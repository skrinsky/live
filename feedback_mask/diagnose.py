"""
feedback_mask/diagnose.py — diagnose why ring bin mask is not being suppressed.

Usage:
    python feedback_mask/diagnose.py
    python feedback_mask/diagnose.py --checkpoint checkpoints/gtcrn_feedback/best.pt
    python feedback_mask/diagnose.py --ring-freq 800 --gain 0.95
"""
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model import FeedbackMaskNet, SR, N_FFT, HOP, N_FREQ
from eval import generate_test_clip


def diagnose(ring_freq_hz=800.0, gain=0.95, ckpt_path=None):
    ckpt_path = ckpt_path or str(
        PROJECT_ROOT / 'checkpoints' / 'gtcrn_feedback' / 'best.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model = FeedbackMaskNet().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
    model.load_state_dict(state['model'])
    model.eval()
    print(f'Checkpoint: epoch {state.get("epoch","?")}, '
          f'best_loss={state.get("best_loss", float("nan")):.4f}')

    mic_np, ring_bin = generate_test_clip(ring_freq_hz, gain)
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)

    with torch.no_grad():
        enh_spec, pred_mask, _ = model(mic_spec)

    mic_mag  = mic_stft.abs()                        # (1, N_FREQ, T)
    mask     = pred_mask[0]                          # (N_FREQ, T)
    logits   = torch.logit(mask.clamp(1e-4, 1-1e-4))

    nb = 20
    lo = max(0, ring_bin - nb)
    hi = min(N_FREQ, ring_bin + nb + 1)
    bg_bins = list(range(lo, ring_bin)) + list(range(ring_bin + 1, hi))

    print(f'\n── Input signal ──')
    print(f'  Ring bin {ring_bin} ({ring_freq_hz:.0f} Hz)  mic_mag mean: '
          f'{mic_mag[0, ring_bin].mean():.4f}')
    print(f'  Background bins ±{nb}       mic_mag mean: '
          f'{mic_mag[0, bg_bins].mean():.4f}')
    print(f'  Ring / background ratio: '
          f'{mic_mag[0,ring_bin].mean() / mic_mag[0,bg_bins].mean():.2f}x  '
          f'(expect ~{1/(1-gain):.0f}x at gain={gain})')

    print(f'\n── Predicted mask ──')
    print(f'  Ring bin mask  mean={mask[ring_bin].mean():.4f}  '
          f'min={mask[ring_bin].min():.4f}  max={mask[ring_bin].max():.4f}')
    print(f'  Background mask mean={mask[bg_bins].mean():.4f}')
    delta = mask[bg_bins].mean() - mask[ring_bin].mean()
    if delta > 0.05:
        print(f'  ✓ Ring suppressed by {delta:.3f}')
    else:
        print(f'  ✗ No suppression (delta={delta:.3f})')

    print(f'\n── Logits (pre-sigmoid) ──')
    print(f'  Ring bin logit  mean={logits[ring_bin].mean():.3f}  '
          f'(negative = suppressing, positive = passing)')
    print(f'  Background logit mean={logits[bg_bins].mean():.3f}')

    print(f'\n── per_bin_ratio (oracle: 0=ring, 1=nonring) ──')
    # Recompute with same normalisation as training
    # (both independently normalised to RMS=0.1, so mic_mag > tgt_mag at non-ring)
    # Oracle visible here WITHOUT target — use mic_mag temporal slope as proxy
    # (ring bin grows, non-ring bins are stationary)
    T = mic_mag.shape[2]
    first_q  = mic_mag[0, :, :T//4].mean(dim=1)   # (N_FREQ,)
    last_q   = mic_mag[0, :, -T//4:].mean(dim=1)  # (N_FREQ,)
    slope    = last_q - first_q                    # positive = growing (ring)

    print(f'  Ring bin  slope (last_q - first_q): {slope[ring_bin]:.4f}  '
          f'(positive = ring is building up)')
    print(f'  Background slope mean: {slope[bg_bins].mean():.4f}')
    print(f'  Is ring clearly visible? '
          f'{"YES" if slope[ring_bin] > slope[bg_bins].mean() + 0.001 else "NO — ring may not be visible in features"}')

    print(f'\n── GRU feature check ──')
    # Check that Δ features at ring bin are distinct from background
    from model import _prepare_features
    feat = _prepare_features(mic_spec)   # (1, N_DELTA, N_FREQ, T)
    # feat[0, 0] = log_mag, feat[0, 1] = Δ1, feat[0, 2] = Δ4, feat[0, 3] = Δ10
    for i, name in enumerate(['log_mag', 'Δ1(10ms)', 'Δ4(40ms)', 'Δ10(100ms)']):
        ring_val = feat[0, i, ring_bin, T//2:].mean().item()
        bg_val   = feat[0, i, bg_bins, T//2:].mean().item()
        print(f'  {name:12s}  ring={ring_val:+.4f}  background={bg_val:+.4f}  '
              f'diff={ring_val-bg_val:+.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ring-freq',  type=float, default=800.0)
    ap.add_argument('--gain',       type=float, default=0.95)
    ap.add_argument('--checkpoint', type=str,   default=None)
    args = ap.parse_args()
    diagnose(args.ring_freq, args.gain, args.checkpoint)
