"""
feedback_mask/eval.py — Qualitative evaluation of FeedbackMaskNet.

Generates a 10s synthetic clip with a known ring at a specified frequency,
runs the best checkpoint, and checks whether the mask suppresses at that bin.

Usage:
    python feedback_mask/eval.py
    python feedback_mask/eval.py --ring-freq 1200 --gain 0.95
    python feedback_mask/eval.py --checkpoint checkpoints/gtcrn_feedback/best.pt
"""

import sys
import argparse
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lfilter, butter, sosfilt, resample_poly
from math import gcd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model import FeedbackMaskNet, SR, N_FFT, HOP, N_FREQ


def _make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def generate_test_clip(ring_freq_hz=800.0, gain=0.95, duration=10.0, seed=42,
                       vocal_file=None):
    """
    Clip with a known ring at ring_freq_hz run through an IIR feedback loop.

    vocal_file : path to a real speech file (wav/flac).  If None, falls back
                 to synthetic bandpass noise (useless for voice-preservation tests).

    Returns
    -------
    mic_np   : (N,) float32  — RMS-normalised mic signal (as in training)
    ring_bin : int           — STFT bin closest to ring_freq_hz
    """
    n = int(duration * SR)

    if vocal_file is not None:
        audio, file_sr = sf.read(str(vocal_file), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)          # stereo → mono
        if file_sr != SR:
            g = gcd(file_sr, SR)
            audio = resample_poly(audio, SR // g, file_sr // g).astype(np.float64)
        # Loop or truncate to exactly n samples
        if len(audio) < n:
            reps = int(np.ceil(n / len(audio)))
            audio = np.tile(audio, reps)
        vocal = audio[:n].astype(np.float64)
        vocal = vocal / (np.abs(vocal).max() + 1e-8) * 0.3
    else:
        rng = np.random.default_rng(seed)
        # Bandpass noise ≈ speech (100–3000 Hz) — no real voice, only for ring detection test
        vocal = rng.standard_normal(n).astype(np.float64)
        sos = butter(4, [100 / (SR / 2), 3000 / (SR / 2)], btype='bandpass', output='sos')
        vocal = sosfilt(sos, vocal)
        vocal = vocal / (np.abs(vocal).max() + 1e-8) * 0.3

    # Narrow resonator IR at ring_freq_hz (Q=50 → ~16 Hz bandwidth at 800 Hz)
    ir_len = int(0.05 * SR)
    ir_t   = np.arange(ir_len) / SR
    Q      = 50.0
    decay  = np.pi * ring_freq_hz / Q
    h      = np.exp(-decay * ir_t) * np.cos(2 * np.pi * ring_freq_hz * ir_t)
    peak   = np.abs(np.fft.rfft(h, n=max(len(h) * 4, 4096))).max()
    h      = h / (peak + 1e-8) * gain   # spectral peak = gain

    # IIR feedback loop
    a      = np.concatenate([[1.0], -h])
    mic_np = lfilter([1.0], a, vocal)
    mic_np = np.nan_to_num(mic_np, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    mic_np = np.clip(mic_np, -1.0, 1.0)

    # HPF + RMS normalise — must match train.py
    mic_np = sosfilt(_make_hpf(90), mic_np).astype(np.float32)
    rms    = float(np.sqrt(np.mean(mic_np ** 2))) + 1e-8
    mic_np = np.clip(mic_np * (0.1 / rms), -1.0, 1.0).astype(np.float32)

    ring_bin = int(round(ring_freq_hz * N_FFT / SR))
    return mic_np, ring_bin


def run_eval(ring_freq_hz=800.0, gain=0.95, ckpt_path=None, vocal_file=None):
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

    mic_np, ring_bin = generate_test_clip(ring_freq_hz, gain, vocal_file=vocal_file)
    print(f'Ring: {ring_freq_hz:.0f} Hz → bin {ring_bin} '
          f'({ring_bin * SR / N_FFT:.1f} Hz), gain={gain}')

    # STFT → model
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)

    with torch.no_grad():
        enh_spec, pred_mask, _ = model(mic_spec)

    mask_np = pred_mask.squeeze(0).cpu().numpy()   # (N_FREQ, T)

    # ── Numerical report ──────────────────────────────────────────────────────
    ring_vals  = mask_np[ring_bin, :]
    lo_vals    = mask_np[max(0, ring_bin - 20):ring_bin, :].mean(0)
    hi_vals    = mask_np[ring_bin + 1:ring_bin + 21, :].mean(0)
    bg_mean    = np.concatenate([lo_vals, hi_vals]).mean()

    print(f'\nMask @ ring bin {ring_bin} ({ring_freq_hz:.0f} Hz):  '
          f'mean={ring_vals.mean():.3f}  min={ring_vals.min():.3f}  max={ring_vals.max():.3f}')
    print(f'Mask @ background bins (±20):  mean={bg_mean:.3f}')
    delta = bg_mean - ring_vals.mean()
    if delta > 0.05:
        print(f'✓ Ring bin is {delta:.3f} lower than background — suppression detected')
    elif delta > 0.01:
        print(f'~ Weak suppression: ring bin {delta:.3f} below background')
    else:
        print(f'✗ No selective suppression (delta={delta:.3f})')

    # ── Save audio ────────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / 'checkpoints' / 'gtcrn_feedback'
    sf.write(str(out_dir / 'eval_mic.wav'), mic_np, SR)

    enh_cplx  = torch.complex(enh_spec[0, :, :, 0], enh_spec[0, :, :, 1])
    enh_audio = torch.istft(enh_cplx, N_FFT, HOP, N_FFT, window).cpu().numpy()
    sf.write(str(out_dir / 'eval_enhanced.wav'), enh_audio, SR)
    print(f'\nAudio → {out_dir}/eval_mic.wav  +  eval_enhanced.wav')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    times = np.arange(mask_np.shape[1]) * HOP / SR
    show_bins = min(N_FREQ, int(6000 * N_FFT / SR))   # up to 6 kHz

    # Input spectrogram
    mic_mag = (mic_spec[0, :show_bins, :, 0] ** 2
               + mic_spec[0, :show_bins, :, 1] ** 2 + 1e-12).sqrt().cpu().numpy()
    axes[0].imshow(20 * np.log10(mic_mag + 1e-8),
                   aspect='auto', origin='lower',
                   extent=[0, times[-1], 0, show_bins * SR / N_FFT],
                   vmin=-60, vmax=0, cmap='inferno')
    axes[0].axhline(ring_freq_hz, color='cyan', lw=1.5, linestyle='--', label='ring freq')
    axes[0].set_title(f'Input spectrogram  (ring={ring_freq_hz:.0f} Hz, gain={gain})')
    axes[0].set_ylabel('Freq (Hz)')
    axes[0].legend(loc='upper right', fontsize=8)

    # Mask heatmap
    im = axes[1].imshow(mask_np[:show_bins, :],
                        aspect='auto', origin='lower',
                        extent=[0, times[-1], 0, show_bins * SR / N_FFT],
                        vmin=0, vmax=1, cmap='RdYlGn')
    plt.colorbar(im, ax=axes[1], fraction=0.02)
    axes[1].axhline(ring_freq_hz, color='blue', lw=1.5, linestyle='--', label='ring freq')
    axes[1].set_title('Predicted mask  (green=pass, red=suppress)')
    axes[1].set_ylabel('Freq (Hz)')
    axes[1].legend(loc='upper right', fontsize=8)

    # Mask over time at ring bin vs background
    axes[2].plot(times, ring_vals,
                 color='red', lw=1.5, label=f'ring bin {ring_bin} ({ring_freq_hz:.0f} Hz)')
    axes[2].plot(times, lo_vals,  color='steelblue', lw=1, alpha=0.7, label=f'bg bins {ring_bin-20}..{ring_bin-1}')
    axes[2].plot(times, hi_vals,  color='green',     lw=1, alpha=0.7, label=f'bg bins {ring_bin+1}..{ring_bin+20}')
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_title('Mask over time: ring bin vs background')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Mask value')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'FeedbackMaskNet eval — ring={ring_freq_hz:.0f} Hz, gain={gain}',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plot_path = str(out_dir / 'eval_mask.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Plot  → {plot_path}')
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ring-freq',   type=float, default=800.0)
    ap.add_argument('--gain',        type=float, default=0.95)
    ap.add_argument('--checkpoint',  type=str,   default=None)
    ap.add_argument('--vocal-file',  type=str,   default=None,
                    help='Real speech file to use instead of synthetic noise. '
                         'Pass any wav/flac from your training data.')
    args = ap.parse_args()
    run_eval(args.ring_freq, args.gain, args.checkpoint, args.vocal_file)
