"""
feedback_mask/run_inference.py — Run FeedbackMaskNet on mic*.wav val files.

Single-channel: only mic*.wav needed, no ref*.wav.
Processes full file at once (causal model, same result as streaming).
Output: data/eval_output_gtcrn/enhanced_*.wav

Usage:
    # Val set (default):
    python feedback_mask/run_inference.py

    # Listening test scenario:
    python feedback_mask/run_inference.py --val-dir data/listening_test/1_loud_feedback \
                                          --out-dir data/listening_test/1_loud_feedback

    # All listening test scenarios at once:
    for d in data/listening_test/*/; do
        python feedback_mask/run_inference.py --val-dir "$d" --out-dir "$d"
    done
"""

import sys
import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model import FeedbackMaskNet, SR, N_FFT, HOP

_console_hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')


def run_batch(val_dir=None, out_dir=None, checkpoint=None):
    val_dir    = Path(val_dir    or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')
    out_dir    = Path(out_dir    or PROJECT_ROOT / 'data' / 'eval_output_gtcrn')
    checkpoint = Path(checkpoint or PROJECT_ROOT / 'checkpoints' / 'gtcrn_feedback' / 'best.pt')

    assert checkpoint.exists(), f'No checkpoint at {checkpoint} — train first.'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model = FeedbackMaskNet().to(device).eval()
    ckpt  = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    print(f'FeedbackMaskNet loaded from {checkpoint}')

    out_dir.mkdir(parents=True, exist_ok=True)

    mic_files = sorted(val_dir.glob('mic*.wav'))
    if not mic_files:
        print(f'No mic*.wav files found in {val_dir}')
        return

    for mic_path in mic_files:
        out_stem = mic_path.stem.replace('mic', 'enhanced_gtcrn', 1) or 'enhanced_gtcrn'
        mic_np, sr = sf.read(str(mic_path), dtype='float32')
        assert sr == SR, f'{mic_path.name}: expected {SR} Hz, got {sr} Hz'
        if mic_np.ndim > 1: mic_np = mic_np.mean(1)

        # Console HPF — matches training conditions
        mic_np = sosfilt(_console_hpf, mic_np).astype(np.float32)

        mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)   # (1, N)
        mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
        mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)  # (1, F, T, 2)

        with torch.no_grad():
            enh_spec = model(mic_spec)

        enh_c   = enh_spec[..., 0] + 1j * enh_spec[..., 1]   # (1, F, T)
        enh_t   = torch.istft(enh_c, N_FFT, HOP, N_FFT, window)
        enh_np  = enh_t[0].cpu().numpy()

        out_path = out_dir / f'{out_stem}.wav'
        sf.write(str(out_path), enh_np[:len(mic_np)], SR, subtype='PCM_16')
        print(f'  {mic_path.name} → {out_path.name}')

    print(f'Done — {len(mic_files)} files → {out_dir}/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--val-dir',    default=None)
    p.add_argument('--out-dir',    default=None)
    p.add_argument('--checkpoint', default=None)
    args = p.parse_args()
    run_batch(args.val_dir, args.out_dir, args.checkpoint)
