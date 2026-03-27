"""
eval/run_inference.py — Run trained FDKFNet on a directory of mic+ref files.

Works on both the val set (mic_000001.wav + ref_000001.wav → enhanced_000001.wav)
and listening test files (mic.wav + ref.wav → enhanced.wav).
Falls back to FreqDomainNLMS v0 if no FDKFNet checkpoint exists.

Usage:
    # Val set (default):
    python eval/run_inference.py

    # Listening test — one scenario at a time:
    python eval/run_inference.py --val-dir data/listening_test/1_loud_feedback \\
                                  --out-dir data/listening_test/1_loud_feedback

    # All listening test scenarios at once (bash):
    for d in data/listening_test/*/; do
        python eval/run_inference.py --val-dir "$d" --out-dir "$d"
    done
"""

import argparse
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'train'))
sys.path.insert(0, str(PROJECT_ROOT / 'simulator'))

from model import FDKFNet
from fdaf import FreqDomainNLMS

SR    = 48000
HOP   = 480
N_FFT = 960
WIN   = torch.hann_window(N_FFT)

# Console HPF — must match recursive_train.py training conditions exactly.
# Training applies a 2nd-order Butterworth HPF at 90Hz to every frame.
# Running without it creates a train/inference spectral mismatch in the sub-100Hz
# bins that the Kalman filter has never seen unfiltered.
_console_hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')


def stft_frame(x_np: np.ndarray) -> torch.Tensor:
    """HOP-sample numpy block → complex (1, N_FFT//2+1) tensor. Causal left-pad."""
    x        = torch.from_numpy(x_np).unsqueeze(0)              # (1, HOP)
    x_padded = torch.nn.functional.pad(x, (N_FFT - HOP, 0))     # (1, N_FFT)
    return torch.fft.rfft(x_padded * WIN, n=N_FFT)               # (1, F)


def istft_frame(X: torch.Tensor) -> np.ndarray:
    """Complex (1, F) tensor → HOP-sample numpy block."""
    x = torch.fft.irfft(X, n=N_FFT)   # (1, N_FFT)
    return x[0, -HOP:].numpy()


def run_batch(val_dir=None, out_dir=None, checkpoint=None):
    val_dir    = Path(val_dir    or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')
    out_dir    = Path(out_dir    or PROJECT_ROOT / 'data' / 'eval_output')
    checkpoint = Path(checkpoint or PROJECT_ROOT / 'checkpoints' / 'fdkfnet' / 'best.pt')

    if checkpoint.exists():
        model = FDKFNet()
        ckpt  = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
        model.eval()
        use_fdkf = True
        print(f"FDKFNet loaded from {checkpoint}")
    else:
        print(f"No FDKFNet checkpoint at {checkpoint} — falling back to v0 FreqDomainNLMS")
        use_fdkf = False

    out_dir.mkdir(parents=True, exist_ok=True)

    mic_files = sorted(val_dir.glob('mic*.wav'))
    if not mic_files:
        print(f"No mic*.wav files found in {val_dir}")
        return

    for mic_path in mic_files:
        out_stem = mic_path.stem.replace('mic', 'enhanced', 1) or 'enhanced'
        mic_np, sr = sf.read(str(mic_path), dtype='float32')
        assert sr == SR, f"{mic_path.name}: expected {SR}Hz, got {sr}Hz"

        ref_path = mic_path.parent / mic_path.name.replace('mic', 'ref', 1)
        if ref_path.exists():
            ref_np, _ = sf.read(str(ref_path), dtype='float32')
            min_len   = min(len(mic_np), len(ref_np))
            mic_np    = mic_np[:min_len]
            ref_np    = ref_np[:min_len]
        else:
            ref_np  = np.zeros_like(mic_np)
            min_len = len(mic_np)

        # Console HPF — apply to the full file before the frame loop (stateful, causal).
        # mic: matches training (HPF applied to mic in recursive_train.py).
        # ref: HPF'd here because training teacher-forcing ref is HPF'd reverberant vocal.
        mic_np = sosfilt(_console_hpf, mic_np)
        ref_np = sosfilt(_console_hpf, ref_np)

        # Pad to a whole number of HOP blocks so the last partial frame is not silently dropped.
        pad      = (-min_len) % HOP
        mic_pad  = np.pad(mic_np, (0, pad))
        ref_pad  = np.pad(ref_np, (0, pad))
        n_frames = len(mic_pad) // HOP
        enhanced = np.zeros(len(mic_pad), dtype=np.float32)

        if use_fdkf:
            H, P, gru_h = model.init_state(batch_size=1, device='cpu')
            for i in range(0, n_frames * HOP, HOP):
                mic_f = stft_frame(mic_pad[i:i + HOP])
                ref_f = stft_frame(ref_pad[i:i + HOP])
                with torch.no_grad():
                    speech_f, H, P, gru_h = model.forward_frame(mic_f, ref_f, H, P, gru_h)
                enhanced[i:i + HOP] = istft_frame(speech_f)
        else:
            fdaf = FreqDomainNLMS(filter_len=1024, block_size=HOP, mu=0.02)
            for i in range(0, n_frames * HOP, HOP):
                enhanced[i:i + HOP] = fdaf.process(mic_pad[i:i + HOP], ref_pad[i:i + HOP])

        sf.write(str(out_dir / f'{out_stem}.wav'), enhanced[:min_len], SR, subtype='PCM_16')

    print(f"Done — {len(mic_files)} files enhanced → {out_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--val-dir',    default=None)
    p.add_argument('--out-dir',    default=None)
    p.add_argument('--checkpoint', default=None)
    args = p.parse_args()
    run_batch(args.val_dir, args.out_dir, args.checkpoint)
