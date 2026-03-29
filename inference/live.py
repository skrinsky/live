"""
inference/live.py — Real-time feedback suppression (FDKFNet v1 or FreqDomainNLMS v0)

Architecture (v1 — FDKFNet checkpoint present + --ref flag):
  Neural Kalman filter per frequency bin + GRU covariance estimation.
  Trained with recursive feedback simulation (arXiv 2309.16048).
  Requires hardware loopback reference (ch1 on --input device = PA send signal).

Architecture (v0 — no checkpoint or no --ref):
  FreqDomainNLMS overlap-save FDAF. 1024 taps, O(N log N) per block.
  No training needed. Use this to verify the audio pipeline before training.

Usage:
    python inference/live.py --list
    python inference/live.py --input 2 --output 4           # v0: FDAF fallback
    python inference/live.py --input 2 --output 4 --ref     # v1: FDKFNet with loopback
    python inference/live.py --checkpoint checkpoints/fdkfnet/best.pt --input 2 --output 4 --ref

Latency: HOP=480 samples = 10ms callback at 48kHz. FDKFNet is causal (per-frame update).
Total end-to-end latency including ADC/DAC buffers: typically ~21–30ms.
"""

import sys
import sounddevice as sd
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfilt_zi

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'train'))
sys.path.insert(0, str(PROJECT_ROOT / 'simulator'))

from model import FDKFNet
from fdaf import FreqDomainNLMS

SR    = 48000
HOP   = 480      # 10ms at 48kHz
N_FFT = 960      # HOP * 2

# Rectangular window (ones): irfft(rfft([zeros|x]))[-HOP:] = x exactly.
# Hann window was a bug — hann[HOP:] tapers each frame 1→0 over 10ms (100Hz AM distortion).
WIN_T = torch.ones(N_FFT)   # module-level constant — not reallocated in callback

TARGET_RMS   = 0.1
rms_smoother = np.array([TARGET_RMS])

# DC blocking filter (~5Hz 1st-order highpass) — removes DC offset from cheap ADCs.
# Applied to mic only; ref is hardware loopback from box output and is already clean.
# Using sosfilt (vectorized C) instead of a Python sample-loop — the loop version costs
# ~1–2ms per 480-sample block on Pi 5 (~10–20% of the real-time budget).
_dc_sos = butter(1, 5.0 / (SR / 2), btype='high', output='sos')
_dc_zi  = sosfilt_zi(_dc_sos) * 0.0

# Console HPF (90Hz 2nd-order Butterworth) — models standard vocal channel HPF on PA consoles.
# Applied to both mic and ref to match training conditions (recursive_train.py applies the same).
_console_hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
_hpf_zi_mic  = sosfilt_zi(_console_hpf) * 0.0
_hpf_zi_ref  = sosfilt_zi(_console_hpf) * 0.0


def stft_frame(x_np: np.ndarray) -> torch.Tensor:
    """HOP-sample numpy block → complex (1, N_FFT//2+1) tensor. Causal left-pad, rectangular window."""
    x        = torch.from_numpy(x_np).unsqueeze(0)   # (1, HOP)
    x_padded = F.pad(x, (N_FFT - HOP, 0))            # (1, N_FFT)
    return torch.fft.rfft(x_padded * WIN_T, n=N_FFT)  # (1, F)


def istft_frame(X: torch.Tensor) -> np.ndarray:
    """Complex (1, F) tensor → HOP-sample numpy block."""
    x = torch.fft.irfft(X, n=N_FFT)   # (1, N_FFT)
    return x[0, -HOP:].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      type=int, default=None,
                        help='Input device index (see --list)')
    parser.add_argument('--output',     type=int, default=None,
                        help='Output device index (see --list)')
    parser.add_argument('--ref',        action='store_true', default=False,
                        help='Enable v1 FDKFNet mode: open 2 input channels (ch0=mic, '
                             'ch1=loopback ref). The ref must be ch1 on the same --input '
                             'device — not a separate device index.')
    parser.add_argument('--list',       action='store_true',
                        help='List audio devices and exit')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'checkpoints' / 'fdkfnet' / 'best.pt'))
    args = parser.parse_args()

    print(sd.query_devices())
    if args.list:
        return

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model  = FDKFNet()
        saved  = torch.load(str(ckpt), map_location='cpu')
        model.load_state_dict(
            saved['model'] if isinstance(saved, dict) and 'model' in saved else saved
        )
        model.eval()
        H, P, gru_h = model.init_state(batch_size=1, device='cpu')
        use_fdkf    = True
        print(f"FDKFNet loaded — SR={SR}Hz, HOP={HOP} ({HOP/SR*1000:.0f}ms/block)")
    else:
        fdaf     = FreqDomainNLMS(filter_len=1024, block_size=HOP, mu=0.02)
        use_fdkf = False
        H = P = gru_h = None
        print(f"No FDKFNet checkpoint at {ckpt} — using v0 FreqDomainNLMS fallback")
        print("Run train/recursive_train.py first to enable the neural model.")

    def callback(indata, outdata, frames, time, status):
        nonlocal H, P, gru_h
        global _hpf_zi_mic, _hpf_zi_ref, _dc_zi

        if status:
            print(status)

        mono = indata[:, 0].copy()

        # DC blocking — removes ADC offset before normalization/filtering
        mono, _dc_zi = sosfilt(_dc_sos, mono, zi=_dc_zi)

        # Running RMS normalization — compute scale BEFORE applying it
        block_rms       = float(np.sqrt(np.mean(mono ** 2))) + 1e-8
        rms_smoother[0] = 0.95 * rms_smoother[0] + 0.05 * block_rms
        rms_scale       = TARGET_RMS / max(rms_smoother[0], 1e-4)
        mono            = mono * rms_scale

        if args.ref:
            ref_block = indata[:, 1].copy()
            # Apply the SAME RMS scale to ref so the Kalman H(z) estimate is gain-consistent.
            # Without this, the mic/ref ratio depends on input level, which miscalibrates
            # both H(z) and the VAD gate threshold.
            ref_block = ref_block * rms_scale
            # Sanity check — silent ref usually means loopback cable is unplugged.
            ref_rms = float(np.sqrt(np.mean(ref_block ** 2)))
            if ref_rms < 1e-5 and block_rms > 1e-4:
                print("[WARNING] ref channel is silent while mic has signal — "
                      "check loopback cable. FDKFNet will fall back to FDAF behavior.",
                      flush=True)
        else:
            ref_block = np.zeros(HOP, dtype=np.float32)

        # Console HPF — matches recursive_train.py training conditions (90Hz HPF on vocal channel)
        mono,      _hpf_zi_mic = sosfilt(_console_hpf, mono.astype(np.float64),      zi=_hpf_zi_mic)
        ref_block, _hpf_zi_ref = sosfilt(_console_hpf, ref_block.astype(np.float64), zi=_hpf_zi_ref)
        mono      = mono.astype(np.float32)
        ref_block = ref_block.astype(np.float32)

        if use_fdkf:
            mic_f = stft_frame(mono)
            ref_f = stft_frame(ref_block)
            with torch.no_grad():
                speech_f, H, P, gru_h = model.forward_frame(mic_f, ref_f, H, P, gru_h)
            if not torch.isfinite(speech_f).all():
                # Kalman diverged — pass mic through unmodified rather than going silent.
                print("[WARNING] NaN/Inf in model output — passing mic through", flush=True)
                speech_f = mic_f
            enhanced = istft_frame(speech_f)
        else:
            enhanced = fdaf.process(mono, ref_block)

        # Undo input RMS normalization so output level matches the original mic level.
        # The model was trained with a normalized input; inverting the scale here means
        # the performer hears no unexpected level change when feedback is suppressed.
        outdata[:, 0] = np.clip(enhanced[:frames] / rms_scale, -1.0, 1.0)

    # When --ref is set, open 2 input channels: ch0 = mic, ch1 = loopback ref.
    # The loopback ref must be ch1 on the same audio interface as the mic (same device index)
    # — typically ch0 = XLR mic input, ch1 = line input wired to PA send (loopback).
    input_channels = 2 if args.ref else 1
    device         = (args.input, args.output)

    if args.ref:
        print("v1 mode: 2-channel input (ch0=mic, ch1=loopback ref) — FDKFNet active")
    else:
        print("v0 mode: 1-channel input, no loopback ref — FreqDomainNLMS fallback")
    print(f"Running on devices {device} — Ctrl+C to stop")

    with sd.Stream(samplerate=SR, blocksize=HOP, dtype='float32',
                   channels=(input_channels, 1), device=device, callback=callback):
        while True:
            sd.sleep(1000)


if __name__ == '__main__':
    main()
