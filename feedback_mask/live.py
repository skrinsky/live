"""
feedback_mask/live.py — Real-time single-channel feedback suppression (FeedbackMaskNet/GTCRN)

Single mic input only. No reference channel, no loopback cable required.
Plug mic in, processed audio out.

Architecture: GTCRN spectral mask (causal, ~24K params).
Latency: HOP=480 samples = 10ms callback at 48kHz.
Total end-to-end latency including ADC/DAC buffers: typically ~21–30ms.

The model processes a rolling 320ms context window (32 STFT frames) each callback.
GRU state is implicitly maintained via the sliding buffer — the model sees enough
history to track feedback buildup and suppress it within a frame or two of onset.

Usage:
    python feedback_mask/live.py --list
    python feedback_mask/live.py --input 2 --output 4
    python feedback_mask/live.py --input 2 --output 4 --checkpoint checkpoints/gtcrn_feedback/best.pt
"""

import sys
import sounddevice as sd
import torch
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfilt_zi

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model import FeedbackMaskNet, SR, N_FFT, HOP, N_FREQ

# sqrt(Hann) WOLA — matches training exactly
WIN_T  = torch.hann_window(N_FFT).sqrt()
WIN_NP = WIN_T.numpy()

# DC blocking (~5Hz highpass) — removes ADC offset before normalization
_dc_sos = butter(1, 5.0 / (SR / 2), btype='high', output='sos')

# Console HPF (90Hz) — must match feedback_mask/train.py training conditions
_console_hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')

# How many STFT frames of context to feed the model each callback.
# 32 frames = 320ms. Enough for the GRU to track feedback onset and decay.
CONTEXT_FRAMES = 32

TARGET_RMS = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',      type=int, default=None,
                    help='Input device index (see --list)')
    ap.add_argument('--output',     type=int, default=None,
                    help='Output device index (see --list)')
    ap.add_argument('--checkpoint', type=str,
                    default=str(PROJECT_ROOT / 'checkpoints' / 'gtcrn_feedback' / 'best.pt'))
    ap.add_argument('--list',       action='store_true',
                    help='List audio devices and exit')
    args = ap.parse_args()

    print(sd.query_devices())
    if args.list:
        return

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), \
        f'No checkpoint at {ckpt_path} — run feedback_mask/train.py first.'

    model = FeedbackMaskNet().eval()
    ckpt  = torch.load(str(ckpt_path), map_location='cpu')
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    print(f'FeedbackMaskNet loaded — SR={SR}Hz, HOP={HOP} ({HOP/SR*1000:.0f}ms/block), '
          f'context={CONTEXT_FRAMES} frames ({CONTEXT_FRAMES*HOP/SR*1000:.0f}ms)')

    # ── Persistent callback state ───────────────────────────────────────────────
    dc_zi       = sosfilt_zi(_dc_sos)       * 0.0
    hpf_zi      = sosfilt_zi(_console_hpf) * 0.0
    prev_block  = np.zeros(HOP,   dtype=np.float32)
    prev_synth  = np.zeros(N_FFT, dtype=np.float32)
    rms_smooth  = np.array([TARGET_RMS])
    # STFT buffer: (1, N_FREQ, CONTEXT_FRAMES, 2) — rolled each callback
    stft_buf    = torch.zeros(1, N_FREQ, CONTEXT_FRAMES, 2)

    def callback(indata, outdata, frames, time, status):
        nonlocal dc_zi, hpf_zi, prev_block, prev_synth, rms_smooth, stft_buf

        if status:
            print(status)

        mono = indata[:, 0].copy()

        # DC block
        mono, dc_zi = sosfilt(_dc_sos, mono, zi=dc_zi)

        # Running RMS normalisation — keeps model input level stable regardless of mic gain
        block_rms    = float(np.sqrt(np.mean(mono ** 2))) + 1e-8
        rms_smooth[0] = 0.95 * rms_smooth[0] + 0.05 * block_rms
        rms_scale    = TARGET_RMS / max(rms_smooth[0], 1e-4)
        mono         = mono * rms_scale

        # Console HPF — matches training conditions
        mono, hpf_zi = sosfilt(_console_hpf, mono.astype(np.float64), zi=hpf_zi)
        mono = mono.astype(np.float32)

        # STFT frame: [prev_block | mono] windowed → (1, N_FREQ) complex
        x   = torch.from_numpy(np.concatenate([prev_block, mono])).unsqueeze(0)
        frm = torch.fft.rfft(x * WIN_T, n=N_FFT)   # (1, N_FREQ)
        prev_block = mono.copy()

        # Slide buffer left by 1, insert new frame at the end
        stft_buf = torch.roll(stft_buf, -1, dims=2)
        stft_buf[:, :, -1, 0] = frm.real
        stft_buf[:, :, -1, 1] = frm.imag

        # Model inference on full context window
        with torch.no_grad():
            enh_spec = model(stft_buf)   # (1, N_FREQ, CONTEXT_FRAMES, 2)

        # Take only the current (last) output frame for synthesis
        enh_c = enh_spec[:, :, -1, 0] + 1j * enh_spec[:, :, -1, 1]   # (1, N_FREQ)

        if not torch.isfinite(enh_c).all():
            print('[WARNING] NaN/Inf in model output — passing mic through', flush=True)
            enh_c = frm

        # WOLA synthesis: irfft → sqrt-Hann → OLA
        x_raw    = torch.fft.irfft(enh_c, n=N_FFT)[0].numpy()
        x_win    = x_raw * WIN_NP
        enhanced = x_win[:HOP] + prev_synth[HOP:]
        prev_synth = x_win

        # Undo RMS normalisation so output level matches original mic level
        outdata[:, 0] = np.clip(enhanced[:frames] / rms_scale, -1.0, 1.0)

    print(f'Running on devices (input={args.input}, output={args.output}) — Ctrl+C to stop')

    with sd.Stream(samplerate=SR, blocksize=HOP, dtype='float32',
                   channels=(1, 1), device=(args.input, args.output),
                   callback=callback):
        while True:
            sd.sleep(1000)


if __name__ == '__main__':
    main()
