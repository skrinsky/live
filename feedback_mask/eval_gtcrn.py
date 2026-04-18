"""
feedback_mask/eval_gtcrn.py — Evaluation of GTCRN48k source-separation model.

Generates a test clip (vocal + room reverb + feedback ring), runs the checkpoint,
and reports SI-SNR improvement over the degraded mic signal.

Unlike eval.py (which checks mask delta at a ring bin), this eval measures source
separation quality: how much does the enhanced output resemble the clean reference?

Metrics:
  SI-SNR improvement = SI-SNR(enhanced, clean) - SI-SNR(mic, clean)
  Positive = model is helping. Target: > 3 dB improvement.

Usage:
    python feedback_mask/eval_gtcrn.py
    python feedback_mask/eval_gtcrn.py --ring-freq 1200 --gain 0.95
    python feedback_mask/eval_gtcrn.py --checkpoint checkpoints/gtcrn_sep/best.pt
    python feedback_mask/eval_gtcrn.py --vocal-file /path/to/speech.wav
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
from scipy.signal import lfilter, butter, sosfilt, resample_poly, fftconvolve
from math import gcd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model_gtcrn import GTCRN48k, SR, N_FFT, HOP, N_FREQ


def _make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def si_snr(estimate, target, eps=1e-8):
    """Scale-invariant SNR in dB. Higher = better."""
    estimate = estimate - estimate.mean()
    target   = target   - target.mean()
    dot      = np.dot(target, estimate)
    s_target = dot / (np.dot(target, target) + eps) * target
    e_noise  = estimate - s_target
    return 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + eps) + eps)


def generate_test_clip(ring_freq_hz=800.0, gain=0.95, duration=10.0, seed=42,
                       vocal_file=None):
    """
    Generates (clean_np, mic_np, ring_bin):
      clean_np : dry vocal, no room, no feedback — the ideal model output
      mic_np   : clean vocal → room reverb → +noise → IIR feedback loop
      ring_bin : STFT bin index for ring_freq_hz

    Having clean_np lets us measure SI-SNR improvement, not just mask delta.
    """
    rng = np.random.default_rng(seed)
    n   = int(duration * SR)

    # ── Vocal source ─────────────────────────────────────────────────────────
    if vocal_file is not None:
        audio, file_sr = sf.read(str(vocal_file), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            g     = gcd(file_sr, SR)
            audio = resample_poly(audio, SR // g, file_sr // g).astype(np.float64)
        if len(audio) < n:
            audio = np.tile(audio, int(np.ceil(n / len(audio))))
        vocal = audio[:n].astype(np.float64)
    else:
        from scipy.signal import butter as _butter, sosfilt as _sosfilt
        vocal = rng.standard_normal(n).astype(np.float64)
        sos   = _butter(4, [100 / (SR / 2), 3000 / (SR / 2)], btype='bandpass', output='sos')
        vocal = _sosfilt(sos, vocal)

    vocal = vocal / (np.abs(vocal).max() + 1e-8) * 0.3

    # ── Clean reference: just the dry vocal, HPF'd and normalised ────────────
    hpf       = _make_hpf(90)
    clean_np  = sosfilt(hpf, vocal).astype(np.float32)
    clean_rms = float(np.sqrt(np.mean(clean_np**2))) + 1e-8
    clean_np  = np.clip(clean_np * (0.1 / clean_rms), -1.0, 1.0).astype(np.float32)

    # ── Room reverb: simple exponential decay IR ──────────────────────────────
    rt60     = 0.6   # seconds — typical small venue
    t_arr    = np.arange(int(rt60 * SR)) / SR
    room_ir  = rng.standard_normal(len(t_arr)).astype(np.float32)
    room_ir *= np.exp(-6.9 * t_arr / rt60).astype(np.float32)
    room_ir /= np.abs(room_ir).max() + 1e-8

    reverb   = fftconvolve(vocal, room_ir)[:n].astype(np.float64)
    reverb   = sosfilt(_make_hpf(90), reverb).astype(np.float64)

    # ── Background noise (SNR ~20 dB) ────────────────────────────────────────
    noise      = rng.standard_normal(n).astype(np.float64)
    vocal_rms  = float(np.sqrt(np.mean(reverb**2))) + 1e-8
    noise_rms  = float(np.sqrt(np.mean(noise**2)))  + 1e-8
    noise      = noise * (vocal_rms / noise_rms) * 10**(-20 / 20)
    noisy      = (reverb + noise).astype(np.float64)

    # ── Narrow resonator IR at ring_freq_hz ──────────────────────────────────
    ir_len = int(0.05 * SR)
    ir_t   = np.arange(ir_len) / SR
    Q      = 50.0
    decay  = np.pi * ring_freq_hz / Q
    h      = np.exp(-decay * ir_t) * np.cos(2 * np.pi * ring_freq_hz * ir_t)
    peak   = np.abs(np.fft.rfft(h, n=max(len(h) * 4, 4096))).max()
    h      = h / (peak + 1e-8) * gain

    # ── IIR feedback loop ─────────────────────────────────────────────────────
    a      = np.concatenate([[1.0], -h])
    mic_np = lfilter([1.0], a, noisy)
    mic_np = np.nan_to_num(mic_np, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    mic_np = np.clip(mic_np, -1.0, 1.0)

    # ── HPF + RMS normalise (must match train_gtcrn.py) ──────────────────────
    mic_np  = sosfilt(_make_hpf(90), mic_np).astype(np.float32)
    mic_rms = float(np.sqrt(np.mean(mic_np**2))) + 1e-8
    mic_np  = np.clip(mic_np * (0.1 / mic_rms), -1.0, 1.0).astype(np.float32)

    ring_bin = int(round(ring_freq_hz * N_FFT / SR))
    return clean_np, mic_np, ring_bin


def run_eval(ring_freq_hz=800.0, gain=0.95, ckpt_path=None, vocal_file=None):
    ckpt_path = ckpt_path or str(
        PROJECT_ROOT / 'checkpoints' / 'gtcrn_sep' / 'best.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model = GTCRN48k().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
    model.load_state_dict(state['model'])
    model.eval()
    print(f'Checkpoint: epoch {state.get("epoch", "?")}, '
          f'best_loss={state.get("best_loss", float("nan")):.4f}')
    print(f'Ring: {ring_freq_hz:.0f} Hz → bin {int(round(ring_freq_hz * N_FFT / SR))} '
          f'({int(round(ring_freq_hz * N_FFT / SR)) * SR / N_FFT:.1f} Hz), gain={gain}')

    clean_np, mic_np, ring_bin = generate_test_clip(
        ring_freq_hz, gain, vocal_file=vocal_file)

    # ── STFT → model → ISTFT ─────────────────────────────────────────────────
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)

    with torch.no_grad():
        enh_spec = model(mic_spec)   # (1, F, T, 2)

    enh_cplx  = torch.complex(enh_spec[0, :, :, 0], enh_spec[0, :, :, 1])
    enh_audio = torch.istft(enh_cplx, N_FFT, HOP, N_FFT, window).cpu().numpy()

    # Trim to same length as clean reference
    n = len(clean_np)
    enh_audio = enh_audio[:n]
    mic_trim  = mic_np[:n]

    # ── SI-SNR metrics ────────────────────────────────────────────────────────
    sisnr_mic = si_snr(mic_trim.astype(np.float64),   clean_np.astype(np.float64))
    sisnr_enh = si_snr(enh_audio.astype(np.float64),  clean_np.astype(np.float64))
    improvement = sisnr_enh - sisnr_mic

    print(f'\nSI-SNR (mic vs clean):      {sisnr_mic:+.2f} dB')
    print(f'SI-SNR (enhanced vs clean): {sisnr_enh:+.2f} dB')
    print(f'SI-SNR improvement:         {improvement:+.2f} dB', end='  ')
    if improvement > 3.0:
        print('✓ Good suppression')
    elif improvement > 0.5:
        print('~ Marginal improvement')
    else:
        print('✗ No improvement (model not learning separation yet)')

    # ── Spectral check at ring bin ────────────────────────────────────────────
    enh_mag = np.abs(np.fft.rfft(enh_audio)) / len(enh_audio)
    mic_mag = np.abs(np.fft.rfft(mic_trim))  / len(mic_trim)
    freqs   = np.fft.rfftfreq(len(mic_trim), 1 / SR)
    ring_idx = np.argmin(np.abs(freqs - ring_freq_hz))
    ring_suppression = 20 * np.log10((mic_mag[ring_idx] + 1e-8) / (enh_mag[ring_idx] + 1e-8))
    print(f'Ring bin suppression:       {ring_suppression:+.1f} dB at {ring_freq_hz:.0f} Hz')

    # ── Save audio ────────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / 'checkpoints' / 'gtcrn_sep'
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / 'eval_mic.wav'),      mic_trim,  SR)
    sf.write(str(out_dir / 'eval_enhanced.wav'), enh_audio, SR)
    sf.write(str(out_dir / 'eval_clean.wav'),    clean_np,  SR)
    print(f'\nAudio → {out_dir}/eval_mic.wav  eval_enhanced.wav  eval_clean.wav')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    times     = np.linspace(0, n / SR, n)
    show_hz   = 6000
    show_bins = int(show_hz * N_FFT / SR)

    def _spectrogram(audio):
        stft = np.abs(np.array([
            np.fft.rfft(audio[i:i+N_FFT])
            for i in range(0, len(audio) - N_FFT, HOP)
        ])).T   # (F, T)
        return 20 * np.log10(stft[:show_bins] + 1e-8)

    t_frames = np.arange(0, len(mic_trim) - N_FFT, HOP) / SR
    extent   = [t_frames[0], t_frames[-1], 0, show_hz]

    spec_mic = _spectrogram(mic_trim)
    spec_enh = _spectrogram(enh_audio)
    vmin = max(spec_mic.max() - 60, -80)
    vmax = spec_mic.max()

    axes[0].imshow(spec_mic, aspect='auto', origin='lower', extent=extent,
                   vmin=vmin, vmax=vmax, cmap='inferno')
    axes[0].axhline(ring_freq_hz, color='cyan', lw=1.5, linestyle='--', label='ring freq')
    axes[0].set_title(f'Input mic  (ring={ring_freq_hz:.0f} Hz, gain={gain})')
    axes[0].set_ylabel('Freq (Hz)')
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].imshow(spec_enh, aspect='auto', origin='lower', extent=extent,
                   vmin=vmin, vmax=vmax, cmap='inferno')
    axes[1].axhline(ring_freq_hz, color='cyan', lw=1.5, linestyle='--', label='ring freq')
    axes[1].set_title(f'Enhanced  (SI-SNR {improvement:+.2f} dB improvement, '
                      f'ring suppression {ring_suppression:+.1f} dB)')
    axes[1].set_ylabel('Freq (Hz)')
    axes[1].legend(loc='upper right', fontsize=8)

    # Difference spectrogram (what got removed)
    diff = spec_mic - spec_enh
    im   = axes[2].imshow(diff, aspect='auto', origin='lower', extent=extent,
                          vmin=-20, vmax=20, cmap='RdYlGn')
    plt.colorbar(im, ax=axes[2], fraction=0.02, label='dB removed')
    axes[2].axhline(ring_freq_hz, color='blue', lw=1.5, linestyle='--', label='ring freq')
    axes[2].set_title('Difference (green = suppressed, red = amplified)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Freq (Hz)')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.suptitle(f'GTCRN48k eval — ring={ring_freq_hz:.0f} Hz, gain={gain}, '
                 f'SI-SNR improvement={improvement:+.2f} dB',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plot_path = str(out_dir / 'eval_spectrogram.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Plot  → {plot_path}')
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ring-freq',  type=float, default=800.0)
    ap.add_argument('--gain',       type=float, default=0.95)
    ap.add_argument('--checkpoint', type=str,   default=None)
    ap.add_argument('--vocal-file', type=str,   default=None,
                    help='Real speech/singing file instead of synthetic noise.')
    args = ap.parse_args()
    run_eval(args.ring_freq, args.gain, args.checkpoint, args.vocal_file)
