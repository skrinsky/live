"""
feedback_detect/eval_loop.py — Closed-loop simulation: does the model break the loop?

Runs two simulations on the same audio + feedback IR:

  A) Raw path:   voice + noise → room feedback loop → mic  (no processing)
  B) Suppressed: voice + noise → room feedback loop → mic → model+notch → back into loop

In A the loop closes with gain > 1 and the signal rings up.
In B the model detects the growth, the notch attenuates the speaker output,
less energy returns to the mic, and the loop is broken or kept subcritical.

This is the only meaningful evaluation for a feedback suppressor.
File-based metrics (run_inference.py) test the wrong thing — the loop never
closes in a recording, so the notch has no effect on what the mic hears.

Usage:
    python feedback_detect/eval_loop.py
    python feedback_detect/eval_loop.py --gain 1.5 --duration 8
"""

import sys
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_detect'))

from model import FeedbackDetector, SR, N_FFT, HOP, N_FREQ
from notch  import NotchBank
from live   import _cluster_bins

CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'
_hpf_sos   = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
_bin_freqs  = np.fft.rfftfreq(N_FFT, d=1.0 / SR)


# ── Simulation ──────────────────────────────────────────────────────────────

def simulate(voice_np, noise_np, feedback_ir, gain,
             model=None, notch_bank=None, device='cpu', window=None):
    """
    Closed-loop simulation for one audio clip.

    Signal flow:
        mic[t] = voice[t] + noise[t] + (feedback_ir * gain) ★ box_out[t-1]
        box_out[t] = notch_bank(model(mic[t]))   if model provided
        box_out[t] = mic[t]                       otherwise (raw path)

    The overlap-add accumulator carries the tail of each convolution forward,
    so the full IR length is respected across block boundaries.

    Returns
    -------
    mic_out : (n,) float32  — what the mic hears each frame
    box_out : (n,) float32  — what goes to the speaker (after notch, or raw)
    """
    ir  = (feedback_ir * gain).astype(np.float32)
    n   = len(voice_np)
    acc = np.zeros(n + len(ir), dtype=np.float64)   # overlap-add accumulator

    mic_out      = np.zeros(n, dtype=np.float32)
    box_out      = np.zeros(n, dtype=np.float32)
    gru_h        = None
    lm_history   = np.zeros((N_FREQ, 11), dtype=np.float32)
    analysis_buf = np.zeros(N_FFT, dtype=np.float32)

    for i in range(n // HOP):
        s = i * HOP

        # Mic = direct signal + accumulated feedback from previous speaker outputs
        mic_block = (voice_np[s:s+HOP].astype(np.float64) +
                     noise_np[s:s+HOP].astype(np.float64) +
                     acc[s:s+HOP])
        mic_block = np.clip(mic_block, -1.0, 1.0).astype(np.float32)
        mic_out[s:s+HOP] = mic_block

        if model is not None:
            # ── Detection (same logic as live.py) ─────────────────────────
            analysis_buf = np.roll(analysis_buf, -HOP)
            analysis_buf[-HOP:] = mic_block

            buf_t    = torch.from_numpy(analysis_buf).unsqueeze(0).to(device)
            stft     = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window,
                                  return_complex=True)
            mag      = stft.abs()

            lm_now          = torch.log(mag[0, :, 0] + 1e-8).cpu().numpy()
            lm_history      = np.roll(lm_history, 1, axis=1)
            lm_history[:, 0] = lm_now

            feat_np  = np.stack([
                lm_history[:, 0],
                lm_history[:, 0] - lm_history[:, 1],
                lm_history[:, 0] - lm_history[:, 4],
                lm_history[:, 0] - lm_history[:, 10],
            ], axis=0)
            features = (torch.from_numpy(feat_np).to(device)
                            .unsqueeze(0).unsqueeze(-1))   # (1, 4, N_FREQ, 1)

            with torch.no_grad():
                prob, gru_h = model(features, gru_h)

            prob_np = prob[0, :, 0].cpu().numpy()
            above   = (prob_np > 0.5) & (_bin_freqs >= 80.0)
            freqs   = _cluster_bins(_bin_freqs, prob_np, above)

            notch_bank.update(freqs)
            out_block = notch_bank.process(mic_block)
        else:
            out_block = mic_block.copy()

        box_out[s:s+HOP] = out_block

        # Overlap-add: speaker output convolved with feedback IR feeds back to mic
        fb = np.convolve(out_block.astype(np.float64), ir.astype(np.float64))
        acc[s:s + len(fb)] += fb

    return mic_out[:n], box_out[:n]


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_eval(gain=1.3, duration_s=8.0, checkpoint=None, out_dir=None):
    ckpt_path = Path(checkpoint or CHECKPOINT)
    assert ckpt_path.exists(), f'No checkpoint at {ckpt_path} — train first.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model = FeedbackDetector().to(device).eval()
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(
        ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    print(f'Loaded {ckpt_path}')

    out_dir = Path(out_dir or PROJECT_ROOT / 'data' / 'eval_loop')
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(duration_s * SR)

    # ── Build feedback IR from ir_pool ────────────────────────────────────
    ir_pool       = PROJECT_ROOT / 'data' / 'ir_pool'
    mains_files   = sorted(ir_pool.glob('mains_*.wav'))
    monitor_files = sorted(ir_pool.glob('monitor_*.wav'))
    assert mains_files   and monitor_files, \
        'No IRs found in data/ir_pool/ — run simulator/generate_ir_pool.py first.'

    mains_np,   _ = sf.read(str(mains_files[0]),   dtype='float32')
    monitor_np, _ = sf.read(str(monitor_files[0]), dtype='float32')
    if mains_np.ndim   > 1: mains_np   = mains_np.sum(1)
    if monitor_np.ndim > 1: monitor_np = monitor_np.sum(1)

    trunc  = min(len(mains_np), len(monitor_np), int(0.05 * SR))
    ir     = (mains_np[:trunc] + monitor_np[:trunc]).astype(np.float32)
    peak   = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    ir    /= (peak + 1e-8)   # normalise: gain=1.0 → loop just at unity

    # ── Source audio ─────────────────────────────────────────────────────
    vocal_files = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                   if not f.name.startswith('._')]
    if vocal_files:
        import soundfile as sf2
        v, vsr = sf2.read(str(vocal_files[0]), dtype='float32')
        if v.ndim > 1: v = v.mean(1)
        if vsr != SR:
            raise ValueError(f'Vocal SR={vsr}, expected {SR}')
        v = sosfilt(_hpf_sos, v).astype(np.float32)
        # tile/trim to n samples
        if len(v) < n:
            v = np.tile(v, n // len(v) + 1)
        voice_np = v[:n]
    else:
        # Fallback: synthetic chirp to excite the feedback path
        print('No vocal files found — using synthetic chirp as source.')
        t = np.linspace(0, duration_s, n, dtype=np.float32)
        voice_np = (0.1 * np.sin(2 * np.pi * (200 + 800 * t / duration_s) * t)
                    ).astype(np.float32)

    noise_np = (np.random.randn(n) * 1e-4).astype(np.float32)

    print(f'\nGain={gain}  duration={duration_s}s  IR_len={len(ir)} samples')
    print(f'{"":─<60}')

    # ── A) Raw feedback ───────────────────────────────────────────────────
    print('Running A: raw feedback (no model)…')
    mic_raw, box_raw = simulate(voice_np, noise_np, ir, gain=gain)
    rms_raw = _rms_db(mic_raw)
    print(f'  mic RMS: {rms_raw:.1f} dB')

    # ── B) Suppressed ─────────────────────────────────────────────────────
    print('Running B: with FeedbackDetector + NotchBank…')
    notch_bank = NotchBank(sr=SR, q=30.0, depth_db=-24.0)
    mic_sup, box_sup = simulate(voice_np, noise_np, ir, gain=gain,
                                model=model, notch_bank=notch_bank,
                                device=device, window=window)
    rms_sup = _rms_db(mic_sup)
    print(f'  mic RMS: {rms_sup:.1f} dB  (Δ {rms_sup - rms_raw:+.1f} dB vs raw)')
    print(f'  Notches held at end: {notch_bank.active_notches}')

    # ── Save audio ────────────────────────────────────────────────────────
    sf.write(str(out_dir / 'raw_feedback.wav'),   mic_raw, SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_mic.wav'), mic_sup, SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_out.wav'), box_sup, SR, subtype='PCM_16')
    print(f'\nAudio saved to {out_dir}/')
    print('  raw_feedback.wav   — what the mic hears with no suppression')
    print('  suppressed_mic.wav — what the mic hears with the model in the loop')
    print('  suppressed_out.wav — what goes to the speaker (post-notch)')

    _save_plot(mic_raw, mic_sup, box_sup, gain,
               out_dir / 'loop_comparison.png')


def _rms_db(x):
    return 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-8)


def _save_plot(raw, sup_mic, sup_out, gain, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import librosa, librosa.display

        sigs   = [raw,       sup_mic,          sup_out]
        titles = [
            f'A: raw mic (gain={gain}) — loop closes, rings up',
            'B: suppressed mic — what the mic hears with model in loop',
            'B: speaker output — what the notch passes to the PA',
        ]
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for ax, sig, title in zip(axes, sigs, titles):
            db = librosa.amplitude_to_db(
                np.abs(librosa.stft(sig, n_fft=N_FFT, hop_length=HOP)),
                ref=np.max)
            librosa.display.specshow(db, sr=SR, hop_length=HOP,
                                     x_axis='time', y_axis='hz',
                                     ax=ax, vmin=-80)
            ax.set_title(title)
            ax.set_ylim(0, 8000)
        plt.suptitle('Closed-loop feedback simulation')
        plt.tight_layout()
        plt.savefig(str(path), dpi=120)
        plt.close(fig)
        print(f'  Plot → {path.name}')
    except Exception:
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gain',       type=float, default=1.3,
                    help='Feedback loop gain (>1 = unstable, try 1.2–1.8)')
    ap.add_argument('--duration',   type=float, default=8.0,
                    help='Simulation length in seconds')
    ap.add_argument('--checkpoint', type=str,   default=None)
    ap.add_argument('--out-dir',    type=str,   default=None)
    args = ap.parse_args()
    run_eval(gain=args.gain, duration_s=args.duration,
             checkpoint=args.checkpoint, out_dir=args.out_dir)
