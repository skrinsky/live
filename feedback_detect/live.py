"""
feedback_detect/live.py — Real-time feedback suppressor: detector + notch + predictor.

Signal flow:
  mic input  →  STFT magnitude  →  FeedbackDetector  →  detected frequencies
                                                               ↓
  mic input  ──────────────────────────────────────→  NotchBank.process()  →  output
                                         ↑
                              FeedbackPredictor (voice-conditioned pre-emption)

Usage:
    python feedback_detect/live.py
    python feedback_detect/live.py --threshold 0.25 --device 2
    python -m sounddevice          # list available audio devices
"""

import sys
import time
import argparse
import numpy as np
import torch
import sounddevice as sd
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_detect'))

from model            import FeedbackDetector, SR, N_FFT, HOP, N_FREQ
from notch            import NotchBank
from predictor        import FeedbackPredictor
from spectral_flatten import ChronicRingEQ, AdaptiveMakeupGain

# ── Defaults ───────────────────────────────────────────────────────────────────
CHECKPOINT    = PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'
PROFILE_PATH  = PROJECT_ROOT / 'data' / 'feedback_risk_profile.json'  # diagnostic only
DETECT_THRESH = 0.25
MIN_FREQ_HZ   = 80.0
NOTCH_DEPTH   = -48.0
BLOCK_SIZE    = HOP      # samples per callback = one STFT frame (10 ms at 48 kHz)

# Log-scale cluster tolerance — matches NotchBank.FREQ_TOL_RATIO
CLUSTER_TOL_RATIO = 0.18   # wide — merges all bins of the same ring into one detection


def _parabolic_peak(bin_freqs: np.ndarray, prob: np.ndarray, k: int) -> float:
    """
    Parabolic interpolation around bin k to find sub-bin peak frequency.
    Fits a parabola to prob[k-1], prob[k], prob[k+1] and returns the
    interpolated frequency at the parabola's peak.
    Falls back to bin_freqs[k] if k is at the edge or the parabola is flat.
    """
    if k <= 0 or k >= len(prob) - 1:
        return float(bin_freqs[k])
    y0, y1, y2 = float(prob[k - 1]), float(prob[k]), float(prob[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-10:
        return float(bin_freqs[k])
    delta = 0.5 * (y0 - y2) / denom          # fractional bin offset, in [-0.5, 0.5]
    delta = max(-0.5, min(0.5, delta))
    bin_spacing = bin_freqs[1] - bin_freqs[0]  # linear spacing (SR / N_FFT)
    return float(bin_freqs[k] + delta * bin_spacing)


def _cluster_bins(bin_freqs: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> list[float]:
    """
    Group above-threshold bins into clusters using log-scale frequency tolerance
    and return the parabolic-interpolated peak frequency per cluster.
    """
    if not mask.any():
        return []

    indices = np.where(mask)[0]
    clusters: list[list[int]] = []
    current = [indices[0]]

    for idx in indices[1:]:
        # Merge bins within CLUSTER_TOL_RATIO of the cluster's current peak freq
        cluster_freq = bin_freqs[current[np.argmax(prob[current])]]
        if abs(bin_freqs[idx] - cluster_freq) / cluster_freq < CLUSTER_TOL_RATIO:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    result = []
    for c in clusters:
        peak_bin = c[np.argmax(prob[c])]
        result.append(_parabolic_peak(bin_freqs, prob, peak_bin))
    return result


def run(threshold=DETECT_THRESH, depth_db=NOTCH_DEPTH,
        checkpoint=None, device=None, save_profile=None):

    ckpt_path    = Path(checkpoint or CHECKPOINT)
    assert ckpt_path.exists(), f'No checkpoint at {ckpt_path} — train first.'

    audio_device = device   # None → sounddevice default

    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(torch_device)

    model = FeedbackDetector().to(torch_device).eval()
    ckpt  = torch.load(str(ckpt_path), map_location=torch_device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    print(f'FeedbackDetector  loaded from {ckpt_path}')

    bin_freqs  = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    # Always start fresh — no profile loaded on boot (plug-and-play, no room assumptions)
    predictor   = FeedbackPredictor(bin_freqs, sr=SR, profile_path=None)
    notch_bank  = NotchBank(sr=SR, depth_db=depth_db)
    chronic_eq  = ChronicRingEQ(bin_freqs, sr=SR)
    makeup_gain = AdaptiveMakeupGain()

    hpf_sos    = butter(2, 90.0 / (SR / 2), btype='high', output='sos')

    # Per-session state (mutated inside callback via nonlocal)
    analysis_buf = np.zeros(N_FFT, dtype=np.float32)
    hpf_zi       = np.zeros((hpf_sos.shape[0], 2))
    gru_h        = None
    lm_history   = np.zeros((N_FREQ, 11), dtype=np.float32)

    def callback(indata, outdata, frames, time, status):
        nonlocal analysis_buf, hpf_zi, gru_h, lm_history

        # ── Pre-processing ────────────────────────────────────────────────
        block = indata[:, 0].copy()
        block_hpf, hpf_zi = sosfilt(hpf_sos, block, zi=hpf_zi)
        block_hpf = block_hpf.astype(np.float32)

        analysis_buf = np.roll(analysis_buf, -BLOCK_SIZE)
        analysis_buf[-BLOCK_SIZE:] = block_hpf

        # ── Detection ─────────────────────────────────────────────────────
        buf_t = torch.from_numpy(analysis_buf).unsqueeze(0).to(torch_device)
        stft  = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window,
                           center=False, return_complex=True)
        mag   = stft.abs()

        lm_now             = torch.log(mag[0, :, 0] + 1e-8).cpu().numpy()
        lm_history         = np.roll(lm_history, 1, axis=1)
        lm_history[:, 0]   = lm_now

        feat_np  = np.stack([
            lm_history[:, 0],
            lm_history[:, 0] - lm_history[:, 1],
            lm_history[:, 0] - lm_history[:, 4],
            lm_history[:, 0] - lm_history[:, 10],
        ], axis=0)
        features = torch.from_numpy(feat_np).to(torch_device).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            prob, gru_h = model(features, gru_h)

        prob_np = prob[0, :, 0].cpu().numpy()

        above         = (prob_np > threshold) & (bin_freqs >= MIN_FREQ_HZ) & (bin_freqs < SR / 2)
        detected_freqs = _cluster_bins(bin_freqs, prob_np, above)

        # ── Predictor (voice-conditioned pre-emption) ─────────────────────
        preemptive = predictor.update(
            mag[0, :, 0].cpu().numpy(), prob_np, notch_bank.active_notches)

        # ── Notch bank ────────────────────────────────────────────────────
        notch_bank.update(detected_freqs, bin_freqs, prob_np,
                          preemptive_freqs=preemptive)
        processed = notch_bank.process(block_hpf)


        # ── ChronicRingEQ (absorbed-ring overflow cuts) ───────────────────
        chronic_eq.update(prob_np, notch_bank.active_notches)
        processed = chronic_eq.process(processed)

        # ── Adaptive makeup gain ───────────────────────────────────────────
        gain_scalar = makeup_gain.update(detected_freqs, notch_bank.active_notches)
        processed   = np.clip(processed * gain_scalar, -1.0, 1.0)

        # ── Output ────────────────────────────────────────────────────────
        outdata[:, 0] = processed
        if outdata.shape[1] > 1:
            outdata[:, 1] = processed

        # Status line — always print so you can see what's happening
        notch_str = ', '.join(f'{f:.0f}Hz/{d:.0f}dB'
                              for f, d, _ in notch_bank.active_notches) or '—'
        det_str   = ', '.join(f'{f:.0f}' for f in detected_freqs) or '—'
        pre_str   = ', '.join(f'{f:.0f}' for f in preemptive)     or '—'
        peak_prob = float(prob_np.max())
        print(f'\rDet:[{det_str}]  Pre:[{pre_str}]  Notches:[{notch_str}]'
              f'  PeakP:{peak_prob:.2f}  Makeup:{makeup_gain.current_db:+.1f}dB    ',
              end='', flush=True)

    def _shutdown():
        print('\n')
        print(predictor.summary())
        if save_profile:
            profile_path = Path(save_profile)
            predictor.profile_path = profile_path
            predictor.save()
            print(f'Profile saved to {profile_path}')

    print(f'Running at {SR} Hz  block={BLOCK_SIZE} samples ({1000*BLOCK_SIZE/SR:.1f} ms)')
    print(f'threshold={threshold}  depth={depth_db} dB')
    print('Ctrl+C to stop.\n')

    try:
        with sd.Stream(samplerate=SR, blocksize=BLOCK_SIZE,
                       dtype='float32', channels=1,
                       device=audio_device,
                       callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        _shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold',  type=float, default=DETECT_THRESH,
                    help='detection probability threshold (default 0.25)')
    ap.add_argument('--depth',      type=float, default=NOTCH_DEPTH,
                    help='max notch depth dB (default -48)')
    ap.add_argument('--checkpoint',    type=str,   default=None)
    ap.add_argument('--save-profile',  type=str,   default=None,
                    help='if set, save learned risk profile to this path on exit (diagnostic)')
    ap.add_argument('--device',        type=int,   default=None,
                    help='sounddevice device index (default: system default)')
    args = ap.parse_args()
    run(threshold=args.threshold, depth_db=args.depth,
        checkpoint=args.checkpoint, device=args.device,
        save_profile=args.save_profile)
