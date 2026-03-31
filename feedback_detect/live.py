"""
feedback_detect/live.py — Real-time feedback suppressor: detector + parametric notch.

Signal flow:
  mic input  →  STFT magnitude  →  FeedbackDetector  →  detected frequencies
                                                               ↓
  mic input  ──────────────────────────────────────→  NotchBank.process()  →  output

The neural model runs on the STFT to decide WHAT to notch.
The notch filters run on the raw time-domain signal — no STFT artifacts,
no ISTFT, lower latency than the feedback_mask approach.

Usage:
    python feedback_detect/live.py
    python feedback_detect/live.py --threshold 0.4 --q 40
"""

import sys
import argparse
import numpy as np
import torch
import sounddevice as sd
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_detect'))

from model import FeedbackDetector, SR, N_FFT, HOP, N_FREQ
from notch import NotchBank

# ── Defaults ───────────────────────────────────────────────────────────────────
CHECKPOINT    = PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'
DETECT_THRESH = 0.5      # probability threshold to declare a bin in feedback
MIN_FREQ_HZ   = 80.0     # ignore feedback detections below this (HPF region)
NOTCH_Q       = 30.0     # Q for biquad notches
NOTCH_DEPTH   = -24.0    # dB depth of each notch
BLOCK_SIZE    = HOP      # samples per sounddevice callback (= one STFT frame)


def run(threshold=DETECT_THRESH, q=NOTCH_Q, depth_db=NOTCH_DEPTH, checkpoint=None):
    ckpt_path = Path(checkpoint or CHECKPOINT)
    assert ckpt_path.exists(), f'No checkpoint at {ckpt_path} — train first.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model = FeedbackDetector().to(device).eval()
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    print(f'FeedbackDetector loaded from {ckpt_path}')

    notch_bank   = NotchBank(sr=SR, q=q, depth_db=depth_db)
    hpf_sos      = butter(2, 90.0 / (SR / 2), btype='high', output='sos')

    # Analysis buffer: accumulate BLOCK_SIZE chunks into N_FFT window
    analysis_buf = np.zeros(N_FFT, dtype=np.float32)
    # HPF state
    hpf_zi       = np.zeros((2, 1))
    # GRU hidden state — persists across callbacks
    gru_h        = None
    # Log-magnitude history for delta features (last 10 frames per bin)
    lm_history   = np.zeros((N_FREQ, 11), dtype=np.float32)

    bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)   # Hz per bin

    def callback(indata, outdata, frames, time, status):
        nonlocal analysis_buf, hpf_zi, gru_h, lm_history

        # ── Pre-processing ────────────────────────────────────────────────
        block = indata[:, 0].copy()                             # mono
        block_hpf, hpf_zi = sosfilt(hpf_sos, block, zi=hpf_zi)
        block_hpf = block_hpf.astype(np.float32)

        # Shift analysis buffer and add new block
        analysis_buf = np.roll(analysis_buf, -BLOCK_SIZE)
        analysis_buf[-BLOCK_SIZE:] = block_hpf

        # ── Detection ─────────────────────────────────────────────────────
        buf_t   = torch.from_numpy(analysis_buf).unsqueeze(0).to(device)
        stft    = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window,
                             return_complex=True)           # (1, N_FREQ, 1)
        mag     = stft.abs()                                # (1, N_FREQ, 1)

        # Build delta features from history buffer.
        # torch.roll is useless with T=1 (delta=0 always), so we maintain
        # lm_history manually.  lm_history[:, k] = log_mag from k frames ago.
        lm_now  = torch.log(mag[0, :, 0] + 1e-8).cpu().numpy()  # (N_FREQ,)
        lm_history = np.roll(lm_history, 1, axis=1)
        lm_history[:, 0] = lm_now

        feat_np = np.stack([
            lm_history[:, 0],
            lm_history[:, 0] - lm_history[:, 1],
            lm_history[:, 0] - lm_history[:, 4],
            lm_history[:, 0] - lm_history[:, 10],
        ], axis=0)                                          # (N_DELTA, N_FREQ)
        features = torch.from_numpy(feat_np).to(device)
        features = features.unsqueeze(0).unsqueeze(-1)     # (1, N_DELTA, N_FREQ, 1)

        with torch.no_grad():
            prob, gru_h = model(features, gru_h)           # (1, N_FREQ, 1)

        prob_np = prob[0, :, 0].cpu().numpy()              # (N_FREQ,)

        # Feedback bins: probability above threshold, above min frequency
        above_thresh = (prob_np > threshold) & (bin_freqs >= MIN_FREQ_HZ)
        detected_freqs = _cluster_bins(bin_freqs, prob_np, above_thresh)

        # ── Notch bank ────────────────────────────────────────────────────
        notch_bank.update(detected_freqs)
        processed = notch_bank.process(block_hpf)

        # ── Output ────────────────────────────────────────────────────────
        outdata[:, 0] = processed
        if outdata.shape[1] > 1:
            outdata[:, 1] = processed

        if notch_bank.active_notches:
            notch_str = ', '.join(f'{f:.0f}Hz/{d:.0f}dB'
                                  for f, d in notch_bank.active_notches)
            det_str   = ', '.join(f'{f:.0f}' for f in detected_freqs) or '—'
            print(f'\rDetected: [{det_str}]   Active: [{notch_str}]    ', end='')

    print(f'Running at {SR} Hz, block={BLOCK_SIZE} samples ({1000*BLOCK_SIZE/SR:.1f} ms)')
    print(f'Detection threshold={threshold}, Q={q}, depth={depth_db} dB')
    print('Press Ctrl+C to stop.\n')

    try:
        with sd.Stream(samplerate=SR, blocksize=BLOCK_SIZE,
                       dtype='float32', channels=1,
                       callback=callback):
            sd.sleep(10 * 3600 * 1000)
    except KeyboardInterrupt:
        print('\nStopped.')


def _cluster_bins(bin_freqs, prob, mask):
    """
    Find feedback frequencies from a per-bin probability mask.
    Groups adjacent positive bins and returns the peak-probability bin's
    frequency for each cluster.
    """
    if not mask.any():
        return []

    indices    = np.where(mask)[0]
    clusters   = []
    current    = [indices[0]]

    for idx in indices[1:]:
        if idx - current[-1] <= 3:   # bins within 3 = ~150 Hz gap → same cluster
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    freqs = []
    for cluster in clusters:
        peak_idx = cluster[np.argmax(prob[cluster])]
        freqs.append(float(bin_freqs[peak_idx]))
    return freqs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold',  type=float, default=DETECT_THRESH)
    ap.add_argument('--q',          type=float, default=NOTCH_Q)
    ap.add_argument('--depth',      type=float, default=NOTCH_DEPTH)
    ap.add_argument('--checkpoint', type=str,   default=None)
    args = ap.parse_args()
    run(threshold=args.threshold, q=args.q, depth_db=args.depth,
        checkpoint=args.checkpoint)
