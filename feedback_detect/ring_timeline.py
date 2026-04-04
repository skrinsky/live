"""
feedback_detect/ring_timeline.py — Print a per-second ringing timeline for a wav file.

Ringing shows up as a narrow spectral peak far above the local noise floor.
For each second we report:
  - peak peakiness (max bin / median of surrounding bins)
  - the frequency of the worst peak
  - a bar scaled to that peakiness

Usage:
    python feedback_detect/ring_timeline.py data/eval_loop_restore_v5/suppressed_out.wav
    python feedback_detect/ring_timeline.py data/eval_loop_restore_v5/suppressed_out.wav --compare data/eval_loop_restore_v5/clean_reference.wav
"""

import sys
import argparse
import numpy as np
import soundfile as sf

N_FFT   = 2048
HOP     = 512
SR_EXPECTED = 48000
NEIGHBOR_BINS = 20   # bins either side used to estimate local floor


def peakiness(mag: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Return (max_ratio, peak_freq_hz, per_bin_ratio) for a magnitude spectrum.
    ratio = mag[i] / median(mag[i-N:i+N])  — how much a bin sticks up above neighbours.
    """
    n = len(mag)
    ratio = np.ones(n, dtype=np.float32)
    for i in range(NEIGHBOR_BINS, n - NEIGHBOR_BINS):
        local = np.concatenate([mag[i-NEIGHBOR_BINS:i], mag[i+1:i+NEIGHBOR_BINS+1]])
        floor = np.median(local) + 1e-8
        ratio[i] = mag[i] / floor
    bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR_EXPECTED)
    peak_idx  = int(np.argmax(ratio))
    return float(ratio[peak_idx]), float(bin_freqs[peak_idx]), ratio


def analyse(path: str) -> list[tuple[float, float, float]]:
    """Returns list of (time_s, max_peakiness, peak_freq_hz) per second."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SR_EXPECTED:
        print(f'  warning: SR={sr}, expected {SR_EXPECTED}')

    window   = np.hanning(N_FFT).astype(np.float32)
    n_frames = (len(audio) - N_FFT) // HOP
    results  = []

    bucket_peak  = 0.0
    bucket_freq  = 0.0
    bucket_start = 0.0
    frames_per_s = SR_EXPECTED / HOP

    for i in range(n_frames):
        s    = i * HOP
        frame= audio[s:s + N_FFT] * window
        mag  = np.abs(np.fft.rfft(frame))
        pk, freq, _ = peakiness(mag)
        if pk > bucket_peak:
            bucket_peak = pk
            bucket_freq = freq
        t = s / SR_EXPECTED
        if t - bucket_start >= 1.0:
            results.append((bucket_start, bucket_peak, bucket_freq))
            bucket_start = t
            bucket_peak  = 0.0
            bucket_freq  = 0.0

    if bucket_peak > 0:
        results.append((bucket_start, bucket_peak, bucket_freq))
    return results


def bar(value: float, scale: float = 30.0, threshold: float = 6.0) -> str:
    filled = min(int((value / scale) * 30), 30)
    char   = '█' if value >= threshold else '░'
    return char * filled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--compare', default=None, help='clean reference wav to subtract baseline')
    p.add_argument('--threshold', type=float, default=6.0,
                   help='peakiness ratio above which a ring is flagged (default 6)')
    args = p.parse_args()

    print(f'\nAnalysing: {args.path}')
    results = analyse(args.path)

    baseline = {}
    if args.compare:
        print(f'Baseline:  {args.compare}')
        for t, pk, freq in analyse(args.compare):
            baseline[int(t)] = pk

    print(f'\n{"t":>5}  {"peakiness":>9}  {"freq":>7}   ring?\n' + '-' * 50)
    for t, pk, freq in results:
        bl  = baseline.get(int(t), 1.0)
        adj = pk / max(bl, 1.0)   # subtract baseline voice peakiness
        flag = '  ← RING' if pk >= args.threshold else ''
        b    = bar(pk, threshold=args.threshold)
        print(f'{t:5.1f}s  {pk:9.1f}  {freq:6.0f}Hz  {b}{flag}')


if __name__ == '__main__':
    main()
