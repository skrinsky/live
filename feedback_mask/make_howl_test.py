"""
feedback_mask/make_howl_test.py — Generate listening test files with real recursive feedback.

Creates three severity scenarios for each vocal+IR combination:
  ringing   — gain just below threshold (sustained tone, no runaway)
  howl      — gain just above threshold (slow buildup)
  squeal    — gain well above threshold (fast explosive howl, clips)

Output structure:
  data/howl_test/<scenario>/mic.wav    — input with real Larsen feedback
  data/howl_test/<scenario>/clean.wav  — clean reverberant vocal (ground truth)

Usage:
    python feedback_mask/make_howl_test.py
    python feedback_mask/make_howl_test.py --n-scenarios 6 --duration 8
"""

import sys
import argparse
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve, butter, sosfilt, lfilter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))
from model import SR

FEEDBACK_TRUNC = int(0.05 * SR)   # 50ms — resonant modes only


def _norm_ir(ir):
    """Normalise IR to unit spectral peak → gain=1.0 is exactly stability threshold."""
    peak = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    return ir / (peak + 1e-8)


def simulate(vocal_np, room_ir, feedback_ir, gain, duration_samps, hpf_sos):
    """
    Returns (mic_np, clean_np) for a given gain level.
    mic[n] = speech[n] + gain × Σ h[k] × mic[n-k]   — true recursive Larsen loop
    """
    # Reverberant clean vocal = target
    clean = fftconvolve(vocal_np, room_ir)[:duration_samps].astype(np.float32)
    clean = sosfilt(hpf_sos, clean).astype(np.float32)

    h = _norm_ir(feedback_ir[:FEEDBACK_TRUNC]) * gain
    if gain == 0.0:
        mic = clean.copy()
    else:
        a = np.concatenate([[1.0], -h.astype(np.float64)])
        mic = lfilter([1.0], a, clean.astype(np.float64)).astype(np.float32)

    mic = np.clip(mic, -1.0, 1.0)
    return mic, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenarios', type=int, default=4,
                    help='Number of vocal+IR combinations to generate')
    ap.add_argument('--duration',    type=float, default=6.0,
                    help='Clip duration in seconds')
    ap.add_argument('--out-dir',     default=None,
                    help='Output directory (default: data/howl_test)')
    args = ap.parse_args()

    duration_samps = int(args.duration * SR)
    out_root = Path(args.out_dir or PROJECT_ROOT / 'data' / 'howl_test')

    vocal_files  = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                    if not f.name.startswith('._') and '__MACOSX' not in str(f)]
    ir_pool_dir  = PROJECT_ROOT / 'data' / 'ir_pool'
    mains_irs    = list(ir_pool_dir.glob('mains_*.wav'))
    monitor_irs  = list(ir_pool_dir.glob('monitor_*.wav'))
    room_irs     = [f for f in (PROJECT_ROOT / 'data' / 'public_irs').rglob('*.wav')
                    if not f.name.startswith('._') and '__MACOSX' not in str(f)]

    assert vocal_files, 'No files in data/clean_vocals/'
    assert mains_irs,   'No mains IRs — run simulator/generate_ir_pool.py'

    # Filter vocals long enough
    vocal_files = [f for f in vocal_files
                   if sf.info(str(f)).frames / sf.info(str(f)).samplerate >= args.duration]
    assert vocal_files, f'No vocal files >= {args.duration}s'

    hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')

    scenarios = [
        ('ringing', 0.85),   # sub-threshold — sustained resonance, no runaway
        ('howl',    1.10),   # just above — slow exponential buildup
        ('squeal',  1.80),   # well above — fast explosive, clips hard
    ]

    for i in range(args.n_scenarios):
        vocal_np, sr = sf.read(str(random.choice(vocal_files)), dtype='float32')
        assert sr == SR
        if vocal_np.ndim > 1: vocal_np = vocal_np.mean(1)
        offset   = random.randint(0, max(0, len(vocal_np) - duration_samps - 1))
        vocal_np = vocal_np[offset:offset + duration_samps]

        # Combine mains + monitor into one feedback IR
        mains_ir,   _ = sf.read(str(random.choice(mains_irs)),   dtype='float32')
        monitor_ir, _ = sf.read(str(random.choice(monitor_irs)), dtype='float32')
        fb_ir = (mains_ir[:FEEDBACK_TRUNC] + monitor_ir[:FEEDBACK_TRUNC]) * 0.5

        if room_irs:
            room_ir, _ = sf.read(str(random.choice(room_irs)), dtype='float32')
            if room_ir.ndim > 1: room_ir = room_ir[:, random.randint(0, room_ir.shape[1] - 1)]
        else:
            t = np.arange(int(0.4 * SR)) / SR
            room_ir = np.random.randn(len(t)).astype(np.float32) * np.exp(-6.9 * t / 0.4)
            room_ir /= np.abs(room_ir).max() + 1e-8

        for name, gain in scenarios:
            mic_np, clean_np = simulate(vocal_np, room_ir, fb_ir, gain,
                                        duration_samps, hpf)
            out_dir = out_root / f'{i+1:02d}_{name}'
            out_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_dir / 'mic.wav'),   mic_np,   SR, subtype='PCM_16')
            sf.write(str(out_dir / 'ref.wav'),   clean_np, SR, subtype='PCM_16')  # PA send ≈ clean vocal
            sf.write(str(out_dir / 'clean.wav'), clean_np, SR, subtype='PCM_16')
            print(f'  {out_dir.name}/mic.wav   gain={gain}')

    print(f'\nDone — {args.n_scenarios * len(scenarios)} files → {out_root}/')
    print('Run inference on each scenario:')
    print('  for d in data/howl_test/*/; do')
    print('    python eval/run_inference.py --val-dir "$d" --out-dir "$d"          # FDKFNet')
    print('    python feedback_mask/run_inference.py --val-dir "$d" --out-dir "$d" # GTCRN')
    print('  done')


if __name__ == '__main__':
    main()
