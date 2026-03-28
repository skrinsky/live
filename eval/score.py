"""
eval/score.py — Objective metrics for FDKFNet evaluation.

Compares enhanced_*.wav against clean_*.wav (HPF'd reverberant vocal) from the val set.
Also computes mic baseline (unprocessed input vs clean) so you can see the improvement.
PESQ requires 16kHz — files are downsampled before scoring.
STOI supports 48kHz natively.

Usage:
    python eval/score.py
    python eval/score.py --enhanced-dir data/eval_output --clean-dir data/training_pairs/val
    python eval/score.py --enhanced-dir data/listening_test --clean-dir data/listening_test
"""

import argparse
import sys
import soundfile as sf
import numpy as np
from pathlib import Path

try:
    from pesq import pesq
    _HAS_PESQ = True
except (ImportError, Exception):
    _HAS_PESQ = False

try:
    from pystoi import stoi
    _HAS_STOI = True
except ImportError:
    _HAS_STOI = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

PROJECT_ROOT = Path(__file__).parent.parent
SR           = 48000
PESQ_SR      = 16000   # pesq library hard requirement — only 8kHz (NB) and 16kHz (WB) supported


def _to_pesq_sr(audio: np.ndarray) -> np.ndarray:
    if not _HAS_LIBROSA:
        raise ImportError("librosa required for PESQ scoring — pip install librosa")
    return librosa.resample(audio.astype(np.float32), orig_sr=SR, target_sr=PESQ_SR)


def _score_pair(clean_48, est_48, label, scores):
    """Score one clean/estimate pair, append to scores dict. Returns stoi score or None."""
    min_len  = min(len(clean_48), len(est_48))
    clean_48 = clean_48[:min_len]
    est_48   = est_48[:min_len]

    if _HAS_PESQ and _HAS_LIBROSA:
        try:
            clean_16 = _to_pesq_sr(clean_48)
            est_16   = _to_pesq_sr(est_48)
            scores['pesq'].append(pesq(PESQ_SR, clean_16, est_16, 'wb'))
        except Exception as e:
            if 'pesq' not in scores.get('_pesq_warned', set()):
                print(f"  PESQ error ({e}) — skipping PESQ for all files")
                scores.setdefault('_pesq_warned', set()).add('pesq')

    if _HAS_STOI:
        try:
            val = stoi(clean_48, est_48, SR)
            scores['stoi'].append(val)
            return val
        except Exception as e:
            print(f"  STOI error on {label}: {e}")
    return None


def evaluate(enhanced_dir=None, clean_dir=None):
    enhanced_dir = Path(enhanced_dir or PROJECT_ROOT / 'data' / 'eval_output')
    clean_dir    = Path(clean_dir    or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')

    if not _HAS_STOI:
        print("Warning: pystoi not installed — skipping STOI. pip install pystoi")

    scores     = {'pesq': [], 'stoi': []}
    mic_scores = {'stoi': []}   # unprocessed mic baseline

    # --- Indexed val-set format: clean_000001.wav / mic_000001.wav / enhanced_000001.wav ---
    for clean_path in sorted(clean_dir.glob('clean_*.wav')):
        idx           = clean_path.stem.split('_', 1)[1]
        enhanced_path = enhanced_dir / f'enhanced_{idx}.wav'
        if not enhanced_path.exists():
            print(f"  [SKIP] missing enhanced_{idx}.wav in {enhanced_dir}")
            continue
        clean_48,    _ = sf.read(str(clean_path),    dtype='float32')
        enhanced_48, _ = sf.read(str(enhanced_path), dtype='float32')
        _score_pair(clean_48, enhanced_48, idx, scores)

        mic_path = clean_dir / f'mic_{idx}.wav'
        if mic_path.exists():
            mic_48, _ = sf.read(str(mic_path), dtype='float32')
            _score_pair(clean_48, mic_48, f'mic_{idx}', mic_scores)

    # --- Listening-test format: scenario_dir/clean.wav + mic.wav + enhanced.wav ---
    for clean_path in sorted(clean_dir.rglob('clean.wav')):
        enhanced_path = clean_path.parent / 'enhanced.wav'
        if not enhanced_path.exists():
            print(f"  [SKIP] no enhanced.wav in {clean_path.parent.name}/")
            continue
        clean_48,    _ = sf.read(str(clean_path),    dtype='float32')
        enhanced_48, _ = sf.read(str(enhanced_path), dtype='float32')
        label = clean_path.parent.name
        _score_pair(clean_48, enhanced_48, label, scores)

        mic_path = clean_path.parent / 'mic.wav'
        if mic_path.exists() and _HAS_STOI:
            mic_48, _ = sf.read(str(mic_path), dtype='float32')
            mic_val = _score_pair(clean_48, mic_48, f'mic_{label}', mic_scores)

    if not scores['pesq'] and not scores['stoi']:
        print("No scored files found.")
        print("  Listening test: python eval/score.py --enhanced-dir data/listening_test "
              "--clean-dir data/listening_test")
        return scores

    # --- Print results ---
    if mic_scores['stoi'] and scores['stoi']:
        mic_mean = np.mean(mic_scores['stoi'])
        enh_mean = np.mean(scores['stoi'])
        print(f"\nSTOI  (48kHz):")
        print(f"  Mic (unprocessed): {mic_mean:.3f}")
        print(f"  Enhanced (model):  {enh_mean:.3f}  (+{enh_mean - mic_mean:+.3f})")
        print(f"  Target: >0.85")
    elif scores['stoi']:
        print(f"STOI  (scored at {SR//1000}kHz):  "
              f"{np.mean(scores['stoi']):.3f}  (n={len(scores['stoi'])})")
        print(f"  Target: >0.85")

    if scores['pesq']:
        print(f"\nPESQ  (wideband, {PESQ_SR//1000}kHz):  "
              f"{np.mean(scores['pesq']):.3f}  "
              f"(min={np.min(scores['pesq']):.3f}, max={np.max(scores['pesq']):.3f})")
        print(f"  Target: >2.5")

    return scores


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--enhanced-dir', default=None)
    p.add_argument('--clean-dir',    default=None)
    args = p.parse_args()
    evaluate(args.enhanced_dir, args.clean_dir)
