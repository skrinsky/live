"""
eval/score.py — Objective metrics for FDKFNet evaluation.

Compares enhanced_*.wav against clean_*.wav (HPF'd reverberant vocal) from the val set.
PESQ requires 16kHz — files are downsampled before scoring.
STOI supports 48kHz natively.

Usage:
    python eval/score.py
    python eval/score.py --enhanced-dir data/eval_output --clean-dir data/training_pairs/val
"""

import argparse
import sys
import soundfile as sf
import numpy as np
from pathlib import Path

try:
    from pesq import pesq
    _HAS_PESQ = True
except ImportError:
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


def evaluate(enhanced_dir=None, clean_dir=None):
    enhanced_dir = Path(enhanced_dir or PROJECT_ROOT / 'data' / 'eval_output')
    clean_dir    = Path(clean_dir    or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')

    if not _HAS_PESQ:
        print("Warning: pesq not installed — skipping PESQ. pip install pesq")
    if not _HAS_STOI:
        print("Warning: pystoi not installed — skipping STOI. pip install pystoi")

    scores = {'pesq': [], 'stoi': []}

    for clean_path in sorted(clean_dir.glob('clean_*.wav')):
        parts = clean_path.stem.split('_', 1)
        if len(parts) < 2:
            # Listening test files are named 'clean.wav' — no suffix index, skip here
            continue
        idx           = parts[1]
        enhanced_path = enhanced_dir / f'enhanced_{idx}.wav'
        if not enhanced_path.exists():
            print(f"  [SKIP] missing enhanced_{idx}.wav in {enhanced_dir}")
            continue

        clean_48,    _ = sf.read(str(clean_path),    dtype='float32')
        enhanced_48, _ = sf.read(str(enhanced_path), dtype='float32')

        min_len     = min(len(clean_48), len(enhanced_48))
        clean_48    = clean_48[:min_len]
        enhanced_48 = enhanced_48[:min_len]

        if _HAS_PESQ and _HAS_LIBROSA:
            try:
                clean_16    = _to_pesq_sr(clean_48)
                enhanced_16 = _to_pesq_sr(enhanced_48)
                scores['pesq'].append(pesq(PESQ_SR, clean_16, enhanced_16, 'wb'))
            except Exception as e:
                print(f"  PESQ error on {idx}: {e}")

        if _HAS_STOI:
            try:
                scores['stoi'].append(stoi(clean_48, enhanced_48, SR))
            except Exception as e:
                print(f"  STOI error on {idx}: {e}")

    if not scores['pesq'] and not scores['stoi']:
        print("No scored files found. Check that enhanced_*.wav files exist in --enhanced-dir.")
        return scores

    if scores['pesq']:
        print(f"PESQ  (wideband, scored at {PESQ_SR//1000}kHz): "
              f"{np.mean(scores['pesq']):.3f}  "
              f"(n={len(scores['pesq'])}, "
              f"min={np.min(scores['pesq']):.3f}, "
              f"max={np.max(scores['pesq']):.3f})")
        print(f"  Target: >2.5  (input baseline typically ~1.5–2.0)")

    if scores['stoi']:
        print(f"STOI  (scored at {SR//1000}kHz):           "
              f"{np.mean(scores['stoi']):.3f}  "
              f"(n={len(scores['stoi'])})")
        print(f"  Target: >0.85")

    return scores


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--enhanced-dir', default=None)
    p.add_argument('--clean-dir',    default=None)
    args = p.parse_args()
    evaluate(args.enhanced_dir, args.clean_dir)
