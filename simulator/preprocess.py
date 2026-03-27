"""
simulator/preprocess.py — Resample all training audio to 48kHz in-place.

Run once after downloading datasets and again after any new venue IR collection.
Converts stereo/multi-channel files to mono. Skips files already at 48kHz.
Safe to re-run: existing 48kHz files are untouched.

Usage:
    python simulator/preprocess.py              # process all directories
    python simulator/preprocess.py --dry-run    # show what would be changed, don't write
"""

import argparse, sys
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import librosa
except ImportError:
    sys.exit("librosa not installed — run: pip install librosa")

TARGET_SR   = 48000
AUDIO_EXTS  = {'.wav', '.flac', '.aiff', '.aif', '.ogg'}

PROJECT_ROOT = Path(__file__).parent.parent

DIRS_TO_PROCESS = [
    PROJECT_ROOT / 'data' / 'clean_vocals',
    PROJECT_ROOT / 'data' / 'venue_irs' / 'mains',
    PROJECT_ROOT / 'data' / 'venue_irs' / 'monitors',
    PROJECT_ROOT / 'data' / 'venue_irs' / 'combined',
    PROJECT_ROOT / 'data' / 'public_irs',
    PROJECT_ROOT / 'data' / 'noise',
]


def process_file(path: Path, dry_run: bool) -> str:
    """
    Resample a single audio file to TARGET_SR in-place.
    Returns a status string: 'skipped', 'resampled', or 'error: <msg>'.
    """
    try:
        info = sf.info(str(path))
    except Exception as e:
        return f"error: sf.info failed — {e}"

    if info.samplerate == TARGET_SR:
        return 'skipped'

    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception as e:
        return f"error: read failed — {e}"

    # Mix down to mono — take mean across channels
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    if dry_run:
        return f"would resample {sr}Hz → {TARGET_SR}Hz ({audio.shape[0]} samples)"

    # Resample using librosa (high-quality sinc resampling)
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Preserve bit depth where possible
    subtype = 'PCM_16'
    if info.subtype in ('PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'):
        subtype = 'PCM_24'

    try:
        sf.write(str(path), resampled, TARGET_SR, subtype=subtype)
    except Exception as e:
        return f"error: write failed — {e}"

    return f"resampled {sr}Hz → {TARGET_SR}Hz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Show what would be changed without writing files')
    args = ap.parse_args()

    counts = {'skipped': 0, 'resampled': 0, 'error': 0}

    for d in DIRS_TO_PROCESS:
        if not d.exists():
            print(f"[skip] {d} — does not exist yet")
            continue

        files = [f for f in d.rglob('*') if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            print(f"[empty] {d}")
            continue

        print(f"\n{d.relative_to(PROJECT_ROOT)}  ({len(files)} files)")
        for f in tqdm(files, desc=d.name, unit='file'):
            status = process_file(f, args.dry_run)
            if status == 'skipped':
                counts['skipped'] += 1
            elif status.startswith('error'):
                counts['error'] += 1
                print(f"  ERROR {f.name}: {status}")
            else:
                counts['resampled'] += 1
                if args.dry_run:
                    print(f"  {f.name}: {status}")

    print(f"\nDone: {counts['resampled']} resampled, "
          f"{counts['skipped']} already at {TARGET_SR}Hz, "
          f"{counts['error']} errors")
    if counts['error']:
        sys.exit(1)


if __name__ == '__main__':
    main()
