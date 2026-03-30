"""
simulator/download_public_irs.py — Download public IR datasets to data/public_irs/.

Sources (all open-licensed, suitable for commercial use except where noted):
  1. C4DM — Queen Mary University London (CC BY-NC-SA)   Zenodo 6497436
     Great Hall (800-seat venue) + Octagon (Victorian domed hall)
  2. ARNI — Variable acoustics lab (CC BY 4.0)            Zenodo 6985104
     132K IRs from one room — sample 500 spread by RT60 range
  3. Aachen AIR — RWTH Aachen (MIT)                       manual URL
  4. OpenAIR — openairlib.net (CC, per-space license)     manual download

Usage:
    python simulator/download_public_irs.py              # all automated sources
    python simulator/download_public_irs.py --skip-arni  # skip the large ARNI download
"""

import sys
import argparse
import json
import random
import shutil
import zipfile
import tarfile
import tempfile
import requests
import numpy as np
import soundfile as sf
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR      = PROJECT_ROOT / 'data' / 'public_irs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Room dimensions for known spaces ──────────────────────────────────────────
# Used by _add_resonators() in train.py to place modal frequencies correctly.
# Format: [Lx, Ly, Lz] in metres. Approximate but physically grounded.

ROOM_DIMENSIONS = {
    'c4dm_great_hall':  [29.0, 15.0, 10.5],   # Queen Mary Great Hall
    'c4dm_octagon':     [20.0, 20.0, 14.0],   # Octagon (approx inscribed rect)
    'arni_lab':         [ 7.3,  6.2,  2.7],   # ARNI variable-acoustics lab
    'aachen_stairway':  [ 6.0,  3.5, 15.0],   # Aachen AIR stairway
    'aachen_aula':      [30.0, 13.0, 10.0],   # Aula Carolina great hall
}


def rt60_from_ir(ir, sr):
    """Estimate RT60 via Schroeder backward integration (T20 × 3)."""
    energy    = np.cumsum(ir[::-1] ** 2)[::-1]
    energy_db = 10 * np.log10(energy / (energy[0] + 1e-12) + 1e-12)
    t = np.arange(len(ir)) / sr
    try:
        idx_5  = np.where(energy_db < -5)[0][0]
        idx_35 = np.where(energy_db < -35)[0][0]
        return float((t[idx_35] - t[idx_5]) * 2)
    except IndexError:
        return None


def axial_modes(dims, c=343.0, n_max=3, f_max=4000):
    """
    Return list of axial mode frequencies for a rectangular room.
    dims: [Lx, Ly, Lz] in metres.
    """
    Lx, Ly, Lz = dims
    modes = set()
    for nx in range(n_max + 1):
        for ny in range(n_max + 1):
            for nz in range(n_max + 1):
                if nx == ny == nz == 0:
                    continue
                f = c / 2 * np.sqrt((nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2)
                if f <= f_max:
                    modes.add(round(f, 1))
    return sorted(modes)


def zenodo_files(record_id):
    """Return list of {key, links.self} dicts from a Zenodo record."""
    r = requests.get(f'https://zenodo.org/api/records/{record_id}', timeout=30)
    r.raise_for_status()
    return r.json()['files']


def download_file(url, dest, desc=''):
    """Stream-download url → dest, showing progress."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  Downloading {desc or dest.name} ...', end=' ', flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        done  = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  Downloading {desc or dest.name} ... '
                          f'{done/total*100:.0f}%', end='', flush=True)
    print(' done')
    return dest


# ── C4DM ──────────────────────────────────────────────────────────────────────

def download_c4dm():
    print('\n=== C4DM (Queen Mary) — Zenodo 6497436 ===')
    sub = OUT_DIR / 'c4dm'
    sub.mkdir(exist_ok=True)
    files = zenodo_files('6497436')
    for f in files:
        name = f['key']
        url  = f['links']['self']
        dest = sub / name
        if dest.exists():
            print(f'  {name} already exists, skipping')
            continue
        tmp = download_file(url, sub / ('_tmp_' + name), desc=name)
        tmp.rename(dest)
    # Extract any archives
    for arc in sub.glob('*.zip'):
        print(f'  Extracting {arc.name} ...')
        with zipfile.ZipFile(arc) as z:
            z.extractall(sub)
        arc.unlink()
    for arc in sub.glob('*.tar.gz'):
        print(f'  Extracting {arc.name} ...')
        with tarfile.open(arc) as t:
            t.extractall(sub)
        arc.unlink()
    wavs = list(sub.rglob('*.wav'))
    print(f'  C4DM: {len(wavs)} wav files in {sub}')
    return wavs


# ── ARNI ──────────────────────────────────────────────────────────────────────

def download_arni(n_target=500):
    """
    Sample n_target IRs from ARNI spread evenly across RT60 bins.
    Avoids over-representing any single panel configuration.
    """
    print(f'\n=== ARNI (Zenodo 6985104) — sampling {n_target} by RT60 ===')
    sub = OUT_DIR / 'arni'
    sub.mkdir(exist_ok=True)

    existing = list(sub.glob('*.wav'))
    if len(existing) >= n_target:
        print(f'  Already have {len(existing)} files, skipping')
        return existing

    print('  Fetching file list from Zenodo ...')
    files = zenodo_files('6985104')
    wav_files = [f for f in files if f['key'].endswith('.wav')]
    print(f'  Total ARNI files: {len(wav_files)}')

    # Sample a manageable subset to compute RT60 on before deciding which to keep
    # Download 2000 random files, compute RT60, bin, keep n_target
    sample_size = min(2000, len(wav_files))
    sampled     = random.sample(wav_files, sample_size)

    rt60s = []
    print(f'  Downloading {sample_size} files to compute RT60 ...')
    tmp_dir = sub / '_tmp'
    tmp_dir.mkdir(exist_ok=True)

    for i, f in enumerate(sampled):
        if i % 100 == 0:
            print(f'  [{i}/{sample_size}]', flush=True)
        dest = tmp_dir / f['key'].replace('/', '_')
        if not dest.exists():
            try:
                download_file(f['links']['self'], dest)
            except Exception as e:
                print(f'  Warning: {f["key"]} failed: {e}')
                continue
        try:
            ir, sr = sf.read(str(dest), dtype='float32')
            if ir.ndim > 1: ir = ir.mean(1)
            rt60 = rt60_from_ir(ir, sr)
            if rt60 is not None:
                rt60s.append((rt60, dest, f))
        except Exception:
            pass

    # Bin by RT60 and sample evenly
    bins    = np.arange(0.1, 3.0, 0.25)
    n_bins  = len(bins) - 1
    per_bin = max(1, n_target // n_bins)
    chosen  = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        bucket = [(rt, p, f) for rt, p, f in rt60s if lo <= rt < hi]
        chosen.extend(random.sample(bucket, min(per_bin, len(bucket))))

    print(f'  Keeping {len(chosen)} files across {n_bins} RT60 bins')
    for rt, tmp_path, _ in chosen:
        dest = sub / tmp_path.name
        shutil.copy(tmp_path, dest)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    wavs = list(sub.glob('*.wav'))
    print(f'  ARNI: {len(wavs)} files saved to {sub}')
    return wavs


# ── Aachen AIR ────────────────────────────────────────────────────────────────

def download_aachen():
    """
    Aachen AIR — try direct download, print manual instructions if unavailable.
    """
    print('\n=== Aachen AIR ===')
    sub = OUT_DIR / 'aachen'
    sub.mkdir(exist_ok=True)

    url = 'https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip'
    dest = sub / 'air_database.zip'

    if list(sub.glob('*.wav')):
        print(f'  Already have files in {sub}, skipping')
        return list(sub.glob('*.wav'))

    try:
        download_file(url, dest, desc='Aachen AIR database')
        print('  Extracting ...')
        with zipfile.ZipFile(dest) as z:
            z.extractall(sub)
        dest.unlink()
        wavs = list(sub.rglob('*.wav'))
        print(f'  Aachen AIR: {len(wavs)} files')
        return wavs
    except Exception as e:
        print(f'  Auto-download failed ({e})')
        print('  Manual download:')
        print('    https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/')
        print(f'  Extract to: {sub}/')
        return []


# ── Save room dimensions metadata ─────────────────────────────────────────────

def save_dimensions():
    meta_path = OUT_DIR / 'room_dimensions.json'
    with open(meta_path, 'w') as f:
        json.dump(ROOM_DIMENSIONS, f, indent=2)
    print(f'\nRoom dimensions saved to {meta_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-arni',  action='store_true',
                    help='Skip ARNI (large download — ~2GB for 500-file sample)')
    ap.add_argument('--skip-c4dm',  action='store_true')
    ap.add_argument('--skip-aachen', action='store_true')
    ap.add_argument('--arni-n',     type=int, default=500,
                    help='Number of ARNI IRs to sample (default 500)')
    args = ap.parse_args()

    total = []
    if not args.skip_c4dm:
        total += download_c4dm()
    if not args.skip_arni:
        total += download_arni(n_target=args.arni_n)
    if not args.skip_aachen:
        total += download_aachen()

    save_dimensions()

    print(f'\n=== Done — {len(total)} IR files in {OUT_DIR} ===')
    print('\nManual downloads still needed:')
    print('  OpenAIR: https://www.openairlib.net  → download per-space, extract to data/public_irs/openair/')
    print('\nThen run:')
    print('  python simulator/preprocess.py   # resample all to 48kHz')
    print('  python feedback_mask/train.py    # training will pick up new IRs automatically')


if __name__ == '__main__':
    main()
