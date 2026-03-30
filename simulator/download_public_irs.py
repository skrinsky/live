"""
simulator/download_public_irs.py — Download public IR datasets to data/public_irs/.

Sources (all open-licensed):
  1. EchoThief (free, non-commercial)  — 115 real spaces: studios, halls, churches
     Direct zip download from echothief.com
  2. ARNI (CC BY 4.0)                  Zenodo 6985104
     Variable acoustics lab — 132K IRs, sample 500 spread by RT60
  3. Aachen AIR (MIT)                  direct zip from RWTH Aachen
  4. OpenAIR (CC per-space)            manual download — openairlib.net

Usage:
    python simulator/download_public_irs.py              # all automated sources
    python simulator/download_public_irs.py --skip-arni  # skip the large ARNI download
"""

import sys
import re
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
    'arni':         [ 7.3,  6.2,  2.7],   # ARNI variable-acoustics lab (Aalto)
    'aachen':       [30.0, 13.0, 10.0],   # Aachen AIR — Aula Carolina great hall
    # EchoThief spaces have no published dimensions — resonators use RT60 fallback
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


# ── EchoThief ─────────────────────────────────────────────────────────────────

def download_echothief():
    """
    EchoThief — 115 real spaces (recording studios, concert halls, churches,
    stairwells, tunnels). Free for non-commercial use.
    http://www.echothief.com
    """
    print('\n=== EchoThief — 115 real spaces ===')
    sub = OUT_DIR / 'echothief'
    sub.mkdir(exist_ok=True)

    existing = list(sub.rglob('*.wav'))
    if existing:
        print(f'  Already have {len(existing)} files, skipping')
        return existing

    url  = 'https://www.echothief.com/wp-content/uploads/2024/07/EchoThiefImpulseResponseLibrary.zip'
    dest = sub / 'EchoThief.zip'
    download_file(url, dest, desc='EchoThief IR library')
    print('  Extracting ...')
    with zipfile.ZipFile(dest) as z:
        z.extractall(sub)
    dest.unlink()
    wavs = list(sub.rglob('*.wav'))
    print(f'  EchoThief: {len(wavs)} wav files in {sub}')
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


# ── OpenAIR ───────────────────────────────────────────────────────────────────

OPENAIR_PAGE_IDS = [
    406, 1293, 416, 425, 435, 1413, 441, 1346, 459, 1167, 468, 476, 1323,
    483, 494, 502, 508, 518, 1565, 525, 571, 577, 584, 595, 602, 611, 1361,
    626, 632, 638, 652, 659, 665, 670, 683, 715, 688, 696, 702, 709, 722,
    729, 1516, 678, 1274, 735, 740, 452, 644, 1543, 745, 752, 764, 770,
    776, 782, 790, 797,
]

def _scrape_openair_zip_url(page_id):
    """Fetch an OpenAIR space page and return the webfiles.york.ac.uk ZIP URL."""
    url = f'https://www.openair.hosted.york.ac.uk/?page_id={page_id}'
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        match = re.search(r'https://webfiles\.york\.ac\.uk/OPENAIR/IRs/[^"\'>\s]+\.zip', r.text)
        return match.group(0) if match else None
    except Exception:
        return None


def download_openair():
    """
    Download all 59 OpenAIR spaces by scraping each space page for its ZIP URL.
    Includes churches, cathedrals, concert halls, and performance spaces.
    """
    print('\n=== OpenAIR — 59 spaces (churches, halls, cathedrals) ===')
    sub = OUT_DIR / 'openair'
    sub.mkdir(exist_ok=True)

    existing = list(sub.rglob('*.wav'))
    if len(existing) > 100:
        print(f'  Already have {len(existing)} files, skipping')
        return existing

    downloaded = 0
    failed     = []
    for i, page_id in enumerate(OPENAIR_PAGE_IDS):
        zip_url = _scrape_openair_zip_url(page_id)
        if not zip_url:
            failed.append(page_id)
            print(f'  [{i+1}/{len(OPENAIR_PAGE_IDS)}] page_id={page_id} — no ZIP link found')
            continue

        slug     = zip_url.split('/')[-2]
        zip_dest = sub / f'{slug}.zip'
        space_dir = sub / slug

        if space_dir.exists() and list(space_dir.rglob('*.wav')):
            print(f'  [{i+1}/{len(OPENAIR_PAGE_IDS)}] {slug} already extracted')
            downloaded += 1
            continue

        try:
            print(f'  [{i+1}/{len(OPENAIR_PAGE_IDS)}] {slug}', end=' ', flush=True)
            download_file(zip_url, zip_dest)
            space_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_dest) as z:
                z.extractall(space_dir)
            zip_dest.unlink()
            downloaded += 1
        except Exception as e:
            print(f'  Warning: {slug} failed: {e}')
            failed.append(page_id)

    wavs = list(sub.rglob('*.wav'))
    print(f'  OpenAIR: {downloaded} spaces downloaded, {len(wavs)} wav files')
    if failed:
        print(f'  Failed page IDs: {failed}')
    return wavs


# ── Save room dimensions metadata ─────────────────────────────────────────────

def save_dimensions():
    meta_path = OUT_DIR / 'room_dimensions.json'
    with open(meta_path, 'w') as f:
        json.dump(ROOM_DIMENSIONS, f, indent=2)
    print(f'\nRoom dimensions saved to {meta_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-arni',      action='store_true',
                    help='Skip ARNI (large download — ~2GB for 500-file sample)')
    ap.add_argument('--skip-echothief', action='store_true')
    ap.add_argument('--skip-aachen',    action='store_true')
    ap.add_argument('--skip-openair',   action='store_true')
    ap.add_argument('--arni-n',     type=int, default=500,
                    help='Number of ARNI IRs to sample (default 500)')
    args = ap.parse_args()

    total = []
    if not args.skip_echothief:
        total += download_echothief()
    if not args.skip_openair:
        total += download_openair()
    if not args.skip_aachen:
        total += download_aachen()
    if not args.skip_arni:
        total += download_arni(n_target=args.arni_n)

    save_dimensions()

    print(f'\n=== Done — {len(total)} IR files in {OUT_DIR} ===')
    print('\nThen run:')
    print('  python simulator/preprocess.py   # resample all to 48kHz')
    print('  python feedback_mask/train.py    # training will pick up new IRs automatically')


if __name__ == '__main__':
    main()
