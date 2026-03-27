"""
simulator/generate_ir_pool.py — Pre-compute synthetic IR pool for recursive training

Generates a fixed pool of acoustic IRs from build_room_simulation() once before
training. Training samples from this pool at runtime rather than calling pra per-step,
which avoids pyroomacoustics in the training worker loop entirely (much faster, no macOS
fork deadlock risk).

Each IR has random transducer coloration (mic FR × speaker FR) applied, weighted so
'flat' (identity) appears ~15% of the time regardless of library size.

Output files: data/ir_pool/mains_XXXXXX.wav, monitor_XXXXXX.wav, sub_XXXXXX.wav

Runtime: ~2–8 hours for 2000 IRs depending on max_order and hardware.
"""

import argparse
import random
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))   # allows: python simulator/generate_ir_pool.py

from generate_pairs import build_room_simulation
from transducer_frs import build_transducer_library

SR = 48000


def main():
    ap = argparse.ArgumentParser(description='Pre-compute IR pool for FDKFNet training')
    ap.add_argument('--n',   type=int, default=2000,
                    help='Number of IR sets to generate (default: 2000)')
    ap.add_argument('--out', default='data/ir_pool',
                    help='Output directory (default: data/ir_pool)')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load transducer FR library once — FIR design is not free (~100ms per curve)
    tx_lib    = build_transducer_library()
    mic_names = list(tx_lib['mics'].keys())
    spk_names = list(tx_lib['speakers'].keys())
    mic_firs  = list(tx_lib['mics'].values())
    spk_firs  = list(tx_lib['speakers'].values())

    # Weight 'flat' (identity) to ~15% regardless of library size.
    # Without this, flat's weight drops to 1/(N+1) ≈ 6% with 14 mic entries —
    # the model sees too little uncolored signal.
    FLAT_PROB = 0.15

    def _weighted_choice(names, firs):
        n = len(names)
        w = [FLAT_PROB if name == 'flat' else (1.0 - FLAT_PROB) / (n - 1)
             for name in names]
        return random.choices(firs, weights=w, k=1)[0]

    n_errors = 0

    for i in tqdm(range(args.n), desc='generating IRs'):
        try:
            mains_ir, mon_ir, _, sub_ir, _ = build_room_simulation()
        except Exception as e:
            print(f"\nWarning: build_room_simulation() failed on iteration {i}: {e}")
            n_errors += 1
            if n_errors / (i + 1) > 0.1:
                sys.exit(f"ERROR: >10% simulation failure rate after {i+1} iterations. "
                         f"Check pyroomacoustics installation.")
            continue

        # Apply random transducer coloration
        # Same mic for both paths (performer has one mic regardless of speaker type).
        # Independent speaker FIRs (mains ≠ monitors in most venues).
        mic_fir         = _weighted_choice(mic_names, mic_firs)
        mains_spk_fir   = _weighted_choice(spk_names, spk_firs)
        monitor_spk_fir = _weighted_choice(spk_names, spk_firs)

        mains_ir = fftconvolve(fftconvolve(mains_ir, mains_spk_fir), mic_fir).astype(np.float32)
        mon_ir   = fftconvolve(fftconvolve(mon_ir,   monitor_spk_fir), mic_fir).astype(np.float32)

        for name, ir in [('mains', mains_ir), ('monitor', mon_ir), ('sub', sub_ir)]:
            sf.write(str(out / f'{name}_{i:06d}.wav'), ir, SR)

    generated = args.n - n_errors
    print(f"\nDone: {generated}/{args.n} generated successfully, {n_errors} errors")
    print(f"Output: {out.resolve()}")
    print(f"Files: {len(list(out.glob('mains_*.wav')))} mains, "
          f"{len(list(out.glob('monitor_*.wav')))} monitor, "
          f"{len(list(out.glob('sub_*.wav')))} sub")


if __name__ == '__main__':
    main()
