"""
eval/generate_test_set.py — Generate 7 targeted listening test scenarios.

Writes to data/listening_test/ — each scenario folder contains:
  mic.wav   — microphone input (feedback + room reverb + noise)
  clean.wav — ground truth (reverberant vocal)
  ref.wav   — reference signal (what the PA plays; used by run_inference.py)

After running this, generate enhanced.wav files with:
  python eval/run_inference.py --val-dir data/listening_test/<scenario> \\
                                --out-dir data/listening_test/<scenario>

Or all at once (bash):
  for d in data/listening_test/*/; do
      python eval/run_inference.py --val-dir "$d" --out-dir "$d"
  done
"""

import sys
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'simulator'))

from generate_pairs import build_room_simulation

SR = 48000

SCENARIOS = [
    {'name': '1_loud_feedback',     'mains_gain': 0.80, 'mon_gain': 0.75, 'rt60': 0.6},
    {'name': '2_moderate_feedback', 'mains_gain': 0.50, 'mon_gain': 0.45, 'rt60': 0.6},
    {'name': '3_reverberant_room',  'mains_gain': 0.55, 'mon_gain': 0.50, 'rt60': 2.2},
    {'name': '4_dry_direct',        'mains_gain': 0.70, 'mon_gain': 0.60, 'rt60': 0.2},
    {'name': '5_sibilant_vocal',    'mains_gain': 0.60, 'mon_gain': 0.55, 'rt60': 0.5},
    {'name': '6_sustained_vowel',   'mains_gain': 0.65, 'mon_gain': 0.60, 'rt60': 0.5},
    {'name': '7_multi_freq_fb',     'mains_gain': 0.75, 'mon_gain': 0.80, 'rt60': 0.8},
]


def synthetic_room_ir(rt60: float, sr: int = SR) -> np.ndarray:
    """
    Exponentially-decaying white noise RIR for room reverb of the direct vocal.
    Peak-normalized — correct for a room IR where direct path has unit gain.
    Do NOT use for feedback path IRs (those need physically calibrated gain from pra).
    """
    n   = int(rt60 * sr)
    t   = np.arange(n) / sr
    ir  = np.random.randn(n).astype(np.float32)
    ir *= np.exp(-6.9 * t / rt60).astype(np.float32)
    ir /= np.abs(ir).max() + 1e-8
    return ir


def generate_test_set(vocal_dir='data/clean_vocals', out_dir='data/listening_test'):
    out          = PROJECT_ROOT / out_dir
    vocal_root   = PROJECT_ROOT / vocal_dir
    vocal_files  = [
        f for f in vocal_root.rglob('*.wav')
        if (info := sf.info(str(f))).frames / info.samplerate >= 3.0
    ]
    assert vocal_files, f"No vocal files >= 3s found in {vocal_dir}"

    # Fixed seeds — reproducible test set across runs
    random.seed(42)
    np.random.seed(42)   # build_room_simulation() uses np.random internally

    print(f"Generating {len(SCENARIOS)} scenarios from {len(vocal_files)} vocal files...")

    for sc in SCENARIOS:
        sc_dir = out / sc['name']
        sc_dir.mkdir(parents=True, exist_ok=True)

        vocal_path = random.choice(vocal_files)
        vocal, _   = sf.read(str(vocal_path), dtype='float32', always_2d=False)
        if vocal.ndim > 1:
            vocal = vocal.mean(axis=1)
        target_len = 3 * SR
        start  = random.randint(0, len(vocal) - target_len)
        vocal  = vocal[start:start + target_len]

        # Feedback path IRs — full physics simulator for correct acoustic attenuation.
        # build_room_simulation() models directional loudspeaker→mic paths via
        # pyroomacoustics ISM with physically realistic gain (typically -20 to -40dB).
        mains_ir, mon_ir, _, _, _meta = build_room_simulation()
        # Override room reverb with scenario-specific RT60 (exponential decay proxy)
        room_ir = synthetic_room_ir(sc['rt60'])

        reverberant_vocal = fftconvolve(vocal, room_ir)[:target_len]
        mains_fb          = fftconvolve(reverberant_vocal, mains_ir)[:target_len] * sc['mains_gain']
        mon_fb            = fftconvolve(reverberant_vocal, mon_ir)[:target_len]   * sc['mon_gain']

        mic_signal = (reverberant_vocal + mains_fb + mon_fb).astype(np.float32)
        clean      = reverberant_vocal.astype(np.float32)
        ref_signal = reverberant_vocal.astype(np.float32)

        # Normalise to -18 dBFS peak (same scale for all three files)
        peak  = max(np.abs(mic_signal).max(), np.abs(clean).max(), 1e-8)
        scale = 0.125 / peak   # 0.125 ≈ -18 dBFS
        sf.write(str(sc_dir / 'mic.wav'),   mic_signal * scale, SR)
        sf.write(str(sc_dir / 'clean.wav'), clean      * scale, SR)
        sf.write(str(sc_dir / 'ref.wav'),   ref_signal * scale, SR)
        print(f"  {sc['name']} — mains_gain={sc['mains_gain']}, rt60={sc['rt60']}s")

    print(f"\nDone. Run eval/run_inference.py to generate enhanced.wav files.")
    print("Then A/B mic / clean / enhanced in each scenario folder.")


if __name__ == '__main__':
    generate_test_set()
