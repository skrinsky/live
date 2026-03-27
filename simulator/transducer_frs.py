"""
simulator/transducer_frs.py — Transducer frequency response library

Loads mic and speaker FR curves from data/transducer_frs/mics/*.csv and
data/transducer_frs/speakers/*.csv and converts them to minimum-phase FIR filters.

The feedback loop gain at each frequency is the product of room IR × mic FR × speaker FR.
Without transducer coloration, the model only sees IRs driven by room modes and never
sees the characteristic 8kHz ring of an SM58 presence peak through a QSC K12, or the
2–4kHz coloration of a cheap PA cabinet.

CSV format (one header row, then data):
    freq_hz,magnitude_db
    100,-2.5
    200,-1.0
    ...

Data acquisition:
  Mics/speakers: digitize manufacturer spec sheet FR chart using WebPlotDigitizer
  (~5 min per curve). For wireless mics, measure the complete system (capsule +
  TX + RX). For speakers, use on-axis at 1m.
"""

import numpy as np
from scipy.signal import firwin2, minimum_phase
from scipy.ndimage import uniform_filter1d
from pathlib import Path


SR     = 48000
N_TAPS = 512   # ~10ms at 48kHz; minimum_phase() returns (N_TAPS+1)//2 = 256 taps


def load_fr_csv(path):
    """Load freq_hz,magnitude_db CSV. Returns (freqs_hz, magnitudes_db) arrays."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


def fr_to_fir(freqs_hz, magnitudes_db, sr=SR, n_taps=N_TAPS):
    """
    Convert a frequency response curve to a minimum-phase FIR filter.

    Minimum-phase ensures causality — a transducer cannot output energy before
    receiving the signal. Normalized to unity gain at 1kHz so it models coloration,
    not absolute level.

    NOTE: minimum_phase(h, 'homomorphic') returns a filter of length (len(h)+1)//2,
    NOT len(h). With n_taps=512, the output is 256 taps (~5.3ms at 48kHz). This is
    undocumented scipy behavior — do not assume output length == n_taps.
    """
    n_pts      = n_taps * 4
    f_grid     = np.linspace(0, sr / 2, n_pts)
    mag_interp = np.interp(f_grid, freqs_hz, 10 ** (magnitudes_db / 20))
    mag_interp = np.clip(mag_interp, 1e-4, None)   # floor at -80dB

    # 1/6-octave smoothing before firwin2 — removes WebPlotDigitizer digitization noise
    # and prevents Gibbs ringing from sharp spectral transitions in coarse FR data.
    smooth_hz   = 0.5 * f_grid[-1] * (2 ** (1/6) - 1)   # approx 1/6-oct bandwidth at Nyquist
    smooth_bins = max(1, int(smooth_hz / (f_grid[1] - f_grid[0])))
    mag_interp  = uniform_filter1d(mag_interp, size=smooth_bins)

    # Normalize at 1kHz — models coloration, not level
    ref_idx    = np.argmin(np.abs(f_grid - 1000))
    mag_interp /= mag_interp[ref_idx]

    f_norm     = f_grid / (sr / 2)
    f_norm[-1] = 1.0   # ensure exactly 1.0 at Nyquist for firwin2

    h = firwin2(n_taps, f_norm, mag_interp)
    h = minimum_phase(h, method='homomorphic')
    return h.astype(np.float32)


def build_transducer_library(frs_dir='data/transducer_frs', sr=SR):
    """
    Load all FR CSVs from data/transducer_frs/mics/ and data/transducer_frs/speakers/.
    Returns {'mics': {name: fir_np, ...}, 'speakers': {name: fir_np, ...}}.
    Always includes a 'flat' entry (identity filter = no transducer coloration).
    Call once at startup — FIR design is not free (~100ms per curve).
    """
    lib = {
        'mics':     {'flat': np.array([1.0], dtype=np.float32)},
        'speakers': {'flat': np.array([1.0], dtype=np.float32)},
    }
    frs_path = Path(frs_dir)
    for category in ('mics', 'speakers'):
        cat_dir = frs_path / category
        if not cat_dir.exists():
            continue
        for csv in sorted(cat_dir.glob('*.csv')):
            if csv.stem == 'flat':
                continue   # already present as identity
            try:
                freqs, mags = load_fr_csv(csv)
                lib[category][csv.stem] = fr_to_fir(freqs, mags, sr)
            except Exception as e:
                print(f"Warning: skipping {csv.name}: {e}")

    print(f"Transducer library: {len(lib['mics'])} mics, {len(lib['speakers'])} speakers")
    return lib
