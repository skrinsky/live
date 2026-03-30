"""
feedback_mask/mic_profiles.py — Parametric EQ approximations of 20 live vocal mics.

Each mic is defined as a list of EQ bands applied as cascaded biquad filters.
Applying a random mic response during training exposes the model to the spectral
coloration variety it will encounter in real deployments, helping close the
sim-to-real gap without needing measured impulse responses.

Band format: (type, freq_hz, gain_db, Q)
  type: 'peak' | 'lowshelf' | 'highshelf'

Formulas: Audio EQ Cookbook (R. Bristow-Johnson).
"""

import numpy as np
import random
from scipy.signal import sosfilt, sosfilt_zi


# ── Biquad design (Audio EQ Cookbook) ─────────────────────────────────────────

def _peak_sos(fc, gain_db, Q, sr):
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    b0 =  1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 =  1 - alpha * A
    a0 =  1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 =  1 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _lowshelf_sos(fc, gain_db, Q, sr):
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    cos_w = np.cos(w0)
    sq    = 2 * np.sqrt(A) * alpha
    b0 =    A * ((A+1) - (A-1)*cos_w + sq)
    b1 =  2*A * ((A-1) - (A+1)*cos_w)
    b2 =    A * ((A+1) - (A-1)*cos_w - sq)
    a0 =         (A+1) + (A-1)*cos_w + sq
    a1 =   -2 * ((A-1) + (A+1)*cos_w)
    a2 =         (A+1) + (A-1)*cos_w - sq
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _highshelf_sos(fc, gain_db, Q, sr):
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    cos_w = np.cos(w0)
    sq    = 2 * np.sqrt(A) * alpha
    b0 =    A * ((A+1) + (A-1)*cos_w + sq)
    b1 = -2*A * ((A-1) + (A+1)*cos_w)
    b2 =    A * ((A+1) + (A-1)*cos_w - sq)
    a0 =         (A+1) - (A-1)*cos_w + sq
    a1 =    2 * ((A-1) - (A+1)*cos_w)
    a2 =         (A+1) - (A-1)*cos_w - sq
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _make_sos(bands, sr):
    """Convert list of (type, fc, gain_db, Q) to stacked SOS array."""
    sections = []
    for btype, fc, gain_db, Q in bands:
        if btype == 'peak':      sections.append(_peak_sos(fc, gain_db, Q, sr))
        elif btype == 'lowshelf':  sections.append(_lowshelf_sos(fc, gain_db, Q, sr))
        elif btype == 'highshelf': sections.append(_highshelf_sos(fc, gain_db, Q, sr))
    return np.vstack(sections) if sections else None


# ── Mic profiles ───────────────────────────────────────────────────────────────
# Each entry: list of (type, freq_hz, gain_db, Q)
# Derived from published frequency response curves and well-known tonal characters.

MIC_PROFILES = {

    # ── Dynamic handheld ──────────────────────────────────────────────────────

    'SM58': [
        ('lowshelf',  150,  +1.5, 0.7),   # proximity warmth
        ('peak',     400,   -1.0, 1.0),   # slight lower-mid recess
        ('peak',    6500,  +3.0, 1.5),   # classic presence peak
        ('highshelf', 14000, -5.0, 0.7),  # natural HF rolloff
    ],

    'Beta58A': [
        ('peak',     9000,  +5.0, 2.0),   # bright, forward presence
        ('peak',     400,   -1.5, 1.0),   # mid scoop
        ('highshelf', 15000, -3.0, 0.7),
    ],

    'e835': [
        ('peak',    8000,  +2.0, 1.5),   # gentle presence
        ('highshelf', 15000, -2.0, 0.7),
    ],

    'e945': [
        ('peak',    8500,  +2.5, 1.5),   # similar to e835, slightly brighter
        ('lowshelf',  100,  -1.0, 0.7),  # tighter low end (supercardioid)
        ('highshelf', 15000, -2.0, 0.7),
    ],

    'AKG_D5': [
        ('peak',     400,   -2.0, 0.8),  # mid scoop
        ('peak',    8000,  +4.0, 2.0),   # prominent presence
        ('highshelf', 14000, -3.0, 0.7),
    ],

    'EV_ND767a': [
        ('lowshelf',  200,  +2.0, 0.7),  # warm low end
        ('peak',    5000,  +3.0, 1.5),   # presence
        ('highshelf', 13000, -4.0, 0.7),
    ],

    'AT_AE6100': [
        ('peak',    10000, +4.0, 1.5),   # extended, detailed top
        ('highshelf', 16000, -2.0, 0.7),
    ],

    'Telefunken_M80': [
        ('lowshelf',  150,  +2.0, 0.7),  # vintage warmth
        ('peak',    6000,  +2.0, 1.0),   # smooth presence
        ('highshelf', 14000, -4.0, 0.7),
    ],

    'Heil_PR35': [
        ('lowshelf',  100,  +3.0, 0.7),  # extended low end
        ('peak',    4000,  +3.0, 1.5),   # forward presence
        ('highshelf', 14000, -3.0, 0.7),
    ],

    'Beyerdynamic_TGV70d': [
        ('lowshelf',  180,  +1.5, 0.7),
        ('peak',    5000,  +2.0, 2.0),
        ('highshelf', 13000, -4.0, 0.7),
    ],

    # ── Condenser handheld ────────────────────────────────────────────────────

    'Neumann_KMS105': [
        ('peak',    12000, +1.0, 1.0),   # condenser air, near-flat
        ('highshelf', 18000, -2.0, 0.7),
    ],

    'Neumann_KMS104': [
        ('peak',    10000, +1.5, 1.0),
        ('highshelf', 17000, -2.0, 0.7),
    ],

    'Shure_KSM9': [
        ('peak',    8000,  +1.0, 1.0),   # very flat, slight presence
    ],

    'AKG_C535': [
        ('peak',    10000, +3.0, 2.0),   # condenser brightness
        ('lowshelf',  120,  -1.0, 0.7),  # tighter low end
        ('highshelf', 17000, -2.0, 0.7),
    ],

    'Sennheiser_e965': [
        ('peak',    12000, +2.0, 1.5),   # air
        ('highshelf', 18000, -1.5, 0.7),
    ],

    'AT_AE5400': [
        ('peak',    9000,  +2.0, 2.0),
        ('highshelf', 17000, -2.0, 0.7),
    ],

    'DPA_2028': [
        # Very flat reference-grade response
        ('peak',    15000, +1.0, 1.0),
    ],

    'Earthworks_SR40V': [
        # Ruler-flat, measurement quality — essentially no coloration
    ],

    'Blue_enCORE300': [
        ('peak',    10000, +3.0, 2.0),
        ('lowshelf',   80,  -2.0, 0.7),  # LF rolloff
        ('highshelf', 16000, -2.0, 0.7),
    ],

    'Audix_VX5': [
        ('lowshelf',  150,  +1.0, 0.7),
        ('peak',    8000,  +3.0, 2.0),
        ('highshelf', 15000, -3.0, 0.7),
    ],
}

MIC_NAMES = list(MIC_PROFILES.keys())


def apply_random_mic_response(audio_np, sr, mic_name=None):
    """
    Apply a random mic frequency response to audio_np (float32 1-D array).
    Returns float32 array of same length.
    If mic_name is None, picks randomly.
    """
    name   = mic_name or random.choice(MIC_NAMES)
    bands  = MIC_PROFILES[name]
    if not bands:
        return audio_np   # Earthworks SR40V — ruler flat

    sos = _make_sos(bands, sr)
    if sos is None:
        return audio_np

    out = sosfilt(sos, audio_np.astype(np.float64)).astype(np.float32)
    # Normalise to prevent level changes biasing the loss
    rms_in  = np.sqrt(np.mean(audio_np**2)) + 1e-8
    rms_out = np.sqrt(np.mean(out**2))       + 1e-8
    return out * (rms_in / rms_out)
