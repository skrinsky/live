"""
feedback_detect/spectral_flatten.py — Adaptive spectral coloration correction.

After the NotchBank cuts at ring frequencies, neighboring bands can stand
out perceptually by contrast — the notch changes the spectral balance even
though it doesn't touch those frequencies directly.

This module corrects that by tracking the voice's spectral balance over
time and applying gentle wide-Q cuts to bands that become over-prominent
relative to their own history.

Design
------
24 log-spaced bands from 100 Hz to 12 kHz.
Per-band energy is NORMALISED by total power so the comparison is about
spectral SHAPE (balance), not absolute level. When a notch cuts energy
at 492 Hz, the total power drops, and any band that is now a larger-than-
usual fraction of total power is flagged.

Two guard conditions prevent false cuts:
  1. Bands within NOTCH_GUARD (25%) of an active notch are excluded — the
     notch itself is handled by NotchBank, not here.
  2. The long-term reference freezes entirely when any deep notch is active,
     preserving the pre-ring natural voice spectral shape as the reference
     across ALL bands — not just the bands adjacent to the notch.

Only cuts — never boosts. Cuts cannot cause feedback.
"""

import numpy as np
from scipy.signal import lfilter


class PeakingEQ:
    """
    Stateful biquad peaking/cutting EQ.
    gain_db < 0 → cut at freq_hz with bandwidth proportional to Q.
    Coefficients from the Audio EQ Cookbook (peakingEQ filter).
    """

    def __init__(self, freq_hz: float, sr: int = 48000,
                 q: float = 3.0, gain_db: float = 0.0):
        self.sr       = sr
        self.freq_hz  = freq_hz
        self.q        = q
        self.gain_db  = gain_db
        self._set_coeffs()
        self.zi = np.zeros(2)

    def _set_coeffs(self):
        A     = 10.0 ** (self.gain_db / 40.0)   # sqrt of linear gain
        w0    = 2.0 * np.pi * self.freq_hz / self.sr
        alpha = np.sin(w0) / (2.0 * self.q)
        cosw  = np.cos(w0)
        b0 =  1.0 + alpha * A
        b1 = -2.0 * cosw
        b2 =  1.0 - alpha * A
        a0 =  1.0 + alpha / A
        a1 = -2.0 * cosw
        a2 =  1.0 - alpha / A
        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1.0,   a1/a0, a2/a0])

    def set_gain(self, gain_db: float):
        if abs(gain_db - self.gain_db) > 0.05:
            self.gain_db = gain_db
            self._set_coeffs()

    def process(self, x: np.ndarray) -> np.ndarray:
        if abs(self.gain_db) < 0.1:
            return x
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


class SpectralFlattener:

    N_BANDS       = 24
    F_MIN         = 100.0
    F_MAX         = 12000.0

    # Time constants (frames at ~100 fps)
    SHORT_ALPHA   = 0.10    # ~10 frames = 100 ms
    LONG_ALPHA    = 0.001   # ~1000 frames = 10 s half-life

    # Cut trigger
    PROMINENCE_DB = 1.5     # normalised band must exceed reference by this much
    MAX_CUT_DB    = -6.0
    CUT_Q         = 3.0     # wide, musical cut
    CUT_SMOOTHING = 0.05    # slow attack (~20 frames) to avoid modulation

    # Guard zones
    VOICE_FLOOR   = 1e-5    # RMS below this = silence, freeze reference
    NOTCH_GUARD   = 0.25    # ±25% around active notch freq = don't cut there

    def __init__(self, bin_freqs: np.ndarray, sr: int = 48000):
        self.bin_freqs = bin_freqs
        self.sr        = sr

        self.band_freqs = np.logspace(
            np.log10(self.F_MIN), np.log10(self.F_MAX), self.N_BANDS)

        # Bin-to-band mapping: log-equal-width edges
        edges = np.logspace(
            np.log10(self.F_MIN * 2 ** (-1.0 / self.N_BANDS)),
            np.log10(self.F_MAX * 2 ** ( 1.0 / self.N_BANDS)),
            self.N_BANDS + 1)
        self._band_masks = [
            (bin_freqs >= edges[i]) & (bin_freqs < edges[i + 1])
            for i in range(self.N_BANDS)
        ]

        self._short_norm = np.ones(self.N_BANDS, dtype=np.float64) / self.N_BANDS
        self._long_norm  = np.ones(self.N_BANDS, dtype=np.float64) / self.N_BANDS
        self._cut_db     = np.zeros(self.N_BANDS, dtype=np.float64)

        self._filters = [
            PeakingEQ(f, sr=sr, q=self.CUT_Q, gain_db=0.0)
            for f in self.band_freqs
        ]

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, stft_mag: np.ndarray, rms: float,
               active_notches: 'list[tuple[float,float,float]]'):
        """
        Update envelope estimates and target cut depths.

        stft_mag       : (N_FREQ,) linear magnitude from current STFT frame
        rms            : RMS of current audio block (for voice-activity gate)
        active_notches : notch_bank.active_notches → [(freq, depth_db, q), ...]
        """
        # Per-band power, normalised to spectral SHAPE
        band_power = np.array([
            float((stft_mag[m] ** 2).sum()) + 1e-20
            for m in self._band_masks
        ])
        total = band_power.sum() + 1e-20
        band_norm = band_power / total

        # Guard zone: bands within NOTCH_GUARD of a deep active notch
        guard = np.zeros(self.N_BANDS, dtype=bool)
        for nf, nd, _ in active_notches:
            if nd < -10.0:
                for i, bf in enumerate(self.band_freqs):
                    if abs(bf - nf) / bf < self.NOTCH_GUARD:
                        guard[i] = True

        # Short-term: always update
        self._short_norm = (self.SHORT_ALPHA * band_norm +
                            (1.0 - self.SHORT_ALPHA) * self._short_norm)

        # Long-term: freeze entirely when any deep notch is active so the
        # reference preserves the pre-ring natural voice spectral shape.
        # Any notch at all shifts spectral balance; we want to catch that shift
        # across all bands, not just the bands adjacent to the notch.
        any_deep_notch = any(nd < -10.0 for _, nd, _ in active_notches)
        if rms > self.VOICE_FLOOR and not any_deep_notch:
            self._long_norm = (
                self.LONG_ALPHA * band_norm +
                (1.0 - self.LONG_ALPHA) * self._long_norm)

        # Prominence in dB (normalised shape comparison)
        with np.errstate(divide='ignore', invalid='ignore'):
            prom_db = 10.0 * np.log10(
                self._short_norm / (self._long_norm + 1e-20))
        prom_db = np.nan_to_num(prom_db, nan=0.0, posinf=0.0, neginf=0.0)

        # Target cut: zero in guard zones, proportional to excess elsewhere
        excess  = np.maximum(prom_db - self.PROMINENCE_DB, 0.0)
        target  = np.clip(-excess, self.MAX_CUT_DB, 0.0)
        target[guard] = 0.0

        # Smooth toward target
        self._cut_db += self.CUT_SMOOTHING * (target - self._cut_db)

        for i, filt in enumerate(self._filters):
            filt.set_gain(float(self._cut_db[i]))

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Apply current EQ to the audio block."""
        y = audio_block.astype(np.float32)
        for filt in self._filters:
            y = filt.process(y)
        return y

    def summary(self) -> str:
        active = [
            (self.band_freqs[i], self._cut_db[i])
            for i in range(self.N_BANDS) if self._cut_db[i] < -0.5
        ]
        if not active:
            return 'SpectralFlattener: no active cuts'
        lines = ['SpectralFlattener cuts:']
        for f, db in active:
            lines.append(f'  {f:.0f} Hz: {db:.1f} dB')
        return '\n'.join(lines)
