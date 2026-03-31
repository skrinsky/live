"""
feedback_detect/notch.py — Parametric biquad notch filter bank.

Pure signal processing — no learning. The neural detector decides WHICH
frequencies to notch; this module does the actual notching with precision
that is independent of the model's internal frequency resolution.

Biquad coefficients from the Audio EQ Cookbook (R. Bristow-Johnson).
Notch depth: blend of notch-filtered and dry signal so depth is finite
and controllable rather than infinite (which causes audible artifacts).
"""

import numpy as np
from scipy.signal import lfilter


class BiquadNotch:
    """
    Single stateful parametric biquad notch filter.

    Coefficients are fixed at construction. To change frequency, create
    a new instance (or call _set_coeffs). State (zi) persists across blocks
    for seamless streaming.
    """

    def __init__(self, freq_hz, sr=48000, q=30.0, depth_db=-24.0):
        """
        freq_hz   : centre frequency in Hz
        sr        : sample rate
        q         : Q factor — higher = narrower notch
                    Q=30 → bandwidth ≈ freq/30 (e.g. ±17 Hz at 1 kHz)
        depth_db  : notch depth in dB (negative). -24 dB is deep but not
                    a perfect null, preventing phase-artefact 'clunk' sounds.
        """
        self.sr       = sr
        self.freq_hz  = freq_hz
        self.q        = q
        self.depth_db = depth_db
        self._set_coeffs(freq_hz)
        self.zi = np.zeros(2)

    def _set_coeffs(self, freq_hz):
        """Biquad notch coefficients (Audio EQ Cookbook)."""
        w0    = 2.0 * np.pi * freq_hz / self.sr
        alpha = np.sin(w0) / (2.0 * self.q)
        cosw  = np.cos(w0)

        b0, b1, b2 = 1.0,        -2.0 * cosw,  1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha

        self.b = np.array([b0 / a0, b1 / a0, b2 / a0])
        self.a = np.array([1.0,     a1 / a0, a2 / a0])

        # Blend factor: 0 = full notch, 1 = dry
        # depth_db < 0, so dry_mix < 1 → partial suppression
        self.dry_mix = 10.0 ** (self.depth_db / 20.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Process audio block x (float32 numpy array, any length).
        State carries over between calls — call reset() to clear.
        Returns notch-filtered output.
        """
        notched, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        # Finite depth: blend filtered and dry
        return (1.0 - self.dry_mix) * notched + self.dry_mix * x

    def reset(self):
        self.zi[:] = 0.0


class NotchBank:
    """
    Bank of up to MAX_NOTCHES active biquad notch filters.

    Workflow:
      1. Each audio frame, call update(detected_freqs) with the list of
         feedback frequencies the detector fired on.
      2. Call process(audio_block) to apply all active notches.

    Notches persist for HOLD_FRAMES after the last detection at that
    frequency, then expire. New frequencies add new notches; if the bank
    is full, the oldest notch is evicted.
    """

    MAX_NOTCHES  = 8       # max simultaneous notch filters
    HOLD_FRAMES  = 300     # frames to hold after last detection (~3 s at 100 fps)
    FREQ_TOL_HZ  = 75      # Hz — frequencies within this range are the same notch

    def __init__(self, sr=48000, q=30.0, depth_db=-24.0):
        self.sr        = sr
        self.q         = q
        self.depth_db  = depth_db
        # {freq_hz: [BiquadNotch, hold_counter]}
        self._notches: dict[float, list] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detected_freqs: list[float]):
        """
        Update the notch bank from a list of detected feedback frequencies
        (in Hz). Call once per audio frame, before process().
        """
        for freq in detected_freqs:
            existing = self._find_close(freq)
            if existing is not None:
                self._notches[existing][1] = self.HOLD_FRAMES   # reset hold
            else:
                self._add(freq)

        # Decrement hold counters; remove expired notches
        for freq in list(self._notches):
            self._notches[freq][1] -= 1
            if self._notches[freq][1] <= 0:
                del self._notches[freq]

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Apply all active notches to audio_block (in-series)."""
        y = audio_block.astype(np.float32)
        for notch, _ in self._notches.values():
            y = notch.process(y)
        return y

    @property
    def active_freqs(self) -> list[float]:
        return list(self._notches.keys())

    def reset(self):
        self._notches.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _find_close(self, freq: float) -> float | None:
        """Return the existing notch frequency closest to freq, if within FREQ_TOL_HZ."""
        for existing in self._notches:
            if abs(existing - freq) < self.FREQ_TOL_HZ:
                return existing
        return None

    def _add(self, freq: float):
        """Add a new notch, evicting the oldest if the bank is full."""
        if len(self._notches) >= self.MAX_NOTCHES:
            # Evict the notch with the lowest hold counter (most stale)
            oldest = min(self._notches, key=lambda f: self._notches[f][1])
            del self._notches[oldest]
        notch = BiquadNotch(freq, sr=self.sr, q=self.q, depth_db=self.depth_db)
        self._notches[freq] = [notch, self.HOLD_FRAMES]
