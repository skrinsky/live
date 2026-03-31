"""
feedback_detect/notch.py — Parametric biquad notch filter bank with attack/release.

Pure signal processing — no learning. The neural detector decides WHICH
frequencies to notch; this module does the actual notching.

Notch depth is dynamic, not fixed:
  - Detection fires → attack: depth slams toward max_depth in ~3 frames (30ms)
  - No detection    → release: depth creeps back toward 0 over ~2.5s
  - Re-triggered during release → re-attack → settles at minimum stable depth

This mirrors what a live engineer does: cut hard to break the loop, then
slowly raise the fader back until it just starts to ring, then back off —
finding the minimum cut that keeps the system stable.

Biquad coefficients from the Audio EQ Cookbook (R. Bristow-Johnson).
Depth is implemented as a wet/dry blend so it is always finite (no perfect
null), preventing the phase-artefact 'clunk' of an infinite notch.
"""

import numpy as np
from scipy.signal import lfilter


class BiquadNotch:
    """
    Single stateful parametric biquad notch filter with variable depth.

    Filter coefficients (freq, Q) are fixed at construction.
    Depth (dry_mix) is updated dynamically via set_depth() — no recomputation
    of coefficients needed, just changes the wet/dry blend ratio.
    """

    def __init__(self, freq_hz, sr=48000, q=30.0, depth_db=0.0):
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

        b0, b1, b2 = 1.0,         -2.0 * cosw, 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha

        self.b = np.array([b0 / a0, b1 / a0, b2 / a0])
        self.a = np.array([1.0,     a1 / a0, a2 / a0])
        self._update_dry_mix()

    def _update_dry_mix(self):
        # depth_db=0 → dry_mix=1.0 (pass-through), depth_db=-24 → dry_mix≈0.063 (deep cut)
        self.dry_mix = 10.0 ** (self.depth_db / 20.0)

    def set_depth(self, depth_db: float):
        """Update notch depth without recomputing filter coefficients."""
        self.depth_db = depth_db
        self._update_dry_mix()

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Process audio block x (float32 numpy array).
        Filter state (zi) persists across calls for seamless streaming.
        """
        notched, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return (1.0 - self.dry_mix) * notched + self.dry_mix * x

    def reset(self):
        self.zi[:] = 0.0


class NotchBank:
    """
    Bank of up to MAX_NOTCHES biquad notch filters with attack/release envelopes.

    Each notch has a dynamic depth_db that:
      - Attacks toward max_depth_db when its frequency is detected (fast, ~3 frames)
      - Releases back toward 0 when not detected (slow, ~2.5 s)
      - Is removed once it releases past EXPIRE_THRESH_DB

    This creates a per-frequency compressor that finds the minimum stable cut:
    if the loop is just barely unstable, the notch settles shallow; if heavily
    ringing, it stays deep.

    Workflow each audio frame:
      1. notch_bank.update(detected_freqs)   — drive attack/release
      2. out = notch_bank.process(block)     — apply all active notches
    """

    MAX_NOTCHES          = 8
    FREQ_TOL_HZ          = 75       # Hz — bins within this are the same resonance
    ATTACK_DB_PER_FRAME  = 8.0      # depth change per frame on detection  (~3 frames to -24)
    RELEASE_DB_PER_FRAME = 0.1      # depth change per frame on release    (~240 frames = 2.4s)
    EXPIRE_THRESH_DB     = -0.5     # remove notch when it releases past this

    def __init__(self, sr=48000, q=30.0, depth_db=-24.0):
        self.sr            = sr
        self.q             = q
        self.max_depth_db  = depth_db   # deepest allowed cut (typically -24 dB)
        # {freq_hz: [BiquadNotch, current_depth_db]}
        self._notches: dict[float, list] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detected_freqs: list[float]):
        """
        Drive attack/release from a list of detected feedback frequencies (Hz).
        Call once per audio frame, before process().
        """
        triggered: set[float] = set()

        # Attack: deepen notches for every detected frequency
        for freq in detected_freqs:
            existing = self._find_close(freq)
            if existing is not None:
                triggered.add(existing)
                new_depth = max(self._notches[existing][1] - self.ATTACK_DB_PER_FRAME,
                                self.max_depth_db)
                self._notches[existing][1] = new_depth
                self._notches[existing][0].set_depth(new_depth)
            else:
                key = self._add(freq)
                triggered.add(key)

        # Release: shallow-up notches that weren't triggered this frame
        for freq in list(self._notches):
            if freq not in triggered:
                new_depth = self._notches[freq][1] + self.RELEASE_DB_PER_FRAME
                if new_depth >= self.EXPIRE_THRESH_DB:
                    del self._notches[freq]    # fully released — remove
                else:
                    self._notches[freq][1] = new_depth
                    self._notches[freq][0].set_depth(new_depth)

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Apply all active notches in series."""
        y = audio_block.astype(np.float32)
        for notch, _ in self._notches.values():
            y = notch.process(y)
        return y

    @property
    def active_freqs(self) -> list[float]:
        return list(self._notches.keys())

    @property
    def active_notches(self) -> list[tuple[float, float]]:
        """(freq_hz, current_depth_db) for each active notch."""
        return [(f, v[1]) for f, v in self._notches.items()]

    def reset(self):
        self._notches.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _find_close(self, freq: float) -> float | None:
        """Return the existing notch frequency closest to freq, if within FREQ_TOL_HZ."""
        for existing in self._notches:
            if abs(existing - freq) < self.FREQ_TOL_HZ:
                return existing
        return None

    def _add(self, freq: float) -> float:
        """Create a new notch at depth=0 (attack will drive it deep). Returns its key."""
        if len(self._notches) >= self.MAX_NOTCHES:
            # Evict the shallowest notch (most released, least active)
            shallowest = max(self._notches, key=lambda f: self._notches[f][1])
            del self._notches[shallowest]
        notch = BiquadNotch(freq, sr=self.sr, q=self.q, depth_db=0.0)
        self._notches[freq] = [notch, 0.0]
        return freq
