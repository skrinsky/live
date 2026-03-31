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

    Two-mode attack, stepped release, harmonic pre-emption:

    COLD attack (new frequency, no history):
        Slam to max_depth_db instantly. Cut hard, ask questions later.

    WARM re-attack (frequency already tracked, currently releasing):
        Step back WARM_REATTACK_DB from current depth. Already has context,
        a moderate additional cut is enough.

    Release (stepped):
        Hold HOLD_FRAMES_PER_STEP frames, then give back RELEASE_STEP_DB.
        Each step probes stability before releasing further. Settles at the
        shallowest depth that keeps the loop stable.

    Harmonic pre-emption:
        When update() is called with bin_freqs + prob_np, for each confirmed
        ringing fundamental F it scans 2F, 3F, 4F, 5F. If any harmonic bin
        shows sub-threshold activity (prob > HARMONIC_PROB_THRESH), a light
        preemptive notch is placed at HARMONIC_DEPTH_DB. This mirrors the
        live-engineer instinct: if 200Hz is ringing and 400Hz is stirring,
        get ahead of it now rather than wait for a full ring.
        Harmonic notches behave identically to primary ones after placement —
        if they later cross the primary threshold they simply get deepened.

    Workflow each audio frame:
      1. notch_bank.update(detected_freqs, bin_freqs, prob_np)
      2. out = notch_bank.process(block)
    """

    MAX_NOTCHES           = 8
    FREQ_TOL_HZ           = 75      # Hz — bins within this are the same resonance
    WARM_REATTACK_DB      = 12.0    # re-trigger while releasing: step back this far
    RELEASE_STEP_DB       = 6.0     # give back this many dB per release step
    HOLD_FRAMES_PER_STEP  = 50      # frames to hold before trying next release step (~0.5s)
    EXPIRE_THRESH_DB      = -0.5    # remove notch when it releases past this
    HARMONIC_PROB_THRESH  = 0.15    # sub-threshold prob to trigger harmonic pre-emption
    HARMONIC_DEPTH_DB     = -12.0   # initial depth for harmonic notches (lighter than primary)
    HARMONIC_MULTIPLES    = (2, 3, 4, 5)

    def __init__(self, sr=48000, q=30.0, depth_db=-48.0):
        self.sr            = sr
        self.q             = q
        self.max_depth_db  = depth_db
        # {freq_hz: [BiquadNotch, current_depth_db, hold_counter]}
        self._notches: dict[float, list] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detected_freqs: list[float],
               bin_freqs: 'np.ndarray | None' = None,
               prob_np:   'np.ndarray | None' = None):
        """
        Drive cold/warm attack, stepped release, and harmonic pre-emption.
        Call once per audio frame, before process().

        detected_freqs : confirmed ringing frequencies (prob > primary threshold)
        bin_freqs      : Hz value of each STFT bin  — required for harmonic scan
        prob_np        : per-bin probability array   — required for harmonic scan
        """
        triggered: set[float] = set()

        # ── Primary detections ────────────────────────────────────────────
        for freq in detected_freqs:
            existing = self._find_close(freq)
            if existing is not None:
                # Warm re-attack: already tracked, step back moderately
                triggered.add(existing)
                new_depth = max(self._notches[existing][1] - self.WARM_REATTACK_DB,
                                self.max_depth_db)
                self._notches[existing][1] = new_depth
                self._notches[existing][2] = self.HOLD_FRAMES_PER_STEP
                self._notches[existing][0].set_depth(new_depth)
            else:
                # Cold attack: new frequency — slam to max depth immediately
                key = self._add(freq, self.max_depth_db)
                triggered.add(key)

        # ── Harmonic pre-emption ─────────────────────────────────────────
        # For each confirmed fundamental, check 2x–5x harmonic bins.
        # If sub-threshold activity is present, place a light preemptive notch.
        if bin_freqs is not None and prob_np is not None:
            for fund_freq in detected_freqs:
                for n in self.HARMONIC_MULTIPLES:
                    h_freq = fund_freq * n
                    if h_freq > bin_freqs[-1]:
                        break                           # above Nyquist
                    # Find closest bin to this harmonic
                    idx = int(np.argmin(np.abs(bin_freqs - h_freq)))
                    if prob_np[idx] < self.HARMONIC_PROB_THRESH:
                        continue                        # no activity here
                    existing = self._find_close(h_freq)
                    if existing is not None:
                        triggered.add(existing)         # already tracked, hold it
                        self._notches[existing][2] = self.HOLD_FRAMES_PER_STEP
                    else:
                        # Light preemptive notch — shallower than primary cold attack
                        key = self._add(h_freq, self.HARMONIC_DEPTH_DB)
                        triggered.add(key)

        # ── Stepped release for untriggered notches ───────────────────────
        for freq in list(self._notches):
            if freq not in triggered:
                hold = self._notches[freq][2] - 1
                if hold > 0:
                    self._notches[freq][2] = hold
                else:
                    new_depth = self._notches[freq][1] + self.RELEASE_STEP_DB
                    if new_depth >= self.EXPIRE_THRESH_DB:
                        del self._notches[freq]
                    else:
                        self._notches[freq][1] = new_depth
                        self._notches[freq][2] = self.HOLD_FRAMES_PER_STEP
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

    def _add(self, freq: float, depth_db: float) -> float:
        """Add a new notch at depth_db. Returns its key."""
        if len(self._notches) >= self.MAX_NOTCHES:
            # Evict the shallowest notch (most released, least active)
            shallowest = max(self._notches, key=lambda f: self._notches[f][1])
            del self._notches[shallowest]
        notch = BiquadNotch(freq, sr=self.sr, q=self.q, depth_db=depth_db)
        self._notches[freq] = [notch, depth_db, self.HOLD_FRAMES_PER_STEP]
        return freq
