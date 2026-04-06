"""
feedback_detect/notch.py — Parametric biquad notch filter bank with attack/release.

Pure signal processing — no learning. The neural detector decides WHICH
frequencies to notch; this module does the actual notching.

Both depth and Q are dynamic:

  Depth (how deep to cut):
    Detection fires → slam to max_depth_db immediately
    No detection    → step back 6 dB every ~1 s until 0 dB (pass-through)
    Re-triggered    → slam back to max; re-probe from full depth
    Convergence     → minimum stable cut found dynamically, not hardcoded

  Q (how wide to cut):
    Cold attack     → start at MAX_Q (30) — surgical, minimal collateral
    Re-triggered    → widen by Q_WIDEN_FACTOR each time (30→22→17→13→10→8→5)
    Stabilises      → minimum Q that stops re-triggering IS the mode width
    New cold        → back to MAX_Q

  This mirrors what a live engineer does: cut hard and narrow first, widen
  if the ring escapes the notch, then back off depth until it just holds.

Biquad coefficients from the Audio EQ Cookbook (R. Bristow-Johnson).
Depth is implemented as a wet/dry blend so it is always finite (no perfect
null), preventing the phase-artefact 'clunk' of an infinite notch.
"""

import numpy as np
from scipy.signal import lfilter


class BiquadNotch:
    """
    Single stateful parametric biquad notch filter with variable depth and Q.

    Coefficients are recomputed on set_q(); filter state (zi) is preserved
    across the recomputation so there is no audible click.
    Depth (dry_mix) is updated via set_depth() — no coefficient recomputation.
    """

    def __init__(self, freq_hz, sr=48000, q=30.0, depth_db=0.0):
        self.sr       = sr
        self.freq_hz  = freq_hz
        self.q        = q
        self.depth_db = depth_db
        self._set_coeffs()
        self.zi = np.zeros(2)

    def _set_coeffs(self):
        """Biquad notch coefficients (Audio EQ Cookbook)."""
        w0    = 2.0 * np.pi * self.freq_hz / self.sr
        alpha = np.sin(w0) / (2.0 * self.q)
        cosw  = np.cos(w0)

        b0, b1, b2 = 1.0,         -2.0 * cosw, 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha

        self.b = np.array([b0 / a0, b1 / a0, b2 / a0])
        self.a = np.array([1.0,     a1 / a0, a2 / a0])
        self._update_dry_mix()

    def _update_dry_mix(self):
        # depth_db=0 → dry_mix=1.0 (pass-through), depth_db=-48 → very deep cut
        self.dry_mix = 10.0 ** (self.depth_db / 20.0)

    def set_depth(self, depth_db: float):
        """Update notch depth without recomputing filter coefficients."""
        self.depth_db = depth_db
        self._update_dry_mix()

    def set_q(self, q: float):
        """Widen or narrow the notch. Recomputes coefficients; zi preserved — no click."""
        self.q = q
        self._set_coeffs()

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
    Bank of up to MAX_NOTCHES biquad notch filters with adaptive depth and Q.

    Per-notch state: [BiquadNotch, current_depth_db, hold_counter, current_q]

    Depth behaviour:
      - Any detection (cold or re-trigger) → slam to max_depth_db immediately
      - Release: hold HOLD_FRAMES_PER_STEP (~1s), give back RELEASE_STEP_DB (6dB)
      - At 0dB: sit silently for IDLE_FRAMES_TO_EXPIRE (~60s), then remove
      - Re-trigger at any depth → slam back to max, re-probe from full depth

    Q behaviour (adaptive):
      - Cold attack: Q = MAX_Q (30) — start surgical
      - Re-trigger:  Q = max(MIN_Q, current_q × Q_WIDEN_FACTOR) — widen each time
      - Stabilises at the minimum Q that stops the re-trigger = actual mode width
      - Cold add always starts at MAX_Q

    Convergence:
      Over multiple speech events, depth and Q both settle at their minimums
      for this room mode — found dynamically, not hardcoded.

    Harmonic pre-emption:
      Scans 2F–5F harmonics of each confirmed fundamental. Sub-threshold
      activity → light notch placed preemptively at HARMONIC_DEPTH_DB.

    Workflow each audio frame:
      1. notch_bank.update(detected_freqs, bin_freqs, prob_np)
      2. out = notch_bank.process(block)
    """

    MAX_NOTCHES           = 24
    FREQ_TOL_RATIO        = 0.18    # ±18% (~3 semitones) — log-scale mode matching
    RELEASE_STEP_DB       = 3.0     # give back this many dB per probe step
    HOLD_FRAMES_PER_STEP  = 50      # frames between probe steps (~500ms at 100fps)
    LOCKED_HOLD_FRAMES    = 500     # frames between probes once locked (~5s at 100fps)
    LOCK_THRESHOLD        = 2       # re-triggers to declare minimum found → lock in
    IDLE_FRAMES_TO_EXPIRE = 6000    # frames at 0dB with no detection before removal (~60s)
    HARMONIC_PROB_THRESH  = 0.50    # prob to trigger harmonic pre-emption — must be clearly ringing
    HARMONIC_DEPTH_DB     = -12.0   # initial depth for harmonic notches
    PREEMPTIVE_DEPTH_DB   = -36.0   # initial depth for risk-based pre-emptive notches
    HARMONIC_MULTIPLES    = (2, 3, 4, 5)
    # Q is frequency-proportional so bandwidth stays perceptually consistent.
    # INITIAL_BW_HZ: surgical starting cut (~50 Hz at 1 kHz → Q=20)
    # MAX_BW_HZ:     widest allowed cut  (~200 Hz at 1 kHz → Q=5)
    # Floors prevent Q from going below 2 (useless notch) or above 200 (inaudible BW).
    INITIAL_BW_HZ         = 100.0
    MIN_Q_ABS             = 5.0     # absolute floor — matches old MIN_Q, ensures rings are caught
    Q_WIDEN_FACTOR        = 0.65    # multiply Q by this when re-triggered at full depth

    def __init__(self, sr=48000, q=30.0, depth_db=-48.0):
        self.sr            = sr
        self.q             = q          # kept for external callers; internal uses MAX_Q
        self.max_depth_db  = depth_db
        # {freq_hz: [BiquadNotch, current_depth_db, hold_counter, current_q, retrigger_count]}
        self._notches: dict[float, list] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detected_freqs: list[float],
               bin_freqs:        'np.ndarray | None' = None,
               prob_np:          'np.ndarray | None' = None,
               preemptive_freqs: 'list[float] | None' = None):
        """
        Drive cold/warm attack, adaptive Q, stepped release, harmonic pre-emption.
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
                triggered.add(existing)
                self._retrigger(existing)
            else:
                key = self._add(freq, self.max_depth_db)
                triggered.add(key)

        # ── Harmonic pre-emption ──────────────────────────────────────────
        if bin_freqs is not None and prob_np is not None:
            for fund_freq in detected_freqs:
                for n in self.HARMONIC_MULTIPLES:
                    h_freq = fund_freq * n
                    if h_freq > bin_freqs[-1]:
                        break
                    idx = int(np.argmin(np.abs(bin_freqs - h_freq)))
                    if prob_np[idx] < self.HARMONIC_PROB_THRESH:
                        continue
                    existing = self._find_close(h_freq)
                    if existing is not None:
                        triggered.add(existing)
                        self._retrigger(existing)
                    else:
                        key = self._add(h_freq, self.HARMONIC_DEPTH_DB)
                        triggered.add(key)

        # ── Risk-based pre-emptive notches ───────────────────────────────
        if preemptive_freqs:
            for freq in preemptive_freqs:
                existing = self._find_close(freq)
                if existing is not None:
                    triggered.add(existing)
                    # Don't slam to max depth — just hold at current depth
                    self._notches[existing][2] = self.HOLD_FRAMES_PER_STEP
                else:
                    key = self._add(freq, self.PREEMPTIVE_DEPTH_DB)
                    triggered.add(key)

        # ── Stepped release / idle expiry for untriggered notches ─────────
        for freq in list(self._notches):
            if freq not in triggered:
                hold = self._notches[freq][2] - 1
                if hold > 0:
                    self._notches[freq][2] = hold
                    continue

                current_depth = self._notches[freq][1]

                if current_depth >= 0.0:
                    # Already at 0dB — idle countdown expired → remove
                    del self._notches[freq]
                else:
                    new_depth = current_depth + self.RELEASE_STEP_DB
                    if new_depth >= 0.0:
                        # Reached 0dB — sit silently, start idle countdown
                        self._notches[freq][1] = 0.0
                        self._notches[freq][2] = self.IDLE_FRAMES_TO_EXPIRE
                        self._notches[freq][0].set_depth(0.0)
                    else:
                        # Normal probe step — release depth only, Q stays where it is
                        self._notches[freq][1] = new_depth
                        self._notches[freq][2] = self.HOLD_FRAMES_PER_STEP
                        self._notches[freq][0].set_depth(new_depth)

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Apply all active notches in series."""
        y = audio_block.astype(np.float32)
        for notch, _, __, ___, ____ in self._notches.values():
            y = notch.process(y)
        return y

    @property
    def active_freqs(self) -> list[float]:
        return list(self._notches.keys())

    @property
    def active_notches(self) -> list[tuple[float, float, float]]:
        """(freq_hz, current_depth_db, current_q) for each active notch."""
        return [(f, v[1], v[3]) for f, v in self._notches.items()]

    def reset(self):
        self._notches.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _initial_q(self, freq_hz: float) -> float:
        """Starting Q — proportional to frequency, floored at 2, capped at 30.
        Below ~3 kHz: wider than old fixed Q=30. Above ~3 kHz: same as old Q=30."""
        return max(2.0, min(30.0, freq_hz / self.INITIAL_BW_HZ))

    def _min_q(self, freq_hz: float) -> float:
        """Widest allowed Q — same absolute floor as before (MIN_Q_ABS=5)."""
        return self.MIN_Q_ABS

    def _retrigger(self, freq: float):
        """
        Slam depth to max. Widen Q only if already at max depth — width is the problem.
        If re-triggered at a shallower depth, depth was the problem — Q stays narrow.
        """
        state = self._notches[freq]
        at_max_depth = state[1] <= self.max_depth_db + 0.1   # already fully slammed
        # Only widen Q when depth alone isn't enough
        if at_max_depth:
            new_q = max(self._min_q(freq), state[3] * self.Q_WIDEN_FACTOR)
            if new_q != state[3]:
                state[3] = new_q
                state[0].set_q(new_q)
        # Slam depth
        state[1] = self.max_depth_db
        state[0].set_depth(self.max_depth_db)
        # Lock in after LOCK_THRESHOLD re-triggers — minimum found, probe slowly
        state[4] += 1
        state[2] = (self.LOCKED_HOLD_FRAMES if state[4] >= self.LOCK_THRESHOLD
                    else self.HOLD_FRAMES_PER_STEP)

    def _find_close(self, freq: float) -> float | None:
        """Return the existing notch frequency closest to freq, but only if the
        notch's current -3dB half-bandwidth (existing / (2*Q)) actually covers
        freq.  This prevents a narrow notch at 775 Hz from absorbing an 855 Hz
        detection that falls outside its suppression range — without tightening
        FREQ_TOL_RATIO (which causes stacked-notch phase distortion)."""
        for existing, state in self._notches.items():
            if abs(existing - freq) / freq >= self.FREQ_TOL_RATIO:
                continue
            current_q = state[3]
            half_bw = existing / (2.0 * current_q)
            if abs(existing - freq) <= half_bw:
                return existing
        return None

    def _add(self, freq: float, depth_db: float) -> float:
        """Add a new notch at MAX_Q and depth_db. Returns its key."""
        if len(self._notches) >= self.MAX_NOTCHES:
            # Evict the shallowest notch (most released, least active)
            shallowest = max(self._notches, key=lambda f: self._notches[f][1])
            del self._notches[shallowest]
        q0 = self._initial_q(freq)
        notch = BiquadNotch(freq, sr=self.sr, q=q0, depth_db=depth_db)
        self._notches[freq] = [notch, depth_db, self.HOLD_FRAMES_PER_STEP, q0, 0]
        return freq
