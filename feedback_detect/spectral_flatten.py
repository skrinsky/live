"""
feedback_detect/spectral_flatten.py — Chronic-ring background EQ.

Applies gentle, wide peaking cuts at the frequencies the FeedbackPredictor
has identified as persistent ringers.  Runs in series after the surgical
NotchBank.

The NotchBank handles acute ring onsets: narrow, deep, reactive.
ChronicRingEQ handles the background: frequencies that ring often enough
to fill a notch slot most of the time get a persistent wide cut that
reduces the base ring level.  The acute breaks then require less surgical
depth, and the notch bank evicts fewer slots.

Source of truth
---------------
predictor.risk_profile — risk[freq] = np.ndarray(N_STATES,) of accumulated
ring counts per voice state.  The top-N frequencies by max risk across all
states are the chronic ringers.  Cut depth scales linearly from MIN_CUT_DB
at MIN_RISK up to MAX_CUT_DB at RISK_MAX.

Only cuts — never boosts.  Cuts cannot cause feedback.
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
                 q: float = 2.0, gain_db: float = 0.0):
        self.sr       = sr
        self.freq_hz  = freq_hz
        self.q        = q
        self.gain_db  = gain_db
        self._set_coeffs()
        self.zi = np.zeros(2)

    def _set_coeffs(self):
        A     = 10.0 ** (self.gain_db / 40.0)
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


class ChronicRingEQ:

    TOP_N               = 8       # top N risk frequencies to cut
    MIN_RISK            = 3.0     # minimum risk score to qualify (> predictor RISK_THRESHOLD=2.0)
    MIN_CUT_DB          = -2.0    # cut at MIN_RISK
    MAX_CUT_DB          = -6.0    # cut at RISK_MAX — "a little wide EQ cut"
    RISK_MAX            = 20.0    # matches FeedbackPredictor.RISK_MAX
    CUT_Q               = 2.0     # wide — not surgical
    # Skip chronic cut if the notch bank already has a deep notch whose
    # -3dB half-bandwidth (nf / 2Q) actually covers the ring frequency.
    # Uses physical coverage, not a fixed ratio, to avoid skipping overflow
    # frequencies that sit just outside a notch's suppression range.
    NOTCH_SKIP_DB       = -20.0   # if active notch is deeper than this, check coverage
    UPDATE_INTERVAL     = 100     # frames between risk profile re-reads (~1 s at 100 fps)
    CUT_SMOOTHING       = 0.02    # EMA per frame toward target (~50-frame / 500 ms time constant)

    def __init__(self, sr: int = 48000):
        self.sr = sr
        # {freq_hz: PeakingEQ}
        self._filters:      dict[float, PeakingEQ] = {}
        self._current_cuts: dict[float, float]     = {}
        self._targets:      dict[float, float]     = {}
        self._peak_cuts:    dict[float, float]     = {}
        self._frame_count    = 0
        self._active_notches: 'list[tuple[float, float, float]]' = []

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, risk_profile: 'dict[float, np.ndarray]',
               active_notches: 'list[tuple[float, float, float]]'):
        """
        risk_profile   : predictor.risk_profile → {freq: np.ndarray(N_STATES,)}
        active_notches : notch_bank.active_notches → [(freq, depth_db, q), ...]
        """
        self._active_notches = active_notches
        self._frame_count += 1
        if self._frame_count >= self.UPDATE_INTERVAL:
            self._frame_count = 0
            self._rebuild_targets(risk_profile)
        self._smooth()

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Apply all active chronic cuts in series."""
        y = audio_block.astype(np.float32)
        for freq, filt in self._filters.items():
            if abs(self._current_cuts.get(freq, 0.0)) > 0.05:
                y = filt.process(y)
        return y

    def summary(self) -> str:
        active = [(f, self._current_cuts[f])
                  for f in self._filters if abs(self._current_cuts.get(f, 0.0)) > 0.1]
        lines = []
        if not active:
            lines.append('ChronicRingEQ: no active cuts')
        else:
            lines.append('ChronicRingEQ cuts:')
            for f, cut in sorted(active, key=lambda x: x[1]):
                tgt = self._targets.get(f, 0.0)
                lines.append(f'  {f:.0f} Hz: {cut:.1f} dB  (target {tgt:.1f} dB)')
        if self._peak_cuts:
            peak = [(f, v) for f, v in self._peak_cuts.items() if v < -0.1]
            if peak:
                lines.append('  peak cuts during run:')
                for f, cut in sorted(peak, key=lambda x: x[1]):
                    lines.append(f'    {f:.0f} Hz: {cut:.2f} dB')
        return '\n'.join(lines)

    # ── private ───────────────────────────────────────────────────────────────

    def _risk_to_cut(self, score: float) -> float:
        """Linear interpolation: MIN_RISK → MIN_CUT_DB, RISK_MAX → MAX_CUT_DB."""
        t = min(1.0, max(0.0,
                (score - self.MIN_RISK) / (self.RISK_MAX - self.MIN_RISK)))
        return self.MIN_CUT_DB + t * (self.MAX_CUT_DB - self.MIN_CUT_DB)

    def _rebuild_targets(self, risk_profile: 'dict[float, np.ndarray]'):
        """Re-read risk table and update cut targets."""
        scored = [(freq, float(risk.max()))
                  for freq, risk in risk_profile.items()
                  if float(risk.max()) >= self.MIN_RISK]
        scored.sort(key=lambda x: -x[1])
        # Filter out frequencies already covered by a deep notch bank slot
        deep_notches = [(f, q) for f, d, q in self._active_notches
                        if d <= self.NOTCH_SKIP_DB]
        top = {}
        for freq, score in scored:
            if len(top) >= self.TOP_N:
                break
            # Skip if any deep notch's half-BW physically covers this frequency
            covered = any(abs(freq - nf) <= nf / (2.0 * q)
                          for nf, q in deep_notches)
            if not covered:
                top[freq] = self._risk_to_cut(score)

        # Set releasing targets for frequencies no longer in top-N
        for freq in list(self._targets):
            if freq not in top:
                self._targets[freq] = 0.0

        # Update or add targets
        for freq, cut in top.items():
            self._targets[freq] = cut
            if freq not in self._filters:
                self._filters[freq]      = PeakingEQ(freq, sr=self.sr,
                                                     q=self.CUT_Q, gain_db=0.0)
                self._current_cuts[freq] = 0.0

    def _smooth(self):
        """Step each filter's gain one EMA frame toward its target."""
        to_remove = []
        for freq in list(self._filters):
            target  = self._targets.get(freq, 0.0)
            current = self._current_cuts.get(freq, 0.0)
            new     = current + self.CUT_SMOOTHING * (target - current)
            self._current_cuts[freq] = new
            self._filters[freq].set_gain(float(new))
            self._peak_cuts[freq] = min(self._peak_cuts.get(freq, 0.0), new)
            # Remove filter once it has fully released to 0
            if target == 0.0 and abs(new) < 0.05:
                to_remove.append(freq)
        for freq in to_remove:
            del self._filters[freq]
            del self._current_cuts[freq]
            self._targets.pop(freq, None)
