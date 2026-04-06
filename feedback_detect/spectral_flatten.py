"""
feedback_detect/spectral_flatten.py — Notch-pair bridge EQ correction.

When two deep notches sit within a critical band of each other (~1.25x,
≈4 semitones), the unnotched region between them becomes perceptually
elevated by contrast — the spectral gap makes the bridge stand out.

Example: notches at 775 Hz and 855 Hz (-48 dB each) leave ~814 Hz
sounding elevated even though its absolute level has not changed.

Design
------
For each qualifying pair of active deep notches, place a gentle peaking
cut at their geometric mean.  Cut depth scales with notch depth and is
capped at MAX_CUT_DB.  Cuts attack smoothly when a pair is active and
release when either notch lifts.

This is driven entirely by the notch bank state — it never fires during
clean speech regardless of voice formant shape, and produces no false
positives from spectral analysis.

Only cuts — never boosts.  Cuts cannot cause feedback.
"""

import math
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


class SpectralFlattener:

    # A pair qualifies when the two notches are within ~4 semitones (1.25×).
    # Wider pairs are rare as isolated bridge cases — in a dense cluster,
    # the bridge is usually already covered by another notch.
    MAX_PAIR_RATIO    = 1.25    # ≈ 4 semitones
    MIN_NOTCH_DB      = -20.0   # both notches must be at least this deep
    BRIDGE_CUT_FACTOR = 0.08    # cut = min(notch_depths_db) × factor
    MAX_CUT_DB        = -4.0    # absolute ceiling on any bridge cut
    CUT_Q             = 2.5     # wide, musical cut
    CUT_SMOOTHING     = 0.05    # EMA frames toward target (attack)
    RELEASE_SMOOTHING = 0.02    # slower release than attack
    IDLE_FRAMES       = 200     # frames at ~0 dB before removing filter

    def __init__(self, bin_freqs: np.ndarray, sr: int = 48000):
        self.bin_freqs = bin_freqs
        self.sr        = sr
        # {(f_low, f_high): [PeakingEQ, current_cut_db, idle_counter]}
        self._bridges: dict[tuple[float, float], list] = {}
        self._peak_cuts: dict[tuple[float, float], float] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, stft_mag: np.ndarray, rms: float,
               active_notches: 'list[tuple[float, float, float]]'):
        """
        stft_mag       : unused (kept for API compatibility)
        rms            : unused
        active_notches : notch_bank.active_notches → [(freq, depth_db, q), ...]
        """
        deep = [(f, d, q) for f, d, q in active_notches if d <= self.MIN_NOTCH_DB]

        # Build target cuts for each qualifying pair
        targets: dict[tuple[float, float], tuple[float, float]] = {}
        for i in range(len(deep)):
            f1, d1, q1 = deep[i]
            for j in range(i + 1, len(deep)):
                f2, d2, q2 = deep[j]
                f_lo, f_hi = (f1, f2) if f1 < f2 else (f2, f1)
                if f_hi / f_lo > self.MAX_PAIR_RATIO:
                    continue
                bridge = math.sqrt(f_lo * f_hi)
                # Skip if bridge is already within the suppression range
                # of another active notch (it's handled, no bridge effect)
                if any(abs(bridge - f) <= f / (2.0 * q)
                       for f, _, q in active_notches):
                    continue
                cut = max(min(d1, d2) * self.BRIDGE_CUT_FACTOR, self.MAX_CUT_DB)
                targets[(f_lo, f_hi)] = (bridge, cut)

        # Update existing bridge filters
        for key in list(self._bridges):
            filt, cur_cut, idle = self._bridges[key]
            if key in targets:
                _, target_cut = targets[key]
                new_cut = cur_cut + self.CUT_SMOOTHING * (target_cut - cur_cut)
                self._bridges[key][1] = new_cut
                self._bridges[key][2] = 0
                filt.set_gain(float(new_cut))
            else:
                new_cut = cur_cut + self.RELEASE_SMOOTHING * (0.0 - cur_cut)
                if abs(new_cut) < 0.05:
                    new_cut = 0.0
                self._bridges[key][1] = new_cut
                filt.set_gain(float(new_cut))
                if abs(new_cut) < 0.05:
                    self._bridges[key][2] = idle + 1
                    if self._bridges[key][2] >= self.IDLE_FRAMES:
                        del self._bridges[key]
                        continue
            self._peak_cuts[key] = min(
                self._peak_cuts.get(key, 0.0), self._bridges[key][1])

        # Add new bridge filters
        for key, (bridge_freq, _) in targets.items():
            if key not in self._bridges:
                filt = PeakingEQ(bridge_freq, sr=self.sr,
                                 q=self.CUT_Q, gain_db=0.0)
                self._bridges[key] = [filt, 0.0, 0]

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Apply all active bridge cuts in series."""
        y = audio_block.astype(np.float32)
        for filt, cut, _ in self._bridges.values():
            if abs(cut) > 0.05:
                y = filt.process(y)
        return y

    def summary(self) -> str:
        active = [(k, v[1]) for k, v in self._bridges.items() if abs(v[1]) > 0.1]
        lines = []
        if not active:
            lines.append('SpectralFlattener: no active bridge cuts')
        else:
            lines.append('SpectralFlattener bridge cuts:')
            for (f1, f2), cut in sorted(active, key=lambda x: x[1]):
                bridge = math.sqrt(f1 * f2)
                lines.append(f'  {bridge:.0f} Hz '
                              f'(between {f1:.0f}/{f2:.0f} Hz): {cut:.1f} dB')
        if self._peak_cuts:
            peak = [(k, v) for k, v in self._peak_cuts.items() if v < -0.1]
            if peak:
                lines.append('  peak bridge cuts during run:')
                for (f1, f2), cut in sorted(peak, key=lambda x: x[1]):
                    bridge = math.sqrt(f1 * f2)
                    lines.append(f'    {bridge:.0f} Hz '
                                 f'({f1:.0f}/{f2:.0f} Hz): {cut:.2f} dB')
        return '\n'.join(lines)
