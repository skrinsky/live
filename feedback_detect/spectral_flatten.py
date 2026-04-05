"""
feedback_detect/spectral_flatten.py — Adaptive spectral coloration correction.

After the NotchBank cuts at ring frequencies, neighboring bands stand out by
contrast — the notch creates a spectral gap, making the untouched bands on
either side sound perceptually prominent.

Example: notches at ~680 Hz and ~987 Hz leave the 800 Hz region elevated
by ~2.5 dB relative to its neighbors.  The historical-reference approach
can't detect this because the effect is caused by the CURRENT spectral shape,
not by a change over time.

Design
------
Compare each band's current smoothed energy to the average of its non-adjacent
neighbors (skip ±1, average ±2–4).  Bands that are locally elevated because
the notches dipped their neighbors get cut.  Only fires when deep notches are
active — during clean speech the local spectral shape is not distorted by
notches so false cuts are prevented by the notch-active gate.

Guard conditions:
  1. Only when any notch is deeper than NOTCH_ACTIVE_THRESHOLD_DB — no cuts
     during clean speech so natural voice formants are untouched.
  2. Bands within NOTCH_GUARD (25%) of an active notch are excluded — the
     notch itself is handled by NotchBank, not here.

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

    # Short-term smoothing (frames at ~100 fps)
    SHORT_ALPHA   = 0.05    # ~20 frames = 200 ms — smooth enough to survive
                            # phoneme transitions without chasing transients

    # Local-neighbor comparison window
    # For band i, skip ±N_SKIP immediate neighbors, then average ±N_COMPARE bands.
    # Example: N_SKIP=1, N_COMPARE=3 → compares band i to [i-4,i-3,i-2,i+2,i+3,i+4].
    N_SKIP        = 1
    N_COMPARE     = 3

    # Cut trigger: band must be this many dB above its local neighbor average
    PROMINENCE_DB = 2.0
    MAX_CUT_DB    = -6.0
    CUT_Q         = 2.5     # wide, musical cut — 1/3-oct bandwidth at this Q
    CUT_SMOOTHING = 0.05    # slow attack (~20 frames) to avoid modulation

    # Gate: only compare / cut when deep notches are active.
    # During clean speech the spectral shape is not distorted by notches,
    # so this prevents cutting natural voice formants.
    NOTCH_ACTIVE_DB   = -10.0   # notch must be deeper than this to count
    NOTCH_GUARD       = 0.25    # ±25% log-scale around active notch → no cut

    # Silence gate
    VOICE_FLOOR   = 1e-5

    def __init__(self, bin_freqs: np.ndarray, sr: int = 48000):
        self.bin_freqs  = bin_freqs
        self.sr         = sr

        self.band_freqs = np.logspace(
            np.log10(self.F_MIN), np.log10(self.F_MAX), self.N_BANDS)

        # Bin-to-band mapping
        edges = np.logspace(
            np.log10(self.F_MIN * 2 ** (-1.0 / self.N_BANDS)),
            np.log10(self.F_MAX * 2 ** ( 1.0 / self.N_BANDS)),
            self.N_BANDS + 1)
        self._band_masks = [
            (bin_freqs >= edges[i]) & (bin_freqs < edges[i + 1])
            for i in range(self.N_BANDS)
        ]

        # Pre-compute neighbor index sets for local comparison
        self._neighbor_idx = []
        for i in range(self.N_BANDS):
            nbrs = [j for j in range(self.N_BANDS)
                    if (self.N_SKIP < abs(j - i) <= self.N_SKIP + self.N_COMPARE)]
            self._neighbor_idx.append(nbrs)

        self._short_norm = np.ones(self.N_BANDS, dtype=np.float64) / self.N_BANDS
        self._cut_db     = np.zeros(self.N_BANDS, dtype=np.float64)
        self._max_prom   = np.zeros(self.N_BANDS, dtype=np.float64)  # diagnostics

        self._filters = [
            PeakingEQ(f, sr=sr, q=self.CUT_Q, gain_db=0.0)
            for f in self.band_freqs
        ]

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, stft_mag: np.ndarray, rms: float,
               active_notches: 'list[tuple[float,float,float]]'):
        """
        stft_mag       : (N_FREQ,) linear magnitude from current STFT frame
        rms            : RMS of current audio block (silence gate)
        active_notches : notch_bank.active_notches → [(freq, depth_db, q), ...]
        """
        # Per-band power, normalised to spectral shape
        band_power = np.array([
            float((stft_mag[m] ** 2).sum()) + 1e-20
            for m in self._band_masks
        ])
        total      = band_power.sum() + 1e-20
        band_norm  = band_power / total

        # Short-term smoothing — 200 ms, survives phoneme transitions
        self._short_norm = (self.SHORT_ALPHA * band_norm +
                            (1.0 - self.SHORT_ALPHA) * self._short_norm)

        # ── Gate: only act when deep notches are present ───────────────────
        deep_notches = [(nf, nd, nq) for nf, nd, nq in active_notches
                        if nd < self.NOTCH_ACTIVE_DB]
        if not deep_notches or rms <= self.VOICE_FLOOR:
            # Release cuts toward 0 while no notches are active
            self._cut_db *= (1.0 - self.CUT_SMOOTHING)
            for i, filt in enumerate(self._filters):
                filt.set_gain(float(self._cut_db[i]))
            return

        # Guard zone: bands within NOTCH_GUARD of an active notch
        guard = np.zeros(self.N_BANDS, dtype=bool)
        for nf, _, _ in deep_notches:
            for i, bf in enumerate(self.band_freqs):
                if abs(bf - nf) / bf < self.NOTCH_GUARD:
                    guard[i] = True

        # Local-neighbor prominence: how much is each band above its neighbors?
        prom_db = np.zeros(self.N_BANDS, dtype=np.float64)
        for i, nbrs in enumerate(self._neighbor_idx):
            if not nbrs:
                continue
            neighbor_avg = self._short_norm[nbrs].mean() + 1e-20
            with np.errstate(divide='ignore', invalid='ignore'):
                p = 10.0 * np.log10(self._short_norm[i] / neighbor_avg)
            prom_db[i] = 0.0 if not np.isfinite(p) else p

        self._max_prom = np.maximum(self._max_prom, prom_db)

        # Target cut: proportional to excess above threshold
        excess = np.maximum(prom_db - self.PROMINENCE_DB, 0.0)
        target = np.clip(-excess, self.MAX_CUT_DB, 0.0)
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
        active = [(self.band_freqs[i], self._cut_db[i])
                  for i in range(self.N_BANDS) if self._cut_db[i] < -0.5]
        lines = []
        if not active:
            lines.append('SpectralFlattener: no active cuts')
        else:
            lines.append('SpectralFlattener cuts:')
            for f, db in active:
                lines.append(f'  {f:.0f} Hz: {db:.1f} dB')
        lines.append(f'  threshold={self.PROMINENCE_DB} dB  '
                     f'(local-neighbor: skip±{self.N_SKIP}, avg±{self.N_COMPARE})')
        top = sorted(enumerate(self._max_prom), key=lambda x: -x[1])[:6]
        lines.append('  peak local-prominence per band (top 6):')
        for i, pk in top:
            marker = ' ← cut' if pk >= self.PROMINENCE_DB else ''
            lines.append(f'    {self.band_freqs[i]:.0f} Hz: {pk:.2f} dB{marker}')
        return '\n'.join(lines)
