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
    SHORT_ALPHA   = 0.05    # ~20 frames = 200 ms

    # Local-neighbor comparison window
    # N_SKIP=0 means immediate neighbors ARE included — the notched adjacent
    # bands (e.g. 651 Hz and 987 Hz) are directly compared to 802 Hz, which
    # is exactly what exposes the notch-induced spectral hump.
    N_SKIP        = 0
    N_COMPARE     = 3

    # Prominence integration: slow EMA of local prominence (notch-gate open only).
    # Phoneme bursts (brief, large) average toward zero over the integration window.
    # Persistent notch-induced elevation (~1-2 dB sustained) accumulates above threshold.
    PROM_INT_ALPHA  = 0.002   # 500-frame = 5 s time constant
    PROM_INT_DECAY  = 0.0005  # release when gate closes (slower than attack)

    # Cut trigger: integrated WEIGHTED band prominence must exceed this.
    # Prominence is scaled by _prom_weight before integration, so the effective
    # threshold is lower in the perceptually sensitive voice-presence region
    # (~800–2000 Hz) and higher at the extremes.
    PROMINENCE_DB = 1.0
    MAX_CUT_DB    = -4.0
    CUT_Q         = 2.5     # wide, musical cut
    CUT_SMOOTHING = 0.05    # smooth attack to avoid modulation

    # Gate: only compare / cut when deep notches are active.
    # During clean speech the spectral shape is not distorted by notches,
    # so this prevents cutting natural voice formants.
    NOTCH_ACTIVE_DB   = -10.0   # notch must be deeper than this to count
    # Guard zone: don't cut bands AT the notch frequency — the notch bank
    # already handles those. ±8% is just wider than the notch's own -3dB
    # bandwidth (Q≈5 → ±9%), so the notch itself is protected but the
    # "bridge" band between adjacent notches (e.g. 802 Hz between 680/987)
    # is eligible for SpectralFlattener cuts.
    NOTCH_GUARD       = 0.08    # ±8% around active notch → no cut

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

        self._short_norm  = np.ones(self.N_BANDS, dtype=np.float64) / self.N_BANDS
        self._cut_db      = np.zeros(self.N_BANDS, dtype=np.float64)
        self._smooth_prom = np.zeros(self.N_BANDS, dtype=np.float64)  # integrated prominence
        self._peak_cut_db = np.zeros(self.N_BANDS, dtype=np.float64)  # most negative cut seen

        # Perceptual prominence weights — log-Gaussian centred at 1200 Hz.
        # The voice-presence region (~800–2000 Hz) gets weight ~2×, so coloration
        # there crosses PROMINENCE_DB at half the raw prominence required at extremes.
        # Effective threshold: PROMINENCE_DB / weight(f).
        log_oct = np.log2(self.band_freqs / 1200.0)   # octaves from 1200 Hz
        self._prom_weight = np.clip(
            0.5 + 1.5 * np.exp(-0.5 * (log_oct / 1.5) ** 2),
            0.5, 2.0).astype(np.float64)

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

        # ── Gate: only integrate when deep notches are present ────────────
        deep_notches = [(nf, nd, nq) for nf, nd, nq in active_notches
                        if nd < self.NOTCH_ACTIVE_DB]

        if not deep_notches or rms <= self.VOICE_FLOOR:
            # Decay integrated prominence and release cuts while gate is closed
            self._smooth_prom *= (1.0 - self.PROM_INT_DECAY)
            self._cut_db      *= (1.0 - self.CUT_SMOOTHING)
            for i, filt in enumerate(self._filters):
                filt.set_gain(float(self._cut_db[i]))
            return

        # Guard zone: bands within NOTCH_GUARD of an active notch
        guard = np.zeros(self.N_BANDS, dtype=bool)
        for nf, _, _ in deep_notches:
            for i, bf in enumerate(self.band_freqs):
                if abs(bf - nf) / bf < self.NOTCH_GUARD:
                    guard[i] = True

        # Instantaneous local-neighbor prominence
        prom_db = np.zeros(self.N_BANDS, dtype=np.float64)
        for i, nbrs in enumerate(self._neighbor_idx):
            if not nbrs:
                continue
            neighbor_avg = self._short_norm[nbrs].mean() + 1e-20
            with np.errstate(divide='ignore', invalid='ignore'):
                p = 10.0 * np.log10(self._short_norm[i] / neighbor_avg)
            prom_db[i] = 0.0 if not np.isfinite(p) else p

        # Integrate prominence over time — phoneme bursts average toward zero,
        # persistent notch-induced elevation accumulates above threshold.
        # Weight by perceptual sensitivity before integrating so the voice-presence
        # region (~800–2000 Hz) triggers at a lower raw prominence.
        self._smooth_prom += self.PROM_INT_ALPHA * (
            np.maximum(prom_db * self._prom_weight, 0.0) - self._smooth_prom)

        # Target cut based on INTEGRATED prominence
        excess = np.maximum(self._smooth_prom - self.PROMINENCE_DB, 0.0)
        target = np.clip(-excess, self.MAX_CUT_DB, 0.0)
        target[guard] = 0.0

        # Smooth toward target
        self._cut_db += self.CUT_SMOOTHING * (target - self._cut_db)
        self._peak_cut_db = np.minimum(self._peak_cut_db, self._cut_db)

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
                     f'(local-neighbor: skip±{self.N_SKIP}, avg±{self.N_COMPARE}  '
                     f'int_alpha={self.PROM_INT_ALPHA})')
        top = sorted(enumerate(self._smooth_prom), key=lambda x: -x[1])[:8]
        lines.append('  integrated prominence per band (top 8):')
        for i, sp in top:
            marker = ' ← cut' if sp >= self.PROMINENCE_DB else ''
            lines.append(f'    {self.band_freqs[i]:.0f} Hz: {sp:.3f} dB{marker}')
        # Peak cuts show what was actually applied during the run (cuts release at end)
        peak = [(self.band_freqs[i], self._peak_cut_db[i])
                for i in range(self.N_BANDS) if self._peak_cut_db[i] < -0.1]
        if peak:
            lines.append('  peak cuts applied during run:')
            for f, db in sorted(peak, key=lambda x: x[1]):
                lines.append(f'    {f:.0f} Hz: {db:.2f} dB')
        else:
            lines.append('  peak cuts applied during run: none')
        return '\n'.join(lines)
