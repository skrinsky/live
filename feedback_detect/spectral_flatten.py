"""
feedback_detect/spectral_flatten.py — Absorbed-ring EQ correction.

Detects frequencies that the notch bank cannot handle because they are
being absorbed (merged via FREQ_TOL_RATIO) by a nearby notch whose
actual -3dB bandwidth does NOT cover them.  These bins have high
detector probability but zero suppression — they just ring.

Per frame:
  1. Mark bins as "covered" if any active notch's half-bandwidth
     (notch_freq / 2*Q) physically includes them.
  2. Bins with prob > PROB_THRESHOLD that are NOT covered are
     "absorbed overflow" — ringing without suppression.
  3. Accumulate a slow per-bin running average of overflow state.
  4. When a bin's average exceeds UNCOV_TRIGGER (persistent, not
     transient), cluster it with neighbours and apply a gentle wide
     peaking cut at the cluster peak frequency.

Only cuts — never boosts.  Cuts cannot cause feedback.
"""

import numpy as np
from scipy.signal import lfilter


class AdaptiveMakeupGain:
    """
    Feedback-aware broadband makeup gain.

    Slowly ramps gain up after the notch bank quiets the room.
    Backs off immediately when the detector fires new rings.
    Finds its own ceiling — no manual tuning needed.

    Signal: len(detected_freqs) per frame.
      - 0 detections → room is quiet → ramp up
      - any detection → ring is active → back off + hold
    """

    MAX_DB            = 6.0    # never boost more than this
    RAMP_DB_PER_FRAME = 0.002  # +0.2 dB/s at 100 fps — slow climb
    BACK_OFF_DB       = 1.5    # step back this many dB on each detection event
    HOLD_FRAMES       = 300    # frames to hold after backing off before ramping (~3 s)

    def __init__(self):
        self.current_db = 0.0
        self._hold      = 0

    def update(self, n_detections: int) -> float:
        """
        n_detections : len(detected_freqs) from the detector this frame.
        Returns linear gain scalar to multiply the output block by.
        """
        if n_detections > 0:
            self.current_db = max(0.0, self.current_db - self.BACK_OFF_DB)
            self._hold      = self.HOLD_FRAMES
        elif self._hold > 0:
            self._hold -= 1
        else:
            self.current_db = min(self.MAX_DB,
                                  self.current_db + self.RAMP_DB_PER_FRAME)
        return 10.0 ** (self.current_db / 20.0)


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

    PROB_THRESHOLD  = 0.25     # bins above this are "ringing"
    NOTCH_MIN_DB    = -10.0    # notch must be deeper to count as coverage
    MIN_FREQ_HZ     = 80.0     # ignore below HPF cutoff

    # Per-bin running average of "high-prob and uncovered"
    UNCOV_ALPHA     = 0.005    # attack  ~200 frames / ~2 s at 100 fps
    UNCOV_DECAY     = 0.002    # release ~500 frames / ~5 s
    UNCOV_TRIGGER   = 0.10     # fire cut when avg exceeds this

    # Clustering overflow bins into cuts
    CLUSTER_RATIO   = 0.10     # merge bins within 10% of each other

    # Cut parameters — gentle, wide
    MIN_CUT_DB      = -2.0     # depth at UNCOV_TRIGGER
    MAX_CUT_DB      = -6.0     # depth at full saturation
    CUT_Q           = 2.0
    CUT_SMOOTHING   = 0.02     # EMA per frame toward target
    UPDATE_INTERVAL = 50       # frames between rebuilding cut targets
    IDLE_FRAMES     = 200      # frames at ~0 dB before removing filter

    def __init__(self, bin_freqs: np.ndarray, sr: int = 48000):
        self.bin_freqs    = bin_freqs
        self.sr           = sr
        self._uncov_prob  = np.zeros(len(bin_freqs), dtype=np.float32)
        # {freq: [PeakingEQ, current_cut_db, idle_counter]}
        self._filters:    dict[float, list]  = {}
        self._targets:    dict[float, float] = {}
        self._peak_cuts:  dict[float, float] = {}
        self._frame_count = 0

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, prob_np: np.ndarray,
               active_notches: 'list[tuple[float, float, float]]'):
        """
        prob_np        : (N_FREQ,) per-bin detector probabilities
        active_notches : notch_bank.active_notches → [(freq, depth_db, q), ...]
        """
        # ── Which bins are physically covered by an active notch? ──────────
        covered = np.zeros(len(self.bin_freqs), dtype=bool)
        for f, d, q in active_notches:
            if d > self.NOTCH_MIN_DB:
                continue
            half_bw = f / (2.0 * q)
            covered |= np.abs(self.bin_freqs - f) <= half_bw

        # ── Accumulate per-bin overflow state ──────────────────────────────
        overflow = ((prob_np > self.PROB_THRESHOLD)
                    & ~covered
                    & (self.bin_freqs >= self.MIN_FREQ_HZ))
        self._uncov_prob[overflow]  += self.UNCOV_ALPHA * (
            1.0 - self._uncov_prob[overflow])
        self._uncov_prob[~overflow]  = np.maximum(
            0.0, self._uncov_prob[~overflow] - self.UNCOV_DECAY)

        # ── Periodically rebuild cut targets from accumulated state ────────
        self._frame_count += 1
        if self._frame_count >= self.UPDATE_INTERVAL:
            self._frame_count = 0
            self._rebuild_targets()
        self._smooth()

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        y = audio_block.astype(np.float32)
        for filt, cut, _ in self._filters.values():
            if abs(cut) > 0.05:
                y = filt.process(y)
        return y

    def summary(self) -> str:
        active = [(f, v[1]) for f, v in self._filters.items()
                  if abs(v[1]) > 0.1]
        lines = []
        if not active:
            lines.append('ChronicRingEQ: no active cuts')
        else:
            lines.append('ChronicRingEQ absorbed-ring cuts:')
            for f, cut in sorted(active, key=lambda x: x[1]):
                bi = int(np.argmin(np.abs(self.bin_freqs - f)))
                lines.append(f'  {f:.0f} Hz: {cut:.1f} dB'
                              f'  (uncov_avg={self._uncov_prob[bi]:.3f})')
        if self._peak_cuts:
            peak = [(f, v) for f, v in self._peak_cuts.items() if v < -0.1]
            if peak:
                lines.append('  peak cuts during run:')
                for f, cut in sorted(peak, key=lambda x: x[1]):
                    lines.append(f'    {f:.0f} Hz: {cut:.2f} dB')
        return '\n'.join(lines)

    # ── private ───────────────────────────────────────────────────────────────

    def _rebuild_targets(self):
        trigger_bins = np.where(self._uncov_prob > self.UNCOV_TRIGGER)[0]

        new_targets: dict[float, float] = {}
        if len(trigger_bins):
            # Cluster bins by log-scale proximity
            clusters: list[list[int]] = []
            current = [int(trigger_bins[0])]
            for idx in trigger_bins[1:]:
                f_last = self.bin_freqs[current[-1]]
                f_this = self.bin_freqs[idx]
                if abs(f_this - f_last) / f_last < self.CLUSTER_RATIO:
                    current.append(int(idx))
                else:
                    clusters.append(current)
                    current = [int(idx)]
            clusters.append(current)

            for cluster in clusters:
                peak_i  = cluster[int(np.argmax(self._uncov_prob[cluster]))]
                freq    = float(self.bin_freqs[peak_i])
                level   = float(self._uncov_prob[peak_i])
                t       = min(1.0, (level - self.UNCOV_TRIGGER)
                              / max(1e-6, 1.0 - self.UNCOV_TRIGGER))
                new_targets[freq] = self.MIN_CUT_DB + t * (
                    self.MAX_CUT_DB - self.MIN_CUT_DB)

        # Release targets no longer active
        for freq in list(self._targets):
            if not any(abs(freq - nf) / freq < self.CLUSTER_RATIO
                       for nf in new_targets):
                self._targets[freq] = 0.0

        # Add / update
        for freq, cut in new_targets.items():
            existing = next((ef for ef in self._filters
                             if abs(ef - freq) / freq < self.CLUSTER_RATIO),
                            None)
            key = existing if existing is not None else freq
            self._targets[key] = cut
            if key not in self._filters:
                self._filters[key] = [
                    PeakingEQ(key, sr=self.sr, q=self.CUT_Q, gain_db=0.0),
                    0.0, 0]

    def _smooth(self):
        to_remove = []
        for freq, state in self._filters.items():
            filt, cur, idle = state
            target  = self._targets.get(freq, 0.0)
            new_cut = cur + self.CUT_SMOOTHING * (target - cur)
            state[1] = new_cut
            filt.set_gain(float(new_cut))
            self._peak_cuts[freq] = min(
                self._peak_cuts.get(freq, 0.0), new_cut)
            if target == 0.0 and abs(new_cut) < 0.05:
                state[2] = idle + 1
                if state[2] >= self.IDLE_FRAMES:
                    to_remove.append(freq)
            else:
                state[2] = 0
        for freq in to_remove:
            del self._filters[freq]
            self._targets.pop(freq, None)
