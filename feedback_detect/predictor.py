"""
feedback_detect/predictor.py — Voice-conditioned anticipatory feedback suppression.

Learns which frequencies tend to ring given the CURRENT VOICE SPECTRAL STATE,
not just which frequencies ring in general. This is the key distinction from a
simple room profile: pre-emption only fires during the voice patterns that have
historically preceded rings — sibilants pre-attenuate sibilant-prone frequencies,
vowels do not.

Voice state
-----------
Quantised into N_STATES=4 buckets using high-frequency energy ratio (HFR):
    HFR = energy(>HF_SPLIT_HZ) / total_energy

    State 0: HFR < 0.10  — silence / low voice / bass-heavy
    State 1: HFR 0.10–0.25 — voiced sounds, vowels
    State 2: HFR 0.25–0.45 — voiced fricatives (v, z, voiced th)
    State 3: HFR > 0.45   — sibilants (s, sh, f, unvoiced th)

Risk model
----------
2-D table: risk[freq][voice_state] — counts how often freq rings during each
voice state. A frequency is pre-triggered only when:
  1. risk[freq][current_state] > RISK_THRESHOLD  (historically risky in THIS voice state)
  2. detector prob[bin] > PRE_TRIGGER_PROB        (activity building right now)

This means:
- A freq that rings during sibilants but not vowels → pre-attenuated during
  sibilants only, leaving vowels untouched.
- A freq that rings regardless of voice state → pre-attenuated always
  (same as the old room-only approach, as a special case).

Persistence
-----------
Risk table saved as JSON across sessions so it builds up over multiple performances
with the same mic/PA/room. Decays with ~8 min half-life so profile adapts to
room/equipment changes over time.

Usage
-----
    predictor = FeedbackPredictor(bin_freqs, profile_path='feedback_profile.json')

    # each frame, after detector runs:
    preemptive = predictor.update(
        stft_mag_np,          # (N_FREQ,) current frame magnitude
        prob_np,              # (N_FREQ,) detector per-bin probabilities
        notch_bank.active_notches,  # [(freq, depth_db, q), ...]
    )
    notch_bank.update(detected_freqs, bin_freqs, prob_np,
                      preemptive_freqs=preemptive)

    # at session end:
    predictor.save()
"""

import json
from pathlib import Path

import numpy as np


class FeedbackPredictor:

    # ── Voice state ────────────────────────────────────────────────────────────
    N_STATES       = 4
    HF_SPLIT_HZ    = 3000.0    # above this = high-frequency energy
    # HFR bucket boundaries (0, B1, B2, B3, 1.0)
    HFR_THRESHOLDS = (0.10, 0.25, 0.45)   # 4 buckets

    # ── Risk accumulation ──────────────────────────────────────────────────────
    RISK_INCREMENT    = 1.0
    RISK_MAX          = 20.0
    RISK_DECAY_FACTOR = 0.5 ** (1.0 / 48_000)   # half-life ≈ 8 min at 100 fps
    RISK_MIN_KEEP     = 0.05

    # ── Pre-trigger ───────────────────────────────────────────────────────────
    RISK_THRESHOLD      = 2.0    # must ring this many times in a voice state to pre-trigger
    PRE_TRIGGER_PROB    = 0.05   # sub-threshold detector prob to fire pre-emption
    PREEMPTIVE_DEPTH_DB = -36.0

    # ── Matching ──────────────────────────────────────────────────────────────
    FREQ_TOL_RATIO = 0.18   # ±18% (~3 semitones) — log-scale mode matching

    def __init__(self,
                 bin_freqs:    np.ndarray,
                 sr:           int = 48000,
                 profile_path: 'str | Path | None' = None):
        self.bin_freqs    = bin_freqs
        self.sr           = sr
        self.profile_path = Path(profile_path) if profile_path else None
        self._hf_mask     = bin_freqs >= self.HF_SPLIT_HZ

        # risk[freq_hz] = np.array of shape (N_STATES,)
        self._risk: dict[float, np.ndarray] = {}
        self._decay_acc = 0
        self._prev_slammed: set[float] = set()   # notches that were at full depth last frame

        if self.profile_path:
            self.load()

    # ── public API ────────────────────────────────────────────────────────────

    def update(self,
               stft_mag_np:    np.ndarray,
               prob_np:        np.ndarray,
               active_notches: 'list[tuple[float, float, float]]') -> list[float]:
        """
        Main per-frame call. Returns list of frequencies for pre-emptive notching.

        stft_mag_np    : (N_FREQ,) linear magnitude of current frame
        prob_np        : (N_FREQ,) per-bin detector probabilities
        active_notches : notch_bank.active_notches → [(freq, depth_db, q), ...]
        """
        state = self._voice_state(stft_mag_np)
        self._accumulate(active_notches, state)
        preemptive = self._get_preemptive(prob_np, state)
        self.decay()
        return preemptive

    def save(self):
        if self.profile_path is None:
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {str(k): v.tolist() for k, v in self._risk.items()}
        with open(self.profile_path, 'w') as fh:
            json.dump(serialisable, fh, indent=2)

    def load(self):
        if self.profile_path is None or not self.profile_path.exists():
            return
        try:
            with open(self.profile_path) as fh:
                data = json.load(fh)
            self._risk = {}
            for k, v in data.items():
                arr = np.array(v, dtype=np.float32)
                if arr.ndim == 0:
                    # old scalar format — discard, start fresh for this entry
                    continue
                if arr.shape != (self.N_STATES,):
                    continue
                self._risk[float(k)] = arr
            print(f'[FeedbackPredictor] loaded {len(self._risk)} voice-conditioned '
                  f'risk entries from {self.profile_path}')
        except (json.JSONDecodeError, ValueError):
            self._risk = {}

    def decay(self):
        self._decay_acc += 1
        if self._decay_acc >= 100:
            factor = float(self.RISK_DECAY_FACTOR ** self._decay_acc)
            to_delete = [f for f, r in self._risk.items() if r.max() * factor < self.RISK_MIN_KEEP]
            for f in to_delete:
                del self._risk[f]
            for f in self._risk:
                self._risk[f] *= factor
            self._decay_acc = 0

    @property
    def risk_profile(self) -> dict[float, np.ndarray]:
        return dict(self._risk)

    def summary(self) -> str:
        lines = [f'FeedbackPredictor: {len(self._risk)} tracked frequencies']
        state_names = ['silence/bass', 'vowels', 'voiced-fric', 'sibilants']
        for freq in sorted(self._risk):
            r = self._risk[freq]
            active = [(state_names[s], f'{r[s]:.1f}') for s in range(self.N_STATES) if r[s] >= self.RISK_THRESHOLD]
            if active:
                lines.append(f'  {freq:.0f} Hz: ' + ', '.join(f'{n}={v}' for n, v in active))
        return '\n'.join(lines)

    # ── private ───────────────────────────────────────────────────────────────

    def _voice_state(self, stft_mag_np: np.ndarray) -> int:
        total = float(stft_mag_np.sum()) + 1e-8
        hf    = float(stft_mag_np[self._hf_mask].sum())
        hfr   = hf / total
        for i, thresh in enumerate(self.HFR_THRESHOLDS):
            if hfr < thresh:
                return i
        return self.N_STATES - 1

    def _accumulate(self,
                    active_notches: 'list[tuple[float, float, float]]',
                    state: int):
        # Only credit the voice state at ONSET (first frame the notch slams to full depth).
        # Counting every frame produces equal scores across all voice states regardless
        # of which state actually caused the ring.
        currently_slammed = {freq for freq, depth, _ in active_notches if depth <= -40.0}
        new_onsets = currently_slammed - self._prev_slammed
        for freq in new_onsets:
            existing = self._find_close(freq)
            if existing is not None:
                self._risk[existing][state] = min(
                    self._risk[existing][state] + self.RISK_INCREMENT, self.RISK_MAX)
            else:
                r = np.zeros(self.N_STATES, dtype=np.float32)
                r[state] = self.RISK_INCREMENT
                self._risk[freq] = r
        self._prev_slammed = currently_slammed

    def _get_preemptive(self, prob_np: np.ndarray, state: int) -> list[float]:
        preemptive = []
        for freq, risk in self._risk.items():
            if risk[state] < self.RISK_THRESHOLD:
                continue
            bin_idx = int(np.argmin(np.abs(self.bin_freqs - freq)))
            if prob_np[bin_idx] > self.PRE_TRIGGER_PROB:
                preemptive.append(freq)
        return preemptive

    def _find_close(self, freq: float) -> 'float | None':
        for existing in self._risk:
            if abs(existing - freq) / freq < self.FREQ_TOL_RATIO:
                return existing
        return None
