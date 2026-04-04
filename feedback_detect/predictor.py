"""
feedback_detect/predictor.py — Anticipatory feedback suppression.

Learns which frequencies tend to ring for a specific room/system/voice and
pre-attenuates them when sub-threshold activity is detected at those bins —
before the ring fully develops.

How it works
------------
1. Every time a frequency rings (NotchBank fires), its risk score increments.
2. Risk decays slowly over time (~8 min half-life) so the profile adapts to
   room/setup changes across sessions.
3. Each frame, if risk[freq] > RISK_THRESHOLD AND the detector's per-bin
   probability exceeds PRE_TRIGGER_PROB (0.15, well below the main threshold
   of 0.4), a pre-emptive notch is requested at that frequency.
4. Pre-emptive notches start at PREEMPTIVE_DEPTH_DB (-24 dB) — enough to
   prevent build-up without audibly coloring the sound. If they retrigger,
   NotchBank's normal adaptive depth/Q takes over.
5. The profile is persisted to JSON so learning carries over across sessions.

Usage
-----
    predictor = FeedbackPredictor(profile_path='feedback_profile.json')

    # each frame, after detector runs:
    predictor.on_notch_events(notch_bank.active_notches)
    preemptive = predictor.get_preemptive(bin_freqs, prob_np)
    notch_bank.update(detected_freqs, bin_freqs, prob_np,
                      preemptive_freqs=preemptive)
    predictor.decay()

    # at session end:
    predictor.save()
"""

import json
from pathlib import Path

import numpy as np


class FeedbackPredictor:
    # ── Risk profile ──────────────────────────────────────────────────────────
    RISK_INCREMENT    = 1.0       # score added each time a frequency rings
    RISK_MAX          = 10.0      # cap so one very persistent ring doesn't dominate
    # Half-life ≈ 8 min at 100 fps → decay factor per frame
    RISK_DECAY_FACTOR = 0.5 ** (1.0 / 48_000)   # ~0.9999856 per frame
    RISK_MIN_KEEP     = 0.05      # prune entries below this

    # ── Pre-trigger ───────────────────────────────────────────────────────────
    RISK_THRESHOLD    = 2.0       # minimum score to enable pre-trigger for a freq
    PRE_TRIGGER_PROB  = 0.15      # sub-threshold prob that fires pre-emption
    PREEMPTIVE_DEPTH_DB = -24.0   # initial depth of pre-emptive notch (lighter than reactive -48)

    # ── Matching ──────────────────────────────────────────────────────────────
    FREQ_TOL_HZ       = 150       # same tolerance as NotchBank

    def __init__(self, sr: int = 48000, profile_path: str | Path | None = None):
        self.sr           = sr
        self.profile_path = Path(profile_path) if profile_path else None
        self._risk: dict[float, float] = {}
        self._decay_acc   = 0
        if self.profile_path:
            self.load()

    # ── public API ────────────────────────────────────────────────────────────

    def on_notch_events(self, active_notches: list[tuple[float, float, float]]):
        """
        Call every frame with notch_bank.active_notches → (freq, depth_db, q).
        Increments risk for any notch that is fully slammed (at or near max depth).
        """
        for freq, depth_db, _ in active_notches:
            if depth_db <= -40.0:   # only count fully active notches, not decaying ones
                existing = self._find_close(freq)
                if existing is not None:
                    self._risk[existing] = min(
                        self._risk[existing] + self.RISK_INCREMENT, self.RISK_MAX)
                else:
                    self._risk[freq] = self.RISK_INCREMENT

    def get_preemptive(self,
                       bin_freqs: np.ndarray,
                       prob_np:   np.ndarray) -> list[float]:
        """
        Returns frequencies that should receive a pre-emptive notch this frame.
        Call after the detector runs, before NotchBank.update().

        A frequency is pre-triggered when:
          - risk score > RISK_THRESHOLD  (historically prone to ring here)
          - detector prob > PRE_TRIGGER_PROB  (activity building at that bin)
        """
        preemptive: list[float] = []
        for freq, risk in self._risk.items():
            if risk < self.RISK_THRESHOLD:
                continue
            bin_idx = int(np.argmin(np.abs(bin_freqs - freq)))
            if prob_np[bin_idx] > self.PRE_TRIGGER_PROB:
                preemptive.append(freq)
        return preemptive

    def decay(self):
        """Decay all risk scores. Call once per frame."""
        self._decay_acc += 1
        # Apply decay in batches of 100 frames to avoid per-frame float overhead
        if self._decay_acc >= 100:
            factor = self.RISK_DECAY_FACTOR ** self._decay_acc
            to_delete = [f for f, r in self._risk.items()
                         if r * factor < self.RISK_MIN_KEEP]
            for f in to_delete:
                del self._risk[f]
            for f in self._risk:
                self._risk[f] *= factor
            self._decay_acc = 0

    def save(self):
        """Persist risk profile to disk."""
        if self.profile_path is None:
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, 'w') as fh:
            json.dump({str(k): v for k, v in self._risk.items()}, fh, indent=2)

    def load(self):
        """Load risk profile from disk. Silent if file doesn't exist."""
        if self.profile_path is None or not self.profile_path.exists():
            return
        try:
            with open(self.profile_path) as fh:
                data = json.load(fh)
            self._risk = {float(k): float(v) for k, v in data.items()}
            print(f'[FeedbackPredictor] loaded {len(self._risk)} risk entries '
                  f'from {self.profile_path}')
        except (json.JSONDecodeError, ValueError):
            self._risk = {}

    @property
    def risk_profile(self) -> dict[float, float]:
        """Read-only view of current risk scores."""
        return dict(self._risk)

    # ── private ───────────────────────────────────────────────────────────────

    def _find_close(self, freq: float) -> float | None:
        for existing in self._risk:
            if abs(existing - freq) < self.FREQ_TOL_HZ:
                return existing
        return None
