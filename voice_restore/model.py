"""
voice_restore/model.py — Learned spectral envelope restorer for post-notch voice recovery.

After the NotchBank cuts feedback frequencies, this model predicts how much to
restore each STFT bin based on:
  - The notched signal's log-magnitude spectrum
  - Which bins were cut and by how much (notch mask)
  - A harmonic template derived from the current F0 (CREPE pitch estimate)
  - Pitch trajectory features (F0, confidence, ΔF0, Δ²F0)

The output is a per-bin gain in [0, 1], where:
  0 = leave the notch as-is
  1 = restore fully up to the pre-notch level

Hard constraint: the model can never boost a bin above its pre-notch level.
This preserves feedback suppression unconditionally — the restorer can only
give back energy that the notch bank removed, not add new energy.

Predictive behaviour comes from the GRU: it carries pitch trajectory state
across frames, allowing it to anticipate where harmonic peaks are heading
before the next frame arrives.

Architecture mirrors FeedbackDetector:
  per-frame freq conv → per-bin GRU → per-bin gain output
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent

# Mirror feedback_detect/model.py constants — keep in sync if those change
SR     = 48000
N_FFT  = 1024
HOP    = 480
N_FREQ = N_FFT // 2 + 1   # 513

N_SPECTRAL = 3   # log_mag_notched, notch_mask_db (normalised), harmonic_template
N_PITCH    = 4   # f0_norm, confidence, delta_f0_norm, delta2_f0_norm
N_IN       = N_SPECTRAL + N_PITCH   # 7 channels total (pitch broadcast to freq axis)

# F0 range for normalisation
F0_MIN_HZ  = 50.0
F0_MAX_HZ  = 2000.0

_bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)  # (N_FREQ,)


# ── Helpers ───────────────────────────────────────────────────────────────────

def harmonic_template(f0_hz: float, n_harmonics: int = 16,
                      sigma_bins: float = 1.5) -> np.ndarray:
    """
    Synthesise a per-bin harmonic comb for a given F0.

    Each harmonic k*F0 contributes a Gaussian bump centred on the nearest bin,
    weighted by 1/k (higher harmonics are quieter). Unvoiced / no-pitch frames
    should pass f0_hz=0, which returns a zero vector.

    Returns (N_FREQ,) float32.
    """
    template = np.zeros(N_FREQ, dtype=np.float32)
    if f0_hz <= 0.0:
        return template
    bin_width = _bin_freqs[1] - _bin_freqs[0]   # Hz per bin
    for k in range(1, n_harmonics + 1):
        h_freq = k * f0_hz
        if h_freq > _bin_freqs[-1]:
            break
        distances = (_bin_freqs - h_freq) / bin_width   # distance in bins
        template += (1.0 / k) * np.exp(-0.5 * (distances / sigma_bins) ** 2)
    # Normalise peak to 1
    peak = template.max()
    if peak > 0:
        template /= peak
    return template


def normalise_f0(f0_hz: float) -> float:
    """Map F0 from [F0_MIN_HZ, F0_MAX_HZ] → [0, 1]. 0 = unvoiced."""
    if f0_hz <= 0.0:
        return 0.0
    return float(np.clip((f0_hz - F0_MIN_HZ) / (F0_MAX_HZ - F0_MIN_HZ), 0.0, 1.0))


# ── Model ─────────────────────────────────────────────────────────────────────

class VoiceRestorer(nn.Module):
    """
    Per-bin gain predictor for spectral envelope restoration.

    Parameters
    ----------
    freq_ch    : channels in the per-frame frequency conv
    gru_hidden : GRU hidden size (shared across all frequency bins)
    """

    def __init__(self, freq_ch: int = 32, gru_hidden: int = 64):
        super().__init__()
        self.freq_ch    = freq_ch
        self.gru_hidden = gru_hidden

        self.freq_conv = nn.Sequential(
            nn.Conv1d(N_IN, freq_ch, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(freq_ch, freq_ch, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(freq_ch, gru_hidden, batch_first=True)
        self.fc  = nn.Linear(gru_hidden, 1)

    def forward(self,
                spectral: torch.Tensor,
                pitch:    torch.Tensor,
                h:        torch.Tensor | None = None):
        """
        spectral : (B, N_SPECTRAL, N_FREQ, T)
                   channels: [log_mag_notched, notch_mask_db_norm, harmonic_template]
        pitch    : (B, N_PITCH, T)
                   channels: [f0_norm, confidence, delta_f0_norm, delta2_f0_norm]
        h        : GRU hidden (1, B*N_FREQ, gru_hidden) or None

        Returns
        -------
        gain  : (B, N_FREQ, T)  in [0, 1]
                  0 = keep notch, 1 = restore fully to pre-notch level
        h_new : updated GRU hidden state
        """
        B, _, F, T = spectral.shape

        # Broadcast pitch features across frequency axis → (B, N_PITCH, F, T)
        p = pitch.unsqueeze(2).expand(-1, -1, F, -1)
        x = torch.cat([spectral, p], dim=1)          # (B, N_IN, F, T)

        # Per-frame frequency conv
        x = x.permute(0, 3, 1, 2).reshape(B * T, N_IN, F)
        x = self.freq_conv(x)                         # (B*T, freq_ch, F)
        x = x.reshape(B, T, self.freq_ch, F).permute(0, 3, 1, 2)
        # → (B, F, T, freq_ch)

        # Per-bin GRU along time axis
        x = x.reshape(B * F, T, self.freq_ch)
        x, h_new = self.gru(x, h)                     # (B*F, T, gru_hidden)

        gain = torch.sigmoid(self.fc(x)).squeeze(-1)  # (B*F, T)
        gain = gain.reshape(B, F, T)                  # (B, N_FREQ, T)

        return gain, h_new

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Inference helper ──────────────────────────────────────────────────────────

def apply_restoration(notched_mag:   torch.Tensor,
                      notch_mask_db: torch.Tensor,
                      gain:          torch.Tensor) -> torch.Tensor:
    """
    Apply the model's gain prediction to the notched magnitude.

    restored_mag = notched_mag * 10^( gain * (-notch_mask_db) / 20 )

    The exponent is always ≥ 0 (since notch_mask_db ≤ 0), so we can only
    ever boost — and only up to the pre-notch level. No feedback energy added.

    notched_mag   : (B, N_FREQ, T) linear magnitude
    notch_mask_db : (B, N_FREQ, T) ≤ 0  — how much each bin was attenuated
    gain          : (B, N_FREQ, T) ∈ [0, 1]

    Returns restored_mag : (B, N_FREQ, T)
    """
    recovery_db  = gain * (-notch_mask_db)          # ∈ [0, abs(notch_depth)]
    scale        = 10.0 ** (recovery_db / 20.0)
    return notched_mag * scale
