"""
voice_restore/model_v3.py — VoiceRestorer V3.

V3 keeps the hard notch constraint but changes the learning objective:
the model is trained to make the output perceptually similar to an "allowed"
target in ERB-like bands, rather than trying to numerically reconstruct the
removed bins.

The architecture intentionally stays close to V2 so we can isolate the effect
of the psychoacoustic objective.
"""

import numpy as np
import torch
import torch.nn as nn

SR     = 48000
N_FFT  = 1024
HOP    = 480
N_FREQ = N_FFT // 2 + 1

N_SPECTRAL = 4   # log_mag_notched, harmonic_template, notch_strength, aperiodic_residual
N_COND     = 7   # f0_norm, confidence, delta_f0, delta2_f0, voiced_gate, flux, flatness
N_IN       = N_SPECTRAL + N_COND

F0_MIN_HZ    = 50.0
F0_MAX_HZ    = 2000.0
MAX_NOTCH_DB = 48.0
MAX_COMP_DB  = 6.0

_bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)


def harmonic_template(f0_hz: float, n_harmonics: int = 16,
                      sigma_bins: float = 1.5) -> np.ndarray:
    template = np.zeros(N_FREQ, dtype=np.float32)
    if f0_hz <= 0.0:
        return template
    bin_width = _bin_freqs[1] - _bin_freqs[0]
    for k in range(1, n_harmonics + 1):
        h_freq = k * f0_hz
        if h_freq > _bin_freqs[-1]:
            break
        distances = (_bin_freqs - h_freq) / bin_width
        template += (1.0 / k) * np.exp(-0.5 * (distances / sigma_bins) ** 2)
    peak = template.max()
    if peak > 0:
        template /= peak
    return template


def normalise_f0(f0_hz: float) -> float:
    if f0_hz <= 0.0:
        return 0.0
    return float(np.clip((f0_hz - F0_MIN_HZ) / (F0_MAX_HZ - F0_MIN_HZ), 0.0, 1.0))


def notch_strength_from_mask(notch_mask_db: torch.Tensor) -> torch.Tensor:
    return (-notch_mask_db / MAX_NOTCH_DB).clamp(0.0, 1.0)


def repair_region_from_mask(notch_mask_db: torch.Tensor,
                            kernel_size: int = 17) -> torch.Tensor:
    """
    Slightly wider than V2 because V3 is about perceptual redistribution.
    """
    strength = notch_strength_from_mask(notch_mask_db)
    pad = kernel_size // 2
    region = torch.nn.functional.avg_pool1d(
        strength.permute(0, 2, 1).reshape(-1, 1, strength.shape[1]),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    region = region.reshape(strength.shape[0], strength.shape[2], strength.shape[1]).permute(0, 2, 1)
    return region.clamp(0.0, 1.0)


def make_aperiodic_residual(log_mag: torch.Tensor,
                            kernel_size: int = 9) -> torch.Tensor:
    pad = kernel_size // 2
    smooth = torch.nn.functional.avg_pool1d(
        log_mag.permute(0, 2, 1).reshape(-1, 1, log_mag.shape[1]),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    smooth = smooth.reshape(log_mag.shape[0], log_mag.shape[2], log_mag.shape[1]).permute(0, 2, 1)
    return log_mag - smooth


class VoiceRestorerV3(nn.Module):
    """
    Same basic model family as V2; the main experiment in V3 is the loss.
    """

    def __init__(self, freq_ch: int = 32, gru_hidden: int = 64):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv1d(N_IN, freq_ch, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(freq_ch, freq_ch, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(freq_ch, gru_hidden, batch_first=True)
        self.fc_harmonic  = nn.Linear(gru_hidden, 1)
        self.fc_aperiodic = nn.Linear(gru_hidden, 1)

    def forward(self,
                spectral: torch.Tensor,
                cond:     torch.Tensor,
                h:        torch.Tensor | None = None):
        spectral = spectral.float()
        cond     = cond.float()
        B, _, F, T = spectral.shape

        cond_b = cond.unsqueeze(2).expand(-1, -1, F, -1)
        x = torch.cat([spectral, cond_b], dim=1)

        x = x.permute(0, 3, 1, 2).reshape(B * T, N_IN, F)
        x = self.freq_conv(x)
        x = x.reshape(B, T, -1, F).permute(0, 3, 1, 2)

        x = x.reshape(B * F, T, x.shape[-1])
        x, h_new = self.gru(x, h)

        harmonic_gain  = torch.sigmoid(self.fc_harmonic(x)).squeeze(-1).reshape(B, F, T)
        aperiodic_gain = torch.sigmoid(self.fc_aperiodic(x)).squeeze(-1).reshape(B, F, T)

        voiced_gate = cond[:, 4:5, :].expand(-1, F, -1)
        gain = voiced_gate * harmonic_gain + (1.0 - voiced_gate) * aperiodic_gain
        return gain, h_new

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def apply_compensation(notched_mag:   torch.Tensor,
                       notch_mask_db: torch.Tensor,
                       gain:          torch.Tensor) -> torch.Tensor:
    """
    Forbidden bins remain untouched. Gains only affect safe bins around the notch.
    """
    safe_bins     = (notch_mask_db > -3.0).float()
    repair_region = repair_region_from_mask(notch_mask_db)
    boost_db      = gain * MAX_COMP_DB * safe_bins * repair_region
    return notched_mag * (10.0 ** (boost_db / 20.0))
