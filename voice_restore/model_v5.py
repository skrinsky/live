"""
voice_restore/model_v5.py — VoiceRestorer V5.

V5 uses a deterministic shoulder compensation baseline plus a learned residual.
This removes the all-zero collapse mode: even if the neural residual learns
slowly, the baseline still applies notch-safe restoration.
"""

import numpy as np
import torch
import torch.nn as nn

SR = 48000
N_FFT = 1024
HOP = 480
N_FREQ = N_FFT // 2 + 1

N_SPECTRAL = 4
N_COND = 7
N_IN = N_SPECTRAL + N_COND

F0_MIN_HZ = 50.0
F0_MAX_HZ = 2000.0
MAX_NOTCH_DB = 48.0
MAX_COMP_DB = 8.0

BASE_GAIN_SCALE = 0.10
RESIDUAL_GAIN_SCALE = 1.00

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


def repair_region_from_mask(notch_mask_db: torch.Tensor, kernel_size: int = 101) -> torch.Tensor:
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


def make_aperiodic_residual(log_mag: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    pad = kernel_size // 2
    smooth = torch.nn.functional.avg_pool1d(
        log_mag.permute(0, 2, 1).reshape(-1, 1, log_mag.shape[1]),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    smooth = smooth.reshape(log_mag.shape[0], log_mag.shape[2], log_mag.shape[1]).permute(0, 2, 1)
    return log_mag - smooth


def compute_base_gain(notch_mask_db: torch.Tensor) -> torch.Tensor:
    safe_bins = (notch_mask_db > -3.0).float()
    shoulder = repair_region_from_mask(notch_mask_db) * safe_bins
    shoulder_peak = shoulder.amax(dim=1, keepdim=True).clamp(min=1e-6)
    shoulder_norm = (shoulder / shoulder_peak).clamp(0.0, 1.0)
    shoulder_gate = (shoulder > 0.02).float()
    return BASE_GAIN_SCALE * shoulder_norm * shoulder_gate


def compute_effective_gain(raw_residual: torch.Tensor,
                           notch_mask_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_gain = compute_base_gain(notch_mask_db)
    safe_bins = (notch_mask_db > -3.0).float()
    shoulder = repair_region_from_mask(notch_mask_db) * safe_bins
    shoulder_gate = (shoulder > 0.02).float()
    residual = raw_residual.clamp(0.0, 1.0)
    # Additive: model directly predicts the gain delta on top of the baseline.
    # Shoulder gate is a hard architectural constraint — no boost outside safe bins.
    effective_gain = (base_gain + residual * shoulder_gate).clamp(0.0, 1.0)
    return base_gain.clamp(0.0, 1.0), effective_gain


class VoiceRestorerV5(nn.Module):
    def __init__(self, freq_ch: int = 48, gru_hidden: int = 64):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv1d(N_IN, freq_ch, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(freq_ch, freq_ch, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(freq_ch, gru_hidden, batch_first=True)
        self.fc_harmonic = nn.Linear(gru_hidden, 1)
        self.fc_aperiodic = nn.Linear(gru_hidden, 1)

        nn.init.constant_(self.fc_harmonic.bias, 0.0)
        nn.init.constant_(self.fc_aperiodic.bias, 0.0)

    def forward(self, spectral: torch.Tensor, cond: torch.Tensor, h: torch.Tensor | None = None):
        spectral = spectral.float()
        cond = cond.float()
        bsz, _, freqs, frames = spectral.shape

        cond_b = cond.unsqueeze(2).expand(-1, -1, freqs, -1)
        x = torch.cat([spectral, cond_b], dim=1)

        x = x.permute(0, 3, 1, 2).reshape(bsz * frames, N_IN, freqs)
        x = self.freq_conv(x)
        x = x.reshape(bsz, frames, -1, freqs).permute(0, 3, 1, 2)

        x = x.reshape(bsz * freqs, frames, x.shape[-1])
        x, h_new = self.gru(x, h)

        harmonic_res = torch.sigmoid(self.fc_harmonic(x)).squeeze(-1).reshape(bsz, freqs, frames)
        aperiodic_res = torch.sigmoid(self.fc_aperiodic(x)).squeeze(-1).reshape(bsz, freqs, frames)
        voiced_gate = cond[:, 4:5, :].expand(-1, freqs, -1)
        raw_residual = voiced_gate * harmonic_res + (1.0 - voiced_gate) * aperiodic_res
        return raw_residual, h_new

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def apply_compensation(notched_mag: torch.Tensor,
                       notch_mask_db: torch.Tensor,
                       raw_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_gain, effective_gain = compute_effective_gain(raw_residual, notch_mask_db)
    boost_db = effective_gain * MAX_COMP_DB
    comp_mag = notched_mag * (10.0 ** (boost_db / 20.0))
    return comp_mag, base_gain, effective_gain
