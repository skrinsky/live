"""
voice_restore/features_v3.py — Shared feature builders for VoiceRestorer V3.

V3 keeps the same input representation as V2 so we can isolate whether the
psychoacoustic loss is the thing that actually improves restoration.
"""

import numpy as np
import torch

from voice_restore.model_v3 import (
    N_FREQ,
    harmonic_template,
    normalise_f0,
    notch_strength_from_mask,
    make_aperiodic_residual,
)


def _fit_length(arr: np.ndarray, n_frames: int) -> np.ndarray:
    if len(arr) >= n_frames:
        return arr[:n_frames]
    return np.pad(arr, (0, n_frames - len(arr)))


def build_condition_features(
    f0_np: np.ndarray,
    conf_np: np.ndarray,
    notched_mag: torch.Tensor,
) -> torch.Tensor:
    """
    Build (7, T) conditioning features from pitch tracks + notched magnitude.
    """
    T = int(notched_mag.shape[1])
    f0_fit = _fit_length(f0_np, T)
    conf_fit = _fit_length(conf_np, T).astype(np.float32)

    f0_norm = np.array([normalise_f0(float(f)) for f in f0_fit], dtype=np.float32)
    delta = np.gradient(f0_norm).astype(np.float32)
    delta2 = np.gradient(delta).astype(np.float32)
    voiced = ((f0_norm > 0.0).astype(np.float32) * conf_fit).astype(np.float32)

    log_mag = torch.log(notched_mag + 1e-8)
    flux = torch.zeros(T, dtype=notched_mag.dtype, device=notched_mag.device)
    if T > 1:
        flux[1:] = torch.mean(torch.abs(log_mag[:, 1:] - log_mag[:, :-1]), dim=0)
    flux = torch.tanh(flux).cpu().numpy().astype(np.float32)

    geom = torch.exp(torch.mean(torch.log(notched_mag + 1e-8), dim=0))
    arith = torch.mean(notched_mag + 1e-8, dim=0)
    flatness = (geom / arith).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    return torch.from_numpy(
        np.stack([f0_norm, conf_fit, delta, delta2, voiced, flux, flatness], axis=0)
    )


def build_harmonic_features(f0_np: np.ndarray, n_frames: int) -> np.ndarray:
    f0_fit = _fit_length(f0_np, n_frames)
    return np.stack([harmonic_template(float(f)) for f in f0_fit], axis=1).astype(np.float32)


def make_v3_inputs(
    notched_mag: torch.Tensor,
    notch_mask_db: torch.Tensor,
    f0_np: np.ndarray,
    conf_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble VoiceRestorer V3 spectral and conditioning inputs.

    notched_mag   : (1, F, T)
    notch_mask_db : (1, F, T)
    Returns:
      spectral : (1, 4, F, T)
      cond     : (1, 7, T)
    """
    log_notched = torch.log(notched_mag + 1e-8)
    T = int(notched_mag.shape[-1])
    harm_np = build_harmonic_features(f0_np, T)
    harm_t = torch.from_numpy(harm_np).to(notched_mag.device).unsqueeze(0)
    notch_strength = notch_strength_from_mask(notch_mask_db)[..., :T]

    T_final = min(
        log_notched.shape[-1],
        harm_t.shape[-1],
        notch_strength.shape[-1],
        notch_mask_db.shape[-1],
    )

    log_notched = log_notched[..., :T_final]
    harm_t = harm_t[..., :T_final]
    notch_strength = notch_strength[..., :T_final]
    notch_mask_db = notch_mask_db[..., :T_final]

    aperiodic = make_aperiodic_residual(log_notched)

    spectral = torch.stack(
        [
            log_notched[:, :N_FREQ, :],
            harm_t[:, :N_FREQ, :],
            notch_strength[:, :N_FREQ, :],
            aperiodic[:, :N_FREQ, :],
        ],
        dim=1,
    )

    cond = build_condition_features(f0_np, conf_np, log_notched[0]).to(notched_mag.device).unsqueeze(0)
    return spectral, cond
