"""
voice_restore/train_v3.py — Train VoiceRestorer V3.

V3 keeps the hard forbidden-bin constraint from V2 but changes the objective:
the model is rewarded for perceptual similarity in ERB-like bands, with
loudness weighting and temporal modulation matching, rather than literal
reconstruction of the notched bins.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure repo root is on sys.path for direct script execution.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_restore.model_v3 import (  # noqa: E402
    HOP,
    N_FFT,
    N_FREQ,
    SR,
    VoiceRestorerV3,
    apply_compensation,
    repair_region_from_mask,
)
from voice_restore.features_v3 import make_v3_inputs  # noqa: E402
from voice_restore import train as v1_train  # noqa: E402


SEQ_SECS = 2.0
SEQ_LEN = int(SEQ_SECS * SR)
BATCH_SIZE = 8
EPOCHS = 200
LR = 3e-4
GRAD_CLIP = 1.0
N_STEPS = 800
WARMUP_EPOCHS = 0

ERB_BANDS = 40
IDENTITY_W = 0.10
SMOOTH_W = 0.05
GAIN_REG_W = 0.01
ERB_W = 1.0
MOD_W = 0.30
SHOULDER_W = 0.75
MASK_AWARE_W = 1.0
MASK_FLOOR = 0.25


def hz_to_erb(f_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 0.00437 * f_hz)


def erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    return (10.0 ** (erb / 21.4) - 1.0) / 0.00437


def make_erb_fb(n_bands: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.linspace(0.0, SR / 2.0, N_FREQ, device=device)
    erb_freqs = hz_to_erb(freqs)
    edges = torch.linspace(erb_freqs[1], erb_freqs[-1], n_bands + 2, device=device)
    centers = edges[1:-1]

    fb = torch.zeros(n_bands, N_FREQ, device=device)
    for i in range(n_bands):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        up = (erb_freqs - left) / (center - left + 1e-8)
        down = (right - erb_freqs) / (right - center + 1e-8)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)

    hz_centers = erb_to_hz(centers)
    return fb, hz_centers


def make_loudness_weights(center_freqs_hz: torch.Tensor) -> torch.Tensor:
    """
    A-weighting approximates where small spectral errors are more audible.
    """
    f2 = center_freqs_hz.square().clamp(min=1.0)
    ra_num = (12194.0 ** 2) * f2.square()
    ra_den = (
        (f2 + 20.6 ** 2)
        * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    ).clamp(min=1e-8)
    a_db = 20.0 * torch.log10(ra_num / ra_den) + 2.0
    weights = 10.0 ** (a_db / 20.0)
    weights = weights / weights.mean().clamp(min=1e-8)
    return weights.clamp(min=0.1)


def erb_log_energies(mag: torch.Tensor, erb_fb: torch.Tensor) -> torch.Tensor:
    power = mag.square().clamp(min=1e-10)
    erb_power = torch.einsum("ef,bft->bet", erb_fb, power).clamp(min=1e-10)
    return torch.log(erb_power)


def make_mask_aware_weights(mask_db_t: torch.Tensor,
                            erb_fb: torch.Tensor,
                            loudness_w: torch.Tensor,
                            strength_scale: float,
                            floor: float) -> torch.Tensor:
    """
    Emphasize bands near notch shoulders, where perceptual compensation is
    possible, while keeping a small floor everywhere else.
    """
    safe_bins = (mask_db_t > -3.0).float()
    repair = repair_region_from_mask(mask_db_t) * safe_bins
    notch_strength = (-mask_db_t / 48.0).clamp(0.0, 1.0)

    norm = erb_fb.sum(dim=1, keepdim=False).view(1, -1, 1).clamp(min=1e-8)
    band_repair = torch.einsum("ef,bft->bet", erb_fb, repair) / norm
    band_strength = torch.einsum("ef,bft->bet", erb_fb, notch_strength) / norm

    emphasis = (band_repair + 0.5 * band_strength).clamp(0.0, 1.0)
    weights = loudness_w.view(1, -1, 1) * (floor + strength_scale * emphasis)
    return weights


def make_allowed_target(clean_mag: torch.Tensor,
                        notched_mag: torch.Tensor,
                        mask_db_t: torch.Tensor) -> torch.Tensor:
    """
    The model is not penalized for energy it is physically forbidden to restore.
    Forbidden bins keep their notched target value.
    """
    safe_bins = (mask_db_t > -3.0).float()
    return clean_mag * safe_bins + notched_mag * (1.0 - safe_bins)


def psychoacoustic_band_loss(comp_mag: torch.Tensor,
                             allowed_target_mag: torch.Tensor,
                             erb_fb: torch.Tensor,
                             band_weights: torch.Tensor) -> torch.Tensor:
    comp_erb = erb_log_energies(comp_mag, erb_fb)
    tgt_erb = erb_log_energies(allowed_target_mag, erb_fb)
    diff = (comp_erb - tgt_erb).square()
    return (diff * band_weights).sum() / band_weights.sum().clamp(min=1e-8)


def modulation_envelope_loss(comp_mag: torch.Tensor,
                             allowed_target_mag: torch.Tensor,
                             erb_fb: torch.Tensor,
                             band_weights: torch.Tensor,
                             kernel_size: int = 9) -> torch.Tensor:
    comp_erb = erb_log_energies(comp_mag, erb_fb)
    tgt_erb = erb_log_energies(allowed_target_mag, erb_fb)

    comp_env = F.avg_pool1d(comp_erb, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    tgt_env = F.avg_pool1d(tgt_erb, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    env_loss = ((comp_env - tgt_env).abs() * band_weights).sum() / band_weights.sum().clamp(min=1e-8)
    if comp_env.shape[-1] < 2:
        return env_loss

    comp_delta = comp_env[:, :, 1:] - comp_env[:, :, :-1]
    tgt_delta = tgt_env[:, :, 1:] - tgt_env[:, :, :-1]
    delta_w = band_weights[:, :, 1:]
    delta_loss = ((comp_delta - tgt_delta).abs() * delta_w).sum() / delta_w.sum().clamp(min=1e-8)
    return env_loss + 0.5 * delta_loss


def shoulder_spectral_loss(comp_mag: torch.Tensor,
                           clean_mag: torch.Tensor,
                           mask_db_t: torch.Tensor) -> torch.Tensor:
    """
    Give the model a direct signal in the safe shoulder region around the notch.
    This stays outside forbidden bins, but avoids ERB bands washing the target out.
    """
    safe_bins = (mask_db_t > -3.0).float()
    shoulder = repair_region_from_mask(mask_db_t) * safe_bins
    log_comp = torch.log(comp_mag + 1e-8)
    log_clean = torch.log(clean_mag + 1e-8)
    return ((log_comp - log_clean).square() * shoulder).sum() / shoulder.sum().clamp(min=1.0)


def identity_preservation_loss(comp_mag: torch.Tensor,
                               notched_mag: torch.Tensor,
                               mask_db_t: torch.Tensor) -> torch.Tensor:
    repair = repair_region_from_mask(mask_db_t)
    preserve = (mask_db_t > -3.0).float() * (1.0 - repair)
    log_comp = torch.log(comp_mag + 1e-8)
    log_notched = torch.log(notched_mag + 1e-8)
    denom = preserve.sum().clamp(min=1.0)
    return ((log_comp - log_notched).square() * preserve).sum() / denom


def temporal_smoothness_loss(gain: torch.Tensor,
                             mask_db_t: torch.Tensor) -> torch.Tensor:
    if gain.shape[-1] < 2:
        return gain.new_zeros(())
    repair = repair_region_from_mask(mask_db_t)
    delta = gain[:, :, 1:] - gain[:, :, :-1]
    weight = 0.25 + 0.75 * repair[:, :, 1:]
    return (delta.square() * weight).mean()


def make_training_pair_v3(vocal_path: Path,
                          noise_np: np.ndarray,
                          device: torch.device,
                          window: torch.Tensor,
                          f0_cache: dict) -> tuple[torch.Tensor, ...] | None:
    """
    Returns:
      spectral, cond, mask_db_t, clean_mag, notched_mag
    """
    audio_np, sr = sf.read(str(vocal_path), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(1)
    if sr != SR or len(audio_np) < SEQ_LEN:
        return None

    off = random.randint(0, len(audio_np) - SEQ_LEN)
    vocal_np = audio_np[off:off + SEQ_LEN]
    vocal_np = v1_train.sosfilt(v1_train._hpf_sos, vocal_np).astype(np.float32)

    snr_db = random.uniform(10, 40)
    v_rms = float(np.sqrt(np.mean(vocal_np ** 2))) + 1e-8
    n_rms = float(np.sqrt(np.mean(noise_np[:SEQ_LEN] ** 2))) + 1e-8
    noisy = (
        vocal_np
        + noise_np[:SEQ_LEN] * (v_rms / n_rms * 10 ** (-snr_db / 20.0))
    ).astype(np.float32)

    def _stft(x_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(x_np).unsqueeze(0).to(device)
        return torch.stft(wav, N_FFT, HOP, N_FFT, window, return_complex=True)

    clean_stft = _stft(noisy)
    clean_mag = clean_stft.abs()[0]
    T = int(clean_mag.shape[1])

    notches = v1_train.simulate_notch_bank(random.randint(1, v1_train.MAX_NOTCHES_SIM))
    notched_np, mask_db = v1_train.apply_notch_bank_to_audio(noisy, notches, T)
    notched_stft = _stft(notched_np)
    notched_mag = notched_stft.abs()[0]

    key = str(vocal_path)
    if key not in f0_cache:
        f0_cache[key] = v1_train.extract_f0(vocal_path, device=str(device))
    f0_full, conf_full = f0_cache[key]
    frame_off = off // HOP
    f0_slice = f0_full[frame_off:frame_off + T + 10]
    conf_slice = conf_full[frame_off:frame_off + T + 10]

    mask_db_t = torch.from_numpy(mask_db).to(device).unsqueeze(0)
    spectral, cond = make_v3_inputs(
        notched_mag.unsqueeze(0),
        mask_db_t,
        f0_slice,
        conf_slice,
    )

    return (
        spectral,
        cond,
        mask_db_t,
        clean_mag.unsqueeze(0),
        notched_mag.unsqueeze(0),
    )


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    ap.add_argument("--erb-bands", type=int, default=ERB_BANDS)
    ap.add_argument("--erb-w", type=float, default=ERB_W)
    ap.add_argument("--mod-w", type=float, default=MOD_W)
    ap.add_argument("--shoulder-w", type=float, default=SHOULDER_W)
    ap.add_argument("--identity-w", type=float, default=IDENTITY_W)
    ap.add_argument("--smooth-w", type=float, default=SMOOTH_W)
    ap.add_argument("--gain-reg-w", type=float, default=GAIN_REG_W)
    ap.add_argument("--mask-aware-w", type=float, default=MASK_AWARE_W)
    ap.add_argument("--mask-floor", type=float, default=MASK_FLOOR)
    args, _ = ap.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = torch.hann_window(N_FFT).sqrt().to(device)
    model = VoiceRestorerV3().to(device)
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "voice_restore_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(ckpt_dir / "tb"))

    base_lr = args.lr or LR
    optimizer = Adam(model.parameters(), lr=base_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    vocal_files = [
        f for f in (PROJECT_ROOT / "data" / "clean_vocals").rglob("*.wav")
        if not f.name.startswith("._") and "__MACOSX" not in str(f)
    ]
    noise_files = list((PROJECT_ROOT / "data" / "noise").rglob("*.wav"))
    assert vocal_files, "No vocal files in data/clean_vocals/"
    assert noise_files, "No noise files in data/noise/"

    vocal_files = [
        f for f in vocal_files
        if (info := sf.info(str(f))).frames / info.samplerate >= SEQ_SECS
    ]
    assert vocal_files, f"No vocal files >= {SEQ_SECS}s"

    erb_fb, center_freqs = make_erb_fb(args.erb_bands, device)
    loudness_w = make_loudness_weights(center_freqs)

    print(f"VoiceRestorerV3: {model.n_params:,} params on {device}")
    print(f"Vocals: {len(vocal_files)}, noise: {len(noise_files)}")
    print(
        f"LR: {base_lr:.2e} | epochs: {args.epochs} | warmup: {args.warmup_epochs} | "
        f"ERB bands: {args.erb_bands}"
    )
    if not v1_train.CREPE_AVAILABLE:
        print("NOTE: CREPE unavailable — training without pitch features.")

    best_loss = float("inf")
    f0_cache: dict = {}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt if isinstance(ckpt, dict) and "model" in ckpt else {"model": ckpt}
        model.load_state_dict(state["model"])
        best_loss = state.get("best_loss", float("inf"))
        print(f"Resumed from {args.resume}  (best_loss={best_loss:.4f})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0
        last_erb_loss = torch.tensor(0.0, device=device)
        last_mod_loss = torch.tensor(0.0, device=device)
        last_shoulder_loss = torch.tensor(0.0, device=device)
        last_id_loss = torch.tensor(0.0, device=device)
        last_smooth_loss = torch.tensor(0.0, device=device)
        last_gain_reg = torch.tensor(0.0, device=device)
        last_gain = torch.zeros(1, 1, 1, device=device)
        last_band_weight_mean = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_lr = base_lr * (epoch / args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        elif epoch == args.warmup_epochs + 1:
            for group in optimizer.param_groups:
                group["lr"] = base_lr

        for _ in tqdm(range(N_STEPS), desc=f"Epoch {epoch}/{args.epochs}"):
            noise_np, _ = sf.read(str(random.choice(noise_files)), dtype="float32")
            if noise_np.ndim > 1:
                noise_np = noise_np.mean(1)
            if len(noise_np) < SEQ_LEN:
                noise_np = np.tile(noise_np, math.ceil(SEQ_LEN / len(noise_np)))
            start = random.randint(0, len(noise_np) - SEQ_LEN)
            noise_np = noise_np[start:start + SEQ_LEN]

            result = make_training_pair_v3(
                random.choice(vocal_files),
                noise_np,
                device,
                window,
                f0_cache,
            )
            if result is None:
                continue

            spectral, cond, mask_db_t, clean_mag, notched_mag = result

            gain, _ = model(spectral, cond)
            gain = torch.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
            comp_mag = apply_compensation(notched_mag, mask_db_t, gain)
            comp_mag = torch.nan_to_num(comp_mag, nan=0.0, posinf=0.0, neginf=0.0)

            allowed_target_mag = make_allowed_target(clean_mag, notched_mag, mask_db_t)
            band_weights = make_mask_aware_weights(
                mask_db_t,
                erb_fb,
                loudness_w,
                strength_scale=args.mask_aware_w,
                floor=args.mask_floor,
            )
            erb_loss = psychoacoustic_band_loss(comp_mag, allowed_target_mag, erb_fb, band_weights)
            mod_loss = modulation_envelope_loss(comp_mag, allowed_target_mag, erb_fb, band_weights)
            shoulder_loss = shoulder_spectral_loss(comp_mag, clean_mag, mask_db_t)
            id_loss = identity_preservation_loss(comp_mag, notched_mag, mask_db_t)
            smooth_loss = temporal_smoothness_loss(gain, mask_db_t)
            gain_reg = (gain.square() * repair_region_from_mask(mask_db_t)).mean()

            loss = (
                args.erb_w * erb_loss
                + args.mod_w * mod_loss
                + args.shoulder_w * shoulder_loss
                + args.identity_w * id_loss
                + args.smooth_w * smooth_loss
                + args.gain_reg_w * gain_reg
            )

            if not torch.isfinite(loss):
                continue

            (loss / BATCH_SIZE).backward()
            valid_steps += 1
            last_erb_loss = erb_loss.detach()
            last_mod_loss = mod_loss.detach()
            last_shoulder_loss = shoulder_loss.detach()
            last_id_loss = id_loss.detach()
            last_smooth_loss = smooth_loss.detach()
            last_gain_reg = gain_reg.detach()
            last_gain = gain.detach()
            last_band_weight_mean = band_weights.mean().detach()

            if valid_steps % BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += float(loss.item())

        if valid_steps % BATCH_SIZE != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(valid_steps // BATCH_SIZE, 1)
        if epoch > args.warmup_epochs:
            scheduler.step(avg_loss)

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("loss/erb", float(last_erb_loss.item()), epoch)
        writer.add_scalar("loss/modulation", float(last_mod_loss.item()), epoch)
        writer.add_scalar("loss/shoulder", float(last_shoulder_loss.item()), epoch)
        writer.add_scalar("loss/identity", float(last_id_loss.item()), epoch)
        writer.add_scalar("loss/smooth", float(last_smooth_loss.item()), epoch)
        writer.add_scalar("loss/gain_reg", float(last_gain_reg.item()), epoch)
        writer.add_scalar("loss/mask_weight_mean", float(last_band_weight_mean.item()), epoch)
        writer.add_scalar("gain/min", float(last_gain.min().item()), epoch)
        writer.add_scalar("gain/max", float(last_gain.max().item()), epoch)
        writer.add_scalar("gain/mean", float(last_gain.mean().item()), epoch)
        writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), epoch)

        best_str = f"{best_loss:.4f}" if best_loss < float("inf") else "none"
        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | best {best_str} | "
            f"valid {valid_steps}/{N_STEPS} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_loss": best_loss,
                },
                str(ckpt_dir / "best.pt"),
            )
            print("  ✓ New best")


if __name__ == "__main__":
    train()
