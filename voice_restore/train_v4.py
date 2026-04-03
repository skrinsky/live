"""
voice_restore/train_v4.py — Train VoiceRestorer V4.

V4 uses a direct envelope-compensation target: the model predicts safe shoulder
boosts that match a smoothed clean-vs-notched spectral envelope difference,
while the hard forbidden-bin mask remains intact.
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_restore.model_v4 import (  # noqa: E402
    HOP,
    MAX_COMP_DB,
    N_FFT,
    SR,
    VoiceRestorerV4,
    apply_compensation,
    repair_region_from_mask,
)
from voice_restore.features_v4 import make_v4_inputs  # noqa: E402
from voice_restore import train as v1_train  # noqa: E402

SEQ_SECS = 2.0
SEQ_LEN = int(SEQ_SECS * SR)
BATCH_SIZE = 8
EPOCHS = 200
LR = 3e-4
GRAD_CLIP = 1.0
N_STEPS = 800
WARMUP_EPOCHS = 0

ENV_KERNEL = 33
ENV_TARGET_W = 1.0
ENV_MATCH_W = 0.75
IDENTITY_W = 0.05
SMOOTH_W = 0.05
GAIN_REG_W = 0.0
TARGET_ONLY_EPOCHS = 3
TARGET_SCALE = 2.0
TARGET_FLOOR = 0.04


def smooth_log_spectrum(mag: torch.Tensor, kernel_size: int = ENV_KERNEL) -> torch.Tensor:
    """
    Smooth along frequency (not time) to get a coarse spectral envelope.
    Input/Output shape: (B, F, T)
    """
    log_mag = torch.log(mag + 1e-8)
    pad = kernel_size // 2
    bsz, n_freq, n_frames = log_mag.shape
    x = log_mag.permute(0, 2, 1).reshape(bsz * n_frames, 1, n_freq)
    x = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    return x.reshape(bsz, n_frames, n_freq).permute(0, 2, 1)


def target_gain_from_envelope(clean_mag: torch.Tensor,
                              notched_mag: torch.Tensor,
                              mask_db_t: torch.Tensor,
                              kernel_size: int = ENV_KERNEL,
                              target_scale: float = TARGET_SCALE,
                              target_floor: float = TARGET_FLOOR) -> torch.Tensor:
    """
    Estimate how much shoulder compensation is perceptually useful by comparing
    smoothed spectral envelopes, then projecting that boost onto safe repair
    regions only.
    """
    safe_bins = (mask_db_t > -3.0).float()
    shoulder = repair_region_from_mask(mask_db_t) * safe_bins

    clean_env = smooth_log_spectrum(clean_mag, kernel_size=kernel_size)
    notched_env = smooth_log_spectrum(notched_mag, kernel_size=kernel_size)

    env_boost_db = (clean_env - notched_env) * (20.0 / math.log(10.0))
    env_boost_db = env_boost_db.clamp(min=0.0, max=MAX_COMP_DB)

    # Normalize by the allowed max compensation and restrict to the safe shoulder.
    target_gain = (env_boost_db / MAX_COMP_DB) * shoulder
    target_gain = (target_gain * target_scale).clamp(0.0, 1.0)

    # Keep a small nonzero prior in active shoulder regions so the model
    # does not collapse to all-zero gains when envelope deltas are tiny.
    notch_strength = (-mask_db_t / 48.0).clamp(0.0, 1.0)
    shoulder_active = (shoulder > 0.05).float()
    prior_gain = target_floor * shoulder_active * notch_strength
    target_gain = torch.maximum(target_gain, prior_gain)
    return target_gain.clamp(0.0, 1.0)


def gain_target_loss(gain: torch.Tensor,
                     target_gain: torch.Tensor,
                     mask_db_t: torch.Tensor) -> torch.Tensor:
    shoulder = repair_region_from_mask(mask_db_t) * (mask_db_t > -3.0).float()
    active = ((target_gain > 0.02).float() * shoulder).clamp(0.0, 1.0)
    if float(active.sum()) < 1.0:
        active = (shoulder > 0.05).float()
    # Only supervise where compensation is actually expected; otherwise the
    # dominant zero-target region collapses the whole gain map to zero.
    return ((gain - target_gain).square() * active).sum() / active.sum().clamp(min=1.0)


def envelope_match_loss(comp_mag: torch.Tensor,
                        clean_mag: torch.Tensor,
                        mask_db_t: torch.Tensor,
                        kernel_size: int = ENV_KERNEL) -> torch.Tensor:
    safe_bins = (mask_db_t > -3.0).float()
    shoulder = repair_region_from_mask(mask_db_t) * safe_bins
    comp_env = smooth_log_spectrum(comp_mag, kernel_size=kernel_size)
    clean_env = smooth_log_spectrum(clean_mag, kernel_size=kernel_size)
    return ((comp_env - clean_env).square() * shoulder).sum() / shoulder.sum().clamp(min=1.0)


def identity_preservation_loss(comp_mag: torch.Tensor,
                               notched_mag: torch.Tensor,
                               mask_db_t: torch.Tensor) -> torch.Tensor:
    repair = repair_region_from_mask(mask_db_t)
    preserve = (mask_db_t > -3.0).float() * (1.0 - repair)
    log_comp = torch.log(comp_mag + 1e-8)
    log_notched = torch.log(notched_mag + 1e-8)
    return ((log_comp - log_notched).square() * preserve).sum() / preserve.sum().clamp(min=1.0)


def temporal_smoothness_loss(gain: torch.Tensor,
                             mask_db_t: torch.Tensor) -> torch.Tensor:
    if gain.shape[-1] < 2:
        return gain.new_zeros(())
    repair = repair_region_from_mask(mask_db_t)
    delta = gain[:, :, 1:] - gain[:, :, :-1]
    weight = 0.25 + 0.75 * repair[:, :, 1:]
    return (delta.square() * weight).mean()


def make_training_pair_v4(vocal_path: Path,
                          noise_np: np.ndarray,
                          device: torch.device,
                          window: torch.Tensor,
                          f0_cache: dict) -> tuple[torch.Tensor, ...] | None:
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
    t_frames = int(clean_mag.shape[1])

    notches = v1_train.simulate_notch_bank(random.randint(1, v1_train.MAX_NOTCHES_SIM))
    notched_np, mask_db = v1_train.apply_notch_bank_to_audio(noisy, notches, t_frames)
    notched_stft = _stft(notched_np)
    notched_mag = notched_stft.abs()[0]

    key = str(vocal_path)
    if key not in f0_cache:
        f0_cache[key] = v1_train.extract_f0(vocal_path, device=str(device))
    f0_full, conf_full = f0_cache[key]
    frame_off = off // HOP
    f0_slice = f0_full[frame_off:frame_off + t_frames + 10]
    conf_slice = conf_full[frame_off:frame_off + t_frames + 10]

    mask_db_t = torch.from_numpy(mask_db).to(device).unsqueeze(0)
    spectral, cond = make_v4_inputs(
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
    ap.add_argument("--env-kernel", type=int, default=ENV_KERNEL)
    ap.add_argument("--env-target-w", type=float, default=ENV_TARGET_W)
    ap.add_argument("--env-match-w", type=float, default=ENV_MATCH_W)
    ap.add_argument("--identity-w", type=float, default=IDENTITY_W)
    ap.add_argument("--smooth-w", type=float, default=SMOOTH_W)
    ap.add_argument("--gain-reg-w", type=float, default=GAIN_REG_W)
    ap.add_argument("--target-only-epochs", type=int, default=TARGET_ONLY_EPOCHS)
    ap.add_argument("--target-scale", type=float, default=TARGET_SCALE)
    ap.add_argument("--target-floor", type=float, default=TARGET_FLOOR)
    args, _ = ap.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = torch.hann_window(N_FFT).sqrt().to(device)
    model = VoiceRestorerV4().to(device)
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "voice_restore_v4"
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

    print(f"VoiceRestorerV4: {model.n_params:,} params on {device}")
    print(f"Vocals: {len(vocal_files)}, noise: {len(noise_files)}")
    print(
        f"LR: {base_lr:.2e} | epochs: {args.epochs} | warmup: {args.warmup_epochs} | "
        f"env_kernel: {args.env_kernel}"
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
        last_env_target_loss = torch.tensor(0.0, device=device)
        last_env_match_loss = torch.tensor(0.0, device=device)
        last_identity_loss = torch.tensor(0.0, device=device)
        last_smooth_loss = torch.tensor(0.0, device=device)
        last_gain_reg = torch.tensor(0.0, device=device)
        last_gain = torch.zeros(1, 1, 1, device=device)
        last_target_gain = torch.zeros(1, 1, 1, device=device)
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

            result = make_training_pair_v4(
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

            target_gain = target_gain_from_envelope(
                clean_mag,
                notched_mag,
                mask_db_t,
                kernel_size=args.env_kernel,
                target_scale=args.target_scale,
                target_floor=args.target_floor,
            )
            env_target_loss = gain_target_loss(gain, target_gain, mask_db_t)
            env_match_loss = envelope_match_loss(
                comp_mag,
                clean_mag,
                mask_db_t,
                kernel_size=args.env_kernel,
            )
            identity_loss = identity_preservation_loss(comp_mag, notched_mag, mask_db_t)
            smooth_loss = temporal_smoothness_loss(gain, mask_db_t)
            gain_reg = (gain.square() * repair_region_from_mask(mask_db_t)).mean()

            if epoch <= args.target_only_epochs:
                loss = (
                    args.env_target_w * env_target_loss
                    + 0.5 * args.smooth_w * smooth_loss
                )
            else:
                loss = (
                    args.env_target_w * env_target_loss
                    + args.env_match_w * env_match_loss
                    + args.identity_w * identity_loss
                    + args.smooth_w * smooth_loss
                    + args.gain_reg_w * gain_reg
                )

            if not torch.isfinite(loss):
                continue

            (loss / BATCH_SIZE).backward()
            valid_steps += 1
            last_env_target_loss = env_target_loss.detach()
            last_env_match_loss = env_match_loss.detach()
            last_identity_loss = identity_loss.detach()
            last_smooth_loss = smooth_loss.detach()
            last_gain_reg = gain_reg.detach()
            last_gain = gain.detach()
            last_target_gain = target_gain.detach()

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
        writer.add_scalar("loss/env_target", float(last_env_target_loss.item()), epoch)
        writer.add_scalar("loss/env_match", float(last_env_match_loss.item()), epoch)
        writer.add_scalar("loss/identity", float(last_identity_loss.item()), epoch)
        writer.add_scalar("loss/smooth", float(last_smooth_loss.item()), epoch)
        writer.add_scalar("loss/gain_reg", float(last_gain_reg.item()), epoch)
        writer.add_scalar("gain/min", float(last_gain.min().item()), epoch)
        writer.add_scalar("gain/max", float(last_gain.max().item()), epoch)
        writer.add_scalar("gain/mean", float(last_gain.mean().item()), epoch)
        writer.add_scalar("target_gain/min", float(last_target_gain.min().item()), epoch)
        writer.add_scalar("target_gain/max", float(last_target_gain.max().item()), epoch)
        writer.add_scalar("target_gain/mean", float(last_target_gain.mean().item()), epoch)
        writer.add_scalar("target_gain/active_pct", float((last_target_gain > 0.02).float().mean().item()), epoch)
        writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), epoch)

        best_str = f"{best_loss:.4f}" if best_loss < float("inf") else "none"
        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | best {best_str} | "
            f"valid {valid_steps}/{N_STEPS} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_loss": best_loss},
                str(ckpt_dir / "best.pt"),
            )
            print("  ✓ New best")


if __name__ == "__main__":
    train()
