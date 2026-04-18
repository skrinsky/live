"""
feedback_mask/train.py — Single-channel feedback suppression training.

No recursive BPTT. No reference channel. Just:
    mic (speech + feedback + noise) → clean speech

Why this is more convex than FDKFNet:
  - No Kalman filter equations in the gradient path (GRU → mask directly)
  - No self-referential contamination (model output not in training loop)
  - HybridLoss gives STFT-domain gradient signal at every bin, every frame —
    much denser than sequence-level SI-SDR alone
  - Recursive IIR feedback simulation (lfilter closes the PA→room→mic loop)

Approach mirrors De-Feedback (Alpha Labs): learn to separate vocal from
feedback/reverb/noise as a spectral separation problem, not echo cancellation.

Usage:
    python feedback_mask/train.py
    python feedback_mask/train.py --resume checkpoints/gtcrn_feedback/best.pt
"""

import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve, butter, sosfilt, lfilter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model import FeedbackMaskNet, SR, N_FFT, HOP, N_FREQ
from mic_profiles import apply_random_mic_response, MIC_NAMES

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_SECS     = 10.0      # seconds per training clip — long enough to distinguish
                         # sustained feedback ring (~6s decay constant at gain=0.85)
                         # from voiced vowels (200-500ms). 4s was too short.
SEQ_LEN      = int(SEQ_SECS * SR)
BATCH_SIZE   = 16        # gradient accumulation steps before optimizer.step()
EPOCHS       = 300
LR           = 1e-4
GRAD_CLIP    = 1.0
MAX_IR_LEN         = int(1.5 * SR)
FEEDBACK_TRUNC     = int(0.05 * SR)   # 50ms — captures resonant modes, keeps IIR order low
N_STEPS            = 800              # 800 steps × 16 batch = 50 optimizer steps/epoch
                                      # Fewer steps than before but each clip is 10s so
                                      # total audio/epoch is similar (~8000s)


def make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def sample_gain(epoch):
    """
    Gain distribution — HARD CAP at 0.99.

    Gain > 1.0 causes the IIR feedback loop to be unstable: the signal grows
    exponentially and clips to ±1 within the first second of a 4s clip.
    Clipping is irreversible — the model cannot recover the vocal from a
    saturated signal, so those examples contribute no useful gradient.
    They actively poison learning by dragging the gradient toward silence.

    0.99 is the hardest tractable case: the ring barely decays (takes ~70
    reflections to halve), representing the real-world scenario just before
    feedback takes off. This is exactly what the model needs to learn to suppress.

    Two phases differ only in how much near-threshold exposure:
      Phase 1 (epochs 1-29):  mostly sub-threshold, model learns what rings look like
      Phase 2 (epochs 30+):   heavier near-threshold, model learns to suppress them
    """
    def _one():
        t = random.random()
        if epoch < 30:
            if t < 0.10:   return 0.0                         # path off (10%)
            elif t < 0.30: return random.uniform(0.5, 0.6)    # clearly sub-threshold (20%)
            elif t < 0.70: return random.uniform(0.6, 0.85)   # moderate ring (40%)
            else:          return random.uniform(0.85, 0.99)  # near-threshold (30%)
        else:
            if t < 0.10:   return 0.0                         # path off (10%)
            elif t < 0.20: return random.uniform(0.5, 0.6)    # clearly sub-threshold (10%)
            elif t < 0.50: return random.uniform(0.6, 0.85)   # moderate ring (30%)
            else:          return random.uniform(0.85, 0.99)  # near-threshold (50%)
    return _one(), _one()


# ── Oracle supervision helpers ──────────────────────────────────────────────────

def _stability_check(h, max_gain=0.99):
    """Re-normalise h so spectral peak stays ≤ max_gain after resonators are added."""
    peak = np.abs(np.fft.rfft(h, n=max(len(h) * 4, 4096))).max()
    if peak > max_gain:
        h = h * (max_gain / (peak + 1e-8))
    return h


# ── Per-step training function ─────────────────────────────────────────────────

def _norm_ir(ir):
    """Normalise IR to unit spectral peak so gain=1.0 is exactly the stability threshold."""
    peak = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    return ir / (peak + 1e-8)


def _load_room_dimensions():
    """Load room_dimensions.json written by download_public_irs.py, or return {}."""
    meta = PROJECT_ROOT / 'data' / 'public_irs' / 'room_dimensions.json'
    if meta.exists():
        import json
        with open(meta) as f:
            return json.load(f)
    return {}

_ROOM_DIMS = _load_room_dimensions()


def _modal_freqs_for_ir(ir_path):
    """
    Return axial mode frequencies for the room this IR was measured in,
    using the room_dimensions.json lookup. Falls back to None if unknown.
    """
    if not _ROOM_DIMS:
        return None
    path_str = str(ir_path).lower()
    for key, dims in _ROOM_DIMS.items():
        if key.split('_')[0] in path_str:   # e.g. 'c4dm', 'arni', 'aachen'
            Lx, Ly, Lz = dims
            c = 343.0
            modes = []
            for nx in range(4):
                for ny in range(4):
                    for nz in range(4):
                        if nx == ny == nz == 0:
                            continue
                        f = c / 2 * np.sqrt((nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2)
                        if f <= 4000:
                            modes.append(f)
            return sorted(modes) if modes else None
    return None


def _add_resonators(h, sr, ir_path=None, n=None):
    """
    Add narrow-band room-mode resonators to feedback IR h.
    If ir_path is from a known room (room_dimensions.json), resonators are
    placed at the actual axial modal frequencies of that room.
    Otherwise, frequencies are random in the 100–4000 Hz feedback range.

    Returns (h_modified, ring_freqs) so callers can build ground-truth ring labels.
    """
    n = n if n is not None else random.randint(0, 3)
    if n == 0:
        return h, []
    modal_freqs = _modal_freqs_for_ir(ir_path) if ir_path else None
    t = np.arange(len(h)) / sr
    ring_freqs = []
    for _ in range(n):
        if modal_freqs:
            freq = random.choice(modal_freqs)
        else:
            freq = random.uniform(100, 4000)
        ring_freqs.append(freq)
        Q         = random.uniform(15, 80)
        gain      = random.uniform(0.05, 0.3)
        decay     = np.pi * freq / Q
        resonance = gain * np.exp(-decay * t) * np.cos(2 * np.pi * freq * t)
        h = h + resonance.astype(h.dtype)
    return h, ring_freqs


def train_one_step(model, vocal_np, mains_ir_np, monitor_ir_np,
                   room_ir_np, noise_np, device, window, room_ir_path=None, epoch=1):
    """
    Recursive feedback simulation → STFT → model → ratio-mask L1 loss.

    mic[n] = speech[n] + gain × Σ h[k] × mic[n-k]   (IIR via lfilter)

    Ideal mask = tgt_mag / (mic_mag + ε), clamped to [0,1].
    Directly observes which bins have feedback energy without any IR oracle.
      Ring bins:  mic_mag >> tgt_mag  →  ideal ≈ 0  (suppress)
      Clean bins: mic_mag ≈ tgt_mag   →  ideal ≈ 1  (pass through)

    This is how FeedbackDetector works (mic/target ratio), extended to a
    continuous mask instead of a binary label.  No class-balance issues,
    no IR-spectrum ambiguity, no threshold tuning.
    Returns scalar loss tensor with graph attached.
    """
    # Room reverb → target (what the model should output)
    target_np = fftconvolve(vocal_np, room_ir_np)[:SEQ_LEN].astype(np.float32)

    # Randomise HPF cutoff per sequence (70–120 Hz covers typical console variation)
    hpf       = make_hpf(np.random.uniform(70, 120))
    target_np = sosfilt(hpf, target_np).astype(np.float32)

    # Noise — 50% convolutive
    noise_np = noise_np[:SEQ_LEN]
    if random.random() < 0.5:
        noise_np = fftconvolve(noise_np, room_ir_np)[:SEQ_LEN].astype(np.float32)
    vocal_rms   = float(np.sqrt(np.mean(target_np**2))) + 1e-8
    noise_rms   = float(np.sqrt(np.mean(noise_np**2)))  + 1e-8
    snr_db      = np.random.uniform(5, 40)
    noise_scale = vocal_rms / noise_rms * 10**(-snr_db / 20)
    noisy_clean = (target_np + noise_np * noise_scale).astype(np.float64)
    # Target = vocal + noise (not clean vocal). The feedback suppressor's job is
    # to remove ONLY the feedback ring; noise and reverb are "wanted" and should
    # pass through. Without this, the model simultaneously learns denoising and
    # feedback suppression. Those objectives conflict: denoising wants mask < 1
    # at every bin (noise is present everywhere), while feedback suppression wants
    # mask ≈ 1 at non-ring bins. The model compromises at a globally low mask
    # (~0.17) that satisfies neither task. Adding noise to target makes the ideal
    # mask = 1 at all non-ring bins and 0 only at the ring bin.
    target_np = noisy_clean.astype(np.float32)

    # ── Pure-feedback case (10%): no vocal, mic is open between songs ──────────
    # Model must learn to output silence when only feedback is present.
    if random.random() < 0.10:
        noisy_clean = (noise_np[:SEQ_LEN] * 0.05).astype(np.float64)   # near-silence input
        target_np   = np.zeros(SEQ_LEN, dtype=np.float32)               # target: silence

    # ── Recursive feedback — IIR filter closes the PA→room→mic loop ─────────
    # Truncate to FEEDBACK_TRUNC samples (50ms): captures the resonant modes
    # that drive instability; longer tail is already covered by room_ir reverb.
    trunc      = min(len(mains_ir_np), len(monitor_ir_np), FEEDBACK_TRUNC)
    mains_norm = _norm_ir(mains_ir_np[:trunc])
    mon_norm   = _norm_ir(monitor_ir_np[:trunc])
    # Renormalise combined IR: mains and monitor may have aligned resonances,
    # making the sum exceed unit spectral peak even at gain=1.0 — causing
    # unexpected instability and NaN losses before the gain is applied.
    _combined_check = mains_norm + mon_norm
    _combined_peak  = np.abs(np.fft.rfft(_combined_check, n=max(len(_combined_check)*4, 4096))).max()
    if _combined_peak > 1.0:
        mains_norm = mains_norm / (_combined_peak + 1e-8)
        mon_norm   = mon_norm   / (_combined_peak + 1e-8)

    # 50% of clips ramp from stable → howling to simulate feedback building up
    # (vocalist walks toward monitor, engineer nudges fader, mic cups, etc.)
    is_ramp = random.random() < 0.50
    if is_ramp:
        gain_lo  = random.uniform(0.2, 0.7)
        gain_hi  = random.uniform(0.85, 0.99)  # never exceeds stability threshold
        split    = random.randint(int(0.2 * SEQ_LEN), int(0.5 * SEQ_LEN))
        # _stability_check: resonators can push spectral peak above gain level → overflow fix
        h_lo_raw, ring_freqs_lo = _add_resonators(mains_norm * gain_lo + mon_norm * gain_lo,
                                                  SR, ir_path=room_ir_path)
        h_hi_raw, ring_freqs_hi = _add_resonators(mains_norm * gain_hi + mon_norm * gain_hi,
                                                  SR, ir_path=room_ir_path)
        h_lo = _stability_check(h_lo_raw)
        h_hi = _stability_check(h_hi_raw)
        ring_freqs = ring_freqs_lo + ring_freqs_hi
        a_lo = np.concatenate([[1.0], -h_lo.astype(np.float64)])
        a_hi = np.concatenate([[1.0], -h_hi.astype(np.float64)])
        zi   = np.zeros(len(a_lo) - 1)
        y1, zi = lfilter([1.0], a_lo, noisy_clean[:split], zi=zi)
        y2, _  = lfilter([1.0], a_hi, noisy_clean[split:], zi=zi)
        mic_np = np.concatenate([y1, y2])
        h_combined = None  # not used in ramp case
    else:
        mains_gain, monitor_gain = sample_gain(epoch)
        h_combined_raw, ring_freqs = _add_resonators(
            mains_norm * mains_gain + mon_norm * monitor_gain,
            SR, ir_path=room_ir_path)
        h_combined = _stability_check(h_combined_raw)
        h_lo = h_hi = None  # not used in constant case
        split = 0
        if h_combined.max() == 0:
            mic_np = noisy_clean.copy()
        else:
            a      = np.concatenate([[1.0], -h_combined.astype(np.float64)])
            mic_np = lfilter([1.0], a, noisy_clean)

    mic_np = np.nan_to_num(mic_np, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    mic_np = np.clip(mic_np, -1.0, 1.0)

    # Console HPF — must match live.py and run_inference.py (90 Hz, randomised ±20 Hz)
    hpf_mic = make_hpf(np.random.uniform(70, 120))
    mic_np  = sosfilt(hpf_mic, mic_np).astype(np.float32)

    # RMS normalisation — mic is always scaled to TARGET_RMS so inference sees
    # the same amplitude distribution as training (matches live.py TARGET_RMS=0.1).
    # Target is normalised INDEPENDENTLY to its own RMS.
    #
    # Why independent normalisation for target:
    #   At near-threshold feedback (gain=0.85-0.99), mic_rms >> tgt_rms because
    #   the resonator dominates the mic signal. Scaling both by the same factor
    #   (0.1 / mic_rms) makes tgt_mag ≈ 0 everywhere — not just at ring bins.
    #   The L1 loss then pushes mask → 0 everywhere on these clips, and because
    #   their loss magnitude is large (mic at 0.1 RMS, target ≈ 0), they
    #   dominate the gradient and override the correct signal from low-gain clips.
    #   Independent normalisation ensures tgt_mag reflects the spectral shape of
    #   the clean vocal (ring bins quiet, speech bins energetic) at any feedback
    #   gain level, giving the model a consistent selective-suppression target.
    TARGET_RMS = 0.1
    mic_rms    = float(np.sqrt(np.mean(mic_np ** 2))) + 1e-8
    mic_np     = np.clip(mic_np * (TARGET_RMS / mic_rms), -1.0, 1.0).astype(np.float32)

    tgt_rms    = float(np.sqrt(np.mean(target_np ** 2))) + 1e-8
    if tgt_rms > 0.001:   # skip for pure-feedback/silence targets — stays near-zero
        target_np = np.clip(target_np * (TARGET_RMS / tgt_rms), -1.0, 1.0).astype(np.float32)
    else:
        target_np = target_np.astype(np.float32)   # silence target → L1 pushes mask → 0 (correct)

    # Mic frequency response — same profile applied to both mic and target so the
    # model sees consistent coloration and only needs to suppress the feedback
    mic_name  = random.choice(MIC_NAMES)
    mic_np    = apply_random_mic_response(mic_np,    SR, mic_name=mic_name)
    target_np = apply_random_mic_response(target_np, SR, mic_name=mic_name)

    # STFT mic and target → (1, N_FREQ, T)
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    tgt_t    = torch.from_numpy(target_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    tgt_stft = torch.stft(tgt_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)

    # Target and mic magnitudes
    tgt_mag = tgt_stft.abs()                                           # (1, N_FREQ, T)
    mic_mag = mic_stft.abs()                                           # (1, N_FREQ, T)

    # Log-magnitude L1: drives mask toward ideal ratio mask (tgt_mag / mic_mag).
    #
    # Why log and not linear:
    #   Linear L1 — non-ring gradient wins aggregate 23:1 → model learns
    #   "pass everything", ring barely suppressed (stuck at mask=0.918).
    #
    #   Log L1 — at ring bin: |log(mask×mic_ring) - log(tgt_ring)| = |log(mask) + log(20)|
    #   At convergence, ring bin loss = 0 when mask = tgt_ring/mic_ring = 1/20 = 0.05.
    #   Gradient balance ~2.5:1 non-ring wins (healthy), vs 23:1 for linear.
    #   This is the Wiener filter optimality criterion — model learns the exact
    #   ratio mask without labels or RING_WEIGHT.
    eps = 1e-8
    enh_spec, _, _ = model(mic_spec)
    enhanced_mag = (enh_spec[..., 0] ** 2 + enh_spec[..., 1] ** 2 + eps).sqrt()
    return F.l1_loss(torch.log(enhanced_mag), torch.log(tgt_mag + eps))


# ── Main training loop ─────────────────────────────────────────────────────────

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to warm-start from.')
    ap.add_argument('--reset-best', action='store_true',
                    help='Reset best_loss to inf on resume (use when switching loss functions).')
    ap.add_argument('--lr', type=float, default=None)
    args, _ = ap.parse_known_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window    = torch.hann_window(N_FFT).sqrt().to(device)
    model     = FeedbackMaskNet().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr or LR)
    # CosineAnnealingLR cycles lr between LR and 1e-6 over EPOCHS.
    # Unlike ReduceLROnPlateau, it never freezes permanently — even if a
    # local minimum is hit, the next cycle restarts at LR and can escape.
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    ckpt_dir  = PROJECT_ROOT / 'checkpoints' / 'gtcrn_feedback'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer    = SummaryWriter(str(ckpt_dir / 'tb'))

    # ── Load files ─────────────────────────────────────────────────────────────
    vocal_files = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                   if not f.name.startswith('._') and '__MACOSX' not in f.parts]
    ir_pool_dir      = PROJECT_ROOT / 'data' / 'ir_pool'
    mains_ir_files   = list(ir_pool_dir.glob('mains_*.wav'))
    monitor_ir_files = list(ir_pool_dir.glob('monitor_*.wav'))
    noise_files      = list((PROJECT_ROOT / 'data' / 'noise').rglob('*.wav'))
    room_ir_files    = [f for f in (PROJECT_ROOT / 'data' / 'public_irs').rglob('*.wav')
                        if not f.name.startswith('._') and '__MACOSX' not in f.parts]

    assert vocal_files,      'No vocal files in data/clean_vocals/'
    assert mains_ir_files,   'No mains IRs — run simulator/generate_ir_pool.py'
    assert monitor_ir_files, 'No monitor IRs — run simulator/generate_ir_pool.py'
    assert noise_files,      'No noise files in data/noise/'

    # Filter to files long enough
    vocal_files = [f for f in vocal_files
                   if (info := sf.info(str(f))).frames / info.samplerate >= SEQ_SECS]
    assert vocal_files, f'No vocal files >= {SEQ_SECS}s'

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'FeedbackMaskNet (per-bin GRU): {n_params:,} parameters on {device}')
    print(f'Clip: {SEQ_SECS}s ({SEQ_LEN} samples), batch accumulation: {BATCH_SIZE}')
    print(f'Vocal files: {len(vocal_files)}, noise: {len(noise_files)}, '
          f'mains IRs: {len(mains_ir_files)}, monitor IRs: {len(monitor_ir_files)}')

    best_loss   = float('inf')
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
        model.load_state_dict(state['model'])
        best_loss   = float('inf') if args.reset_best else state.get('best_loss', float('inf'))
        start_epoch = state.get('epoch', 0) + 1
        print(f'Resumed from {args.resume} (epoch {start_epoch}, best_loss={best_loss:.4f})')

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        valid_steps = 0
        optimizer.zero_grad()

        for step in tqdm(range(N_STEPS), desc=f'Epoch {epoch}/{EPOCHS}'):
            # Sample audio files
            vocal_np, _ = sf.read(str(random.choice(vocal_files)), dtype='float32')
            if vocal_np.ndim > 1: vocal_np = vocal_np.mean(1)
            offset   = random.randint(0, max(0, len(vocal_np) - SEQ_LEN - 1))
            vocal_np = vocal_np[offset:offset + SEQ_LEN]

            mains_ir_np,   _ = sf.read(str(random.choice(mains_ir_files)),   dtype='float32')
            monitor_ir_np, _ = sf.read(str(random.choice(monitor_ir_files)), dtype='float32')
            # Stereo mains/monitor = L and R speakers; mono mic picks up both — sum channels
            if mains_ir_np.ndim > 1:   mains_ir_np   = mains_ir_np.sum(1)
            if monitor_ir_np.ndim > 1: monitor_ir_np = monitor_ir_np.sum(1)
            mains_ir_np   = mains_ir_np[:MAX_IR_LEN]
            monitor_ir_np = monitor_ir_np[:MAX_IR_LEN]

            if room_ir_files:
                room_ir_path = random.choice(room_ir_files)
                room_ir_np, _ = sf.read(str(room_ir_path), dtype='float32')
                # Stereo room IR = two mic positions; pick one randomly for training diversity
                if room_ir_np.ndim > 1: room_ir_np = room_ir_np[:, random.randint(0, room_ir_np.shape[1] - 1)]
            else:
                room_ir_path = None
                rt60       = np.random.uniform(0.2, 2.0)
                t_arr      = np.arange(int(rt60 * SR)) / SR
                room_ir_np = np.random.randn(len(t_arr)).astype(np.float32)
                room_ir_np *= np.exp(-6.9 * t_arr / rt60).astype(np.float32)
                room_ir_np /= np.abs(room_ir_np).max() + 1e-8

            noise_np, _ = sf.read(str(random.choice(noise_files)), dtype='float32')
            if noise_np.ndim > 1: noise_np = noise_np.mean(1)
            if len(noise_np) < SEQ_LEN:
                noise_np = np.tile(noise_np, SEQ_LEN // len(noise_np) + 1)
            n0       = random.randint(0, len(noise_np) - SEQ_LEN)
            noise_np = noise_np[n0:n0 + SEQ_LEN]

            loss = train_one_step(
                model, vocal_np,
                mains_ir_np, monitor_ir_np, room_ir_np, noise_np,
                device, window, room_ir_path=room_ir_path, epoch=epoch
            )

            if not torch.isfinite(loss):
                continue

            (loss / BATCH_SIZE).backward()
            valid_steps += 1

            if valid_steps % BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += float(loss.item())

        # Flush remaining gradients
        if valid_steps % BATCH_SIZE != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(valid_steps // BATCH_SIZE, 1)
        scheduler.step()
        writer.add_scalar('loss/train', avg_loss, epoch)
        cur_lr  = optimizer.param_groups[0]['lr']
        best_str = f'{best_loss:.4f}' if best_loss < float('inf') else 'none'
        phase   = 1 if epoch < 150 else 2
        print(f'Epoch {epoch:3d} | loss {avg_loss:.4f} | best {best_str} | '
              f'phase {phase} | lr {cur_lr:.2e} | valid {valid_steps}/{N_STEPS}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_loss': best_loss},
                       str(ckpt_dir / 'best.pt'))
            print('  ✓ New best')


if __name__ == '__main__':
    train()
