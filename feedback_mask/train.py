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
SEQ_SECS     = 4.0       # seconds per training clip
SEQ_LEN      = int(SEQ_SECS * SR)
BATCH_SIZE   = 16        # gradient accumulation steps before optimizer.step()
EPOCHS       = 300
LR           = 1e-4      # reduced from 3e-4: mask supervision adds variance, needs stable LR
GRAD_CLIP    = 1.0
MAX_IR_LEN         = int(1.5 * SR)
FEEDBACK_TRUNC     = int(0.05 * SR)   # 50ms — captures resonant modes, keeps IIR order low
N_STEPS            = 1600             # sequences per epoch — scaled up with BATCH_SIZE=16 to
                                      # maintain ~100 optimizer steps/epoch (was 400/4=100)


def make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def sample_gain(epoch):
    """
    Curriculum gain distribution.

    Phase 1 (epochs 1-149): onset-heavy.
      Model learns to detect and notch at the stability threshold.
      Mostly near-threshold cases where the mic still has vocal information.
      10% off / 15% sub-threshold / 60% near-threshold (0.7-1.2) / 15% moderate (1.2-1.8)
      No extreme cases — too little vocal info to learn from.

    Phase 2 (epochs 150+): refine for severe feedback.
      Model already knows onset detection; now learns to break a full howl.
      10% off / 10% sub-threshold / 35% near-threshold / 30% moderate / 15% extreme (1.8-2.5)
    """
    def _one():
        t = random.random()
        if epoch < 150:
            if t < 0.10:   return 0.0                        # path off
            elif t < 0.25: return random.uniform(0.2, 0.7)   # sub-threshold (15%)
            elif t < 0.85: return random.uniform(0.7, 1.2)   # near-threshold onset (60%)
            else:          return random.uniform(1.2, 1.4)   # moderate severe (15%) — capped at 1.4 to keep examples learnable
        else:
            if t < 0.10:   return 0.0                        # path off
            elif t < 0.20: return random.uniform(0.2, 0.7)   # sub-threshold (10%)
            elif t < 0.55: return random.uniform(0.7, 1.2)   # near-threshold (35%)
            elif t < 0.85: return random.uniform(1.2, 1.8)   # moderate severe (30%)
            else:          return random.uniform(1.8, 2.5)   # extreme (15%)
    return _one(), _one()


# ── HybridLoss (STFT-domain + SI-SDR) ─────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Compressed complex STFT loss + SI-SDR + onset speed + early vocal quality.

    Base terms (dense learning signal):
      30 × RI compressed MSE   — phase-aware per-bin gradient every frame
      70 × magnitude MSE       — perceptual spectral reconstruction
       1 × SI-SDR full seq     — overall signal quality

    Refinement terms (nudge toward fast suppression and clean vocals at onset):
       2 × onset-weighted mag  — frames near t=0 weighted by exp(-t/50),
                                  penalises residual feedback surviving early frames
     0.5 × SI-SDR first second — rewards output sounding clean immediately at onset,
                                  not just averaged over the whole 4s clip
    """
    ONSET_TAU   = 50    # frames — onset weight decay (~500ms at 10ms/frame)
    EARLY_SECS  = 1.0   # seconds of signal used for early SI-SDR term

    def __init__(self):
        super().__init__()
        win = torch.hann_window(N_FFT).sqrt()
        self.register_buffer('window', win)

    @staticmethod
    def _sdr(y_pred, y_true):
        """Scale-sensitive SDR on (..., T) tensors. Returns scalar.
        Unlike SI-SDR, penalises amplitude mismatch — a quieter prediction
        is NOT equivalent to a full-amplitude one. This prevents the model
        from learning broadband attenuation as a shortcut."""
        return -(y_true.pow(2).sum(-1) /
                 (y_pred - y_true).pow(2).sum(-1).clamp(1e-8)).log10().mean()

    def forward(self, pred, true, mic):
        """pred, true, mic: (B, F, T, 2)"""
        pr, pi = pred[..., 0], pred[..., 1]
        tr, ti = true[..., 0], true[..., 1]
        mr, mi = mic[..., 0],  mic[..., 1]
        pm = (pr**2 + pi**2 + 1e-12).sqrt()
        tm = (tr**2 + ti**2 + 1e-12).sqrt()
        mm = (mr**2 + mi**2 + 1e-12).sqrt()

        # ── Base STFT terms ────────────────────────────────────────────────────
        # Compression 0.5 (was 0.3): less aggressive compression = amplitude
        # mismatches hurt more, preventing the model outputting a quiet but
        # spectrally-correct signal and getting away with it.
        real_loss = F.mse_loss(pr / pm.pow(0.5), tr / tm.pow(0.5))
        imag_loss = F.mse_loss(pi / pm.pow(0.5), ti / tm.pow(0.5))
        mag_loss  = F.mse_loss(pm.pow(0.5),       tm.pow(0.5))

        # ── Time-domain signals ────────────────────────────────────────────────
        # Clamp before istft: early-training model output can be extreme,
        # causing istft overflow → NaN SDR loss. Clamp is tight enough to
        # not affect well-trained outputs (normal STFT bins << 1e3).
        y_pred = torch.istft((pr + 1j * pi).clamp(-1e3, 1e3), N_FFT, HOP, N_FFT, self.window)
        y_true = torch.istft(tr + 1j * ti,                    N_FFT, HOP, N_FFT, self.window)

        # ── Full-sequence SDR (scale-sensitive) ────────────────────────────────
        sdr_full = self._sdr(y_pred, y_true)

        # ── Onset-weighted magnitude loss ──────────────────────────────────────
        T = pm.shape[2]
        t = torch.arange(T, dtype=pm.dtype, device=pm.device)
        onset_w = torch.exp(-t / self.ONSET_TAU)
        onset_w = onset_w / onset_w.sum()
        frame_mag = ((pm.pow(0.5) - tm.pow(0.5))**2).mean(dim=(0, 1))   # (T,)
        onset_mag_loss = (frame_mag * onset_w).sum()

        # ── Early SDR (first EARLY_SECS seconds) ───────────────────────────────
        early_samps = int(self.EARLY_SECS * SR)
        sdr_early = self._sdr(y_pred[..., :early_samps],
                               y_true[..., :early_samps])

        # ── Direct mask supervision ────────────────────────────────────────────
        # We generate synthetic data, so we know exactly where feedback is.
        # Ideal mask: target_mag / mic_mag — suppress where mic has extra energy
        # (feedback), pass through where mic ≈ target (clean vocal).
        # Predicted mask: enhanced_mag / mic_mag — what the model actually did.
        # Per-bin, per-frame MSE gives direct gradient signal about WHERE to
        # suppress, not just indirect signal from reconstruction error.
        ideal_mask = (tm / (mm + 1e-8)).clamp(0.0, 1.0)
        pred_mask  = (pm / (mm + 1e-8)).clamp(0.0, 1.0)
        mask_loss  = F.mse_loss(pred_mask, ideal_mask)

        return (30 * (real_loss + imag_loss)
                + 70 * mag_loss
                +  1 * sdr_full
                +  2 * onset_mag_loss
                + 0.5 * sdr_early
                + 40 * mask_loss)


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
    """
    n = n if n is not None else random.randint(0, 3)
    if n == 0:
        return h
    modal_freqs = _modal_freqs_for_ir(ir_path) if ir_path else None
    t = np.arange(len(h)) / sr
    for _ in range(n):
        if modal_freqs:
            freq = random.choice(modal_freqs)
        else:
            freq = random.uniform(100, 4000)
        Q         = random.uniform(15, 80)
        gain      = random.uniform(0.05, 0.3)
        decay     = np.pi * freq / Q
        resonance = gain * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)
        h = h + resonance.astype(h.dtype)
    return h


def train_one_step(model, criterion, vocal_np, mains_ir_np, monitor_ir_np,
                   room_ir_np, noise_np, device, window, room_ir_path=None, epoch=1):
    """
    Recursive feedback simulation → STFT → model → loss.

    mic[n] = speech[n] + gain × Σ h[k] × mic[n-k]   (IIR via lfilter)

    gain < 1.0  → decaying resonance
    gain = 1.0  → sustained tone (marginal stability)
    gain > 1.0  → exponential howl, clipped at ±1 (saturated squeal)

    Target is always the reverberant clean vocal — model learns to invert the
    feedback loop and output clean speech regardless of gain level.
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
    if random.random() < 0.50:
        gain_lo  = random.uniform(0.2, 0.9)
        gain_hi  = random.uniform(1.0, 1.4 if epoch < 150 else 2.5)
        split    = random.randint(int(0.2 * SEQ_LEN), int(0.5 * SEQ_LEN))
        h_lo     = _add_resonators(mains_norm * gain_lo + mon_norm * gain_lo,
                                   SR, ir_path=room_ir_path)
        h_hi     = _add_resonators(mains_norm * gain_hi + mon_norm * gain_hi,
                                   SR, ir_path=room_ir_path)
        a_lo = np.concatenate([[1.0], -h_lo.astype(np.float64)])
        a_hi = np.concatenate([[1.0], -h_hi.astype(np.float64)])
        zi   = np.zeros(len(a_lo) - 1)
        y1, zi = lfilter([1.0], a_lo, noisy_clean[:split], zi=zi)
        y2, _  = lfilter([1.0], a_hi, noisy_clean[split:], zi=zi)
        mic_np = np.concatenate([y1, y2])
    else:
        mains_gain, monitor_gain = sample_gain(epoch)
        h_combined = _add_resonators(mains_norm * mains_gain + mon_norm * monitor_gain,
                                     SR, ir_path=room_ir_path)
        if h_combined.max() == 0:
            mic_np = noisy_clean.copy()
        else:
            a      = np.concatenate([[1.0], -h_combined.astype(np.float64)])
            mic_np = lfilter([1.0], a, noisy_clean)

    mic_np = np.nan_to_num(mic_np, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    mic_np = np.clip(mic_np, -1.0, 1.0)

    # Mic frequency response — same profile applied to both mic and target so the
    # model sees consistent coloration and only needs to suppress the feedback
    mic_name  = random.choice(MIC_NAMES)
    mic_np    = apply_random_mic_response(mic_np,    SR, mic_name=mic_name)
    target_np = apply_random_mic_response(target_np, SR, mic_name=mic_name)

    # STFT → (1, F, T, 2)
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    tgt_t    = torch.from_numpy(target_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    tgt_stft = torch.stft(tgt_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)
    tgt_spec = torch.stack([tgt_stft.real, tgt_stft.imag], dim=-1)

    enh_spec, _ = model(mic_spec)

    return criterion(enh_spec, tgt_spec, mic_spec)


# ── Main training loop ─────────────────────────────────────────────────────────

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to warm-start from.')
    ap.add_argument('--lr', type=float, default=None)
    args, _ = ap.parse_known_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window    = torch.hann_window(N_FFT).sqrt().to(device)
    model     = FeedbackMaskNet().to(device)
    criterion = HybridLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr or LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=5, min_lr=1e-6)
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
    print(f'FeedbackMaskNet (GTCRN@48kHz): {n_params:,} parameters on {device}')
    print(f'Clip: {SEQ_SECS}s ({SEQ_LEN} samples), batch accumulation: {BATCH_SIZE}')
    print(f'Vocal files: {len(vocal_files)}, noise: {len(noise_files)}, '
          f'mains IRs: {len(mains_ir_files)}, monitor IRs: {len(monitor_ir_files)}')

    best_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
        model.load_state_dict(state['model'])
        best_loss = state.get('best_loss', float('inf'))
        print(f'Warm-started from {args.resume} (best_loss={best_loss:.4f})')

    for epoch in range(1, EPOCHS + 1):
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
                model, criterion, vocal_np,
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
        scheduler.step(avg_loss)
        writer.add_scalar('loss/train', avg_loss, epoch)
        best_str = f'{best_loss:.4f}' if best_loss < float('inf') else 'none'
        print(f'Epoch {epoch:3d} | loss {avg_loss:.4f} | best {best_str} | valid {valid_steps}/{N_STEPS}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_loss': best_loss},
                       str(ckpt_dir / 'best.pt'))
            print('  ✓ New best')


if __name__ == '__main__':
    train()
