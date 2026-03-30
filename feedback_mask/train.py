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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model import FeedbackMaskNet, SR, N_FFT, HOP, N_FREQ

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_SECS     = 4.0       # seconds per training clip
SEQ_LEN      = int(SEQ_SECS * SR)
BATCH_SIZE   = 4         # gradient accumulation steps before optimizer.step()
EPOCHS       = 300
LR           = 3e-4      # higher than FDKFNet — simpler gradient path allows faster LR
GRAD_CLIP    = 1.0
MAX_IR_LEN         = int(1.5 * SR)
FEEDBACK_TRUNC     = int(0.05 * SR)   # 50ms — captures resonant modes, keeps IIR order low
N_STEPS            = 200              # sequences per epoch


def make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def sample_gain():
    """
    Gain relative to normalised IR (unit spectral peak).
    <1.0 = stable (decaying resonance), =1.0 = marginal, >1.0 = howl.
    Distribution: 15% silent path / 20% sub-threshold / 25% near-threshold / 40% howling.
    """
    def _one():
        t = random.random()
        if t < 0.15:   return 0.0                       # path completely off
        elif t < 0.35: return random.uniform(0.2, 0.7)  # sub-threshold
        elif t < 0.60: return random.uniform(0.7, 1.0)  # near-threshold ringing
        else:          return random.uniform(1.0, 2.5)  # above-threshold howl
    return _one(), _one()


# ── HybridLoss (STFT-domain + SI-SDR) ─────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Compressed complex STFT loss + SI-SDR.
    Adapted from GTCRN original (gtcrn/loss.py) for 48 kHz / N_FFT=960.

    Compressed MSE gives dense per-bin gradient at every frame, making this
    substantially more convex than sequence-level SI-SDR alone.
    Weight: 30*(RI loss) + 70*(magnitude loss) + SI-SDR  (same as original).
    """
    def __init__(self):
        super().__init__()
        win = torch.hann_window(N_FFT).sqrt()
        self.register_buffer('window', win)

    def forward(self, pred, true):
        """pred, true: (B, F, T, 2)"""
        pr, pi = pred[..., 0], pred[..., 1]
        tr, ti = true[..., 0], true[..., 1]
        pm = (pr**2 + pi**2 + 1e-12).sqrt()
        tm = (tr**2 + ti**2 + 1e-12).sqrt()

        real_loss = F.mse_loss(pr / pm.pow(0.7), tr / tm.pow(0.7))
        imag_loss = F.mse_loss(pi / pm.pow(0.7), ti / tm.pow(0.7))
        mag_loss  = F.mse_loss(pm.pow(0.3),       tm.pow(0.3))

        # SI-SDR in time domain
        pred_c  = pr + 1j * pi
        true_c  = tr + 1j * ti
        y_pred  = torch.istft(pred_c, N_FFT, HOP, N_FFT, self.window)
        y_true  = torch.istft(true_c, N_FFT, HOP, N_FFT, self.window)
        dot     = (y_pred * y_true).sum(-1, keepdim=True)
        s_tgt   = dot / (y_true.pow(2).sum(-1, keepdim=True) + 1e-8) * y_true
        sisnr   = -(s_tgt.norm(dim=-1)**2 / (y_pred - s_tgt).norm(dim=-1).pow(2).clamp(1e-8)).log10().mean()

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr


# ── Per-step training function ─────────────────────────────────────────────────

def _norm_ir(ir):
    """Normalise IR to unit spectral peak so gain=1.0 is exactly the stability threshold."""
    peak = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    return ir / (peak + 1e-8)


def train_one_step(model, criterion, vocal_np, mains_ir_np, monitor_ir_np,
                   room_ir_np, noise_np, device, window):
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

    # Recursive feedback — IIR filter closes the PA→room→mic loop.
    # Truncate to FEEDBACK_TRUNC samples (50ms): captures the resonant modes
    # that drive instability; longer tail is already covered by room_ir reverb.
    mains_gain, monitor_gain = sample_gain()
    trunc      = min(len(mains_ir_np), len(monitor_ir_np), FEEDBACK_TRUNC)
    mains_h    = _norm_ir(mains_ir_np[:trunc])   * mains_gain
    monitor_h  = _norm_ir(monitor_ir_np[:trunc]) * monitor_gain
    h_combined = mains_h + monitor_h

    if h_combined.max() == 0:
        # Both paths silent — no feedback, pass noisy clean through
        mic_np = noisy_clean.astype(np.float32)
    else:
        # lfilter denominator: [1, -h[0], -h[1], ...] implements the closed loop
        a = np.concatenate([[1.0], -h_combined.astype(np.float64)])
        mic_np = lfilter([1.0], a, noisy_clean).astype(np.float32)

    mic_np = np.clip(mic_np, -1.0, 1.0)

    # STFT → (1, F, T, 2)
    mic_t   = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    tgt_t   = torch.from_numpy(target_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    tgt_stft = torch.stft(tgt_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)
    tgt_spec = torch.stack([tgt_stft.real, tgt_stft.imag], dim=-1)

    enh_spec = model(mic_spec)
    return criterion(enh_spec, tgt_spec)


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
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
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
    room_ir_files    = list((PROJECT_ROOT / 'data' / 'public_irs').rglob('*.wav'))

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
            mains_ir_np   = mains_ir_np[:MAX_IR_LEN]
            monitor_ir_np = monitor_ir_np[:MAX_IR_LEN]

            if room_ir_files:
                room_ir_np, _ = sf.read(str(random.choice(room_ir_files)), dtype='float32')
            else:
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
                device, window
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
        best_str = f'{best_loss:.4f}' if best_loss < float('inf') else 'none'
        print(f'Epoch {epoch:3d} | loss {avg_loss:.4f} | best {best_str} | valid {valid_steps}/{N_STEPS}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_loss': best_loss},
                       str(ckpt_dir / 'best.pt'))
            print('  ✓ New best')


if __name__ == '__main__':
    train()
