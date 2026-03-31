"""
feedback_detect/train.py — Train the FeedbackDetector.

Task: binary classification per (frequency bin, frame).
Label = 1 where mic has significantly more energy than clean target
        (= that bin is in a Larsen feedback loop right now).

Loss: weighted binary cross-entropy.
  pos_weight = 80  (feedback bins are ~1% of all bin/frame pairs)

This is far simpler than the reconstruction approach in feedback_mask/:
  - No ISTFT, no SDR, no complex mask
  - Direct supervision: we tell the model exactly which bins are feedback
  - The model only needs to learn "growing energy = feedback"
  - Notch placement / depth / persistence are handled by notch.py, not learned

Usage:
    python feedback_detect/train.py
    python feedback_detect/train.py --resume checkpoints/feedback_detect/best.pt
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_detect'))

from model import FeedbackDetector, SR, N_FFT, HOP, N_FREQ, prepare_features

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_SECS        = 4.0
SEQ_LEN         = int(SEQ_SECS * SR)
BATCH_SIZE      = 16
EPOCHS          = 300
LR              = 1e-4
GRAD_CLIP       = 1.0
N_STEPS         = 1600
MAX_IR_LEN      = int(1.5 * SR)
FEEDBACK_TRUNC  = int(0.05 * SR)
POS_WEIGHT      = 80.0    # upweight rare positive (feedback) bins
DETECT_THRESH   = 1.5     # mic_mag / target_mag > this → label = 1


def make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def _norm_ir(ir):
    peak = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    return ir / (peak + 1e-8)


def _add_resonators(h, sr, n=None):
    n = n if n is not None else random.randint(0, 3)
    if n == 0:
        return h
    t = np.arange(len(h)) / sr
    for _ in range(n):
        freq  = random.uniform(100, 4000)
        Q     = random.uniform(15, 80)
        gain  = random.uniform(0.05, 0.3)
        decay = np.pi * freq / Q
        h     = h + (gain * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)).astype(h.dtype)
    return h


def sample_gain(epoch):
    """Curriculum gain distribution — same as feedback_mask/train.py."""
    def _one():
        t = random.random()
        if epoch < 150:
            if t < 0.10:   return 0.0
            elif t < 0.25: return random.uniform(0.2, 0.7)
            elif t < 0.85: return random.uniform(0.7, 1.2)
            else:          return random.uniform(1.2, 1.4)
        else:
            if t < 0.10:   return 0.0
            elif t < 0.20: return random.uniform(0.2, 0.7)
            elif t < 0.55: return random.uniform(0.7, 1.2)
            elif t < 0.85: return random.uniform(1.2, 1.8)
            else:          return random.uniform(1.8, 2.5)
    return _one(), _one()


def make_training_pair(vocal_np, mains_ir_np, monitor_ir_np,
                       room_ir_np, noise_np, device, window, epoch):
    """
    Returns (mic_mag, labels) as torch tensors on device.

    mic_mag : (1, N_FREQ, T)  — STFT magnitude of mic signal
    labels  : (1, N_FREQ, T)  — 1 where mic_mag / target_mag > DETECT_THRESH
    """
    # Reverberant clean vocal = what should come out
    hpf       = make_hpf(np.random.uniform(70, 120))
    target_np = fftconvolve(vocal_np, room_ir_np)[:SEQ_LEN].astype(np.float32)
    target_np = sosfilt(hpf, target_np).astype(np.float32)

    noise_np  = noise_np[:SEQ_LEN]
    if random.random() < 0.5:
        noise_np = fftconvolve(noise_np, room_ir_np)[:SEQ_LEN].astype(np.float32)
    v_rms        = float(np.sqrt(np.mean(target_np ** 2))) + 1e-8
    n_rms        = float(np.sqrt(np.mean(noise_np ** 2)))  + 1e-8
    snr_db       = np.random.uniform(5, 40)
    noisy_clean  = (target_np + noise_np * (v_rms / n_rms * 10 ** (-snr_db / 20))).astype(np.float64)

    # Pure-feedback clip (10%): near-silence vocal, label = 1 where feedback builds
    if random.random() < 0.10:
        noisy_clean = (noise_np[:SEQ_LEN] * 0.05).astype(np.float64)
        target_np   = np.zeros(SEQ_LEN, dtype=np.float32)

    # ── Build feedback IR ──────────────────────────────────────────────────
    trunc      = min(len(mains_ir_np), len(monitor_ir_np), FEEDBACK_TRUNC)
    mains_norm = _norm_ir(mains_ir_np[:trunc])
    mon_norm   = _norm_ir(monitor_ir_np[:trunc])
    # Renormalise combined IR to prevent aligned-resonance instability
    combined_check = mains_norm + mon_norm
    combined_peak  = np.abs(np.fft.rfft(combined_check, n=max(len(combined_check) * 4, 4096))).max()
    if combined_peak > 1.0:
        mains_norm /= (combined_peak + 1e-8)
        mon_norm   /= (combined_peak + 1e-8)

    # ── Recursive IIR feedback simulation ─────────────────────────────────
    if random.random() < 0.50:
        gain_lo  = random.uniform(0.2, 0.9)
        gain_hi  = random.uniform(1.0, 1.4 if epoch < 150 else 2.5)
        split    = random.randint(int(0.2 * SEQ_LEN), int(0.5 * SEQ_LEN))
        h_lo     = _add_resonators(mains_norm * gain_lo + mon_norm * gain_lo, SR)
        h_hi     = _add_resonators(mains_norm * gain_hi + mon_norm * gain_hi, SR)
        a_lo     = np.concatenate([[1.0], -h_lo.astype(np.float64)])
        a_hi     = np.concatenate([[1.0], -h_hi.astype(np.float64)])
        zi       = np.zeros(len(a_lo) - 1)
        y1, zi   = lfilter([1.0], a_lo, noisy_clean[:split], zi=zi)
        y2, _    = lfilter([1.0], a_hi, noisy_clean[split:],  zi=zi)
        mic_np   = np.concatenate([y1, y2])
    else:
        mg, mong = sample_gain(epoch)
        h        = _add_resonators(mains_norm * mg + mon_norm * mong, SR)
        if h.max() == 0:
            mic_np = noisy_clean.copy()
        else:
            a      = np.concatenate([[1.0], -h.astype(np.float64)])
            mic_np = lfilter([1.0], a, noisy_clean)

    mic_np = np.nan_to_num(mic_np, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    mic_np = np.clip(mic_np, -1.0, 1.0)

    # ── STFT magnitudes ────────────────────────────────────────────────────
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    tgt_t    = torch.from_numpy(target_np).unsqueeze(0).to(device)

    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    tgt_stft = torch.stft(tgt_t, N_FFT, HOP, N_FFT, window, return_complex=True)

    mic_mag  = mic_stft.abs()   # (1, N_FREQ, T)
    tgt_mag  = tgt_stft.abs()   # (1, N_FREQ, T)

    # ── Labels: where is the mic significantly louder than clean? ──────────
    # mic_mag >> tgt_mag at a bin → that bin is in a Larsen loop right now
    ratio  = mic_mag / (tgt_mag + 1e-8)
    labels = (ratio > DETECT_THRESH).float()

    return mic_mag, labels


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--lr',     type=float, default=None)
    args, _ = ap.parse_known_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window  = torch.hann_window(N_FFT).sqrt().to(device)
    model   = FeedbackDetector().to(device)
    ckpt_dir = PROJECT_ROOT / 'checkpoints' / 'feedback_detect'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(str(ckpt_dir / 'tb'))

    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    optimizer  = Adam(model.parameters(), lr=args.lr or LR)
    scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=5, min_lr=1e-6)

    # ── Load audio files ───────────────────────────────────────────────────
    vocal_files    = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                      if not f.name.startswith('._') and '__MACOSX' not in f.parts]
    ir_pool        = PROJECT_ROOT / 'data' / 'ir_pool'
    mains_files    = list(ir_pool.glob('mains_*.wav'))
    monitor_files  = list(ir_pool.glob('monitor_*.wav'))
    noise_files    = list((PROJECT_ROOT / 'data' / 'noise').rglob('*.wav'))
    room_ir_files  = [f for f in (PROJECT_ROOT / 'data' / 'public_irs').rglob('*.wav')
                      if not f.name.startswith('._') and '__MACOSX' not in f.parts]

    assert vocal_files,   'No vocal files in data/clean_vocals/'
    assert mains_files,   'No mains IRs'
    assert monitor_files, 'No monitor IRs'
    assert noise_files,   'No noise files'

    vocal_files = [f for f in vocal_files
                   if (i := sf.info(str(f))).frames / i.samplerate >= SEQ_SECS]
    assert vocal_files, f'No vocal files >= {SEQ_SECS}s'

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'FeedbackDetector: {n_params:,} params on {device}')
    print(f'Vocals: {len(vocal_files)}, noise: {len(noise_files)}, '
          f'mains: {len(mains_files)}, monitor: {len(monitor_files)}')

    best_loss = float('inf')

    if args.resume:
        ckpt      = torch.load(args.resume, map_location=device)
        state     = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
        model.load_state_dict(state['model'])
        best_loss = state.get('best_loss', float('inf'))
        print(f'Resumed from {args.resume}  (best_loss={best_loss:.4f})')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        valid_steps = 0
        optimizer.zero_grad()

        for step in tqdm(range(N_STEPS), desc=f'Epoch {epoch}/{EPOCHS}'):
            # ── Sample audio ───────────────────────────────────────────────
            vocal_np, _ = sf.read(str(random.choice(vocal_files)), dtype='float32')
            if vocal_np.ndim > 1: vocal_np = vocal_np.mean(1)
            off      = random.randint(0, max(0, len(vocal_np) - SEQ_LEN - 1))
            vocal_np = vocal_np[off:off + SEQ_LEN]

            mains_np,   _ = sf.read(str(random.choice(mains_files)),   dtype='float32')
            monitor_np, _ = sf.read(str(random.choice(monitor_files)), dtype='float32')
            if mains_np.ndim   > 1: mains_np   = mains_np.sum(1)
            if monitor_np.ndim > 1: monitor_np = monitor_np.sum(1)
            mains_np   = mains_np[:MAX_IR_LEN]
            monitor_np = monitor_np[:MAX_IR_LEN]

            if room_ir_files:
                room_np, _ = sf.read(str(random.choice(room_ir_files)), dtype='float32')
                if room_np.ndim > 1:
                    room_np = room_np[:, random.randint(0, room_np.shape[1] - 1)]
            else:
                rt60    = np.random.uniform(0.2, 2.0)
                t_arr   = np.arange(int(rt60 * SR)) / SR
                room_np = np.random.randn(len(t_arr)).astype(np.float32)
                room_np *= np.exp(-6.9 * t_arr / rt60).astype(np.float32)
                room_np /= np.abs(room_np).max() + 1e-8

            noise_np, _ = sf.read(str(random.choice(noise_files)), dtype='float32')
            if noise_np.ndim > 1: noise_np = noise_np.mean(1)
            if len(noise_np) < SEQ_LEN:
                noise_np = np.tile(noise_np, SEQ_LEN // len(noise_np) + 1)
            n0       = random.randint(0, len(noise_np) - SEQ_LEN)
            noise_np = noise_np[n0:n0 + SEQ_LEN]

            # ── Forward pass ───────────────────────────────────────────────
            mic_mag, labels = make_training_pair(
                vocal_np, mains_np, monitor_np, room_np, noise_np,
                device, window, epoch
            )

            prob, _ = model(prepare_features(mic_mag))   # (1, N_FREQ, T)

            loss = F.binary_cross_entropy_with_logits(
                torch.logit(prob.clamp(1e-6, 1 - 1e-6)),
                labels,
                pos_weight=pos_weight
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
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'best_loss': best_loss},
                       str(ckpt_dir / 'best.pt'))
            print('  ✓ New best')


if __name__ == '__main__':
    train()
