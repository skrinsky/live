"""
feedback_mask/train_gtcrn.py — Source-separation feedback suppression training (GTCRN48k).

De-Feedback-style approach: single-channel, no reference signal, no oracle ring labels.
  mic (speech + room + feedback + noise) → clean dry vocal

Loss: HybridLoss48k (complex MSE + magnitude MSE + SI-SNR). No BCE, no RING_WEIGHT.
Model: GTCRN48k (~24K params, Complex Ratio Mask, Dual-Path GRNN).

Compare with train.py (per-bin GRU, BCE, ring labels) to evaluate whether source
separation generalises better to real-world feedback vs the narrow-notch approach.

Usage:
    python feedback_mask/train_gtcrn.py
    python feedback_mask/train_gtcrn.py --resume checkpoints/gtcrn_sep/best.pt
"""

import sys
import random
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve, butter, sosfilt, lfilter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_mask'))

from model_gtcrn import GTCRN48k, HybridLoss48k, SR, N_FFT, HOP, N_FREQ
from mic_profiles import apply_random_mic_response, MIC_NAMES

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_SECS    = 10.0
SEQ_LEN     = int(SEQ_SECS * SR)
BATCH_SIZE  = 4           # smaller than BCE run — HybridLoss is heavier per step
EPOCHS      = 300
LR          = 1e-4
GRAD_CLIP   = 1.0
MAX_IR_LEN  = int(1.5 * SR)
FEEDBACK_TRUNC = int(0.05 * SR)
N_STEPS     = 200         # 200 steps × 4 batch = 50 optimizer steps/epoch


def make_hpf(cutoff_hz=90):
    return butter(2, cutoff_hz / (SR / 2), btype='high', output='sos')


def sample_gain(epoch):
    """Same curriculum as train.py."""
    def _one():
        t = random.random()
        if epoch < 30:
            if t < 0.10:   return 0.0
            elif t < 0.30: return random.uniform(0.5, 0.6)
            elif t < 0.70: return random.uniform(0.6, 0.85)
            else:          return random.uniform(0.85, 0.99)
        else:
            if t < 0.10:   return 0.0
            elif t < 0.20: return random.uniform(0.5, 0.6)
            elif t < 0.50: return random.uniform(0.6, 0.85)
            else:          return random.uniform(0.85, 0.99)
    return _one(), _one()


def _stability_check(h, max_gain=0.99):
    peak = np.abs(np.fft.rfft(h, n=max(len(h) * 4, 4096))).max()
    if peak > max_gain:
        h = h * (max_gain / (peak + 1e-8))
    return h


def _norm_ir(ir):
    peak = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    return ir / (peak + 1e-8)


def _load_room_dimensions():
    meta = PROJECT_ROOT / 'data' / 'public_irs' / 'room_dimensions.json'
    if meta.exists():
        import json
        with open(meta) as f:
            return json.load(f)
    return {}


_ROOM_DIMS = _load_room_dimensions()


def _modal_freqs_for_ir(ir_path):
    if not _ROOM_DIMS:
        return None
    path_str = str(ir_path).lower()
    for key, dims in _ROOM_DIMS.items():
        if key.split('_')[0] in path_str:
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
    n = n if n is not None else random.randint(0, 3)
    if n == 0:
        return h, []
    modal_freqs = _modal_freqs_for_ir(ir_path) if ir_path else None
    t = np.arange(len(h)) / sr
    ring_freqs = []
    for _ in range(n):
        freq = random.choice(modal_freqs) if modal_freqs else random.uniform(100, 4000)
        ring_freqs.append(freq)
        Q         = random.uniform(15, 80)
        gain      = random.uniform(0.05, 0.3)
        decay     = np.pi * freq / Q
        resonance = gain * np.exp(-decay * t) * np.cos(2 * np.pi * freq * t)
        h = h + resonance.astype(h.dtype)
    return h, ring_freqs


def train_one_step(model, loss_fn, vocal_np, mains_ir_np, monitor_ir_np,
                   room_ir_np, noise_np, device, window, room_ir_path=None, epoch=1):
    """
    Source-separation training step.

    Target: dry vocal (no room, no feedback, no noise).
    The model must learn to recover clean voice from the degraded mic signal.

    This mirrors De-Feedback's approach: classify each spectral component as
    "voice" vs "not-voice" (via Complex Ratio Mask) and pass only voice through.

    Why dry vocal as target (not vocal+room or vocal+noise):
      - Forces the model to learn what a clean voice sounds like, not just
        "voice-plus-everything-else-except-feedback"
      - Feedback, room reverb, and noise are all "not-voice" — the model removes
        them all in one pass, matching De-Feedback's reported behaviour
      - HybridLoss is scale-invariant (SI-SNR component) so the amplitude
        difference between reverberant mic and dry target is handled automatically
    """
    TARGET_RMS = 0.1

    # ── Target: dry vocal (before room/feedback/noise) ─────────────────────
    target_np = vocal_np[:SEQ_LEN].copy()
    hpf_tgt   = make_hpf(np.random.uniform(70, 120))
    target_np = sosfilt(hpf_tgt, target_np).astype(np.float32)

    # ── Mic: vocal → room → +noise → feedback IIR ──────────────────────────
    reverb_np = fftconvolve(vocal_np, room_ir_np)[:SEQ_LEN].astype(np.float32)
    hpf_rev   = make_hpf(np.random.uniform(70, 120))
    reverb_np = sosfilt(hpf_rev, reverb_np).astype(np.float32)

    noise_np = noise_np[:SEQ_LEN]
    if random.random() < 0.5:
        noise_np = fftconvolve(noise_np, room_ir_np)[:SEQ_LEN].astype(np.float32)
    vocal_rms   = float(np.sqrt(np.mean(reverb_np**2))) + 1e-8
    noise_rms   = float(np.sqrt(np.mean(noise_np**2)))  + 1e-8
    snr_db      = np.random.uniform(5, 40)
    noise_scale = vocal_rms / noise_rms * 10**(-snr_db / 20)
    noisy_reverb = (reverb_np + noise_np * noise_scale).astype(np.float64)

    # 10% pure-feedback case (open mic, no vocalist)
    if random.random() < 0.10:
        noisy_reverb = (noise_np[:SEQ_LEN] * 0.05).astype(np.float64)
        target_np    = np.zeros(SEQ_LEN, dtype=np.float32)

    # ── Recursive feedback IIR ──────────────────────────────────────────────
    trunc      = min(len(mains_ir_np), len(monitor_ir_np), FEEDBACK_TRUNC)
    mains_norm = _norm_ir(mains_ir_np[:trunc])
    mon_norm   = _norm_ir(monitor_ir_np[:trunc])
    _combined  = mains_norm + mon_norm
    _peak      = np.abs(np.fft.rfft(_combined, n=max(len(_combined)*4, 4096))).max()
    if _peak > 1.0:
        mains_norm = mains_norm / (_peak + 1e-8)
        mon_norm   = mon_norm   / (_peak + 1e-8)

    is_ramp = random.random() < 0.50
    if is_ramp:
        gain_lo = random.uniform(0.2, 0.7)
        gain_hi = random.uniform(0.85, 0.99)
        split   = random.randint(int(0.2 * SEQ_LEN), int(0.5 * SEQ_LEN))
        h_lo_raw, _ = _add_resonators(mains_norm * gain_lo + mon_norm * gain_lo,
                                      SR, ir_path=room_ir_path)
        h_hi_raw, _ = _add_resonators(mains_norm * gain_hi + mon_norm * gain_hi,
                                      SR, ir_path=room_ir_path)
        h_lo = _stability_check(h_lo_raw)
        h_hi = _stability_check(h_hi_raw)
        a_lo = np.concatenate([[1.0], -h_lo.astype(np.float64)])
        a_hi = np.concatenate([[1.0], -h_hi.astype(np.float64)])
        zi   = np.zeros(len(a_lo) - 1)
        y1, zi = lfilter([1.0], a_lo, noisy_reverb[:split], zi=zi)
        y2, _  = lfilter([1.0], a_hi, noisy_reverb[split:], zi=zi)
        mic_np = np.concatenate([y1, y2])
    else:
        mains_gain, monitor_gain = sample_gain(epoch)
        h_raw, _ = _add_resonators(
            mains_norm * mains_gain + mon_norm * monitor_gain,
            SR, ir_path=room_ir_path)
        h = _stability_check(h_raw)
        if h.max() == 0:
            mic_np = noisy_reverb.copy()
        else:
            a      = np.concatenate([[1.0], -h.astype(np.float64)])
            mic_np = lfilter([1.0], a, noisy_reverb)

    mic_np = np.nan_to_num(mic_np, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    mic_np = np.clip(mic_np, -1.0, 1.0)

    hpf_mic = make_hpf(np.random.uniform(70, 120))
    mic_np  = sosfilt(hpf_mic, mic_np).astype(np.float32)

    # ── Normalise ───────────────────────────────────────────────────────────
    mic_rms = float(np.sqrt(np.mean(mic_np**2)))  + 1e-8
    mic_np  = np.clip(mic_np * (TARGET_RMS / mic_rms), -1.0, 1.0).astype(np.float32)

    tgt_rms = float(np.sqrt(np.mean(target_np**2))) + 1e-8
    if tgt_rms > 0.001:
        target_np = np.clip(target_np * (TARGET_RMS / tgt_rms), -1.0, 1.0).astype(np.float32)

    # ── Mic profile coloration (applied to both so model sees consistent FR) ─
    mic_name  = random.choice(MIC_NAMES)
    mic_np    = apply_random_mic_response(mic_np,    SR, mic_name=mic_name)
    target_np = apply_random_mic_response(target_np, SR, mic_name=mic_name)

    # ── STFT ────────────────────────────────────────────────────────────────
    mic_t    = torch.from_numpy(mic_np).unsqueeze(0).to(device)
    tgt_t    = torch.from_numpy(target_np).unsqueeze(0).to(device)
    mic_stft = torch.stft(mic_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    tgt_stft = torch.stft(tgt_t, N_FFT, HOP, N_FFT, window, return_complex=True)
    mic_spec = torch.stack([mic_stft.real, mic_stft.imag], dim=-1)   # (1, F, T, 2)
    tgt_spec = torch.stack([tgt_stft.real, tgt_stft.imag], dim=-1)   # (1, F, T, 2)

    enh_spec = model(mic_spec)   # (1, F, T, 2)
    return loss_fn(enh_spec, tgt_spec)


# ── Main training loop ─────────────────────────────────────────────────────────

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume',     type=str, default=None)
    ap.add_argument('--reset-best', action='store_true')
    ap.add_argument('--lr',         type=float, default=None)
    args, _ = ap.parse_known_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # GTCRN expects sqrt-Hann window (same as original gtcrn.py)
    window    = torch.hann_window(N_FFT).sqrt().to(device)
    model     = GTCRN48k().to(device)
    loss_fn   = HybridLoss48k().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr or LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    ckpt_dir  = PROJECT_ROOT / 'checkpoints' / 'gtcrn_sep'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer    = SummaryWriter(str(ckpt_dir / 'tb'))

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

    vocal_files = [f for f in vocal_files
                   if (info := sf.info(str(f))).frames / info.samplerate >= SEQ_SECS]
    assert vocal_files, f'No vocal files >= {SEQ_SECS}s'

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'GTCRN48k (source separation): {n_params:,} parameters on {device}')
    print(f'Clip: {SEQ_SECS}s, batch accumulation: {BATCH_SIZE}, steps/epoch: {N_STEPS}')
    print(f'Vocal: {len(vocal_files)}, noise: {len(noise_files)}, '
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
            vocal_np, _ = sf.read(str(random.choice(vocal_files)), dtype='float32')
            if vocal_np.ndim > 1: vocal_np = vocal_np.mean(1)
            offset   = random.randint(0, max(0, len(vocal_np) - SEQ_LEN - 1))
            vocal_np = vocal_np[offset:offset + SEQ_LEN]

            mains_ir_np,   _ = sf.read(str(random.choice(mains_ir_files)),   dtype='float32')
            monitor_ir_np, _ = sf.read(str(random.choice(monitor_ir_files)), dtype='float32')
            if mains_ir_np.ndim > 1:   mains_ir_np   = mains_ir_np.sum(1)
            if monitor_ir_np.ndim > 1: monitor_ir_np = monitor_ir_np.sum(1)
            mains_ir_np   = mains_ir_np[:MAX_IR_LEN]
            monitor_ir_np = monitor_ir_np[:MAX_IR_LEN]

            if room_ir_files:
                room_ir_path = random.choice(room_ir_files)
                room_ir_np, _ = sf.read(str(room_ir_path), dtype='float32')
                if room_ir_np.ndim > 1:
                    room_ir_np = room_ir_np[:, random.randint(0, room_ir_np.shape[1] - 1)]
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
                model, loss_fn, vocal_np,
                mains_ir_np, monitor_ir_np, room_ir_np, noise_np,
                device, window, room_ir_path=room_ir_path, epoch=epoch)

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
        scheduler.step()
        writer.add_scalar('loss/train', avg_loss, epoch)
        cur_lr   = optimizer.param_groups[0]['lr']
        best_str = f'{best_loss:.4f}' if best_loss < float('inf') else 'none'
        print(f'Epoch {epoch:3d} | loss {avg_loss:.4f} | best {best_str} | '
              f'lr {cur_lr:.2e} | valid {valid_steps}/{N_STEPS}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_loss': best_loss},
                       str(ckpt_dir / 'best.pt'))
            print('  ✓ New best')


if __name__ == '__main__':
    train()
