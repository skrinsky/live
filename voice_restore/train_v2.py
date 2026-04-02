"""
voice_restore/train_v2.py — Train VoiceRestorer V2.

V2 adds:
  - notch-mask awareness
  - aperiodic / speech-friendly conditioning
  - dual harmonic vs aperiodic output heads
  - identity preservation outside repair regions
  - temporal smoothness on predicted gains

This file leaves the original voice_restore path untouched.
"""

import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchaudio

from voice_restore.model_v2 import (
    SR, N_FFT, HOP, N_FREQ,
    VoiceRestorerV2,
    apply_compensation,
    repair_region_from_mask,
)
from voice_restore.features_v2 import make_v2_inputs
from voice_restore import train as v1_train


PROJECT_ROOT = Path(__file__).parent.parent

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_SECS    = 2.0
SEQ_LEN     = int(SEQ_SECS * SR)
BATCH_SIZE  = 8
EPOCHS      = 200
LR          = 3e-4
GRAD_CLIP   = 1.0
N_STEPS     = 800
N_MELS      = 80

IDENTITY_W  = 0.25
SMOOTH_W    = 0.05


def make_mel_fb(device: torch.device) -> torch.Tensor:
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FREQ, f_min=80.0, f_max=float(SR // 2),
        n_mels=N_MELS, sample_rate=SR,
    )
    return fb.T.to(device)


def mel_compensation_loss(compensated_mag: torch.Tensor,
                          clean_mag:       torch.Tensor,
                          mel_fb:          torch.Tensor,
                          harm_template:   torch.Tensor) -> torch.Tensor:
    c_mel = torch.einsum('mf,bft->bmt', mel_fb, compensated_mag ** 2).clamp(1e-8)
    k_mel = torch.einsum('mf,bft->bmt', mel_fb, clean_mag ** 2).clamp(1e-8)

    log_c = torch.log(c_mel)
    log_k = torch.log(k_mel)

    voiced = (harm_template.mean(0) > 0.1).float().unsqueeze(0).unsqueeze(0)
    silence = (torch.log(clean_mag + 1e-8) > -10.0).float().mean(1, keepdim=True)
    w = (1.0 + voiced) * silence
    denom = (w.expand_as(log_c)).sum().clamp(1.0)
    return (w * (log_c - log_k) ** 2).sum() / denom


def identity_preservation_loss(comp_mag: torch.Tensor,
                               notched_mag: torch.Tensor,
                               mask_db_t: torch.Tensor) -> torch.Tensor:
    """
    Outside the repair neighborhood, leave the already-good signal alone.
    """
    repair = repair_region_from_mask(mask_db_t)
    preserve = (mask_db_t > -3.0).float() * (1.0 - repair)
    log_comp = torch.log(comp_mag + 1e-8)
    log_notched = torch.log(notched_mag + 1e-8)
    denom = preserve.sum().clamp(1.0)
    return ((log_comp - log_notched) ** 2 * preserve).sum() / denom


def temporal_smoothness_loss(gain: torch.Tensor,
                             mask_db_t: torch.Tensor) -> torch.Tensor:
    """
    Reduce frame-to-frame flicker in the restoration gains.
    """
    if gain.shape[-1] < 2:
        return gain.new_zeros(())
    repair = repair_region_from_mask(mask_db_t)
    delta = gain[:, :, 1:] - gain[:, :, :-1]
    weight = 0.25 + 0.75 * repair[:, :, 1:]
    return ((delta ** 2) * weight).mean()


def make_training_pair_v2(vocal_path: Path,
                          noise_np: np.ndarray,
                          device: torch.device,
                          window: torch.Tensor,
                          f0_cache: dict) -> tuple[torch.Tensor, ...] | None:
    """
    Returns (spectral, cond, mask_db_t, clean_mag, notched_mag) on device.
    """
    audio_np, sr = sf.read(str(vocal_path), dtype='float32')
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
    noisy = (vocal_np + noise_np[:SEQ_LEN] * (v_rms / n_rms * 10 ** (-snr_db / 20))
             ).astype(np.float32)

    def _stft_mag(x):
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        st = torch.stft(t, N_FFT, HOP, N_FFT, window, return_complex=True)
        return st.abs()[0]

    clean_mag = _stft_mag(noisy)
    T = clean_mag.shape[1]

    notches = v1_train.simulate_notch_bank(random.randint(1, v1_train.MAX_NOTCHES_SIM))
    notched_np, mask_db = v1_train.apply_notch_bank_to_audio(noisy, notches, T)
    notched_mag = _stft_mag(notched_np)

    key = str(vocal_path)
    if key not in f0_cache:
        f0_cache[key] = v1_train.extract_f0(vocal_path, device=str(device))
    f0_full, conf_full = f0_cache[key]
    frame_off = off // HOP
    f0_slice = f0_full[frame_off:frame_off + T + 10]
    conf_slice = conf_full[frame_off:frame_off + T + 10]

    mask_db_t = torch.from_numpy(mask_db).to(device).unsqueeze(0)
    spectral, cond = make_v2_inputs(
        notched_mag.unsqueeze(0),
        mask_db_t,
        f0_slice,
        conf_slice,
    )

    return spectral, cond, mask_db_t, clean_mag.unsqueeze(0), notched_mag.unsqueeze(0)


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--lr',     type=float, default=None)
    args, _ = ap.parse_known_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window   = torch.hann_window(N_FFT).sqrt().to(device)
    model    = VoiceRestorerV2().to(device)
    ckpt_dir = PROJECT_ROOT / 'checkpoints' / 'voice_restore_v2'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(str(ckpt_dir / 'tb'))

    optimizer = Adam(model.parameters(), lr=args.lr or LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=8, min_lr=1e-6)

    vocal_files = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                   if not f.name.startswith('._') and '__MACOSX' not in str(f)]
    noise_files = list((PROJECT_ROOT / 'data' / 'noise').rglob('*.wav'))
    assert vocal_files, 'No vocal files in data/clean_vocals/'
    assert noise_files, 'No noise files in data/noise/'

    vocal_files = [f for f in vocal_files
                   if (i := sf.info(str(f))).frames / i.samplerate >= SEQ_SECS]
    assert vocal_files, f'No vocal files >= {SEQ_SECS}s'

    print(f'VoiceRestorerV2: {model.n_params:,} params on {device}')
    print(f'Vocals: {len(vocal_files)}, noise: {len(noise_files)}')
    if not v1_train.CREPE_AVAILABLE:
        print('NOTE: CREPE unavailable — training without pitch features.')

    mel_fb    = make_mel_fb(device)
    best_loss = float('inf')
    f0_cache: dict = {}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
        model.load_state_dict(state['model'])
        best_loss = state.get('best_loss', float('inf'))
        print(f'Resumed from {args.resume}  (best_loss={best_loss:.4f})')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0
        optimizer.zero_grad()

        for step in tqdm(range(N_STEPS), desc=f'Epoch {epoch}/{EPOCHS}'):
            noise_np, _ = sf.read(str(random.choice(noise_files)), dtype='float32')
            if noise_np.ndim > 1:
                noise_np = noise_np.mean(1)
            if len(noise_np) < SEQ_LEN:
                noise_np = np.tile(noise_np, SEQ_LEN // len(noise_np) + 1)
            n0 = random.randint(0, len(noise_np) - SEQ_LEN)
            noise_np = noise_np[n0:n0 + SEQ_LEN]

            result = make_training_pair_v2(
                random.choice(vocal_files), noise_np, device, window, f0_cache
            )
            if result is None:
                continue

            spectral, cond, mask_db_t, clean_mag, notched_mag = result

            gain, _ = model(spectral, cond)
            comp_mag = apply_compensation(notched_mag, mask_db_t, gain)

            harm_t = spectral[0, 1]
            mel_loss = mel_compensation_loss(comp_mag, clean_mag, mel_fb, harm_t)
            id_loss = identity_preservation_loss(comp_mag, notched_mag, mask_db_t)
            smooth_loss = temporal_smoothness_loss(gain, mask_db_t)
            loss = mel_loss + IDENTITY_W * id_loss + SMOOTH_W * smooth_loss

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
