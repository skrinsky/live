"""
voice_restore/train.py — Train the VoiceRestorer.

Task: given a notched vocal signal + notch mask + pitch features,
      predict per-bin gain to restore the clean spectral envelope.

Training pair construction:
  1. Load clean vocal clip
  2. Extract F0 with CREPE-tiny (cached as .f0.npz alongside audio)
  3. Add noise at random SNR
  4. Simulate random notch bank (1–8 notches, frequencies/depths drawn from
     the same distribution the real NotchBank produces)
  5. Compute STFT of clean and notched audio
  6. Build harmonic templates from F0 trajectory
  7. Train: predict gain that recovers clean_mag from notched_mag

Loss:
  Weighted MSE on log-magnitude.
  - Full weight at voice harmonic bins (where accuracy matters most)
  - Down-weighted at deep notch bins (can't recover through -48 dB)
  - Down-weighted at near-silence bins

Usage:
    python voice_restore/train.py
    python voice_restore/train.py --resume checkpoints/voice_restore/best.pt
"""

import sys
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torchaudio
from voice_restore.model import (SR, N_FFT, HOP, N_FREQ,
                                  VoiceRestorer, harmonic_template,
                                  normalise_f0, apply_compensation)

# Stub resampy before importing torchcrepe — resampy→numba breaks on Python 3.13.
# torchcrepe only calls resampy.resample; replace with torchaudio (already required).
try:
    import sys as _sys, types as _types
    import torchaudio.functional as _taf
    _resampy = _types.ModuleType('resampy')
    _resampy.resample = lambda x, sr_orig, sr_target, **kw: _taf.resample(
        torch.from_numpy(x.copy()), int(sr_orig), int(sr_target)).numpy()
    _sys.modules.setdefault('resampy', _resampy)
    import torchcrepe
    CREPE_AVAILABLE = True
except Exception as _crepe_err:
    CREPE_AVAILABLE = False
    print(f'WARNING: torchcrepe unavailable ({type(_crepe_err).__name__}: {_crepe_err})')
    print('         F0 will be zero-initialised (no pitch information).')


# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_SECS    = 2.0
SEQ_LEN     = int(SEQ_SECS * SR)
BATCH_SIZE  = 8
EPOCHS      = 200
LR          = 3e-4
GRAD_CLIP   = 1.0
N_STEPS     = 800

# Notch simulation distribution
MAX_NOTCHES_SIM = 8
FREQ_RANGE      = (80.0, 18000.0)
DEPTH_RANGE_DB  = (-48.0, -12.0)
Q_RANGE         = (5.0, 30.0)

ATTACK_SAMPS  = int(0.020 * SR)   # 20 ms attack ramp
RELEASE_SAMPS = int(0.020 * SR)   # 20 ms release ramp

_hpf_sos = butter(2, 90.0 / (SR / 2), btype='high', output='sos')


# ── F0 extraction / caching ───────────────────────────────────────────────────

def extract_f0(audio_path: Path, device='cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 with CREPE-tiny, caching result as <audio_path>.f0.npz.

    Returns (f0_hz, confidence) both shape (n_frames,) aligned to HOP.
    If CREPE unavailable, returns zeros.
    """
    cache = audio_path.with_suffix('.f0.npz')
    if cache.exists():
        d = np.load(str(cache))
        return d['f0'], d['confidence']

    try:
        audio_np, sr = sf.read(str(audio_path), dtype='float32')
    except Exception:
        zeros = np.zeros(4, dtype=np.float32)
        return zeros, zeros

    if not CREPE_AVAILABLE:
        n_frames = int(len(audio_np) // HOP)
        zeros = np.zeros(n_frames, dtype=np.float32)
        return zeros, zeros

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(1)
    if sr != SR:
        return np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)

    audio_t = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        f0, confidence = torchcrepe.predict(
            audio_t, SR,
            hop_length=HOP,
            fmin=50, fmax=2000,
            model='tiny',
            return_periodicity=True,
            device=device,
        )
    f0_np   = f0[0].cpu().numpy().astype(np.float32)
    conf_np = confidence[0].cpu().numpy().astype(np.float32)
    f0_np[conf_np < 0.5] = 0.0

    np.savez(str(cache), f0=f0_np, confidence=conf_np)
    return f0_np, conf_np


# ── Notch simulation ──────────────────────────────────────────────────────────

def simulate_notch_bank(n_notches: int, sr: int = SR) -> list[dict]:
    """Draw random notch parameters including random onset/duration within the clip."""
    notches = []
    used_freqs = []
    for _ in range(n_notches):
        for _ in range(10):   # retry if too close to existing notch
            freq  = random.uniform(*FREQ_RANGE)
            if all(abs(freq - f) > 100.0 for f in used_freqs):
                break
        depth      = random.uniform(*DEPTH_RANGE_DB)
        q          = random.uniform(*Q_RANGE)
        onset_frac = random.uniform(0.0, 0.6)
        dur_frac   = random.uniform(0.15, max(0.16, 1.0 - onset_frac))
        notches.append({'freq': freq, 'depth_db': depth, 'q': q,
                        'onset_frac': onset_frac, 'dur_frac': dur_frac})
        used_freqs.append(freq)
    return notches


def make_depth_envelope(depth_db: float, onset_frac: float, dur_frac: float,
                         n_samples: int) -> np.ndarray:
    """
    Per-sample depth envelope for one notch (values in [depth_db, 0]).
    Ramps in over ATTACK_SAMPS, holds at depth_db, ramps out over RELEASE_SAMPS.
    """
    env    = np.zeros(n_samples, dtype=np.float32)
    onset  = int(onset_frac * n_samples)
    dur    = max(1, int(dur_frac * n_samples))
    end    = min(onset + dur, n_samples)
    seg    = end - onset
    if seg <= 0:
        return env
    atk = min(ATTACK_SAMPS, seg // 2)
    rel = min(RELEASE_SAMPS, seg // 2)
    atk_end   = onset + atk
    hold_end  = max(atk_end, end - rel)
    if atk_end > onset:
        env[onset:atk_end] = np.linspace(0.0, depth_db, atk_end - onset, dtype=np.float32)
    if hold_end > atk_end:
        env[atk_end:hold_end] = depth_db
    if end > hold_end:
        env[hold_end:end] = np.linspace(depth_db, 0.0, end - hold_end, dtype=np.float32)
    return env


def notch_frequency_response(freq_hz: float, depth_db: float,
                              q: float, sr: int = SR) -> np.ndarray:
    """
    Compute the per-bin gain (dB) of a single biquad notch.
    Returns (N_FREQ,) array ≤ 0.
    """
    from scipy.signal import freqz
    w0    = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cosw  = np.cos(w0)
    b0, b1, b2 = 1.0, -2.0 * cosw, 1.0
    a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    dry_mix = 10.0 ** (depth_db / 20.0)
    # Wet path frequency response
    _, h_wet = freqz(b, a, worN=N_FREQ, fs=sr)
    h_total = (1.0 - dry_mix) * h_wet + dry_mix   # wet/dry blend
    return 20.0 * np.log10(np.abs(h_total) + 1e-8).astype(np.float32)


def apply_notch_bank_to_audio(audio_np: np.ndarray,
                               notches: list[dict],
                               n_frames: int,
                               sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply notches with time-varying depth envelopes.

    Each notch has a random onset and duration within the clip, with 20 ms
    attack/release ramps, so the mask changes per frame — forcing the model
    to use pitch context rather than memorising a static notch pattern.

    Returns:
      notched_audio : (n_samples,) float32
      mask_db       : (N_FREQ, n_frames) float32, ≤ 0
    """
    from scipy.signal import lfilter
    n_samples = len(audio_np)
    notched   = audio_np.copy().astype(np.float32)
    mask_db   = np.zeros((N_FREQ, n_frames), dtype=np.float32)

    for n in notches:
        freq, depth_db, q = n['freq'], n['depth_db'], n['q']
        onset_frac, dur_frac = n['onset_frac'], n['dur_frac']

        # Biquad coefficients (fixed)
        w0    = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / (2.0 * q)
        cosw  = np.cos(w0)
        b0, b1, b2 = 1.0, -2.0 * cosw, 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha
        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])

        # Filter full signal (one pass, fixed coefficients)
        filtered = lfilter(b, a, notched).astype(np.float32)

        # Per-sample depth envelope → time-varying wet/dry blend
        depth_env = make_depth_envelope(depth_db, onset_frac, dur_frac, n_samples)
        dry_mix   = (10.0 ** (depth_env / 20.0)).astype(np.float32)
        notched   = (1.0 - dry_mix) * filtered + dry_mix * notched

        # Per-frame mask: scale full-depth frequency shape by envelope at each frame
        fr_shape = notch_frequency_response(freq, depth_db, q, sr)  # (N_FREQ,) ≤ 0
        for t in range(n_frames):
            s           = min(t * HOP, n_samples - 1)
            scale       = float(depth_env[s]) / depth_db if depth_db != 0 else 0.0
            mask_db[:, t] += fr_shape * scale

    return notched, np.clip(mask_db, -96.0, 0.0)


# ── Feature construction ──────────────────────────────────────────────────────

def build_pitch_features(f0_np: np.ndarray,
                          conf_np: np.ndarray,
                          n_frames: int) -> np.ndarray:
    """
    Build (4, n_frames) pitch feature array:
      [f0_norm, confidence, delta_f0_norm, delta2_f0_norm]

    Handles length mismatch between CREPE output and STFT frames.
    """
    # Trim / pad to n_frames
    def _fit(arr):
        if len(arr) >= n_frames:
            return arr[:n_frames]
        return np.pad(arr, (0, n_frames - len(arr)))

    f0_norm   = np.array([normalise_f0(f) for f in _fit(f0_np)], dtype=np.float32)
    conf      = _fit(conf_np).astype(np.float32)
    delta     = np.gradient(f0_norm)
    delta2    = np.gradient(delta)

    return np.stack([f0_norm, conf, delta, delta2], axis=0).astype(np.float32)   # (4, T)


def build_harmonic_features(f0_np: np.ndarray, n_frames: int) -> np.ndarray:
    """Build (N_FREQ, n_frames) harmonic template array from F0 trajectory."""
    def _fit(arr):
        if len(arr) >= n_frames:
            return arr[:n_frames]
        return np.pad(arr, (0, n_frames - len(arr)))

    f0_fit = _fit(f0_np)
    return np.stack([harmonic_template(float(f)) for f in f0_fit], axis=1)
    # → (N_FREQ, T)


# ── Training pair ─────────────────────────────────────────────────────────────

def make_training_pair(vocal_path: Path,
                        noise_np: np.ndarray,
                        device: torch.device,
                        window: torch.Tensor,
                        f0_cache: dict) -> tuple[torch.Tensor, ...] | None:
    """
    Returns (spectral, pitch_feats, notch_mask_t, clean_mag, notched_mag) on device,
    or None if the clip is too short.
    """
    # ── Load vocal ────────────────────────────────────────────────────────────
    audio_np, sr = sf.read(str(vocal_path), dtype='float32')
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(1)
    if sr != SR or len(audio_np) < SEQ_LEN:
        return None

    off      = random.randint(0, len(audio_np) - SEQ_LEN)
    vocal_np = audio_np[off:off + SEQ_LEN]
    vocal_np = sosfilt(_hpf_sos, vocal_np).astype(np.float32)

    # ── Mix with noise ────────────────────────────────────────────────────────
    snr_db  = random.uniform(10, 40)
    v_rms   = float(np.sqrt(np.mean(vocal_np ** 2))) + 1e-8
    n_rms   = float(np.sqrt(np.mean(noise_np[:SEQ_LEN] ** 2))) + 1e-8
    noisy   = (vocal_np + noise_np[:SEQ_LEN] * (v_rms / n_rms * 10 ** (-snr_db / 20))
               ).astype(np.float32)

    # ── STFT helper + get T from clean signal first ───────────────────────────
    def _stft_mag(x):
        t  = torch.from_numpy(x).unsqueeze(0).to(device)
        st = torch.stft(t, N_FFT, HOP, N_FFT, window, return_complex=True)
        return st.abs()[0]   # (N_FREQ, T)

    clean_mag = _stft_mag(noisy)
    T = clean_mag.shape[1]

    # ── Simulate notch bank (time-varying depths) ──────────────────────────────
    n_notches   = random.randint(1, MAX_NOTCHES_SIM)
    notches     = simulate_notch_bank(n_notches)
    notched_np, mask_db = apply_notch_bank_to_audio(noisy, notches, T)
    # mask_db: (N_FREQ, T) — depth varies per frame, model must use pitch

    notched_mag = _stft_mag(notched_np)

    # ── F0 / pitch features ───────────────────────────────────────────────────
    key = str(vocal_path)
    if key not in f0_cache:
        f0_cache[key] = extract_f0(vocal_path, device=str(device))
    f0_full, conf_full = f0_cache[key]

    # Slice to the same offset (in frames)
    frame_off  = off // HOP
    f0_slice   = f0_full[frame_off:frame_off + T + 10]   # slight overrun then fit
    conf_slice = conf_full[frame_off:frame_off + T + 10]

    pitch_np   = build_pitch_features(f0_slice, conf_slice, T)  # (4, T)
    harm_np    = build_harmonic_features(f0_slice, T)            # (N_FREQ, T)

    # ── Assemble spectral input ───────────────────────────────────────────────
    log_notched = torch.log(notched_mag + 1e-8)                  # (N_FREQ, T)
    harm_t      = torch.from_numpy(harm_np).to(device)

    # Mask is NOT a model input — the model must infer suppressed bins from
    # pitch context, not from an explicit mask channel. Mask is only used as
    # a hard ceiling in apply_restoration and for loss weighting.
    spectral = torch.stack([log_notched, harm_t], dim=0).unsqueeze(0)
    # → (1, 2, N_FREQ, T)

    pitch_t  = torch.from_numpy(pitch_np).to(device).unsqueeze(0)
    # → (1, 4, T)

    mask_db_t = torch.from_numpy(mask_db).to(device).unsqueeze(0)
    # → (1, N_FREQ, T) — actual dB values for loss weighting

    return spectral, pitch_t, mask_db_t, clean_mag.unsqueeze(0), notched_mag.unsqueeze(0)


# ── Loss ──────────────────────────────────────────────────────────────────────

N_MELS = 80

def make_mel_fb(device: torch.device) -> torch.Tensor:
    """Mel filterbank matrix (N_MELS, N_FREQ) on device."""
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FREQ, f_min=80.0, f_max=float(SR // 2),
        n_mels=N_MELS, sample_rate=SR,
    )  # (N_FREQ, N_MELS)
    return fb.T.to(device)  # (N_MELS, N_FREQ)


def mel_compensation_loss(compensated_mag: torch.Tensor,
                           clean_mag:       torch.Tensor,
                           mel_fb:          torch.Tensor,
                           harm_template:   torch.Tensor) -> torch.Tensor:
    """
    Log mel-energy MSE between compensated and clean signals.

    Using mel-scale bands instead of per-bin MSE means gradient from a
    notched bin flows to all un-notched bins in the same mel band — the
    mechanism that teaches "1 kHz cut → boost 900 Hz / 1.1 kHz."

    compensated_mag : (B, N_FREQ, T)
    clean_mag       : (B, N_FREQ, T)
    mel_fb          : (N_MELS, N_FREQ)
    harm_template   : (N_FREQ, T)  — for voiced-frame weighting
    """
    # Per-band energy  (B, N_MELS, T)
    c_mel = torch.einsum('mf,bft->bmt', mel_fb, compensated_mag ** 2).clamp(1e-8)
    k_mel = torch.einsum('mf,bft->bmt', mel_fb, clean_mag       ** 2).clamp(1e-8)

    log_c = torch.log(c_mel)
    log_k = torch.log(k_mel)

    # Weight voiced frames more (harmonic template active)
    voiced = (harm_template.mean(0) > 0.1).float()       # (T,)
    voiced = voiced.unsqueeze(0).unsqueeze(0)              # (1, 1, T)

    # Silence gate: drop frames where clean is very quiet
    silence = (torch.log(clean_mag + 1e-8) > -10.0).float().mean(1, keepdim=True)
    # → (B, 1, T)

    w     = (1.0 + voiced) * silence                      # (B, 1, T) broadcast over mels
    denom = (w.expand_as(log_c)).sum().clamp(1.0)
    return (w * (log_c - log_k) ** 2).sum() / denom


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--lr',     type=float, default=None)
    args, _ = ap.parse_known_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window   = torch.hann_window(N_FFT).sqrt().to(device)
    model    = VoiceRestorer().to(device)
    ckpt_dir = PROJECT_ROOT / 'checkpoints' / 'voice_restore'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(str(ckpt_dir / 'tb'))

    optimizer = Adam(model.parameters(), lr=args.lr or LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=8, min_lr=1e-6)

    # ── Load audio files ───────────────────────────────────────────────────────
    vocal_files = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                   if not f.name.startswith('._') and '__MACOSX' not in str(f)]
    noise_files = list((PROJECT_ROOT / 'data' / 'noise').rglob('*.wav'))
    assert vocal_files, 'No vocal files in data/clean_vocals/'
    assert noise_files, 'No noise files in data/noise/'

    vocal_files = [f for f in vocal_files
                   if (i := sf.info(str(f))).frames / i.samplerate >= SEQ_SECS]
    assert vocal_files, f'No vocal files >= {SEQ_SECS}s'

    n_params = model.n_params
    print(f'VoiceRestorer: {n_params:,} params on {device}')
    print(f'Vocals: {len(vocal_files)}, noise: {len(noise_files)}')
    if not CREPE_AVAILABLE:
        print('NOTE: CREPE unavailable — training without pitch features.')

    mel_fb    = make_mel_fb(device)   # (N_MELS, N_FREQ)
    best_loss = float('inf')
    f0_cache: dict = {}

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
            # ── Sample noise ───────────────────────────────────────────────────
            noise_np, _ = sf.read(str(random.choice(noise_files)), dtype='float32')
            if noise_np.ndim > 1:
                noise_np = noise_np.mean(1)
            if len(noise_np) < SEQ_LEN:
                noise_np = np.tile(noise_np, SEQ_LEN // len(noise_np) + 1)
            n0       = random.randint(0, len(noise_np) - SEQ_LEN)
            noise_np = noise_np[n0:n0 + SEQ_LEN]

            # ── Build training pair ────────────────────────────────────────────
            result = make_training_pair(
                random.choice(vocal_files), noise_np,
                device, window, f0_cache
            )
            if result is None:
                continue

            spectral, pitch_t, mask_db_t, clean_mag, notched_mag = result

            # ── Forward ────────────────────────────────────────────────────────
            gain, _ = model(spectral, pitch_t)   # (1, N_FREQ, T)

            comp_mag = apply_compensation(notched_mag, mask_db_t, gain)

            harm_t = spectral[0, 1]   # (N_FREQ, T) — channel 1 is harmonic_template

            loss = mel_compensation_loss(comp_mag, clean_mag, mel_fb, harm_t)

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
