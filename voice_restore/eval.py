"""
voice_restore/eval.py — Listen test for the VoiceRestorer.

Applies a simulated notch bank to a clean vocal, runs the restorer,
saves three wav files for A/B comparison:
  eval_clean.wav    — original (pre-notch)
  eval_notched.wav  — after notch bank (what you'd hear without restorer)
  eval_restored.wav — after restorer (what the model predicts)

Usage:
    python voice_restore/eval.py --input path/to/vocal.wav
    python voice_restore/eval.py --input path/to/vocal.wav \
        --notch 800:-24:20 --notch 2400:-18:15

    # notch format: freq_hz:depth_db:q  (depth_db should be negative)
    # if no --notch args, draws 3 random notches
"""

import sys
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.signal import lfilter, butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_restore.model import (SR, N_FFT, HOP, N_FREQ,
                                  VoiceRestorer, harmonic_template,
                                  normalise_f0, apply_restoration)
from voice_restore.train import (notch_frequency_response,
                                  build_pitch_features, build_harmonic_features,
                                  make_depth_envelope, ATTACK_SAMPS, RELEASE_SAMPS)

# Stub resampy → torchaudio before importing torchcrepe (same as train.py)
try:
    import types as _types, torchaudio.functional as _taf
    _resampy = _types.ModuleType('resampy')
    _resampy.resample = lambda x, sr_orig, sr_target, **kw: _taf.resample(
        torch.from_numpy(x.copy()), int(sr_orig), int(sr_target)).numpy()
    sys.modules.setdefault('resampy', _resampy)
    import torchcrepe
    CREPE_AVAILABLE = True
except Exception as e:
    CREPE_AVAILABLE = False
    print(f'WARNING: torchcrepe unavailable — F0 will be zero ({e})')

CKPT = PROJECT_ROOT / 'checkpoints' / 'voice_restore' / 'best.pt'
_hpf_sos = butter(2, 90.0 / (SR / 2), btype='high', output='sos')


def extract_f0_direct(audio_np: np.ndarray, device='cpu'):
    """Extract F0 directly from audio array (no caching)."""
    if not CREPE_AVAILABLE:
        n_frames = 1 + (len(audio_np) - N_FFT) // HOP
        return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)
    audio_t = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        f0, conf = torchcrepe.predict(audio_t, SR, hop_length=HOP,
                                       fmin=50, fmax=2000, model='tiny',
                                       return_periodicity=True, device=device)
    f0_np   = f0[0].cpu().numpy().astype(np.float32)
    conf_np = conf[0].cpu().numpy().astype(np.float32)
    f0_np[conf_np < 0.5] = 0.0
    return f0_np, conf_np


def apply_static_notch(audio_np: np.ndarray, freq: float,
                        depth_db: float, q: float) -> np.ndarray:
    """Apply a single biquad notch to audio, return notched audio."""
    w0    = 2.0 * np.pi * freq / SR
    alpha = np.sin(w0) / (2.0 * q)
    cosw  = np.cos(w0)
    b0, b1, b2 = 1.0, -2.0 * cosw, 1.0
    a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    dry_mix  = 10.0 ** (depth_db / 20.0)
    filtered = lfilter(b, a, audio_np).astype(np.float32)
    return (1.0 - dry_mix) * filtered + dry_mix * audio_np


def run_eval(input_path: str, notch_specs: list[tuple], ckpt_path: Path, out_dir: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    assert ckpt_path.exists(), f'No checkpoint at {ckpt_path} — train first.'
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model = VoiceRestorer().to(device).eval()
    model.load_state_dict(state)
    print(f'Loaded VoiceRestorer from {ckpt_path}')

    # ── Load audio ────────────────────────────────────────────────────────────
    audio_np, sr = sf.read(input_path, dtype='float32')
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(1)
    assert sr == SR, f'Expected {SR} Hz, got {sr}'
    audio_np = sosfilt(_hpf_sos, audio_np).astype(np.float32)
    print(f'Audio: {len(audio_np)/SR:.2f}s  ({len(audio_np)} samples)')

    # ── Apply notches (static, full duration) ────────────────────────────────
    notched_np = audio_np.copy()
    mask_db_static = np.zeros(N_FREQ, dtype=np.float32)
    for freq, depth_db, q in notch_specs:
        notched_np     = apply_static_notch(notched_np, freq, depth_db, q)
        mask_db_static += notch_frequency_response(freq, depth_db, q)
        print(f'  Notch: {freq:.0f} Hz  {depth_db:.0f} dB  Q={q:.1f}')
    mask_db_static = np.clip(mask_db_static, -96.0, 0.0)

    # ── STFT ──────────────────────────────────────────────────────────────────
    def _stft_mag(x):
        t  = torch.from_numpy(x).unsqueeze(0).to(device)
        st = torch.stft(t, N_FFT, HOP, N_FFT, window, return_complex=True)
        return st[0]   # complex (N_FREQ, T)

    clean_stft   = _stft_mag(audio_np)
    notched_stft = _stft_mag(notched_np)
    T = clean_stft.shape[1]

    clean_mag   = clean_stft.abs()
    notched_mag = notched_stft.abs()

    # ── F0 / pitch features ───────────────────────────────────────────────────
    print('Extracting F0...')
    f0_np, conf_np = extract_f0_direct(audio_np, device=str(device))
    pitch_np = build_pitch_features(f0_np, conf_np, T)       # (4, T)
    harm_np  = build_harmonic_features(f0_np, T)              # (N_FREQ, T)

    voiced_frac = float((f0_np > 0).mean())
    print(f'F0: voiced {voiced_frac*100:.0f}% of frames, '
          f'median={float(np.median(f0_np[f0_np>0])) if voiced_frac>0 else 0:.0f} Hz')

    # ── Assemble model input ──────────────────────────────────────────────────
    log_notched = torch.log(notched_mag + 1e-8)
    harm_t      = torch.from_numpy(harm_np).to(device)
    spectral    = torch.stack([log_notched, harm_t], dim=0).unsqueeze(0)  # (1,2,F,T)
    pitch_t     = torch.from_numpy(pitch_np).to(device).unsqueeze(0)      # (1,4,T)

    # Static mask broadcast to (1, N_FREQ, T)
    mask_db_t = (torch.from_numpy(mask_db_static[:, None]).to(device)
                 .expand(-1, T).unsqueeze(0))

    # ── Run model ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        gain, _ = model(spectral, pitch_t)   # (1, N_FREQ, T)

    restored_mag = apply_restoration(notched_mag.unsqueeze(0), mask_db_t, gain)[0]
    # → (N_FREQ, T)

    print(f'Gain stats: min={gain.min():.3f}  max={gain.max():.3f}  '
          f'mean={gain.mean():.3f}')

    # ── Reconstruct time-domain via Griffin-Lim ───────────────────────────────
    # Use notched phase (best available — avoids phase hallucination)
    notched_phase = notched_stft / (notched_mag + 1e-8)
    restored_stft = restored_mag * notched_phase

    def _istft(stft_c):
        return torch.istft(stft_c.unsqueeze(0), N_FFT, HOP, N_FFT, window)[0].cpu().numpy()

    out_clean    = audio_np
    out_notched  = notched_np
    out_restored = _istft(restored_stft)

    # Trim to original length
    L = len(audio_np)
    out_restored = out_restored[:L]

    # Loudness-match to input
    def _rms(x): return float(np.sqrt(np.mean(x**2))) + 1e-8
    out_restored = out_restored * (_rms(out_clean) / _rms(out_restored))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / 'eval_clean.wav'),    out_clean,    SR, subtype='PCM_16')
    sf.write(str(out_dir / 'eval_notched.wav'),  out_notched,  SR, subtype='PCM_16')
    sf.write(str(out_dir / 'eval_restored.wav'), out_restored, SR, subtype='PCM_16')
    print(f'\nSaved to {out_dir}/:')
    print('  eval_clean.wav    — pre-notch reference')
    print('  eval_notched.wav  — after notch (what you hear without restorer)')
    print('  eval_restored.wav — after restorer')

    # ── Quick SNR estimate ─────────────────────────────────────────────────────
    err_notch    = out_clean - out_notched[:L]
    err_restored = out_clean - out_restored
    snr_notch    = 10*np.log10(_rms(out_clean)**2 / _rms(err_notch)**2)
    snr_restored = 10*np.log10(_rms(out_clean)**2 / _rms(err_restored)**2)
    print(f'\nSNR vs clean:  notched={snr_notch:.1f} dB  restored={snr_restored:.1f} dB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',    required=True,  help='Path to clean vocal .wav (48kHz)')
    ap.add_argument('--notch',    action='append', metavar='FREQ:DEPTH:Q',
                    help='Notch spec e.g. 800:-24:20  (can repeat)')
    ap.add_argument('--checkpoint', default=str(CKPT))
    ap.add_argument('--out-dir',    default=str(PROJECT_ROOT / 'data' / 'eval_restore'))
    args = ap.parse_args()

    if args.notch:
        notch_specs = []
        for s in args.notch:
            parts = s.split(':')
            notch_specs.append((float(parts[0]), float(parts[1]), float(parts[2])))
    else:
        # Default: 3 notches at typical feedback frequencies
        import random
        random.seed(42)
        notch_specs = [
            (random.uniform(200, 600),   random.uniform(-36, -18), random.uniform(10, 25)),
            (random.uniform(800, 2000),  random.uniform(-36, -18), random.uniform(10, 25)),
            (random.uniform(2000, 6000), random.uniform(-36, -18), random.uniform(10, 25)),
        ]

    run_eval(args.input, notch_specs, Path(args.checkpoint), Path(args.out_dir))


if __name__ == '__main__':
    main()
