"""
voice_restore/eval_v4.py — Listen test for VoiceRestorer V4.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import lfilter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_restore.model_v4 import (  # noqa: E402
    HOP,
    N_FFT,
    SR,
    VoiceRestorerV4,
    apply_compensation,
    compute_effective_gain,
)
from voice_restore.features_v4 import make_v4_inputs  # noqa: E402
from voice_restore import train as v1_train  # noqa: E402

CKPT = PROJECT_ROOT / "checkpoints" / "voice_restore_v4" / "best.pt"


def apply_static_notch(audio_np: np.ndarray, freq: float,
                       depth_db: float, q: float) -> np.ndarray:
    w0 = 2.0 * np.pi * freq / SR
    alpha = np.sin(w0) / (2.0 * q)
    cosw = np.cos(w0)
    b0, b1, b2 = 1.0, -2.0 * cosw, 1.0
    a0, a1, a2 = 1.0 + alpha, -2.0 * cosw, 1.0 - alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
    dry_mix = 10.0 ** (depth_db / 20.0)
    filtered = lfilter(b, a, audio_np).astype(np.float32)
    return (1.0 - dry_mix) * filtered + dry_mix * audio_np


def extract_f0_direct(audio_np: np.ndarray, device: str = "cpu"):
    if not v1_train.CREPE_AVAILABLE:
        n_frames = 1 + max(0, len(audio_np) - N_FFT) // HOP
        return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)
    audio_t = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        f0, conf = v1_train.torchcrepe.predict(
            audio_t, SR, hop_length=HOP, fmin=50, fmax=2000, model="tiny",
            return_periodicity=True, device=device,
        )
    f0_np = f0[0].cpu().numpy().astype(np.float32)
    conf_np = conf[0].cpu().numpy().astype(np.float32)
    f0_np[conf_np < 0.5] = 0.0
    return f0_np, conf_np


def run_eval(input_path: str, notch_specs: list[tuple], ckpt_path: Path, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = torch.hann_window(N_FFT).sqrt().to(device)

    assert ckpt_path.exists(), f"No checkpoint at {ckpt_path} — train v4 first."
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = VoiceRestorerV4().to(device).eval()
    model.load_state_dict(state)
    print(f"Loaded VoiceRestorerV4 from {ckpt_path}")

    audio_np, sr = sf.read(input_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(1)
    assert sr == SR, f"Expected {SR} Hz, got {sr}"
    audio_np = v1_train.sosfilt(v1_train._hpf_sos, audio_np).astype(np.float32)
    print(f"Audio: {len(audio_np) / SR:.2f}s  ({len(audio_np)} samples)")

    notched_np = audio_np.copy()
    mask_db_static = np.zeros(N_FFT // 2 + 1, dtype=np.float32)
    for freq, depth_db, q in notch_specs:
        notched_np = apply_static_notch(notched_np, freq, depth_db, q)
        mask_db_static += v1_train.notch_frequency_response(freq, depth_db, q)
        print(f"  Notch: {freq:.0f} Hz  {depth_db:.0f} dB  Q={q:.1f}")
    mask_db_static = np.clip(mask_db_static, -96.0, 0.0)

    def _stft_complex(x_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(x_np).unsqueeze(0).to(device)
        return torch.stft(wav, N_FFT, HOP, N_FFT, window, return_complex=True)[0]

    clean_stft = _stft_complex(audio_np)
    notched_stft = _stft_complex(notched_np)
    clean_mag = clean_stft.abs().unsqueeze(0)
    notched_mag = notched_stft.abs().unsqueeze(0)
    t_frames = int(clean_mag.shape[-1])

    print("Extracting F0...")
    f0_np, conf_np = extract_f0_direct(audio_np, device=str(device))
    voiced_frac = float((f0_np > 0).mean()) if len(f0_np) else 0.0
    print(f"F0: voiced {voiced_frac * 100:.0f}% of frames")

    mask_db_t = torch.from_numpy(mask_db_static[:, None]).to(device).expand(-1, t_frames).unsqueeze(0)
    spectral, cond = make_v4_inputs(notched_mag, mask_db_t, f0_np, conf_np)

    with torch.no_grad():
        raw_gain, _ = model(spectral, cond)
    raw_gain = torch.nan_to_num(raw_gain, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    gain = compute_effective_gain(raw_gain, mask_db_t)

    comp_mag = apply_compensation(notched_mag, mask_db_t, raw_gain)[0]
    boost_db = 20.0 * torch.log10((comp_mag + 1e-8) / (notched_mag[0] + 1e-8))
    boosted = boost_db > 0.1
    pct_boosted = float(boosted.float().mean().item()) * 100.0
    print(
        f"Raw gain: min={float(raw_gain.min()):.3f}  max={float(raw_gain.max()):.3f}  "
        f"mean={float(raw_gain.mean()):.3f}"
    )
    print(
        f"Effective gain: min={float(gain.min()):.3f}  max={float(gain.max()):.3f}  "
        f"mean={float(gain.mean()):.3f}  | pct_bins_boosted={pct_boosted:.2f}%"
    )

    notched_phase = notched_stft / (notched_stft.abs() + 1e-8)
    restored_stft = comp_mag * notched_phase
    out_clean = audio_np
    out_notched = notched_np
    out_restored = torch.istft(restored_stft.unsqueeze(0), N_FFT, HOP, N_FFT, window)[0].cpu().numpy()

    length = min(len(out_clean), len(out_restored))
    out_clean = out_clean[:length]
    out_notched = out_notched[:length]
    out_restored = out_restored[:length]

    def _rms(x_np: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x_np ** 2))) + 1e-8

    out_restored = out_restored * (_rms(out_clean) / _rms(out_restored))

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "eval_clean.wav"), out_clean, SR, subtype="PCM_16")
    sf.write(str(out_dir / "eval_notched.wav"), out_notched, SR, subtype="PCM_16")
    sf.write(str(out_dir / "eval_restored.wav"), out_restored, SR, subtype="PCM_16")
    print(f"\nSaved to {out_dir}/")
    print("  eval_clean.wav    — pre-notch reference")
    print("  eval_notched.wav  — after notch")
    print("  eval_restored.wav — after VoiceRestorer V4")

    err_notched = out_clean - out_notched
    err_restored = out_clean - out_restored
    snr_notched = 10.0 * np.log10((_rms(out_clean) ** 2) / (_rms(err_notched) ** 2))
    snr_restored = 10.0 * np.log10((_rms(out_clean) ** 2) / (_rms(err_restored) ** 2))
    print(f"\nSNR vs clean:  notched={snr_notched:.1f} dB  restored={snr_restored:.1f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to clean vocal .wav (48kHz)")
    ap.add_argument("--notch", action="append", metavar="FREQ:DEPTH:Q")
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "eval_restore_v4"))
    args = ap.parse_args()

    if args.notch:
        notch_specs = []
        for spec in args.notch:
            freq, depth, q = spec.split(":")
            notch_specs.append((float(freq), float(depth), float(q)))
    else:
        import random
        random.seed(42)
        notch_specs = [
            (random.uniform(200, 600), random.uniform(-36, -18), random.uniform(10, 25)),
            (random.uniform(800, 2000), random.uniform(-36, -18), random.uniform(10, 25)),
            (random.uniform(2000, 6000), random.uniform(-36, -18), random.uniform(10, 25)),
        ]

    run_eval(args.input, notch_specs, Path(args.checkpoint), Path(args.out_dir))


if __name__ == "__main__":
    main()
