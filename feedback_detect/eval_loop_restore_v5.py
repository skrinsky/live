"""
feedback_detect/eval_loop_restore_v5.py — Closed-loop simulation with detector+notch+VoiceRestorerV5.

Runs three paths on the same audio + feedback IR:
  A) clean_reference.wav       — voice + noise, no loop (target)
  B) suppressed_out.wav        — detector+NotchBank in loop, speaker feed
  C) suppressed_out_restored.wav — same but with V5 restorer applied post-loop

Usage:
    python feedback_detect/eval_loop_restore_v5.py --gain 1.3 --duration 60
    python feedback_detect/eval_loop_restore_v5.py --gain 1.5 --duration 60
"""

import sys
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'feedback_detect'))
sys.path.insert(0, str(PROJECT_ROOT))

from feedback_detect.model import FeedbackDetector, SR, N_FFT, HOP, N_FREQ
from feedback_detect.notch import NotchBank
from feedback_detect.predictor import FeedbackPredictor
from feedback_detect.live import _cluster_bins
from feedback_detect.eval_loop import _rms_db
from feedback_detect.spectral_flatten import SpectralFlattener

from voice_restore.model_v5 import VoiceRestorerV5, apply_compensation
from voice_restore.model_v5 import N_FREQ as VR_N_FREQ, N_FFT as VR_N_FFT
VR_HOP = VR_N_FFT // 2   # 512 — 50% overlap required for perfect OLA with sqrt-Hann window
from voice_restore.features_v5 import make_v5_inputs
from voice_restore import train as vr_train

CHECKPOINT_DET = PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'
CHECKPOINT_RES = PROJECT_ROOT / 'checkpoints' / 'voice_restore_v5' / 'best.pt'
_hpf_sos   = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
_bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)


def simulate_notch(voice_np, noise_np, feedback_ir, gain,
                   model_det, notch_bank, device, window, threshold,
                   predictor=None, flattener=None):
    ir  = (feedback_ir * gain).astype(np.float32)
    n   = len(voice_np)
    acc = np.zeros(n + len(ir), dtype=np.float64)

    mic_out      = np.zeros(n, dtype=np.float32)
    box_out      = np.zeros(n, dtype=np.float32)
    flat_out     = np.zeros(n, dtype=np.float32)
    gru_h        = None
    lm_history   = np.zeros((N_FREQ, 11), dtype=np.float32)
    analysis_buf = np.zeros(N_FFT, dtype=np.float32)
    notch_logs   = []

    for i in range(n // HOP):
        s = i * HOP
        mic_block = (voice_np[s:s+HOP].astype(np.float64) +
                     noise_np[s:s+HOP].astype(np.float64) +
                     acc[s:s+HOP])
        mic_block = np.clip(mic_block, -1.0, 1.0).astype(np.float32)
        mic_out[s:s+HOP] = mic_block

        analysis_buf = np.roll(analysis_buf, -HOP)
        analysis_buf[-HOP:] = mic_block

        buf_t = torch.from_numpy(analysis_buf).unsqueeze(0).to(device)
        stft  = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window, center=False, return_complex=True)
        mag   = stft.abs()

        lm_now            = torch.log(mag[0, :, 0] + 1e-8).cpu().numpy()
        lm_history        = np.roll(lm_history, 1, axis=1)
        lm_history[:, 0]  = lm_now

        feat_np  = np.stack([
            lm_history[:, 0],
            lm_history[:, 0] - lm_history[:, 1],
            lm_history[:, 0] - lm_history[:, 4],
            lm_history[:, 0] - lm_history[:, 10],
        ], axis=0)
        features = torch.from_numpy(feat_np).to(device).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            prob, gru_h = model_det(features, gru_h)

        prob_np = prob[0, :, 0].cpu().numpy()
        above   = (prob_np > threshold) & (_bin_freqs >= 80.0) & (_bin_freqs < SR / 2)
        freqs   = _cluster_bins(_bin_freqs, prob_np, above)

        stft_mag_np = mag[0, :, 0].cpu().numpy()
        preemptive = predictor.update(
            stft_mag_np, prob_np, notch_bank.active_notches
        ) if predictor else []
        notch_bank.update(freqs, _bin_freqs, prob_np, preemptive_freqs=preemptive)
        out_block = notch_bank.process(mic_block)
        box_out[s:s+HOP] = out_block
        notch_logs.append(list(notch_bank.active_notches))

        # SpectralFlattener: adaptive wide-Q coloration correction
        if flattener is not None:
            rms = float(np.sqrt(np.mean(out_block ** 2) + 1e-20))
            flattener.update(stft_mag_np, rms, notch_bank.active_notches)
            flat_block = flattener.process(out_block)
        else:
            flat_block = out_block
        flat_out[s:s+HOP] = flat_block

        fb = np.convolve(flat_block.astype(np.float64), ir.astype(np.float64))
        acc[s:s + len(fb)] += fb

    return mic_out[:n], box_out[:n], flat_out[:n], notch_logs


def build_notch_mask_from_logs(notch_logs, n_frames):
    mask = np.zeros((VR_N_FREQ, n_frames), dtype=np.float32)
    for t, notches in enumerate(notch_logs[:n_frames]):
        cur = np.zeros(VR_N_FREQ, dtype=np.float32)
        for f, d, q in notches:
            cur += vr_train.notch_frequency_response(f, d, q)
        mask[:, t] = np.clip(cur, -96.0, 0.0)
    return mask


def run_eval(gain=1.3, duration_s=60.0, threshold=0.4,
             det_ckpt=None, res_ckpt=None, out_dir=None):
    det_path = Path(det_ckpt or CHECKPOINT_DET)
    res_path = Path(res_ckpt or CHECKPOINT_RES)
    assert det_path.exists(), f'No detector checkpoint at {det_path}'
    assert res_path.exists(), f'No restorer checkpoint at {res_path}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model_det = FeedbackDetector().to(device).eval()
    ckpt_det  = torch.load(str(det_path), map_location=device)
    model_det.load_state_dict(
        ckpt_det['model'] if isinstance(ckpt_det, dict) and 'model' in ckpt_det else ckpt_det)
    print(f'FeedbackDetector loaded from {det_path}')

    model_res = VoiceRestorerV5().cpu().eval()
    ckpt_res  = torch.load(str(res_path), map_location='cpu')
    model_res.load_state_dict(
        ckpt_res['model'] if isinstance(ckpt_res, dict) and 'model' in ckpt_res else ckpt_res)
    print(f'VoiceRestorerV5 loaded from {res_path}')

    out_dir = Path(out_dir or PROJECT_ROOT / 'data' / 'eval_loop_restore_v5')
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(duration_s * SR)

    # ── Load IR ────────────────────────────────────────────────────────────────
    ir_pool       = PROJECT_ROOT / 'data' / 'ir_pool'
    mains_files   = sorted(ir_pool.glob('mains_*.wav'))
    monitor_files = sorted(ir_pool.glob('monitor_*.wav'))
    assert mains_files and monitor_files, 'No IRs found in data/ir_pool/.'

    mains_np,   _ = sf.read(str(mains_files[0]),   dtype='float32')
    monitor_np, _ = sf.read(str(monitor_files[0]), dtype='float32')
    if mains_np.ndim   > 1: mains_np   = mains_np.sum(1)
    if monitor_np.ndim > 1: monitor_np = monitor_np.sum(1)

    trunc = min(len(mains_np), len(monitor_np), int(0.05 * SR))
    ir    = (mains_np[:trunc] + monitor_np[:trunc]).astype(np.float32)
    peak  = np.abs(np.fft.rfft(ir, n=max(len(ir) * 4, 4096))).max()
    ir   /= (peak + 1e-8)

    # ── Load vocal ─────────────────────────────────────────────────────────────
    vocal_files = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                   if not f.name.startswith('._') and '__MACOSX' not in str(f)]
    vocal_path_used = None
    if vocal_files:
        vocal_path_used = vocal_files[0]
        v, vsr = sf.read(str(vocal_path_used), dtype='float32')
        if v.ndim > 1: v = v.mean(1)
        if vsr != SR:
            raise ValueError(f'Vocal SR={vsr}, expected {SR}')
        v = sosfilt(_hpf_sos, v).astype(np.float32)
        print(f'  vocal file: {vocal_path_used.name}  ({len(v)/SR:.1f}s)')
        if len(v) < n:
            # Crossfade tile boundaries to avoid phase discontinuities
            fade = min(int(0.05 * SR), len(v) // 8)   # 50ms fade
            v_fade = v.copy()
            v_fade[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            v_fade[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            v = np.tile(v_fade, n // len(v) + 1)
        voice_np = v[:n]
    else:
        t = np.linspace(0, duration_s, n, dtype=np.float32)
        voice_np = (0.1 * np.sin(2 * np.pi * (200 + 800 * t / duration_s) * t)).astype(np.float32)

    noise_np = (np.random.randn(n) * 1e-4).astype(np.float32)
    clean_np = np.clip(sosfilt(_hpf_sos, voice_np + noise_np).astype(np.float32), -1.0, 1.0)

    print(f'\nGain={gain}  duration={duration_s}s  IR_len={len(ir)} samples')

    # ── Run detector + notch loop ──────────────────────────────────────────────
    print('Running detector + NotchBank loop…')
    profile_path = PROJECT_ROOT / 'data' / 'feedback_risk_profile.json'
    predictor  = FeedbackPredictor(_bin_freqs, sr=SR, profile_path=profile_path)
    predictor.seed_from_ir(ir, gain=gain)
    notch_bank = NotchBank(sr=SR, q=15.0, depth_db=-48.0)
    flattener  = SpectralFlattener(_bin_freqs, sr=SR)
    mic_sup, box_sup, flat_sup, notch_logs = simulate_notch(
        voice_np, noise_np, ir, gain=gain,
        model_det=model_det, notch_bank=notch_bank,
        device=device, window=window, threshold=threshold,
        predictor=predictor, flattener=flattener)
    predictor.save()
    print(predictor.summary())
    print(f'  mic RMS: {_rms_db(mic_sup):.1f} dB')

    # ── Build notch mask from logs ─────────────────────────────────────────────
    # Use voice_restore FFT params (N_FFT=1024, HOP=480) — restorer expects 513 bins
    window_res   = torch.hann_window(VR_N_FFT).sqrt()
    notched_t    = torch.from_numpy(box_sup).unsqueeze(0)
    notched_stft = torch.stft(notched_t, VR_N_FFT, VR_HOP, VR_N_FFT, window_res, center=True, return_complex=True)
    notched_mag  = notched_stft.abs()
    T_stft       = int(notched_mag.shape[-1])

    mask_db      = build_notch_mask_from_logs(notch_logs, T_stft)
    mask_db_t    = torch.from_numpy(mask_db).unsqueeze(0)   # (1, N_FREQ, T)

    # Align
    T_final      = min(T_stft, mask_db_t.shape[-1])
    mask_db_t    = mask_db_t[..., :T_final]
    notched_mag  = notched_mag[..., :T_final]
    notched_stft = notched_stft[..., :T_final]

    # ── F0 ─────────────────────────────────────────────────────────────────────
    try:
        if vocal_path_used is not None:
            f0_np, conf_np = vr_train.extract_f0(vocal_path_used, device='cpu')
        else:
            raise RuntimeError('no vocal path')
    except Exception:
        f0_np  = np.zeros(T_final, dtype=np.float32)
        conf_np = np.zeros(T_final, dtype=np.float32)

    # ── Run V5 restorer ────────────────────────────────────────────────────────
    print('Running VoiceRestorerV5…')
    spectral, cond = make_v5_inputs(notched_mag, mask_db_t, f0_np, conf_np)
    spectral = torch.nan_to_num(spectral, nan=0.0, posinf=0.0, neginf=0.0)
    cond     = torch.nan_to_num(cond,     nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        raw_residual, _ = model_res(spectral, cond)
    raw_residual = torch.nan_to_num(raw_residual, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    comp_mag, base_gain, eff_gain = apply_compensation(notched_mag, mask_db_t, raw_residual)
    comp_mag = torch.nan_to_num(comp_mag, nan=0.0, posinf=0.0, neginf=0.0)

    boost_db      = 20.0 * torch.log10((comp_mag + 1e-8) / (notched_mag + 1e-8))
    pct_boosted   = float((boost_db > 0.1).float().mean()) * 100
    max_boost     = float(boost_db.max())
    mean_boost    = float(boost_db[boost_db > 0.1].mean()) if pct_boosted > 0 else 0.0

    print(f'  raw_residual: min={float(raw_residual.min()):.3f} '
          f'max={float(raw_residual.max()):.3f} mean={float(raw_residual.mean()):.3f}')
    print(f'  eff_gain:     min={float(eff_gain.min()):.3f} '
          f'max={float(eff_gain.max()):.3f} mean={float(eff_gain.mean()):.3f}')
    print(f'  boost: {pct_boosted:.1f}% bins > 0.1 dB | max={max_boost:.2f} dB | mean={mean_boost:.2f} dB')

    # ── Reconstruct audio ──────────────────────────────────────────────────────
    notched_phase = notched_stft / (notched_mag + 1e-8)
    restored_stft = comp_mag * notched_phase
    restored_stft = torch.where(torch.isfinite(restored_stft),
                                restored_stft, torch.zeros_like(restored_stft))
    restored_wav  = torch.istft(restored_stft, VR_N_FFT, VR_HOP, VR_N_FFT, window_res, center=True)[0].numpy()

    L = min(len(box_sup), len(restored_wav))

    # ── Save ───────────────────────────────────────────────────────────────────
    L_flat = min(len(flat_sup), len(box_sup))
    sf.write(str(out_dir / 'clean_reference.wav'),         clean_np,           SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_out.wav'),          box_sup[:L],        SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_out_flattened.wav'), flat_sup[:L_flat], SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_out_restored.wav'),  restored_wav[:L],  SR, subtype='PCM_16')

    print(f'\nSpectralFlattener: {flattener.summary()}')
    print(f'\nRMS dB — clean {_rms_db(clean_np):.1f} | '
          f'suppressed {_rms_db(box_sup[:L]):.1f} | '
          f'flattened {_rms_db(flat_sup[:L_flat]):.1f} | '
          f'restored {_rms_db(restored_wav[:L]):.1f}')
    print(f'\nSaved to {out_dir}/:')
    print('  clean_reference.wav           — voice with no loop (target)')
    print('  suppressed_out.wav            — speaker feed after notch (no restorer)')
    print('  suppressed_out_flattened.wav  — notch + SpectralFlattener (coloration fix)')
    print('  suppressed_out_restored.wav   — notch + V5 restorer')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gain',      type=float, default=1.3)
    p.add_argument('--duration',  type=float, default=60.0)
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--detector',  type=str,   default=None)
    p.add_argument('--restorer',  type=str,   default=None)
    p.add_argument('--out-dir',   type=str,   default=None)
    args = p.parse_args()
    run_eval(gain=args.gain, duration_s=args.duration, threshold=args.threshold,
             det_ckpt=args.detector, res_ckpt=args.restorer, out_dir=args.out_dir)
