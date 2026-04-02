"""
feedback_detect/eval_loop_restore_v2.py — Closed-loop simulation with detector+notch+VoiceRestorerV2.

Runs three paths on the same audio + feedback IR:
  A) Raw:        voice + noise -> loop (no processing)
  B) Notched:    voice + noise -> detector+NotchBank in loop
  C) Restored:   voice + noise -> detector+NotchBank in loop -> VoiceRestorerV2 out-of-loop (post)

Note: VoiceRestorerV2 is applied after the loop to avoid destabilizing the loop;
      it does not change the feedback path, only the listener output.
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
from feedback_detect.live import _cluster_bins
from feedback_detect.eval_loop import _rms_db

from voice_restore.model_v2 import VoiceRestorerV2, apply_compensation
from voice_restore.features_v2 import make_v2_inputs
from voice_restore import train as vr_train

CHECKPOINT_DET = PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'
CHECKPOINT_RES = PROJECT_ROOT / 'checkpoints' / 'voice_restore_v2' / 'best.pt'
_hpf_sos   = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
_bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)


def simulate_notch(voice_np, noise_np, feedback_ir, gain,
                   model_det, notch_bank, device, window, threshold):
    ir  = (feedback_ir * gain).astype(np.float32)
    n   = len(voice_np)
    acc = np.zeros(n + len(ir), dtype=np.float64)

    mic_out      = np.zeros(n, dtype=np.float32)
    box_out      = np.zeros(n, dtype=np.float32)
    gru_h        = None
    lm_history   = np.zeros((N_FREQ, 11), dtype=np.float32)
    analysis_buf = np.zeros(N_FFT, dtype=np.float32)

    notch_logs = []

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
        stft  = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window, return_complex=True)
        mag   = stft.abs()

        lm_now          = torch.log(mag[0, :, 0] + 1e-8).cpu().numpy()
        lm_history      = np.roll(lm_history, 1, axis=1)
        lm_history[:, 0] = lm_now

        feat_np  = np.stack([
            lm_history[:, 0],
            lm_history[:, 0] - lm_history[:, 1],
            lm_history[:, 0] - lm_history[:, 4],
            lm_history[:, 0] - lm_history[:, 10],
        ], axis=0)
        features = (torch.from_numpy(feat_np).to(device)
                        .unsqueeze(0).unsqueeze(-1))

        with torch.no_grad():
            prob, gru_h = model_det(features, gru_h)

        prob_np = prob[0, :, 0].cpu().numpy()
        above   = (prob_np > threshold) & (_bin_freqs >= 80.0) & (_bin_freqs < SR / 2)
        freqs   = _cluster_bins(_bin_freqs, prob_np, above)

        notch_bank.update(freqs, _bin_freqs, prob_np)
        out_block = notch_bank.process(mic_block)
        box_out[s:s+HOP] = out_block
        notch_logs.append(list(notch_bank.active_notches))

        fb = np.convolve(out_block.astype(np.float64), ir.astype(np.float64))
        acc[s:s + len(fb)] += fb

    return mic_out[:n], box_out[:n], notch_logs


def build_notch_mask_from_logs(notch_logs, n_frames, n_freqs=512):
    mask = np.zeros((n_freqs + 1, n_frames), dtype=np.float32)
    for t, notches in enumerate(notch_logs):
        cur = np.zeros(n_freqs + 1, dtype=np.float32)
        for f, d, q in notches:
            cur += vr_train.notch_frequency_response(f, d, q)
        cur = np.clip(cur, -96.0, 0.0)
        mask[:, t] = cur
    return mask


def run_eval(gain=1.3, duration_s=30.0, threshold=0.4,
             det_ckpt=None, res_ckpt=None, out_dir=None):
    det_path = Path(det_ckpt or CHECKPOINT_DET)
    res_path = Path(res_ckpt or CHECKPOINT_RES)
    assert det_path.exists(), f'No detector checkpoint at {det_path}'
    assert res_path.exists(), f'No restorer checkpoint at {res_path}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)

    model_det = FeedbackDetector().to(device).eval()
    ckpt_det  = torch.load(str(det_path), map_location=device)
    model_det.load_state_dict(ckpt_det['model'] if isinstance(ckpt_det, dict) and 'model' in ckpt_det else ckpt_det)

    model_res = VoiceRestorerV2().to(device).eval()
    ckpt_res  = torch.load(str(res_path), map_location=device)
    model_res.load_state_dict(ckpt_res['model'] if isinstance(ckpt_res, dict) and 'model' in ckpt_res else ckpt_res)

    out_dir = Path(out_dir or PROJECT_ROOT / 'data' / 'eval_loop_restore_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(duration_s * SR)

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

    vocal_files = [f for f in (PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav')
                   if not f.name.startswith('._')]
    if vocal_files:
        v, vsr = sf.read(str(vocal_files[0]), dtype='float32')
        if v.ndim > 1: v = v.mean(1)
        if vsr != SR:
            raise ValueError(f'Vocal SR={vsr}, expected {SR}')
        v = sosfilt(_hpf_sos, v).astype(np.float32)
        if len(v) < n:
            v = np.tile(v, n // len(v) + 1)
        voice_np = v[:n]
    else:
        t = np.linspace(0, duration_s, n, dtype=np.float32)
        voice_np = (0.1 * np.sin(2 * np.pi * (200 + 800 * t / duration_s) * t)).astype(np.float32)

    noise_np = (np.random.randn(n) * 1e-4).astype(np.float32)

    clean_np = sosfilt(_hpf_sos, (voice_np + noise_np)).astype(np.float32)
    clean_np = np.clip(clean_np, -1.0, 1.0)

    print(f'\nGain={gain}  duration={duration_s}s  IR_len={len(ir)} samples')
    print(f'{"":─<60}')

    print('Running A: raw feedback (no model)…')
    mic_raw, box_raw = vr_train.simulate(voice_np, noise_np, ir, gain=gain,
                                         model=None, notch_bank=None,
                                         device=device, window=window,
                                         threshold=threshold) if hasattr(vr_train, 'simulate') else (None, None)

    print('Running B/C: detector + notch (and restorer post-loop)…')
    notch_bank = NotchBank(sr=SR, q=15.0, depth_db=-48.0)
    mic_sup, box_sup, notch_logs = simulate_notch(
        voice_np, noise_np, ir, gain=gain,
        model_det=model_det, notch_bank=notch_bank,
        device=device, window=window, threshold=threshold)
    rms_sup = _rms_db(mic_sup)
    print(f'  mic RMS: {rms_sup:.1f} dB')

    # Build notch mask for restorer from logs
    n_frames = len(mic_sup) // HOP
    mask_db = build_notch_mask_from_logs(notch_logs, n_frames)
    mask_db_t = torch.from_numpy(mask_db).to(device).unsqueeze(0)

    window_res = torch.hann_window(1024).sqrt().to(device)
    notched_t = torch.from_numpy(box_sup).unsqueeze(0).to(device)
    notched_stft = torch.stft(notched_t, 1024, 480, 1024, window_res, return_complex=True)
    notched_mag = notched_stft.abs()

    # Align time dimension between notch mask and STFT
    T_final = min(notched_mag.shape[-1], mask_db_t.shape[-1])
    mask_db_t = mask_db_t[..., :T_final]
    notched_mag = notched_mag[..., :T_final]
    notched_stft = notched_stft[..., :T_final]

    # F0 for conditioning (fallback zeros on failure)
    try:
        f0_np, conf_np = vr_train.extract_f0(Path('temp.wav'), device=str(device))
    except Exception:
        T = mask_db.shape[1]
        f0_np = np.zeros(T, dtype=np.float32)
        conf_np = np.zeros(T, dtype=np.float32)

    spectral, cond = make_v2_inputs(notched_mag, mask_db_t, f0_np, conf_np)
    with torch.no_grad():
        gain, _ = model_res(spectral, cond)
    comp_mag = apply_compensation(notched_mag, mask_db_t, gain)[0]

    notched_phase = notched_stft / (notched_mag + 1e-8)
    restored_stft = comp_mag * notched_phase            # (1, F, T)
    # istft expects (batch, freq, time) or (freq, time); comp_mag is (1,F,T)
    restored_wav = torch.istft(restored_stft, 1024, 480, 1024, window_res)[0].cpu().numpy()
    L = min(len(box_sup), len(restored_wav))
    out_restored = restored_wav[:L]

    sf.write(str(out_dir / 'clean_reference.wav'), clean_np, SR, subtype='PCM_16')
    if mic_raw is not None:
        sf.write(str(out_dir / 'raw_feedback.wav'), mic_raw, SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_mic.wav'), mic_sup, SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_out.wav'), box_sup[:L], SR, subtype='PCM_16')
    sf.write(str(out_dir / 'suppressed_out_restored.wav'), out_restored, SR, subtype='PCM_16')

    print(f'RMS dB — clean { _rms_db(clean_np):.1f} | suppressed_out { _rms_db(box_sup[:L]):.1f} '
          f'| restored { _rms_db(out_restored):.1f}')
    print(f'\nAudio saved to {out_dir}/')
    print('  clean_reference.wav          — voice with no loop (target)')
    print('  suppressed_mic.wav           — loop closed, detector+notch in loop')
    print('  suppressed_out.wav           — speaker feed after notch')
    print('  suppressed_out_restored.wav  — speaker feed after notch + restorer (post)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gain',      type=float, default=1.3)
    p.add_argument('--duration',  type=float, default=30.0)
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--detector',  type=str,   default=None)
    p.add_argument('--restorer',  type=str,   default=None)
    p.add_argument('--out-dir',   type=str,   default=None)
    args = p.parse_args()
    run_eval(gain=args.gain, duration_s=args.duration, threshold=args.threshold,
             det_ckpt=args.detector, res_ckpt=args.restorer, out_dir=args.out_dir)
