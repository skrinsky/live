"""
feedback_detect/run_inference_restore_v2.py — Detector + NotchBank + VoiceRestorerV2 offline pipeline.

Flow per file:
  1) Run FeedbackDetector + NotchBank frame-by-frame to produce a notched signal.
  2) Log active notches each frame to build a notch mask.
  3) Run VoiceRestorerV2 once on the full notched signal using the notch mask.
  4) Reconstruct with the notched phase and write enhanced_restore_v2_*.wav.

This leaves the original scripts untouched.
"""

import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from feedback_detect.model import FeedbackDetector, SR, N_FFT, HOP, N_FREQ, prepare_features
from feedback_detect.notch import NotchBank
from feedback_detect.live import _cluster_bins

from voice_restore.model_v2 import VoiceRestorerV2, apply_compensation
from voice_restore.features_v2 import make_v2_inputs
from voice_restore import train as vr_train  # for notch_frequency_response, extract_f0, HPF

_hpf_sos = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
BIN_FREQS = np.fft.rfftfreq(1024, d=1.0 / SR)  # voice_restore N_FFT


def build_notch_mask(active_logs, n_frames, n_freqs=512):
    """
    Build (n_freqs+1, T) notch mask in dB from per-frame active_notches logs.
    active_logs: list of lists of (freq_hz, depth_db, q)
    """
    mask = np.zeros((n_freqs + 1, n_frames), dtype=np.float32)
    for t, notches in enumerate(active_logs):
        cur = np.zeros(n_freqs + 1, dtype=np.float32)
        for f, d, q in notches:
            cur += vr_train.notch_frequency_response(f, d, q)
        cur = np.clip(cur, -96.0, 0.0)
        mask[:, t] = cur
    return mask


def run_file(mic_path: Path,
             detector_ckpt: Path,
             restorer_ckpt: Path,
             out_dir: Path,
             threshold: float = 0.5,
             q: float = 30.0,
             depth_db: float = -24.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_det = torch.hann_window(N_FFT).sqrt().to(device)

    # Load detector
    model_det = FeedbackDetector().to(device).eval()
    ckpt_det = torch.load(str(detector_ckpt), map_location=device)
    model_det.load_state_dict(ckpt_det['model'] if isinstance(ckpt_det, dict) and 'model' in ckpt_det else ckpt_det)

    # Load restorer
    model_res = VoiceRestorerV2().to(device).eval()
    ckpt_res = torch.load(str(restorer_ckpt), map_location=device)
    model_res.load_state_dict(ckpt_res['model'] if isinstance(ckpt_res, dict) and 'model' in ckpt_res else ckpt_res)

    # Read mic
    mic_np, sr = sf.read(str(mic_path), dtype='float32')
    assert sr == SR, f'{mic_path.name}: expected {SR} Hz, got {sr}'
    if mic_np.ndim > 1:
        mic_np = mic_np.mean(1)
    mic_np = sosfilt(_hpf_sos, mic_np).astype(np.float32)

    # Frame loop: detector + notch
    notch_bank   = NotchBank(sr=SR, q=q, depth_db=depth_db)
    gru_h        = None
    output       = np.zeros_like(mic_np)
    analysis_buf = np.zeros(N_FFT, dtype=np.float32)
    notch_logs   = []
    bin_freqs_det = np.fft.rfftfreq(N_FFT, d=1.0 / SR)

    n_frames = len(mic_np) // HOP
    for i in range(n_frames):
        block = mic_np[i * HOP:(i + 1) * HOP]

        analysis_buf = np.roll(analysis_buf, -HOP)
        analysis_buf[-HOP:] = block

        buf_t = torch.from_numpy(analysis_buf).unsqueeze(0).to(device)
        stft  = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window_det, return_complex=True)
        mag   = stft.abs()

        with torch.no_grad():
            prob, gru_h = model_det(prepare_features(mag), gru_h)

        prob_np = prob[0, :, 0].cpu().numpy()
        above   = (prob_np > threshold) & (bin_freqs_det >= 80.0)
        freqs   = _cluster_bins(bin_freqs_det, prob_np, above)

        notch_bank.update(freqs)
        output[i * HOP:(i + 1) * HOP] = notch_bank.process(block)
        notch_logs.append([(f, d, q) for f, d, q in notch_bank.active_notches])

    tail = mic_np[n_frames * HOP:]
    if len(tail) > 0:
        output[n_frames * HOP:] = notch_bank.process(tail)

    # Build notch mask for restorer
    mask_db = build_notch_mask(notch_logs, n_frames, n_freqs=512)
    mask_db_t = torch.from_numpy(mask_db).to(device).unsqueeze(0)

    # STFT of notched signal for restorer
    window_res = torch.hann_window(1024).sqrt().to(device)
    notched_t = torch.from_numpy(output).unsqueeze(0).to(device)
    notched_stft = torch.stft(notched_t, 1024, 480, 1024, window_res, return_complex=True)
    notched_mag = notched_stft.abs()

    # F0 for conditioning
    try:
        f0_np, conf_np = vr_train.extract_f0(mic_path, device=str(device))
    except Exception:
        T = mask_db.shape[1]
        f0_np = np.zeros(T, dtype=np.float32)
        conf_np = np.zeros(T, dtype=np.float32)

    spectral, cond = make_v2_inputs(notched_mag, mask_db_t, f0_np, conf_np)

    with torch.no_grad():
        gain, _ = model_res(spectral, cond)
    comp_mag = apply_compensation(notched_mag, mask_db_t, gain)[0]

    notched_phase = notched_stft / (notched_mag + 1e-8)
    restored_stft = comp_mag * notched_phase
    restored_wav = torch.istft(restored_stft.unsqueeze(0), 1024, 480, 1024, window_res)[0].cpu().numpy()

    # Align lengths
    L = min(len(output), len(restored_wav))
    out_restored = restored_wav[:L]

    out_stem = mic_path.stem.replace('mic', 'enhanced_restore_v2', 1) or 'enhanced_restore_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / f'{out_stem}.wav'), out_restored, SR, subtype='PCM_16')
    print(f'{mic_path.name} → {out_stem}.wav')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir',    default=None)
    ap.add_argument('--out-dir',    default=None)
    ap.add_argument('--detector',   default=str(PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'))
    ap.add_argument('--restorer',   default=str(PROJECT_ROOT / 'checkpoints' / 'voice_restore_v2' / 'best.pt'))
    ap.add_argument('--threshold',  type=float, default=0.5)
    ap.add_argument('--q',          type=float, default=30.0)
    ap.add_argument('--depth',      type=float, default=-24.0)
    args = ap.parse_args()

    val_dir = Path(args.val_dir or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')
    out_dir = Path(args.out_dir or PROJECT_ROOT / 'data' / 'eval_output_restore_v2')
    detector_ckpt = Path(args.detector)
    restorer_ckpt = Path(args.restorer)

    mic_files = sorted(val_dir.glob('mic*.wav'))
    if not mic_files:
        print(f'No mic*.wav in {val_dir}')
        return

    for mic_path in mic_files:
        run_file(mic_path, detector_ckpt, restorer_ckpt, out_dir,
                 threshold=args.threshold, q=args.q, depth_db=args.depth)


if __name__ == '__main__':
    main()
