"""
feedback_detect/run_inference.py — Run the detector+notch pipeline on mic*.wav files.

Produces enhanced_detect_*.wav alongside each mic*.wav.
Also produces notch_mask_*.png showing which bins were notched over time.

Usage:
    # Howl test scenarios:
    python feedback_detect/run_inference.py --val-dir data/howl_test/01_howl

    # All scenarios:
    for d in data/howl_test/*/; do
        python feedback_detect/run_inference.py --val-dir "$d" --out-dir "$d"
    done
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

from model import FeedbackDetector, SR, N_FFT, HOP, N_FREQ
from notch  import NotchBank
from live   import _cluster_bins

CHECKPOINT    = PROJECT_ROOT / 'checkpoints' / 'feedback_detect' / 'best.pt'
DETECT_THRESH = 0.5
NOTCH_Q       = 30.0
NOTCH_DEPTH   = -24.0

_hpf_sos = butter(2, 90.0 / (SR / 2), btype='high', output='sos')


def run_batch(val_dir=None, out_dir=None, checkpoint=None,
              threshold=DETECT_THRESH, q=NOTCH_Q, depth_db=NOTCH_DEPTH):
    val_dir    = Path(val_dir    or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')
    out_dir    = Path(out_dir    or PROJECT_ROOT / 'data' / 'eval_output_detect')
    ckpt_path  = Path(checkpoint or CHECKPOINT)

    assert ckpt_path.exists(), f'No checkpoint at {ckpt_path} — train first.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).sqrt().to(device)
    model  = FeedbackDetector().to(device).eval()
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    print(f'FeedbackDetector loaded from {ckpt_path}')

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)

    for mic_path in sorted(val_dir.glob('mic*.wav')):
        mic_np, sr = sf.read(str(mic_path), dtype='float32')
        assert sr == SR, f'{mic_path.name}: expected {SR} Hz, got {sr}'
        if mic_np.ndim > 1: mic_np = mic_np.mean(1)

        mic_np = sosfilt(_hpf_sos, mic_np).astype(np.float32)

        # ── Frame-by-frame streaming simulation ───────────────────────────
        notch_bank   = NotchBank(sr=SR, q=q, depth_db=depth_db)
        gru_h        = None
        output       = np.zeros_like(mic_np)
        analysis_buf = np.zeros(N_FFT, dtype=np.float32)
        notch_log    = []   # (frame, [active_freqs])

        n_frames = len(mic_np) // HOP
        for i in range(n_frames):
            block = mic_np[i * HOP:(i + 1) * HOP]

            # Update analysis buffer
            analysis_buf = np.roll(analysis_buf, -HOP)
            analysis_buf[-HOP:] = block

            # Detection
            buf_t = torch.from_numpy(analysis_buf).unsqueeze(0).to(device)
            stft  = torch.stft(buf_t, N_FFT, N_FFT, N_FFT, window, return_complex=True)
            mag   = stft.abs()

            with torch.no_grad():
                prob, gru_h = model(mag, gru_h)

            prob_np = prob[0, :, 0].cpu().numpy()
            above   = (prob_np > threshold) & (bin_freqs >= 80.0)
            freqs   = _cluster_bins(bin_freqs, prob_np, above)

            notch_bank.update(freqs)
            output[i * HOP:(i + 1) * HOP] = notch_bank.process(block)
            notch_log.append(notch_bank.active_freqs[:])

        # Handle remaining samples
        tail = mic_np[n_frames * HOP:]
        if len(tail) > 0:
            output[n_frames * HOP:] = notch_bank.process(tail)

        stem     = mic_path.stem.replace('mic', 'enhanced_detect', 1) or 'enhanced_detect'
        out_path = out_dir / f'{stem}.wav'
        sf.write(str(out_path), output[:len(mic_np)], SR, subtype='PCM_16')
        print(f'  {mic_path.name} → {out_path.name}')

        # ── Spectrogram comparison plot ────────────────────────────────────
        _save_plot(mic_np, output[:len(mic_np)], notch_log,
                   out_dir / f'{stem}_mask.png', val_dir)

    print(f'Done — output in {out_dir}/')


def _save_plot(mic_np, enh_np, notch_log, plot_path, val_dir):
    """Save mic / enhanced / mask comparison spectrogram."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display

        n_fft   = N_FFT
        hop     = HOP
        mic_db  = librosa.amplitude_to_db(np.abs(librosa.stft(mic_np, n_fft=n_fft, hop_length=hop)), ref=np.max)
        enh_db  = librosa.amplitude_to_db(np.abs(librosa.stft(enh_np, n_fft=n_fft, hop_length=hop)), ref=np.max)
        mask    = np.abs(librosa.stft(enh_np, n_fft=n_fft, hop_length=hop)) / \
                  (np.abs(librosa.stft(mic_np, n_fft=n_fft, hop_length=hop)) + 1e-8)

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        librosa.display.specshow(mic_db, sr=SR, hop_length=hop, x_axis='time',
                                 y_axis='hz', ax=axes[0], vmin=-80)
        axes[0].set_title('mic (input)')
        axes[0].set_ylim(0, 8000)

        librosa.display.specshow(enh_db, sr=SR, hop_length=hop, x_axis='time',
                                 y_axis='hz', ax=axes[1], vmin=-80)
        axes[1].set_title('enhanced (output)')
        axes[1].set_ylim(0, 8000)

        librosa.display.specshow(mask, sr=SR, hop_length=hop, x_axis='time',
                                 y_axis='hz', ax=axes[2], vmin=0, vmax=1,
                                 cmap='RdYlGn')
        axes[2].set_title('suppression ratio (green=pass, red=notched)')
        axes[2].set_ylim(0, 8000)

        # Load clean reference for title if available
        clean_path = val_dir / 'clean.wav'
        title_sfx  = f' — {val_dir.name}' if clean_path.exists() else ''
        fig.suptitle(f'FeedbackDetector{title_sfx}')
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=120)
        plt.close(fig)
        print(f'  Saved plot → {plot_path.name}')
    except ImportError:
        pass   # librosa / matplotlib not available — skip plot


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir',    default=None)
    ap.add_argument('--out-dir',    default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--threshold',  type=float, default=DETECT_THRESH)
    ap.add_argument('--q',          type=float, default=NOTCH_Q)
    ap.add_argument('--depth',      type=float, default=NOTCH_DEPTH)
    args = ap.parse_args()
    run_batch(args.val_dir, args.out_dir, args.checkpoint,
              threshold=args.threshold, q=args.q, depth_db=args.depth)
