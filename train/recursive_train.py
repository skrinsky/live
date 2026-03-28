"""
train/recursive_train.py — True BPTT recursive training for FDKFNet

Architecture:
  Single forward pass with full gradient tracking through the feedback loop.
  The model's output at frame t feeds back through a differentiable acoustic
  convolution (F.conv1d) to produce the mic input at frame t+1. Gradients
  flow through the complete causal chain — the model is penalised for outputs
  at t that cause divergence at t+k, which teacher-forcing cannot do.

  Reference: arXiv 2309.16048 (recursive AHS training methodology)

Dual-path:
  Both mains (long-path, reverberant) and monitor (short-path, direct) feedback
  IRs are active simultaneously every sequence — matching real-world conditions
  where a performer faces both paths at once.

Scheduled sampling (ref signal):
  TF=1.0 for TF_WARMUP_EPOCHS, then decays TF_DECAY_RATE/epoch to 0.
  During TF phase, ref = clean reverberant vocal (stable warm start).
  After decay, ref = model's own previous output (true deployment condition).
  VAD gate override is tied per-step to which ref is used (clean→override,
  model output→normal gate), not to the epoch-level TF probability.

Memory:
  BPTT graph spans SEQ_FRAMES. The feedback convolution at frame t only needs
  the last ceil((L_ir + HOP - 1) / HOP) frames from the outputs list — older
  frames are excluded from the graph automatically, bounding memory.
  Start at SEQ_FRAMES=50; increase to 100 → 200 once loss is decreasing smoothly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))   # makes: from model import FDKFNet work

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import random
from scipy.signal import fftconvolve, butter, sosfilt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import FDKFNet


# ── Constants (SR defined BEFORE any function that uses it as a default arg) ──

SR             = 48000
N_FFT          = 960
HOP            = 480
N_FREQS        = N_FFT // 2 + 1
MAX_IR_SAMPLES = int(1.5 * SR)  # 1.5s cap — F.conv1d cost is O(L_ir * HOP) per frame;
                                  # gymnasium rooms produce 2s+ IRs without this cap

SEQ_FRAMES       = 100    # BPTT window. Run 2: 100 (1s); next: 200 once stable
BATCH_SIZE       = 4      # gradient accumulation
EPOCHS           = 300    # extended: SEQ_FRAMES=100 from scratch needs ~2x more epochs
LR               = 1e-4
GRAD_CLIP        = 0.5
CLIP_LEVEL       = 0.95   # soft-clip ceiling on model output in feedback path

# Teacher-forcing schedule:
#   Phase 1 (epochs 1–TF_WARMUP_EPOCHS): TF=1.0 — model sees only clean ref.
#     Model learns supervised speech enhancement before any recursive training.
#     Prevents the shortcut-collapse that occurs when random-init model output
#     is fed back as ref before the model produces anything meaningful.
#   Phase 2 (after warmup): TF decays 0.01/epoch (100 epochs to reach 0).
#     Slow enough that GRU can adapt incrementally to model output as ref.
TF_WARMUP_EPOCHS = 25     # epochs of pure supervised training before decay starts
TF_DECAY_RATE    = 0.01   # TF reduction per epoch after warmup (0.01 = 1%/epoch)

PROJECT_ROOT = Path(__file__).parent.parent


# ── Console HPF ────────────────────────────────────────────────────────────────

def make_console_hpf(cutoff_hz=90, sr=SR):
    """
    2nd-order Butterworth HPF — standard vocal channel console high-pass (80–100Hz).
    Randomized cutoff per sequence covers different engineer preferences.
    Returns SOS coefficients for sosfilt().
    """
    return butter(2, cutoff_hz / (sr / 2), btype='high', output='sos')


# ── Gain sampling ──────────────────────────────────────────────────────────────

def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), in dB.
    Both inputs are (HOP,) time-domain tensors. Returns a scalar.
    Minimise -SI-SDR during training (higher SI-SDR = better).
    Bounded and scale-invariant — unaffected by feedback amplitude.
    """
    target  = target - target.mean()
    estimate = estimate - estimate.mean()
    dot     = (estimate * target).sum()
    s_target = dot / (target.pow(2).sum() + eps) * target
    noise   = estimate - s_target
    return 10 * torch.log10(s_target.pow(2).sum() / (noise.pow(2).sum() + eps))


def sample_gain():
    """
    Returns (mains_gain, monitor_gain) independently sampled.
    30% normal (0.2–0.6) | 25% near-threshold (0.6–0.9) | 45% active (0.9–1.5)
    Both paths sampled independently — monitor can be hot while mains is quiet,
    which is the most common failure mode in small venues and HOW settings.
    Above-threshold gains (> 1.0) are safe with SI-SDR loss because SI-SDR is
    scale-invariant and bounded regardless of feedback amplitude.
    Raised above-threshold from 25%→45%: model needs to see actual Larsen buildup
    as the common case, not the exception. Below-threshold sequences are easy (do
    nothing gets good SI-SDR) and dilute the gradient signal for suppression.
    """
    def _one():
        t = random.random()
        if t < 0.30:   return random.uniform(0.2, 0.6)   # 30% low
        elif t < 0.55: return random.uniform(0.6, 0.9)   # 25% near-threshold
        else:          return random.uniform(0.9, 1.5)   # 45% active Larsen
    return _one(), _one()


# ── Differentiable STFT / ISTFT ────────────────────────────────────────────────

def torch_stft(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """(HOP,) tensor → (N_FREQS,) complex. Fully differentiable. Causal left-pad."""
    xp = F.pad(x.unsqueeze(0), (N_FFT - HOP, 0))          # (1, N_FFT)
    return torch.fft.rfft(xp * window, n=N_FFT).squeeze(0) # (N_FREQS,)


def torch_istft(X: torch.Tensor) -> torch.Tensor:
    """(N_FREQS,) complex → (HOP,) tensor. Fully differentiable."""
    return torch.fft.irfft(X, n=N_FFT)[-HOP:]


# ── Differentiable feedback convolution ────────────────────────────────────────

def feedback_conv(outputs: list, ir_t: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Compute the feedback contribution for the current frame by convolving the
    model's output history with the acoustic IR.

    Fully differentiable — gradients flow through the outputs list back to the
    model weights that produced each prior frame. This is the core of BPTT:
    the gradient at frame t propagates through the feedback path to frames t-1,
    t-2, etc., teaching the model that its previous outputs caused the current
    feedback state.

    Only the last ceil((L_ir + HOP - 1) / HOP) frames are needed — older frames
    cannot affect the current output through the IR, so they're excluded from
    the graph automatically. Memory is bounded regardless of sequence length.
    """
    if not outputs:
        return ir_t.new_zeros(HOP)

    L      = len(ir_t)
    needed = L + HOP - 1

    # Only stack the relevant suffix of the output history
    n_frames_needed = (needed + HOP - 1) // HOP
    relevant = outputs[-n_frames_needed:]
    playback = torch.cat(relevant)   # ∈ compute graph — gradients flow here

    if playback.shape[0] < needed:
        playback = F.pad(playback, (needed - playback.shape[0], 0))
    else:
        playback = playback[-needed:]

    # Linear convolution via F.conv1d (differentiable cross-correlation with flipped kernel)
    fb = F.conv1d(
        playback.view(1, 1, -1),
        ir_t.flip(0).view(1, 1, -1),
        padding=0
    ).squeeze()   # (HOP,)

    return fb * gain


# ── Per-sequence training function ─────────────────────────────────────────────

def train_one_sequence(model, vocal_np, mains_ir_np, monitor_ir_np,
                       room_ir_np, noise_np, device, window, teacher_forcing_prob):
    """
    Full BPTT pass over one SEQ_FRAMES sequence with simultaneous dual-path feedback.

    Returns scalar loss tensor (graph attached — caller calls .backward()).
    Returns None if sequence is degenerate (too short, all-NaN inputs, etc.).
    """
    target_len = SEQ_FRAMES * HOP

    # ── Pre-processing (numpy, outside compute graph) ─────────────────────────
    # IR length cap: F.conv1d cost = O(L_ir * HOP) per frame
    mains_ir_np   = mains_ir_np[:MAX_IR_SAMPLES]
    monitor_ir_np = monitor_ir_np[:MAX_IR_SAMPLES]

    # Room reverb: dry vocal → reverberant target (what the model should output)
    reverb_np = fftconvolve(vocal_np[:target_len], room_ir_np)[:target_len].astype(np.float32)

    # Console HPF applied to the reverberant target (pre-filter, not in graph).
    # Since the PA chain HPF's everything before it reaches the speaker, the
    # feedback path naturally carries HPF'd content once the model is trained.
    # Pre-filtering the target teaches the model not to produce sub-90Hz output,
    # which means the feedback that accumulates is also HPF'd. Clean and consistent.
    hpf = make_console_hpf(np.random.uniform(70, 120))
    reverb_np = sosfilt(hpf, reverb_np).astype(np.float32)

    # Noise: 50% convolutive (crowd/ambient convolved with room IR), 50% additive.
    # Convolutive noise forces the model to use reference coherence to distinguish
    # feedback from crowd — both arrive with the same reverb character. Without this,
    # the model can identify feedback by spectral pattern alone, ignoring the reference.
    if random.random() < 0.5:
        noise_np = fftconvolve(noise_np, room_ir_np)[:target_len].astype(np.float32)
    else:
        noise_np = noise_np[:target_len]
    vocal_rms   = float(np.sqrt(np.mean(reverb_np ** 2))) + 1e-8
    noise_rms   = float(np.sqrt(np.mean(noise_np  ** 2))) + 1e-8
    snr_db      = np.random.uniform(5, 40)
    noise_scale = (vocal_rms / noise_rms) * (10 ** (-snr_db / 20))

    # ── Convert to tensors ────────────────────────────────────────────────────
    reverb_t     = torch.from_numpy(reverb_np).float().to(device)
    noise_t      = torch.from_numpy(noise_np.astype(np.float32)).float().to(device)
    mains_ir_t   = torch.from_numpy(mains_ir_np).float().to(device)
    monitor_ir_t = torch.from_numpy(monitor_ir_np).float().to(device)

    # ── Gain + path dropout ───────────────────────────────────────────────────
    mains_gain, monitor_gain = sample_gain()
    drop_mains   = random.random() < 0.2   # 20% IEM/no-mains scenarios
    drop_monitor = random.random() < 0.2

    # ── Initial state ─────────────────────────────────────────────────────────
    H, P, gru_h = model.init_state(1, device)

    outputs     = []   # (HOP,) tensors per frame — in compute graph, used for feedback

    for t in range(SEQ_FRAMES):
        start, end = t * HOP, t * HOP + HOP
        reverb_frame = reverb_t[start:end]   # clean target for this frame
        noise_frame  = noise_t[start:end] * noise_scale

        # ── Feedback (differentiable through outputs) ─────────────────────────
        mains_fb   = (feedback_conv(outputs, mains_ir_t,   mains_gain)
                      if not drop_mains   else mains_ir_t.new_zeros(HOP))
        monitor_fb = (feedback_conv(outputs, monitor_ir_t, monitor_gain)
                      if not drop_monitor else monitor_ir_t.new_zeros(HOP))

        # ── Mic signal ────────────────────────────────────────────────────────
        mic_frame = reverb_frame + mains_fb + monitor_fb + noise_frame
        mic_frame = torch.clamp(mic_frame, -1.0, 1.0)   # prevent early-training divergence

        # ── Ref signal (scheduled sampling) ──────────────────────────────────
        # Teacher-forcing phase (early training): use clean reverb vocal as ref.
        # This gives the Kalman filter a correct H(z) reference while GRU learns.
        # Recursive phase (later training): use model's own last output as ref.
        # This matches inference: ref = what the box is actually sending to the PA.
        # vad_override is tied to the per-step ref choice, not the epoch-level TF prob.
        # When clean ref is used: override VAD to 1.0 (always update H), because
        #   mic_power/ref_power >> VAD_RATIO at feedback freqs with clean teacher ref,
        #   which would freeze H exactly when it should be learning the feedback path.
        # When model output is used as ref: normal VAD gate (None), matching inference.
        use_model_ref = bool(outputs) and random.random() >= teacher_forcing_prob
        if use_model_ref:
            ref_frame    = outputs[-1]            # model output — deployment condition
            vad_override = None                   # use normal VAD gate
        else:
            ref_frame    = reverb_frame.detach()  # teacher signal — stable warm start
            vad_override = 1.0                    # clean ref → always update H

        # ── Differentiable STFT ───────────────────────────────────────────────
        mic_f = torch_stft(mic_frame, window).unsqueeze(0)   # (1, N_FREQS)
        ref_f = torch_stft(ref_frame, window).unsqueeze(0)

        # ── Model forward ─────────────────────────────────────────────────────
        speech_f, H, P, gru_h = model.forward_frame(
            mic_f, ref_f, H, P, gru_h, vad_override=vad_override
        )

        # ── Output frame (differentiable, soft-clipped, feeds back next frame)
        out_frame = torch_istft(speech_f.squeeze(0))
        out_frame = torch.tanh(out_frame / CLIP_LEVEL) * CLIP_LEVEL
        outputs.append(out_frame)

    if not outputs:
        return None

    # ── Loss: SI-SDR over the full sequence ───────────────────────────────────
    # Computed once over 500ms, not per-frame. Per-frame SI-SDR on 480 samples
    # is numerically unstable: near-silent frames have target.pow(2) ≈ 0, making
    # SI-SDR → -inf. A single sequence-level computation is stable and matches
    # standard practice in speech enhancement literature.
    out_full   = torch.cat(outputs)          # (SEQ_FRAMES * HOP,) — in compute graph
    clean_full = reverb_t.detach()           # (SEQ_FRAMES * HOP,) — no grad
    return -si_sdr(out_full, clean_full)


# ── Main training loop ─────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = torch.hann_window(N_FFT).to(device)

    model     = FDKFNet().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    writer    = SummaryWriter(str(PROJECT_ROOT / 'checkpoints' / 'fdkfnet' / 'tb'))

    # ── Load files ────────────────────────────────────────────────────────────
    vocal_files      = list((PROJECT_ROOT / 'data' / 'clean_vocals').rglob('*.wav'))
    ir_pool_dir      = PROJECT_ROOT / 'data' / 'ir_pool'
    mains_ir_files   = list(ir_pool_dir.glob('mains_*.wav'))
    monitor_ir_files = list(ir_pool_dir.glob('monitor_*.wav'))
    noise_files      = list((PROJECT_ROOT / 'data' / 'noise').rglob('*.wav'))
    room_ir_files    = list((PROJECT_ROOT / 'data' / 'public_irs').rglob('*.wav'))

    assert vocal_files,      "No vocal files in data/clean_vocals/ — run preprocess.py"
    assert mains_ir_files,   "No mains IRs — run simulator/generate_ir_pool.py first"
    assert monitor_ir_files, "No monitor IRs — run simulator/generate_ir_pool.py first"
    assert noise_files,      "No noise files in data/noise/ — download DNS noise set"

    # Filter to sequences long enough for SEQ_FRAMES
    min_dur     = SEQ_FRAMES * HOP / SR
    vocal_files = [f for f in vocal_files
                   if (info := sf.info(str(f))).frames / info.samplerate >= min_dur]
    assert vocal_files, f"No vocal files >= {min_dur:.1f}s after filtering"

    ckpt_dir = PROJECT_ROOT / 'checkpoints' / 'fdkfnet'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FDKFNet: {n_params:,} parameters on {device}")
    print(f"BPTT: {SEQ_FRAMES} frames ({SEQ_FRAMES * HOP / SR * 1000:.0f}ms), "
          f"IR cap {MAX_IR_SAMPLES/SR:.1f}s")
    print(f"Dual-path: {len(mains_ir_files)} mains + {len(monitor_ir_files)} monitor IRs")
    print(f"TF schedule: 1.0 for {TF_WARMUP_EPOCHS} epochs, then -{TF_DECAY_RATE}/epoch to 0")

    # ── Optional warm-start from a previous checkpoint ────────────────────────
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to warm-start weights from. '
                         'TF schedule and epoch counter always restart from 1.')
    ap.add_argument('--lr', type=float, default=None,
                    help='Override learning rate (default: LR constant in script). '
                         'Use a lower value (e.g. 3e-5) when resuming from a checkpoint.')
    args, _ = ap.parse_known_args()
    resume_tf   = 0.0          # TF to continue from (overridden if checkpoint has it)
    resume_best = float('inf') # best loss to continue from (overridden if checkpoint has it)
    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(str(resume_path), map_location=device)
        state = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else {'model': ckpt}
        model.load_state_dict(state['model'])
        # Restore TF and best_loss so warm-start continues smoothly without regression.
        # Checkpoints without these keys default to tf=0.0 (safe: assumes TF fully decayed).
        resume_tf   = state.get('tf_prob',   0.0)
        resume_best = state.get('best_loss', float('inf'))
        # Back up the source checkpoint before anything overwrites it.
        backup = resume_path.parent / (resume_path.stem + '_prev.pt')
        import shutil
        shutil.copy2(str(resume_path), str(backup))
        print(f"Warm-started weights from {args.resume}")
        print(f"  TF continuing from {resume_tf:.2f}, best_loss continuing from {resume_best:.4f}")
        print(f"  Backup saved → {backup}")

    if args.lr is not None:
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr
        scheduler.base_lrs = [args.lr for _ in scheduler.base_lrs]
        print(f"Learning rate overridden to {args.lr}")

    best_loss = resume_best

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Teacher-forcing prob: frozen at 1.0 during warmup, then 0.01/epoch decay.
        # On resume, continues decaying from the saved tf_prob rather than resetting to 1.0.
        if args.resume:
            tf_prob = max(0.0, resume_tf - (epoch - 1) * TF_DECAY_RATE)
        elif epoch <= TF_WARMUP_EPOCHS:
            tf_prob = 1.0
        else:
            tf_prob = max(0.0, 1.0 - (epoch - TF_WARMUP_EPOCHS) * TF_DECAY_RATE)

        epoch_loss  = 0.0
        n_steps     = 200
        valid_steps = 0

        optimizer.zero_grad()

        for step in tqdm(range(n_steps), desc=f'Epoch {epoch}/{EPOCHS} tf={tf_prob:.2f}'):
            # ── Sample files ──────────────────────────────────────────────────
            vocal_np, _ = sf.read(str(random.choice(vocal_files)), dtype='float32')
            if vocal_np.ndim > 1: vocal_np = vocal_np.mean(1)
            offset   = random.randint(0, max(0, len(vocal_np) - SEQ_FRAMES * HOP - 1))
            vocal_np = vocal_np[offset:offset + SEQ_FRAMES * HOP]

            mains_ir_np,   _ = sf.read(str(random.choice(mains_ir_files)),   dtype='float32')
            monitor_ir_np, _ = sf.read(str(random.choice(monitor_ir_files)), dtype='float32')

            if room_ir_files:
                room_ir_np, _ = sf.read(str(random.choice(room_ir_files)), dtype='float32')
            else:
                # Synthetic fallback: exponential-decay white noise room IR
                rt60 = np.random.uniform(0.2, 2.0)
                t_arr = np.arange(int(rt60 * SR)) / SR
                room_ir_np = np.random.randn(len(t_arr)).astype(np.float32)
                room_ir_np *= np.exp(-6.9 * t_arr / rt60).astype(np.float32)
                room_ir_np /= np.abs(room_ir_np).max() + 1e-8

            noise_np, _ = sf.read(str(random.choice(noise_files)), dtype='float32')
            if noise_np.ndim > 1: noise_np = noise_np.mean(1)
            if len(noise_np) < SEQ_FRAMES * HOP:
                noise_np = np.tile(noise_np, (SEQ_FRAMES * HOP // len(noise_np)) + 1)
            n_start  = random.randint(0, len(noise_np) - SEQ_FRAMES * HOP)
            noise_np = noise_np[n_start:n_start + SEQ_FRAMES * HOP]

            # ── BPTT forward pass (single pass, graph maintained) ─────────────
            loss = train_one_sequence(
                model, vocal_np, mains_ir_np, monitor_ir_np,
                room_ir_np, noise_np, device, window, tf_prob
            )
            if loss is None or not torch.isfinite(loss):
                continue

            (loss / BATCH_SIZE).backward()
            valid_steps += 1

            if valid_steps % BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += float(loss.item())

        # Flush remaining accumulated gradients
        if valid_steps % BATCH_SIZE != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(valid_steps // BATCH_SIZE, 1)
        scheduler.step()
        writer.add_scalar('loss/train',           avg_loss, epoch)
        writer.add_scalar('teacher_forcing_prob', tf_prob,  epoch)
        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
              f"valid {valid_steps}/{n_steps} | tf={tf_prob:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'tf_prob':   tf_prob,
                'best_loss': best_loss,
            }, str(ckpt_dir / 'best.pt'))
            print("  ✓ New best")


if __name__ == '__main__':
    train()
