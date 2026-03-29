# Training Debugging — What To Try If Current Run Fails

Current status: **FRESH RUN REQUIRED** — windowing bug found and fixed (see P0 below). Previous best checkpoint (-2.1174 dB) used Hann window, producing severe AM distortion at inference. Now using rectangular window. Starting from scratch.

**A run is "failing" if:** loss doesn't move below -1.5 dB SI-SDR after 100+ more epochs, or keeps oscillating positive.

---

## P0 — Hann Window Bug (FIXED 2026-03-29) — ROOT CAUSE OF "BAD RADIO" DISTORTION

**Problem:** `torch_stft` left-pads with zeros and applies Hann window. `torch_istft` takes `irfft(X)[-HOP:]`. For any input x:

```
irfft(rfft(hann * [zeros(HOP) | x]))[-HOP:] = hann[HOP:] * x
```

`hann[HOP:]` decays from 1.0 → 0 over HOP=480 samples (10ms). Every frame of audio fades to silence at its end, creating **100Hz amplitude modulation on the voice**. This sounds like "bad radio / car stereo" distortion — the exact symptom observed in the listening test.

**Why the training loss didn't catch it:** SI-SDR is scale-invariant and computed over the full sequence. The windowed output `hann[HOP:]*vocal` has some correlation with `vocal` (enough to give +2.1 dB SI-SDR), but the perceptual quality is terrible. The metric was lying about the model quality.

**Fix (applied):** Use rectangular window (ones) instead of Hann. Then:
```
irfft(rfft([zeros(HOP) | x]))[-HOP:] = x exactly
```
Perfect reconstruction — no windowing artifact.

**Tradeoff:** Rectangular window has worse spectral isolation than Hann (-13dB first sidelobe vs -31dB). This slightly degrades per-bin Kalman filter accuracy. But it's vastly better than a model that sounds like a broken radio. If future experiments need better spectral isolation, implement proper WOLA (sqrt-Hann analysis + sqrt-Hann synthesis + OLA at 50% overlap) — but this requires careful implementation to avoid the same reconstruction bug.

**Checkpoint compatibility:** Any checkpoint trained with Hann window is incompatible with the rectangular-window inference. **Do not attempt to resume from the old best.pt — retrain from scratch.**

---

---

## P0 — Fix TF Reset on Resume (should be next code change regardless)

**Problem:** `--resume` resets TF to 1.0. A model trained to TF=0 (self-referencing) gets clean references it's no longer calibrated for. This causes epoch 1 loss regression every warm-start (~2.4 dB vs -0.85 baseline).

**Fix:** Save and restore TF state in the checkpoint.

In the save block:
```python
torch.save({
    'epoch': epoch,
    'model': model.state_dict(),
    'best_loss': best_loss,        # so resume knows what it's improving on
    'tf_prob': tf_prob,            # so resume continues from where TF left off
}, str(ckpt_dir / 'best.pt'))
```

In the `--resume` load block, after `model.load_state_dict(...)`:
```python
resume_tf   = ckpt.get('tf_prob',   0.0)   if isinstance(ckpt, dict) else 0.0
resume_best = ckpt.get('best_loss', float('inf')) if isinstance(ckpt, dict) else float('inf')
best_loss = resume_best
# override the epoch-computed tf_prob at the start of the first epoch:
# add a flag so the loop uses resume_tf on epoch 1 instead of recomputing
```

This means epoch 1 of a warm-start starts where the previous run ended — no distribution shift, no regression.

---

## P1a — Convolutive Noise (high impact, easy code change)

**Problem:** Noise is currently added additively to the mic signal (`mic_frame = reverb_frame + feedback + noise_frame`). The noise clips from DNS Challenge are never convolved with the room IR. This means the model can learn to suppress feedback by spectral pattern-matching, without ever actually learning to use the reference signal for discrimination.

**Why this matters for FDKFNet specifically:** The whole point of the reference signal is to distinguish "sounds that came from the PA (feedback)" from "sounds that didn't (crowd, room tone)." In a real venue, crowd noise, HVAC, and other ambient sounds pass through the same room acoustics as the feedback — they arrive at the mic with the same reverb character. With additive training noise, the feedback is the only reverberant thing in the mix and the model can identify it without needing the reference at all. With convolutive noise, both crowd noise and feedback are reverberant — the only way to tell them apart is coherence with the reference signal. This forces the Kalman filter to actually learn to use the reference.

**Research basis:** REVERB Challenge (2014), WHAMR! dataset (2020), all CHiME challenges — all built around convolving ambient noise with room IRs to simulate realistic conditions.

**Fix:** In `train_one_sequence()` in `recursive_train.py`, convolve the noise with the room IR before mixing (50% of the time — some noise genuinely is non-reverberant, e.g., electrical hum):

```python
# Current (additive only):
noise_np = noise_np[:target_len]

# Fix (convolutive 50% of time):
if random.random() < 0.5:
    noise_np = fftconvolve(noise_np, room_ir_np)[:target_len].astype(np.float32)
else:
    noise_np = noise_np[:target_len]
```

This uses the DNS Challenge crowd/ambient clips we already have — no new data needed.

---

## P1 — Increase Above-Threshold Training Ratio (easy, high impact)

**Problem:** 75% of training sequences use sub-Larsen gains (0.3–0.9). For those, doing nothing achieves good SI-SDR. The model may learn "mostly pass through, occasionally suppress." The above-threshold sequences (25%) are the only ones that actually require suppression — and they're a minority.

**Current setup** (`recursive_train.py`):
```python
ABOVE_THRESHOLD_PROB = 0.25   # check the actual variable name in the script
```

**Fix:** Increase to 0.5 (50/50 above/below threshold). The model needs to see feedback suppression as the norm, not the exception.

Also consider: a **curriculum** that starts at 50% above-threshold for the first 50 epochs, then drops to 30% — forces the model to learn suppression before learning refinement.

---

## P1c — Reference Channel Dropout (easy, ~5 lines)

**Source:** DCCRN-E (Microsoft AEC Challenge 2022)

**Problem:** If the reference is always present during training, the model learns to do nearly as well without it via blind enhancement shortcuts. It has no training signal that specifically rewards using the reference over ignoring it.

**Fix:** On 10% of frames, zero out `ref_f` and set `vad_override=0.0` (freeze H/P). The model must do blind enhancement for those frames. This creates a gradient that rewards reference-awareness: the model gets a better loss outcome when the reference is present and used correctly vs. absent.

```python
# In the frame loop in train_one_sequence(), after computing ref_frame:
if random.random() < 0.10:
    ref_frame    = torch.zeros_like(ref_frame)
    vad_override = 0.0   # freeze H/P when ref is dropped
```

---

## P1b — Echo-Path Auxiliary Loss (moderate effort, high impact)

**Source:** DPCRN (INTERSPEECH 2022 AEC Challenge)

**Problem:** H (the feedback path estimate) only receives gradient through the final SI-SDR loss on the model output — a weak, indirect signal. The model can achieve decent SI-SDR without H converging to the true feedback transfer function (e.g., by learning a fixed spectral mask). If H is wrong, the model fails catastrophically at inference when the acoustic path changes.

**Fix:** Add a small auxiliary loss that directly supervises H against the known ground-truth feedback component (available in training since we synthesize it):

```python
# In train_one_sequence(), accumulate per-frame:
feedback_gt_f = torch_stft((mains_fb + monitor_fb).detach(), window)  # (1, N_FREQS) complex
H_fb_estimate = H_new * ref_f                                          # (1, N_FREQS) complex
echo_loss += F.mse_loss(H_fb_estimate.abs(), feedback_gt_f.abs().detach())

# Final loss:
loss = -si_sdr(out_full, clean_full) + 0.05 * echo_loss / SEQ_FRAMES
```

Use coefficient 0.05–0.1. The primary objective stays SI-SDR; this just anchors H to the true path. This is uniquely possible in our setup because we synthesize the training data and know the ground-truth feedback signal — real AEC systems don't have this luxury.

---

## P2 — Add Phase Features to GRU Input (moderate effort, likely high impact)

**Problem:** The GRU only sees log-magnitude features (`abs().pow(2)` → ERB bands). It has no access to phase. But feedback identification is fundamentally phase-sensitive — feedback is correlated with the reference in both magnitude AND phase. A model that can't see phase has to infer it indirectly from magnitude patterns over time.

**Current GRU input** (`model.py` line 67-71):
```python
feat = torch.cat([
    self._to_erb(mic_f.abs().pow(2)),    # log mic magnitude
    self._to_erb(ref_power),              # log ref magnitude
    self._to_erb(innovation.abs().pow(2)), # log innovation magnitude
], ...)
```

**Fix:** Add the normalized cross-spectrum between mic and ref per ERB band — **both real and imaginary parts**. The real part is cosine of IPD (phase alignment magnitude); the imaginary part is sine of IPD (phase lead/lag direction). Using only real discards whether the reference leads or lags the mic — exactly the information that distinguishes feedback coherence from coincidental correlation. Standard form in IPD-based AEC/beamforming literature (arXiv 2111.04904).

```python
# Numerically stable normalized cross-spectrum (both real and imaginary parts)
denom     = (mic_f.abs() * ref_f.abs()).clamp(min=1e-8)   # (B, F)
ncs       = mic_f * ref_f.conj() / denom                  # (B, F) complex, |ncs|=1
ncs_real  = ncs.real                                       # cosine of IPD, [-1, 1]
ncs_imag  = ncs.imag                                       # sine of IPD,   [-1, 1]
coh_real_erb = ncs_real @ self.mel_fb                      # (B, n_bands)
coh_imag_erb = ncs_imag @ self.mel_fb                      # (B, n_bands)

feat = torch.cat([
    self._to_erb(mic_f.abs().pow(2)),
    self._to_erb(ref_power),
    self._to_erb(innovation.abs().pow(2)),
    coh_real_erb,                                          # +1 feature group
    coh_imag_erb,                                          # +1 feature group
], ...)
```

GRU input grows from `3 × n_bands` to `5 × n_bands`. Minor param increase (~13%).

**Engineering notes:**
- `.clamp(min=1e-8)` on the denominator is required — without it, near-silence frames produce NaN/±∞ that immediately destabilize the GRU hidden state.
- Feature scale mismatch: coherence features are bounded [-1, 1] while log-power features are unbounded. This slows early learning of coherence but does not cause divergence — no additional normalization needed.
- NeuralKalmanAHS (arXiv 2309.16049), our direct architecture basis, does NOT include coherence features. Both IPD components are an improvement over the paper.
- The CADB-Conformer ablation (Interspeech 2024) found inter-channel features contributed +0.84 dB SDRi — the closest published evidence of impact for this type of feature.

---

## P3 — Bigger GRU / More ERB Bands (moderate effort, moderate impact)

**Current:** `HIDDEN=128, N_LAYERS=2, N_BANDS=64`

**The constraint:** This runs on a Pi 5 in real time. At 10ms blocks (HOP=480), each block budget is ~5ms for inference. The GRU is the bottleneck.

**Do P2 (coherence features) before P3.** ICASSP 2023 AEC Challenge found a negative correlation (PCC = -0.54) between parameter count and score — better features beat more capacity for reference-conditioned tasks. A HIDDEN=128 model with coherence features will likely outperform HIDDEN=256 without them.

**Compute cost of HIDDEN=256 (verified):**
- GRU MACs per frame: ~221K (current) → ~737K (HIDDEN=256) — 3.3× increase
- Estimated Pi 5 latency: 1.5–10ms per frame (wide range — PyTorch kernel overhead dominates at small matrix sizes)
- The upper end busts the budget. **Must benchmark empirically before training.**

**Options in priority order:**
1. `N_BANDS=128` — finer Q/R resolution, near-free compute (matrix multiply not recurrence).
2. `HIDDEN=256` — only attempt after: (a) P2 coherence features are working, (b) Pi 5 timing is measured empirically and shows headroom. Use `torch.quantization.quantize_dynamic` (qnnpack) for 2–4× ARM speedup if needed.
3. `N_LAYERS=3` — adds depth not width. Less impact. Not recommended without profiling.

**Before attempting HIDDEN=256**, measure current model on Pi 5:
```python
import time
model.eval()
for _ in range(1000):
    t0 = time.perf_counter()
    with torch.no_grad():
        _, H, P, h = model.forward_frame(mic_f, ref_f, H, P, h)
    print(time.perf_counter() - t0)
```

Target: current model < 2ms, leaving headroom for HIDDEN=256 at 3.3× cost.

---

## P4 — More / Better Training Data (high effort, high uncertainty)

**Current data:**
- Clean vocals: `data/clean_vocals/` — unknown count, check with `ls data/clean_vocals/ | wc -l`
- Room IRs: generated by `simulator/generate_pairs.py` via pyroomacoustics
- Noise: `data/noise/`

**Known issues:**
1. **Simulated IRs ≠ real IRs.** pyroomacoustics ISM doesn't model: speaker directivity, room modes below Schroeder frequency, temperature/humidity drift, nonlinear clipping at high SPL. Model trained on simulated paths may not transfer to real hardware.
2. **If vocal data is < 100 files,** the model hears the same singers too often and learns voice-specific suppression rather than generalizing.

**Fixes:**
1. **Real room IRs:** Measure actual IRs with a speaker + mic + sine sweep in a real room. Tools: `pyroomacoustics`, `acoustics`, or even Audacity + logsweep. 10-20 measured IRs mixed with simulated ones would help.
2. **More vocal diversity:** DAPS dataset, VoxCeleb (clean version), or any freely licensed speech corpus with varied speakers, accents, mic types.
3. **Real feedback recordings:** If you can record actual Larsen feedback on the hardware prototype, even 10-20 examples mixed into training would dramatically reduce the sim-to-real gap.

---

## P5 — Architecture Replacement (last resort, major effort)

**When to consider:** If all P0-P4 have been tried and the model still doesn't show good feedback suppression on scenarios 8 and 9 in the listening test.

**Option A: Bigger FDKFNet**
Keep the same per-bin Kalman structure but expand the GRU substantially (HIDDEN=512, N_BANDS=128, phase features). This preserves the reference-signal advantage.

**Option B: GTCRN backbone**
`gtcrn/` is already in the repo. GTCRN achieves PESQ 2.87 / STOI 0.940 on VCTK-DEMAND with 48K params. It's a denoiser (no reference signal) but the temporal convolutional backbone could replace the GRU in FDKFNet, giving better cross-bin and cross-time modeling.

Adaptation needed: GTCRN processes magnitude spectrograms. Would need to:
- Add a reference signal input path
- Keep the Kalman update structure for feedback cancellation
- Retrain from scratch

**Option C: Full AEC approach**
Treat the problem as Acoustic Echo Cancellation (AEC) rather than speech enhancement. AEC models (e.g., DCCRN-E, FullSubNet) are specifically designed for reference-signal-based interference cancellation in a closed loop. The feedback suppression task is structurally identical to echo cancellation. Pre-trained AEC models exist and could be fine-tuned.

---

## Diagnostic Experiment — Blind Enhancer Baseline

**What it is:** Train a model that takes only the mic signal and outputs clean vocal — no reference signal, no Kalman filter. Essentially what De-Feedback does. Compare it to FDKFNet.

**Why it matters:** If blind does nearly as well as FDKFNet, it means the Kalman filter isn't actually exploiting the reference signal properly — either the reference is wrong during training (TF mismatch), or the GRU isn't learning to use it. If FDKFNet is clearly better, the reference is doing real work and the architecture is sound.

**The GTCRN in the repo is NOT this experiment.** GTCRN was trained on DNS3 (traffic, cafeteria noise) — not feedback. Running it on our scenarios would test a noise suppressor against feedback, which aren't the same problem and would tell us nothing useful.

**What to build:** A separate training script `train/blind_train.py` that:
- Takes only `mic_signal` as input (no ref)
- Uses a simple GRU-based encoder/decoder (or GTCRN architecture) operating on STFT magnitude
- Trains on the same simulated data with SI-SDR loss
- At inference: `enhanced = model(mic_stft)` — no Kalman, no reference

**Training script outline:**
```python
# blind_train.py — train a blind vocal enhancer (no reference) as a diagnostic baseline
# Architecture: GTCRN-style (use gtcrn/gtcrn.py as backbone)
# Input: mic STFT magnitude (B, T, F)
# Output: enhanced STFT (B, T, F)
# Loss: SI-SDR vs clean target
# Purpose: measure how much the reference signal actually helps FDKFNet
```

**How to interpret results:**
- Blind STOI > FDKFNet STOI → reference isn't helping; training distribution or architecture issue
- FDKFNet STOI > Blind by >0.05 → reference is working, worth continuing FDKFNet path
- Both score poorly on scenarios 8/9 → data is the ceiling (need singing data + real IRs)

**When to run:** After FDKFNet training stabilizes past -2 dB SI-SDR and VocalSet is added. Not urgent now — first confirm FDKFNet is learning at all on the hard scenarios.

---

## What The Listening Test Will Tell Us

Before doing P3-P5, listen to `mic.wav` vs `enhanced.wav` in scenarios **8_near_threshold** and **9_above_threshold**. These are the only honest tests.

- **If the model suppresses feedback but introduces artifacts:** capacity/architecture issue → P2/P3
- **If the model does nothing (pass-through):** training distribution issue → P1
- **If the model sounds good in simulation but bad on hardware:** sim-to-real gap → P4
- **If suppression is partial and inconsistent:** more training needed, SEQ_FRAMES=200 next

Run:
```bash
python eval/run_inference.py --val-dir data/listening_test/8_near_threshold \
                              --out-dir data/listening_test/8_near_threshold
python eval/run_inference.py --val-dir data/listening_test/9_above_threshold \
                              --out-dir data/listening_test/9_above_threshold
python eval/score.py --enhanced-dir data/listening_test --clean-dir data/listening_test
```
