# Training Debugging — What To Try If Current Run Fails

Current status: Training from scratch with SEQ_FRAMES=100, EPOCHS=300, best at -0.85 dB SI-SDR after epoch 150. Warm-starting with `--resume best.pt --lr 1e-5` for another 300 epochs.

**A run is "failing" if:** loss doesn't move below -1.5 dB SI-SDR after 100+ more epochs, or keeps oscillating positive.

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

## P1 — Increase Above-Threshold Training Ratio (easy, high impact)

**Problem:** 75% of training sequences use sub-Larsen gains (0.3–0.9). For those, doing nothing achieves good SI-SDR. The model may learn "mostly pass through, occasionally suppress." The above-threshold sequences (25%) are the only ones that actually require suppression — and they're a minority.

**Current setup** (`recursive_train.py`):
```python
ABOVE_THRESHOLD_PROB = 0.25   # check the actual variable name in the script
```

**Fix:** Increase to 0.5 (50/50 above/below threshold). The model needs to see feedback suppression as the norm, not the exception.

Also consider: a **curriculum** that starts at 50% above-threshold for the first 50 epochs, then drops to 30% — forces the model to learn suppression before learning refinement.

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

**Fix:** Add cosine similarity (real part of normalized cross-correlation) between mic and ref per ERB band — this captures phase alignment without blowing up the feature dimension:
```python
# Normalized cross-correlation (real part = cosine similarity in complex domain)
mic_norm  = mic_f / (mic_f.abs() + 1e-8)
ref_norm  = ref_f / (ref_f.abs() + 1e-8)
coherence = (mic_norm * ref_norm.conj()).real   # (B, F), range [-1, 1]
# ERB-compress it
coh_erb   = coherence @ self.mel_fb             # (B, n_bands)

feat = torch.cat([
    self._to_erb(mic_f.abs().pow(2)),
    self._to_erb(ref_power),
    self._to_erb(innovation.abs().pow(2)),
    coh_erb,                                    # +1 feature group
], ...)
```

GRU input grows from `3 × n_bands` to `4 × n_bands`. Minor param increase (~10%).

---

## P3 — Bigger GRU / More ERB Bands (moderate effort, moderate impact)

**Current:** `HIDDEN=128, N_LAYERS=2, N_BANDS=64`

**The constraint:** This runs on a Pi 5 in real time. At 10ms blocks (HOP=480), each block budget is ~5-8ms for inference. The GRU is the bottleneck.

**Options in rough priority:**
1. `N_BANDS=128` — finer frequency resolution for Q/R estimates. Narrow feedback peaks might fall between 64-band boundaries and get over-smoothed. Near-free in compute (ERB compression is a matrix multiply, not a recurrence).
2. `HIDDEN=256` — doubles GRU capacity. ~3-4x parameter increase but same recurrence depth. Estimate: still real-time on Pi 5 (profile first).
3. `N_LAYERS=3` — adds depth. Less impact than width. Not recommended without profiling.

Before changing: profile the current model on Pi 5 to know the real-time budget headroom:
```python
import time
model.eval()
for _ in range(1000):
    t0 = time.perf_counter()
    with torch.no_grad():
        _, H, P, h = model.forward_frame(mic_f, ref_f, H, P, h)
    print(time.perf_counter() - t0)
```

Target: < 3ms per block (leaves margin for STFT, I/O).

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
