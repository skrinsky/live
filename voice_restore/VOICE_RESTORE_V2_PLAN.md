# Voice Restore V2 Plan

## Goal

Improve post-notch vocal color restoration so it works better on both singing and speech, especially when the notch bank is moving over time with varying depth/Q/onset/release.

V1 is already a good proof of concept, but it is still biased toward voiced, pitch-stable material. V2 should stay lightweight and causal-friendly while getting better at:

- restoring speech consonants and noisy/aperiodic content
- using the actual notch pattern instead of inferring it indirectly
- avoiding unnecessary repainting of healthy bins
- staying temporally stable when notch states change frame to frame

## Current V1 Limits

- Input features are dominated by `F0` and the harmonic template.
- The model does not receive the actual notch mask, even though runtime knows it exactly.
- Loss focuses on mel reconstruction only, so the model can overcompensate healthy regions.
- There is no explicit temporal continuity term on the predicted restoration gain.
- Compensation can be applied too broadly because it is only gated by "not notched", not by "close enough to a notch to need repair".

## V2 Direction

Keep the same overall decomposition:

1. `feedback_detect` finds unstable frequencies
2. `NotchBank` keeps the loop stable
3. `voice_restore` repairs perceived color after the notch stage

But move the restorer from "pitch-first harmonic compensation" toward a more balanced "harmonic + aperiodic + mask-aware" restorer.

## V2 Architecture

### Inputs

Per-frequency spectral inputs:

- `log_mag_notched`
- `harmonic_template`
- `notch_strength` derived from the real dynamic notch mask
- `aperiodic_residual` = local detail remaining after subtracting a smoothed spectral envelope

Per-frame conditioning inputs:

- `f0_norm`
- `confidence`
- `delta_f0`
- `delta2_f0`
- `voiced_gate`
- `spectral_flux`
- `spectral_flatness`

### Model

- retain the lightweight frequency-conv + per-bin GRU structure
- add two output heads:
  - harmonic gain head for voiced frames
  - aperiodic gain head for unvoiced / noisy frames
- blend the two with `voiced_gate`

This keeps the model small while making speech restoration less dependent on harmonic structure.

### Compensation Rule

Compensation should stay restricted to safe bins, but also be concentrated near actual notch regions:

- compute a smoothed `repair_region` from the notch mask
- allow gain mainly in bins near the cuts
- keep far-away healthy bins close to identity

## V2 Training Changes

### Data / Features

- continue training with dynamic notch onset, release, depth, and Q
- build condition features from the notched signal, not only cached pitch
- keep speech and singing mixed in the training pool

### Losses

Primary:

- mel reconstruction loss on compensated vs clean magnitude

Add:

- identity loss outside the repair region so untouched bins stay untouched
- temporal continuity loss on predicted gains to reduce frame-to-frame flicker

## Implementation Phases

### Phase 1

- add the plan document
- add mask-aware / aperiodic-aware features
- add voiced vs aperiodic dual-head blending
- add repair-region-aware compensation
- add identity + temporal smoothness losses

### Phase 2

- wire the restorer after the live notch stage
- train on real notch logs captured from the detector in loop tests
- add objective speech intelligibility checks and speech-heavy listening tests

### Phase 3

- benchmark latency for live use
- test lighter GRU widths or quantization if needed
- explore a tiny waveform/postfilter branch only if STFT-only V2 plateaus

## Success Criteria

- clearer speech after dynamic notching, especially fricatives and consonants
- less "repainted" or over-bright sound on unnotched regions
- less pumping / flicker when notches engage and release
- better A/B result on both sung vowels and spoken voice
