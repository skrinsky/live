# AI in Live Sound: De-Feedback & Beyond

*Research compiled March 2026*

---

## De-Feedback V1 (Alpha Labs) — Overview

**Alpha Labs LLC** spun out of Alpha Sound, a Pacific Northwest PA rental and installation company. Devin Sheets (founder) built De-Feedback after becoming frustrated with existing feedback solutions — generating custom training data from venue impulse responses gathered nationwide, training the model on dedicated hardware, and running beta tests throughout 2024 in regional churches and his own productions.

Launched November 11, 2024. Reportedly appeared on Mariah Carey's holiday tour within days of release.

---

## How It Works (Technically)

De-Feedback is **the first neural-network-based feedback suppressor for live sound.** Instead of the traditional approach (detect squeal → notch filter that frequency), it performs **continuous spectral separation** — the model was trained on human vocals and attempts to subtract everything that isn't direct voice:

- Feedback loop energy
- Room reverberation
- Ambient noise / floor

All three in a single real-time pass. Zero algorithmic latency added.

Trained on several terabytes of custom audio data. Processing range: 100 Hz – 22 kHz. Supported sample rates: 44.1–96 kHz. Mono and stereo.

The single user-facing control is a **Strength slider (0–100%).** Developers recommend leaving it at 100%.

---

## What's Genuinely Impressive

- **No notch filter accumulation** — traditional tools leave a growing graveyard of carved frequencies that audibly color the system over a long show. This doesn't.
- **Proactive, not reactive** — it doesn't wait for a squeal to begin.
- **Three tools in one** — feedback suppression + room dereverb + noise reduction, simultaneously.
- **Zero added plugin latency** — a real engineering achievement. Most traditional hardware feedback suppressors add 3–10ms of algorithmic delay on top of I/O latency.
- Real-world validation on major touring productions.

---

## Where It Falls Short

| Problem | Detail |
|---|---|
| **Voice only** | Trained on human vocals — destroys instruments. Not usable elsewhere in the signal chain. |
| **6ft+ mic-to-speaker minimum** | Physics still wins at close range. Performance degrades below this distance. |
| **Dedicated hardware required** | 1 instance per channel requires a specific Intel NUC + Windows LTSC. A $2k AMD workstation barely outperforms it. |
| **No AAX** | Pro Tools users are out. Developer response: "Sell your Bitcoin, fund us, then maybe." |
| **Mac is Beta / unsigned** | Requires manual Gatekeeper security workarounds. |
| **Cost** | $499 license + $500–$1,500 hardware = $1,000–$2,000+ per rig. A dbx DriveRack PA2 handles most gigs for $350 total. |
| **One instance = one channel** | 4 vocalists = 4 instances = near the limit of their recommended hardware configuration. |
| **Black box** | No frequency visualization, no insight into what the model is doing or suppressing. |

### Hardware Reality

The plugin is so CPU-intensive that the developers spent ~1 year testing hardware before committing to two verified configurations:

- **Entry:** Intel NUC (N95) + Focusrite Solo — ~$514 total, 1 instance at ~4.9ms roundtrip
- **Pro:** Larger Intel NUC + Focusrite 18i16 Gen4 — ~$1,569 total, up to 4 instances at ~4.9ms

A Waves SuperRack LiveBox gets ~8 instances. Moving windows or accessing menus on the entry NUC can cause audio glitches during live use.

---

## Versus Traditional Tools

### How Traditional Feedback Suppressors Work

Acoustic feedback occurs when loop gain exceeds unity at a frequency where loop phase aligns. Traditional suppressors detect the sine wave and insert a notch filter at that frequency. Differences between products come down to filter speed, notch width, number of simultaneous filters, and whether filters are fixed or adaptive.

### dbx AFS2 (~$300)
The most technically precise traditional option. 1/80-octave notch width (nearly inaudible per notch), adaptive bandwidth consolidation, fixed + live filter modes, setup wizard. For most applications this is 80% of the result at 20% the cost of De-Feedback. Sound on Sound: "the increase in available volume was impressive."

### dbx DriveRack PA2 (~$350)
Full speaker management system (crossover, limiter, EQ, sub alignment) with AFS built in. Best value for full PA management.

### Behringer FBQ2496 (~$150)
Budget entry point. Up to 40 filters, 1/60-octave width, 0.2-second detection. Mixed reviews at very low signal levels. Adequate for budget applications.

### Sabine FBX Series (~$150–$300)
One of the original automatic feedback suppressors. Fixed + live filter banks. Notch width wider than dbx — more coloration per suppressed frequency.

### Waves X-FDBK (~$80) / Feedback Hunter (~$100)
Software-based within the Waves SoundGrid ecosystem. X-FDBK is reactive/dynamic during performance. Feedback Hunter is a pre-show analysis tool that generates a static GBF-maximizing EQ curve for soundcheck. Both require SoundGrid infrastructure or a DAW host.

---

## How To Make De-Feedback Better

### Near-Term

1. **Multi-source models** — Train separate models for instruments. The vocal-only constraint is the single biggest limitation. A keyboard player holding a mic, a guitar amp in a small room, drum overheads — all need this.

2. **Suppression visualization** — A real-time FFT overlay showing what the model removed would build engineer trust and help diagnose artifacts. The black-box single-knob interface is a faith-based proposition for skeptical engineers.

3. **Adaptive strength** — Instead of a flat slider, auto-modulate suppression depth based on direct signal quality: pull back when the vocal is strong and clean, push harder when room energy rises.

4. **NPU/GPU inference offload** — Modern Intel NPUs (Neural Processing Units, 12th Gen+ Core) are specifically designed for this kind of model inference. Offloading from CPU to NPU could dramatically increase instances-per-machine and reduce hardware costs.

5. **Console DSP integration** — The real game-changer. Licensing the model to run natively on Yamaha/Allen & Heath/DiGiCo DSP hardware would enable mass adoption. Licensing complexity is the current stated barrier.

### Bigger Swings

6. **Cross-channel acoustic modeling** — One model aware of all open mics simultaneously, modeling the full acoustic loop. Currently every instance is isolated and independent; coordinated suppression would be more physically accurate.

7. **Room-adaptive fine-tuning** — A brief soundcheck measurement that adapts the model to the specific venue's acoustic fingerprint, rather than relying on generalized training data across all venues.

---

## Other High-Value Places for AI in the Live Signal Chain

### 1. Continuous Per-Channel Adaptive EQ
All current room EQ is a **snapshot** — measure at soundcheck, apply a fixed curve, done. But rooms change: crowd density absorbs HF, temperature and humidity drift over a 3-hour show. An AI model continuously monitoring the FOH response and micro-adjusting system EQ in real time (below perceptible threshold of change) would meaningfully improve consistency through a show. Architecturally similar to De-Feedback but applied to PA output rather than mic input.

### 2. Intelligent Automixing
Dan Dugan's gain-sharing automix (embedded in Yamaha CL/QL series, Waves, Shure SCM820) is 1970s-era algorithmic DSP — not ML. An ML-based automixer trained on thousands of hours of live performance could learn **context**: distinguish intentional pause from finished sentence, handle overlapping speech, recognize who is likely to speak next. High-value for house of worship, corporate AV, and theatrical applications.

### 3. Real-Time Source Separation / Bleed Reduction
Post-production already has this (iZotope Music Rebalance, Demucs, Spleeter). Bringing ML source separation into the live chain at usable latency would enable: reducing drum bleed in a vocal mic in real time, reducing guitar amp bleed into drum overheads, recovering GBF from monitor mics that pick up stage wash. Compute and latency requirements are the current barrier — but they're approaching tractable.

### 4. Predictive Feedback Detection
Every current tool — including De-Feedback — responds to feedback energy that's already present. A model trained on the **pre-feedback spectral signature** (the resonance accumulation that occurs 50–200ms before audible squeal) could suppress the loop before it's audible. This requires a specialized dataset of pre-feedback audio, but it's a solvable data problem.

### 5. Intelligent Monitor Mix Generation
Monitor mixing is often the most technically demanding and time-pressured part of live sound. An AI that learns each musician's preferences across multiple shows and auto-generates a starting point monitor mix (based on input list, genre, and historical preference data) would save significant soundcheck time. More "ML-assisted workflow" than real-time signal processing, but highly practical and immediately monetizable.

### 6. Hearing-Aid-Derived Directional Processing
The hearing aid industry has spent billions of dollars training ML models to isolate a target speaker in a noisy crowd. That research is directly applicable to isolating a vocalist against a loud stage — a fundamentally identical acoustic problem. Those neural architectures, adapted for professional audio fidelity and latency requirements, are a largely untapped crossover opportunity.

---

## Current AI Audio Landscape (What Already Exists)

| Product | Type | AI? | Live Use? |
|---|---|---|---|
| De-Feedback V1 (Alpha Labs) | Feedback suppression | Neural network | Yes — purpose built |
| CEDAR DNS 8D | Noise suppression | ML-hybrid | Yes — broadcast/theater |
| iZotope RX 11 Dialogue Isolate | Source separation | Neural network | Limited — post-production |
| NVIDIA Broadcast / RTX Voice | Noise suppression | Neural network | No — streaming/conferencing only |
| Waves X-FDBK | Feedback suppression | No — DSP notch filter | Yes |
| Dugan Automixer | Automixing | No — algorithmic gain-sharing | Yes |

---

## Bottom Line

De-Feedback is genuinely novel and the results are credible — it solves a real problem in a fundamentally different way than anything else on the market. But it's a V1, and it shows: vocal-only, hardware-intensive, platform-limited, expensive, and opaque.

The right customers are touring production companies and high-budget HOW installs with difficult acoustics who can absorb the infrastructure cost and justify the per-channel price.

The broader opportunity space is large. The live sound field is roughly 5 years behind studio/post-production in ML adoption, almost entirely because of the hard latency constraint — a studio reverb tail can absorb 500ms of processing delay invisibly; a live signal chain cannot tolerate more than ~5ms before performers feel it. That constraint eliminates most of the ML architectures powering impressive studio tools today.

As NPU compute and model compression/quantization techniques mature, that gap will close fast. The console manufacturers that figure out how to run this class of inference natively in their DSP hardware will have a significant competitive advantage.
