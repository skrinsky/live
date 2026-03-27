# Standalone AI Feedback Suppressor — Project Notes

*Started March 2026 | Updated ongoing*

---

## The Idea

A standalone hardware box that suppresses acoustic feedback in live sound — **mic XLR in, XLR out to board or snake**. No DAW, no laptop, no send/return routing. Plug and play.

**Target market:** Venues, churches, small productions, and operators who don't have the technical background to EQ feedback by ear or configure a send/return chain. The people who need feedback suppression most are often the least equipped to set up the current solutions.

**Key differentiator vs. De-Feedback (Alpha Labs):**
- De-Feedback requires: dedicated Intel NUC + Windows LTSC + audio interface + DAW host + send/return routing + $499 license = $1,000–$2,000+ and significant setup
- This box: plug mic in, XLR out to snake. Done.
- Target price point: **$249**

---

## Competitive Landscape

### De-Feedback V1 (Alpha Labs) — The Benchmark

The first neural-network-based feedback suppressor for live sound. Launched November 11, 2024.

**How it works:** Continuous spectral separation via ML model trained on human vocals. Subtracts everything that isn't direct voice: feedback energy, room reverberation, ambient noise — all in one pass. Zero algorithmic latency.

**What's good:**
- No notch filter accumulation (traditional tools carve up the frequency response over a show)
- Proactive, not reactive — doesn't wait for a squeal
- Feedback suppression + dereverb + denoising in one pass
- Zero added plugin latency

**What's bad:**
- Vocal only — destroys instruments
- Requires 6ft+ mic-to-speaker minimum
- Needs dedicated Intel NUC + Windows LTSC (1 instance per channel)
- No AAX, Mac is Beta/unsigned
- $499 license + $500–$1,500 hardware = $1,000–$2,000+ per rig
- Single Strength slider — no visualization, no tuning
- No hardware card or console-native version; requires send/return loop through external computer

**Hardware reality:**
- Entry: Intel NUC (N95) + Focusrite Solo — ~$514, 1 instance at ~4.9ms roundtrip
- Pro: Larger Intel NUC + Focusrite 18i16 Gen4 — ~$1,569, up to 4 instances

**Data approach:** Devin Sheets collected real venue impulse responses nationwide (churches, venues) physically — likely sine sweep measurements. "Several terabytes of customized audio data." This real-world IR collection is believed to be a major reason the product generalizes to real PA environments better than academic models trained only on synthetic shoebox rooms.

**IR measurement configuration for feedback suppression:**
- Source: PA loudspeakers driven with sine sweep
- Receiver: measurement mic at performer position (on stage, at working height)
- This captures the loudspeaker → mic feedback path specifically
- Need multiple positions per venue, multiple speaker configs (mains only, monitors only, combined)
- *Not* mix position — the feedback path is speaker → stage mic, not speaker → FOH

### Traditional Feedback Suppressors

| Product | Price | Method | Notes |
|---|---|---|---|
| dbx AFS2 | ~$300 | Notch filters (1/80 oct) | Most precise traditional option; nearly inaudible per notch |
| dbx DriveRack PA2 | ~$350 | AFS + full speaker management | Best value for full PA management |
| Behringer FBQ2496 | ~$150 | Notch filters (1/60 oct) | Budget entry; mixed reviews at low signal levels |
| Sabine FBX series | $150–$300 | Notch filters (1/10 oct) | Wider notch = more coloration; original category creator |
| Waves X-FDBK | ~$80 | Notch filters (adaptive) | Software only; SoundGrid or DAW required |

**Core problem with all traditional tools:** They leave a growing graveyard of carved frequencies that audibly color the system over a long show. The AI approach avoids this entirely.

---

## AI in Live Sound — Broader Opportunity Map

Beyond feedback suppression, high-value places for AI in the live signal chain:

1. **Continuous per-channel adaptive EQ** — rooms change over a show (crowd, temp, humidity); all current room EQ is a soundcheck snapshot
2. **Intelligent automixing** — Dan Dugan's algorithm is 1970s DSP; ML could understand context (intentional pause vs. finished sentence, overlapping speech)
3. **Real-time source separation / bleed reduction** — reduce drum bleed in a vocal mic, guitar amp bleed into overheads, in real time
4. **Predictive feedback detection** — detect pre-feedback resonance buildup before audible squeal (50–200ms before onset)
5. **Intelligent monitor mix generation** — learn musician preferences across shows, auto-generate starting point monitor mix
6. **Hearing-aid-derived directional processing** — billions spent training models to isolate a speaker in crowd noise; directly applicable to live vocal isolation

---

## Training Data

### What's Publicly Available

| Data Need | Best Public Source | Notes |
|---|---|---|
| Clean dry vocals | **EARS** (Meta/UHH, 107 speakers, 100h, 48kHz anechoic) | Best single source; open source |
| Clean dry vocals | VCTK (110 speakers, studio-dry) | Standard fallback |
| Clean singing | NUS-48E, OpenSinger | Relevant since target is vocalists |
| Room IRs | **OpenSLR-28** (115K synthetic + 3K real) | Apache 2.0; used by DNS Challenge |
| Room IRs | RIR-Mega (50K simulated, with metadata) | On Hugging Face |
| Real venue IRs | Pori Concert Hall (Aalto University) | Multi-position real measurements |
| Noise floor | AudioSet, Freesound, DEMAND | Standard noise sets |
| Synthesis pipeline | **Microsoft DNS Challenge scripts** | Open-source; handles convolution + noise mixing |

### What Must Be Generated

**Real acoustic feedback/howling data — no public dataset exists at scale.** Every academic paper synthesizes it.

**Feedback loop simulation approaches:**

- **Teacher forcing** (start here): Use clean vocal in the loop instead of model output during training. Turns AHS into a speech separation problem. Creates train-inference mismatch but acceptable for prototype.
  - `mic_signal = clean_vocal + (RIR_feedback * clean_vocal * loop_gain)`
  - Vary loop gain across range below Larsen threshold

- **Recursive / in-a-loop training**: Feed model's own output back through the simulated loop during training. Closes the mismatch but can go numerically unstable. Requires pretrained initialization.

- **Hybrid**: Teacher forcing → recursive fine-tuning. Best results in practice.

**Key insight:** Real venue IRs (measured physically) are what separate a model that generalizes to real PA environments from one that only works on simulated shoebox rooms. This is the moat Sheets built. Collecting real loudspeaker → mic position IRs across venue types (churches, theaters, gymnasiums, outdoor stages) is a direct path to outperforming academic baselines.

### Relevant Academic Work

| Paper | arXiv | Key Contribution | Code |
|---|---|---|---|
| Deep AHS | 2302.09252 | First DL approach to acoustic howling suppression; teacher forcing | No |
| Recursive Training AHS | 2309.16048 | In-a-loop recursive training to close train-test mismatch | No |
| NN-Kalman AHS | 2309.16049 | Neural network + Kalman filter hybrid | No |
| Multichannel AHS (Meta) | 2505.15914 | CRN 684K params / 82M MACs/s; Interspeech 2025 | No |

No open-source deep learning AHS implementation exists. Will need to build on speech enhancement code and implement the feedback loop simulator from scratch.

---

## Model Architecture

### Target: GTCRN

**[GTCRN](https://github.com/Xiaobin-Rong/gtcrn)** (Grouped Temporal CRN) — best fit for embedded deployment:
- **48K parameters, 33 MMACs/s**
- Real-time factor: **0.07 on an i5 CPU** (runs 14x faster than real-time on a laptop)
- Fully causal / streaming (dedicated `stream/` folder in repo)
- Pretrained checkpoints available (DNS3, VCTK-DEMAND)
- Open source, fine-tunable

**Backup: DeepFilterNet3** — 2.3M params, best pretrained weights of any open-source model, handles denoising + partial dereverb. Use as fine-tuning base if GTCRN proves insufficient quality.

**Reference: Meta/Inria CRN** (arXiv 2505.15914) — 684K params, 82M MACs/s, Interspeech 2025. Not open source but architecture is reproducible from the paper. Sits between GTCRN and DCCRN in size — good quality target for final model.

### Training Pipeline

1. Clean vocals: EARS dataset
2. Room IRs: OpenSLR-28 + real measured IRs (collected in field)
3. Synthesis: DNS Challenge scripts (extend to add feedback loop step)
4. Feedback simulation: pyroomacoustics for RIR generation, teacher forcing to start
5. Loss: multi-scale STFT loss + perceptual loss (not MSE — MSE models over-suppress and artifact)
6. Compute needed: ~6–24 GPU hours on A100 depending on model size; 16GB VRAM sufficient

---

## Hardware Platform

### Compute

**Target: Raspberry Pi CM4 or Pi 5 on custom PCB**

| Platform | Notes |
|---|---|
| RPi CM4 | Cortex-A72, proven embedded audio platform, custom PCB friendly, Elk Audio OS support |
| RPi 5 | Cortex-A76, ~2x faster than CM4 for inference — preferred |
| Rockchip RK3588 | 6 TOPS NPU — could run larger models; Radxa/Orange Pi boards available |

GTCRN at 0.07x real-time factor on an i5 likely has headroom on Pi 5's A76 cores for a single channel. Needs validation on actual hardware.

### Audio I/O Hardware

- **XLR balanced input** with phantom power (48V) — input stage IC (that128, SSM2019, or similar)
- **ADC/DAC** — low-latency audio codec via I2S (AK4558, CS4272)
- **XLR balanced output** — differential driver
- **Power** — wall wart or IEC inlet; needs 48V rail for phantom + 5V/3.3V for compute

### Latency Budget

At 48kHz, 256-sample buffer:
- Buffer period: 5.3ms
- ADC + DAC: ~1ms each end
- Model inference must complete within one buffer period
- **Target total: <10ms end-to-end** — under perceptible threshold for most live applications

### Software Stack

- **OS:** Elk Audio OS (real-time Linux tuned for embedded audio) or Pi OS with RT kernel patch
- **Inference:** ONNX Runtime (export from PyTorch → ONNX)
- **Audio I/O:** ALSA or JACK at low buffer sizes

### Reference Projects

- **Elk Audio OS** — real-time Linux for embedded audio on CM4 + custom hardware
- **Bela** — BeagleBone-based embedded audio platform, ultra-low latency PRU
- **Neutone SDK** — Python/PyTorch → VST wrapper (for DAW prototype validation before hardware)
- **ANIRA + JUCE** — C++ ONNX inference in a VST3 plugin (for distributable plugin version if needed)
- **RTNeural** — optimized LSTM/CRN inference in C++ for embedded audio

---

## Market & Sales Analysis

### De-Feedback Pricing (Confirmed)
- **$499 perpetual license** — covers all sub-version patches, not major upgrades
- 10% discount code ran through February 2026 via ProSoundWeb promo
- No low-cost tier, no subscription model
- Total system cost with required hardware: **$1,000–$2,000+**
- Priced and architected for professional touring production — not accessible to small church or working band market

### Online Presence Summary
| Platform | Signal |
|---|---|
| ProSoundWeb | Active forum thread + Signal to Noise Ep. 315 podcast feature |
| Gig Gab Podcast | Ep. 524 (March 2026) |
| Sonic State | News article Jan 2026 |
| Gig Performer forum | 3 replies — mixed |
| Fourier Audio forum | 9 posts — mixed/critical |
| Allen & Heath forum | Feature request thread (native integration) |
| AudioSEX Pro | 0 replies |
| Facebook | Official user group exists; size unknown |
| **Reddit** | **Effectively zero** — no indexed threads in r/livesound, r/churchsound, r/audioengineering, or anywhere else |

The Reddit gap is significant. r/livesound and r/churchsound are exactly the target customer for a simpler solution — semi-pro engineers, church AV volunteers, venue house engineers. De-Feedback has no presence there not because those people don't have the problem, but because the solution doesn't fit their world.

### Estimated De-Feedback Sales (16 months, Nov 2024 – Mar 2026)
Based on online signal density, forum engagement, and niche professional podcast reach:
- **Estimated range: 300–2,000 units**
- **Midpoint estimate: ~800 units**
- **Estimated gross revenue: ~$400K at midpoint**
- Entirely bootstrapped / self-funded; no VC, no crowdfunding found

### Cost Per Unit Estimates
| Component | 100 units | 1,000 units | 5,000 units |
|---|---|---|---|
| Raspberry Pi CM4 (2GB/16GB) | ~$40 | ~$30 | ~$25 |
| Audio codec + passives | ~$15 | ~$12 | ~$9 |
| XLR connectors (2x Neutrik) | ~$8 | ~$6 | ~$4 |
| Phantom power circuit | ~$12 | ~$10 | ~$8 |
| PCB fab + assembly | ~$50 | ~$20 | ~$12 |
| Enclosure | ~$20 | ~$15 | ~$12 |
| Power supply | ~$8 | ~$6 | ~$5 |
| Misc (LEDs, hardware, packaging) | ~$12 | ~$8 | ~$6 |
| **Total BOM** | **~$165** | **~$107** | **~$81** |
| **With 20% overhead/QA** | **~$200** | **~$130** | **~$97** |

At $249 retail: ~$50 margin at 100 units (R&D/early territory), ~$119 margin at 1,000 units (viable), ~$152 margin at 5,000 units (healthy).

### Sales Projections at $249

| Scenario | Year 1 | Year 2 | Year 3 |
|---|---|---|---|
| Conservative | 300 units / $75K | 700 / $175K | 1,200 / $300K |
| Moderate | 800 units / $200K | 2,000 / $500K | 4,000 / $1M |
| Optimistic | 2,000 units / $500K | 5,000 / $1.25M | 10,000 / $2.5M |

**Sweetwater is the key distribution unlock** — they have a dedicated HOW sales team and actively court the church AV market. A Sweetwater placement likely 3–5x organic sales rate. The dbx AFS2 at ~$300 with consistent backorder demand proves the market buys dedicated hardware feedback processors at this price point.

### Market Segments
- **House of worship (primary):** ~30,000–70,000 US churches with meaningful sound reinforcement; $1B+ annual HOW AV market; volunteer operators are exactly the "can't EQ feedback by ear" customer
- **Small venues / bars:** ~50,000 potential buyers, consistent feedback problems, no budget/expertise for complex solutions
- **Working bands / small production companies:** More technically capable but value simplicity and reliability of hardware over laptop-dependent setup

---

## Development Roadmap

### Phase 1: Software Prototype (Weeks 1–4)
- [ ] Set up training environment; download EARS + OpenSLR-28
- [ ] Extend DNS Challenge synthesis scripts with feedback loop simulator (pyroomacoustics)
- [ ] Fine-tune GTCRN on synthesized feedback data (teacher forcing)
- [ ] Wrap with Neutone SDK; validate in DAW
- [ ] Listening tests; iterate

### Phase 2: Real IR Collection (Ongoing from Phase 1)
- [ ] Define measurement protocol: sine sweep through PA, capture at mic position on stage
- [ ] Collect IRs across venue types: churches, small theaters, gymnasiums, outdoor
- [ ] Multiple mic positions per venue, multiple speaker configs (mains, monitors, combined)
- [ ] Integrate real IRs into training pipeline alongside simulated ones
- [ ] Retrain; observe generalization improvement

### Phase 3: Embedded Validation (Weeks 5–8)
- [ ] Port ONNX model to Raspberry Pi 5 (dev board + USB audio interface)
- [ ] Validate inference latency and CPU headroom at target buffer sizes
- [ ] Stress test: multiple hours continuous operation, thermal stability

### Phase 4: Hardware Design (Weeks 8–16)
- [ ] Custom PCB: CM4 compute module socket + audio codec + XLR I/O + phantom power
- [ ] Enclosure design: small rackmount (1/2U) or DI-box form factor
- [ ] Prototype assembly and testing

### Phase 5: Product
- [ ] Determine enclosure / form factor
- [ ] Pricing strategy ($150–$300 target)
- [ ] Licensing / distribution

---

## Open Questions

- How many real venue IRs are needed before training meaningfully outperforms public IR datasets?
- GTCRN inference speed on Pi 5 at 256-sample buffer — needs hardware validation
- Phantom power: pass-through vs. generated on-board
- Form factor preference: rackmount vs. DI box style
- Single channel only, or stereo/dual-channel version?
- Does the model need separate training for handheld dynamic vs. condenser vs. headset mic types?
