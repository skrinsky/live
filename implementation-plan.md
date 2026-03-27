# Implementation Plan — Standalone AI Feedback Suppressor

*Every step required to get the algorithm working on a live mic signal.*
*Hardware / PCB / enclosure work is tracked separately.*

---

## Phase 0: Environment Setup

### 0.1 Python Environment
```bash
python3 -m venv venv && source venv/bin/activate

pip install torch torchaudio
pip install pyroomacoustics
pip install soundfile librosa numpy scipy freesound
pip install sounddevice            # live mic I/O
pip install "pesq>=0.0.4"          # evaluation metrics — >=0.0.4 required for wb mode at 16kHz
pip install "pystoi>=0.3.3"        # STOI — >=0.3.3 for correct 48kHz handling
pip install tensorboard            # training monitoring
pip install matplotlib             # visualization
pip install tqdm                   # progress bars
```

### 0.2 Clone Repos
```bash
git clone https://github.com/microsoft/DNS-Challenge      # synthesis pipeline + noise data
git clone https://github.com/LCAV/pyroomacoustics         # RIR simulation
git clone https://github.com/RoyJames/room-impulse-responses  # IR dataset index

# Reference papers (not direct code dependencies):
# - NeuralKalmanAHS (arXiv 2309.16049) — FDKFNet architecture basis
# - Recursive AHS training (arXiv 2309.16048) — recursive training methodology
# FDKFNet model is implemented in train/model.py (home-grown PyTorch, ~500K params)
```

### 0.3 Directory Structure
```
project/
├── data/
│   ├── clean_vocals/          # EARS + VCTK dry vocal files
│   ├── public_irs/            # downloaded public IR libraries
│   ├── venue_irs/
│   │   ├── mains/             # IRs captured with mains only active
│   │   ├── monitors/          # IRs captured with monitors only active
│   │   └── combined/          # IRs captured with full system active (reserved for future use)
│   ├── noise/                 # AudioSet / DEMAND noise files
│   ├── transducer_frs/
│   │   ├── mics/              # per-mic FR CSVs (freq_hz, magnitude_db)
│   │   └── speakers/          # per-speaker FR CSVs
│   └── training_pairs/
│       ├── train/             # 50,000 generated (mic, clean, ref) triples
│       └── val/               # 1,000 held-out triples for evaluation
├── simulator/                 # feedback loop synthesis code
├── measurement/               # venue sweep scripts
├── train/                     # training scripts + configs
├── checkpoints/
│   └── fdkfnet/               # FDKFNet trained checkpoints (best.pt, last.pt)
├── eval/                      # evaluation scripts
└── inference/                 # real-time inference script
```

> **Training pairs are triples:** `mic_{idx}.wav` (input), `ref_{idx}.wav` (reference — reverberant vocal before speaker, used by FDKFNet Kalman filter as H(z) reference), `clean_{idx}.wav` (target). See Phase 3 for why the reference signal is needed.

### 0.4 API Smoke Test

**Run this before generating IRs.** Two minutes now saves hours of debugging:

```bash
# Run from project root after pip installs complete
python - <<'EOF'
import sys

# ── 1. pyroomacoustics directivity API ──────────────────────────────────────
try:
    from pyroomacoustics.directivities import (
        CardioidFamily, DirectivityPattern, Rotation3D
    )
    import pyroomacoustics as pra
    import numpy as np

    # Test Rotation3D construction
    rot = Rotation3D([0.0, 0.0, 0.0], 'yzx', degrees=True)

    # Test CardioidFamily construction
    src = CardioidFamily(
        orientation=rot,
        pattern_enum=DirectivityPattern.CARDIOID,
        fs=48000,
    )

    # Test a minimal room simulation with directivity
    room = pra.ShoeBox([5, 4, 3], fs=48000,
                       materials=pra.Material(0.2), max_order=3)
    room.add_source([1, 1, 1.5], directivity=src)
    room.add_microphone([3, 2, 1.5])
    room.simulate()
    ir = room.rir[0][0]
    assert len(ir) > 10, "IR too short — simulation may have failed"
    print("✓  pyroomacoustics directivity API OK "
          f"(pra {pra.__version__}, IR length {len(ir)} samples)")

    # ── Monitor wedge orientation sanity check ────────────────────────────────
    # A wedge on the floor (0.3m height) should fire backward+upward toward a
    # performer's capsule (~1.5m height, ~1m behind). Verify that Rotation3D
    # with azimuth=180 + positive elevation produces a non-trivial IR vs a
    # no-directivity source in the same position. If the directivity IR RMS is
    # more than 40dB below the omni IR RMS, the elevation rotation is probably
    # horizontal (panning) rather than vertical (tilting), and the rotation
    # order 'yzx' may need to change to 'zyx'.
    room2 = pra.ShoeBox([6, 6, 4], fs=48000,
                        materials=pra.Material(0.3), max_order=3)
    # Monitor wedge: floor level in front of performer, firing backward+upward
    wedge_pos = np.array([3.0, 2.5, 0.3])   # downstage of mic
    mic2_pos  = np.array([3.0, 3.5, 1.5])   # performer capsule, 1m upstage
    wedge_dir = CardioidFamily(
        orientation=Rotation3D([180, 40, 0], 'yzx', degrees=True),
        pattern_enum=DirectivityPattern.CARDIOID,
        fs=48000
    )
    room2.add_source(wedge_pos.tolist(), directivity=wedge_dir)
    room2.add_microphone(mic2_pos.tolist())
    room2.simulate()
    ir_wedge = room2.rir[0][0]
    # Omni reference
    room3 = pra.ShoeBox([6, 6, 4], fs=48000,
                        materials=pra.Material(0.3), max_order=3)
    room3.add_source(wedge_pos.tolist())
    room3.add_microphone(mic2_pos.tolist())
    room3.simulate()
    ir_omni = room3.rir[0][0]
    min_len = min(len(ir_wedge), len(ir_omni))
    rms_wedge = np.sqrt(np.mean(ir_wedge[:min_len]**2)) + 1e-12
    rms_omni  = np.sqrt(np.mean(ir_omni[:min_len]**2))  + 1e-12
    db_diff = 20 * np.log10(rms_wedge / rms_omni)
    if db_diff < -40:
        print(f"⚠️  WEDGE ORIENTATION WARNING: directivity IR is {db_diff:.1f}dB below omni.")
        print("   The elevation rotation may be horizontal (panning) rather than vertical.")
        print("   Expected range: -10 to -25dB (cardioid rejection at ~140° off-axis).")
        print("   If this is wrong, change 'yzx' → 'zyx' in build_room_simulation() and retest.")
    else:
        print(f"✓  Monitor wedge orientation OK ({db_diff:.1f}dB vs omni — expected -10 to -25dB)")

except Exception as e:
    print(f"✗  pyroomacoustics directivity FAILED: {e}")
    print("   Check: pip install pyroomacoustics>=0.7.3")
    sys.exit(1)

# ── 2. torchaudio melscale_fbanks ───────────────────────────────────────────
try:
    import torchaudio
    import torch

    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=481, f_min=80.0, f_max=24000.0,
        n_mels=64, sample_rate=48000, norm=None
    )
    assert fb.shape == (481, 64), \
        f"Expected (481, 64), got {tuple(fb.shape)}"
    print(f"✓  torchaudio.functional.melscale_fbanks OK "
          f"(torchaudio {torchaudio.__version__}, shape {tuple(fb.shape)})")

except Exception as e:
    print(f"✗  melscale_fbanks FAILED: {e}")
    print("   Check: pip install torchaudio>=0.12")
    sys.exit(1)

# ── 3. FDKFNet model instantiation ──────────────────────────────────────────
try:
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path('train')))
    from model import FDKFNet
    import torch

    m = FDKFNet()
    n_params = sum(p.numel() for p in m.parameters())
    H, P, h = m.init_state(1, 'cpu')
    mic_f = torch.zeros(1, 481, dtype=torch.cfloat)
    ref_f = torch.zeros(1, 481, dtype=torch.cfloat)
    speech_f, H2, P2, h2 = m.forward_frame(mic_f, ref_f, H, P, h)
    assert speech_f.shape == (1, 481)
    print(f"✓  FDKFNet OK ({n_params:,} parameters, output shape {tuple(speech_f.shape)})")

except Exception as e:
    print(f"✗  FDKFNet FAILED: {e}")
    sys.exit(1)

print("\nAll checks passed — safe to run generate_ir_pool.py")
EOF
```

If any check fails, fix it before proceeding to Phase 3.

---

## Phase 1: Public Dataset Acquisition

### 1.1 Clean Vocal Data

**Critical:** include both speech and singing datasets. EARS and VCTK are speech-only. Your product is primarily used for singing voices. Sustained vowels with vibrato, belting, falsetto, and held consonants ("sss", "fff") are phonetically different from speech in exactly the ways that matter for feedback — they produce sustained narrow-band energy that looks like incipient feedback to a model trained only on speech. A model trained on speech alone will likely produce artifacts (tremolo, pitch smearing) on held notes — which is the most common real-world failure mode.

**Speech datasets:**
- [ ] Download **EARS** anechoic vocal dataset
  - URL: https://sp-uhh.github.io/ears_dataset/
  - 107 speakers, 100+ hours, 48kHz anechoic — primary clean speech source
- [ ] Download **VCTK** corpus as secondary speech source
  - URL: https://datashare.ed.ac.uk/handle/10283/3443
  - 110 speakers, studio-dry quality

**Singing datasets (required — not optional):**
- [ ] Download **VocalSet**
  - URL: https://zenodo.org/record/1442513
  - CC BY 4.0 — 10 singers, 17 vocal techniques: belt, vibrato, straight, breathy, trillo, marcato, falsetto, spoken
  - Covers exactly the techniques most vulnerable to feedback
  - 6.6 hours, 48kHz. Primary singing source.
- [ ] Download singing voice data for natural phrasing (VocalSet covers technique but not phrase variety)
  - **⚠️ LICENSE WARNING — this is a $249 commercial product:**
    - **iKala**: research-only license — **cannot be used commercially**. Do not use.
    - **MUSDB18**: CC BY-NC-SA 4.0 — non-commercial only. **Cannot be used commercially.**
  - **Safe options (permissive license):**
    - **VocalSet** (CC BY 4.0) — already required above; use it as the primary singing source
    - **OpenSinger** (check license per release) — large Chinese singing corpus, verify before use
    - **CSD (Children's Song Dataset)** — CC BY-SA 4.0, children's voices, limited applicability
    - **Record your own** — 30 minutes of isolated singing (scales, phrases, sustained notes) in a dead room is legally clean and acoustically ideal
  - If research-only pipeline is acceptable (not for the commercial product), iKala/MUSDB18 are fine for internal prototyping, but the final trained checkpoint must not be distributed if built on them
- [ ] Download **NUS-48E** (optional, adds diversity)
  - 12 singers performing 48 English songs — good for phrase-level singing
  - Available via NUS Computing (request form)

**Note on VCTK quality:** some VCTK files have audible room tone. If your preprocessing step (`preprocess.py`) doesn't filter these, the model will train on partially reverberant "clean" targets, which degrades training signal quality. Consider a quick energy/spectral flatness check to flag and exclude problematic files.

### 1.2 Public Room Impulse Responses
These are useful for the **dereverberation component** of training.
They are NOT measured with the right protocol for feedback path simulation
(all use omnidirectional sources, not PA loudspeakers).
Use them to teach the model about room acoustics generally.

- [ ] **OpenAIR** — openairlib.net
  - 13 confirmed churches (1st Baptist Nashville, York Minster, St. Paul's, etc.)
  - 4+ performance venues and large halls
  - B-format Ambisonics + mono/stereo, 48kHz/24-bit
  - CC licensed (check each space; some NC — fine for research)
  - Download bulk archive if available; otherwise download per-space

- [ ] **C4DM — Queen Mary University London**
  - URL: https://zenodo.org/record/6497436  (old isophonics.net URL archived ~2023 — use Zenodo)
  - Great Hall (800-seat venue with stage) + Octagon (Victorian domed hall)
  - 468 IRs at 96kHz/32-bit — highest quality source in this list
  - CC BY-NC-SA

- [ ] **Freesound CC0 church + venue IRs**
  - URL: freesound.org — filter by CC0 license
  - Search terms: "impulse response church", "impulse response theater",
    "impulse response auditorium", "impulse response gymnasium"
  - Use Freesound API to batch download:
    ```python
    import freesound, os
    client = freesound.FreesoundClient()
    client.set_token("YOUR_API_KEY")
    # NOTE: text_search() returns a Pager object — max 15 results per page.
    # Must call results.next_page() to iterate. Collect all pages before downloading.
    all_sounds = []
    results = client.text_search(
        query="impulse response church",
        filter='license:"Creative Commons 0"',
        fields="id,name,previews,download",
        page_size=15
    )
    while results:
        all_sounds.extend(results)
        results = results.next_page() if results.count > len(all_sounds) else None
    for sound in all_sounds:
        sound.retrieve_preview(os.getcwd(), sound.name + ".mp3")
    ```

- [ ] **Aachen AIR**
  - URL: iks.rwth-aachen.de (AIR database section)
  - Includes Aula Carolina — large reverberant medieval church/great hall
  - MIT license — fully open

- [ ] **ARNI (Zenodo 6985104)**
  - URL: https://zenodo.org/record/6985104
  - 132,037 IRs from one variable-acoustics lab, CC BY 4.0
  - **Do NOT use all 132K** — they all come from one room with adjustable panels.
    Using the full set would massively over-represent one acoustic profile.
    Sample ~500 IRs spread evenly across the absorption range (RT60 range) instead.
  - **How to sort by RT60:** The ARNI filenames encode the panel configuration.
    Use REW or python-acoustics (`from acoustics.room import t60_impulse`) to compute
    RT60 of each IR, bin into ~10 RT60 ranges (0.2–0.4s, 0.4–0.6s, ... 2.0–2.5s),
    then sample ~50 from each bin. This gives acoustic diversity rather than random
    sampling (which would cluster at the most common panel setting).
    ```python
    from scipy.signal import fftconvolve
    import soundfile as sf, numpy as np
    def rt60_from_ir(ir, sr=48000):
        """Estimate RT60 via Schroeder backward integration."""
        energy = np.cumsum(ir[::-1]**2)[::-1]
        energy_db = 10 * np.log10(energy / (energy[0] + 1e-12) + 1e-12)
        # Find -5dB and -35dB crossings for EDT/T20
        t = np.arange(len(ir)) / sr
        try:
            idx_5  = np.where(energy_db < -5)[0][0]
            idx_35 = np.where(energy_db < -35)[0][0]
            return (t[idx_35] - t[idx_5]) * 2   # extrapolate to -60dB
        except IndexError:
            return None   # IR too short
    ```

- [ ] **BUT ReverbDB**
  - URL: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
  - 1,300+ IRs, office/lecture rooms, CC BY 4.0
  - These are small, close-mic'd, dead rooms — acoustically nothing like venues.
    Useful for raw training volume but deprioritize vs OpenAIR churches.
    Sample ~200 IRs max.

### 1.3 Noise Data
- [ ] Download DNS Challenge noise set (subset sufficient)
  - URL: https://github.com/microsoft/DNS-Challenge
  - AudioSet + Freesound clips across 150 audio classes

### 1.4 Transducer Frequency Response Data

Collect FR curves for the mics and speakers actually present in your target market.
These are applied as minimum-phase FIR filters to the simulated feedback path in
`generate_ir_pool.py`, so training data reflects the spectral coloration that determines
which frequencies feed back first in real use.

**Source options (in priority order):**
1. Manufacturer spec sheets — FR chart in the PDF. Digitize with WebPlotDigitizer (free, ~5 min/curve). Most curves are 1/3-octave smoothed at 0° on-axis — accurate enough for this purpose.
2. AudioScienceReview — measured PA speakers (QSC, JBL, Yamaha, EV, Mackie)
3. Sonarworks Reference database — live vocal mic measurements

**Format:** CSV with two columns: `freq_hz,magnitude_db`. Save to `data/transducer_frs/mics/` or `data/transducer_frs/speakers/` with a descriptive filename (e.g., `shure_sm58.csv`, `qsc_k12_2.csv`). One file per model.

**Target mic list:**

| Model | Type | Why it matters |
|---|---|---|
| Shure SM58 | Dynamic handheld | The default mic everywhere — presence peak ~5–10kHz |
| Shure Beta 58A | Dynamic handheld | More aggressive presence peak than SM58, ~8–12kHz |
| Sennheiser e835 | Dynamic handheld | Slightly flatter than SM58, very common |
| Sennheiser e945 | Dynamic handheld | Tighter supercardioid, different feedback profile |
| AKG D5 | Dynamic handheld | Multiple mid-presence bumps — common in smaller venues |
| Shure SM86 | Condenser handheld | HOW/praise band — flatter response, different feedback character |
| Shure KSM9 | Condenser handheld | Premium condenser — nearly flat |
| Shure ULX-D (TA87A capsule) | Wireless | Wireless HF rolloff + TA87A coloration |
| Shure ULX-D (Beta87A capsule) | Wireless | Beta87A presence curve through wireless path |
| Sennheiser EW (835-p capsule) | Wireless | Very common in churches — 835 character with HF rolloff |
| Sennheiser EW (945 capsule) | Wireless | Tighter pattern, different HF rollout than 835 |
| DPA 4099 | Instrument clip-on | Common in worship settings on acoustic instruments |
| Countryman E6 | Headset lav | Church headset — small capsule, brighter character |
| Shure MX153 | Headset lav | Common in HOW — small diaphragm, very flat |

**Target speaker list:**

| Model | Type | Why it matters |
|---|---|---|
| QSC K12.2 | Powered PA | Most common small venue/church main — relatively flat |
| QSC K10.2 | Powered PA | Smaller version — different HF extension |
| JBL SRX835P | Powered PA | Mid-size venue standard |
| JBL PRX835 | Powered PA | Very common church install |
| Yamaha DXR12 | Powered PA | Strong HOW market presence |
| EV ZLX-12P | Powered PA | Budget workhorse — significant cabinet coloration |
| Mackie Thump15A | Powered PA | Budget coloration characteristic of cheap boxes |
| RCF ART 745-A | Powered PA | Common in Italian/European market |
| Yamaha SM12V | Monitor wedge | Common passive wedge |
| JBL EON515XT | Monitor wedge (repurposed) | Very common as cheap monitor |
| QSC K12 used as monitor | Monitor wedge | Extremely common in small churches |

**Always include a `flat.csv` entry** (two points: 20Hz/0dB and 20000Hz/0dB) to represent the "no transducer coloration" case — about 15% of samples should use this to prevent the model from over-correcting signal that doesn't have transducer coloration.

**Acquisition notes:**
- Wireless mics: look for measurements of the complete system (capsule + bodypack/handheld transmitter + receiver), not just the capsule alone. The wireless RF path adds HF rolloff that the capsule-only spec doesn't show.
- Speakers: on-axis measurement at 1m is the correct reference. Avoid off-axis measurements — feedback arrives near on-axis at the mic position.
- Many QSC/JBL/Yamaha measurements are on AudioScienceReview's "Speaker Measurements" forum; EV and Mackie are patchier.

---

## Phase 2: Venue IR Measurement Program

> **Why this matters:** Every public IR library uses omnidirectional sources
> measured for general room acoustics. The feedback path we need is specifically
> PA loudspeaker → performer mic position — a directional, geometry-specific
> measurement that doesn't exist in any public dataset. These measurements are
> the proprietary training data that will differentiate the model.

### 2.0 Synthetic vs. Real IR Strategy (Research-Backed)

**The question:** Can we ship a competitive product without real venue measurements, or are they required?

**Short answer:** Synthetic IRs alone are enough to beat traditional notch-filter suppressors. Real venue IRs improve generalization further but are not required for v1.

---

#### Why synthetic IRs are sufficient to beat traditional suppressors

Traditional notch-filter feedback suppressors have no concept of voice. They detect a sustained narrow-band frequency spike and cut it — which often cuts fundamental voice frequencies, causing the classic "honky" artifact. They're also reactive: feedback is already audible before they trigger.

The ML model's advantage is voice-awareness baked into the weights from training on thousands of hours of clean speech. That advantage exists regardless of whether training IRs were real or synthetic. The voice/feedback separation capability comes from the speech training data, not the venue IRs.

---

#### The sim-to-real gap (and how to close it without real measurements)

Research confirms that basic ISM (Image Source Method) synthetic IRs — which is what pyroomacoustics generates — leave a measurable generalization gap vs. real-world deployment. The causes:

- **Single absorption coefficient across all frequencies** — physically wrong; real materials absorb differently at different frequencies
- **Cuboid room assumption** — no obstacle diffraction, irregular geometry, or non-parallel walls
- **No source/receiver directivity** — omni speaker, omni mic in simulation
- **Missing late reverberation tails** — ISM at max_order=10 doesn't model diffuse field accurately

The key research finding (MB-RIRs paper, 2025): **adding frequency-dependent absorption to ISM simulation** closed most of the gap vs. real measured IRs, gaining +0.51 dB SDR and +8.9 MUSHRA points in listening tests — *without any real room measurements*. The fix is physics accuracy, not real-world data collection.

SonicSim (ICLR 2025) confirmed: models trained on geometry-accurate synthetic IRs (Matterport3D mesh scenes) outperformed models trained on DNS Challenge data (real + naive ISM synthetic mix) when tested on real-world benchmarks. Quality beats quantity.

---

#### The staged IR strategy

**Prototype (no real venue IRs):**
- Upgrade `synthetic_feedback_ir()` in `simulator/generate_pairs.py` to use frequency-dependent absorption coefficients (pyroomacoustics supports `pra.Material` with per-band coefficients)
- Add source directivity modeling via `pra.DirectivityPattern` (pyroomacoustics has cardioid/subcardioid presets)
- This alone should produce a model substantially better than traditional suppressors
- Expected training mix: 100% synthetic IRs (mains + monitors simulated separately)

**V1 product (post-prototype, with real venue IRs):**
- Add real venue IRs at 60/40 real/synthetic ratio (as currently coded for mains/monitor paths)
- Target: 5–10 venues, 15 IRs each = 75–150 real feedback-path IRs
- Real venue IRs are especially valuable here because generic IR datasets capture room acoustics with omni sources — not the loudspeaker→mic feedback path. Our measurement protocol captures something that doesn't exist in any public dataset.
- Expected generalization improvement: meaningful for edge cases (unusual room shapes, non-standard speaker placement)

**What the DNS Challenge data shows:**
The DNS Challenge (Microsoft/ICASSP) provides ~3,000 real IRs and ~115,000 synthetic IRs — a 37:1 ratio — not because synthetic is preferred, but because real IRs are scarce. DeepFilterNet and GTCRN both use this mixed pool. Our feedback-specific IRs are more targeted than generic room IRs, which could make each real measurement punch above its weight.

---

#### Action items before generating training data

- [ ] **2.0a** — Replace `synthetic_feedback_ir()` in `simulator/generate_pairs.py` with `build_room_simulation()` from Phase 3.0. The full design (materials, archetypes, directivity, sub path, non-convex approximation) is specified in sections 3.0.1–3.0.7. This is the single implementation task — the partial upgrade steps originally listed here (2.0a–2.0d) are superseded by the 3.0 design.
- [ ] **2.0b** — Update `generate_pair()` in `generate_pairs.py` to call `build_room_simulation()` instead of `synthetic_feedback_ir()`, and add the sub_ir path to the combine step: `mic_signal = reverberant_vocal + mains_fb + monitor_fb + sub_fb + noise_scaled`
- [ ] **2.0c** — Update `eval/generate_test_set.py` to use `build_room_simulation()` instead of importing `synthetic_feedback_ir` — after the simulator upgrade is done

**Key papers:**
- MB-RIRs (2025): frequency-dependent absorption in synthetic IRs — https://arxiv.org/html/2507.09750
- SonicSim (ICLR 2025): geometry-accurate synthetic IRs beat naive ISM+real mix — https://arxiv.org/abs/2410.01481
- TS-RIR (IEEE ASRU 2021): GAN-based synthetic→real IR translation — https://arxiv.org/abs/2103.16804
- Ko et al. (ICASSP 2017): small set of real IRs matched to target environment ≈ large set of synthetic; combining both is best

---

### 2.1 Equipment Required

| Item | Purpose | Notes |
|---|---|---|
| Measurement mic | Capture IR at performer position | Dayton EMM-6 (~$25) or miniDSP UMIK-1 (~$75) — flat response to 20kHz |
| Audio interface | ADC for measurement mic | Any interface with phantom power + line in; 48kHz minimum |
| Laptop | Run sweep software | Can be any laptop; doesn't need to be the production machine |
| XLR cable | Mic to interface | Standard |
| Mic stand | Hold mic at performer height | Standard |
| Access to the PA | Play sweeps through mains + monitors | Need cooperation from venue |

**Measurement software (free):**
- **Room EQ Wizard (REW)** — roomeqwizard.com — the standard. Free, generates log sine sweeps, captures IRs, exports WAV. Handles the full deconvolution pipeline.
- **Alternative:** Custom Python script (see Section 2.4 below) — useful if you want to automate batch measurements or capture without a laptop screen visible.

### 2.2 Measurement Protocol

**What you are capturing:**
The acoustic transfer function from PA loudspeaker output → performer mic position.
This is the feedback loop path — the route audio takes from the speaker back into the open mic.

**Setup:**
1. Position the measurement mic on a stand at **performer position** — capsule at standing height (~5.5–6ft). The Dayton EMM-6 and UMIK-1 are omnidirectional — pointing direction doesn't matter. Just get the capsule at the right height and position, secured on a stand (don't hand-hold — movement corrupts the sweep).
2. Connect measurement mic to interface → laptop running REW
3. Connect laptop line out → PA system line input (same signal path the PA would normally receive from the console).
   **⚠️ DI box required:** Laptop headphone output is an unbalanced consumer signal. Running it directly into a balanced XLR mic-level PA input will cause a ground loop, hum, and level mismatch. Use a direct injection (DI) box (passive DI is fine — Radial J48 or similar) to convert unbalanced TS → balanced XLR and match impedance. Your interface's line output (TRS ¼") is also fine if it has one.
4. Set PA to a moderate gain — enough to get good SNR on the sweep capture without feedback during measurement

**Why mains and monitors must both be captured:**

Mains feedback and monitor feedback are fundamentally different acoustic problems:

| | Mains | Monitors |
|---|---|---|
| Distance to mic | 15–50ft typically | 2–6ft |
| Path character | Long, reverberant, room reflections dominant | Short, mostly direct sound |
| Energy at mic | Lower, diffuse | High, very direct |
| Frequency behavior | Room modes dominate | Speaker directivity + proximity dominate |
| Common cause | Mic aimed at FOH at high gain | Wedge aimed directly at capsule |

If the model only trains on mains IRs it will fail on monitor feedback — which is actually the more common problem in small venues and HOW settings. Both paths must be in the training data, and the combined measurement matters because in real use the mic sees both simultaneously.

In the simulator (Phase 3), mains and monitor IRs can be mixed and matched independently at different loop gains — e.g., high-gain monitor path + low-gain mains wash — expanding the effective training combinations significantly.

**Measurement positions per venue (minimum):**

At each position, run all 3 speaker configurations before moving the mic:

- [ ] Center stage, 6ft from front — **mains only**
- [ ] Center stage, 6ft from front — **monitors only**
- [ ] Center stage, 6ft from front — **mains + monitors combined**
- [ ] Stage left, 6ft back — **mains only**
- [ ] Stage left, 6ft back — **monitors only**
- [ ] Stage left, 6ft back — **mains + monitors combined**
- [ ] Stage right, 6ft back — **mains only**
- [ ] Stage right, 6ft back — **monitors only**
- [ ] Stage right, 6ft back — **mains + monitors combined**
- [ ] Center stage, 10ft back — **mains only**
- [ ] Center stage, 10ft back — **monitors only**
- [ ] Center stage, 10ft back — **mains + monitors combined**
- [ ] Front of stage (worst case) — **mains only**
- [ ] Front of stage (worst case) — **monitors only**
- [ ] Front of stage (worst case) — **mains + monitors combined**

**Target per venue:** ~15 IRs minimum (5 positions × 3 configurations)

**Capture per position:**
- 3 sweeps and average them (REW does this automatically) — averages out ambient noise
- Log sine sweep, 10–30 seconds, 20Hz–20kHz
- Capture at 48kHz / 24-bit minimum
- Takes ~1 minute per IR; ~45–90 minutes total per venue including mic moves

**Venue types to prioritize:**
1. Churches (small, medium, large sanctuary) — primary market
2. Gymnasiums / multi-purpose rooms (hard parallel walls, very different acoustic)
3. Small theaters / auditoriums
4. Bars / clubs with low ceilings
5. Outdoor stages (unique — minimal reverb, strong direct path)

**After each measurement:**
- Label files clearly: `{venue_type}_{venue_name}_{position}_{source_config}_{date}.wav`
- Source config values: `mains` / `monitors` / `combined`
- Examples:
  - `church_grace_fellowship_cs6ft_mains_20260318.wav`
  - `church_grace_fellowship_cs6ft_monitors_20260318.wav`
  - `church_grace_fellowship_cs6ft_combined_20260318.wav`
- Log metadata in a CSV: venue name, type, city, room dimensions (estimate),
  ceiling height, floor material, RT60 (REW measures automatically),
  PA system type, monitor type (wedge/IEM/sidefill), source config, mic position, filename

**In the simulator (Phase 3), mains and monitor IRs are used independently:**
- Mains IR drives the long-path reverberant feedback component
- Monitor IR drives the short-path direct feedback component
- Both can be assigned separate loop gains and combined into one mic signal
- This lets one set of real measurements generate many more training combinations

### 2.3 REW Step-by-Step

1. Open REW → Measure → set input/output to your interface
2. Set sweep: Log Sine, 20Hz–20kHz, 15 seconds, 3 averages
3. Set level: play sweep at moderate level, check input meter peaks at -12 to -6 dBFS
4. Click Start Sweep — REW plays sweep through line out → PA, captures on mic
5. When complete: Impulse Response tab → Export → WAV, 48kHz, 24-bit
6. Export the raw IR (**NOT minimum phase processed**) — you want the full causal response including the delay of the room.
   In REW: ensure "Minimum Phase" checkbox is **unchecked** before exporting. The minimum-phase version has the pre-delay removed and the phase reconstructed — it does not represent the actual acoustic path, which the Kalman filter needs to track.

### 2.4 Measurement GUI

**File:** `measurement/venue_sweep.py`

GUI built with Tkinter (included with Python — no extra install).
Run it with:

```bash
python measurement/venue_sweep.py
```

**What it does:**
- Dropdowns for venue type, mic position, source config (mains / monitors / combined), floor material
- Free-text fields for venue name, city, mic type, speaker model, monitor type, ceiling height, room dims, PA/console system, notes
- Audio device selector — lists all available devices, pre-selects system defaults
- Sweep duration (5–30s) and averages (1–5) spinboxes
- **Start Measurement** button — runs sweeps on a background thread so the UI stays responsive
- Progress bar showing sweep completion
- Scrolling log window showing sweep-by-sweep status
- Auto-generates filename from venue + position + config + timestamp
- Saves IR to the correct subdirectory (`data/venue_irs/mains/`, `monitors/`, or `combined/`) based on the Source Config dropdown, and appends a row to `metadata.csv` automatically
- **Reset for Next Position** button — clears position/config/notes fields while keeping venue-level info (name, type, city, room, PA) so you can chain all positions in one session without re-entering everything

### 2.5 IR Metadata Logging

`metadata.csv` is written automatically by the script. No manual entry required.
Fields logged per measurement:

```
filename, date, venue_name, venue_type, city,
mic_position, mic_distance_ft, source_config,
mic_type, speaker_type, monitor_type,
ceiling_height_ft, room_dims_estimate, floor_material,
pa_system, rt60_s, input_device, output_device,
sweep_duration_s, n_averages, ir_length_s, notes
```

This metadata becomes valuable for stratified sampling during training — making sure the training set covers a range of RT60 values, room sizes, and venue types evenly.

---

## Phase 3: Feedback Loop Simulator

This is the custom code at the core of training data generation.
No open-source version exists — must be written from scratch.

### 3.0 High-Fidelity Simulator Design (Upgrade Target)

The current `synthetic_feedback_ir()` uses shoebox rooms with a single scalar absorption value, no directivity on source or receiver, and `max_order=10`. This section specifies the full high-quality design. Implement progressively — each subsection is an independent upgrade.

---

#### 3.0.1 Frequency-Dependent Material Absorption Library

Real surfaces absorb differently across the frequency spectrum. The current single-coefficient model is physically wrong — a concrete wall absorbs 2% at 125 Hz but 5% at 4 kHz; carpet absorbs 2% at 125 Hz but 65% at 4 kHz. Using a single average coefficient produces IRs with incorrect spectral decay shape, which misleads the model about what real feedback sounds like tonally.

pyroomacoustics accepts per-band coefficients via:
```python
pra.Material(energy_absorption={"coeffs": [a125, a250, a500, a1k, a2k, a4k],
                                  "center_freqs": [125, 250, 500, 1000, 2000, 4000]})
```

**Material absorption library** (values from ISO 354 / Sabine tables, per octave band 125–4000 Hz):

```python
MATERIALS = {
    # surface            [125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz]  scattering
    'concrete':         ([0.01, 0.02, 0.02, 0.03, 0.04, 0.05],    0.05),
    'brick':            ([0.03, 0.03, 0.03, 0.04, 0.05, 0.07],    0.05),
    'plaster':          ([0.01, 0.02, 0.03, 0.04, 0.05, 0.05],    0.05),
    'wood_floor':       ([0.15, 0.11, 0.10, 0.07, 0.06, 0.07],    0.10),
    'wood_panel':       ([0.28, 0.22, 0.17, 0.09, 0.10, 0.11],    0.10),
    'carpet_thin':      ([0.02, 0.04, 0.08, 0.20, 0.35, 0.40],    0.20),
    'carpet_thick':     ([0.02, 0.06, 0.14, 0.37, 0.60, 0.65],    0.25),
    'curtains_light':   ([0.03, 0.04, 0.11, 0.17, 0.24, 0.35],    0.35),
    'curtains_heavy':   ([0.05, 0.12, 0.35, 0.45, 0.38, 0.36],    0.40),
    'acoustic_panels':  ([0.10, 0.25, 0.60, 0.90, 0.95, 0.95],    0.70),
    'audience_seated':  ([0.39, 0.57, 0.80, 0.94, 0.92, 0.87],    0.60),
    'glass':            ([0.35, 0.25, 0.18, 0.12, 0.07, 0.04],    0.05),
    'seats_empty':      ([0.44, 0.56, 0.67, 0.74, 0.83, 0.87],    0.40),
    'stage_platform':   ([0.40, 0.30, 0.20, 0.17, 0.15, 0.10],    0.15),
}
```

**Scattering coefficients** (second value per material) model diffuse reflection. High values (carpet, audience) scatter most energy; low values (concrete, glass) produce specular mirror-like reflections. Without scattering, ISM late reverberation sounds metallic and unrealistic.

---

#### 3.0.2 Room Archetypes

Each archetype defines plausible dimension ranges and per-surface material probability distributions. The simulator randomly samples an archetype, then randomizes within it.

```python
ROOM_ARCHETYPES = {
    'small_venue_bar': {
        # Low ceiling, parallel hard walls, terrible acoustics — very common
        'dims': ([5, 8, 2.8], [10, 15, 3.5]),       # [min], [max] in meters
        'floor':   ['wood_floor', 'concrete', 'carpet_thin'],
        'ceiling': ['plaster', 'acoustic_panels'],
        'walls':   ['brick', 'plaster', 'concrete', 'wood_panel'],
        'stage':   ['stage_platform', 'wood_floor'],
        'max_order': 15,
    },
    'church_sanctuary': {
        # High ceiling, hard parallel walls (brick/plaster), reverberant
        'dims': ([8, 12, 5], [18, 25, 10]),
        'floor':   ['wood_floor', 'concrete', 'carpet_thick'],
        'ceiling': ['plaster', 'wood_panel'],
        'walls':   ['brick', 'plaster', 'concrete'],  # stone omitted — not in MATERIALS dict
        'stage':   ['stage_platform', 'wood_floor'],
        'max_order': 20,
    },
    'gymnasium': {
        # Very long RT60, hard on all surfaces, worst-case scenario
        'dims': ([15, 20, 6], [35, 50, 10]),
        'floor':   ['wood_floor', 'concrete'],
        'ceiling': ['concrete', 'plaster'],
        'walls':   ['concrete', 'brick', 'glass'],
        'stage':   ['stage_platform'],
        'max_order': 18,   # was 25 — at 35×50×10m, max_order=25 yields ~15K image sources (minutes per IR); 18 is sufficient for RT60 modeling with ray_tracing=True
    },
    'theater': {
        # Sloped seating, acoustic treatment, moderate RT60
        'dims': ([10, 15, 6], [25, 35, 12]),
        'floor':   ['carpet_thick', 'seats_empty'],
        'ceiling': ['acoustic_panels', 'plaster'],
        'walls':   ['acoustic_panels', 'curtains_heavy', 'wood_panel'],
        'stage':   ['stage_platform'],  # audience_seated removed — physically nonsensical for a stage surface
        'max_order': 18,
    },
    'rehearsal_room': {
        # Small, treated, lowest feedback risk but still trained
        'dims': ([3, 4, 2.5], [6, 8, 3.2]),
        'floor':   ['carpet_thick', 'wood_floor'],
        'ceiling': ['acoustic_panels', 'plaster'],
        'walls':   ['acoustic_panels', 'curtains_heavy', 'plaster'],
        'stage':   ['stage_platform', 'carpet_thick'],
        'max_order': 12,
    },
    'ballroom': {
        # Large open room, hard floors, often glass walls — horrible acoustics
        'dims': ([10, 15, 4], [25, 40, 6]),
        'floor':   ['wood_floor', 'carpet_thick'],
        'ceiling': ['plaster', 'acoustic_panels'],
        'walls':   ['plaster', 'glass', 'curtains_heavy'],
        'stage':   ['stage_platform', 'wood_floor'],
        'max_order': 18,
    },
    'hall': {
        # Multi-purpose room attached to churches and community centers.
        # Low ceiling, bare parallel walls, zero acoustic treatment — one of the
        # worst feedback environments and extremely common in our target market.
        # Distinct from gymnasium (smaller volume, lower ceiling) and theater
        # (no treatment, flat floor). Early reflections arrive fast and hot.
        'dims': ([6, 10, 2.8], [15, 25, 4.0]),
        'floor':   ['concrete', 'carpet_thin', 'wood_floor'],
        'ceiling': ['plaster', 'concrete'],
        'walls':   ['concrete', 'brick', 'plaster'],
        'stage':   ['stage_platform', 'wood_floor', 'concrete'],
        'max_order': 18,
    },
}
```

**Usage:** randomly weight archetype sampling toward the most common target venues — church and small venue are the primary market:
```python
ARCHETYPE_WEIGHTS = {
    'small_venue_bar':  0.20,
    'church_sanctuary': 0.25,
    'hall':  0.20,  # very common in church/community market — bad acoustics, high feedback risk
    'gymnasium':        0.08,
    'theater':          0.12,
    'rehearsal_room':   0.08,
    'ballroom':         0.07,
}
```

---

#### 3.0.3 Microphone Pickup Pattern Library

Different polar patterns have fundamentally different feedback behavior. Cardioid has rear rejection ~20dB but leaks energy at the sides; supercardioid has tighter forward pattern but a rear lobe at ~180° that makes rear-facing monitors *more* dangerous, not less. The model must learn these different failure modes.

pyroomacoustics supports receiver directivity via `CardioidFamily`:
```python
from pyroomacoustics.directivities import (
    CardioidFamily, DirectivityPattern, Rotation3D
)
```

**Pattern library:**

| Pattern | DirectivityPattern enum | Common mics | Rear lobe? | Monitor risk |
|---|---|---|---|---|
| Cardioid | `CARDIOID` | SM58, e935, Beta58A | No rear lobe, broad side leakage | Moderate |
| Supercardioid | `HYPERCARDIOID` | SM86, KSM9, e945 | Small rear lobe at 126° | Rear wedge = dangerous |
| Hypercardioid | `HYPERCARDIOID` + tighter | Beta91A class | Rear lobe at 110° | Rear wedge = very dangerous |
| Subcardioid | `SUBCARDIOID` | Some condensers, boundary mics | Wide pattern, more side leakage | High all around |
| Omnidirectional | `OMNI` | Lavalier, headset mics | No rejection anywhere | Maximum risk |

> **Note:** pyroomacoustics `DirectivityPattern` enum has: `OMNI`, `FIGURE_EIGHT`, `HYPERCARDIOID`, `CARDIOID`, `SUBCARDIOID`. For true supercardioid vs. hypercardioid distinction, vary the `pattern_enum` and optionally the `gain` parameter of `CardioidFamily`.

**Mic orientation randomization** — performers don't hold mics on-axis with speakers:
```python
# Azimuth: left/right rotation — singer turning head or pointing at monitor
# Elevation: up/down — mic tilted toward floor vs. aimed at house
azimuth   = np.random.normal(0, 20)   # degrees — centered forward, ±20° std dev
elevation = np.random.normal(0, 10)   # degrees — slight upward tilt is common
orientation = Rotation3D([azimuth, elevation, 0], 'yzx', degrees=True)

mic_directivity = CardioidFamily(
    orientation=orientation,
    pattern_enum=chosen_pattern,
    gain=1.0,
    fs=SR
)
room.add_microphone(mic_pos, directivity=mic_directivity)
```

**Sampling weights** (based on live sound usage distribution):
```python
MIC_PATTERN_WEIGHTS = {
    DirectivityPattern.CARDIOID:      0.55,  # most common handheld dynamic
    DirectivityPattern.HYPERCARDIOID: 0.25,  # supercardioid — tight performance mic
    DirectivityPattern.SUBCARDIOID:   0.10,  # headset / wide-pattern condensers
    DirectivityPattern.OMNI:          0.10,  # lav mics, boundary mics, pulpit mics
}
```

---

#### 3.0.4 Speaker Directivity Library

PA loudspeakers and monitor wedges have very different polar patterns. A line array has near-omnidirectional horizontal dispersion but extremely narrow vertical coverage (~10°). A wedge monitor fires at ≈ 40° upward angle toward the performer's capsule.

```python
SPEAKER_CONFIGS = {
    'mains_point_source': {
        # Single box speaker — most common in small venues/churches
        'pattern':         DirectivityPattern.CARDIOID,
        'azimuth':         0,              # facing stage
        'elevation_range': (-30, -10),     # sampled per-call in build_room_simulation()
        'n_sources': 1,
        'positions': 'center_elevated',
    },
    'mains_left_right': {
        # Stereo L/R hang — typical in larger venues
        'pattern':       DirectivityPattern.CARDIOID,
        'n_sources':     2,
        'positions':     'lr_elevated',
        'toe_in_range':  (10, 25),         # sampled per-call
        'elevation_range': (-25, -10),
    },
    'mains_line_array': {
        # Very narrow vertical, wide horizontal — worst case for off-axis stage positions
        'pattern':         DirectivityPattern.HYPERCARDIOID,  # tightest available as proxy
        'elevation_range': (-20, -5),
        'n_sources':       2,
        'positions':       'lr_elevated',
    },
    'monitor_wedge_front': {
        # Standard floor wedge — fired directly at performer
        'pattern':         DirectivityPattern.CARDIOID,
        'azimuth':         180,            # facing back toward performer
        'elevation_range': (35, 50),       # tilted up at capsule height — sampled per-call
        'n_sources':       1,
        'positions':       'downstage_floor',
    },
    'monitor_wedge_side': {
        # Side-fill or drummer monitor — less common in small venues
        'pattern':         DirectivityPattern.CARDIOID,
        'azimuth':         90,             # firing across stage
        'elevation_range': (20, 40),
        'n_sources':       1,
        'positions':       'stage_left_floor',
    },
    'monitor_in_ear': {
        # IEM — produces no acoustic feedback at all
        # Include at ~15% of samples to train the "no feedback" case
        'pattern': None,   # skip feedback path entirely
        'n_sources': 0,
    },
    'subwoofer_ground_stack': {
        # Subs stacked on each other on the floor (ground-stack deployment)
        # as opposed to flown/rigged. Omnidirectional radiation pattern because
        # wavelengths at 80-100Hz (3-4 meters) are >> cabinet size — the sub
        # is acoustically a point source at these frequencies.
        # Feedback manifests as low room-mode ringing ("woof"), especially in gyms + churches.
        'pattern':   DirectivityPattern.OMNI,
        'azimuth':   0,
        'elevation': 0,
        'n_sources': 1,
        'positions': 'ground_center_front',
        'lowpass_hz': 100,     # crossover typically 80-100Hz
        'gain_range': (0.1, 0.4),  # lower gain than mains — sub feedback is slower to build
    },
    'subwoofer_cardioid_array': {
        # End-fire cardioid sub array — subs deployed in a specific spacing/delay
        # arrangement to create rear cancellation (common in installed systems).
        # Reduces rear bleed toward stage but still broad pattern forward.
        'pattern':   DirectivityPattern.SUBCARDIOID,
        'azimuth':   0,
        'elevation': 0,
        'n_sources': 1,
        'positions': 'ground_center_front',
        'lowpass_hz': 100,
        'gain_range': (0.05, 0.25),  # cardioid config has lower rear energy toward stage
    },
}
```

**IEM note:** 15% of monitor samples should use `monitor_in_ear` (zero feedback path). This teaches the model not to suppress signal when there's no feedback.

---

#### 3.0.5 Integrated Room Builder Function

Replaces the current `synthetic_feedback_ir()` with a single `build_room_simulation()` that:
1. Samples an archetype and builds the room with per-surface frequency-dependent materials + scattering
2. Places the mic with sampled directivity pattern and random orientation
3. Places mains speaker(s) with appropriate directivity and position
4. Places monitor wedge with appropriate directivity and angle
5. Runs one `room.simulate()` call and returns separate IRs for each source→mic path

```python
def build_room_simulation():
    """
    Returns:
        mains_ir    : np.ndarray  (loudspeaker→mic feedback path for mains)
        monitor_ir  : np.ndarray  (wedge→mic feedback path)
        room_ir     : np.ndarray  (direct vocal reverb — close-mic diffuse path)
        sub_ir      : np.ndarray  (sub feedback path, zeros if no sub in this sample)
        meta        : dict        (archetype, mic_pattern, speaker_config, dims)
    """
    # 1. Sample archetype
    archetype_name = random.choices(
        list(ARCHETYPE_WEIGHTS.keys()),
        weights=list(ARCHETYPE_WEIGHTS.values())
    )[0]
    arch = ROOM_ARCHETYPES[archetype_name]

    # 2. Random room dimensions within archetype bounds
    dims = np.random.uniform(arch['dims'][0], arch['dims'][1])

    # 3. Per-surface material sampling
    def make_material(surface_key):
        mat_name = random.choice(arch[surface_key])
        coeffs, scatter = MATERIALS[mat_name]
        return pra.Material(
            energy_absorption={"coeffs": coeffs,
                               "center_freqs": [125, 250, 500, 1000, 2000, 4000]},
            scattering=scatter
        )

    # pra.ShoeBox accepts a plain dict keyed by wall name — pra.MixedMaterial does not exist
    materials = {
        'floor':   make_material('floor'),
        'ceiling': make_material('ceiling'),
        'east':    make_material('walls'),
        'west':    make_material('walls'),
        'north':   make_material('walls'),
        'south':   make_material('walls'),
    }

    room = pra.ShoeBox(
        dims, fs=SR,
        materials=materials,
        max_order=arch['max_order'],
        ray_tracing=True,          # better late reverb beyond ISM max_order
        air_absorption=True,       # high-freq rolloff over distance
    )

    # 4. Mic: performer position — center-to-slightly-back of stage, standing height
    mic_pos = [
        dims[0] * np.random.uniform(0.3, 0.7),           # left/right: anywhere across stage
        dims[1] * np.random.uniform(0.4, 0.65),           # front/back: mid-to-back stage
        np.random.uniform(1.5, 1.9)                        # height: standing mic height
    ]
    mic_pattern = random.choices(
        list(MIC_PATTERN_WEIGHTS.keys()),
        weights=list(MIC_PATTERN_WEIGHTS.values())
    )[0]
    azimuth   = np.random.normal(0, 20)
    elevation = np.random.normal(0, 10)
    mic_dir = CardioidFamily(
        orientation=Rotation3D([azimuth, elevation, 0], 'yzx', degrees=True),
        pattern_enum=mic_pattern, fs=SR
    )
    room.add_microphone(np.array(mic_pos), directivity=mic_dir)

    # 5. Mains speaker — elevated near front of room, aimed at audience/stage
    mains_config_name = random.choices(
        ['mains_point_source', 'mains_left_right', 'mains_line_array'],
        weights=[0.5, 0.35, 0.15]
    )[0]
    mains_config = SPEAKER_CONFIGS[mains_config_name]
    mains_elev = np.random.uniform(*mains_config['elevation_range'])  # sampled fresh each call
    mains_dir = CardioidFamily(
        orientation=Rotation3D([180, mains_elev, 0], 'yzx', degrees=True),
        pattern_enum=mains_config['pattern'], fs=SR
    )
    mains_src_pos = [
        dims[0] * 0.5,
        dims[1] * 0.05,                # near front wall
        dims[2] * np.random.uniform(0.5, 0.85)  # elevated
    ]
    room.add_source(np.array(mains_src_pos), directivity=mains_dir)
    mains_src_idx = len(room.sources) - 1  # add_source() doesn't return index

    # 6. Monitor wedge — downstage floor, angled at performer
    monitor_config_name = random.choices(
        ['monitor_wedge_front', 'monitor_wedge_side', 'monitor_in_ear'],
        weights=[0.65, 0.20, 0.15]
    )[0]
    monitor_config = SPEAKER_CONFIGS[monitor_config_name]
    if monitor_config['n_sources'] == 0:
        monitor_src_idx = None   # IEM — no feedback path
    else:
        monitor_elev = np.random.uniform(*monitor_config['elevation_range'])  # sampled fresh
        mon_dir = CardioidFamily(
            orientation=Rotation3D([monitor_config['azimuth'], monitor_elev, 0],
                                   'yzx', degrees=True),
            pattern_enum=monitor_config['pattern'], fs=SR
        )
        mon_src_pos = [
            mic_pos[0] + np.random.uniform(-0.3, 0.3),  # directly in front of performer
            mic_pos[1] - np.random.uniform(0.5, 1.5),    # downstage of performer
            np.random.uniform(0.1, 0.4)                   # floor level
        ]
        room.add_source(np.array(mon_src_pos), directivity=mon_dir)
        monitor_src_idx = len(room.sources) - 1

    # 7. Room reverb source — close to mic (direct vocal reflection path)
    room_src_pos = [
        mic_pos[0] + np.random.uniform(0.1, 0.4),
        mic_pos[1] + np.random.uniform(-0.2, 0.2),
        mic_pos[2]
    ]
    room.add_source(np.array(room_src_pos))
    room_src_idx = len(room.sources) - 1

    # 8. Subwoofer — decide upfront so we can add to room before the single simulate() call
    # Include in 60% of samples for archetypes that typically have subs
    sub_config_name = None
    sub_src_idx = None
    if archetype_name in ('gymnasium', 'church_sanctuary', 'hall', 'theater', 'ballroom') \
            and random.random() < 0.60:
        sub_config_name = random.choice(['subwoofer_ground_stack', 'subwoofer_cardioid_array'])
        sub_config = SPEAKER_CONFIGS[sub_config_name]
        sub_dir = CardioidFamily(
            orientation=Rotation3D([0, 0, 0], 'yzx', degrees=True),
            pattern_enum=sub_config['pattern'], fs=SR
        )
        sub_src_pos = [
            dims[0] * np.random.uniform(0.35, 0.65),   # center-ish front
            dims[1] * 0.03,                              # near front wall
            np.random.uniform(0.1, 0.5)                  # ground level
        ]
        room.add_source(np.array(sub_src_pos), directivity=sub_dir)
        sub_src_idx = len(room.sources) - 1

    # 9. Single simulate() call covers all sources added above
    room.simulate()

    def _safe_ir(raw):
        """Return IR, or raise if it contains NaN/Inf or is all-zero (silent simulation failure)."""
        ir = raw.astype(np.float32)
        if not np.isfinite(ir).all():
            raise ValueError(f"IR contains NaN or Inf — pyroomacoustics simulation diverged "
                             f"(room dims {dims.tolist()}, max_order={arch['max_order']})")
        if np.max(np.abs(ir)) < 1e-12:
            raise ValueError(f"IR is all-zero — silent simulation failure "
                             f"(room dims {dims.tolist()}, source/mic positions may be coincident)")
        return ir

    mains_ir   = _safe_ir(room.rir[0][mains_src_idx])
    monitor_ir = (_safe_ir(room.rir[0][monitor_src_idx])
                  if monitor_src_idx is not None
                  else np.zeros(512, dtype=np.float32))
    room_ir    = _safe_ir(room.rir[0][room_src_idx])

    sub_ir = np.zeros(512, dtype=np.float32)   # default: no sub feedback
    if sub_src_idx is not None:
        raw_sub_ir = room.rir[0][sub_src_idx].astype(np.float32)
        # Lowpass to sub frequency range — crossover at 80-100Hz
        from scipy.signal import butter, sosfilt
        sos = butter(4, sub_config['lowpass_hz'] / (SR / 2), btype='low', output='sos')
        sub_ir = sosfilt(sos, raw_sub_ir).astype(np.float32)

    meta = {
        'archetype': archetype_name,
        'mic_pattern': mic_pattern.name,
        'mains_config': mains_config_name,
        'monitor_config': monitor_config_name,
        'sub_config': sub_config_name,
        'dims': dims.tolist(),
    }
    return mains_ir, monitor_ir, room_ir, sub_ir, meta
```

> **`ray_tracing=True`** — pyroomacoustics can combine ISM (early reflections) with stochastic ray tracing (late reverb diffuse field) when `max_order` is insufficient to model the full tail. This significantly improves realism of long RT60 rooms (gyms, churches) without needing extremely high `max_order` values that are computationally prohibitive.

---

#### 3.0.6 What This Covers vs. Doesn't Cover

**Covered:**
- 6 room archetypes × randomized dimensions = enormous geometric variety
- Per-surface frequency-dependent absorption + scattering on every wall
- 5 mic pickup patterns (omni → hypercardioid)
- 3 mains configurations (point source, L/R stereo, line array proxy)
- 3 monitor configurations (front wedge, side fill, IEM/no-feedback)
- Random mic orientation within realistic performer behavior bounds
- Speaker directivity and positioning within plausible constraints
- All acoustic paths computed in a single room (mains reflections interact correctly)
- Air absorption (high-freq rolloff over distance — critical for mains at 30+ feet)
- IEM scenario (no monitor feedback — teaches model not to suppress clean signal)

**Not covered (known limitations):**
- Moving sources/receivers (performer walking, turning) — static snapshot per sample; training diversity compensates
- Non-linear speaker behavior (distortion at high SPL) — out of scope for acoustic IR modeling

**Covered via approximation (see 3.0.7):**
- Non-convex rooms — cascaded shoebox simulation (see 3.0.7)
- Subwoofer feedback — omnidirectional source + lowpass-shaped IR (see 3.0.4/3.0.5)

---

#### 3.0.7 Non-Convex Room Approximation (Cascaded Shoebox)

pyroomacoustics ISM requires convex geometry — the image source method breaks down when walls are non-parallel and non-planar in a concave arrangement (the "images" would fall outside the room). True non-convex simulation requires FEM/BEM or FDTD solvers, which are orders of magnitude too slow for training data generation.

**Why non-convex rooms matter for live sound:**
- L-shaped rooms (church with side wings, bar with back room) are extremely common in the target market
- Rooms with balconies create a coupled upper-lower space — classic feedback scenario
- Fan-shaped auditoriums have non-parallel wall pairs that create distinct acoustic behavior
- In these geometries, feedback frequencies differ from a simple rectangular model — the model needs to have seen this

**The approximation: cascaded (series) shoebox convolution**

The key acoustic effect of an L-shaped or multi-space room is that sound traveling from source to mic passes *through* or *around* a junction. Acoustically, the result is a double-reverb tail: the energy decays in the primary space, reflects off the junction geometry, and a secondary decay from the secondary space follows. This can be approximated by convolving two independent shoebox IRs in series.

```python
def non_convex_room_ir(near_field=False):
    """
    Approximate non-convex room IR via cascaded shoebox simulation.
    Models L-shaped rooms, rooms with alcoves, stage + auditorium coupling.

    The primary IR represents the main performance space.
    The secondary IR represents the connected sub-space (side wing, back room, balcony).
    Convolving in series approximates sound traveling through the acoustic junction.

    near_field=True: monitor path (short primary space, small secondary alcove)
    near_field=False: mains path (large primary hall, secondary side wing or balcony)
    """
    # ── Primary space (main performance area) ───────────────────────────────
    if near_field:
        primary_dims = np.random.uniform([4, 6, 2.8], [8, 12, 4.0])
    else:
        primary_dims = np.random.uniform([8, 12, 4], [20, 30, 10])

    primary_mat_name = random.choice(['plaster', 'brick', 'wood_panel', 'concrete'])
    p_coeffs, p_scatter = MATERIALS[primary_mat_name]
    primary_room = pra.ShoeBox(
        primary_dims, fs=SR,
        materials=pra.Material(
            energy_absorption={"coeffs": p_coeffs,
                               "center_freqs": [125, 250, 500, 1000, 2000, 4000]},
            scattering=p_scatter
        ),
        max_order=15,
        ray_tracing=True,
        air_absorption=True,
    )
    if near_field:
        src_pos = [primary_dims[0]*0.2, primary_dims[1]*0.1, primary_dims[2]*0.3]
        mic_pos = [primary_dims[0]*0.5, primary_dims[1]*0.5, 1.7]
    else:
        src_pos = [primary_dims[0]*0.5, primary_dims[1]*0.05, primary_dims[2]*0.7]
        mic_pos = [primary_dims[0]*0.5, primary_dims[1]*0.55, 1.7]
    primary_room.add_source(src_pos)
    primary_room.add_microphone(mic_pos)
    primary_room.simulate()
    primary_ir = primary_room.rir[0][0].astype(np.float32)

    # ── Secondary space (connected sub-space) ───────────────────────────────
    # Smaller than primary — a side chapel, back corridor, balcony overhang
    secondary_scale = np.random.uniform(0.3, 0.6)   # secondary is 30–60% of primary volume
    secondary_dims = primary_dims * secondary_scale
    secondary_dims[2] = np.random.uniform(2.5, primary_dims[2])  # independent ceiling height

    secondary_mat_name = random.choice(list(MATERIALS.keys()))
    s_coeffs, s_scatter = MATERIALS[secondary_mat_name]
    secondary_room = pra.ShoeBox(
        secondary_dims, fs=SR,
        materials=pra.Material(
            energy_absorption={"coeffs": s_coeffs,
                               "center_freqs": [125, 250, 500, 1000, 2000, 4000]},
            scattering=s_scatter
        ),
        max_order=12,
        ray_tracing=True,
        air_absorption=True,
    )
    # Place source and mic at the junction point (entrance to secondary space)
    junc_src = [secondary_dims[0]*0.1, secondary_dims[1]*0.5, secondary_dims[2]*0.5]
    junc_mic = [secondary_dims[0]*0.8, secondary_dims[1]*0.5, secondary_dims[2]*0.5]
    secondary_room.add_source(junc_src)
    secondary_room.add_microphone(junc_mic)
    secondary_room.simulate()
    secondary_ir = secondary_room.rir[0][0].astype(np.float32)

    # ── Cascade: convolve primary → secondary ───────────────────────────────
    # Attenuation at junction: sound loses energy passing through the opening
    junction_atten = np.random.uniform(0.2, 0.6)   # 6–14 dB loss at the junction
    cascaded_ir = fftconvolve(primary_ir, secondary_ir * junction_atten)

    # Do NOT normalize the cascaded IR — the junction_atten factor encodes physically
    # realistic path loss (6–14 dB) through the junction opening. Normalizing would undo
    # that attenuation and make every L-shaped room look like it has full direct-path gain.
    # The Kalman filter calibrates its covariances based on absolute IR amplitude.
    return cascaded_ir.astype(np.float32)
```

**Integration into `build_room_simulation()`:**

Replace ~20% of shoebox-generated mains IRs with `non_convex_room_ir(near_field=False)`, and ~10% of monitor IRs with `non_convex_room_ir(near_field=True)`:

```python
# In build_room_simulation(), after computing mains_ir from the main room:
if random.random() < 0.20:
    mains_ir = non_convex_room_ir(near_field=False)   # override with L-shaped approx

if monitor_src_idx is not None and random.random() < 0.10:
    monitor_ir = non_convex_room_ir(near_field=True)
```

**What this captures:**
- Double reverb tail characteristic of L-shaped / multi-space rooms
- Junction attenuation (the opening between spaces acts as a low-pass filter at the boundary)
- Independent material properties for primary vs. secondary space
- Different ceiling heights between spaces (very common in church side chapels)
- A range of secondary space sizes — from small alcove (scale 0.3) to large side wing (scale 0.6)

**What it doesn't capture:**
- The exact diffraction geometry at the junction (we use convolution as a linear proxy)
- Rooms where source and mic are in *different* convex subspaces (here they share the primary space)
- Geometry-specific modal behavior of the non-convex shape

For training data diversity purposes, these approximations are sufficient — the model learns that feedback can have irregular multi-decay tails, without needing the exact geometry to be modeled.

#### 3.0.8 Transducer Frequency Response Library

The feedback loop gain at each frequency is the product of room IR, mic FR, and speaker FR.
Without transducer coloration in training data, the model only sees feedback driven by room
modes — it never sees the characteristic 8kHz ring of an SM58 presence peak through a QSC K12,
or the 2–4kHz coloration of a cheap PA cabinet. The Kalman filter adapts online regardless,
but the GRU's Q_k/R_k estimates will be poorly calibrated for the spectral profiles that
real transducers produce.

**Implementation — `simulator/transducer_frs.py`:**

```python
"""simulator/transducer_frs.py"""
import numpy as np
from scipy.signal import firwin2, minimum_phase
from pathlib import Path

SR     = 48000
N_TAPS = 512   # ~10ms at 48kHz — captures transducer coloration without excessive latency

def load_fr_csv(path):
    """Load freq_hz,magnitude_db CSV. Returns (freqs_hz, magnitudes_db) arrays."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]

def fr_to_fir(freqs_hz, magnitudes_db, sr=SR, n_taps=N_TAPS):
    """
    Convert a frequency response curve to a minimum-phase FIR filter.

    Minimum-phase ensures causality — the filter cannot output energy before its input
    arrives. Appropriate for transducer modeling (a speaker can't radiate before receiving
    the signal). Normalized to unity gain at 1kHz so it models coloration, not level.
    """
    n_pts      = n_taps * 4
    f_grid     = np.linspace(0, sr / 2, n_pts)
    mag_interp = np.interp(f_grid, freqs_hz, 10 ** (magnitudes_db / 20))
    mag_interp = np.clip(mag_interp, 1e-4, None)           # floor at -80dB

    # 1/6-octave smoothing before firwin2 — removes digitization noise from WebPlotDigitizer
    # and prevents Gibbs ringing in the FIR from sharp spectral transitions in coarse FR data.
    # Kernel width = 1/6 octave in log-frequency space ≈ ~3 bins at low freqs, ~20 at high.
    from scipy.ndimage import uniform_filter1d
    # Approximate 1/6-octave in linear frequency space at the geometric mean of f_grid
    smooth_hz = 0.5 * f_grid[-1] * (2 ** (1/6) - 1)   # avg bandwidth of 1/6-oct bin
    smooth_bins = max(1, int(smooth_hz / (f_grid[1] - f_grid[0])))
    mag_interp = uniform_filter1d(mag_interp, size=smooth_bins)

    ref_idx    = np.argmin(np.abs(f_grid - 1000))
    mag_interp /= mag_interp[ref_idx]                       # normalize at 1kHz
    f_norm     = f_grid / (sr / 2)
    f_norm[-1] = 1.0
    h = firwin2(n_taps, f_norm, mag_interp)
    # NOTE: minimum_phase(h, 'homomorphic') returns a filter of length (len(h)+1)//2,
    # NOT len(h). With n_taps=512, the returned filter is 256 taps (~5.3ms at 48kHz).
    # This is undocumented behavior in scipy — do not assume output length == n_taps.
    h = minimum_phase(h, method='homomorphic')
    return h.astype(np.float32)

def build_transducer_library(frs_dir='data/transducer_frs', sr=SR):
    """
    Load all FR CSVs from data/transducer_frs/mics/ and data/transducer_frs/speakers/.
    Returns {'mics': {name: fir_np, ...}, 'speakers': {name: fir_np, ...}}.
    Always includes a 'flat' entry in each category (identity filter).
    Call once at startup — FIR design is not free.
    """
    lib = {'mics': {'flat': np.array([1.0], dtype=np.float32)},
           'speakers': {'flat': np.array([1.0], dtype=np.float32)}}
    frs_path = Path(frs_dir)
    for category in ('mics', 'speakers'):
        cat_dir = frs_path / category
        if not cat_dir.exists():
            continue
        for csv in sorted(cat_dir.glob('*.csv')):
            try:
                freqs, mags = load_fr_csv(csv)
                lib[category][csv.stem] = fr_to_fir(freqs, mags, sr)
            except Exception as e:
                print(f"Warning: skipping {csv.name}: {e}")
    return lib
```

**Integration in `generate_ir_pool.py`:**

Load the library once before the loop, then apply random mic + speaker FR to each IR.
The IR already encodes room coloration; convolving with mic+speaker FIR adds transducer
coloration on top. Apply to mains and monitor IRs independently (same mic, different
speaker type is common — monitors and mains are often different models).

```python
from transducer_frs import build_transducer_library
from scipy.signal import fftconvolve

# Load once before the generation loop
tx_lib = build_transducer_library()
mic_firs     = list(tx_lib['mics'].values())
speaker_firs = list(tx_lib['speakers'].values())

# Inside the loop, after build_room_simulation() returns mains_ir / monitor_ir:
mic_fir         = random.choice(mic_firs)
mains_spk_fir   = random.choice(speaker_firs)
monitor_spk_fir = random.choice(speaker_firs)

mains_ir   = fftconvolve(fftconvolve(mains_ir,   mains_spk_fir), mic_fir).astype(np.float32)
monitor_ir = fftconvolve(fftconvolve(monitor_ir, monitor_spk_fir), mic_fir).astype(np.float32)
# Same mic for both paths (the performer has one mic regardless of which speaker type feeds back)
# Independent speaker FIRs (mains ≠ monitors in most venues)
```

**Data acquisition — `data/transducer_frs/mics/`:**

| File | Model | Key character |
|---|---|---|
| `shure_sm58.csv` | Shure SM58 | Classic presence peak 5–10kHz (+4–6dB) |
| `shure_beta58a.csv` | Shure Beta 58A | More aggressive presence peak 8–12kHz (+8dB) |
| `sennheiser_e835.csv` | Sennheiser e835 | Slightly flatter than SM58, very common |
| `sennheiser_e945.csv` | Sennheiser e945 | Tighter supercardioid, different HF character |
| `akg_d5.csv` | AKG D5 | Multiple mid-presence bumps |
| `shure_sm86.csv` | Shure SM86 | Condenser — flatter, slight HF lift |
| `shure_ksm9.csv` | Shure KSM9 | Premium condenser — near-flat |
| `shure_ulxd_ta87a.csv` | Shure ULX-D / TA87A | Wireless HF rolloff + TA87A character |
| `shure_ulxd_beta87a.csv` | Shure ULX-D / Beta87A | Beta87A presence curve through wireless path |
| `sennheiser_ew_835p.csv` | Sennheiser EW / 835-p | Most common church wireless — 835 + HF rolloff |
| `sennheiser_ew_945.csv` | Sennheiser EW / 945 | Tighter pattern, different HF rollout |
| `dpa_4099.csv` | DPA 4099 | Instrument clip-on, bright character |
| `countryman_e6.csv` | Countryman E6 | Church headset — small capsule, brighter |
| `shure_mx153.csv` | Shure MX153 | HOW headset — small diaphragm, very flat |
| `flat.csv` | (identity) | No transducer coloration — ~15% of samples |

**Data acquisition — `data/transducer_frs/speakers/`:**

| File | Model | Key character |
|---|---|---|
| `qsc_k12_2.csv` | QSC K12.2 | Dominant small venue/church main — relatively flat |
| `qsc_k10_2.csv` | QSC K10.2 | Smaller version — different HF extension |
| `jbl_srx835p.csv` | JBL SRX835P | Mid-size venue standard |
| `jbl_prx835.csv` | JBL PRX835 | Very common church install |
| `yamaha_dxr12.csv` | Yamaha DXR12 | Strong HOW market presence |
| `ev_zlx12p.csv` | EV ZLX-12P | Budget workhorse — significant cabinet coloration |
| `mackie_thump15a.csv` | Mackie Thump15A | Budget coloration — characteristic cheap-box 3kHz bump |
| `rcf_art745a.csv` | RCF ART 745-A | Common in European market |
| `yamaha_sm12v.csv` | Yamaha SM12V | Common passive monitor wedge |
| `jbl_eon515xt.csv` | JBL EON515XT | Frequently used as cheap monitor |
| `flat.csv` | (identity) | No speaker coloration — ~15% of samples |

**Where to get the data:**
- **Manufacturer spec sheets**: FR chart in the PDF. Digitize with WebPlotDigitizer (free browser tool) — ~5 min per curve. Export as CSV. Most curves are 1/3-octave smoothed at 0° on-axis — sufficient for this purpose.
- **AudioScienceReview** (forums) — community measurements for QSC, JBL, Yamaha, EV, Mackie boxes. Usually CEA-2034 spinorama data; use the on-axis curve.
- **Sonarworks Reference** — database covers most live vocal mics including wireless systems.
- **Wireless mics**: use measurements of the complete system (capsule + transmitter + receiver), not capsule alone. RF path adds HF rolloff the capsule spec doesn't show.
- **Speakers**: on-axis at 1m is the correct reference. Feedback arrives near on-axis at the mic position.

**Smoke test** (add to Phase 0.4 checks):
```python
from simulator.transducer_frs import build_transducer_library
lib = build_transducer_library()
assert 'flat' in lib['mics']
assert 'flat' in lib['speakers']
# With real data loaded:
# assert len(lib['mics']) >= 5, "Need at least 5 mic FRs before training"
# assert len(lib['speakers']) >= 4, "Need at least 4 speaker FRs before training"
print(f"Transducer library: {len(lib['mics'])} mics, {len(lib['speakers'])} speakers")
```

**Timing note:** data acquisition (digitizing FR charts) is a parallel task to Phase 1 dataset downloads — they're independent and can be done at the same time. The FIR conversion runs in `generate_ir_pool.py` at IR pool generation time, not during training, so it adds zero training latency.

---

### 3.1 Simulator Design

> **Note:** This section describes the original simulator design. The synthetic IR generation path (`synthetic_feedback_ir()`) is superseded by `build_room_simulation()` from section 3.0. The `generate_pair()` function and combine step in 3.2 still need to be updated per action items 2.0a–2.0b. The real IR loading path (60% real, 40% synthetic) remains correct.

```
inputs:
  - clean_vocal (dry anechoic audio from EARS/VCTK)
  - mains_ir  (from venue_irs/mains/ — long-path reverberant feedback)
  - monitor_ir (from venue_irs/monitors/ — short-path direct feedback)
    → each with independent loop gain
    → both fall back to build_room_simulation() if real ones unavailable  [updated]
  - sub_ir (from build_room_simulation() sub path — 60% of large-venue samples)  [new]
  - room_ir (from public_irs/ — room reverb of direct vocal path)
  - noise (from DNS noise set, random SNR)

output:
  - mic_signal = reverberant_vocal + mains_feedback + monitor_feedback + sub_feedback + noise
  - clean_vocal (the training target)
```

**IR directory structure (required before running simulator):**
```
data/venue_irs/
  mains/        ← IRs captured with mains only active
  monitors/     ← IRs captured with monitors only active
  combined/     ← IRs captured with full system active
data/public_irs/  ← room acoustic IRs from public datasets (for reverb path)
```

**Pre-processing step — resample all audio to 48kHz before generating pairs:**
This avoids running librosa inside worker processes, which causes multiprocessing
failures on macOS (spawn start method re-imports C extensions unreliably).
Run once after downloading datasets, and again after any new IR collection:
```bash
python simulator/preprocess.py
# resamples data/clean_vocals/, data/venue_irs/, data/public_irs/, data/noise/ to 48kHz in-place
# covers all four directories the simulator loads from — not just vocals and venue IRs
```

```python
"""
simulator/preprocess.py — Resample all training audio to 48kHz in-place.

Run once after downloading datasets and again after any new venue IR collection.
Converts stereo/multi-channel files to mono. Skips files already at 48kHz.
Safe to re-run: existing 48kHz files are untouched.

Usage:
    python simulator/preprocess.py              # process all four directories
    python simulator/preprocess.py --dry-run    # show what would be changed, don't write
"""

import argparse, sys
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import librosa
except ImportError:
    sys.exit("librosa not installed — run: pip install librosa")

TARGET_SR   = 48000
AUDIO_EXTS  = {'.wav', '.flac', '.aiff', '.aif', '.ogg'}

PROJECT_ROOT = Path(__file__).parent.parent

DIRS_TO_PROCESS = [
    PROJECT_ROOT / 'data' / 'clean_vocals',
    PROJECT_ROOT / 'data' / 'venue_irs' / 'mains',
    PROJECT_ROOT / 'data' / 'venue_irs' / 'monitors',
    PROJECT_ROOT / 'data' / 'venue_irs' / 'combined',
    PROJECT_ROOT / 'data' / 'public_irs',
    PROJECT_ROOT / 'data' / 'noise',
]


def process_file(path: Path, dry_run: bool) -> str:
    """
    Resample a single audio file to TARGET_SR in-place.
    Returns a status string: 'skipped', 'resampled', or 'error: <msg>'.
    """
    try:
        info = sf.info(str(path))
    except Exception as e:
        return f"error: sf.info failed — {e}"

    if info.samplerate == TARGET_SR:
        return 'skipped'

    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception as e:
        return f"error: read failed — {e}"

    # Mix down to mono — take mean across channels
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    if dry_run:
        return f"would resample {sr}Hz → {TARGET_SR}Hz ({audio.shape[0]} samples)"

    # Resample using librosa (high-quality sinc resampling)
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Determine output subtype — preserve bit depth where possible
    subtype = 'PCM_16'
    if info.subtype in ('PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'):
        subtype = 'PCM_24'

    try:
        sf.write(str(path), resampled, TARGET_SR, subtype=subtype)
    except Exception as e:
        return f"error: write failed — {e}"

    return f"resampled {sr}Hz → {TARGET_SR}Hz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Show what would be changed without writing files')
    args = ap.parse_args()

    counts = {'skipped': 0, 'resampled': 0, 'error': 0}

    for d in DIRS_TO_PROCESS:
        if not d.exists():
            print(f"[skip] {d} — does not exist yet")
            continue

        files = [f for f in d.rglob('*') if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            print(f"[empty] {d}")
            continue

        print(f"\n{d.relative_to(PROJECT_ROOT)}  ({len(files)} files)")
        for f in tqdm(files, desc=d.name, unit='file'):
            status = process_file(f, args.dry_run)
            if status == 'skipped':
                counts['skipped'] += 1
            elif status.startswith('error'):
                counts['error'] += 1
                print(f"  ERROR {f.name}: {status}")
            else:
                counts['resampled'] += 1
                if args.dry_run:
                    print(f"  {f.name}: {status}")

    print(f"\nDone: {counts['resampled']} resampled, "
          f"{counts['skipped']} already at {TARGET_SR}Hz, "
          f"{counts['error']} errors")
    if counts['error']:
        sys.exit(1)


if __name__ == '__main__':
    main()
```

### 3.2 Simulator Code

```python
"""
simulator/generate_pairs.py
Generates (mic_signal, clean_vocal) training pairs
using teacher forcing feedback loop simulation.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import multiprocessing
import random
import sys
from scipy.signal import fftconvolve, butter, sosfilt  # fftconvolve: O(N log N) required for long IRs
from tqdm import tqdm

# NOTE: pyroomacoustics is NOT imported at module level.
# It must be imported inside worker functions (synthetic_feedback_ir, generate_pair) only.
# Reason: multiprocessing fork copies the parent's memory including any OpenBLAS locks
# that pyroomacoustics/numpy acquired. Workers that call pra after forking from a parent
# that imported pra will deadlock silently. Importing inside the worker avoids this.

# NOTE: librosa is NOT imported here intentionally.
# All audio is pre-resampled to SR before generating pairs (see preprocess.py).
# This avoids librosa re-initialization inside worker processes, which causes
# multiprocessing failures on macOS (spawn start method).

SR      = 48000   # all audio must be pre-resampled to this rate before running
N_FFT   = 960     # DeepFilterNet3: 20ms frame at 48kHz → 960 samples → 481 freq bins
HOP     = 480     # DeepFilterNet3: 10ms hop at 48kHz (50% overlap)
# 48kHz covers full vocal range (100Hz–22kHz). No prototype simplification — we train
# at the final product sample rate from the start.

PROJECT_ROOT = Path(__file__).parent.parent  # anchored — works regardless of cwd

def load_ir(path):
    """Load a single IR file. All files must already be at SR."""
    ir, sr = sf.read(path)
    assert sr == SR, f"IR {path} is {sr}Hz — run preprocess.py first"
    if ir.ndim > 1:
        ir = ir[:, 0]
    return ir.astype(np.float32)

def synthetic_feedback_ir(near_field=False):
    """
    Generate a synthetic loudspeaker → mic IR.
    near_field=True simulates a stage monitor (short path, high direct energy).
    near_field=False simulates PA mains (long path, more reverberant).
    """
    import pyroomacoustics as pra  # local import — avoids macOS fork+OpenBLAS deadlock
    if near_field:
        # Monitor: small room slice, speaker close to mic
        room_dims  = np.random.uniform([3, 3, 2.5], [6, 5, 3.5])
        speaker_pos = [
            np.random.uniform(0.3, 1.0),
            np.random.uniform(0.3, 0.8),
            np.random.uniform(0.3, 0.8)
        ]
    else:
        # Mains: larger room, speaker elevated and distant
        room_dims  = np.random.uniform([6, 6, 3], [15, 12, 6])
        speaker_pos = [
            np.random.uniform(0.5, room_dims[0] - 0.5),
            np.random.uniform(0.3, 1.5),
            np.random.uniform(2.5, room_dims[2] - 0.5)
        ]

    absorption = np.random.uniform(0.1, 0.5)
    mic_pos    = [
        np.random.uniform(1.0, room_dims[0] - 1.0),
        np.random.uniform(room_dims[1] * 0.4, room_dims[1] * 0.8),
        np.random.uniform(1.5, 1.9)
    ]

    room = pra.ShoeBox(
        room_dims, fs=SR,
        materials=pra.Material(absorption),
        max_order=10
    )
    room.add_source(speaker_pos)
    room.add_microphone(mic_pos)
    room.simulate()
    return room.rir[0][0].astype(np.float32)

def generate_pair(args):
    vocal_path, mains_irs, monitor_irs, room_irs, noise_files, output_dir, idx = args
    try:
        # Load clean vocal (already at SR after preprocess.py)
        vocal, vocal_sr = sf.read(vocal_path)
        assert vocal_sr == SR, f"{vocal_path} is {vocal_sr}Hz — run preprocess.py first"
        if vocal.ndim > 1:
            vocal = vocal[:, 0]

        target_len = 3 * SR
        start = random.randint(0, len(vocal) - target_len)
        vocal = vocal[start:start + target_len].astype(np.float32)

        # ── MAINS feedback path ──────────────────────────────────────────────
        # 60% real measured, 40% synthetic when real available
        if mains_irs and random.random() < 0.6:
            mains_ir = load_ir(random.choice(mains_irs))
        else:
            mains_ir = synthetic_feedback_ir(near_field=False)
        mains_gain = np.random.uniform(0.2, 0.75)   # lower gain — long path attenuates
        mains_fb   = fftconvolve(vocal, mains_ir)[:target_len] * mains_gain

        # ── MONITOR feedback path ────────────────────────────────────────────
        # Independently sampled gain — monitors can be much hotter than mains
        if monitor_irs and random.random() < 0.6:
            monitor_ir = load_ir(random.choice(monitor_irs))
        else:
            monitor_ir = synthetic_feedback_ir(near_field=True)
        monitor_gain = np.random.uniform(0.3, 0.85)  # higher gain — short direct path
        monitor_fb   = fftconvolve(vocal, monitor_ir)[:target_len] * monitor_gain

        # Randomly drop one path entirely (not every scenario has both)
        if random.random() < 0.2:
            mains_fb   = np.zeros(target_len, dtype=np.float32)  # mains-off scenario
        if random.random() < 0.2:
            monitor_fb = np.zeros(target_len, dtype=np.float32)  # monitor-off scenario

        # ── Room reverb on direct vocal ──────────────────────────────────────
        if room_irs:
            room_ir = load_ir(random.choice(room_irs))
        else:
            # Fallback: generate a diffuse-field-like synthetic room IR.
            # NOTE: synthetic_feedback_ir() is NOT used here — it models a
            # directional loudspeaker→mic path. For room reverb of the direct
            # vocal we want a more diffuse reflection profile. With pyroomacoustics
            # we simulate a receiver near the source (close-mic), high absorption
            # to keep it short and relatively direct.
            room_dims   = np.random.uniform([4, 4, 3], [12, 10, 5])
            absorption  = np.random.uniform(0.3, 0.7)   # moderate-to-high absorption
            source_pos  = [room_dims[0]/2, room_dims[1]/2, 1.6]
            mic_pos     = [room_dims[0]/2 + np.random.uniform(0.1, 0.5),
                           room_dims[1]/2, 1.6]
            import pyroomacoustics as pra  # local import — avoids macOS fork+OpenBLAS deadlock
            room_obj = pra.ShoeBox(room_dims, fs=SR,
                                   materials=pra.Material(absorption), max_order=8)
            room_obj.add_source(source_pos)
            room_obj.add_microphone(mic_pos)
            room_obj.simulate()
            room_ir = room_obj.rir[0][0].astype(np.float32)
        reverberant_vocal = fftconvolve(vocal, room_ir)[:target_len]

        # ── Noise ────────────────────────────────────────────────────────────
        noise_path = random.choice(noise_files)
        noise, noise_sr = sf.read(noise_path)
        assert noise_sr == SR, f"{noise_path} is {noise_sr}Hz — run preprocess.py first"
        if noise.ndim > 1:
            noise = noise[:, 0]
        if len(noise) < target_len:
            noise = np.tile(noise, (target_len // len(noise)) + 1)
        noise = noise[random.randint(0, len(noise) - target_len):][:target_len]

        snr_db       = np.random.uniform(5, 40)
        vocal_rms    = np.sqrt(np.mean(reverberant_vocal ** 2)) + 1e-8
        noise_rms    = np.sqrt(np.mean(noise ** 2)) + 1e-8
        noise_scaled = noise * (vocal_rms / noise_rms) * (10 ** (-snr_db / 20))

        # ── Combine ──────────────────────────────────────────────────────────
        mic_signal = reverberant_vocal + mains_fb + monitor_fb + noise_scaled

        # Reference signal: the reverberant vocal — what the box outputs to the PA.
        # In hardware, the box has direct access to its own line output (the signal
        # sent to the amp). In simulation, reverberant_vocal is the correct proxy —
        # it's the direct signal path before feedback is added.
        # Meta-AF uses (mic, ref) pairs to estimate and cancel H(z) in real time.
        ref_signal = reverberant_vocal

        # HPF the reverberant vocal for the evaluation target (clean_*.wav).
        # The model is trained to output HPF'd signal (reverb_np in recursive_train.py is HPF'd).
        # Comparing a non-HPF'd reference against HPF'd model output in score.py would penalize
        # the model for sub-90Hz content that it correctly removed — unfair PESQ/STOI penalty.
        # ref_*.wav stays non-HPF'd: run_inference.py HPFs it before passing to the model,
        # matching the HPF'd teacher-forcing ref used in training.
        _clean_hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
        reverberant_vocal_hpf = sosfilt(_clean_hpf, reverberant_vocal).astype(np.float32)

        peak      = max(np.max(np.abs(mic_signal)), np.max(np.abs(reverberant_vocal_hpf))) + 1e-8
        scale     = 0.707 / peak
        mic_signal = (mic_signal          * scale).astype(np.float32)
        clean      = (reverberant_vocal_hpf * scale).astype(np.float32)  # HPF'd — matches model output
        ref_signal = (ref_signal            * scale).astype(np.float32)  # non-HPF'd — HPF'd in run_inference.py

        out = Path(output_dir)
        sf.write(out / f'mic_{idx:06d}.wav',   mic_signal, SR, subtype='PCM_16')
        sf.write(out / f'clean_{idx:06d}.wav', clean,      SR, subtype='PCM_16')
        sf.write(out / f'ref_{idx:06d}.wav',   ref_signal, SR, subtype='PCM_16')

        return None   # success

    except Exception as e:
        print(f"Worker error on idx {idx}: {e}", flush=True)
        return f"idx {idx}: {e}"   # returned to generate_dataset for error rate tracking

def generate_dataset(n_pairs=50000, n_workers=8, val_pairs=1000):
    data = PROJECT_ROOT / 'data'
    vocal_files  = list((data / 'clean_vocals').rglob('*.wav'))
    mains_irs    = list((data / 'venue_irs' / 'mains').rglob('*.wav'))
    monitor_irs  = list((data / 'venue_irs' / 'monitors').rglob('*.wav'))
    room_irs     = list((data / 'public_irs').rglob('*.wav'))
    noise_files  = list((data / 'noise').rglob('*.wav'))

    if not vocal_files:
        sys.exit("ERROR: No vocal files in data/clean_vocals/ — download EARS/VCTK first.")
    if not noise_files:
        sys.exit("ERROR: No noise files in data/noise/ — download DNS noise set first.")
    if not mains_irs and not monitor_irs:
        print("WARNING: No real venue IRs found — using synthetic only. "
              "Run venue_sweep.py to collect real IRs.")

    # Filter to files >= 3s so workers never silently skip short clips
    target_len = 3 * SR
    vocal_files = [
        f for f in vocal_files
        if (info := sf.info(str(f))).frames / info.samplerate >= 3.0
    ]
    if not vocal_files:
        sys.exit("ERROR: No vocal files >= 3 seconds found after filtering.")

    # Split into train/ and val/ at generation time so each stage's dataloader
    # can point at separate directories — avoids relying on a val_split config key.
    train_dir = data / 'training_pairs' / 'train'
    val_dir   = data / 'training_pairs' / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total = n_pairs + val_pairs
    print(f"Generating {n_pairs} train + {val_pairs} val pairs across {n_workers} workers...")
    print(f"  Vocals:       {len(vocal_files)} files >= 3s")
    print(f"  Mains IRs:    {len(mains_irs)} real + synthetic fallback")
    print(f"  Monitor IRs:  {len(monitor_irs)} real + synthetic fallback")
    print(f"  Room IRs:     {len(room_irs)} files (public datasets)")
    print(f"  Noise clips:  {len(noise_files)} files")

    # First n_pairs go to train/, last val_pairs go to val/
    args = [
        (random.choice(vocal_files), mains_irs, monitor_irs,
         room_irs, noise_files, str(train_dir if i < n_pairs else val_dir), i)
        for i in range(total)
    ]

    # Use 'fork' on macOS/Linux. 'spawn' is safer for CUDA but breaks librosa/scipy workers.
    # RISK: If pyroomacoustics or numpy was imported in the parent process before forking,
    # OpenBLAS internal locks can deadlock in fork'd children. Workaround: import pra and
    # numpy ONLY inside generate_pair() (inside the worker), not at module level. If you
    # see workers hanging silently, switch to ctx = multiprocessing.get_context('spawn')
    # and add if __name__ == '__main__': guard. Windows requires 'spawn' regardless.
    ctx = multiprocessing.get_context('fork')
    results = []
    with ctx.Pool(n_workers) as p:
        for result in tqdm(p.imap_unordered(generate_pair, args, chunksize=50),
                           total=total, desc='generating pairs'):
            results.append(result)

    error_count = sum(1 for r in results if r is not None)   # generate_pair returns None on success, str on error
    error_rate  = error_count / total
    print(f"Done — {total - error_count}/{total} succeeded, {error_count} errors ({error_rate:.1%})")
    if error_rate > 0.05:
        sys.exit(f"ERROR: {error_rate:.1%} failure rate exceeds 5% threshold — "
                 "check worker errors above. Likely cause: pyroomacoustics simulation "
                 "produced NaN or zero-length IR. Fix before training.")

if __name__ == '__main__':
    generate_dataset(n_pairs=50000, val_pairs=1000)
```

**Run:**
```bash
python simulator/generate_pairs.py
# generates 50,000 pairs overnight on 8 CPU cores
```

---

## Phase 4: Model Architecture + Training

### 4.0 Architecture

```
mic_in ──┬──────────────────────────────────────────► FDKFNet
         │                                          (Kalman filter + GRU)
line_out ─┘  ← hardware loopback (v1 with hardware)     │
                                                         ▼
                                                      clean_out

v0 (no hardware yet):
mic_in ─────────────────────────────────────────────► FreqDomainNLMS
                                                      (classical NLMS, no training)
```

**v0 vs v1:**
- **v0 (laptop + USB interface, no reference signal):** Run FreqDomainNLMS only.
  Classical FDAF, no training required, instant startup, validates the signal chain.
- **v1 (hardware box with loopback):** Run FDKFNet. The trained neural Kalman filter
  uses the hardware line output as reference — the box physically owns both sides of
  the signal chain, which De-Feedback (a plugin) cannot do.

**FDKFNet — Frequency-Domain Kalman Filter with neural covariance estimation:**

A single integrated model that replaces both the FDAF and the neural enhancement stage.
Directly models the acoustic feedback loop — no separate "enhancement" step.

- **Kalman filter** (per frequency bin): tracks H(z), the acoustic feedback path from
  speaker output to mic. Per-frame update using the reference signal. Produces a speech
  estimate by subtracting the estimated feedback component.
- **GRU** (small, ERB-band compressed): estimates the Kalman covariances Q_k
  (how fast H(z) is changing) and R_k (speech + noise level at each bin). This is
  what makes the filter adaptive rather than fixed.

This is the same class of architecture as **NeuralKalmanAHS** (Zhang et al. 2023,
arXiv 2309.16049) — the published state-of-the-art for streaming acoustic howling
suppression. Key difference from DN3: the Kalman filter explicitly models the feedback
loop dynamics rather than applying a spectral mask that could protect tonal feedback.

**Why not DeepFilterNet3:**
DN3 was designed to *enhance* periodic/harmonic content — the opposite of what's needed
for feedback suppression. Published research on acoustic howling suppression consistently
uses Kalman filter + LSTM architectures, not speech enhancement models. No published
paper evaluates DN3 on feedback suppression.

**Trained with recursive training (not teacher forcing):**
Teacher forcing (training on pre-computed mic+feedback pairs) is the documented central
failure mode of early AHS approaches — models train on signal statistics that don't exist
at inference (because their own output feeds back). The fix: during training, the model's
output is fed back through the simulated acoustic path to generate the next frame's input.
See arXiv 2309.16048 (ICASSP 2024) for the approach.

**Failure mode analysis — why this is lower risk than DN3:**
Even with a poorly trained GRU (random or constant covariance estimates), the Kalman
filter still tracks H(z) and suppresses feedback — it just uses fixed gain instead of
adaptive gain. The performance floor is a working fixed-gain Kalman AHS, not zero. This
is fundamentally different from DN3: if DN3 training failed or was misconfigured, the
model would output garbage. Here, training improves a working signal processor rather
than enabling one from scratch. Worst realistic outcome: model performs like classical
FDAF. Best case: model outperforms FDAF by adapting suppression speed to room + signal
dynamics.

**Estimated compute on Pi 5:**
- FDKFNet (~500K params, ERB-compressed GRU): RTF ~0.05–0.08 (estimated)
- STFT/ISTFT wrapper: negligible
- Total: RTF ~0.08 — significant headroom within 10ms budget

---

### 4.1 v0 Fallback: FreqDomainNLMS

Used when no trained FDKFNet checkpoint exists yet, or when testing without hardware
loopback. Classical signal processor — no training, no ML. Lives in `simulator/fdaf.py`.

```python
"""
simulator/fdaf.py — Frequency-Domain NLMS Adaptive Filter (overlap-save method)

Used for Stage 1 acoustic feedback cancellation. Given the reference signal
(what the PA is playing) and the mic signal, estimates the feedback path H(z)
and subtracts it from the mic in real time.

Per-block cost: 4 FFTs of size 2048 — < 1ms on Pi 5. No sample loops.
Filter length 1024 taps = 21.3ms at 48kHz — covers speaker-to-mic acoustic
delays for any stage geometry up to ~7m.

Converges online in ~0.5–2 seconds depending on signal level and mu.
No training required.
"""

import numpy as np


class FreqDomainNLMS:
    """Overlap-save frequency-domain NLMS adaptive filter.

    Args:
        filter_len:  FIR tap count. Must span the max acoustic path delay.
                     At 48kHz: speaker 3m away = 421 taps minimum.
                     Default 1024 (21.3ms) covers any realistic stage geometry.
        block_size:  Samples per callback — must match inference HOP (480 at 48kHz).
        mu:          NLMS step size. 0.02 = moderate (stable, converges in ~1–2s).
                     Increase to 0.05–0.1 for faster convergence if signal is strong.
        eps:         Power floor — prevents divide-by-zero during silence.
    """

    def __init__(self, filter_len: int = 1024, block_size: int = 480,
                 mu: float = 0.02, eps: float = 1e-6):
        self.L   = filter_len
        self.B   = block_size
        # FFT size: smallest power of 2 >= L + B - 1
        self.N   = 1 << (filter_len + block_size - 1).bit_length()
        self.mu  = mu
        self.eps = eps
        # Frequency-domain filter weights (complex)
        self.W       = np.zeros(self.N // 2 + 1, dtype=np.complex128)
        # Overlap-save reference input buffer (N samples)
        self.ref_buf = np.zeros(self.N, dtype=np.float64)

    def process(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """Process one block. Returns the residual (feedback-cancelled signal).

        mic_block: (B,) float32 — raw microphone signal
        ref_block: (B,) float32 — reference = what the speaker is currently playing
        Returns:   (B,) float32 — mic with estimated feedback subtracted
        """
        # Overlap-save: shift buffer left, append new reference block
        self.ref_buf = np.roll(self.ref_buf, -self.B)
        self.ref_buf[-self.B:] = ref_block.astype(np.float64)

        # Frequency-domain reference
        X = np.fft.rfft(self.ref_buf)                         # (N//2+1,) complex

        # Estimated feedback via linear convolution (overlap-save valid output = last B samples)
        y = np.fft.irfft(self.W * X)[self.N - self.B:].astype(np.float32)

        # Residual after cancellation
        e = mic_block - y

        # Constrained NLMS update — enforce causal FIR of length L to prevent circular aliasing
        e_padded               = np.zeros(self.N, dtype=np.float64)
        e_padded[self.N - self.B:] = e.astype(np.float64)
        E                      = np.fft.rfft(e_padded)

        grad           = np.fft.irfft(E * np.conj(X))    # cross-correlation gradient (N,)
        grad[self.L:]  = 0.0                              # zero non-causal taps (constraint)
        G              = np.fft.rfft(grad)

        self.W += (self.mu / (np.mean(np.abs(X) ** 2) + self.eps)) * G

        return e

    def reset(self):
        """Reset filter state — call when acoustic environment changes significantly."""
        self.W[:]       = 0.0
        self.ref_buf[:] = 0.0
```

---

### 4.2 FDKFNet Model Architecture

```python
"""
train/model.py — Frequency-Domain Kalman Filter Network (FDKFNet)

Kalman filter per frequency bin + small GRU for covariance estimation.
Causal, streaming-compatible. ~500K parameters.

Based on NeuralKalmanAHS (Zhang et al. 2023, arXiv 2309.16049).
"""

import torch
import torch.nn as nn
import torchaudio

SR       = 48000
N_FFT    = 960
HOP      = 480
WIN_LEN  = 960
N_FREQS  = N_FFT // 2 + 1   # 481
N_BANDS  = 64                # ERB-compressed bands for GRU input
HIDDEN   = 128
N_LAYERS = 2


class FDKFNet(nn.Module):
    def __init__(self, n_freqs=N_FREQS, n_bands=N_BANDS, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_bands = n_bands

        # Mel filterbank: compresses 481 FFT bins → 64 bands for GRU
        # (ERB and mel are equivalent for this purpose)
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs, n_mels=n_bands,
            f_min=80.0, f_max=24000.0, sample_rate=SR, norm=None
        )  # (n_freqs, n_bands)
        self.register_buffer('mel_fb', fb)

        # GRU: estimates log(Q_k) and log(R_k) per ERB band per frame
        # Input: log power of mic, ref, and innovation per band (3 × n_bands)
        self.gru  = nn.GRU(n_bands * 3, hidden, num_layers=n_layers, batch_first=True)
        # Output: log(Q) and log(R) per band (interpolated back to all bins)
        self.proj = nn.Linear(hidden, n_bands * 2)

    def _to_erb(self, power):
        """power: (B, F) real → (B, n_bands) log-compressed"""
        return torch.log1p(power @ self.mel_fb)   # (B, n_bands)

    def init_state(self, batch_size=1, device='cpu'):
        F = self.n_freqs
        H = torch.zeros(batch_size, F, dtype=torch.cfloat,  device=device)
        P = torch.ones( batch_size, F, dtype=torch.float32, device=device) * 0.1
        h = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=device)
        return H, P, h

    def forward_frame(self, mic_f, ref_f, H_prev, P_prev, gru_h):
        """
        One STFT frame.
        mic_f, ref_f : (B, F) complex
        H_prev       : (B, F) complex  — feedback path estimate
        P_prev       : (B, F) float    — error covariance
        Returns: speech_f, H_new, P_new, gru_h
        """
        ref_power  = ref_f.abs().pow(2)                       # (B, F) real
        innovation = mic_f - H_prev * ref_f                   # (B, F) complex

        # GRU features (ERB-compressed log power)
        feat = torch.cat([
            self._to_erb(mic_f.abs().pow(2)),
            self._to_erb(ref_power),
            self._to_erb(innovation.abs().pow(2)),
        ], dim=-1).unsqueeze(1)                               # (B, 1, 3*n_bands)

        gru_out, gru_h = self.gru(feat, gru_h)               # (B, 1, hidden)
        cov = self.proj(gru_out[:, 0, :])                     # (B, 2*n_bands)
        log_Q = cov[:, :self.n_bands]
        log_R = cov[:, self.n_bands:]

        # Interpolate covariances back to all frequency bins
        Q_k = torch.exp(log_Q @ self.mel_fb.T) + 1e-8        # (B, F)
        R_k = torch.exp(log_R @ self.mel_fb.T) + 1e-8        # (B, F)

        # VAD gate — freeze H and P updates when no feedback is present.
        # When mic energy >> ref energy, the signal is clean speech with no active
        # feedback loop. Updating H(z) in this state causes the filter to adapt to
        # speech harmonics, which corrupts the feedback path estimate. When feedback
        # risk returns, H(z) is wrong and takes time to re-converge — causing audible
        # distortion on exactly the sustained notes most vulnerable to feedback.
        #
        # Gate logic: if mic_power / (ref_power + eps) > VAD_RATIO, freeze.
        # VAD_RATIO=20.0 (13dB): feedback present only when ref is within 13dB of mic.
        # When ref is 13dB or more below mic, speaker output is low relative to mic —
        # no meaningful feedback path is active. Use .detach() so gate doesn't block grad.
        VAD_RATIO   = 20.0
        mic_power   = mic_f.abs().pow(2)                      # (B, F)
        vad_gate    = (mic_power / (ref_power + 1e-8) < VAD_RATIO).float().detach()
        # vad_gate = 1.0 → feedback present, update H/P normally
        # vad_gate = 0.0 → clean speech, hold H/P at previous values

        # Kalman update (gated)
        P_pred = P_prev + Q_k
        S_k    = ref_power * P_pred + R_k                     # innovation covariance
        K_k    = P_pred * ref_f.conj() / S_k                  # Kalman gain (complex)
        H_new  = H_prev + vad_gate * K_k * innovation         # freeze if no feedback
        # (K_k * ref_f).real is provably real: K_k = P_pred*conj(ref_f)/S_k, so
        # K_k*ref_f = P_pred*|ref_f|^2/S_k — a ratio of real scalars.
        # FRAGILE: .real is correct ONLY for this specific K_k formula. If you change
        # the Kalman gain expression, re-derive this identity before keeping .real.
        P_new  = (1.0 - vad_gate * (K_k * ref_f).real) * P_pred
        # Clamp both ends: min prevents divide-by-zero; max prevents unbounded growth
        # during silence (when ref ≈ 0, K_k ≈ 0, and P_new ≈ P_prev + Q_k each frame).
        # Without the max, P inflates over multi-second silences and K_k spikes when
        # the reference returns — causing transient divergence artifacts.
        P_new  = P_new.clamp(min=1e-8, max=100.0)

        speech_f = mic_f - H_new * ref_f                      # clean speech estimate
        return speech_f, H_new, P_new, gru_h

    def forward(self, mic_frames, ref_frames, H_0=None, P_0=None, gru_h=None):
        """
        mic_frames, ref_frames: (B, T, F) complex
        Returns: speech_frames (B, T, F) complex, final state tuple
        """
        B, T, F = mic_frames.shape
        if H_0 is None:
            H_0, P_0, gru_h = self.init_state(B, mic_frames.device)
        H_k, P_k = H_0, P_0
        out = []
        for t in range(T):
            s, H_k, P_k, gru_h = self.forward_frame(
                mic_frames[:, t, :], ref_frames[:, t, :], H_k, P_k, gru_h
            )
            out.append(s)
        return torch.stack(out, dim=1), (H_k, P_k, gru_h)
```

### 4.3 IR Pool Pre-computation

Recursive training loads IRs at runtime — no pre-generated pairs needed. Pre-compute
a pool of synthetic IRs once before training:

```bash
# Generate 2000 synthetic IRs from build_room_simulation() — do this once
python simulator/generate_ir_pool.py --n 2000 --out data/ir_pool/
# Saves: data/ir_pool/mains_XXXXXX.wav, monitors_XXXXXX.wav, sub_XXXXXX.wav
# Takes: ~2–8 hours depending on max_order and hardware
```

```python
"""simulator/generate_ir_pool.py"""
import argparse, random, sys, soundfile as sf, numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))   # allows: python simulator/generate_ir_pool.py from project root
from generate_pairs import build_room_simulation
from transducer_frs import build_transducer_library

SR = 48000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',   type=int, default=2000)
    ap.add_argument('--out', default='data/ir_pool')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load transducer FR library once (FIR design is not free)
    tx_lib      = build_transducer_library()
    mic_firs    = list(tx_lib['mics'].values())
    spk_firs    = list(tx_lib['speakers'].values())
    mic_names   = list(tx_lib['mics'].keys())
    spk_names   = list(tx_lib['speakers'].keys())
    print(f"Transducer library: {len(mic_firs)} mic FRs, {len(spk_firs)} speaker FRs")

    # Build weights so 'flat' (identity) appears ~15% of the time regardless of how many
    # real FRs are in the library. Without this, flat's weight is 1/(N+1) which drops to
    # ~6% with 14 mic entries — the model sees too little uncolored signal.
    FLAT_PROB = 0.15
    def _weighted_choice(names, firs):
        n = len(names)
        weights = [FLAT_PROB if name == 'flat' else (1.0 - FLAT_PROB) / (n - 1) for name in names]
        return random.choices(firs, weights=weights, k=1)[0]

    for i in tqdm(range(args.n)):
        mains_ir, mon_ir, _, sub_ir, _ = build_room_simulation()

        # Apply random transducer coloration (see section 3.0.8)
        # Same mic for both paths (performer has one mic regardless of speaker type)
        # Independent speaker FIRs (mains ≠ monitors in most venues)
        # 'flat' weighted to ~15% to prevent over-correcting clean signal
        mic_fir         = _weighted_choice(mic_names, mic_firs)
        mains_spk_fir   = _weighted_choice(spk_names, spk_firs)
        monitor_spk_fir = _weighted_choice(spk_names, spk_firs)
        mains_ir = fftconvolve(fftconvolve(mains_ir, mains_spk_fir),   mic_fir).astype(np.float32)
        mon_ir   = fftconvolve(fftconvolve(mon_ir,   monitor_spk_fir), mic_fir).astype(np.float32)

        for name, ir in [('mains', mains_ir), ('monitor', mon_ir), ('sub', sub_ir)]:
            sf.write(str(out / f'{name}_{i:06d}.wav'), ir.astype(np.float32), SR)

if __name__ == '__main__':
    main()
```

### 4.4 Recursive Training

```python
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
  teacher_forcing_prob starts at TF_START=0.5 and decays to 0 by TF_DECAY_EPOCHS.
  During TF phase, ref = clean reverberant vocal (stable warm start).
  After decay, ref = model's own previous output (true deployment condition).
  This prevents early training instability when the model outputs garbage.

Memory:
  BPTT graph spans SEQ_FRAMES. The feedback convolution at frame t only needs
  the last ceil((L_ir + HOP - 1) / HOP) frames from the outputs list — older
  frames are excluded from the graph automatically, bounding memory.
  Start at SEQ_FRAMES=50; increase to 100 → 200 once loss is decreasing smoothly.
"""

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import random
from pathlib import Path
from scipy.signal import fftconvolve, butter, sosfilt, sosfilt_zi
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

SEQ_FRAMES       = 50     # BPTT window. Start here; ramp 50 → 100 → 200 once stable
BATCH_SIZE       = 4      # gradient accumulation
EPOCHS           = 100
LR               = 3e-4
GRAD_CLIP        = 1.0
CLIP_LEVEL       = 0.95   # soft-clip ceiling on model output in feedback path

TF_START         = 0.5    # initial teacher-forcing probability (ref = clean vocal)
TF_DECAY_EPOCHS  = 30     # linear decay to 0 — after this, ref = model's own output

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

def sample_gain():
    """
    Returns (mains_gain, monitor_gain) independently sampled.
    40% normal (0.2–0.6) | 35% near-threshold (0.6–0.9) | 25% active (0.9–1.5)
    Both paths sampled independently — monitor can be hot while mains is quiet,
    which is the most common failure mode in small venues and HOW settings.
    """
    def _one():
        t = random.random()
        if t < 0.40:  return random.uniform(0.2, 0.6)
        elif t < 0.75: return random.uniform(0.6, 0.9)
        else:          return random.uniform(0.9, 1.5)
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

    # Linear convolution via F.conv1d (differentiable cross-correlation)
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

    # Noise scaling
    noise_np   = noise_np[:target_len]
    vocal_rms  = float(np.sqrt(np.mean(reverb_np ** 2))) + 1e-8
    noise_rms  = float(np.sqrt(np.mean(noise_np  ** 2))) + 1e-8
    snr_db     = np.random.uniform(5, 40)
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
    total_loss  = torch.tensor(0.0, device=device)

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
        if outputs and random.random() >= teacher_forcing_prob:
            ref_frame = outputs[-1]                      # model output — deployment condition
        else:
            ref_frame = reverb_frame.detach()            # teacher signal — stable warm start

        # ── Differentiable STFT ───────────────────────────────────────────────
        mic_f = torch_stft(mic_frame, window).unsqueeze(0)   # (1, N_FREQS)
        ref_f = torch_stft(ref_frame, window).unsqueeze(0)

        # ── Model forward (gradients flow through H, P, gru_h across frames) ─
        speech_f, H, P, gru_h = model.forward_frame(mic_f, ref_f, H, P, gru_h)

        # ── Output frame (differentiable, soft-clipped, feeds back next frame)
        out_frame = torch_istft(speech_f.squeeze(0))
        out_frame = torch.tanh(out_frame / CLIP_LEVEL) * CLIP_LEVEL
        outputs.append(out_frame)

        # ── Loss: spectral MSE against HPF'd reverberant vocal ────────────────
        clean_f    = torch_stft(reverb_frame.detach(), window).unsqueeze(0)
        total_loss = total_loss + F.mse_loss(speech_f, clean_f)

    if not outputs:
        return None
    return total_loss / SEQ_FRAMES


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
    min_dur = SEQ_FRAMES * HOP / SR
    vocal_files = [f for f in vocal_files
                   if sf.info(str(f)).frames / sf.info(str(f)).samplerate >= min_dur]
    assert vocal_files, f"No vocal files >= {min_dur:.1f}s after filtering"

    ckpt_dir = PROJECT_ROOT / 'checkpoints' / 'fdkfnet'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FDKFNet: {n_params:,} parameters on {device}")
    print(f"BPTT: {SEQ_FRAMES} frames ({SEQ_FRAMES * HOP / SR * 1000:.0f}ms), "
          f"IR cap {MAX_IR_SAMPLES/SR:.1f}s")
    print(f"Dual-path: {len(mains_ir_files)} mains + {len(monitor_ir_files)} monitor IRs")
    print(f"TF schedule: {TF_START:.1f} → 0 over {TF_DECAY_EPOCHS} epochs")

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Teacher-forcing prob: linear decay from TF_START → 0
        tf_prob = max(0.0, TF_START * (1.0 - (epoch - 1) / TF_DECAY_EPOCHS))

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

        torch.save({'epoch': epoch, 'model': model.state_dict()},
                   str(ckpt_dir / f'epoch_{epoch:03d}.pt'))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       str(ckpt_dir / 'best.pt'))
            print("  ✓ New best")


if __name__ == '__main__':
    train()
```

### 4.5 Training Run

```bash
# Step 1: Generate IR pool (once)
python simulator/generate_ir_pool.py --n 2000 --out data/ir_pool/

# Step 2: Train FDKFNet recursively
python train/recursive_train.py

# Monitor:
tensorboard --logdir checkpoints/fdkfnet/

# Expected compute: 8–15 GPU hours on RTX 3090 / A100
# (~500K params, sequences of 200 frames — faster than DN3 fine-tuning)
# Use Lambda Labs, Vast.ai, or Google Colab Pro A100
```

**Mixed precision:** wrap model forward + loss in `torch.cuda.amp.autocast()` and use
`GradScaler` — cuts GPU hours roughly in half on A100/RTX 30xx+.

### 4.6 Iteration Loop

Train → listen at epochs 10, 30, 60, 100 → identify failure modes → fix IR pool or
collect real feedback recordings → retrain from last checkpoint → repeat.

Key things to listen for at each checkpoint:
- Does the model suppress high-frequency feedback (2–8kHz) without adding harshness?
- Does it hold on a sustained vowel (Scenario 6) without pitch artifacts?
- Does it converge within 1–2 seconds at inference (FDKF should stabilize quickly)?

---

## Phase 5: Evaluation Protocol

### 5.1 Objective Metrics (Automated)

Two scripts needed: one to run the model on the val set, one to score the output.

**Step 1 — Run batch inference on val set:**
```python
"""eval/run_inference.py — Run trained FDKFNet on a directory of mic+ref files, save enhanced files.

Works on both val set (mic_000001.wav + ref_000001.wav → enhanced_000001.wav) and
listening test files (mic.wav + ref.wav → enhanced.wav).
Falls back to FreqDomainNLMS v0 if no FDKFNet checkpoint exists.

Usage:
    # Val set (default):
    python eval/run_inference.py

    # Listening test — one scenario at a time:
    python eval/run_inference.py --val-dir data/listening_test/1_loud_feedback \\
                                  --out-dir data/listening_test/1_loud_feedback

    # All listening test scenarios at once (bash):
    for d in data/listening_test/*/; do
        python eval/run_inference.py --val-dir "$d" --out-dir "$d"
    done
"""

import argparse, sys, torch, soundfile as sf, numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfilt_zi

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'train'))
sys.path.insert(0, str(PROJECT_ROOT / 'simulator'))
from model import FDKFNet
from fdaf import FreqDomainNLMS

SR   = 48000
HOP  = 480          # 10ms at 48kHz — matches training frame size
N_FFT = 960         # matches FDKFNet default (HOP*2)
WIN   = torch.hann_window(N_FFT)

# Console HPF — must match recursive_train.py training conditions exactly.
# Training applies a 2nd-order Butterworth HPF at 90Hz to every frame.
# Running without it at inference creates a train/inference spectral mismatch
# in the sub-100Hz bins that the Kalman filter has never seen unfiltered.
_console_hpf = butter(2, 90.0 / (SR / 2), btype='high', output='sos')


def stft_frame(x_np):
    """Single HOP block → complex FFT frame (1, N_FFT//2+1)."""
    x = torch.from_numpy(x_np).unsqueeze(0)          # (1, HOP)
    # pad to N_FFT for first frame
    x_padded = torch.nn.functional.pad(x, (N_FFT - HOP, 0))
    X = torch.fft.rfft(x_padded * WIN, n=N_FFT)      # (1, N_FFT//2+1)
    return X


def istft_frame(X):
    """Complex FFT frame (1, N_FFT//2+1) → HOP samples numpy."""
    x = torch.fft.irfft(X, n=N_FFT)                  # (1, N_FFT)
    # overlap-add: keep last HOP samples
    return x[0, -HOP:].numpy()


def run_batch(val_dir=None, out_dir=None, checkpoint=None):
    val_dir    = Path(val_dir    or PROJECT_ROOT / 'data' / 'training_pairs' / 'val')
    out_dir    = Path(out_dir    or PROJECT_ROOT / 'data' / 'eval_output')
    checkpoint = Path(checkpoint or PROJECT_ROOT / 'checkpoints' / 'fdkfnet' / 'best.pt')

    if checkpoint.exists():
        model = FDKFNet()
        ckpt  = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
        model.eval()
        use_fdkf = True
        print(f"FDKFNet loaded from {checkpoint}")
    else:
        print(f"No FDKFNet checkpoint at {checkpoint} — falling back to v0 FreqDomainNLMS")
        use_fdkf = False

    out_dir.mkdir(parents=True, exist_ok=True)

    mic_files = sorted(val_dir.glob('mic*.wav'))
    if not mic_files:
        print(f"No mic*.wav files found in {val_dir}")
        return

    for mic_path in mic_files:
        out_stem = mic_path.stem.replace('mic', 'enhanced', 1) or 'enhanced'
        mic_np, sr = sf.read(str(mic_path), dtype='float32')
        assert sr == SR, f"{mic_path.name}: expected {SR}Hz, got {sr}Hz"

        ref_path = mic_path.parent / mic_path.name.replace('mic', 'ref', 1)
        if ref_path.exists():
            ref_np, _ = sf.read(str(ref_path), dtype='float32')
            min_len = min(len(mic_np), len(ref_np))
            mic_np, ref_np = mic_np[:min_len], ref_np[:min_len]
        else:
            ref_np = np.zeros_like(mic_np)
            min_len = len(mic_np)

        # Console HPF — apply to full file before frame loop (stateful, causal).
        # Matches the HPF applied during recursive training (recursive_train.py).
        mic_np  = sosfilt(_console_hpf, mic_np)
        ref_np  = sosfilt(_console_hpf, ref_np)

        enhanced = np.zeros(min_len, dtype=np.float32)

        if use_fdkf:
            H, P, gru_h = model.init_state(batch_size=1, device='cpu')
            for i in range(0, min_len - HOP + 1, HOP):
                mic_f = stft_frame(mic_np[i:i + HOP])   # (1, F) complex
                ref_f = stft_frame(ref_np[i:i + HOP])
                with torch.no_grad():
                    speech_f, H, P, gru_h = model.forward_frame(mic_f, ref_f, H, P, gru_h)
                enhanced[i:i + HOP] = istft_frame(speech_f)
        else:
            fdaf = FreqDomainNLMS(filter_len=1024, block_size=HOP, mu=0.02)
            for i in range(0, min_len - HOP + 1, HOP):
                enhanced[i:i + HOP] = fdaf.process(mic_np[i:i + HOP], ref_np[i:i + HOP])

        sf.write(str(out_dir / f'{out_stem}.wav'), enhanced, SR, subtype='PCM_16')

    print(f"Done — {len(mic_files)} files enhanced → {out_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--val-dir',    default=None)
    p.add_argument('--out-dir',    default=None)
    p.add_argument('--checkpoint', default=None)
    args = p.parse_args()
    run_batch(args.val_dir, args.out_dir, args.checkpoint)
```

**Step 2 — Score enhanced vs clean:**
```python
"""eval/score.py — Run objective metrics on model output.

PESQ only supports 8kHz (NB) and 16kHz (WB) — files are downsampled to 16kHz
before scoring. STOI supports arbitrary sample rates natively.
"""

from pesq import pesq
from pystoi import stoi
import soundfile as sf
import numpy as np
from pathlib import Path

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

SR      = 48000
PESQ_SR = 16000   # pesq library hard requirement


def _to_pesq_sr(audio: np.ndarray) -> np.ndarray:
    """Downsample from SR to PESQ_SR for PESQ computation."""
    if not _HAS_LIBROSA:
        raise ImportError("librosa required for PESQ scoring — pip install librosa")
    return librosa.resample(audio.astype(np.float32), orig_sr=SR, target_sr=PESQ_SR)


def evaluate(enhanced_dir='data/eval_output',
             clean_dir='data/training_pairs/val'):
    scores = {'pesq': [], 'stoi': []}

    for clean_path in Path(clean_dir).glob('clean_*.wav'):
        parts = clean_path.stem.split('_', 1)
        if len(parts) < 2:
            # Listening test files are named 'clean.wav' (no suffix index) — skip
            # them here; they are handled separately by generate_test_set.py output
            continue
        idx = parts[1]
        enhanced_path = Path(enhanced_dir) / f'enhanced_{idx}.wav'
        if not enhanced_path.exists():
            continue

        clean_48,    _ = sf.read(str(clean_path),    dtype='float32')
        enhanced_48, _ = sf.read(str(enhanced_path), dtype='float32')

        min_len = min(len(clean_48), len(enhanced_48))
        clean_48    = clean_48[:min_len]
        enhanced_48 = enhanced_48[:min_len]

        # PESQ — must be at 16kHz wideband
        clean_16    = _to_pesq_sr(clean_48)
        enhanced_16 = _to_pesq_sr(enhanced_48)
        scores['pesq'].append(pesq(PESQ_SR, clean_16, enhanced_16, 'wb'))

        # STOI — native 48kHz is fine
        scores['stoi'].append(stoi(clean_48, enhanced_48, SR))

    if not scores['pesq']:
        print("No scored files found.")
        return scores

    print(f"PESQ (wideband, scored at 16kHz): {np.mean(scores['pesq']):.3f}  "
          f"(n={len(scores['pesq'])})")
    print(f"STOI (scored at 48kHz):           {np.mean(scores['stoi']):.3f}")
    return scores

if __name__ == '__main__':
    evaluate()
```

**Run evaluation:**
```bash
python eval/run_inference.py     # generates data/eval_output/enhanced_*.wav
python eval/score.py             # prints PESQ and STOI
```

**Baseline targets to beat:**
- PESQ > 2.5 (input mic signal baseline will be ~1.5–2.0)
- STOI > 0.85

### 5.2 Listening Test Protocol (Subjective — Do This Every Training Checkpoint)

**Step 0 — Generate the 7 controlled test scenarios:**
```python
"""eval/generate_test_set.py — Create targeted listening test pairs with known parameters.
Run from project root: python eval/generate_test_set.py
Writes to data/listening_test/ — each scenario has mic.wav, clean.wav, enhanced.wav (after inference).
"""
import sys, random, numpy as np, soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'simulator'))
from generate_pairs import build_room_simulation   # physics-upgraded pyroomacoustics simulator

SR = 48000

SCENARIOS = [
    {'name': '1_loud_feedback',      'mains_gain': 0.80, 'mon_gain': 0.75, 'rt60': 0.6},
    {'name': '2_moderate_feedback',  'mains_gain': 0.50, 'mon_gain': 0.45, 'rt60': 0.6},
    {'name': '3_reverberant_room',   'mains_gain': 0.55, 'mon_gain': 0.50, 'rt60': 2.2},
    {'name': '4_dry_direct',         'mains_gain': 0.70, 'mon_gain': 0.60, 'rt60': 0.2},
    {'name': '5_sibilant_vocal',     'mains_gain': 0.60, 'mon_gain': 0.55, 'rt60': 0.5},
    {'name': '6_sustained_vowel',    'mains_gain': 0.65, 'mon_gain': 0.60, 'rt60': 0.5},
    {'name': '7_multi_freq_fb',      'mains_gain': 0.75, 'mon_gain': 0.80, 'rt60': 0.8},
]

def synthetic_room_ir(rt60, sr=SR):
    """Exponentially-decaying white noise RIR for room reverb of the direct vocal.
    Peak-normalized to 1.0, which is correct for a room IR (direct path has unit gain;
    reflections decay from there). DO NOT use this for feedback path IRs —
    those use synthetic_feedback_ir() from generate_pairs.py which applies correct
    acoustic path attenuation via pyroomacoustics geometry."""
    n_samples = int(rt60 * sr)
    t = np.arange(n_samples) / sr
    ir = np.random.randn(n_samples).astype(np.float32)
    ir *= np.exp(-6.9 * t / rt60).astype(np.float32)   # -60 dB at rt60
    ir /= np.abs(ir).max() + 1e-8
    return ir

def generate_test_set(vocal_dir='data/clean_vocals', out_dir='data/listening_test'):
    out = PROJECT_ROOT / out_dir
    vocal_files = [f for f in (PROJECT_ROOT / vocal_dir).rglob('*.wav')
                   if sf.info(str(f)).frames / sf.info(str(f)).samplerate >= 3.0]
    assert vocal_files, f"No vocal files >= 3s found in {vocal_dir}"

    # Scenario 5 — prefer sibilant-rich files if tagged; else pick random
    # Scenario 6 — prefer sustained vowel files; else pick random
    # (Without phoneme tags just pick random — operator can re-pick as needed)
    random.seed(42)       # reproducible test set
    np.random.seed(42)    # build_room_simulation() uses np.random internally

    for sc in SCENARIOS:
        sc_dir = out / sc['name']
        sc_dir.mkdir(parents=True, exist_ok=True)

        vocal_path = random.choice(vocal_files)
        vocal, _ = sf.read(str(vocal_path), dtype='float32', always_2d=False)
        if vocal.ndim > 1:
            vocal = vocal.mean(axis=1)
        target_len = 3 * SR
        start = random.randint(0, len(vocal) - target_len)
        vocal = vocal[start:start + target_len]

        # Feedback path IRs — use full physics simulator for correct acoustic attenuation.
        # build_room_simulation() models directional loudspeaker→mic paths with
        # physically realistic gain (typically -20 to -40 dB) via pyroomacoustics ISM.
        mains_ir, mon_ir, room_ir, sub_ir, _meta = build_room_simulation()
        room_ir  = synthetic_room_ir(sc['rt60'])   # override with scenario-specific RT60

        reverberant_vocal = fftconvolve(vocal, room_ir)[:target_len]
        mains_fb  = fftconvolve(vocal, mains_ir)[:target_len] * sc['mains_gain']
        mon_fb    = fftconvolve(vocal, mon_ir)[:target_len]   * sc['mon_gain']
        mic_signal = (reverberant_vocal + mains_fb + mon_fb).astype(np.float32)
        clean      = reverberant_vocal.astype(np.float32)

        # ref.wav = what the speaker is playing = reverberant vocal (no feedback added).
        # This is used by run_inference.py when --ref is available (FDKFNet v1 mode).
        # Without it, run_inference.py falls back to FDAF-only, which tests v0 not v1.
        ref_signal = reverberant_vocal.astype(np.float32)

        # Normalise to -18 dBFS peak (apply same scale to all three files)
        peak = max(np.abs(mic_signal).max(), np.abs(clean).max(), 1e-8)
        scale = 0.125 / peak   # 0.125 ≈ -18 dBFS
        sf.write(str(sc_dir / 'mic.wav'),   mic_signal * scale, SR)
        sf.write(str(sc_dir / 'clean.wav'), clean      * scale, SR)
        sf.write(str(sc_dir / 'ref.wav'),   ref_signal * scale, SR)
        print(f"  {sc['name']} — mains_gain={sc['mains_gain']}, rt60={sc['rt60']}s → {sc_dir}")

    print(f"\nDone. Run eval/run_inference.py on data/listening_test/*/*mic.wav to generate enhanced.wav files.")
    print("Then listen to mic / clean / enhanced side-by-side in each scenario folder.")

if __name__ == '__main__':
    generate_test_set()
```

**Run order:**
```bash
python eval/generate_test_set.py           # writes mic.wav + clean.wav for 7 scenarios
python eval/run_inference.py               # run model on all mic.wav → enhanced.wav
# Then open each data/listening_test/<scenario>/ in an audio editor and A/B all three files
```

**The 7 scenarios and what each one tests:**
| # | Scenario | Key stress |
|---|---|---|
| 1 | Loud feedback (gain 0.80/0.75) | Does it suppress ringing without destroying vocal? |
| 2 | Moderate feedback (0.50/0.45) | Baseline — model should handle this easily |
| 3 | Very reverberant (RT60 2.2s) | Does late reverberation confuse the model? |
| 4 | Dry/direct (RT60 0.2s) | Pure feedback with no masking room tail |
| 5 | Sibilant vocal | S/T/F artifact check — model must not distort high frequencies |
| 6 | Sustained vowel | No pitch-shift or tremolo on held notes |
| 7 | Multi-frequency feedback (gain 0.75/0.80) | Both mains and monitor rings simultaneously |

For each, listen to three files side by side:
1. `clean.wav` — ground truth
2. `mic.wav` — input (before model)
3. `enhanced.wav` — model output

**What to check:**
- [ ] Is feedback energy suppressed without eating the vocal body?
- [ ] Does the model artifact on sibilants? (harshness, distortion on S/T)
- [ ] Does it handle sustained vowels cleanly? (no pitch shifting or tremolo)
- [ ] Does it maintain natural vocal timbre?
- [ ] Is there any "pumping" or gain modulation artifact?
- [ ] Does it hold up at high loop gain (0.8+)?

### 5.3 Real-World Hardware Test Protocol

Once the model sounds good on synthetic test pairs:

**Equipment:**
- Laptop running inference script
- Audio interface (any — this is testing, not production)
- Mic (SM58 or similar — the most common use case)
- PA speaker or monitor + amplifier
- SPL meter app on phone

**Setup:**
- Mic → interface input
- Interface output → powered speaker
- Speaker and mic in the same room, at increasing proximity

**Tests:**

| Test | Pass Condition |
|---|---|
| No mic input (silence) | No artifacts, no self-oscillation |
| Mic active, speaker off | Clean passthrough, no processing artifacts |
| Walk mic toward speaker at low gain | Feedback suppressed before audible squeal |
| Walk mic toward speaker at high gain | Significant GBF improvement vs. bypassed |
| Sustained vowel at moderate gain | No artifacts on sustained "ahhh" |
| Sibilant speech at moderate gain | No harshness on S/T consonants |
| Sustained high-gain near-feedback | Model holds without runaway |
| Bypass comparison | Listener prefers processed signal |

**Measure gain before feedback (GBF):**
1. Bypass model — slowly raise gain until feedback. Note level.
2. Enable model — slowly raise gain until feedback. Note level.
3. GBF improvement = difference in dB.
4. Target: **+6dB or more** improvement. De-Feedback claims ~+10dB.
   Even +4dB is a meaningful real-world improvement.

**Safety check — over-suppression:**
Before any test involving a live PA, verify the model doesn't over-suppress.
An early-stage model can sometimes output near-silence for signals it hasn't seen before.
In a live sound context, unexpected silence is a hard failure in front of an audience.
Test: speak into mic with model active and no feedback risk — confirm output level
is within a few dB of input level. If not, the model needs more training data before
it touches a real system.

---

## Phase 6: Real-Time Inference Script

```python
"""
inference/live.py — Real-time feedback suppression via FDKFNet (v1) or FreqDomainNLMS fallback (v0)

Architecture (v1 — FDKFNet checkpoint present):
  Single integrated model: frequency-domain Kalman filter + GRU estimates per-bin
  covariances Q_k and R_k from mic+ref. Trained with recursive feedback simulation.
  Requires hardware loopback reference (what the PA is playing) → second ADC input.

Architecture (v0 — no checkpoint, fallback):
  FreqDomainNLMS overlap-save FDAF. 1024 taps, O(N log N) per block.
  No training needed. Use this to verify the audio pipeline before training completes.

Usage:
    python inference/live.py --list
    python inference/live.py --input 2 --output 4                # v0 fallback
    python inference/live.py --input 2 --output 4 --ref           # v1: with loopback ref
    python inference/live.py --checkpoint checkpoints/fdkfnet/best.pt

Latency: HOP=480 samples = 10ms callback at 48kHz. FDKFNet is causal (per-frame).
"""

import sys
import sounddevice as sd
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfilt_zi

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'train'))
sys.path.insert(0, str(PROJECT_ROOT / 'simulator'))
from model import FDKFNet
from fdaf import FreqDomainNLMS

SR    = 48000
HOP   = 480      # 10ms at 48kHz
N_FFT = 960      # HOP * 2

WIN_T = torch.hann_window(N_FFT)   # module-level constant, not in callback

TARGET_RMS   = 0.1
rms_smoother = np.array([TARGET_RMS])

# DC blocking filter (~5Hz 1st-order highpass) — removes DC offset from cheap ADCs.
# Applied to mic only; ref is hardware loopback from box output and is already clean.
# Using sosfilt (vectorized C) instead of a Python sample-loop — the loop version costs
# ~1–2ms per 480-sample block on Pi 5 (~10–20% of the real-time budget).
_dc_sos = butter(1, 5.0 / (SR / 2), btype='high', output='sos')
_dc_zi  = sosfilt_zi(_dc_sos) * 0.0   # zero initial conditions

# Console HPF (90Hz 2nd-order Butterworth) — models standard vocal channel HPF on PA consoles.
# Applied to both mic and ref to match training conditions (recursive_train.py applies the same).
_console_hpf     = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
_hpf_zi_mic      = sosfilt_zi(_console_hpf) * 0.0   # zero initial conditions
_hpf_zi_ref      = sosfilt_zi(_console_hpf) * 0.0


def stft_frame(x_np: np.ndarray) -> torch.Tensor:
    """HOP-sample numpy block → complex (1, N_FFT//2+1) tensor."""
    x = torch.from_numpy(x_np).unsqueeze(0)          # (1, HOP)
    x_padded = F.pad(x, (N_FFT - HOP, 0))            # (1, N_FFT)
    return torch.fft.rfft(x_padded * WIN_T, n=N_FFT) # (1, F)


def istft_frame(X: torch.Tensor) -> np.ndarray:
    """Complex (1, F) tensor → HOP-sample numpy block."""
    x = torch.fft.irfft(X, n=N_FFT)   # (1, N_FFT)
    return x[0, -HOP:].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      type=int, default=None)
    parser.add_argument('--output',     type=int, default=None)
    parser.add_argument('--ref',        action='store_true', default=False,
                        help='Enable v1 FDKFNet mode: open 2 input channels (ch0=mic, ch1=loopback ref). '
                             'The ref must be ch1 on the same --input device — not a separate device index.')
    parser.add_argument('--list',       action='store_true')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'checkpoints' / 'fdkfnet' / 'best.pt'))
    args = parser.parse_args()

    print(sd.query_devices())
    if args.list:
        return

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model    = FDKFNet()
        saved    = torch.load(str(ckpt), map_location='cpu')
        model.load_state_dict(saved['model'] if isinstance(saved, dict) and 'model' in saved else saved)
        model.eval()
        H, P, gru_h = model.init_state(batch_size=1, device='cpu')
        use_fdkf = True
        print(f"FDKFNet loaded — SR={SR}Hz, HOP={HOP} ({HOP/SR*1000:.0f}ms/block)")
    else:
        fdaf = FreqDomainNLMS(filter_len=1024, block_size=HOP, mu=0.02)
        use_fdkf = False
        H = P = gru_h = None
        print(f"No FDKFNet checkpoint at {ckpt} — using v0 FreqDomainNLMS fallback")
        print("Run recursive_train.py first to enable the neural model.")

    def callback(indata, outdata, frames, time, status):
        nonlocal H, P, gru_h
        global _hpf_zi_mic, _hpf_zi_ref, _dc_zi

        if status:
            print(status)

        mono = indata[:, 0].copy()

        # DC blocking — removes ADC offset before normalization/filtering
        mono, _dc_zi = sosfilt(_dc_sos, mono, zi=_dc_zi)

        # Running RMS normalization — compute scale BEFORE applying it
        block_rms       = float(np.sqrt(np.mean(mono ** 2))) + 1e-8
        rms_smoother[0] = 0.95 * rms_smoother[0] + 0.05 * block_rms
        rms_scale       = TARGET_RMS / max(rms_smoother[0], 1e-4)
        mono            = mono * rms_scale

        has_ref   = args.ref is not None
        if has_ref:
            ref_block = indata[:, 1].copy()
            # Apply the SAME RMS scale to ref so Kalman H(z) estimate is gain-consistent.
            # Without this, the ratio between mic and ref depends on the input level,
            # which miscalibrates both the H(z) estimate and the VAD gate threshold.
            ref_block = ref_block * rms_scale
            # Sanity check: ref should be non-trivial when mic is active.
            # A silent ref usually means the loopback cable is unplugged or routed wrong.
            ref_rms = float(np.sqrt(np.mean(ref_block ** 2)))
            if ref_rms < 1e-5 and block_rms > 1e-4:
                print("[WARNING] ref channel is silent while mic has signal — "
                      "check loopback cable. FDKFNet will fall back to FDAF behavior.",
                      flush=True)
        else:
            ref_block = np.zeros(HOP, dtype=np.float32)

        # Console HPF — matches recursive_train.py training conditions (90Hz HPF on vocal channel)
        mono,      _hpf_zi_mic = sosfilt(_console_hpf, mono,      zi=_hpf_zi_mic)
        ref_block, _hpf_zi_ref = sosfilt(_console_hpf, ref_block, zi=_hpf_zi_ref)

        if use_fdkf:
            mic_f = stft_frame(mono)
            ref_f = stft_frame(ref_block)
            with torch.no_grad():
                speech_f, H, P, gru_h = model.forward_frame(mic_f, ref_f, H, P, gru_h)
            enhanced = istft_frame(speech_f)
        else:
            enhanced = fdaf.process(mono, ref_block)

        outdata[:, 0] = np.clip(enhanced[:frames], -1.0, 1.0)

    device = (args.input, args.output)
    # When --ref is provided, open 2 input channels: ch0 = mic, ch1 = loopback ref.
    # The loopback ref must be on the same audio interface as the mic (same device index)
    # on a second input channel — not a separate device. For a typical 2-channel interface:
    # ch0 = XLR mic input, ch1 = line input wired to the PA send (loopback).
    # If the ref comes from a different physical interface, a more complex multi-device
    # setup is required (not supported here — use a mixer or patch bay to combine).
    input_channels = 2 if args.ref is not None else 1
    has_ref        = args.ref is not None
    if has_ref:
        print(f"v1 mode: 2-channel input (ch0=mic, ch1=loopback ref) — FDKFNet active")
    else:
        print(f"v0 mode: 1-channel input, no loopback ref — FreqDomainNLMS fallback")
    print(f"Running live inference on devices {device} — Ctrl+C to stop")

    with sd.Stream(samplerate=SR, blocksize=HOP, dtype='float32',
                   channels=(input_channels, 1), device=device, callback=callback):
        while True:
            sd.sleep(1000)


if __name__ == '__main__':
    main()
```

**Run:**
```bash
python inference/live.py --list
python inference/live.py --input 2 --output 4                    # v0: FDAF fallback (no checkpoint needed)
python inference/live.py --input 2 --output 4 --ref              # v1: FDKFNet with hardware loopback
python inference/live.py --checkpoint checkpoints/fdkfnet/best.pt --input 2 --output 4 --ref
```

---

## Phase 7: Iteration Loop

This is the ongoing work after Phase 6. Model quality is a direct function
of how many real acoustic environments are in the training data.

```
Collect real venue IRs (Phase 2 protocol)
        ↓
Add to data/venue_irs/
        ↓
Regenerate training pairs (Phase 3)
        ↓
Fine-tune from last checkpoint (Phase 4)
        ↓
Run evaluation protocol (Phase 5)
        ↓
Real-world hardware test (Phase 5.3)
        ↓
Identify failure modes → note venue type that failed
        ↓
Prioritize measuring that venue type next
        ↓
repeat
```

**Priority venue gaps (not covered by any public IR dataset):**
- [ ] Gymnasium / multi-purpose room (hard parallel walls, flutter echo)
- [ ] Outdoor stage (no reverb, strong direct path only)
- [ ] Small bar / low ceiling club
- [ ] Large sanctuary with PA installed — measure with PA active

**After each new venue session:** run `python simulator/preprocess.py` to resample
new IRs, then regenerate pairs and fine-tune. New IRs go to:
- `data/venue_irs/mains/` — from the mains-only captures
- `data/venue_irs/monitors/` — from the monitors-only captures

---

## Milestone Checklist

- [ ] **M0** — Environment set up, repos cloned. **Phase 0.4 smoke test passes** (pyroomacoustics directivity API, melscale_fbanks shape, FDKFNet forward pass).
- [x] **M0.5** — Architecture finalized: FDKFNet (single integrated model) at 48kHz native.
  Key facts:
  - Model: Frequency-Domain Kalman Filter + GRU (~500K params). Estimates per-bin Q_k/R_k.
  - Training: recursive simulation — model output fed back into simulated acoustic path.
    Eliminates teacher-forcing train/inference mismatch (documented failure mode of all prior work).
  - Reference: NeuralKalmanAHS (arXiv 2309.16049) — +4.94 dB SDR over Hybrid AHS streaming.
  - v0 fallback (before checkpoint exists): FreqDomainNLMS 1024-tap FDAF — no training needed.
  - v1 (with FDKFNet checkpoint): requires hardware loopback reference (second ADC input).
  - N_FFT=960, HOP=480 (10ms/block at 48kHz). Per-frame STFT → forward_frame() → ISTFT.
  - Pi 5 estimated RTF: FDKFNet ~0.10–0.15 (GRU is lighter than DN3 conv stack).
- [ ] **M1** — All public datasets downloaded and organized. Parallel tasks:
  - Speech: EARS + VCTK in `data/clean_vocals/`
  - **Singing: VocalSet (CC BY 4.0) in `data/clean_vocals/`** — required. iKala/MUSDB18 are non-commercial only — do not use in the commercial product build
  - Room IRs in `data/public_irs/`, noise in `data/noise/`
  - Transducer FR CSVs: ≥5 mics and ≥4 speakers in `data/transducer_frs/`
- [ ] **M1.5** — `preprocess.py` run: all directories resampled to 48kHz in-place.
  Verify with `soxi data/clean_vocals/*.wav`. Check that singing files are present:
  `ls data/clean_vocals/ | grep -i vocalset` should return results.
  `build_transducer_library()` smoke test passes.
- [ ] **M2** — Venue sweep GUI working; first real venue measured with mains,
  monitors, and combined captures. IRs saved to correct subdirectories
  (`data/venue_irs/mains/`, `data/venue_irs/monitors/`, `data/venue_irs/combined/`)
- [ ] **M3** — Simulator generating training pairs from synthetic IRs only
  (no real venue IRs yet). Verify output pairs sound reasonable in an audio editor.
- [ ] **M4** — Simulator updated with real venue IRs; mains and monitor paths
  mixing correctly with independent loop gains. Listening check on generated pairs.
- [ ] **M5** — FDKFNet trained on synthetic data; `eval/run_inference.py` + `eval/score.py`
  passing on held-out val set. PESQ > 2.5, STOI > 0.85.
  Listening test on 20 examples (all 7 scenario types).
  **GBF hardware test**: measure gain-before-feedback improvement vs. bypassed — target +3dB minimum.
- [ ] **M6** — Live inference script running on mic input (laptop + USB interface).
  **Total system latency breakdown** (not 10ms):
  - HOP (one frame): 10ms (480 samples)
  - ADC buffer: typically 10–16ms at 48kHz on most USB interfaces (device-dependent)
  - Model compute: ~1–2ms on modern CPU, ~5–10ms on Pi 5
  - DAC output buffer: 1–3ms
  - **Total: ~21–30ms end-to-end** — acceptable for live PA (under 30ms threshold for most performers)
  Confirm latency with clap test (simultaneous input/output record, measure sample offset).
  v0 path: FreqDomainNLMS fallback (no checkpoint needed).
  v1 path: FDKFNet with `--ref` loopback — confirm suppression onset within 1–2s of loop closure.
- [ ] **M7** — Real-world GBF improvement measured (+4dB minimum vs. bypassed).
- [ ] **M8** — 10+ real venues measured across at least 3 venue types. Model
  retrained. GBF improvement holds across venue types.
- [ ] **M8.5** — Target embedded hardware selected and validated. Export FDKFNet to
  TorchScript and benchmark on target device (Pi 5).
  **Do this early** — don't wait for M8.5. Run the Pi 5 benchmark at M5 on whatever
  checkpoint you have. RTF > 1.0 means the model can't run in real time and you need
  to reduce GRU hidden size or N_FFT before investing in more training compute.

  FDKFNet uses standard PyTorch ops — TorchScript export is straightforward:

  ```python
  # Export FDKFNet to TorchScript (run from project root)
  import sys, torch
  sys.path.insert(0, 'train')
  from model import FDKFNet
  model = FDKFNet()
  model.load_state_dict(torch.load('checkpoints/fdkfnet/best.pt', map_location='cpu'))
  model.eval()
  scripted = torch.jit.script(model)
  scripted.save('checkpoints/fdkfnet/fdkfnet_scripted.pt')
  print("Exported fdkfnet_scripted.pt")
  ```
  ```bash
  # On Pi 5: benchmark one HOP block (480 samples = 10ms at 48kHz)
  python -c "
  import sys, torch, time
  sys.path.insert(0, 'train')
  from model import FDKFNet
  model = FDKFNet(); model.eval()
  H, P, h = model.init_state(1, 'cpu')
  mic_f = torch.zeros(1, 481, dtype=torch.cfloat)
  ref_f = torch.zeros(1, 481, dtype=torch.cfloat)
  # warm-up
  for _ in range(10):
      with torch.no_grad(): _, H, P, h = model.forward_frame(mic_f, ref_f, H, P, h)
  t = time.perf_counter()
  for _ in range(200):
      with torch.no_grad(): _, H, P, h = model.forward_frame(mic_f, ref_f, H, P, h)
  print(f'{(time.perf_counter()-t)/200*1000:.2f} ms/frame (budget: 10ms at 48kHz)')
  "
  ```
  Target: < 10ms/frame on Pi 5 (estimated RTF ~0.10–0.15 — GRU is lighter than DN3).
  This must pass before M9.
- [ ] **M9** — Model stable across venue types, inference validated on target
  hardware. Ready to begin embedded software integration.
