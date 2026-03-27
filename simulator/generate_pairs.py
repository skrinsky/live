"""
simulator/generate_pairs.py — Training pair generation for FDKFNet

Generates (mic_signal, clean_target, ref_signal) tuples where:
  mic_signal  = reverberant_vocal + mains_feedback + monitor_feedback + noise
  clean       = HPF'd reverberant vocal (matches model's HPF'd training target)
  ref_signal  = reverberant vocal (non-HPF'd; HPF'd in run_inference.py before the model)

Also exports build_room_simulation() — imported by generate_ir_pool.py for IR pool
pre-computation. Both callers use the same high-fidelity room simulation infrastructure.

NOTE: pyroomacoustics is NOT imported at module level.
Import it only inside worker functions to avoid the macOS fork+OpenBLAS deadlock:
  multiprocessing fork copies the parent's memory including any OpenBLAS mutexes.
  If pra was imported in the parent, workers calling pra after fork inherit locked mutexes
  and deadlock silently. Importing inside the worker (after fork) avoids this entirely.
  If workers hang silently, switch to ctx = multiprocessing.get_context('spawn') and
  add if __name__ == '__main__': guard. Windows requires 'spawn' regardless.

NOTE: librosa is NOT imported. All audio must be pre-resampled to SR with preprocess.py.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import multiprocessing
import random
import sys
from scipy.signal import fftconvolve, butter, sosfilt
from tqdm import tqdm

SR           = 48000
PROJECT_ROOT = Path(__file__).parent.parent


# ── Material absorption library ──────────────────────────────────────────────
# Values from ISO 354 / Sabine tables, per octave band 125–4000 Hz.
# Per-band coefficients produce correct spectral decay shape — a single-coefficient
# model is physically wrong (carpet absorbs 2% at 125Hz but 65% at 4kHz).
# Format: ([125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz], scattering)
MATERIALS = {
    'concrete':        ([0.01, 0.02, 0.02, 0.03, 0.04, 0.05], 0.05),
    'brick':           ([0.03, 0.03, 0.03, 0.04, 0.05, 0.07], 0.05),
    'plaster':         ([0.01, 0.02, 0.03, 0.04, 0.05, 0.05], 0.05),
    'wood_floor':      ([0.15, 0.11, 0.10, 0.07, 0.06, 0.07], 0.10),
    'wood_panel':      ([0.28, 0.22, 0.17, 0.09, 0.10, 0.11], 0.10),
    'carpet_thin':     ([0.02, 0.04, 0.08, 0.20, 0.35, 0.40], 0.20),
    'carpet_thick':    ([0.02, 0.06, 0.14, 0.37, 0.60, 0.65], 0.25),
    'curtains_light':  ([0.03, 0.04, 0.11, 0.17, 0.24, 0.35], 0.35),
    'curtains_heavy':  ([0.05, 0.12, 0.35, 0.45, 0.38, 0.36], 0.40),
    'acoustic_panels': ([0.10, 0.25, 0.60, 0.90, 0.95, 0.95], 0.70),
    'audience_seated': ([0.39, 0.57, 0.80, 0.94, 0.92, 0.87], 0.60),
    'glass':           ([0.35, 0.25, 0.18, 0.12, 0.07, 0.04], 0.05),
    'seats_empty':     ([0.44, 0.56, 0.67, 0.74, 0.83, 0.87], 0.40),
    'stage_platform':  ([0.40, 0.30, 0.20, 0.17, 0.15, 0.10], 0.15),
}


# ── Room archetypes ───────────────────────────────────────────────────────────
ROOM_ARCHETYPES = {
    'small_venue_bar': {
        'dims':      ([5, 8, 2.8], [10, 15, 3.5]),
        'floor':     ['wood_floor', 'concrete', 'carpet_thin'],
        'ceiling':   ['plaster', 'acoustic_panels'],
        'walls':     ['brick', 'plaster', 'concrete', 'wood_panel'],
        'stage':     ['stage_platform', 'wood_floor'],
        'max_order': 15,
    },
    'church_sanctuary': {
        'dims':      ([8, 12, 5], [18, 25, 10]),
        'floor':     ['wood_floor', 'concrete', 'carpet_thick'],
        'ceiling':   ['plaster', 'wood_panel'],
        'walls':     ['brick', 'plaster', 'concrete'],
        'stage':     ['stage_platform', 'wood_floor'],
        'max_order': 20,
    },
    'gymnasium': {
        'dims':      ([15, 20, 6], [35, 50, 10]),
        'floor':     ['wood_floor', 'concrete'],
        'ceiling':   ['concrete', 'plaster'],
        'walls':     ['concrete', 'brick', 'glass'],
        'stage':     ['stage_platform'],
        # max_order=18: at 35×50×10m, order=25 yields ~15K image sources (minutes per IR).
        # 18 is sufficient when combined with ray_tracing=True for the late tail.
        'max_order': 18,
    },
    'theater': {
        'dims':      ([10, 15, 6], [25, 35, 12]),
        'floor':     ['carpet_thick', 'seats_empty'],
        'ceiling':   ['acoustic_panels', 'plaster'],
        'walls':     ['acoustic_panels', 'curtains_heavy', 'wood_panel'],
        'stage':     ['stage_platform'],   # audience_seated removed — nonsensical for a stage
        'max_order': 18,
    },
    'rehearsal_room': {
        'dims':      ([3, 4, 2.5], [6, 8, 3.2]),
        'floor':     ['carpet_thick', 'wood_floor'],
        'ceiling':   ['acoustic_panels', 'plaster'],
        'walls':     ['acoustic_panels', 'curtains_heavy', 'plaster'],
        'stage':     ['stage_platform', 'carpet_thick'],
        'max_order': 12,
    },
    'ballroom': {
        'dims':      ([10, 15, 4], [25, 40, 6]),
        'floor':     ['wood_floor', 'carpet_thick'],
        'ceiling':   ['plaster', 'acoustic_panels'],
        'walls':     ['plaster', 'glass', 'curtains_heavy'],
        'stage':     ['stage_platform', 'wood_floor'],
        'max_order': 18,
    },
    'hall': {
        # Multi-purpose hall attached to churches/community centers.
        # Bare parallel walls, zero treatment — common worst-case feedback environment.
        'dims':      ([6, 10, 2.8], [15, 25, 4.0]),
        'floor':     ['concrete', 'carpet_thin', 'wood_floor'],
        'ceiling':   ['plaster', 'concrete'],
        'walls':     ['concrete', 'brick', 'plaster'],
        'stage':     ['stage_platform', 'wood_floor', 'concrete'],
        'max_order': 18,
    },
}

ARCHETYPE_WEIGHTS = {
    'small_venue_bar':  0.20,
    'church_sanctuary': 0.25,
    'hall':             0.20,
    'gymnasium':        0.08,
    'theater':          0.12,
    'rehearsal_room':   0.08,
    'ballroom':         0.07,
}


# ── Speaker and mic constants ─────────────────────────────────────────────────
# These are referenced inside build_room_simulation() which imports pra locally.
# The enum values are stored here as strings and looked up after import.
SPEAKER_CONFIGS = {
    'mains_point_source': {
        'pattern_name':    'CARDIOID',
        'azimuth':         0,
        'elevation_range': (-30, -10),
        'n_sources':       1,
    },
    'mains_left_right': {
        'pattern_name':    'CARDIOID',
        'azimuth':         0,
        'elevation_range': (-25, -10),
        'n_sources':       2,
    },
    'mains_line_array': {
        'pattern_name':    'HYPERCARDIOID',
        'azimuth':         0,
        'elevation_range': (-20, -5),
        'n_sources':       2,
    },
    'monitor_wedge_front': {
        'pattern_name':    'CARDIOID',
        'azimuth':         180,
        'elevation_range': (35, 50),
        'n_sources':       1,
    },
    'monitor_wedge_side': {
        'pattern_name':    'CARDIOID',
        'azimuth':         90,
        'elevation_range': (20, 40),
        'n_sources':       1,
    },
    'monitor_in_ear': {
        'pattern_name':    None,
        'n_sources':       0,
    },
    'subwoofer_ground_stack': {
        'pattern_name':    'OMNI',
        'azimuth':         0,
        'elevation':       0,
        'n_sources':       1,
        'lowpass_hz':      100,
        'gain_range':      (0.1, 0.4),
    },
    'subwoofer_cardioid_array': {
        'pattern_name':    'SUBCARDIOID',
        'azimuth':         0,
        'elevation':       0,
        'n_sources':       1,
        'lowpass_hz':      100,
        'gain_range':      (0.05, 0.25),
    },
}

MIC_PATTERN_WEIGHTS = {
    'CARDIOID':      0.55,
    'HYPERCARDIOID': 0.25,
    'SUBCARDIOID':   0.10,
    'OMNI':          0.10,
}


# ── High-fidelity room simulation (exported for generate_ir_pool.py) ─────────

def build_room_simulation():
    """
    High-fidelity acoustic simulation using pyroomacoustics.
    Samples a room archetype, per-surface materials, mic/speaker configurations,
    and runs a single simulate() call to compute all source→mic IRs.

    Returns:
        mains_ir    : np.ndarray (loudspeaker→mic feedback path, mains)
        monitor_ir  : np.ndarray (wedge→mic feedback path, zeros if IEM)
        room_ir     : np.ndarray (direct vocal reverb — close-mic diffuse path)
        sub_ir      : np.ndarray (sub feedback path, zeros if no sub in this sample)
        meta        : dict (archetype, mic_pattern, configs, dims)
    """
    import pyroomacoustics as pra   # local import — avoids macOS fork+OpenBLAS deadlock
    from pyroomacoustics.directivities import (
        Cardioid, HyperCardioid, SubCardioid, Omnidirectional, DirectionVector
    )
    from scipy.signal import butter, sosfilt

    # pyroomacoustics 0.9+ uses DirectionVector(azimuth, colatitude) for orientation.
    # colatitude = 90 - elevation (measured from zenith, not horizon).
    def _make_dir(pattern_name, azimuth_deg, elevation_deg):
        if pattern_name == 'OMNI':
            return Omnidirectional()
        orientation = DirectionVector(azimuth=azimuth_deg, colatitude=90 - elevation_deg, degrees=True)
        cls = {'CARDIOID': Cardioid, 'HYPERCARDIOID': HyperCardioid, 'SUBCARDIOID': SubCardioid}
        return cls[pattern_name](orientation=orientation)

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
        air_absorption=True, # high-freq rolloff over distance (critical for distant mains)
    )

    # 4. Mic: performer position — mid-to-back stage, standing height
    mic_pos = [
        dims[0] * np.random.uniform(0.3, 0.7),
        dims[1] * np.random.uniform(0.4, 0.65),
        np.random.uniform(1.5, 1.9),
    ]
    mic_pattern_name = random.choices(
        list(MIC_PATTERN_WEIGHTS.keys()),
        weights=list(MIC_PATTERN_WEIGHTS.values())
    )[0]
    mic_dir = _make_dir(mic_pattern_name, np.random.normal(0, 20), np.random.normal(0, 10))
    room.add_microphone(np.array(mic_pos), directivity=mic_dir)

    # 5. Mains speaker — elevated near front of room, aimed at stage
    mains_config_name = random.choices(
        ['mains_point_source', 'mains_left_right', 'mains_line_array'],
        weights=[0.5, 0.35, 0.15]
    )[0]
    mains_cfg   = SPEAKER_CONFIGS[mains_config_name]
    mains_elev  = np.random.uniform(*mains_cfg['elevation_range'])
    mains_dir = _make_dir(mains_cfg['pattern_name'], 180, mains_elev)
    mains_src_pos = [
        dims[0] * 0.5,
        dims[1] * 0.05,
        dims[2] * np.random.uniform(0.5, 0.85),
    ]
    room.add_source(np.array(mains_src_pos), directivity=mains_dir)
    mains_src_idx = len(room.sources) - 1

    # 6. Monitor wedge — downstage floor, angled at performer
    monitor_config_name = random.choices(
        ['monitor_wedge_front', 'monitor_wedge_side', 'monitor_in_ear'],
        weights=[0.65, 0.20, 0.15]
    )[0]
    monitor_cfg = SPEAKER_CONFIGS[monitor_config_name]
    monitor_src_idx = None
    if monitor_cfg['n_sources'] > 0:
        mon_elev = np.random.uniform(*monitor_cfg['elevation_range'])
        mon_dir = _make_dir(monitor_cfg['pattern_name'], monitor_cfg['azimuth'], mon_elev)
        mon_src_pos = [
            mic_pos[0] + np.random.uniform(-0.3, 0.3),
            mic_pos[1] - np.random.uniform(0.5, 1.5),
            np.random.uniform(0.1, 0.4),
        ]
        room.add_source(np.array(mon_src_pos), directivity=mon_dir)
        monitor_src_idx = len(room.sources) - 1

    # 7. Room reverb source — close to mic (direct vocal reflection path)
    room_src_pos = [
        mic_pos[0] + np.random.uniform(0.1, 0.4),
        mic_pos[1] + np.random.uniform(-0.2, 0.2),
        mic_pos[2],
    ]
    room.add_source(np.array(room_src_pos))
    room_src_idx = len(room.sources) - 1

    # 8. Subwoofer — 60% chance in large-room archetypes
    sub_src_idx     = None
    sub_config_name = None
    if archetype_name in ('gymnasium', 'church_sanctuary', 'hall', 'theater', 'ballroom') \
            and random.random() < 0.60:
        sub_config_name = random.choice(['subwoofer_ground_stack', 'subwoofer_cardioid_array'])
        sub_cfg = SPEAKER_CONFIGS[sub_config_name]
        sub_dir = _make_dir(sub_cfg['pattern_name'], 0, 0)
        sub_src_pos = [
            dims[0] * np.random.uniform(0.35, 0.65),
            dims[1] * 0.03,
            np.random.uniform(0.1, 0.5),
        ]
        room.add_source(np.array(sub_src_pos), directivity=sub_dir)
        sub_src_idx = len(room.sources) - 1

    # 9. Single simulate() — all sources at once
    room.compute_rir()

    def _safe_ir(raw):
        ir = raw.astype(np.float32)
        if not np.isfinite(ir).all():
            raise ValueError(
                f"IR contains NaN/Inf — simulation diverged "
                f"(room {dims.tolist()}, max_order={arch['max_order']})"
            )
        if np.max(np.abs(ir)) < 1e-12:
            raise ValueError(
                f"IR is all-zero — silent simulation failure "
                f"(room {dims.tolist()}, source/mic may be coincident)"
            )
        return ir

    mains_ir   = _safe_ir(room.rir[0][mains_src_idx])
    monitor_ir = (
        _safe_ir(room.rir[0][monitor_src_idx])
        if monitor_src_idx is not None
        else np.zeros(512, dtype=np.float32)
    )
    room_ir = _safe_ir(room.rir[0][room_src_idx])

    sub_ir = np.zeros(512, dtype=np.float32)
    if sub_src_idx is not None:
        raw_sub = room.rir[0][sub_src_idx].astype(np.float32)
        sos     = butter(4, sub_cfg['lowpass_hz'] / (SR / 2), btype='low', output='sos')
        sub_ir  = sosfilt(sos, raw_sub).astype(np.float32)

    # 10. Non-convex room override (~20% mains, ~10% monitor)
    # Approximates L-shaped rooms and multi-space coupling via cascaded shoebox convolution.
    if random.random() < 0.20:
        mains_ir = non_convex_room_ir(near_field=False)
    if monitor_src_idx is not None and random.random() < 0.10:
        monitor_ir = non_convex_room_ir(near_field=True)

    meta = {
        'archetype':      archetype_name,
        'mic_pattern':    mic_pattern_name,
        'mains_config':   mains_config_name,
        'monitor_config': monitor_config_name,
        'sub_config':     sub_config_name,
        'dims':           dims.tolist(),
    }
    return mains_ir, monitor_ir, room_ir, sub_ir, meta


def non_convex_room_ir(near_field=False):
    """
    Approximate non-convex room IR via cascaded shoebox simulation.

    Models L-shaped rooms, rooms with alcoves, stage+auditorium coupling.
    The primary IR = main performance space; secondary IR = connected sub-space
    (side chapel, back corridor, balcony). Cascading approximates sound traveling
    through the acoustic junction.

    Do NOT normalize the output — junction_atten encodes physically realistic
    path loss (6–14dB) that the Kalman filter uses for covariance calibration.
    """
    import pyroomacoustics as pra   # local import — avoids macOS fork+OpenBLAS deadlock

    if near_field:
        primary_dims = np.random.uniform([4, 6, 2.8], [8, 12, 4.0])
    else:
        primary_dims = np.random.uniform([8, 12, 4], [20, 30, 10])

    primary_mat      = random.choice(['plaster', 'brick', 'wood_panel', 'concrete'])
    p_coeffs, p_scat = MATERIALS[primary_mat]
    primary_room     = pra.ShoeBox(
        primary_dims, fs=SR,
        materials=pra.Material(
            energy_absorption={"coeffs": p_coeffs,
                               "center_freqs": [125, 250, 500, 1000, 2000, 4000]},
            scattering=p_scat
        ),
        max_order=15, ray_tracing=True, air_absorption=True,
    )
    if near_field:
        src_pos = [primary_dims[0]*0.2, primary_dims[1]*0.1, primary_dims[2]*0.3]
        mic_pos = [primary_dims[0]*0.5, primary_dims[1]*0.5, 1.7]
    else:
        src_pos = [primary_dims[0]*0.5, primary_dims[1]*0.05, primary_dims[2]*0.7]
        mic_pos = [primary_dims[0]*0.5, primary_dims[1]*0.55, 1.7]
    primary_room.add_source(src_pos)
    primary_room.add_microphone(mic_pos)
    primary_room.compute_rir()
    primary_ir = primary_room.rir[0][0].astype(np.float32)

    # Secondary space — 30–60% of primary volume, independent ceiling height
    secondary_scale = np.random.uniform(0.3, 0.6)
    secondary_dims  = primary_dims * secondary_scale
    secondary_dims[2] = np.random.uniform(2.5, primary_dims[2])

    secondary_mat      = random.choice(list(MATERIALS.keys()))
    s_coeffs, s_scat   = MATERIALS[secondary_mat]
    secondary_room     = pra.ShoeBox(
        secondary_dims, fs=SR,
        materials=pra.Material(
            energy_absorption={"coeffs": s_coeffs,
                               "center_freqs": [125, 250, 500, 1000, 2000, 4000]},
            scattering=s_scat
        ),
        max_order=12, ray_tracing=True, air_absorption=True,
    )
    junc_src = [secondary_dims[0]*0.1, secondary_dims[1]*0.5, secondary_dims[2]*0.5]
    junc_mic = [secondary_dims[0]*0.8, secondary_dims[1]*0.5, secondary_dims[2]*0.5]
    secondary_room.add_source(junc_src)
    secondary_room.add_microphone(junc_mic)
    secondary_room.compute_rir()
    secondary_ir = secondary_room.rir[0][0].astype(np.float32)

    # Cascade: junction attenuation of 6–14dB (opening between spaces)
    junction_atten = np.random.uniform(0.2, 0.6)
    cascaded_ir    = fftconvolve(primary_ir, secondary_ir * junction_atten)
    return cascaded_ir.astype(np.float32)


# ── Simple synthetic IR (legacy fallback in generate_pair) ───────────────────

def synthetic_feedback_ir(near_field=False):
    """
    Simple single-room loudspeaker→mic IR (no directivity, single absorption coeff).
    Used as fallback inside generate_pair() when no real venue IRs are available.
    For the IR pool, use build_room_simulation() instead — it is much higher fidelity.
    """
    import pyroomacoustics as pra   # local import — avoids macOS fork+OpenBLAS deadlock

    if near_field:
        room_dims   = np.random.uniform([3, 3, 2.5], [6, 5, 3.5])
        speaker_pos = [
            np.random.uniform(0.3, 1.0),
            np.random.uniform(0.3, 0.8),
            np.random.uniform(0.3, 0.8),
        ]
    else:
        room_dims   = np.random.uniform([6, 6, 3], [15, 12, 6])
        speaker_pos = [
            np.random.uniform(0.5, room_dims[0] - 0.5),
            np.random.uniform(0.3, 1.5),
            np.random.uniform(2.5, room_dims[2] - 0.5),
        ]

    mic_pos = [
        np.random.uniform(1.0, room_dims[0] - 1.0),
        np.random.uniform(room_dims[1] * 0.4, room_dims[1] * 0.8),
        np.random.uniform(1.5, 1.9),
    ]
    room = pra.ShoeBox(
        room_dims, fs=SR,
        materials=pra.Material(np.random.uniform(0.1, 0.5)),
        max_order=10
    )
    room.add_source(speaker_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()
    return room.rir[0][0].astype(np.float32)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_ir(path):
    """Load a single IR file. All files must already be at SR."""
    ir, sr = sf.read(path, dtype='float32')
    assert sr == SR, f"IR {path} is {sr}Hz — run preprocess.py first"
    if ir.ndim > 1:
        ir = ir[:, 0]
    return ir.astype(np.float32)


# ── Worker function ───────────────────────────────────────────────────────────

def generate_pair(args):
    vocal_path, mains_irs, monitor_irs, room_irs, noise_files, output_dir, idx = args
    try:
        vocal, vocal_sr = sf.read(vocal_path, dtype='float32')
        assert vocal_sr == SR, f"{vocal_path} is {vocal_sr}Hz — run preprocess.py first"
        if vocal.ndim > 1:
            vocal = vocal[:, 0]

        target_len = 3 * SR
        start  = random.randint(0, len(vocal) - target_len)
        vocal  = vocal[start:start + target_len].astype(np.float32)

        # ── Mains feedback path ───────────────────────────────────────────────
        # 60% real measured, 40% synthetic when real available
        if mains_irs and random.random() < 0.6:
            mains_ir = load_ir(random.choice(mains_irs))
        else:
            mains_ir = synthetic_feedback_ir(near_field=False)
        mains_gain = np.random.uniform(0.2, 0.75)
        mains_fb   = fftconvolve(vocal, mains_ir)[:target_len] * mains_gain

        # ── Monitor feedback path ─────────────────────────────────────────────
        if monitor_irs and random.random() < 0.6:
            monitor_ir = load_ir(random.choice(monitor_irs))
        else:
            monitor_ir = synthetic_feedback_ir(near_field=True)
        monitor_gain = np.random.uniform(0.3, 0.85)
        monitor_fb   = fftconvolve(vocal, monitor_ir)[:target_len] * monitor_gain

        # Randomly drop one feedback path (IEM / no-monitor scenario)
        if random.random() < 0.2:
            mains_fb   = np.zeros(target_len, dtype=np.float32)
        if random.random() < 0.2:
            monitor_fb = np.zeros(target_len, dtype=np.float32)

        # ── Room reverb on direct vocal ───────────────────────────────────────
        if room_irs:
            room_ir = load_ir(random.choice(room_irs))
        else:
            # Synthetic close-mic room IR fallback
            room_dims  = np.random.uniform([4, 4, 3], [12, 10, 5])
            absorption = np.random.uniform(0.3, 0.7)
            source_pos = [room_dims[0]/2, room_dims[1]/2, 1.6]
            mic_pos    = [
                room_dims[0]/2 + np.random.uniform(0.1, 0.5),
                room_dims[1]/2, 1.6
            ]
            import pyroomacoustics as pra  # local import — avoids macOS fork+OpenBLAS deadlock
            room_obj = pra.ShoeBox(
                room_dims, fs=SR,
                materials=pra.Material(absorption), max_order=8
            )
            room_obj.add_source(source_pos)
            room_obj.add_microphone(mic_pos)
            room_obj.simulate()
            room_ir = room_obj.rir[0][0].astype(np.float32)

        reverberant_vocal = fftconvolve(vocal, room_ir)[:target_len]

        # ── Noise ─────────────────────────────────────────────────────────────
        noise, noise_sr = sf.read(random.choice(noise_files), dtype='float32')
        assert noise_sr == SR, "Noise file not at 48kHz — run preprocess.py first"
        if noise.ndim > 1:
            noise = noise[:, 0]
        if len(noise) < target_len:
            noise = np.tile(noise, (target_len // len(noise)) + 1)
        noise = noise[random.randint(0, len(noise) - target_len):][:target_len]

        snr_db       = np.random.uniform(5, 40)
        vocal_rms    = np.sqrt(np.mean(reverberant_vocal ** 2)) + 1e-8
        noise_rms    = np.sqrt(np.mean(noise ** 2)) + 1e-8
        noise_scaled = noise * (vocal_rms / noise_rms) * (10 ** (-snr_db / 20))

        # ── Combine ───────────────────────────────────────────────────────────
        mic_signal = reverberant_vocal + mains_fb + monitor_fb + noise_scaled
        ref_signal = reverberant_vocal   # direct signal sent to PA — non-HPF'd

        # HPF the reverberant vocal for the evaluation target.
        # The model is trained to output HPF'd signal (reverb_np in recursive_train.py
        # is HPF'd). Comparing a non-HPF'd reference against HPF'd model output in
        # score.py would penalize the model for sub-90Hz content it correctly removed.
        # ref_*.wav stays non-HPF'd: run_inference.py HPFs it before the model,
        # matching the HPF'd teacher-forcing ref used in training.
        _clean_hpf            = butter(2, 90.0 / (SR / 2), btype='high', output='sos')
        reverberant_vocal_hpf = sosfilt(_clean_hpf, reverberant_vocal).astype(np.float32)

        peak       = max(np.max(np.abs(mic_signal)), np.max(np.abs(reverberant_vocal_hpf))) + 1e-8
        scale      = 0.707 / peak
        mic_signal = (mic_signal          * scale).astype(np.float32)
        clean      = (reverberant_vocal_hpf * scale).astype(np.float32)
        ref_signal = (ref_signal            * scale).astype(np.float32)

        out = Path(output_dir)
        sf.write(str(out / f'mic_{idx:06d}.wav'),   mic_signal, SR, subtype='PCM_16')
        sf.write(str(out / f'clean_{idx:06d}.wav'), clean,      SR, subtype='PCM_16')
        sf.write(str(out / f'ref_{idx:06d}.wav'),   ref_signal, SR, subtype='PCM_16')

        return None   # success

    except Exception as e:
        print(f"Worker error on idx {idx}: {e}", flush=True)
        return f"idx {idx}: {e}"


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n_pairs=50000, n_workers=8, val_pairs=1000):
    data = PROJECT_ROOT / 'data'
    vocal_files = list((data / 'clean_vocals').rglob('*.wav'))
    mains_irs   = list((data / 'venue_irs' / 'mains').rglob('*.wav'))
    monitor_irs = list((data / 'venue_irs' / 'monitors').rglob('*.wav'))
    room_irs    = list((data / 'public_irs').rglob('*.wav'))
    noise_files = list((data / 'noise').rglob('*.wav'))

    if not vocal_files:
        sys.exit("ERROR: No vocal files in data/clean_vocals/ — download EARS/VCTK first.")
    if not noise_files:
        sys.exit("ERROR: No noise files in data/noise/ — download DNS noise set first.")
    if not mains_irs and not monitor_irs:
        print("WARNING: No real venue IRs — using synthetic only. "
              "Run venue_sweep.py at a venue to collect real IRs.")

    vocal_files = [
        f for f in vocal_files
        if (info := sf.info(str(f))).frames / info.samplerate >= 3.0
    ]
    if not vocal_files:
        sys.exit("ERROR: No vocal files >= 3 seconds found.")

    train_dir = data / 'training_pairs' / 'train'
    val_dir   = data / 'training_pairs' / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total = n_pairs + val_pairs
    print(f"Generating {n_pairs} train + {val_pairs} val pairs across {n_workers} workers...")
    print(f"  Vocals:      {len(vocal_files)} files >= 3s")
    print(f"  Mains IRs:   {len(mains_irs)} real + synthetic fallback")
    print(f"  Monitor IRs: {len(monitor_irs)} real + synthetic fallback")
    print(f"  Room IRs:    {len(room_irs)} files")
    print(f"  Noise:       {len(noise_files)} files")

    args = [
        (random.choice(vocal_files), mains_irs, monitor_irs,
         room_irs, noise_files,
         str(train_dir if i < n_pairs else val_dir), i)
        for i in range(total)
    ]

    ctx = multiprocessing.get_context('fork')
    results = []
    with ctx.Pool(n_workers) as p:
        for result in tqdm(p.imap_unordered(generate_pair, args, chunksize=50),
                           total=total, desc='generating pairs'):
            results.append(result)

    error_count = sum(1 for r in results if r is not None)
    error_rate  = error_count / total
    print(f"Done — {total - error_count}/{total} succeeded, "
          f"{error_count} errors ({error_rate:.1%})")
    if error_rate > 0.05:
        sys.exit(
            f"ERROR: {error_rate:.1%} failure rate exceeds 5% threshold — "
            f"check worker errors above. Likely cause: pra simulation produced "
            f"NaN or zero-length IR."
        )


if __name__ == '__main__':
    generate_dataset(n_pairs=50000, val_pairs=1000)
