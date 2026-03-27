"""
venue_sweep.py — Venue IR Measurement Tool
GUI for capturing PA loudspeaker → mic impulse responses for training data.

Requirements:
    pip install sounddevice soundfile numpy scipy
    (tkinter is included with Python)

Usage:
    python measurement/venue_sweep.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sounddevice as sd
import soundfile as sf
import numpy as np
import csv
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from scipy.signal import fftconvolve

# ─── Constants ────────────────────────────────────────────────────────────────

SR            = 48000
F_START       = 20
F_END         = 20000
SILENCE_PAD   = 1.0
IR_KEEP_S     = 3.0      # seconds of IR to keep after peak — 3s handles cathedrals (RT60 up to ~4s)
COUNTDOWN_S   = 3        # seconds to count down before each sweep

# Paths are anchored to the project root (two levels up from this file)
PROJECT_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR    = PROJECT_ROOT / 'data' / 'venue_irs'
METADATA_FILE = OUTPUT_DIR / 'metadata.csv'

METADATA_FIELDS = [
    'filename', 'date', 'venue_name', 'venue_type', 'city',
    'mic_position', 'mic_distance_ft', 'source_config',
    'mic_type', 'speaker_type', 'monitor_type',
    'ceiling_height_ft', 'room_dims_estimate', 'floor_material',
    'pa_system', 'rt60_s', 'input_device', 'output_device',
    'sweep_duration_s', 'n_averages', 'ir_length_s', 'notes'
]

VENUE_TYPES    = ['church', 'gymnasium', 'theater', 'bar_club',
                  'outdoor', 'auditorium', 'multipurpose', 'other']
SOURCE_CONFIGS = ['mains', 'monitors', 'combined']
MIC_POSITIONS  = ['cs_6ft', 'cs_10ft', 'cs_front', 'sl_6ft', 'sr_6ft', 'other']
FLOOR_TYPES    = ['carpet', 'hardwood', 'concrete', 'tile', 'mixed', 'other']

# ─── CSV lock (safe for future multi-thread use) ──────────────────────────────

_csv_lock = threading.Lock()


# ─── DSP ──────────────────────────────────────────────────────────────────────

def generate_log_sweep(duration):
    """Generate a log sine sweep and its matched inverse filter."""
    t = np.linspace(0, duration, int(duration * SR))
    sweep = np.sin(
        2 * np.pi * F_START * duration / np.log(F_END / F_START) *
        (np.exp(t / duration * np.log(F_END / F_START)) - 1)
    )
    k = np.exp(t / duration * np.log(F_END / F_START))
    inverse = sweep[::-1] / k
    return sweep.astype(np.float32), inverse.astype(np.float32)


def run_sweep(sweep, inverse_filter, input_device, output_device):
    """
    Play one sweep through the PA and return the deconvolved IR.

    Args:
        input_device:  0-based sounddevice device index for the measurement mic
        output_device: 0-based sounddevice device index for the PA output
    """
    pad      = np.zeros(int(SILENCE_PAD * SR), dtype=np.float32)
    playback = np.concatenate([pad, sweep, pad])

    # Use device= tuple and channels=1 — avoids confusing device index
    # with channel-within-device number.
    recording = sd.playrec(
        playback.reshape(-1, 1),
        samplerate=SR,
        device=(input_device, output_device),
        channels=1,
        dtype='float32'
    )
    sd.wait()

    rec = recording.flatten()[len(pad):]

    # Deconvolve: convolve recording with inverse filter → IR
    ir = fftconvolve(rec, inverse_filter)

    # Robust peak detection: find first sample above 10% of max
    # (avoids false peak from noise spike in quiet preamble)
    threshold  = 0.1 * np.max(np.abs(ir))
    candidates = np.where(np.abs(ir) >= threshold)[0]
    peak       = candidates[0] if len(candidates) > 0 else np.argmax(np.abs(ir))

    # Keep 1ms pre-peak + IR_KEEP_S post-peak (2s handles RT60 up to ~3s)
    pre  = int(0.001 * SR)
    post = int(IR_KEEP_S * SR)
    ir   = ir[max(0, peak - pre): peak + post]

    return ir.astype(np.float32)


def average_irs(irs):
    min_len  = min(len(ir) for ir in irs)
    averaged = np.mean([ir[:min_len] for ir in irs], axis=0)
    # Do NOT normalize — preserve physical amplitude.
    # The Kalman filter calibrates its covariance estimates based on absolute IR gain;
    # normalizing to peak=1.0 would make every feedback path look like full direct gain.
    return averaged.astype(np.float32)


# ─── Metadata ─────────────────────────────────────────────────────────────────

def log_metadata(row):
    with _csv_lock:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_header = not METADATA_FILE.exists()
        with open(METADATA_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def build_filename(venue_type, venue_name, mic_position, source_config):
    date_str   = datetime.now().strftime('%Y%m%d_%H%M')
    venue_slug = venue_name.lower().replace(' ', '_').replace("'", '')[:20]
    return f"{venue_type}_{venue_slug}_{mic_position}_{source_config}_{date_str}.wav"


# ─── GUI ──────────────────────────────────────────────────────────────────────

class VenueSweepApp:
    def __init__(self, root):
        self.root      = root
        self.root.title("Venue IR Measurement")
        self.root.resizable(False, False)

        self.measuring  = False
        self._stop_flag = False
        self.gui_queue  = queue.Queue()   # thread-safe GUI update queue

        self._build_ui()
        self._refresh_devices()
        self._process_gui_queue()         # start polling the queue on main thread

        # Handle window close gracefully — don't kill mid-measurement
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # ── Header ──
        hdr = tk.Frame(self.root, bg='#1a1a2e')
        hdr.grid(row=0, column=0, columnspan=2, sticky='ew')
        tk.Label(hdr, text="Venue IR Measurement",
                 bg='#1a1a2e', fg='white',
                 font=('Helvetica', 16, 'bold'),
                 pady=12).pack()

        # ── Left column: Venue info ──
        left = ttk.LabelFrame(self.root, text="Venue Info", padding=8)
        left.grid(row=1, column=0, sticky='nsew', padx=10, pady=8)

        self.venue_name  = self._field(left,    "Venue Name", 0)
        self.venue_type  = self._dropdown(left, "Venue Type", VENUE_TYPES, 1)
        self.city        = self._field(left,    "City, State", 2)
        self.ceiling_ht  = self._field(left,    "Ceiling Height (ft)", 3, default='unknown')
        self.room_dims   = self._field(left,    "Room Dims (e.g. 40x60ft)", 4, default='unknown')
        self.floor_mat   = self._dropdown(left, "Floor Material", FLOOR_TYPES, 5)
        self.pa_system   = self._field(left,    "Console / PA System", 6)
        self.rt60        = self._field(left,    "RT60 (s, from REW)", 7, default='unknown')

        # ── Right column: Measurement config ──
        right = ttk.LabelFrame(self.root, text="Measurement Config", padding=8)
        right.grid(row=1, column=1, sticky='nsew', padx=10, pady=8)

        self.mic_type      = self._field(right,    "Mic Type (e.g. SM58)", 0)
        self.speaker_type  = self._field(right,    "Main Speaker Model", 1)
        self.monitor_type  = self._field(right,    "Monitor Type", 2, default='none')
        self.mic_position  = self._dropdown(right, "Mic Position", MIC_POSITIONS, 3)
        self.mic_dist      = self._field(right,    "Mic Distance to Speaker (ft)", 4, default='6')
        self.source_config = self._dropdown(right, "Source Config", SOURCE_CONFIGS, 5)
        self.sweep_dur     = self._spinbox(right,  "Sweep Duration (s)", 6, from_=5, to=30, default=15)
        self.n_averages    = self._spinbox(right,  "Averages", 7, from_=1, to=5, default=3)

        # ── Audio Devices ──
        dev = ttk.LabelFrame(self.root, text="Audio Devices", padding=8)
        dev.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=4)

        ttk.Label(dev, text="Input (mic):").grid(row=0, column=0, sticky='w', **pad)
        self.input_dev = ttk.Combobox(dev, width=50, state='readonly')
        self.input_dev.grid(row=0, column=1, sticky='ew', **pad)

        ttk.Label(dev, text="Output (PA):").grid(row=1, column=0, sticky='w', **pad)
        self.output_dev = ttk.Combobox(dev, width=50, state='readonly')
        self.output_dev.grid(row=1, column=1, sticky='ew', **pad)

        ttk.Button(dev, text="Refresh",
                   command=self._refresh_devices).grid(row=0, column=2, rowspan=2, padx=8)

        # ── Notes ──
        notes_frame = ttk.LabelFrame(self.root, text="Notes", padding=8)
        notes_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=4)
        self.notes = tk.Text(notes_frame, height=2, width=72, font=('Helvetica', 11))
        self.notes.grid(row=0, column=0, sticky='ew')

        # ── Progress ──
        prog_frame = ttk.LabelFrame(self.root, text="Progress", padding=8)
        prog_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=4)

        self.status_label = ttk.Label(prog_frame, text="Ready.",
                                      font=('Helvetica', 11, 'bold'))
        self.status_label.grid(row=0, column=0, sticky='w', padx=4, pady=2)

        self.progress = ttk.Progressbar(prog_frame, length=560, mode='determinate')
        self.progress.grid(row=1, column=0, sticky='ew', padx=4, pady=4)

        self.log = scrolledtext.ScrolledText(prog_frame, height=7, width=72,
                                              font=('Courier', 10),
                                              state='disabled', bg='#f5f5f5')
        self.log.grid(row=2, column=0, sticky='ew', padx=4, pady=4)

        # ── Buttons ──
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="▶  Start Measurement",
                                    command=self._start, width=24)
        self.start_btn.grid(row=0, column=0, padx=8)

        self.reset_btn = ttk.Button(btn_frame, text="↺  Next Position",
                                    command=self._reset, width=24)
        self.reset_btn.grid(row=0, column=1, padx=8)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _field(self, parent, label, row, default=''):
        ttk.Label(parent, text=label + ':').grid(row=row, column=0,
                                                  sticky='w', padx=6, pady=3)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=28).grid(row=row, column=1,
                                                            sticky='ew', padx=6, pady=3)
        return var

    def _dropdown(self, parent, label, options, row):
        ttk.Label(parent, text=label + ':').grid(row=row, column=0,
                                                  sticky='w', padx=6, pady=3)
        var = tk.StringVar(value=options[0])
        ttk.Combobox(parent, textvariable=var, values=options,
                     state='readonly', width=26).grid(row=row, column=1,
                                                       sticky='ew', padx=6, pady=3)
        return var

    def _spinbox(self, parent, label, row, from_, to, default):
        ttk.Label(parent, text=label + ':').grid(row=row, column=0,
                                                  sticky='w', padx=6, pady=3)
        var = tk.IntVar(value=default)
        ttk.Spinbox(parent, textvariable=var, from_=from_, to=to,
                    width=6).grid(row=row, column=1, sticky='w', padx=6, pady=3)
        return var

    def _refresh_devices(self):
        devices = sd.query_devices()
        # Store list so we can recover the 0-based index from selection
        self._device_list = list(devices)
        names = [
            f"{i}: {d['name']}  (in:{d['max_input_channels']}  out:{d['max_output_channels']})"
            for i, d in enumerate(devices)
        ]
        self.input_dev['values']  = names
        self.output_dev['values'] = names

        default_in, default_out = sd.default.device
        if default_in  is not None and 0 <= default_in  < len(names):
            self.input_dev.current(default_in)
        if default_out is not None and 0 <= default_out < len(names):
            self.output_dev.current(default_out)

    def _get_device_index(self, combobox):
        """Extract the 0-based device index from the combobox selection string."""
        return int(combobox.get().split(':')[0])

    # ── Thread-safe GUI updates ────────────────────────────────────────────────

    def _process_gui_queue(self):
        """Drain the GUI update queue on the main thread. Called every 100ms."""
        try:
            while True:
                cmd, data = self.gui_queue.get_nowait()
                if cmd == 'log':
                    self.log.config(state='normal')
                    self.log.insert(tk.END, data + '\n')
                    self.log.see(tk.END)
                    self.log.config(state='disabled')
                elif cmd == 'status':
                    self.status_label.config(text=data)
                elif cmd == 'progress':
                    self.progress['value'] = data
                elif cmd == 'btn_enable':
                    self.start_btn.config(state='normal' if data else 'disabled')
                elif cmd == 'done':
                    messagebox.showinfo("Measurement Complete",
                                        f"IR saved:\n{data}\n\n"
                                        f"Click 'Next Position' to run another.")
                elif cmd == 'error':
                    messagebox.showerror("Measurement Error", data)
        except queue.Empty:
            pass
        self.root.after(100, self._process_gui_queue)

    def _log(self, msg):
        self.gui_queue.put(('log', msg))

    def _set_status(self, msg):
        self.gui_queue.put(('status', msg))

    def _set_progress(self, pct):
        self.gui_queue.put(('progress', pct))

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        checks = [
            (self.venue_name.get().strip(),   "Venue Name"),
            (self.city.get().strip(),          "City"),
            (self.mic_type.get().strip(),      "Mic Type"),
            (self.speaker_type.get().strip(),  "Speaker Type"),
            (self.pa_system.get().strip(),     "PA / Console System"),
        ]
        for value, label in checks:
            if not value:
                messagebox.showwarning("Missing Field", f"Please enter: {label}")
                return False
        if not self.input_dev.get():
            messagebox.showwarning("Missing Device", "Please select an input device.")
            return False
        if not self.output_dev.get():
            messagebox.showwarning("Missing Device", "Please select an output device.")
            return False

        # Verify selected devices actually have the required channels
        in_idx  = self._get_device_index(self.input_dev)
        out_idx = self._get_device_index(self.output_dev)
        in_dev  = self._device_list[in_idx]
        out_dev = self._device_list[out_idx]

        if in_dev['max_input_channels'] < 1:
            messagebox.showerror("Device Error",
                                 f"Selected input device '{in_dev['name']}' has no input channels.")
            return False
        if out_dev['max_output_channels'] < 1:
            messagebox.showerror("Device Error",
                                 f"Selected output device '{out_dev['name']}' has no output channels.")
            return False
        return True

    # ── Measurement ───────────────────────────────────────────────────────────

    def _start(self):
        if self.measuring:
            return
        if not self._validate():
            return
        self.measuring = True
        self.gui_queue.put(('btn_enable', False))
        threading.Thread(target=self._measure_thread, daemon=True).start()

    def _measure_thread(self):
        try:
            n_avg      = self.n_averages.get()
            duration   = self.sweep_dur.get()
            in_idx     = self._get_device_index(self.input_dev)
            out_idx    = self._get_device_index(self.output_dev)

            self._log(f"Generating {duration}s log sweep ({F_START}–{F_END}Hz)...")
            sweep, inverse = generate_log_sweep(duration)

            irs = []
            for i in range(n_avg):
                # Countdown before each sweep
                for t in range(COUNTDOWN_S, 0, -1):
                    if self._stop_flag:
                        return
                    self._set_status(
                        f"Sweep {i+1}/{n_avg} — ensure room is quiet — starting in {t}s..."
                    )
                    time.sleep(1)

                self._set_status(f"Sweep {i+1}/{n_avg} — recording...")
                self._log(f"  Sweep {i+1}/{n_avg}: recording {duration}s...")

                try:
                    ir = run_sweep(sweep, inverse, in_idx, out_idx)
                except sd.PortAudioError as e:
                    raise RuntimeError(
                        f"Audio device error on sweep {i+1}: {e}\n\n"
                        "Check that your interface is connected and the correct "
                        "devices are selected."
                    )

                irs.append(ir)
                self._set_progress(int((i + 1) / n_avg * 80))
                self._log(f"  Sweep {i+1}/{n_avg}: done — IR length {len(ir)/SR:.3f}s")

            self._set_status("Processing — averaging sweeps...")
            self._log("Averaging sweeps...")
            averaged = average_irs(irs)
            ir_length = len(averaged) / SR

            self._set_status("Saving...")
            filename = build_filename(
                self.venue_type.get(),
                self.venue_name.get(),
                self.mic_position.get(),
                self.source_config.get()
            )
            # Route to mains/, monitors/, or combined/ so the simulator can
            # load the correct feedback path independently
            subdir = OUTPUT_DIR / self.source_config.get()
            subdir.mkdir(parents=True, exist_ok=True)
            out_path = subdir / filename
            sf.write(str(out_path), averaged, SR, subtype='PCM_24')
            self._log(f"Saved: {out_path}")
            self._log(f"IR length: {ir_length:.3f}s")

            self._set_status("Logging metadata...")
            log_metadata({
                'filename':           filename,
                'date':               datetime.now().strftime('%Y-%m-%d %H:%M'),
                'venue_name':         self.venue_name.get(),
                'venue_type':         self.venue_type.get(),
                'city':               self.city.get(),
                'mic_position':       self.mic_position.get(),
                'mic_distance_ft':    self.mic_dist.get(),
                'source_config':      self.source_config.get(),
                'mic_type':           self.mic_type.get(),
                'speaker_type':       self.speaker_type.get(),
                'monitor_type':       self.monitor_type.get(),
                'ceiling_height_ft':  self.ceiling_ht.get(),
                'room_dims_estimate': self.room_dims.get(),
                'floor_material':     self.floor_mat.get(),
                'pa_system':          self.pa_system.get(),
                'rt60_s':             self.rt60.get(),
                'input_device':       self._device_list[in_idx]['name'],
                'output_device':      self._device_list[out_idx]['name'],
                'sweep_duration_s':   duration,
                'n_averages':         n_avg,
                'ir_length_s':        f"{ir_length:.3f}",
                'notes':              self.notes.get('1.0', tk.END).strip()
            })
            self._log(f"Metadata logged to {METADATA_FILE}")

            self._set_progress(100)
            self._set_status(f"Complete — {filename}")
            self.gui_queue.put(('done', filename))

        except Exception as e:
            self._log(f"ERROR: {e}")
            self._set_status("Error — see log.")
            self.gui_queue.put(('error', str(e)))

        finally:
            self.measuring = False
            self.gui_queue.put(('btn_enable', True))

    # ── Window close ──────────────────────────────────────────────────────────

    def _on_close(self):
        if self.measuring:
            if not messagebox.askyesno(
                "Measurement in progress",
                "A measurement is currently running.\n\n"
                "Closing now may lose the current IR and corrupt metadata.\n\n"
                "Close anyway?"
            ):
                return
        self._stop_flag = True
        self.root.destroy()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset(self):
        """Reset position/config fields for next measurement.
        Venue-level info (name, type, city, room, PA, mic type, speaker type) is kept."""
        self.mic_position.set(MIC_POSITIONS[0])
        self.mic_dist.set('6')
        self.source_config.set(SOURCE_CONFIGS[0])
        self.notes.delete('1.0', tk.END)
        self.progress['value'] = 0
        self.log.config(state='normal')
        self.log.delete('1.0', tk.END)
        self.log.config(state='disabled')
        self._set_status("Ready.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app  = VenueSweepApp(root)
    root.mainloop()
