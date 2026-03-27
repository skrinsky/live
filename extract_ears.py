"""
Extract WAV files from the EARS HuggingFace parquet download.
Run from the project root: python extract_ears.py
"""
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Run: pip install datasets")

import soundfile as sf
import numpy as np

OUT_DIR = Path("data/clean_vocals/wavs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = Path("data/clean_vocals")

print("Loading dataset from parquet files...")
ds = load_dataset("parquet", data_files={
    "train": str(PARQUET_DIR / "data/train-*.parquet"),
    "test":  str(PARQUET_DIR / "data/test-*.parquet"),
})

total = 0
for split in ds:
    for i, row in enumerate(ds[split]):
        audio = row["audio"]
        arr   = np.array(audio["array"], dtype=np.float32)
        sr    = audio["sampling_rate"]
        out   = OUT_DIR / f"{split}_{i:06d}.wav"
        sf.write(str(out), arr, sr)
        total += 1
        if total % 100 == 0:
            print(f"  {total} files extracted...")

print(f"Done: {total} WAV files written to {OUT_DIR}")
