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

# Inspect first row to understand structure
first = ds[list(ds.keys())[0]][0]
print("Keys:", list(first.keys()))
print("Audio type:", type(first["audio"]))
if isinstance(first["audio"], dict):
    print("Audio keys:", list(first["audio"].keys()))
else:
    print("Audio value (first 200 chars):", str(first["audio"])[:200])
sys.exit(0)
