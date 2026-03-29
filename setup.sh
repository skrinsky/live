#!/usr/bin/env bash
# setup.sh — create a venv and install all project dependencies
#
# Usage:
#   bash setup.sh            # uses uv (fast) if available, falls back to pip
#   bash setup.sh --pip      # force pip
#
# After running:
#   source .venv/bin/activate
#   python train/recursive_train.py

set -e

PYTHON=${PYTHON:-python3.11}
TORCH_CUDA_URL="https://download.pytorch.org/whl/cu124"

# ── Installer selection ────────────────────────────────────────────────────────
USE_UV=1
if [[ "$1" == "--pip" ]] || ! command -v uv &>/dev/null; then
    USE_UV=0
fi

install_pkg() {
    if [[ $USE_UV -eq 1 ]]; then
        uv pip install "$@"
    else
        pip install "$@"
    fi
}

# ── Create venv ────────────────────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
    echo "Creating .venv with $PYTHON..."
    if [[ $USE_UV -eq 1 ]]; then
        uv venv --python "$PYTHON" .venv
    else
        "$PYTHON" -m venv .venv
    fi
fi

source .venv/bin/activate

# ── PyTorch (CUDA) ─────────────────────────────────────────────────────────────
echo "Installing PyTorch with CUDA support..."
install_pkg torch torchaudio --index-url "$TORCH_CUDA_URL"

# ── Everything else ────────────────────────────────────────────────────────────
echo "Installing project dependencies..."
install_pkg -r requirements.txt

echo ""
echo "Done. Activate with: source .venv/bin/activate"
