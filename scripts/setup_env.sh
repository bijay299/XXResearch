#!/bin/bash
# Build isolated venv for the CUIG ConAbl / UnlearnCanvas pilot.
# Pins follow upstream CUIG env.yaml (conda is unavailable on this host, so venv+pip is used).
set -euo pipefail

PILOT_ROOT="/data/bijaypandey/cuig_pilot"
VENV="${PILOT_ROOT}/venv-cuig"

# Home partition (/) is 100% full; keep every cache and temp dir on /data.
export PIP_CACHE_DIR="/data/bijaypandey/pipcache"
export TMPDIR="/data/bijaypandey/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade "pip==24.2" "setuptools==75.1.0" "wheel==0.44.0"

# Torch stack from the cu121 index (driver 550.144.03 / CUDA 12.4 is backward compatible).
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" "torchvision==0.19.0"

# xformers 0.0.27.post2 is the build matched to torch 2.4.0.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    "xformers==0.0.27.post2"

# Remaining pins taken verbatim from CUIG env.yaml, limited to the ConAbl +
# UnlearnCanvas import closure (jupyter / lm_eval / NudeNet / TF are not on this path).
python -m pip install \
    "numpy==1.26.4" \
    "diffusers==0.30.2" \
    "transformers==4.44.2" \
    "tokenizers==0.19.1" \
    "accelerate==0.34.2" \
    "huggingface-hub==0.32.4" \
    "safetensors==0.4.5" \
    "timm==1.0.15" \
    "pytorch-lightning==2.5.1.post0" \
    "torchmetrics==1.7.3" \
    "lightning-utilities==0.14.3" \
    "pillow==10.4.0" \
    "openpyxl==3.1.5" \
    "et-xmlfile==2.0.0" \
    "tqdm==4.66.5" \
    "ftfy==6.2.3" \
    "regex==2024.11.6" \
    "scipy==1.15.3" \
    "packaging==25.0" \
    "openai==0.28.0" \
    "gdown==5.2.1" \
    "modelcards==0.1.6" \
    "omegaconf==2.1.1"

echo "=== VENV BUILD OK ==="
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tv', torchvision.__version__)"
