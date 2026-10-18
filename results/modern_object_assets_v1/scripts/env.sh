#!/bin/bash
# modern_object_assets_v1 -- project environment.
# / is ~100% full: every cache/scratch path lives on /data.
# Source before any python/uv/accelerate invocation:
#   source /data/bijaypandey/cuig_pilot/modern_object_assets_v1/env.sh
# Idempotent: only sets values.

export MOA_ROOT="/data/bijaypandey/cuig_pilot/modern_object_assets_v1"
export MOA_GEM="$MOA_ROOT/GEM_src"

# ---- scratch / caches on /data ----
export TMPDIR="/data/bijaypandey/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="/data/bijaypandey/pipcache"
export UV_CACHE_DIR="/data/bijaypandey/uvcache"
export UV_PYTHON_INSTALL_DIR="/data/bijaypandey/uvpython"
export UV_PROJECT_ENVIRONMENT="$MOA_ROOT/venv-gem"

# ---- HuggingFace: reuse the already-populated, pristine hub cache READ-ONLY ----
export HF_HUB_CACHE="/data/bijaypandey/cuig_pilot/hf_home/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/data/bijaypandey/cuig_pilot/torch_home"

# ---- no telemetry, no external services ----
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_SILENT="true"
export HF_HUB_DISABLE_TELEMETRY="1"
export DISABLE_TELEMETRY="1"
export DO_NOT_TRACK="1"
export BITSANDBYTES_NOWELCOME="1"
export TOKENIZERS_PARALLELISM="false"

# ---- bounded CPU/BLAS workers (shared host: respect other users) ----
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export OPENBLAS_NUM_THREADS="8"
export NUMEXPR_NUM_THREADS="8"
export VECLIB_MAXIMUM_THREADS="8"

# ---- GEM's own dotenv-driven output dir ----
export OUTPUT_DIR="$MOA_ROOT/outputs"

export PATH="/data/bijaypandey/bin:$PATH"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$TORCH_HOME/hub/checkpoints" "$OUTPUT_DIR" /data/bijaypandey/bin
