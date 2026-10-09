#!/bin/bash
# ---------------------------------------------------------------------------
# pilot_config.sh - local, non-secret path configuration for this pilot.
#
# Nothing here is a credential. Large assets (weights, classifiers, anchor
# images, generated samples) live on /data because the root filesystem that
# carries /home is at 100% capacity. See docs/ASSETS.md.
# ---------------------------------------------------------------------------

# Our project repository (small text artefacts only: configs, scripts, notes).
export PILOT_REPO_ROOT="${PILOT_REPO_ROOT:-/home/bijaypandey/FINMLResearch}"

# Scratch root on the large filesystem.
export PILOT_DATA_ROOT="${PILOT_DATA_ROOT:-/data/bijaypandey/cuig_pilot}"

# Upstream CUIG checkout, pinned to the commit in third_party/CUIG/UPSTREAM.md.
export CUIG_REPO_ROOT="${CUIG_REPO_ROOT:-${PILOT_DATA_ROOT}/CUIG}"
export REPO_ROOT="${CUIG_REPO_ROOT}"

# Experiment outputs (models, images, metrics).
export CUIG_OUTPUT_ROOT="${CUIG_OUTPUT_ROOT:-${PILOT_DATA_ROOT}/outputs}"
export OUTPUT_ROOT="${CUIG_OUTPUT_ROOT}"

# UnlearnCanvas assets (downloaded; see docs/ASSETS.md).
export CUIG_UNLEARNCANVAS_GENERATOR_DIR="${CUIG_UNLEARNCANVAS_GENERATOR_DIR:-${PILOT_DATA_ROOT}/Checkpoints/Generators/UnlearnCanvas}"
export CUIG_UNLEARNCANVAS_CLASSIFIER_DIR="${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR:-${PILOT_DATA_ROOT}/Checkpoints/Classifiers/UnlearnCanvas}"

# Isolated Python environment.
export PILOT_VENV="${PILOT_VENV:-${PILOT_DATA_ROOT}/venv-cuig}"

# Keep every cache off the full root filesystem.
export HF_HOME="${HF_HOME:-${PILOT_DATA_ROOT}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${PILOT_DATA_ROOT}/torch_home}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/bijaypandey/pipcache}"
export TMPDIR="${TMPDIR:-/data/bijaypandey/tmp}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

# This host has no Slurm. The upstream BashScripts require these three vars to
# be non-empty before they will source their config; they are never used for
# submission here because we execute training directly.
export CUIG_SLURM_ACCOUNT="${CUIG_SLURM_ACCOUNT:-none-direct-execution}"
export CUIG_SLURM_CLUSTER="${CUIG_SLURM_CLUSTER:-none-direct-execution}"
export CUIG_SLURM_PARTITION="${CUIG_SLURM_PARTITION:-none-direct-execution}"
export CUIG_PRIVATE_EXPORTS="${CUIG_PRIVATE_EXPORTS:-}"
