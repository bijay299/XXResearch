#!/bin/bash
# ---------------------------------------------------------------------------
# SD-1.5 exploratory sequential object-erasure pilot - shared configuration.
#
# This is a SEPARATE experiment from the (blocked) UnlearnCanvas baseline. It
# intentionally uses base Stable Diffusion v1.5 as its backbone, and a public
# COCO object detector as its evaluator. Nothing here produces UA/IRA/CRA and
# nothing here is a reproduction of a published result.
# ---------------------------------------------------------------------------

export PILOT_REPO_ROOT="${PILOT_REPO_ROOT:-/home/bijaypandey/FINMLResearch}"
export PILOT_DATA_ROOT="${PILOT_DATA_ROOT:-/data/bijaypandey/cuig_pilot}"
export SEQ_ROOT="${SEQ_ROOT:-${PILOT_DATA_ROOT}/seq_pilot}"

export CUIG_REPO_ROOT="${CUIG_REPO_ROOT:-${PILOT_DATA_ROOT}/CUIG}"
export REPO_ROOT="${CUIG_REPO_ROOT}"
export PILOT_VENV="${PILOT_VENV:-${PILOT_DATA_ROOT}/venv-cuig}"

# M0: the untouched backbone for this experiment (already downloaded).
export SEQ_BASE_MODEL="${SEQ_BASE_MODEL:-${PILOT_DATA_ROOT}/Generators_substitute/sd-v1-5}"

# Caches off the full root filesystem.
export HF_HOME="${HF_HOME:-${PILOT_DATA_ROOT}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${PILOT_DATA_ROOT}/torch_home}"
export TMPDIR="${TMPDIR:-/data/bijaypandey/tmp}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

# Accelerate config keeps gpu_ids: all so our CUDA_VISIBLE_DEVICES pin survives.
export SEQ_ACCEL_CFG="${PILOT_REPO_ROOT}/configs/accelerate_single_visible_gpu.yaml"

# --- Fixed training protocol ------------------------------------------------
export SEQ_ITERATIONS="${SEQ_ITERATIONS:-1000}"   # upstream sequential uses 2000
export SEQ_ANCHOR_IMAGES=200
export SEQ_ANCHOR_PROMPTS=200
export SEQ_BATCH=4                                 # anchor_batch_size default
# 200 anchor images / batch 4 = 50 optimizer steps per epoch. Upstream's default
# --epochs 1 would therefore stop at 50 steps, far short of SEQ_ITERATIONS. We
# raise the epoch cap so the step budget is what actually binds; training still
# breaks at exactly SEQ_ITERATIONS optimizer steps.
export SEQ_EPOCHS="${SEQ_EPOCHS:-25}"              # 25*50 = 1250 >= 1000
export SEQ_L2SP_WEIGHT=25000
export SEQ_PARAM_GROUP="kv-xattn"                  # upstream default, held fixed

# --- Anchor caches (generated once from untouched M0, shared by all arms) ----
export SEQ_ANCHOR_SEED="${SEQ_ANCHOR_SEED:-17}"
export SEQ_ANCHOR_ROOT="${SEQ_ROOT}/anchor_caches"
export SEQ_ANCHOR_HORSES="${SEQ_ANCHOR_ROOT}/Horses"
export SEQ_ANCHOR_FLOWERS="${SEQ_ANCHOR_ROOT}/Flowers"
export SEQ_PROMPTS_HORSES="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/object/Horses.txt"
export SEQ_PROMPTS_FLOWERS="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/object/Flowers.txt"

# --- Evaluation generation settings (identical at every checkpoint) ---------
export SEQ_EVAL_STEPS=30
export SEQ_EVAL_GUIDANCE=7.5
export SEQ_EVAL_RES=512
export SEQ_EVAL_SEEDS="101 202 303 404"
export SEQ_EVAL_CATEGORIES="cat dog sandwich horse bird chair bicycle"
export SEQ_DET_THRESHOLDS="0.3 0.5 0.7"
