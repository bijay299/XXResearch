export CPRD_ROOT=/data/bijaypandey/cuig_pilot/cpr_detector_controls_v1
export CPRD_PY=/data/bijaypandey/cuig_pilot/venv-cuig/bin/python
export HF_HOME=/data/bijaypandey/cuig_pilot/hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# All required downloads (both detectors, COCO annotations) completed during
# preflight. FLUX.1-dev is a GATED repo: without offline mode the hub issues a
# 401 head request instead of using the pristine local cache.
export CPRD_BASE_MODEL_PATH__FLUX=$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/data/bijaypandey/.cache
export TMPDIR=/data/bijaypandey/claude_tmp
export TRITON_CACHE_DIR=/data/bijaypandey/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/bijaypandey/.cache/inductor
export PYTHONPATH=$CPRD_ROOT:$CPRD_ROOT/pydeps:$CPRD_ROOT/GEM_src
export WANDB_DISABLED=true WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
