export CPRX_ROOT=/data/bijaypandey/cuig_pilot/cpr_execution_recovery_v1
export CPRD_PY=/data/bijaypandey/cuig_pilot/venv-cuig/bin/python
export HF_HOME=/data/bijaypandey/cuig_pilot/hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export CPRD_BASE_MODEL_PATH__FLUX=$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=$CPRX_ROOT/tmp/xdg
export TMPDIR=$CPRX_ROOT/tmp
export TRITON_CACHE_DIR=$CPRX_ROOT/tmp/triton
export TORCHINDUCTOR_CACHE_DIR=$CPRX_ROOT/tmp/inductor
export PYTHONPATH=$CPRX_ROOT/code:$CPRX_ROOT/pydeps:$CPRX_ROOT/GEM_src
export WANDB_DISABLED=true WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
