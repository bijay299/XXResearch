# Source before every job. Pins the known-good FLUX stack and keeps every
# cache off the 100%-full root filesystem.
export CPR_ROOT=/data/bijaypandey/cuig_pilot/cpr_real_model_controls_v1
export CPR_PY=/data/bijaypandey/cuig_pilot/venv-cuig/bin/python
export HF_HOME=/data/bijaypandey/cuig_pilot/hf_home
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1                      # pristine caches, read-only, no downloads
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/data/bijaypandey/.cache
export TMPDIR=/data/bijaypandey/claude_tmp
export TRITON_CACHE_DIR=/data/bijaypandey/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/bijaypandey/.cache/inductor
export PYTHONPATH=$CPR_ROOT:$CPR_ROOT/pydeps   # pilot-owned overlay; shared venv stays read-only
# telemetry off
export WANDB_DISABLED=true WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
# CPU/BLAS workers <= 8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
