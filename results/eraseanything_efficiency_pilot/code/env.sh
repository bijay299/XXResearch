# Isolated environment for the EraseAnything efficiency pilot. Everything on /data
# because the root filesystem is 100% full.
export PILOT=/data/bijaypandey/cuig_pilot/eraseanything_efficiency_pilot
export HF_HOME=/data/bijaypandey/cuig_pilot/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TORCH_HOME=$PILOT/torch_home
export TMPDIR=$PILOT/tmp
export PIP_CACHE_DIR=$PILOT/pip_cache
export XDG_CACHE_HOME=$PILOT/xdg_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
# never inherit a stale visible-device mask; each job sets its own
unset CUDA_VISIBLE_DEVICES
export NLTK_DATA=$PILOT/nltk_data
# Shared 255-core host: cap per-process thread pools. Without this, 3 concurrent generation
# jobs each spawned ~255 OMP threads (load average 432 on 255 cores, GPU util only 12-39%,
# 19s/img vs 6.2s/img solo). 0.695 GPU-h was charged to that aborted attempt.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
