# Isolated environment for the BLOCK object-erasure sprint. Source before any run.
export SPRINT=/data/bijaypandey/cuig_pilot/block_object_sprint
export REPO=$SPRINT/code/block-concept-removal
export VENV=$SPRINT/env/venv-block
export HF_HOME=/data/bijaypandey/cuig_pilot/hf_home
export HF_HUB_CACHE=/data/bijaypandey/cuig_pilot/hf_home/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export HF_HUB_OFFLINE=0
export FLUX_PATH=/data/bijaypandey/cuig_pilot/hf_home/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
export SD3_PATH=/nonexistent-not-used-this-sprint
export BASE_DIR=$SPRINT/artifacts/transcoders
export COCO_PROMPTS=$REPO/prompts/coco_train2014_30k_prompts.txt
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$SPRINT/code
export TMPDIR=/data/bijaypandey/cuig_pilot/block_object_sprint/scratch
mkdir -p $TMPDIR $BASE_DIR
# all JIT / compile caches on /data (home filesystem is full)
export TRITON_CACHE_DIR=$SPRINT/scratch/triton_cache
export TORCHINDUCTOR_CACHE_DIR=$SPRINT/scratch/inductor_cache
export XDG_CACHE_HOME=$SPRINT/scratch/xdg_cache
export WANDB_MODE=disabled
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME
export LLAVA_PATH=$SPRINT/assets/models/llava-v1.6-vicuna-7b
export LLAVA_REVISION=deae57a8c0ccb0da4c2661cc1891cc9d06503d11
export VENV_LLAVA=$SPRINT/env/venv-llava
export LLAVA_REPO=$REPO/src/LLaVA
