# Isolated environment for the LLaDA smoke test. Root fs is 100% full:
# every cache must resolve under /data.
W=/data/bijaypandey/llada_smoke
export HF_HOME=$W/hf_home
export HF_MODULES_CACHE=$W/hf_home/modules
export HF_HUB_CACHE=$W/hf_home/hub
export XDG_CACHE_HOME=$W/hf_home/xdg
export TMPDIR=$W/tmp
export TORCH_HOME=$W/hf_home/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
