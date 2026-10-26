#!/bin/bash
# E6: first and only legitimate use of the FINAL prompt set (frozen 2026-10-25T23:30, before any
# final result existed). Arm list pre-committed in scripts/gen_eval_final.py. Writes JSONs only --
# no checkpoints -- so it is safe with /data near capacity.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
venv_t2i/bin/python scripts/gen_eval_final.py >>logs/e6_final.log 2>&1
echo "E6 DONE rc=$? $(date -Is)" >>logs/e6_final.log
