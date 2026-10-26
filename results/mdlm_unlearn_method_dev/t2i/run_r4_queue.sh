#!/bin/bash
# R4: UCE official-vs-port numerical comparison. Waits for a free GPU rather than displacing the
# running experiments (E4m on GPU 0, E2g on GPU 1). GPUs 2-3 belong to other users and are not used.
# Runs at the OFFICIAL defaults (lamb 0.5, erase_scale 1.0, preserve_scale 1.0), then at OUR
# historical default (lamb 0.1) so the default mismatch is measured, not assumed.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4_uce.log
# wait for one of MY two GPUs (0 or 1) to free
while :; do
  G=""
  for i in 0 1; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
    [ "$U" -lt 2000 ] && { G=$i; break; }
  done
  [ -n "$G" ] && break
  sleep 120
done
echo "=== R4 starting on GPU $G $(date -Is)" >>"$L"
export CUDA_VISIBLE_DEVICES=$G
for cfg in "0.5 official-default" "0.1 our-historical-default"; do
  set -- $cfg
  echo "--- lamb=$1 ($2) ---" >>"$L"
  venv_uce/bin/python scripts/r4_uce_official_vs_port.py --concept church \
    --lamb "$1" --erase-scale 1.0 --preserve-scale 1.0 >>"$L" 2>&1
  cp outputs/r4_uce_official_vs_port.json "outputs/r4_uce_lamb$1.json" 2>/dev/null
done
# and at erase_scale != 1, where the two forms MUST diverge if our port lacks the coefficient
echo "--- erase_scale=2.0 (divergence probe) ---" >>"$L"
venv_uce/bin/python scripts/r4_uce_official_vs_port.py --concept church \
  --lamb 0.5 --erase-scale 2.0 --preserve-scale 1.0 >>"$L" 2>&1
cp outputs/r4_uce_official_vs_port.json outputs/r4_uce_erasescale2.json 2>/dev/null
echo "R4 DONE $(date -Is)" >>"$L"
