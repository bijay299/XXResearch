#!/bin/bash
# R2: the OFFICIAL Meta-Unlearning trainer with its THREE data pools, author protocol.
#
# Pool meanings, verbatim from upstream/Meta-Unlearning/README.md:
#   hrm = the UNLEARNED concept
#   irt = retained concepts UNRELATED to it
#   tgt = retained concepts RELATED to it
# (Our manifest previously mislabelled tgt as the forget pool; corrected 2026-10-26.)
#
# Construction follows src/gen_images.py: 30 train + 30 test images per pool, the author's exact
# negative prompt, metadata.jsonl written per directory, loaded by meta.py as `imagefolder`.
# irt prompts are drawn from the author's own data/coco_object_retain.csv (244 generic objects).
#
# Concept = church, so the resulting checkpoint is evaluable under our common protocol.
# This is an OFFICIAL REPRODUCTION attempt: upstream src/meta.py is invoked, not our port.
set -u
cd /data/bijaypandey/t2i_unlearn
R=upstream_runs/meta
L=logs/r2_meta.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R2 on GPU $G $(date -Is)" >>"$L"
mkdir -p "$R"
venv_t2i/bin/python scripts/r2_build_meta_pools.py --root "$R/dataset" >>"$L" 2>&1
echo "  pool build rc=$?" >>"$L"
for d in hrm irt tgt; do
  echo "  $d: $(ls $R/dataset/$d/train/*.png 2>/dev/null | wc -l) train, $(ls $R/dataset/$d/test/*.png 2>/dev/null | wc -l) test" >>"$L"
done
echo "R2 POOLS DONE $(date -Is)" >>"$L"
