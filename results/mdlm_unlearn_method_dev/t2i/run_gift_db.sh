#!/bin/bash
# FINDING58 section 6 names GIFT's own threat model as the obvious next measurement and records it as
# NOT RUN: their experiments defend against DreamBooth on a disjoint attack split, not against
# full-UNet relearning on self-generated images. Running it closes the one escape route the finding
# leaves open -- "it was reversed by an adversary the method was not built for".
#
# Uses the DreamBooth adversary from FINDING54/FINDING55, including the prior-preservation artifact
# correction FINDING55 documents.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p .labels logs
A=gift_church_b1.0
for SD in 0 1; do
  K="H81DB_${A}_sd${SD}"
  mkdir ".labels/$K" 2>/dev/null || continue
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py \
    --concept church --unet "adapters/$A/unet.pt" --seed "$SD" --label "$K" --measure-utility \
    >>logs/gift_db.log 2>&1
done
echo "GIFT DB DONE $(date -Is)" >>logs/gift_db.log
