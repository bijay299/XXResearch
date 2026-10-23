#!/bin/bash
# H68 -- the control FINDING39 S4 queued: a matched v2 rebuild of retain_fisher.
# Everything identical to h64v2 except --protect retain_fisher, so allocation is the ONLY variable
# that differs from H64, and pool recipe is the ONLY variable that differs from h47 (v1).
#   lands near 0.85 -> v2 pools innocent, the frag allocation is the cause (H64 cleanly false)
#   also collapses -> the POOL REPAIR is the cause, and every published damage+distil retain
#                     number in this phase is an in-sample artifact
# Usage: run_h68_v2_control.sh <gpu> <concept>
set -u
cd /data/bijaypandey/t2i_unlearn
GPU=$1; CONCEPT=$2
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/${CONCEPT}_unlearned
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/bijaypandey/hf_home
mkdir -p logs
ARM=h68v2_rf_a0.3_${CONCEPT}
LOG=logs/${ARM}.log
if [ ! -f "adapters/${ARM}/unet.pt" ]; then
  echo "=== BUILD $ARM $(date -Is)" >>$LOG
  venv_t2i/bin/python scripts/undo_t2i.py \
    --teacher-te "$TE" --concept "$CONCEPT" \
    --alpha 0.3 --beta 0.1 --protect retain_fisher \
    --fisher-batches 16 --steps 6000 --lr 1e-5 --n-images 300 \
    --forget-frac 0.2 --seed 0 --log-every 3000 \
    --label "$ARM" --out "adapters/$ARM" >>$LOG 2>&1
  echo "=== BUILD rc=$? $(date -Is)" >>$LOG
fi
[ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
  --label "$ARM" --unet "adapters/$ARM/unet.pt" --target "$CONCEPT" >>$LOG 2>&1
echo "H68 DONE $(date -Is)" >>$LOG
