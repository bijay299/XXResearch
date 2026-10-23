#!/bin/bash
# H64 relaunch -- frag allocation (forget-critical x retain-sparing), pools RECIPE v2.
# Budget per PREREG_H64 S4a: the frozen 6 points, THEN the 3 extra K1 attacks, reported
# separately. New labels (h64v2_*) -- the dead first launch left no numbers and nothing is reused.
#
# Usage: run_h64_frag.sh <gpu> <concept>
set -u
cd /data/bijaypandey/t2i_unlearn
GPU=$1; CONCEPT=$2
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/${CONCEPT}_unlearned
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/bijaypandey/hf_home
mkdir -p logs
ARM=h64v2_frag_a0.3_${CONCEPT}
LOG=logs/${ARM}.log

if [ ! -f "adapters/${ARM}/unet.pt" ]; then
  echo "=== BUILD $ARM $(date -Is)" >>$LOG
  venv_t2i/bin/python scripts/undo_t2i.py \
    --teacher-te "$TE" --concept "$CONCEPT" \
    --alpha 0.3 --beta 0.1 --protect frag \
    --fisher-batches 16 --steps 6000 --lr 1e-5 --n-images 300 \
    --forget-frac 0.2 --seed 0 --log-every 3000 \
    --label "$ARM" --out "adapters/$ARM" >>$LOG 2>&1
  rc=$?; echo "=== BUILD rc=$rc $(date -Is)" >>$LOG
  [ $rc -ne 0 ] && { echo "BUILD FAILED" >>$LOG; exit 1; }
fi

# utility + target suppression before any attack
if [ ! -f "outputs/gen_${ARM}.json" ]; then
  venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" --unet "adapters/$ARM/unet.pt" \
    --target "$CONCEPT" --save-grid >>$LOG 2>&1
fi

# FROZEN 6-point grid, then the K1 extension -- in that order, so the comparable numbers land first
for spec in "0 1e-5 frozen" "1 1e-5 frozen" "6 1e-5 frozen" "0 5e-5 frozen" "1 5e-5 frozen" \
            "0 1e-4 frozen" "2 1e-5 k1ext" "3 1e-5 k1ext" "6 5e-5 k1ext"; do
  set -- $spec; SD=$1; LR=$2; TAG=$3
  L="H64v2_${TAG}_${CONCEPT}_sd${SD}_lr${LR}"
  [ -f "outputs/atk_${L}.json" ] && { echo "[skip] $L" >>$LOG; continue; }
  echo "=== ATK $L $(date -Is)" >>$LOG
  venv_t2i/bin/python scripts/attack_relearn.py \
    --concept "$CONCEPT" --unet "adapters/$ARM/unet.pt" \
    --steps 300 --lr "$LR" --seed "$SD" --n-images 16 \
    --label "$L" >>$LOG 2>&1
  echo "=== ATK rc=$? $(date -Is)" >>$LOG
done
echo "ALL DONE $(date -Is)" >>$LOG
