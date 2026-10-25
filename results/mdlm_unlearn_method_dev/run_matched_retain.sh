#!/bin/bash
# THE DECISIVE COMPARISON FOR THE MANDATE, which matched-disclosure alone cannot settle.
#
# At matched DISCLOSURE (AFC 0.155 vs MDU lam=3 0.152) the corrected n=12/n=8 result is TV parity
# (0.298 vs 0.298) with AFC retaining +0.12 more. That is one slice through the trade-off. The
# complementary slice is matched RETENTION: hold retain fixed and ask who gets closer to pristine.
#
# AFC lam=1 sits at retain 0.5153 (n=12). MDU lam=3 sits at 0.3928 -- far below it, so lam=3 is the
# WRONG baseline for this slice. MDU lam=1's published n=3 row is retain 0.557, TV 0.423, i.e. the
# nearest-retain MDU arm. If AFC's 0.298 beats MDU lam=1's TV at comparable retain, AFC has a real
# frontier advantage and that is the contribution. If MDU lam=1 comes in near 0.30 too, AFC has NO
# advantage on either slice and the honest conclusion is that this method does not beat the baseline.
#
# MDU lam=1's published figures are n=3 and its checkpoints are among those deleted, so it must be
# rebuilt at n=12 under the explicit recorded settings whose bridges passed today.
#
# LABEL UNIQUENESS: a collision between two of my own scripts contaminated mdu_l3_s11_n12 earlier.
# Every label here is claimed with mkdir before any work starts, so a second claimant exits instead
# of racing.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
mkdir -p .labels

claim() { mkdir ".labels/$1" 2>/dev/null; }   # rc!=0 means someone else owns this label

build() {  # $1 seed  $2 gpu
  S=$1; L="mdu_l1_s${S}_mr"
  claim "$L" || { echo "$L already claimed, skipping" >>logs/matched_retain.log; return 0; }
  export CUDA_VISIBLE_DEVICES=$2
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget 1 \
    --lam-retain 1 --seed "$S" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1
}

# GPU 1 is free now. GPU 0 is finishing the seed-11 repair, so that worker waits for it rather than
# putting three of my jobs on the node at once.
( for S in 0 1 2 3 4 5; do build "$S" 1; done; echo "MR A DONE $(date -Is)" >>logs/matched_retain.log ) &
( while pgrep -f "mdu_l3_s11_n12b" >/dev/null; do sleep 30; done
  for S in 6 7 8 9 10 11; do build "$S" 0; done; echo "MR B DONE $(date -Is)" >>logs/matched_retain.log ) &
wait
