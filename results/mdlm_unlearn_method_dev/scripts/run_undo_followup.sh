#!/bin/bash
# Follow-on for the UNDO alpha sweep.
#
# Phase 1  train the alpha=0 CONTROL. This is the control that makes the whole study
#          interpretable: UNDO's claim is that distilling into a DAMAGED copy is what makes
#          suppression robust. alpha=0 distils into an undamaged copy of the same suppressed
#          teacher. If alpha=0 is just as robust, the damage does nothing and any gain is
#          plain distillation; if only alpha>0 is robust, the damage is the active ingredient.
# Phase 2  partition-evaluate each arm, then run the VALIDATED RTT protocol (FINDING20's gate).
#          Two lanes, one GPU each, so we stay at the two-GPU cap.
# Phase 3  H17's still-owed theta0 decoder grid.
#
# This is the test of FINDING18+19's joint prediction: both walls concern taking the forget
# target from the model's own distributions, and the joint prediction is that a working method
# must source it elsewhere. UNDO (damage + distil) is that route. Two separate bars:
#   beat its own teacher AFC lam=1 (recovery 0.200) -> the mechanism works on an MDLM;
#   beat the robustness incumbent MDU lam=3 (recovery 0.075) at comparable DISCLOSE and
#   retain -> a genuine SOTA claim. Clearing only the first is NOT a SOTA result.
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
export OMP_NUM_THREADS=60
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

# ---- Phase 1: the alpha=0 control -------------------------------------------------------
if [ ! -f adapters/undo_a0p0/undo_step300.bin ]; then
  echo "[followup] training alpha=0.0 control (distil into an UNDAMAGED suppressed model)"
  /data/bijaypandey/llada_smoke/venv_llada/bin/deepspeed --include localhost:2,3 \
    --master_port 29577 \
    scripts/train_undo.py --alpha 0.0 --beta 0.1 --steps 300 \
    --probe-every 50 --save-every 300 --batch 1 \
    --out /data/bijaypandey/mdlm_unlearn/adapters/undo_a0p0 \
    > outputs/log_undo_a0p0.txt 2>&1 || echo "[followup] alpha=0 CONTROL FAILED"
fi

# ---- Phase 2: evaluate + attack each arm ------------------------------------------------
lane () {                       # $1 = gpu, $2 = label
  g="$1"; lab="$2"
  ck="adapters/$lab/undo_step300.bin"
  if [ ! -f "$ck" ]; then echo "[lane $g] no checkpoint for $lab, skipping"; return; fi

  if [ ! -f "outputs/arm_${lab}.json" ]; then
    echo "[lane $g] partition eval $lab"
    CUDA_VISIBLE_DEVICES="$g" $PY scripts/eval_arm_partition.py \
      --memo none --state-dict "$ck" --label "$lab" \
      > "outputs/log_part_${lab}.txt" 2>&1 || echo "[lane $g] EVAL FAILED $lab"
  fi

  if [ ! -f "outputs/rtt_${lab}.json" ]; then
    echo "[lane $g] RTT $lab (lr 3e-5, the setting whose control passed)"
    CUDA_VISIBLE_DEVICES="$g" $PY scripts/train_rtt.py \
      --arm "$ck" --lr 3e-5 --probe-steps 0,25,100,300 --n-eval 40 \
      --label "$lab" > "outputs/log_rtt_${lab}.txt" 2>&1 || echo "[lane $g] RTT FAILED $lab"
  fi
}

lane 2 undo_a0p3 &
lane 3 undo_a0p6 &
wait
lane 2 undo_a0p0 &
wait

# ---- Phase 2b: steps ablation, to rule out a false negative ------------------------------
# The frozen probe on alpha=0.3 fell 2.467 -> 2.188 -> 1.663 at steps 0/49/99: still falling,
# and the rate was INCREASING, so 300 steps is very likely an under-trained student. Without
# this arm, poor utility at 300 steps is indistinguishable from "UNDO's route does not work".
# 1000 steps with checkpoints at 500 and 1000 gives a repair-vs-steps curve.
if [ ! -f adapters/undo_a0p3_s1000/undo_step1000.bin ]; then
  echo "[followup] alpha=0.3 extended to 1000 steps (steps ablation)"
  /data/bijaypandey/llada_smoke/venv_llada/bin/deepspeed --include localhost:2,3 \
    --master_port 29577 \
    scripts/train_undo.py --alpha 0.3 --beta 0.1 --steps 1000 \
    --probe-every 50 --save-every 500 --batch 1 \
    --out /data/bijaypandey/mdlm_unlearn/adapters/undo_a0p3_s1000 \
    > outputs/log_undo_a0p3_s1000.txt 2>&1 || echo "[followup] EXTENDED ARM FAILED"
fi
for ck in undo_step500 undo_step1000; do
  f="adapters/undo_a0p3_s1000/$ck.bin"
  [ -f "$f" ] || continue
  lab="undo_a0p3_$ck"
  if [ ! -f "outputs/arm_${lab}.json" ]; then
    CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_arm_partition.py \
      --memo none --state-dict "$f" --label "$lab" \
      > "outputs/log_part_${lab}.txt" 2>&1 || echo "[followup] EVAL FAILED $lab"
  fi
done
# attack only the fully-distilled one; the 500 checkpoint is there for the utility curve
if [ -f adapters/undo_a0p3_s1000/undo_step1000.bin ] && [ ! -f outputs/rtt_undo_a0p3_s1000.json ]; then
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train_rtt.py \
    --arm adapters/undo_a0p3_s1000/undo_step1000.bin --lr 3e-5 \
    --probe-steps 0,25,100,300 --n-eval 40 --label undo_a0p3_s1000 \
    > outputs/log_rtt_undo_a0p3_s1000.txt 2>&1 || echo "[followup] RTT FAILED extended arm"
fi

# ---- Phase 3: H17's mandatory reference arm ---------------------------------------------
# Without it the AFC grid's spread cannot be attributed to unlearning fragility rather than
# decoder variance. No --arm means base + memo, which IS theta0.
if [ ! -f outputs/decatk_theta0.json ]; then
  echo "[followup] theta0 decoder grid (H17 reference arm)"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/decode_attack.py --label theta0 --n-forget 60 \
    > outputs/log_decatk_theta0.txt 2>&1 || echo "[followup] theta0 GRID FAILED"
fi

# ---- Phase 4: adversarial learning-rate search for FINDING20 ----------------------------
# UNDO and Lynch et al. both report the adversarially-OPTIMAL relearning attack over an lr grid.
# FINDING20 used a single lr (3e-5), so its recoveries are lower bounds and the AFC-vs-MDU
# ordering is not yet shown stable under the strongest attack. 1e-4 and 1e-5 already FAILED the
# validity gate (degeneracy 0.93-0.98), so the grid brackets 3e-5 inside the usable window.
AFC=/data/bijaypandey/mdlm_unlearn/adapters/h16_none_f1/ckpt_300
MDU=/data/bijaypandey/mdlm_unlearn/adapters/ll_mdu_f3/ckpt_300
rtt_lr () {                     # $1 = gpu, $2 = arm path, $3 = tag, $4 = lr
  out="outputs/rtt_$3_lr$4.json"
  [ -f "$out" ] && return
  CUDA_VISIBLE_DEVICES="$1" $PY scripts/train_rtt.py --arm "$2" --lr "$4" \
    --probe-steps 0,25,100,300 --n-eval 40 --label "$3_lr$4" \
    > "outputs/log_rtt_$3_lr$4.txt" 2>&1 || echo "[lrsweep] FAILED $3 lr=$4"
}
for lr in 2e-5 5e-5 7e-5; do
  echo "[lrsweep] lr=$lr on AFC(gpu2) and MDU(gpu3)"
  rtt_lr 2 "$AFC" afc_l1 "$lr" &
  rtt_lr 3 "$MDU" mdu_l3 "$lr" &
  wait
done

echo "[followup] done"
for lab in undo_a0p0 undo_a0p3 undo_a0p6; do
  for f in "outputs/arm_${lab}.json" "outputs/rtt_${lab}.json"; do
    [ -f "$f" ] || continue
    echo "== $f"
    $PY -c "
import json
R=json.load(open('$f'))
for k in ('label','DISCLOSE','DEGENERATE','retain_ascr','tv_to_pristine'):
    if k in R: print('  ',k,R[k])
if 'cells' in R: print('   cells',R['cells'])
if 'recovery' in R: print('   recovery',R['recovery'])
if 'attack_validity' in R:
    v=R['attack_validity']; print('   validity valid=',v.get('valid'),'T_rise=',v.get('T_rise'))
"
  done
done
