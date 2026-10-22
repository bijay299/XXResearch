#!/bin/bash
# Follow-on for the UNDO alpha sweep.
#
# WHY THE ORDER CHANGED. The sweep's arms distil on TOFU-retain + RWKU only, which contains none
# of the synthetic TOFU forget authors. UNDO transfers the teacher's suppressed behaviour by
# distilling on data that COVERS the forget distribution (the paper uses an epoch of pretraining
# data). With no forget questions in the pool the student gets no signal at all on the inputs we
# evaluate, so those arms are "damage + retain-only distillation", not a faithful UNDO test.
# They are kept as a legitimate variant; the faithful arms run FIRST because they are the test.
#
# Targets are always the TEACHER's outputs, never gold answers, so including forget questions
# leaks no forget content.
#
# Phase 1  faithful UNDO, alpha=0.3 --include-forget, 1000 steps. 1000 not 300 because the
#          frozen probe on the 300-step arm fell 2.467 -> 2.188 -> 1.663 at steps 0/49/99,
#          still falling with an INCREASING rate: 300 steps is an under-trained student, and
#          weak utility there would be indistinguishable from "UNDO's route does not work".
# Phase 2  the alpha=0 CONTROL in the SAME faithful configuration. UNDO's claim is that
#          distilling into a DAMAGED copy is what robustifies; alpha=0 distils into an
#          undamaged copy of the same teacher. If alpha=0 is equally robust the damage does
#          nothing and any gain is plain distillation.
# Phase 3  partition eval + validated RTT (FINDING20's gate) on every arm, two lanes.
# Phase 4  H17's still-owed theta0 decoder grid.
# Phase 5  adversarial learning-rate search, to turn FINDING20's single-lr recoveries into
#          bounds under the strongest attack.
#
# Two separate bars: beating the teacher AFC lam=1 (recovery 0.200) shows the mechanism works
# on an MDLM; beating the robustness incumbent MDU lam=3 (recovery 0.075) at comparable
# DISCLOSE and retain is what a SOTA claim needs. Clearing only the first is NOT a SOTA result.
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
export OMP_NUM_THREADS=60
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
DS=/data/bijaypandey/llada_smoke/venv_llada/bin/deepspeed

train_undo () {                 # $1 = label, $2 = alpha, $3 = steps, $4 = extra args
  lab="$1"; al="$2"; st="$3"; extra="${4:-}"
  if [ -f "adapters/$lab/undo_step${st}.bin" ]; then
    echo "[train] $lab already done"; return
  fi
  echo "[train] $lab alpha=$al steps=$st $extra"
  $DS --include localhost:2,3 --master_port 29577 \
    scripts/train_undo.py --alpha "$al" --beta 0.1 --steps "$st" \
    --probe-every 50 --save-every $(( st / 2 )) --batch 1 $extra \
    --out "/data/bijaypandey/mdlm_unlearn/adapters/$lab" \
    > "outputs/log_$lab.txt" 2>&1 || echo "[train] FAILED $lab"
}

# ---- Phases 1 and 2: the faithful arm and its control -----------------------------------
train_undo undo_fa0p3 0.3 1000 --include-forget
train_undo undo_fa0p0 0.0 1000 --include-forget

# ---- Phase 3: evaluate + attack ---------------------------------------------------------
evaluate () {                   # $1 = gpu, $2 = label, $3 = checkpoint path, $4 = rtt? yes/no
  g="$1"; lab="$2"; ck="$3"; do_rtt="$4"
  [ -f "$ck" ] || { echo "[lane $g] no checkpoint $ck"; return; }
  if [ ! -f "outputs/arm_${lab}.json" ]; then
    echo "[lane $g] partition eval $lab"
    CUDA_VISIBLE_DEVICES="$g" $PY scripts/eval_arm_partition.py \
      --memo none --state-dict "$ck" --label "$lab" \
      > "outputs/log_part_${lab}.txt" 2>&1 || echo "[lane $g] EVAL FAILED $lab"
  fi
  if [ "$do_rtt" = yes ] && [ ! -f "outputs/rtt_${lab}.json" ]; then
    echo "[lane $g] RTT $lab"
    CUDA_VISIBLE_DEVICES="$g" $PY scripts/train_rtt.py \
      --arm "$ck" --lr 3e-5 --probe-steps 0,25,100,300 --n-eval 40 \
      --label "$lab" > "outputs/log_rtt_${lab}.txt" 2>&1 || echo "[lane $g] RTT FAILED $lab"
  fi
}

# the faithful arm and its control are the priority pair
evaluate 2 undo_fa0p3 adapters/undo_fa0p3/undo_step1000.bin yes &
evaluate 3 undo_fa0p0 adapters/undo_fa0p0/undo_step1000.bin yes &
wait
# mid-distillation checkpoint: utility-vs-steps, no attack needed
evaluate 2 undo_fa0p3_s500 adapters/undo_fa0p3/undo_step500.bin no &
# the retain-only variants from the sweep
evaluate 3 undo_a0p3 adapters/undo_a0p3/undo_step300.bin yes &
wait
evaluate 2 undo_a0p6 adapters/undo_a0p6/undo_step300.bin yes &
wait

# ---- Phase 4: H17's mandatory reference arm ---------------------------------------------
# Without it the AFC grid's spread cannot be attributed to unlearning fragility rather than
# decoder variance. No --arm means base + memo, which IS theta0.
if [ ! -f outputs/decatk_theta0.json ]; then
  echo "[followup] theta0 decoder grid (H17 reference arm)"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/decode_attack.py --label theta0 --n-forget 60 \
    > outputs/log_decatk_theta0.txt 2>&1 || echo "[followup] theta0 GRID FAILED"
fi

# ---- Phase 5: adversarial learning-rate search for FINDING20 ----------------------------
# 1e-4 and 1e-5 already FAILED the validity gate (degeneracy 0.93-0.98), so the grid brackets
# 3e-5 inside the usable window.
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
for f in outputs/arm_undo_*.json outputs/rtt_undo_*.json; do
  [ -f "$f" ] || continue
  echo "== $f"
  $PY -c "
import json
R=json.load(open('$f'))
for k in ('label','DISCLOSE','DEGENERATE','retain_ascr','tv_to_pristine'):
    if k in R: print('  ',k,R[k])
if 'cells' in R: print('   cells',R['cells'])
if 'TV_forget_to' in R: print('   TV',R['TV_forget_to'])
if 'recovery' in R: print('   recovery',R['recovery'])
if 'attack_validity' in R:
    v=R['attack_validity']; print('   validity valid=',v.get('valid'),'T_rise=',v.get('T_rise'))
"
done
