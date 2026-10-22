#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
G=$1
# H39: the MDLM trainability control. FINDING32 showed that in T2I, damage+distil's apparent
# robustness IS general trainability loss, scaling smoothly with alpha. FINDING31 -- this
# project's strongest result -- may rest on the same artifact. This measures it directly on the
# MDLM backbone with a FROZEN fixed-mask probe, so the attacker's fit speed is comparable across
# arms with the mask-rate variance removed.
#
# Arms span the alpha ladder plus a suppression reference. The H37 checkpoints were deleted for
# storage, so they are rebuilt here at the SAME settings.
for spec in "sup_mdu_l0p5|sup|mdu|0.5" "dd_afc_a0|dd|0.0" "dd_afc_a0p3|dd|0.3" \
            "dd_afc_a0p5|dd|0.5" "dd_afc_a0p6|dd|0.6"; do
  IFS='|' read -r lab kind p1 p2 <<< "$spec"
  o=adapters/h39_$lab
  if [ ! -f $o/model.pt ]; then
    if [ "$kind" = "sup" ]; then
      CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_unlearn.py --arm $p1 --lam-forget $p2 \
        --lam-retain 1.0 --steps 1500 --log-every 1500 --seed 0 --out $o > outputs/log_h39_$lab.txt 2>&1
    else
      CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_undo.py --teacher adapters/su_teacher/model.pt \
        --alpha $p1 --beta 0.1 --protect retain_fisher --steps 3000 --log-every 3000 \
        --retain-oversample 3 --seed 0 --out $o > outputs/log_h39_$lab.txt 2>&1
    fi
  fi
  [ -f $o/model.pt ] || { echo "[h39] $lab FAILED to build"; continue; }
  for lr in 3e-5 1e-4; do
    l2="h39_${lab}_lr${lr}"
    [ -f outputs/rtt_${l2}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_rtt.py --arm $o/model.pt --t-frac 0.5 \
      --steps 600 --lr $lr --label $l2 >> outputs/log_h39_$lab.txt 2>&1
  done
  rm -f $o/model.pt
  echo "[h39] $lab done" >> outputs/log_h39_progress.txt
done
echo "[h39] complete"
