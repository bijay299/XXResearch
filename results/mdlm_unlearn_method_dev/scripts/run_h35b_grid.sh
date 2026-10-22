#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
G=${1:-2}
# H35b: the diffusion attack failed FINDING20's validity gate on the treated arm at 2 of 3 lrs
# (T_rise 0.106 and 0.169 < 0.20) -- it could not relearn its own training set. That leaves the
# treatment-side estimate resting on a single valid attack, so extend the grid upward where
# T_rise will be larger. A weak attack must NOT be allowed to read as robustness.
SEL=adapters/sel_af; BASE=adapters/base_mdu_af
[ -f $SEL/model.pt ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_undo.py \
  --teacher adapters/su_teacher/model.pt --alpha 0.6 --beta 0.1 --protect retain_fisher \
  --steps 12000 --log-every 12000 --retain-oversample 3 --seed 0 --out $SEL \
  > outputs/log_sel_af.txt 2>&1
[ -f $BASE/model.pt ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_unlearn.py --arm mdu \
  --lam-forget 0.5 --lam-retain 1.0 --steps 1500 --log-every 1500 --seed 0 --out $BASE \
  > outputs/log_base_mdu_af.txt 2>&1
for spec in "$SEL sel" "$BASE mdu"; do
  set -- $spec
  for lr in 3e-4 1e-3 3e-3; do
    lab="rtt_af_$2_diff_lr${lr}"
    [ -f "outputs/rtt_${lab}.json" ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_rtt.py --arm "$1/model.pt" \
      --attack-mode diffusion --t-frac 0.8 --steps 600 --lr "$lr" --label "$lab" \
      > "outputs/log_$lab.txt" 2>&1
  done
done
echo "[h35b] done"
