#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# SECOND ATTACK FAMILY. All robustness numbers so far use SFT on the answer span. If the selected
# point's protection is specific to that surface, the central claim collapses. The diffusion
# attack optimises the MDLM objective over whole passages instead.
SEL=adapters/sel_af
BASE=adapters/base_mdu_af
[ -f $SEL/model.pt ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py \
  --teacher adapters/su_teacher/model.pt --alpha 0.6 --beta 0.1 --protect retain_fisher \
  --steps 12000 --log-every 12000 --retain-oversample 3 --seed 0 --out $SEL \
  > outputs/log_sel_af.txt 2>&1
[ -f $BASE/model.pt ] || CUDA_VISIBLE_DEVICES=3 $PY scripts/train_small_unlearn.py --arm mdu \
  --lam-forget 0.5 --lam-retain 1.0 --steps 1500 --log-every 1500 --seed 0 --out $BASE \
  > outputs/log_base_mdu_af.txt 2>&1
i=0
for spec in "$SEL sel" "$BASE mdu"; do
  set -- $spec
  for lr in 3e-5 1e-4 1e-5; do
    lab="rtt_af_$2_diff_lr${lr}"
    [ -f "outputs/rtt_${lab}.json" ] && continue
    g=$(( i % 2 + 2 )); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$1/model.pt" \
      --attack-mode diffusion --t-frac 0.8 --steps 600 --lr "$lr" --label "$lab" \
      > "outputs/log_$lab.txt" 2>&1 &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
rm -f $SEL/model.pt $BASE/model.pt
echo "[h35] done"
