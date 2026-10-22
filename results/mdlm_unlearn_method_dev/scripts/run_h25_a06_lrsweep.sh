#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# alpha=0.6's zero-recovery result rests on ONE attack lr (3e-5). That is exactly the limitation
# NOVELTY_attack_validity_gate.md recorded against FINDING20, and it already caught one of our own
# claims (alpha=0.5 "fully robust" -> withdrawn). Max-over-lr is the only defensible number.
T=adapters/su_teacher_emdu/model.pt
A=adapters/undo_a06_lrsweep
if [ ! -f "$A/model.pt" ]; then
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py --teacher "$T" --alpha 0.6 --beta 0.1 \
    --protect retain_fisher --steps 12000 --log-every 12000 --retain-oversample 3 --seed 0 \
    --out "$A" > outputs/log_undo_a06_lrsweep.txt 2>&1 || exit 1
  CUDA_VISIBLE_DEVICES=2 $PY scripts/rank_probe_small.py --ckpt "$A/model.pt" \
    --label undo_a06_lrsweep > outputs/log_rank_undo_a06_lrsweep.txt 2>&1
fi
i=0
for lr in 1e-5 1e-4 3e-4 1e-3; do
  lab="rtt_a06_lr${lr}"
  [ -f "outputs/rtt_${lab}.json" ] && continue
  g=$(( i % 2 + 2 )); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$A/model.pt" \
    --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
done
wait
rm -f "$A/model.pt"
echo "[a06lr] done"
