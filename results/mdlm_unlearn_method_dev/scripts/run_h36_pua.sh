#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
G=${1:-3}
# H36 stage 1: find the lambda that puts the PUA teacher AT CHANCE while staying fluent.
# Both variants: `pua` (closed pool, retain-derived) and `pua_global` (pool-free control, K3).
for arm in pua pua_global; do
  for lam in 1 2 4 8; do
    o=adapters/h36_${arm}_l${lam}
    [ -f outputs/arm_h36_${arm}_l${lam}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_unlearn.py --arm $arm \
      --lam-forget $lam --lam-retain 1.0 --steps 1500 --log-every 750 --seed 0 --out $o \
      > outputs/log_h36_${arm}_l${lam}.txt 2>&1 || continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/rank_probe_small.py --ckpt $o/model.pt \
      --label h36_${arm}_l${lam} >> outputs/log_h36_${arm}_l${lam}.txt 2>&1
    CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_small.py --ckpt $o/model.pt --infill \
      --label h36_${arm}_l${lam} >> outputs/log_h36_${arm}_l${lam}.txt 2>&1
    # teachers are kept only if they are candidates; stage 2 re-trains the chosen one
  done
done
echo "[h36-stage1] done"
