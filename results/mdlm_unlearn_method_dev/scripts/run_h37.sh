#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
G=$1; shift
# H37: designed replication of FINDING28 with a VALIDATED knowledge axis (PREREG_H37).
# Arms are fixed by the prereg. Each arm: both knowledge probes + behavioural eval + 3 gated
# attacks at t_frac 0.5, then the checkpoint is deleted (storage budget) -- measurements kept.
run_arm () {
  lab=$1; o=adapters/h37_$lab
  if [ ! -f outputs/marg_h37_$lab.json ]; then
    [ -f $o/model.pt ] || return 1
    CUDA_VISIBLE_DEVICES=$G $PY scripts/marginal_probe_small.py --ckpt $o/model.pt --label h37_$lab >> outputs/log_h37_$lab.txt 2>&1
    CUDA_VISIBLE_DEVICES=$G $PY scripts/rank_probe_small.py     --ckpt $o/model.pt --label h37_$lab >> outputs/log_h37_$lab.txt 2>&1
    CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_small.py --ckpt $o/model.pt --infill --label h37_$lab >> outputs/log_h37_$lab.txt 2>&1
  fi
  for lr in 3e-5 1e-5 1e-4; do
    [ -f outputs/rtt_h37_${lab}_lr${lr}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_rtt.py --arm $o/model.pt \
      --t-frac 0.5 --steps 600 --lr $lr --label h37_${lab}_lr${lr} >> outputs/log_h37_$lab.txt 2>&1
  done
  rm -f $o/model.pt          # measurements are kept; the 0.46 GiB checkpoint is not
}
for spec in "$@"; do
  IFS='|' read -r lab kind p1 p2 <<< "$spec"
  o=adapters/h37_$lab
  if [ ! -f outputs/marg_h37_$lab.json ] && [ ! -f $o/model.pt ]; then
    if [ "$kind" = "sup" ]; then
      CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_unlearn.py --arm $p1 --lam-forget $p2 \
        --lam-retain 1.0 --steps 1500 --log-every 1500 --seed 0 --out $o > outputs/log_h37_$lab.txt 2>&1
    else
      CUDA_VISIBLE_DEVICES=$G $PY scripts/train_small_undo.py --teacher $p1/model.pt \
        --alpha $p2 --beta 0.1 --protect retain_fisher --steps 3000 --log-every 3000 \
        --retain-oversample 3 --seed 0 --out $o > outputs/log_h37_$lab.txt 2>&1
    fi
  fi
  run_arm $lab
  echo "[h37] $lab done" >> outputs/log_h37_progress.txt
done
echo "[h37] shard complete"
