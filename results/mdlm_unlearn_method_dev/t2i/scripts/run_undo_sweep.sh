#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; shift
# H38 treatment: damage+distil on the UCE-erased teacher (the stronger eraser: church 0.067).
# alpha=0 is the DISCRIMINATING CONTROL -- distillation with no damage. If alpha=0 is as robust
# as alpha>0, the damage does nothing and H38 is false for the same reason FINDING23 tested.
for spec in "$@"; do
  IFS='|' read -r al prot <<< "$spec"
  arm="undo_a${al}_${prot}"
  o=adapters/$arm
  if [ ! -f outputs/undo_${arm}.json ]; then
    CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py --teacher adapters/uce_church/unet.pt \
      --concept church --alpha $al --beta 0.1 --protect $prot --steps 1500 --log-every 500 \
      --n-images 48 --forget-frac 0.34 --seed 0 --label $arm --out $o \
      > outputs/log_$arm.txt 2>&1 || continue
  fi
  [ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
    --unet $o/unet.pt --n-per-prompt 5 --steps 25 --seed 0 --save-grid >> outputs/log_$arm.txt 2>&1
  for cfg in "300|1e-5" "300|5e-5" "150|1e-5"; do
    IFS='|' read -r st lr <<< "$cfg"
    lab="${arm}_s${st}_lr${lr}"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept church \
      --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
  rm -f $o/unet.pt     # measurements kept; 1.7 GiB checkpoint not
  echo "[undo-sweep] $arm done" >> outputs/log_undo_progress.txt
done
echo "[undo-sweep] shard complete"
