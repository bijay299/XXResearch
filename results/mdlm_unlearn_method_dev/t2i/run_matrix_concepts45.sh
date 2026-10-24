#!/bin/bash
# Extend the adversary matrix to the two remaining AdvUnlearn-released concepts (garbage_truck,
# tench), under the two cheapest adversaries. Tests whether the matrix's uniformity is a property
# of the three concepts chosen or of the method.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
B=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1; do
    L=H76_ti_advu_garbage_truck_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept garbage_truck \
      --text-encoder "$B/garbage_truck_unlearned" --steps 1000 --lr 5e-3 --seed $SD --label "$L" \
      >>logs/h76_a.log 2>&1
  done
  for SD in 0 1; do
    L=H76_ti_advu_tench_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept tench \
      --text-encoder "$B/tench_unlearned" --steps 1000 --lr 5e-3 --seed $SD --label "$L" \
      >>logs/h76_a.log 2>&1
  done
  echo "C45 A DONE $(date -Is)" >>logs/h76_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H76_db_advu_garbage_truck_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept garbage_truck \
      --text-encoder "$B/garbage_truck_unlearned" --steps 300 --lr 1e-5 --seed $SD \
      --prior-weight 0.0 --measure-utility --label "$L" >>logs/h76_b.log 2>&1
  done
  for SD in 0 1; do
    L=H76_db_advu_tench_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept tench \
      --text-encoder "$B/tench_unlearned" --steps 300 --lr 1e-5 --seed $SD \
      --prior-weight 0.0 --measure-utility --label "$L" >>logs/h76_b.log 2>&1
  done
  echo "C45 B DONE $(date -Is)" >>logs/h76_b.log ) &
wait
