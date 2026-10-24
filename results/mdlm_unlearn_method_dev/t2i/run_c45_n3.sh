#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/bijaypandey/hf_home
B=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)
for C in garbage_truck tench; do
  L=H76N_ti_advu_${C}_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept $C \
    --text-encoder "$B/${C}_unlearned" --steps 1000 --lr 5e-3 --seed 6 --label "$L" \
    >>logs/h76n.log 2>&1
  L=H76N_db_advu_${C}_pw0_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py --concept $C \
    --text-encoder "$B/${C}_unlearned" --steps 300 --lr 1e-5 --seed 6 --prior-weight 0.0 \
    --measure-utility --label "$L" >>logs/h76n.log 2>&1
done
echo "C45 N3 DONE $(date -Is)" >>logs/h76n.log
