#!/bin/bash
# FINDING48 S4's largest gap: a SECOND CONCEPT. Attack the AdvUnlearn and damage+distil parachute
# arms with --keep-checkpoint, then sweep for the utility-matched comparison.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
  L=KEEP_advunlearn_parachute_sd0_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept parachute --text-encoder "$TE" \
    --steps 300 --lr 1e-5 --seed 0 --n-images 16 --keep-checkpoint --label "$L" \
    >>logs/para_advu.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked "attacked/$L/unet.pt" --concept parachute --state attacked_advunlearn \
    --modes random --fracs 0.10,0.25,0.30,0.40 >>logs/para_advu.log 2>&1
  echo "PARA ADVU DONE $(date -Is)" >>logs/para_advu.log ) &
( export CUDA_VISIBLE_DEVICES=1
  L=KEEP_h68rf_parachute_sd0_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept parachute \
    --unet adapters/h68v2_rf_a0.3_parachute/unet.pt --steps 300 --lr 1e-5 --seed 0 \
    --n-images 16 --keep-checkpoint --label "$L" >>logs/para_h68rf.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_parachute \
    --attacked "attacked/$L/unet.pt" --concept parachute --state attacked_h68rf \
    --modes random --fracs 0.10,0.17,0.25,0.35 >>logs/para_h68rf.log 2>&1
  echo "PARA H68RF DONE $(date -Is)" >>logs/para_h68rf.log ) &
wait
