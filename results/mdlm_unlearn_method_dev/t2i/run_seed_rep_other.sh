#!/bin/bash
# FINDING48 S4 gap: attack-seed replication for the NON-FRP arms (only FRP has 3 seeds).
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
  L=KEEP_advunlearn_church_sd1_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept church --text-encoder "$TE" \
    --steps 300 --lr 1e-5 --seed 1 --n-images 16 --keep-checkpoint --label "$L" \
    >>logs/seedrep_advu.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked "attacked/$L/unet.pt" --concept church --state advunlearn_sd1 \
    --modes random --fracs 0.25,0.27,0.30 >>logs/seedrep_advu.log 2>&1
  echo "SEEDREP ADVU DONE $(date -Is)" >>logs/seedrep_advu.log ) &
( export CUDA_VISIBLE_DEVICES=1
  L=KEEP_h68rf_church_sd1_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-5 --seed 1 \
    --n-images 16 --keep-checkpoint --label "$L" >>logs/seedrep_h68rf.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
    --attacked "attacked/$L/unet.pt" --concept church --state h68rf_sd1 \
    --modes random --fracs 0.15,0.17,0.20 >>logs/seedrep_h68rf.log 2>&1
  echo "SEEDREP H68RF DONE $(date -Is)" >>logs/seedrep_h68rf.log ) &
wait
