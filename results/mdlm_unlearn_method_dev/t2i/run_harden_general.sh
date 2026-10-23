#!/bin/bash
# Does FINDING46's entrenchment generalise beyond the FRP-attacked arm? Two other families:
#   GPU 0: the damage+distil arm (h68v2_rf_a0.3_church) -- a UNet-space erasure
#   GPU 1: official AdvUnlearn church -- a TEXT-ENCODER erasure, attacked UNet-only
# Both need --keep-checkpoint, which only exists as of this session.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  L=KEEP_h68rf_church_sd0_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-5 --seed 0 --n-images 16 \
    --keep-checkpoint --label "$L" >>logs/harden_h68rf.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
    --attacked "attacked/$L/unet.pt" --concept church --state attacked_h68rf \
    >>logs/harden_h68rf.log 2>&1
  echo "HARDEN H68RF DONE $(date -Is)" >>logs/harden_h68rf.log ) &
( export CUDA_VISIBLE_DEVICES=1
  TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
  L=KEEP_advunlearn_church_sd0_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --text-encoder "$TE" --steps 300 --lr 1e-5 --seed 0 --n-images 16 \
    --keep-checkpoint --label "$L" >>logs/harden_advu.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked "attacked/$L/unet.pt" --concept church --state attacked_advunlearn \
    >>logs/harden_advu.log 2>&1
  echo "HARDEN ADVU DONE $(date -Is)" >>logs/harden_advu.log ) &
wait
echo "HARDEN BOTH DONE $(date -Is)" >>logs/harden_chain.log
