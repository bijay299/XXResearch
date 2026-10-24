#!/bin/bash
# AdvUnlearn's DreamBooth cap held at 0.667 across 8 valid attacks and 4 varied hyperparameters --
# the first sub-ceiling number this phase that survived pushing. Second concept before any claim.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TEP=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
TEV=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/vangogh_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1 6; do
    L=H73V_db_advu_parachute_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept parachute --text-encoder "$TEP" \
      --steps 300 --lr 1e-5 --seed $SD --label "$L" >>logs/h73v_a.log 2>&1
  done
  echo "DBV A DONE $(date -Is)" >>logs/h73v_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H73V_db_advu_vangogh_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept van_gogh --text-encoder "$TEV" \
      --steps 300 --lr 1e-5 --seed $SD --label "$L" >>logs/h73v_b.log 2>&1
  done
  # and the damage+distil arm under DreamBooth, for arm coverage
  for SD in 0 1; do
    L=H73V_db_h68rf_church_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-5 --seed $SD --label "$L" \
      >>logs/h73v_b.log 2>&1
  done
  echo "DBV B DONE $(date -Is)" >>logs/h73v_b.log ) &
wait
