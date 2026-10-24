#!/bin/bash
# Parachute reads 0.533 under DreamBooth on 3 BASELINE-config attacks. Church's 0.667 only became
# credible after 8 attacks with lr/steps/prior-weight varied. Match that before the cap is claimed.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TEP=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
TEV=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/vangogh_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for cfg in "600 1e-5 0 1.0" "300 5e-5 0 1.0" "300 1e-5 0 0.2"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3; PW=$4
    L=H73PP_db_advu_parachute_s${ST}_lr${LR}_sd${SD}_pw${PW}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept parachute --text-encoder "$TEP" \
      --steps $ST --lr $LR --seed $SD --prior-weight $PW --label "$L" >>logs/h73pp_a.log 2>&1
  done
  echo "PP A DONE $(date -Is)" >>logs/h73pp_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for cfg in "600 5e-5 1 0.5" "300 1e-4 0 1.0"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3; PW=$4
    L=H73PP_db_advu_parachute_s${ST}_lr${LR}_sd${SD}_pw${PW}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept parachute --text-encoder "$TEP" \
      --steps $ST --lr $LR --seed $SD --prior-weight $PW --label "$L" >>logs/h73pp_b.log 2>&1
  done
  L=H73PP_db_advu_vangogh_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py \
    --concept van_gogh --text-encoder "$TEV" --steps 300 --lr 1e-5 --seed 6 --label "$L" \
    >>logs/h73pp_b.log 2>&1
  echo "PP B DONE $(date -Is)" >>logs/h73pp_b.log ) &
wait
