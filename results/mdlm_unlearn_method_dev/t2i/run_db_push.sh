#!/bin/bash
# AdvUnlearn reads 0.667 under DreamBooth on 2 seeds. Five sub-ceiling numbers of mine have broken
# under a stronger adversary this week, so push before characterising: more seeds, more steps,
# higher lr, and lower prior weight (the prior term constrains the instance fit).
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for cfg in "300 1e-5 6 1.0" "600 1e-5 0 1.0" "300 5e-5 0 1.0"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3; PW=$4
    L=H73P_db_advu_church_s${ST}_lr${LR}_sd${SD}_pw${PW}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church --text-encoder "$TE" \
      --steps $ST --lr $LR --seed $SD --prior-weight $PW --label "$L" >>logs/h73p_a.log 2>&1
  done
  echo "DBP A DONE $(date -Is)" >>logs/h73p_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for cfg in "300 1e-5 0 0.2" "600 5e-5 1 0.5" "300 1e-4 0 1.0"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3; PW=$4
    L=H73P_db_advu_church_s${ST}_lr${LR}_sd${SD}_pw${PW}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church --text-encoder "$TE" \
      --steps $ST --lr $LR --seed $SD --prior-weight $PW --label "$L" >>logs/h73p_b.log 2>&1
  done
  echo "DBP B DONE $(date -Is)" >>logs/h73p_b.log ) &
wait
