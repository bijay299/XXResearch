#!/bin/bash
# DECISIVE TEST. Parachute's cap rose 0.533 -> 0.667 when the prior weight dropped to 0.2, so the
# PRIOR-PRESERVATION TERM is what was holding the attack back -- not AdvUnlearn resisting. At
# prior-weight 0 DreamBooth degenerates to full fine-tuning with an identifier prompt, which
# reaches 1.000 on concept-word prompts. If pw=0 reaches ~1.000 here, the "DreamBooth cap" is an
# artifact of the attacker handicapping itself, and the resistance claim must NOT be made.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TEC=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
TEP=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1 6; do
    L=H73Z_db_advu_church_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church --text-encoder "$TEC" \
      --steps 300 --lr 1e-5 --seed $SD --prior-weight 0.0 --label "$L" >>logs/h73z_a.log 2>&1
  done
  echo "Z A DONE $(date -Is)" >>logs/h73z_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1 6; do
    L=H73Z_db_advu_parachute_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept parachute --text-encoder "$TEP" \
      --steps 300 --lr 1e-5 --seed $SD --prior-weight 0.0 --label "$L" >>logs/h73z_b.log 2>&1
  done
  echo "Z B DONE $(date -Is)" >>logs/h73z_b.log ) &
wait
