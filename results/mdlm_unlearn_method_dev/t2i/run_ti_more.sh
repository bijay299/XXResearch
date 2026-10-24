#!/bin/bash
# The ranking INVERTED between adversaries (LoRA: FRP 1.000 > AdvU 0.867; TI: AdvU 1.000 > FRP
# 0.667). Before any claim: a third seed and a SECOND CONCEPT on both arms.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TEC=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
TEP=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
( export CUDA_VISIBLE_DEVICES=0
  L=H72_ti_advunlearn_church_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
    --text-encoder "$TEC" --steps 1000 --lr 5e-3 --seed 6 --label "$L" >>logs/h72m_a.log 2>&1
  for SD in 0 1; do
    L=H72_ti_advunlearn_parachute_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute --text-encoder "$TEP" \
      --steps 1000 --lr 5e-3 --seed $SD --label "$L" >>logs/h72m_a.log 2>&1
  done
  echo "TIM A DONE $(date -Is)" >>logs/h72m_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  L=H72_ti_frp_church_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
    --unet adapters/frp_church_r3_l05/unet.pt --steps 1000 --lr 5e-3 --seed 6 --label "$L" \
    >>logs/h72m_b.log 2>&1
  for SD in 0 1; do
    L=H72_ti_frp_parachute_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
      --unet adapters/frp_parachute_r5_l05/unet.pt --steps 1000 --lr 5e-3 --seed $SD --label "$L" \
      >>logs/h72m_b.log 2>&1
  done
  echo "TIM B DONE $(date -Is)" >>logs/h72m_b.log ) &
wait
