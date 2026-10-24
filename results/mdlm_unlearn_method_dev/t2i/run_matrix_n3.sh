#!/bin/bash
# Bring the matrix cells that sit at n=2 up to n=3 so the report's table has uniform seed coverage.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TEC=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
TEP=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
( export CUDA_VISIBLE_DEVICES=0
  L=H74N_ti_h68rf_parachute_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
    --unet adapters/h68v2_rf_a0.3_parachute/unet.pt --steps 1000 --lr 5e-3 --seed 6 --label "$L" \
    >>logs/h74n_a.log 2>&1
  L=H74N_db_advu_church_pw0_sd2
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py \
    --concept church --text-encoder "$TEC" --steps 300 --lr 1e-5 --seed 2 --prior-weight 0.0 \
    --label "$L" >>logs/h74n_a.log 2>&1
  echo "N3 A DONE $(date -Is)" >>logs/h74n_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  L=H74N_db_h68rf_church_pw0_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py \
    --concept church --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-5 --seed 6 \
    --prior-weight 0.0 --label "$L" >>logs/h74n_b.log 2>&1
  L=H74N_db_frp_parachute_pw0_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py \
    --concept parachute --unet adapters/frp_parachute_r5_l05/unet.pt --steps 300 --lr 1e-5 \
    --seed 6 --prior-weight 0.0 --label "$L" >>logs/h74n_b.log 2>&1
  echo "N3 B DONE $(date -Is)" >>logs/h74n_b.log ) &
wait
