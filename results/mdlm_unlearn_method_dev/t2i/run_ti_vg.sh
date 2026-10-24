#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1; do
    L=H79_ti_advu_vangogh_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept van_gogh \
      --text-encoder "/data/bijaypandey/t2i_unlearn/hf_home/hub/models--OPTML-Group--AdvUnlearn/snapshots/bca3b27922b1e70cfe5b858e35beccc2fbc3b5ee/vangogh_unlearned" --steps 1000 --lr 5e-3 --seed $SD --label "$L" \
      >>logs/h79_a.log 2>&1
  done
  echo "TIVG A DONE $(date -Is)" >>logs/h79_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  L=H79_ti_advu_vangogh_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept van_gogh \
    --text-encoder "/data/bijaypandey/t2i_unlearn/hf_home/hub/models--OPTML-Group--AdvUnlearn/snapshots/bca3b27922b1e70cfe5b858e35beccc2fbc3b5ee/vangogh_unlearned" --steps 1000 --lr 5e-3 --seed 6 --label "$L" \
    >>logs/h79_b.log 2>&1
  L=H79_ti_advu_parachute_sd6
  [ -f "outputs/atk_${L}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
    --text-encoder "/data/bijaypandey/t2i_unlearn/hf_home/hub/models--OPTML-Group--AdvUnlearn/snapshots/bca3b27922b1e70cfe5b858e35beccc2fbc3b5ee/parachute_unlearned" --steps 1000 --lr 5e-3 --seed 6 --label "$L" \
    >>logs/h79_b.log 2>&1
  echo "TIVG B DONE $(date -Is)" >>logs/h79_b.log ) &
wait
