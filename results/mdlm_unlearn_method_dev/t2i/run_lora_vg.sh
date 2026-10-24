#!/bin/bash
# Third concept (van_gogh, a STYLE rather than an object) for the LoRA grid.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/vangogh_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1 6; do
    L=H71V_advu_sd${SD}_lr1e-3_cross
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept van_gogh --text-encoder "$TE" \
      --steps 300 --lr 1e-3 --rank 16 --alpha 16 --scope cross --seed $SD --label "$L" \
      >>logs/vg_a.log 2>&1
  done
  echo "VG A DONE $(date -Is)" >>logs/vg_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H71V_advu_sd${SD}_lr1e-3_all_attn
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept van_gogh --text-encoder "$TE" \
      --steps 300 --lr 1e-3 --rank 16 --alpha 16 --scope all_attn --seed $SD --label "$L" \
      >>logs/vg_b.log 2>&1
  done
  echo "VG B DONE $(date -Is)" >>logs/vg_b.log ) &
wait
