#!/bin/bash
# FINDING54 built the four-adversary matrix (full-UNet relearning, LoRA, Textual Inversion,
# DreamBooth) for three defences. GIFT now has full-UNet (6/6 -> 1.000) and DreamBooth (2/2 ->
# 1.000); LoRA and TI are missing. Adding them makes GIFT the fourth defence with a complete matrix
# and tests the two adversary families that behaved differently elsewhere in this phase -- TI in
# particular, which could not be made valid at all on the Meta-Unlearning arm (FINDING56 S5a).
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
mkdir -p .labels logs
A=gift_church_b1.0
for SD in 0 1; do
  K="H81L_${A}_sd${SD}"
  mkdir ".labels/$K" 2>/dev/null || continue
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_lora.py \
    --concept church --unet "adapters/$A/unet.pt" --seed "$SD" --label "$K" >>logs/gift_adv.log 2>&1
done
for cfg in "2000 1e-2 0" "3000 1e-2 1"; do
  set -- $cfg; ST=$1; LR=$2; SD=$3
  K="H81T_${A}_s${ST}_lr${LR}_sd${SD}"
  mkdir ".labels/$K" 2>/dev/null || continue
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py \
    --concept church --unet "adapters/$A/unet.pt" --steps "$ST" --lr "$LR" --seed "$SD" \
    --label "$K" >>logs/gift_adv.log 2>&1
done
echo "GIFT ADV DONE $(date -Is)" >>logs/gift_adv.log
