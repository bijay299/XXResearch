#!/bin/bash
# The LoRA attack FAILED ITS VALIDITY GATE on official AdvUnlearn (T 0.333 -> 0.208). Per FINDING20
# a gate failure is NOT evidence of robustness -- it means the adversary did not relearn even the
# prompts it was given. Before any claim, strengthen it along every axis: higher lr, higher rank,
# wider scope (attn1+attn2).
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/attack_lora.py --concept church --text-encoder "$TE" \
    --steps 300 --lr 1e-3 --rank 16 --alpha 16 --scope all_attn --seed 0 \
    --label H71_lora_advunlearn_church_r16_all_lr1e3 >>logs/h71s_a.log 2>&1
  venv_t2i/bin/python scripts/attack_lora.py --concept church --text-encoder "$TE" \
    --steps 600 --lr 5e-3 --rank 32 --alpha 32 --scope all_attn --seed 0 \
    --label H71_lora_advunlearn_church_r32_all_lr5e3_s600 >>logs/h71s_a.log 2>&1
  echo "STRONG A DONE $(date -Is)" >>logs/h71s_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/attack_lora.py --concept church --text-encoder "$TE" \
    --steps 300 --lr 1e-3 --rank 16 --alpha 16 --scope cross --seed 0 \
    --label H71_lora_advunlearn_church_r16_cross_lr1e3 >>logs/h71s_b.log 2>&1
  # and the FRP arm at the ORIGINAL weak config on a second seed, to check its gate-pass replicates
  venv_t2i/bin/python scripts/attack_lora.py --concept church \
    --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-4 --rank 4 --alpha 4 \
    --scope cross --seed 1 --label H71_lora_frp_church_r4_cross_sd1 >>logs/h71s_b.log 2>&1
  echo "STRONG B DONE $(date -Is)" >>logs/h71s_b.log ) &
wait
