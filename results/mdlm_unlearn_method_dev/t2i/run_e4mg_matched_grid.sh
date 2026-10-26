#!/bin/bash
# E4mg: the frozen grid for the SAMPLE-MATCHED H90 series (PREREG_H90 S3 + S8).
# Arms: utrap2_church_l0.15k0 (400 outer, shared) + utrap2m_church_l0.15k{1,8,32}.
# 3 seeds x lr 1e-5 x 300 steps, no escalation. Arms failing K1 are not attacked.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e4mg_grid.log
for A in utrap2_church_l0.15k0 utrap2m_church_l0.15k1 utrap2m_church_l0.15k8 utrap2m_church_l0.15k32; do
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    [ "$FREE" -lt 5 ] && { echo "ABORT $A sd=$SD: ${FREE}GiB free" >>"$L"; break 2; }
    K="E4MG_${A}_sd${SD}_lr1e-5_s300"
    [ -f "outputs/atk_${K}.json" ] && continue
    echo "=== $K $(date -Is)" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps 300 --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
echo "E4MG DONE $(date -Is)" >>"$L"
