#!/bin/bash
# E6b: does the H=1.000 ceiling hold on FINAL prompts, or is it a DEV artifact?
#
# Every H number in this project is DEV-prompt recovery. FINDING65 showed pre-attack suppression is
# ~3x shallower on held-out phrasings; this asks the converse question about RECOVERY.
#
# Uses the UNMODIFIED attack_relearn.py, so the adversary is bit-identical to all 642 historical
# records -- only --keep-checkpoint is added, then gen_eval_final.py scores FINAL on the attacked
# model and the 1.7 GiB checkpoint is deleted immediately. SERIAL, one GPU: /data has ~11 GiB free.
#
# Two arms, pre-committed: the SD-1.4 mandate-qualifying arm and the SD-1.5 best operating point --
# the two whose FINAL suppression held up in FINDING65, so a FINAL ceiling here is most informative.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e6c.log

run_one () {
  ARM=$1; FINALARM=$2; BM=$3; BREV=$4
  K="E6C_${ARM}_sd0_lr1e-5"
  echo "=== $K $(date -Is)" >>"$L"
  df -h /data | tail -1 >>"$L"
  ( export BASE_MODEL="$BM"; [ -n "$BREV" ] && export BASE_REV="$BREV"; export BASE_VARIANT=fp16
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$ARM/unet.pt" --seed 0 --lr 1e-5 --keep-checkpoint \
      --label "$K" >>"$L" 2>&1 )
  if [ -f "attacked/$K/unet.pt" ]; then
    venv_t2i/bin/python scripts/gen_eval_final.py --only "$FINALARM" >>"$L" 2>&1
    rm -rf "attacked/$K"          # 1.7 GiB reclaimed immediately; /data is at capacity
    echo "$K scored on FINAL, checkpoint deleted" >>"$L"
  else
    echo "$K NO KEPT CHECKPOINT -- FINAL not scored (not a result)" >>"$L"
  fi
}

echo "E6B DONE $(date -Is)" >>"$L"

# E6c: the two arms whose PRE-attack FINAL suppression was weakest (frp 0.4667, uce 0.6000).
# Deliberate EXTENSION of E6b, added after seeing its results -- labelled as such, not pre-committed.
# Leaving these out would have been selective: they are where E6b's conclusion is least surprising.
run_one frp_church_r3_l05       ATK_frp_church_r3_l05       CompVis/stable-diffusion-v1-4 ""
run_one uce_sd15_church_lam1000 ATK_uce_sd15_church_lam1000 runwayml/stable-diffusion-v1-5 451f4fe16113bff5a5d2269ed5ad43b0592e9a14
echo "E6C DONE $(date -Is)" >>"$L"
