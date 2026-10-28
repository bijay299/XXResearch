#!/bin/bash
# U39 -- D4: is the ESD arm's erasure level a property of the method, or of one build seed?
#
# WHY THIS ARM. D4 is "n=1 is pervasive": attacks are replicated at n=3 but every arm is ONE BUILD.
# Five claims in this project have already died on seed replication. The arm carrying the most
# weight right now is esd_church: U34/U36/U37 use it for the headline that benign fine-tuning on
# golf balls and french horns completely undoes erasure, verified in images. That entire result is
# conditioned on a single build at seed 0, whose un-attacked suppression happens to be 0.4667.
#
# WHAT WOULD CHANGE. If rebuilding ESD at other seeds gives materially different suppression, then
# "ESD erases to 0.4667 and is fully recovered by benign fine-tuning" is a statement about one
# checkpoint, not about the method as we implement it -- and the benign result must be re-stated at
# whatever spread exists. If the builds agree, the headline is replicated at the BUILD level, which
# nothing in this project currently has.
#
# UCE needs no seed replication: it is a closed-form edit with no stochastic training.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
source scripts/gpu_reserve.sh
gpu_reserve_hold "u39_d4_esd" 240 >/dev/null || { echo "U39 ABORT: no reservation" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U39 ABORT: no CVD" >&2; exit 1; }
echo "U39 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/u39_d4_esd_seeds.log
source scripts/run_guard.sh
rg_init "U39" "$L"
for SD in 1 6; do
  ARM="esd_church_s${SD}"
  D="adapters/$ARM"
  if [ ! -s "$D/unet.pt" ]; then
    free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
    if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
      rg_abort "space guard: ${used}GiB of 500GiB cap, ${free}GiB free"; exit 1
    fi
    echo "=== build $ARM (ESD, church, 500 steps, lr 1e-5, seed $SD) $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/erase_esd.py --concept church \
      --steps 500 --lr 1e-5 --seed "$SD" --label "$ARM" --out "$D"
  fi
  [ -s "$D/unet.pt" ] || { echo "SKIP $ARM: no checkpoint" >>"$L"; continue; }
  if [ ! -f "outputs/gen_${ARM}.json" ]; then
    echo "=== gen_eval $ARM $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" --unet "$D/unet.pt" --target church
    rg_require_loadable "outputs/gen_${ARM}.json" json
  fi
done
rg_finish
echo "U39 DONE $(date -Is)" >>"$L"
