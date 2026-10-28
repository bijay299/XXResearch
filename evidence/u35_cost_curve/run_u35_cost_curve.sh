#!/bin/bash
# U35 -- THE ATTACKER COST CURVE.  Pre-registered in docs/PREREG_U35_COST_CURVE.md.
#
# WHY. Every attack in this project runs at a FIXED 25 steps and reports a binary outcome. Under it
# every arm reads 1.0000, and U33 has now shown capacity does not separate them either -- a rank-4
# LoRA holding 0.09% of the UNet fully reverses UCE (9/9 cells). We have NEVER measured how many
# steps recovery takes. If one method falls at step 2 and another holds to step 20, the instrument
# cannot see it: both read 1.0000 at step 25.
#
# WHY EFFORT IS THE PRINCIPLED AXIS. A church is a configuration of primitives the retained set also
# needs, so erasure cannot remove the bulk of the capability. The attacker is not relearning a
# concept, it is closing a small residual gap -- so the question is not WHETHER it closes but HOW
# FAST. That is also the quantity the mandate accepts: an explicitly measured trade-off.
#
# THE GATE IS DIFFERENT HERE, DELIBERATELY. At 25 steps, T_rise < 0.20 means the attack did not take
# and the cell is inconclusive. At a budget of 1 or 2 steps, T_rise < 0.20 IS the measurement -- the
# attacker genuinely has not succeeded yet. Excluding those cells would discard exactly the data
# this experiment exists to collect. So for U35 ONLY the T_rise gate is not applied; T_rise is
# recorded beside every cell, and the DESTROYED-cell rule still applies in full. This relaxation
# makes recovery look EARLIER AND CHEAPER -- it makes every method look worse, not better.
#
# RATES ARE FIXED PER ARM from the existing 25-step data, chosen to favour the ATTACKER (highest
# rate among those achieving the max). Declared in the prereg before any cell here was run.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
source scripts/gpu_reserve.sh
gpu_reserve_hold "u35_cost_curve" 300 >/dev/null || { echo "U35 ABORT: no reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U35 ABORT: no CVD" >&2; exit 1; }
echo "U35 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/u35_cost_curve.log
source scripts/run_guard.sh
rg_init "U35" "$L"

# arm:lr  -- lr fixed per arm, see prereg s2
CELLS=(
  "uce_church:5e-5"
  "esd_church:1e-4"
  "gift_church_b1.0:1e-4"
  "imma_church_frp_a1e-7_s1500:5e-5"
  "u32_uceT_a0.3:5e-5"
)
for ENTRY in "${CELLS[@]}"; do
  ARM="${ENTRY%%:*}"; LR="${ENTRY##*:}"
  U="adapters/$ARM/unet.pt"
  [ -s "$U" ] || { echo "SKIP $ARM: no checkpoint" >>"$L"; continue; }
  for ST in 1 2 4 8 16; do
    for SD in 0 1 6; do
      K="U35_${ARM}_s${ST}_sd${SD}_lr${LR}"
      [ -f "outputs/atk_${K}.json" ] && continue
      echo "=== $K $(date -Is)" >>"$L"
      rg_cell venv_t2i/bin/python scripts/attack_relearn_v3.py --concept church \
        --unet "$U" --attack-mode full \
        --seed "$SD" --lr "$LR" --steps "$ST" --label "$K" --validate-reload
      rg_require_loadable "outputs/atk_${K}.json" json
    done
  done
  echo "=== ARM DONE $ARM (lr=$LR) $(date -Is)" >>"$L"
done
rg_finish
echo "U35 DONE $(date -Is)" >>"$L"
