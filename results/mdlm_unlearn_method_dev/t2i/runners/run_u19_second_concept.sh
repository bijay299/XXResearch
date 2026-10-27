#!/bin/bash
# U19 -- does the attacker-calibration finding GENERALISE beyond church?
#
# The deck's most prominent scope limit is "one main concept". Its central methodological claim is
# that apparent robustness tracked attacker tuning rather than defence strength: on church, the same
# checkpoint reads H 0.333 at attacker lr 1e-5 and 1.000 at 5e-5 on an identical 25-step budget, and
# the attacker collapses entirely at 2e-4 (0/3 gate-valid, H 0.000). If that pattern is specific to
# church it is a curiosity; if it reproduces on a second concept it is a property of this evaluation
# and bears on anyone measuring relearning robustness at a fixed learning rate.
#
# No new defence training. Two EXISTING parachute arms are swept, chosen to bracket erasure depth
# because on church the depth of erasure determined how misleading a fixed learning rate was:
#     frp_parachute_r3_l05   target 0.6000  retain8 0.9333   (shallow erasure, high retain)
#     frp_parachute_r5_l05   target 0.2000  retain8 0.7417   (deeper erasure, lower retain)
#
# DECLARED BEFORE EXECUTION
#   lrs    : 1e-5, 5e-5, 1e-4, 2e-4   (the church grid, including the point where the attacker died)
#   steps  : 25, held fixed so only attacker strength varies
#   seeds  : 0, 1, 6
#   cells  : 2 arms x 4 lrs x 3 seeds = 24
#
# READING: per-seed points with gate flags; the robustness figure is the maximum over the sweep. A
# gate failure means the attacker could not relearn its own training prompts and the cell is
# INCONCLUSIVE, never evidence of defence.
#
# FALSIFIER: if both parachute arms show a flat response across attacker lr, with no effectiveness
# peak and no collapse at 2e-4, then the calibration finding is church-specific and the deck's
# methodological claim must be narrowed to that concept.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
source scripts/gpu_reserve.sh
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
gpu_reserve_hold "u19_second_concept" 240 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null || {
  echo "U19 ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U19 ABORT: no CVD" >&2; exit 1; }
echo "U19 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/u19_second_concept.log
source scripts/run_guard.sh
rg_init "U19" "$L"
for SPEC in "p_r3 adapters/frp_parachute_r3_l05" "p_r5 adapters/frp_parachute_r5_l05"; do
  set -- $SPEC; TAG=$1; D=$2
  [ -s "$D/unet.pt" ] || { rg_abort "$TAG missing at $D"; continue; }
  for LR in 1e-5 5e-5 1e-4 2e-4; do
    for SD in 0 1 6; do
      K="U19EV_${TAG}_sd${SD}_lr${LR}_s25"
      [ -f "outputs/atk_${K}.json" ] && { echo "   skip: $K" >>"$L"; continue; }
      used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
      free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
      if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
        rg_abort "space guard: project ${used}GiB vs 500GiB authorization, ${free}GiB free on a shared volume"
        break 3
      fi
      echo "=== $K $(date -Is)" >>"$L"
      rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept parachute \
        --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
      rg_require_loadable "outputs/atk_${K}.json" json
      rg_metric "outputs/atk_${K}.json" "recovery.H_after" 0 1 >/dev/null
    done
  done
done
rg_finish
