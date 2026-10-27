#!/bin/bash
# U21 -- bring the published baselines to the SAME attacker effort as the teacher.
#
# WHY: at the 25-step budget the FRAG/FRP teacher is swept across 7 learning rates under the
# evidence schema, while esd_church and imma_church_frp have only lr 1e-5 (legacy schema) and
# uce_church and gift_church_b1.0 have NO 25-step church cells at all. MANDATE v2 s5 requires
# comparable attacker-search effort before baselines are compared, and clause 1c makes a
# single-rate figure a lower bound rather than a robustness estimate. Any write-up that ranks these
# methods against each other needs this closed first.
#
# Grid: lr {1e-5, 5e-5, 1e-4} x seeds {0,1,6}, 25 steps, evidence schema with --validate-reload.
# 1e-5 is re-run for the two arms that have it in legacy form: same (lr, seed) under the new schema
# doubles as a cross-schema reproducibility check (cf. amendment s1.5, where H_after reproduced to
# within one quantum).
#
# Attack only -- no defence training. $1 = space-separated arm list (lets two GPUs split the work).
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
ARMS="${1:?usage: run_u21_baselines.sh \"arm1 arm2\"}"
TAG="${2:-u21}"
source scripts/gpu_reserve.sh
gpu_reserve_hold "${TAG}_baselines" 180 >/dev/null || { echo "U21 ABORT: no reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U21 ABORT: no CVD" >&2; exit 1; }
echo "U21[$TAG] reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/${TAG}_baselines.log
source scripts/run_guard.sh
rg_init "U21_$TAG" "$L"
for A in $ARMS; do
  D="adapters/$A"
  [ -s "$D/unet.pt" ] || { echo "   skip $A: no checkpoint" >>"$L"; continue; }
  for LR in 1e-5 5e-5 1e-4; do
    for SD in 0 1 6; do
      K="U21EV_${A}_sd${SD}_lr${LR}_s25"
      [ -f "outputs/atk_${K}.json" ] && { echo "   skip: $K" >>"$L"; continue; }
      used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
      free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
      if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
        rg_abort "space guard: project ${used}GiB vs 500GiB authorization, ${free}GiB free (shared volume)"
        break 3
      fi
      echo "=== $K $(date -Is)" >>"$L"
      rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept church \
        --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
      rg_require_loadable "outputs/atk_${K}.json" json
    done
  done
done
rg_finish
echo "U21[$TAG] DONE $(date -Is)" >>"$L"
