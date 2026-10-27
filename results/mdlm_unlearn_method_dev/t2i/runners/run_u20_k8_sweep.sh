#!/bin/bash
# U20 -- complete the attacker sweep on the three K=8 builds. Protocol: docs/PREREG_U20_K8_SWEEP.md
#
# WHY: u4_k8_s400_b{0,1,2} are the ONLY arms in this project that retain usable capability
# (retain8 0.9583/0.9583/0.9792) while their max-over-valid H_after is below 1.000 (0.9333/0.9333/
# 0.8000). All nine of their existing cells sit at attacker lr 1e-5, which U18's curve shows is
# below the peak (K=0: 0.267 at 1e-5 vs 0.933 at 5e-5). MANDATE clause 1c makes a single-rate
# result a lower bound, so these arms are probably under-measured and the central negative result
# currently depends on that caveat rather than on measurement.
#
# This does NOT attempt to revive UTRAP: K=8 was retired on cost grounds that no outcome here
# changes. Attack only -- no defence training.
#
# FALSIFIER: if all three builds hold max-over-valid H_after below 1.000 on gate-valid,
# non-destroyed cells with retain8 still >= 0.85, that is the project's first genuine resistance
# and MANDATE v2 section 3.2's "no measurable benefit" wording must be revisited.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
source scripts/gpu_reserve.sh
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
gpu_reserve_hold "u20_k8_sweep" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null || {
  echo "U20 ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U20 ABORT: no CVD" >&2; exit 1; }
echo "U20 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/u20_k8_sweep.log
source scripts/run_guard.sh
rg_init "U20" "$L"
for B in 0 1 2; do
  D="adapters/u4_k8_s400_b${B}"
  [ -s "$D/unet.pt" ] || { rg_abort "b${B} missing at $D"; continue; }
  for LR in 5e-5 1e-4; do
    for SD in 0 1 6; do
      K="U20EV_k8_b${B}_sd${SD}_lr${LR}_s25"
      [ -f "outputs/atk_${K}.json" ] && { echo "   skip: $K" >>"$L"; continue; }
      used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
      free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
      if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
        rg_abort "space guard: project ${used}GiB vs 500GiB authorization, ${free}GiB free on a shared volume"
        break 3
      fi
      echo "=== $K $(date -Is)" >>"$L"
      rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept church \
        --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
      rg_require_loadable "outputs/atk_${K}.json" json
      rg_metric "outputs/atk_${K}.json" "recovery.H_after" 0 1 >/dev/null
    done
  done
done
rg_finish
echo "U20 DONE $(date -Is)" >>"$L"
