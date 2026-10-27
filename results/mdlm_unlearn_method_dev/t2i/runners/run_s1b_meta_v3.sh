#!/bin/bash
# S1b -- the three-term objective with a variance-correct estimator.
# Protocol: docs/PREREG_S1b_META_V3.md   Defect it corrects: docs/PREREG_S1_AMENDMENT_1_2026-10-27.md s6
#
# WHY: S1's meta gradient was noise-dominated because the difference [L_R(theta')-L_R(theta)] was
# estimated on DIFFERENT random draws. Arm 2's measured mean parameter drift was 2.27e-5 -- matching
# a random walk (3.87e-5) and 65x below coherent motion (1.5e-3) -- so S1 cannot answer H1.
# meta_unlearn_v3.py fixes this with common random numbers plus upstream's fixed meta-term
# timesteps {1,2,5,10} (src/meta.py:595, scripts/meta.sh), which our port had silently omitted.
#
# STAGE 0 is a PILOT THAT CAN FALSIFY OUR OWN DIAGNOSIS, and it runs first:
#   if sharing the draw is what matters, retain_gap must become SMALL and CONSISTENTLY SIGNED.
#   v2 measured: -0.402, +0.027, +0.179, +0.346, -0.195 -- magnitude ~0.2-0.4, sign flipping.
#   FALSIFIER: if v3's gap still swings ~0.1+ with sign flips, section 6's diagnosis is WRONG,
#   it must be withdrawn, and NO build is paid for. The script stops after the pilot in that case
#   only by our reading of it -- the gate is a human decision, so the pilot exits here by design
#   and the builds are launched by a separate invocation with --build.
#
# The no-op control is NOT rebuilt: amendment s1.2 proved s1_metaV2_z0.0_g0.0 byte-identical to its
# teacher (max abs delta 0 over 859,520,964 params). The paired control for the s5.1 criterion is
# the TEACHER's own attack cells, tabulated in amendment s1.4.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
BUILD=0
[ "${1:-}" = "--build" ] && BUILD=1
source scripts/gpu_reserve.sh
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
gpu_reserve_hold "s1b_meta_v3" 240 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null || {
  echo "S1b ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "S1b ABORT: no CVD" >&2; exit 1; }
echo "S1b reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/s1b_meta_v3.log
source scripts/run_guard.sh
rg_init "S1b" "$L"
T=adapters/frp_church_r3_l05/unet.pt
[ -s "$T" ] || { rg_abort "teacher missing at $T"; exit 1; }

if [ "$BUILD" = "0" ]; then
  # ---- STAGE 0: pilot. 100 steps each, two estimator variants, no checkpoint kept for science.
  #      variant A: fixed timesteps {1,2,5,10}  (both corrections)
  #      variant B: random timestep per pair    (correction 1 only -- isolates the two fixes)
  for SPEC in "fixed 1,2,5,10" "randpair "; do
    set -- $SPEC; TAG=$1; TS=${2:-}
    K="s1b_pilot_${TAG}"
    [ -f "outputs/metaV3_${K}.json" ] && { echo "   skip: $K" >>"$L"; continue; }
    echo "=== $K $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/meta_unlearn_v3.py \
      --teacher "$T" --concept church --steps 100 --lr 1e-6 --seed 0 \
      --zeta 0.8 --gamma-f 0.0 --fixed-timesteps "$TS" --log-every 10 \
      --label "$K" --out "adapters/_pilot_${K}"
    rg_require_loadable "outputs/metaV3_${K}.json" json
  done
  echo "S1b PILOT DONE $(date -Is) -- read retain_gap and meta_grad_norm, then rerun with --build" >>"$L"
  rg_finish
  exit 0
fi

# ---- STAGE 1: builds. g0.0 first; g1.0 only if g0.0 clears gate 1 (prereg s6 default).
for CFG in "0.8 0.0" "0.8 1.0"; do
  set -- $CFG; Z=$1; GF=$2
  A="s1b_metaV3_z${Z}_g${GF}"; D="adapters/$A"
  if [ "$GF" != "0.0" ]; then
    R8=$(venv_t2i/bin/python -c "
import json
try: print(json.load(open('outputs/gen_s1b_metaV3_z0.8_g0.0.json'))['active_target_metrics']['legacy_retain8'])
except Exception: print(0)")
    OK=$(venv_t2i/bin/python -c "print(1 if float('$R8')>=0.85 else 0)")
    [ "$OK" = "1" ] || { echo "   skip $A: g0.0 retain8=$R8 failed gate 1" >>"$L"; continue; }
  fi
  if [ ! -s "$D/unet.pt" ]; then
    used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
    free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
      rg_abort "space guard: project ${used}GiB vs 500GiB authorization, ${free}GiB free (shared volume)"; break
    fi
    echo "=== build $A $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/meta_unlearn_v3.py \
      --teacher "$T" --concept church --steps 1500 --lr 1e-6 --seed 0 \
      --zeta "$Z" --gamma-f "$GF" --fixed-timesteps "1,2,5,10" --adam-eps 1e-12 \
      --label "$A" --out "$D"
    rg_require_loadable "outputs/metaV3_${A}.json" json
  fi
  [ -s "$D/unet.pt" ] || continue
  if [ ! -f "outputs/gen_${A}.json" ]; then
    echo "=== gen_eval $A $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "$D/unet.pt" --target church
    rg_require_loadable "outputs/gen_${A}.json" json
  fi
  # ---- attacks only at the two rates where the TEACHER control has a clean cell at all 3 seeds
  R8=$(venv_t2i/bin/python -c "
import json
try: print(json.load(open('outputs/gen_${A}.json'))['active_target_metrics']['legacy_retain8'])
except Exception: print(0)")
  OK=$(venv_t2i/bin/python -c "print(1 if float('$R8')>=0.85 else 0)")
  [ "$OK" = "1" ] || { echo "   skip attacks for $A: retain8=$R8 failed gate 1" >>"$L"; continue; }
  for LR in 1e-5 5e-5; do
    for SD in 0 1 6; do
      K="S1bEV_${A}_sd${SD}_lr${LR}_s25"
      [ -f "outputs/atk_${K}.json" ] && continue
      echo "=== $K $(date -Is)" >>"$L"
      rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept church \
        --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
      rg_require_loadable "outputs/atk_${K}.json" json
    done
  done
done
rg_finish
echo "S1b DONE $(date -Is)" >>"$L"
