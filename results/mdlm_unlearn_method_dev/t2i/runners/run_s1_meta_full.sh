#!/bin/bash
# S1 -- the full three-term meta objective. Preregistered in docs/PREREG_S1_META_FULL_OBJECTIVE.md.
# Stage S1 of MANDATE_v2_2026-10-27.md. Decision rule and stopping criteria are in the prereg.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
source scripts/gpu_reserve.sh
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
gpu_reserve_hold "s1_meta_full" 300 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null || {
  echo "S1 ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "S1 ABORT: no CVD" >&2; exit 1; }
echo "S1 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/s1_meta_full.log
T=adapters/frp_church_r3_l05/unet.pt
source scripts/run_guard.sh
rg_init "S1" "$L"
rg_require_file "$T"

space_ok() {
  local used free
  used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
  free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "${used:-0}" -lt 500 ] && [ "${free:-0}" -ge 25 ] && return 0
  rg_abort "space guard: ${used}GiB vs 500GiB authorization, ${free}GiB free"; return 1
}

for CFG in "0.0 0.0" "0.8 0.0" "0.8 1.0"; do
  set -- $CFG; Z=$1; GF=$2
  A="s1_metaV2_z${Z}_g${GF}"; D="adapters/$A"
  if [ ! -s "$D/unet.pt" ]; then
    space_ok "build $A" || break
    echo "=== build $A zeta=$Z gamma_f=$GF $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/meta_unlearn_v2.py --teacher "$T" --concept church \
      --steps 1500 --lr 1e-6 --seed 0 --inner-alpha 1e-5 --zeta "$Z" --gamma-f "$GF" \
      --n-images 64 --log-every 100 --label "$A" --out "$D"
    rg_require_loadable "$D/unet.pt" torch
  fi
  if [ ! -f "outputs/gen_${A}.json" ]; then
    space_ok "gen_eval $A" || break
    rg_cell venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "$D/unet.pt" --target church
    rg_require_loadable "outputs/gen_${A}.json" json
    rg_metric "outputs/gen_${A}.json" "active_target_metrics.legacy_retain8" 0 1 >/dev/null
  fi
done

# attacks only on arms whose utility SURVIVED normal use -- the prereg's first gate. An arm whose
# trap fired un-attacked has already failed and is not worth attacking.
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json,os
print("[S1] gate 1 -- did the trap fire on NORMAL use? (retain8 >= 0.85 required to proceed)")
for z,gf in (("0.0","0.0"),("0.8","0.0"),("0.8","1.0")):
    a="s1_metaV2_z%s_g%s"%(z,gf); f="outputs/gen_%s.json"%a
    if not os.path.exists(f): print("   %-24s (no panel)"%a); continue
    am=json.load(open(f))["active_target_metrics"]
    ok=am["legacy_retain8"]>=0.85
    print("   %-24s target=%.4f retain8=%.4f allnt=%.4f  -> %s"%(
        a,am["target_suppression_acc"],am["legacy_retain8"],am["utility_all_nontarget"],
        "PROCEED to attack" if ok else "FAILED gate 1 (trap fires un-attacked)"))
print("   v1 reference at zeta 0.8 (one-term objective): retain8 0.4750")
PY

for CFG in "0.0 0.0" "0.8 0.0" "0.8 1.0"; do
  set -- $CFG; Z=$1; GF=$2
  A="s1_metaV2_z${Z}_g${GF}"; D="adapters/$A"
  [ -s "$D/unet.pt" ] || continue
  R8=$(venv_t2i/bin/python -c "
import json,sys
try: print(json.load(open('outputs/gen_${A}.json'))['active_target_metrics']['legacy_retain8'])
except Exception: print(0)")
  AWK=$(venv_t2i/bin/python -c "print(1 if float('$R8')>=0.85 else 0)")
  if [ "$AWK" != "1" ]; then
    echo "   skip attacks for $A: retain8=$R8 failed gate 1" >>"$L"; continue
  fi
  for LR in 1e-5 5e-5 1e-4; do
    for SD in 0 1 6; do
      K="S1EV_${A}_sd${SD}_lr${LR}_s25"
      [ -f "outputs/atk_${K}.json" ] && continue
      space_ok "$K" || break 3
      echo "=== $K $(date -Is)" >>"$L"
      rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept church \
        --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
      rg_require_loadable "outputs/atk_${K}.json" json
    done
  done
done
rg_finish
