#!/bin/bash
# S1c -- the LAST attempt at the trap mechanism. Protocol: docs/PREREG_S1b_META_V3.md s3.5.2
#
# WHY: v3 fixed the difference estimator (gap 0.190 -> 3e-6, confirmed) and eps (1e-12), yet the
# model's mean drift was 2.046e-5 -- 0.53x a pure random walk and no better than v2's broken
# 2.27e-5. The gradient magnitude was stable (5.9e-4 to 8.3e-4 over 1500 steps) but its DIRECTION
# was resampled each step, so the updates cancelled. Cause: batch size 1, where upstream ships
# --train_batch_size 10. v4 micro-batches the meta gradient over 8 samples per step.
#
# STOPPING RULE, declared before this ran (s3.5.2):
#   drift >= 1.4e-4 -> objective has purchase; read the panel under the s5.1 criterion
#   drift <= 3e-5   -> STOP. Three corrections have failed; report the direction untestable within
#                      our resources and do NOT attempt a fourth fix.
#   THRESHOLDS CORRECTED 2026-10-27 BEFORE any drift was measured: the originally declared bands
#   (2e-4 / 6e-5) were computed for a 1500-step run and never rescaled to 190 steps. At 190 steps
#   coherent motion is 190*lr = 1.9e-4 and a random walk is sqrt(190)*lr = 1.38e-5, so the old
#   purchase gate of 2e-4 EXCEEDED the fully-coherent value -- a perfect run would have failed it.
#   New bands: 10x random walk for purchase, 2x for stop. Arithmetic error, not a fit to data.
#   between        -> partial; report drift alongside the panel, treat the panel as provisional.
#
# 190 steps x 8 samples = 1520 samples, within 2% of the 1500 used by v1/v2/v3, and 15,200 backward
# passes against S1b's 15,000 -- cost-neutral.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
source scripts/gpu_reserve.sh
gpu_reserve_hold "s1c_meta_v4" 180 >/dev/null || { echo "S1c ABORT: no reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "S1c ABORT: no CVD" >&2; exit 1; }
echo "S1c reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/s1c_meta_v4.log
source scripts/run_guard.sh
rg_init "S1c" "$L"
T=adapters/frp_church_r3_l05/unet.pt
[ -s "$T" ] || { rg_abort "teacher missing at $T"; exit 1; }
A="s1c_metaV4_z0.8_g0.0"; D="adapters/$A"
if [ ! -s "$D/unet.pt" ]; then
  used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
  free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
    rg_abort "space guard: project ${used}GiB vs 500GiB authorization, ${free}GiB free (shared volume)"; exit 1
  fi
  echo "=== build $A $(date -Is)" >>"$L"
  rg_cell venv_t2i/bin/python scripts/meta_unlearn_v4.py \
    --teacher "$T" --concept church --steps 190 --lr 1e-6 --seed 0 \
    --zeta 0.8 --gamma-f 0.0 --fixed-timesteps "1,2,5,10" \
    --meta-batch 8 --adam-eps 1e-12 --log-every 10 \
    --label "$A" --out "$D"
  rg_require_loadable "outputs/metaV4_${A}.json" json
fi
[ -s "$D/unet.pt" ] || { rg_abort "no checkpoint written"; exit 1; }
# ---- the STOPPING RULE gate: drift decides whether the panel is even read
echo "=== drift check $A $(date -Is)" >>"$L"
venv_t2i/bin/python scripts/param_delta.py "$D/unet.pt" "$T" --expect different 2>&1 | grep -vE "FutureWarning|torch.load" >>"$L"
DRIFT=$(venv_t2i/bin/python scripts/param_delta.py "$D/unet.pt" "$T" 2>/dev/null | awk '/mean abs delta/{print $4}')
echo "   measured mean abs drift = $DRIFT  (190 steps: random walk 1.38e-5, coherent 1.9e-4)" >>"$L"
echo "   bands: stop <=3e-5, purchase >=1.4e-4" >>"$L"
GO=$(venv_t2i/bin/python -c "
d=float('$DRIFT')
print('STOP' if d<=3e-5 else ('GO' if d>=1.4e-4 else 'PARTIAL'))")
echo "   STOPPING RULE verdict: $GO" >>"$L"
if [ "$GO" = "STOP" ]; then
  echo "   drift in the stop band -- per s3.5.2 the direction is CLOSED as untestable." >>"$L"
  echo "   No gen_eval, no attacks, no fourth fix." >>"$L"
  rg_finish; echo "S1c DONE (STOP) $(date -Is)" >>"$L"; exit 0
fi
echo "=== gen_eval $A $(date -Is)" >>"$L"
rg_cell venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "$D/unet.pt" --target church
rg_require_loadable "outputs/gen_${A}.json" json
R8=$(venv_t2i/bin/python -c "
import json
try: print(json.load(open('outputs/gen_${A}.json'))['active_target_metrics']['legacy_retain8'])
except Exception: print(0)")
OK=$(venv_t2i/bin/python -c "print(1 if float('$R8')>=0.85 else 0)")
if [ "$OK" != "1" ]; then
  echo "   skip attacks: retain8=$R8 failed gate 1" >>"$L"; rg_finish
  echo "S1c DONE (gate 1 fail) $(date -Is)" >>"$L"; exit 0
fi
for LR in 1e-5 5e-5; do
  for SD in 0 1 6; do
    K="S1cEV_${A}_sd${SD}_lr${LR}_s25"
    [ -f "outputs/atk_${K}.json" ] && continue
    echo "=== $K $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept church \
      --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
    rg_require_loadable "outputs/atk_${K}.json" json
  done
done
rg_finish
echo "S1c DONE $(date -Is)" >>"$L"
