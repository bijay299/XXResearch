#!/bin/bash
# E4: PREREG_H90's arms -- the first test of unroll depth with an inner attacker that IS the
# evaluation attacker (ERRATUM_utrap_inner_attacker.md). Unblocked by the disk decision.
#
# Matched compute, fixed in PREREG_H90 S2a BEFORE any arm existed: outer steps chosen so total
# UNet forward-backwards is held near-constant (400-429, within 7.3%).
#   K=0  ABLATION_K0            400 outer    (the control that was missing)
#   K=1  SHORT_HORIZON_SURROGATE 200 outer
#   K=8  SHORT_HORIZON_SURROGATE  45 outer
#   K=32 SHORT_HORIZON_SURROGATE  13 outer
# Plus the S7 amendment, made before any arm existed: one lambda=1.2 K=8 arm, because
# diag_coeff_scale.py measured lambda parity at 1.23 against the inherited 0.15 (~8x under-weighted).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0          # GPU 1 is running E2
L=logs/e4_h90.log
T=adapters/frp_church_r3_l05/unet.pt
build () {
  LAM=$1; K=$2; OUTER=$3; A="utrap2_church_l${LAM}k${K}"
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$FREE" -lt 8 ] && { echo "ABORT $A: ${FREE}GiB free" >>"$L"; return 1; }
  [ -f "adapters/$A/unet.pt" ] && { echo "$A exists, skipping build" >>"$L"; return 0; }
  echo "=== $A (lam=$LAM K=$K outer=$OUTER) $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/erase_utrap_v2.py --concept church --teacher "$T" \
    --lam "$LAM" --inner-k "$K" --steps "$OUTER" --n-images 16 --n-retain 32 \
    --log-every 25 --probe-every 50 --label "$A" --out "adapters/$A" >>"$L" 2>&1
  [ -f "adapters/$A/unet.pt" ] || { echo "$A BUILD FAILED -- not a kill-rule result" >>"$L"; return 1; }
  venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "adapters/$A/unet.pt" \
    --target church >>"$L" 2>&1
  venv_t2i/bin/python - <<PY >>"$L" 2>&1
import json
am=json.load(open("outputs/gen_$A.json"))["active_target_metrics"]
u=json.load(open("outputs/utrap2_$A.json"))
r,t=am["legacy_retain8"],am["target_suppression_acc"]
pr=u.get("inner_attacker_effectiveness_probe") or []
d=[p["delta"] for p in pr]
print(f"[E4] $A as-is: target={t:.4f} retain8={r:.4f} allnt={am['utility_all_nontarget']:.4f}")
print(f"[E4] inner attacker role={u['inner_attacker']['spec']['horizon_role']} "
      f"divergence={json.dumps(u['inner_attacker']['divergence_from_evaluation_attacker'])}")
print(f"[E4] probe delta over {len(d)} samples: mean={sum(d)/max(len(d),1):+.5f} "
      f"min={min(d) if d else float('nan'):+.5f}")
print(f"[E4] K1: {'CLEAR -> frozen grid' if r>=0.60 else 'FAIL -> not attacked'}")
PY
}
build 0.15 0  400
build 0.15 1  200
build 0.15 8  45
build 0.15 32 13
build 1.2  8  45
echo "E4 BUILDS DONE $(date -Is)" >>"$L"
