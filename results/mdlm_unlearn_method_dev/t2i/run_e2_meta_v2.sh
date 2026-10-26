#!/bin/bash
# E2: the FIRST build of meta_unlearn_v2.py -- Meta-Unlearning's FULL Eq. 8 objective.
#
# Until this runs, arXiv 2410.12777 is UNTESTED in this project: meta_unlearn.py (variant A) omits
# two of Eq. 8's three terms (ERRATUM_meta_objective.md), so FINDING56's "the trap fires on normal
# users" was withdrawn as a claim about the published method.
#
# Coefficients: zeta = lam_target = 1.0, the paper's values, kept because diag_coeff_scale.py
# measured them already NEAR PARITY on this model (||grad L_retain(theta)|| 0.0387 vs
# ||grad L_target(theta')|| 0.0927, ratio 2.4x) -- unlike WDR's alpha, which was 4000x off
# (FINDING67). Scale was measured BEFORE the run, not after.
# Scopes separated as the reference does: meta gradient on xattn, simulated attack on the full UNet.
# --target-source base so the target pool is BASE-model generations with a CLIP content check,
# instead of variant A's unverified already-erased-teacher pool.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=1          # GPU 0 is running E12; two GPUs at a time, within the cap
L=logs/e2_meta_v2.log
T=adapters/frp_church_r3_l05/unet.pt
for LAM in 1.0 0.5; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$FREE" -lt 8 ] && { echo "ABORT lam=$LAM: ${FREE}GiB free" >>"$L"; break; }
  A="meta2_church_lt${LAM}"
  echo "=== $A $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/meta_unlearn_v2.py --teacher "$T" --concept church \
    --steps 1500 --n-images 64 --log-every 250 --zeta 1.0 --lam-target "$LAM" \
    --meta-scope xattn --attack-scope full --target-source base --require-target-acc 0.5 \
    --label "$A" --out "adapters/$A" >>"$L" 2>&1
  [ -f "adapters/$A/unet.pt" ] || { echo "$A BUILD FAILED -- not a kill-rule result" >>"$L"; continue; }
  venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "adapters/$A/unet.pt" \
    --target church >>"$L" 2>&1
  venv_t2i/bin/python - <<PY >>"$L" 2>&1
import json
am=json.load(open("outputs/gen_$A.json"))["active_target_metrics"]
m=json.load(open("outputs/meta2_$A.json"))
r,t=am["legacy_retain8"],am["target_suppression_acc"]
print(f"[E2] $A as-is: target={t:.4f} retain8={r:.4f} allnt={am['utility_all_nontarget']:.4f}")
print(f"[E2] target-pool content check: {m['target_data_provenance']['target_classified_fraction']:.4f} "
      f"({m['target_data_provenance']['source']})")
print(f"[E2] scopes: meta={m['scopes']['meta']} ({m['scopes']['meta_n_params']/1e6:.1f}M) "
      f"attack={m['scopes']['attack']} ({m['scopes']['attack_n_params']/1e6:.1f}M)")
h=m["train_log"][-1]
print(f"[E2] final: L_F(theta')={h['l_target_theta_prime']:.5f} dL_R={h['retain_change']:+.5f}")
print(f"[E2] K1: {'CLEAR -> frozen grid' if r>=0.60 else 'FAIL -> not attacked'} | "
      f"vacuity: {'VACUOUS' if t>=0.95 else 'erased something'}")
PY
done
echo "E2 DONE $(date -Is)" >>"$L"
