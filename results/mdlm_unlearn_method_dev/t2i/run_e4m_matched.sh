#!/bin/bash
# E4m: the SAMPLE-MATCHED H90 series, per PREREG_H90 S8. The first build set used K+1 accounting;
# true cost is K+2 passes / K+4 samples per outer step, so those arms span 1600->468 samples,
# DECREASING with K -- biased against the hypothesis. Corrected outer steps: K=0 400 (unchanged,
# shared), K=1 320, K=8 133, K=32 44. The unmatched arms are KEPT, not deleted.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e4m_matched.log
T=adapters/frp_church_r3_l05/unet.pt
for cfg in "1 320" "8 133" "32 44"; do
  set -- $cfg; K=$1; OUTER=$2; A="utrap2m_church_l0.15k${K}"
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$FREE" -lt 8 ] && { echo "ABORT $A: ${FREE}GiB free" >>"$L"; break; }
  [ -f "adapters/$A/unet.pt" ] && { echo "$A exists" >>"$L"; continue; }
  echo "=== $A K=$K outer=$OUTER (samples=$((OUTER*(K+4)))) $(date -Is)" >>"$L"
  venv_t2i/bin/python scripts/erase_utrap_v2.py --concept church --teacher "$T" \
    --lam 0.15 --inner-k "$K" --steps "$OUTER" --n-images 16 --n-retain 32 \
    --log-every 25 --probe-every 50 --label "$A" --out "adapters/$A" >>"$L" 2>&1
  [ -f "adapters/$A/unet.pt" ] || { echo "$A BUILD FAILED" >>"$L"; continue; }
  venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "adapters/$A/unet.pt" \
    --target church >>"$L" 2>&1
  venv_t2i/bin/python - <<PY >>"$L" 2>&1
import json
am=json.load(open("outputs/gen_$A.json"))["active_target_metrics"]
u=json.load(open("outputs/utrap2_$A.json"))
d=[p["delta"] for p in (u.get("inner_attacker_effectiveness_probe") or [])]
print(f"[E4m] $A as-is: target={am['target_suppression_acc']:.4f} retain8={am['legacy_retain8']:.4f} "
      f"allnt={am['utility_all_nontarget']:.4f} | probe delta mean={sum(d)/max(len(d),1):+.5f} n={len(d)}")
PY
done
echo "E4m BUILDS DONE $(date -Is)" >>"$L"
