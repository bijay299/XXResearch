#!/bin/bash
# E1: build and evaluate ONE arm of the FAITHFUL Weight Dist Reg adaptation (erase_wdr_v2.py),
# the audit's top-priority pending experiment. Until this runs, 'From Dormant to Deleted' is
# UNTESTED in this project -- erase_wsr.py implemented the opposite direction
# (ERRATUM_wsr_direction.md).
#
# ONE arm only (alpha=1.0, the reference's --unlearning-alpha default), because /data has ~6.7 GiB
# free and a checkpoint is 1.7 GiB. The 3-arm alpha sweep in the register stays blocked.
#
# --max-rms 0.05: calibrated from the smoke run, which showed the FRP teacher ALREADY sits at
# RMS 0.009 from base. A bound below that would stop at step 0 (it did, at 0.0005). This is the
# 'measure the scale before sweeping the knob' lesson from ERRATUM_uce_lambda_scale.md applied
# in advance for once.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e1_wdr.log
FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
echo "=== E1 start $(date -Is) free=${FREE}GiB" >>"$L"
if [ "$FREE" -lt 4 ]; then
  echo "ABORT: only ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; exit 0
fi
A=wdr2_church_a1.0
venv_t2i/bin/python scripts/erase_wdr_v2.py --concept church --alpha 1.0 --max-rms 0.05 \
  --steps 1500 --n-retain 32 --log-every 100 --teacher adapters/frp_church_r3_l05/unet.pt \
  --label "$A" --out "adapters/$A" >>"$L" 2>&1
if [ ! -f "adapters/$A/unet.pt" ]; then
  echo "$A BUILD FAILED -- not a kill-rule result; see $L" >>"$L"; exit 0
fi
venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "adapters/$A/unet.pt" \
  --target church >>"$L" 2>&1
venv_t2i/bin/python - <<PY >>"$L" 2>&1
import json
am=json.load(open("outputs/gen_$A.json"))["active_target_metrics"]
r,t=am["legacy_retain8"],am["target_suppression_acc"]
print(f"[E1] $A as-is: target={t:.4f} retain8={r:.4f} allnt={am['utility_all_nontarget']:.4f}")
print(f"[E1] K1 (retain8>=0.60): {'CLEAR -> eligible for the frozen grid' if r>=0.60 else 'FAIL -> not attacked'}")
print(f"[E1] vacuity (target>=0.95 means no erasure): {'VACUOUS' if t>=0.95 else 'erased something'}")
PY
echo "E1 DONE $(date -Is)" >>"$L"
