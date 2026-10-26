#!/bin/bash
# E1b: the faithful WDR arm at the MEASURED alpha (4000), not the reference's inherited default.
# FINDING67: at alpha=1 the distance term's gradient norm is 1/sqrt(N) = 3.41e-5 against the retain
# term's 0.137, so alpha=1 is ~4000x too small on an 859.5M-parameter UNet.
# --max-rms 0.03 is ~3x the FRP teacher's own 0.009, so the trust region can bind before the model
# is destroyed; at alpha=4000 an unbounded ascent plausibly would (FINDING61 S4).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e1b_wdr.log
FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
echo "=== E1b start $(date -Is) free=${FREE}GiB" >>"$L"
[ "$FREE" -lt 3 ] && { echo "ABORT: ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; exit 0; }
A=wdr2_church_a4000
venv_t2i/bin/python scripts/erase_wdr_v2.py --concept church --alpha 4000 --max-rms 0.03 \
  --steps 1500 --n-retain 32 --log-every 100 --teacher adapters/frp_church_r3_l05/unet.pt \
  --label "$A" --out "adapters/$A" >>"$L" 2>&1
[ -f "adapters/$A/unet.pt" ] || { echo "$A BUILD FAILED -- not a kill-rule result" >>"$L"; exit 0; }
venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "adapters/$A/unet.pt" \
  --target church >>"$L" 2>&1
venv_t2i/bin/python - <<PY >>"$L" 2>&1
import json
am=json.load(open("outputs/gen_$A.json"))["active_target_metrics"]
w=json.load(open("outputs/wdr2_$A.json"))
r,t=am["legacy_retain8"],am["target_suppression_acc"]
print(f"[E1b] $A as-is: target={t:.4f} retain8={r:.4f} allnt={am['utility_all_nontarget']:.4f}")
print(f"[E1b] trust_region_stop: {w['trust_region_stop']}")
print(f"[E1b] final RMS: {w['history'][-1]['rms_distance']:.6f} (teacher started at 0.009024)")
print(f"[E1b] K1: {'CLEAR' if r>=0.60 else 'FAIL -> not attacked'} | vacuity: {'VACUOUS' if t>=0.95 else 'erased something'}")
PY
echo "E1b DONE $(date -Is)" >>"$L"
