#!/bin/bash
# E11b: the same recovery-vs-budget curve on a DIFFERENT method, to test whether E11's <=100-step
# threshold is method-specific. frp_church_r3_l05: target 0.1333 / retain8 0.9417, so its curve
# starts from a higher H_before than UTRAP's 0.0000 -- that difference is itself informative.
# Transient checkpoint only; hard disk guard; serial.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11b_budget.log
A=frp_church_r3_l05
for S in 25 50 100 200; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  if [ "$FREE" -lt 3 ]; then
    echo "ABORT before steps=$S: ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; break
  fi
  K="E11B_${A}_sd0_lr1e-5_s${S}"
  echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed 0 --lr 1e-5 --steps "$S" --label "$K" >>"$L" 2>&1
  rm -rf "adapters/_atk_$K" 2>/dev/null
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob
rows=[]
for f in sorted(glob.glob("outputs/atk_E11B_frp_church_r3_l05_*.json")):
    d=json.load(open(f)); hp=d["hparams"]; av=d["attack_validity"]; rc=d["recovery"]
    rows.append((hp["steps"], av["T_after"], av["valid"], rc["H_before"], rc["H_after"]))
for g in glob.glob("outputs/atk_*frp_church_r3_l05*sd0*lr1e-5*.json"):
    if "E11B" in g: continue
    d=json.load(open(g))
    if d["hparams"]["steps"]==300:
        av=d["attack_validity"]; rc=d["recovery"]
        rows.append((300, av["T_after"], av["valid"], rc["H_before"], rc["H_after"])); break
rows.sort()
print("\n=== E11b recovery-vs-budget, frp_church_r3_l05, seed 0, lr 1e-5 ===")
print(f"{'steps':>6} {'T_after':>8} {'gate':>6} {'H_before':>9} {'H_after':>8}")
for s,ta,v,hb,ha in rows: print(f"{s:>6} {ta:>8.4f} {'PASS' if v else 'fail':>6} {hb:>9.4f} {ha:>8.4f}")
json.dump({"study":"recovery_vs_budget","arm":"frp_church_r3_l05","seed":0,"lr":1e-5,
  "rows":[{"steps":s,"T_after":ta,"gate_valid":v,"H_before":hb,"H_after":ha} for s,ta,v,hb,ha in rows]},
  open("outputs/e11b_budget_curve.json","w"), indent=1)
PY
echo "E11B DONE $(date -Is)" >>"$L"
