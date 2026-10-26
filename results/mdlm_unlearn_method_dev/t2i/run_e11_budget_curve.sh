#!/bin/bash
# E11: recovery-versus-budget curve on the mandate-qualifying arm.
#
# A NAMED AUDIT GAP (audit S8): "Use time-to-threshold, recovery-versus-budget curves, area under a
# fixed-budget curve... These are different estimands from final maximum recovery", and "observing
# success at 900 rather than 300 steps does not establish a minimal 3x cost multiplier. Intermediate
# budgets and matched baseline curves are needed." This project has only ever measured 300 (and 900
# for LoRA, 2000-6000 for TI) -- never the interior.
#
# Why it is runnable at 3.5 GiB free: attack_relearn.py writes a TRANSIENT fp16 checkpoint
# (line 383) and deletes it at line 414 unless --keep-checkpoint. No resident checkpoint, no new
# adapter. Serial, one at a time, with a hard pre-run disk guard.
#
# Arm: utrap_church_l0.15k8 -- target 0.0000 / retain8 0.8667, the only SD-1.4 arm that cleared the
# mandate's retain floor, and already measured at 300 steps (H = 1.000, T +1.000).
# Budgets: 25, 50, 100, 200. Seed 0, lr 1e-5 -- identical to the measured 300-step point, so the
# curve is comparable to it by construction.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11_budget.log
A=utrap_church_l0.15k8
for S in 25 50 100 200; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  if [ "$FREE" -lt 3 ]; then
    echo "ABORT before steps=$S: ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; break
  fi
  K="E11_${A}_sd0_lr1e-5_s${S}"
  echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed 0 --lr 1e-5 --steps "$S" \
    --label "$K" >>"$L" 2>&1
  rm -rf "adapters/_atk_$K" 2>/dev/null   # belt-and-braces; the script already removes it
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, os
rows=[]
for f in sorted(glob.glob("outputs/atk_E11_utrap_church_l0.15k8_*.json")):
    d=json.load(open(f)); hp=d["hparams"]; av=d["attack_validity"]; rc=d["recovery"]
    rows.append((hp["steps"], av["T_before"], av["T_after"], av["valid"], rc["H_before"], rc["H_after"]))
f300="outputs/atk_H84_utrap_church_l0.15k8_sd0_lr1e-5.json"
for g in glob.glob("outputs/atk_*utrap_church_l0.15k8*sd0*lr1e-5*.json"):
    if "E11" in g: continue
    d=json.load(open(g))
    if d["hparams"]["steps"]==300:
        av=d["attack_validity"]; rc=d["recovery"]
        rows.append((300, av["T_before"], av["T_after"], av["valid"], rc["H_before"], rc["H_after"])); break
rows.sort()
print("\n=== E11 recovery-vs-budget, utrap_church_l0.15k8, seed 0, lr 1e-5 ===")
print(f"{'steps':>6} {'T_before':>9} {'T_after':>8} {'gate':>6} {'H_before':>9} {'H_after':>8}")
for s,tb,ta,v,hb,ha in rows:
    print(f"{s:>6} {tb:>9.4f} {ta:>8.4f} {'PASS' if v else 'fail':>6} {hb:>9.4f} {ha:>8.4f}")
json.dump({"study":"recovery_vs_budget","arm":"utrap_church_l0.15k8","seed":0,"lr":1e-5,
  "rows":[{"steps":s,"T_before":tb,"T_after":ta,"gate_valid":v,"H_before":hb,"H_after":ha}
          for s,tb,ta,v,hb,ha in rows]},
  open("outputs/e11_budget_curve.json","w"), indent=1)
PY
echo "E11 DONE $(date -Is)" >>"$L"
