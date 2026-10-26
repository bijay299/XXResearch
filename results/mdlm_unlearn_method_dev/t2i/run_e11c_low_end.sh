#!/bin/bash
# E11c: the LOW END, which has never been explored. Both E11 curves are already >=0.87 at the
# smallest budget tried (25 steps), so the real floor of the attacker's cost is unknown.
# steps in {5, 10, 15} on both arms. Transient checkpoints only; hard disk guard; serial.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11c_lowend.log
for A in utrap_church_l0.15k8 frp_church_r3_l05; do
  for S in 5 10 15; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    if [ "$FREE" -lt 3 ]; then
      echo "ABORT $A steps=$S: ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; break 2
    fi
    K="E11C_${A}_sd0_lr1e-5_s${S}"
    echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed 0 --lr 1e-5 --steps "$S" --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob
print("\n=== E11c low end + E11/E11b, seed 0, lr 1e-5 ===")
for arm in ("utrap_church_l0.15k8","frp_church_r3_l05"):
    rows={}
    for pat in (f"outputs/atk_E11C_{arm}_*.json", f"outputs/atk_E11_{arm}_*.json",
                f"outputs/atk_E11B_{arm}_*.json"):
        for f in glob.glob(pat):
            d=json.load(open(f)); rows[d["hparams"]["steps"]]=(
                d["attack_validity"]["T_after"], d["attack_validity"]["valid"],
                d["recovery"]["H_before"], d["recovery"]["H_after"])
    print(f"\n  {arm}")
    print(f"  {'steps':>6} {'T_after':>8} {'gate':>6} {'H_before':>9} {'H_after':>8}")
    for s in sorted(rows):
        ta,v,hb,ha=rows[s]
        print(f"  {s:>6} {ta:>8.4f} {'PASS' if v else 'fail':>6} {hb:>9.4f} {ha:>8.4f}")
    json.dump({"study":"recovery_vs_budget_full","arm":arm,"seed":0,"lr":1e-5,
      "rows":[{"steps":s,"T_after":rows[s][0],"gate_valid":rows[s][1],
               "H_before":rows[s][2],"H_after":rows[s][3]} for s in sorted(rows)]},
      open(f"outputs/e11_full_curve_{arm}.json","w"), indent=1)
PY
echo "E11C DONE $(date -Is)" >>"$L"
