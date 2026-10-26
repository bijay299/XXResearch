#!/bin/bash
# E11o: complete frp_church_r3_l05 at 50 steps to n=3 -- which TESTS, AND MAY REFUTE, my own
# alignment hypothesis from FINDING73 S8b.
#
# That hypothesis: base prompt alignment orders INVERSELY with recovery speed
#   garbage_truck align 0.2456 -> 25 steps
#   parachute     align 0.2702 -> 50 steps
#   tench         align 0.2899 -> 100 steps
# church has the LOWEST alignment of all four (0.2406), so the hypothesis predicts it should be the
# FASTEST -- saturating at or below garbage_truck's 25 steps. But FRP-church at 25 steps is only
# 0.8667 mean / 0.9333 max (n=3, E11i), i.e. NOT saturated. If church needs 50 or more, the monotone
# relation BREAKS at its own extreme and the hypothesis is refuted on four points.
# Church's 50-step cell is currently n=1. This completes it.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11o_church_sat.log
A=frp_church_r3_l05
for SD in 1 6; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$FREE" -lt 3 ] && { echo "ABORT sd=$SD: ${FREE}GiB free" >>"$L"; break; }
  K="E11O_${A}_sd${SD}_lr1e-5_s50"
  echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps 50 --label "$K" >>"$L" 2>&1
  rm -rf "adapters/_atk_$K" 2>/dev/null
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st, collections
ALIGN={"church":0.2406,"garbage_truck":0.2456,"parachute":0.2702,"tench":0.2899}
print("\n=== E11o: alignment hypothesis on FOUR points ===")
arms=[("frp_church_r3_l05","church"),("frp_garbage_truck_r5_l05","garbage_truck"),
      ("frp_parachute_r5_l05","parachute"),("frp_tench_r3_l05","tench")]
print(f"  {'concept':<14} {'base align':>11} {'sat (all-3=1.0)':>16}   per-step mean H")
for arm,conc in arms:
    cells=collections.defaultdict(set)
    for f in glob.glob(f"outputs/atk_E11*_{arm}_sd*_lr1e-5_s*.json"):
        d=json.load(open(f)); hp=d["hparams"]
        cells[hp["steps"]].add((hp["seed"], d["attack_validity"]["valid"],
                                round(d["recovery"]["H_after"],4)))
    sat=None; detail=[]
    for S in sorted(cells):
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]
        if not ok: continue
        detail.append(f"{S}:{st.mean(ok):.3f}(n{len(ok)})")
        if sat is None and len(ok)>=3 and min(ok)==1.0: sat=S
    print(f"  {conc:<14} {ALIGN[conc]:>11.4f} {str(sat) if sat else '>'+str(max(cells)):>16}   {' '.join(detail)}")
PY
echo "E11O DONE $(date -Is)" >>"$L"
