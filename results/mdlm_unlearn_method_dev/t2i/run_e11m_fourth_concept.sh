#!/bin/bash
# E11k: a THIRD concept. E11j showed garbage_truck saturating at 25 steps on all 3 seeds
# where FRP-church was still at 0.8667, so the cost correction is not church-specific. tench
# (frp_tench_r3_l05, target 0.1333 / retain8 0.7917) is the third point, same FRP family. "One concept (church)" is the limit
# named in FINDING68, 69, 70 AND 71 -- every one of them. frp_garbage_truck_r5_l05 is present
# (target 0.0000 / retain8 0.7750), fully suppressed like the church arms, so it is the right test.
#
# lr 1e-5 (the non-destructive rate, per FINDING71), steps {10, 25, 100}, seeds {0,1,6}. n=3.
# PREDICTION, recorded before the run: if the story generalises, H should rise monotonically with
# budget, reach the ceiling by ~100 steps, and leave utility in the 0.9+ band. If garbage_truck
# needs materially more budget than church did, then the <=100-step cost figure is concept-specific
# and FINDING68/70's cost claims must be narrowed to church.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11m_parachute.log
A=frp_parachute_r5_l05
for S in 25 50 100; do
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    [ "$FREE" -lt 3 ] && { echo "ABORT s=$S sd=$SD: ${FREE}GiB free" >>"$L"; break 2; }
    K="E11M_${A}_sd${SD}_lr1e-5_s${S}"
    echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept parachute \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps "$S" --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st, collections
print("\n=== E11j: garbage_truck (2nd concept) vs church, lr 1e-5, n=3 ===")
for arm,conc,hb in (("frp_parachute_r5_l05","parachute","0.2000"),
                    ("frp_tench_r3_l05","tench","0.1333"),
                    ("frp_garbage_truck_r5_l05","garbage_truck","0.0000"),
                    ("utrap_church_l0.15k8","church","0.0000"),
                    ("frp_church_r3_l05","church","0.1333")):
    cells=collections.defaultdict(set)
    for f in glob.glob(f"outputs/atk_E11*_{arm}_sd*_lr1e-5_s*.json"):
        d=json.load(open(f)); hp=d["hparams"]
        rc=(d.get("post_attack_utility") or {}).get("retained_capability") or {}
        cells[hp["steps"]].add((hp["seed"], d["attack_validity"]["valid"],
                                round(d["recovery"]["H_after"],4), round(rc.get("after",-1),4)))
    print(f"\n  {arm}  ({conc}, H_before={hb})")
    print(f"  {'steps':>6} {'valid':>7} {'meanH':>7} {'maxH':>7} {'util':>7}")
    for S in sorted(cells):
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]; u=[x[3] for x in v if x[3]>=0]
        if not ok: 
            print(f"  {S:>6} {0:>3}/{len(v):<3}   all gate-invalid"); continue
        print(f"  {S:>6} {len(ok):>3}/{len(v):<3} {st.mean(ok):>7.4f} {max(ok):>7.4f} "
              f"{st.mean(u) if u else float('nan'):>7.4f}")
PY
echo "E11M DONE $(date -Is)" >>"$L"
