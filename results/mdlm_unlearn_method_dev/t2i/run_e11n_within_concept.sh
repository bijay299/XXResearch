#!/bin/bash
# E11n: is "garbage_truck is the fastest concept" a CONCEPT property or an ARM property?
#
# FINDING73 S8a named this as the main remaining confound: "each concept still has exactly one arm,
# so 'concept' and 'this particular build' remain inseparable", and said separating them needs new
# builds. It does not -- a SECOND well-suppressed garbage_truck arm already exists on disk:
#   frp_garbage_truck_r5_l05  target 0.0000  retain8 0.7750  -> saturates at 25 steps (E11j)
#   frp_garbage_truck_r3_l05  target 0.1333  retain8 0.9667  -> THIS RUN
# Same concept, same FRP family, different suppression depth and much higher retention.
#
# PREDICTION, recorded before the run: if the fast recovery is a CONCEPT property, this arm should
# also saturate by ~25-50 steps. If instead it behaves like tench/church (needing ~100), then
# "garbage_truck is fast" was a property of the r5_l05 BUILD, and FINDING73 S8's concept framing
# must be narrowed from concepts to arms.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11n_within_concept.log
A=frp_garbage_truck_r3_l05
for S in 25 50; do
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    [ "$FREE" -lt 3 ] && { echo "ABORT s=$S sd=$SD: ${FREE}GiB free" >>"$L"; break 2; }
    K="E11N_${A}_sd${SD}_lr1e-5_s${S}"
    echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept garbage_truck \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps "$S" --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st, collections
print("\n=== E11n: TWO garbage_truck arms, same concept and family, lr 1e-5, n=3 ===")
for arm,tgt,ret in (("frp_garbage_truck_r5_l05","0.0000","0.7750"),
                    ("frp_garbage_truck_r3_l05","0.1333","0.9667")):
    cells=collections.defaultdict(set)
    for f in glob.glob(f"outputs/atk_E11*_{arm}_sd*_lr1e-5_s*.json"):
        d=json.load(open(f)); hp=d["hparams"]
        rc=(d.get("post_attack_utility") or {}).get("retained_capability") or {}
        cells[hp["steps"]].add((hp["seed"], d["attack_validity"]["valid"],
                                round(d["recovery"]["H_after"],4), round(rc.get("after",-1),4)))
    print(f"\n  {arm}  (as-is target {tgt}, retain8 {ret})")
    print(f"  {'steps':>6} {'valid':>7} {'meanH':>7} {'maxH':>7} {'util':>7}  per-seed H")
    for S in sorted(cells):
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]; u=[x[3] for x in v if x[3]>=0]
        if not ok:
            print(f"  {S:>6} {0:>3}/{len(v):<3}   all gate-invalid"); continue
        print(f"  {S:>6} {len(ok):>3}/{len(v):<3} {st.mean(ok):>7.4f} {max(ok):>7.4f} "
              f"{st.mean(u) if u else float('nan'):>7.4f}  {[x[2] for x in v]}")
PY
echo "E11N DONE $(date -Is)" >>"$L"
