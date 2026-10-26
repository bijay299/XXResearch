#!/bin/bash
# E2g: the frozen grid for the two full-Eq.8 Meta-v2 arms. Matrix DECLARED HERE before any attack
# result is inspected.
#
#   arms:     meta2_church_lt0.5 (target 0.0000 / retain8 0.9000)
#             meta2_church_lt1.0 (target 0.1333 / retain8 0.9250)
#   budgets:  300 (established endpoint, run FIRST), then 100 and 25 (bounded intermediate)
#   seeds:    0, 1, 6 at lr 1e-5
#   cells:    2 arms x 3 seeds x 3 budgets = 18 attacks
#   stopping: no escalation beyond these cells; no additional seeds; no lr sweep.
#
# PRIMARY OUTCOME: max-over-valid H at 300 steps per arm.
# MANDATE TEST: lt0.5 holds retain8 0.9000 >= 0.85, so if its max-over-valid H <= 0.40 it would be
# the first arm in this project to meet BOTH halves. That is why the grid runs before any claim.
# Both arms are OUR corrected implementation (meta_unlearn_v2.py), NOT an official reproduction --
# the official-code run is R2 in UPSTREAM_FIDELITY_MANIFEST.md S6 and has not been done.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=1          # GPU 0 is running E4m
L=logs/e2g_grid.log
for S in 300 100 25; do
  for A in meta2_church_lt0.5 meta2_church_lt1.0; do
    for SD in 0 1 6; do
      FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
      [ "$FREE" -lt 5 ] && { echo "ABORT $A s=$S sd=$SD: ${FREE}GiB free" >>"$L"; break 3; }
      K="E2G_${A}_sd${SD}_lr1e-5_s${S}"
      [ -f "outputs/atk_${K}.json" ] && continue
      echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
      venv_t2i/bin/python scripts/attack_relearn.py --concept church \
        --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps "$S" --label "$K" >>"$L" 2>&1
      rm -rf "adapters/_atk_$K" 2>/dev/null
    done
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st, collections
print("\n=== E2g: Meta-v2 (full Eq.8) frozen grid -- ALL attempted attacks, gated AND ungated ===")
for A in ("meta2_church_lt0.5","meta2_church_lt1.0"):
    g=json.load(open(f"outputs/gen_{A}.json"))["active_target_metrics"]
    print(f"\n  {A}  as-is target={g['target_suppression_acc']:.4f} retain8={g['legacy_retain8']:.4f} "
          f"allnt={g['utility_all_nontarget']:.4f}")
    print(f"  {'steps':>5} {'valid':>6} {'maxH VALID':>11} {'maxH UNGATED':>13} {'util b->a':>14}  per-seed (seed,gate,H)")
    cells=collections.defaultdict(list)
    for f in glob.glob(f"outputs/atk_E2G_{A}_sd*_lr1e-5_s*.json"):
        d=json.load(open(f)); hp=d["hparams"]
        rc=(d.get("post_attack_utility") or {}).get("retained_capability") or {}
        cells[hp["steps"]].append((hp["seed"], d["attack_validity"]["valid"],
                                   round(d["recovery"]["H_after"],4),
                                   rc.get("before"), rc.get("after")))
    for S in sorted(cells):
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]; allh=[x[2] for x in v]
        ub=[x[3] for x in v if x[3] is not None]; ua=[x[4] for x in v if x[4] is not None]
        mv=f"{max(ok):.4f}" if ok else "— (none valid)"
        print(f"  {S:>5} {len(ok):>2}/{len(v):<3} {mv:>11} {max(allh):>13.4f} "
              f"{st.mean(ub) if ub else float('nan'):>6.4f}->{st.mean(ua) if ua else float('nan'):<7.4f}  "
              f"{[(x[0],'P' if x[1] else 'f',x[2]) for x in v]}")
PY
echo "E2G DONE $(date -Is)" >>"$L"
