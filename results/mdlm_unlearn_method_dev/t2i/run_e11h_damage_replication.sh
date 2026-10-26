#!/bin/bash
# E11h: does FINDING70 S6b's DAMAGE TRANSITION reproduce on a second arm?
#
# FINDING70 S6b claims that past ~10 steps at lr 1e-4 the attack stops relearning and starts
# destroying the model (H falls 0.9555 -> 0.6222 while post-attack utility collapses
# 0.8472 -> 0.5278). That rests on ONE arm. A single-arm mechanism claim is exactly what I have
# retracted four times in this project, so it gets tested on a second arm before it is relied on.
#
# frp_church_r3_l05, lr 1e-4, steps {10, 25}, seeds {0,1,6}. n=3 from the start.
# PREDICTION, recorded before the run: if the mechanism is general, the 25-step cell should show
# BOTH a lower mean H than the 10-step cell AND a collapse in post-attack utility. If utility holds
# up while H still falls, the damage explanation is wrong and something else is going on. If neither
# moves, the non-monotonicity is specific to utrap_church_l0.15k8 and S6b must be narrowed to it.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11h_damage.log
A=frp_church_r3_l05
for S in 10 25; do
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    if [ "$FREE" -lt 3 ]; then echo "ABORT s=$S sd=$SD: ${FREE}GiB free" >>"$L"; break 2; fi
    K="E11H_${A}_sd${SD}_lr1e-4_s${S}"
    echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-4 --steps "$S" --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st, collections
print("\n=== E11h: does the damage transition reproduce on frp_church_r3_l05? ===")
for arm,tag in (("frp_church_r3_l05","E11H"),("utrap_church_l0.15k8","E11G")):
    cells=collections.defaultdict(list)
    for f in glob.glob(f"outputs/atk_{tag}_{arm}_sd*_lr1e-4_s*.json"):
        d=json.load(open(f)); hp=d["hparams"]
        rc=(d.get("post_attack_utility") or {}).get("retained_capability") or {}
        cells[hp["steps"]].append((hp["seed"], d["attack_validity"]["valid"],
                                   round(d["recovery"]["H_after"],4),
                                   round(rc.get("after",float("nan")),4)))
    print(f"\n  {arm} @ lr 1e-4")
    print(f"  {'steps':>6} {'valid':>7} {'meanH':>7} {'maxH':>7} {'util_after':>11}  per-seed (H, util)")
    for S in sorted(cells):
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]
        u=[x[3] for x in v if x[3]==x[3]]
        print(f"  {S:>6} {len(ok):>3}/{len(v):<3} {st.mean(ok) if ok else float('nan'):>7.4f} "
              f"{max(ok) if ok else float('nan'):>7.4f} {st.mean(u) if u else float('nan'):>11.4f}  "
              f"{[(x[2],x[3]) for x in v]}")
PY
echo "E11H DONE $(date -Is)" >>"$L"
