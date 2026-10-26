#!/bin/bash
# E12: does the <=100-step cost correction hold across DEFENCE FAMILIES, not just concepts?
#
# FINDING68-73 curved only FRP and UTRAP. A registry query (the lesson from FINDING73 S9c) found 36
# church arms across SEVEN families already on disk. This curves the best-suppressed K1-clearing arm
# from each remaining family on SD-1.4 church, so the cost claim is tested across methods -- a
# stronger generalisation than concepts, because it asks whether ANY of these methods buys the
# attacker meaningfully more budget.
#
#   GIFT port          gift_church_b1.35           tgt 0.2000  retain8 0.6000
#   IMMA port          imma_church_frp_a1e-7_s1500 tgt 0.1333  retain8 0.9417
#   Meta variant A     h80_meta_church_lam0.7      tgt 0.0000  retain8 0.7000
#   WSR-anchor (ours)  wsr2_church_lr100           tgt 0.0667  retain8 0.9500
# (FRP and UTRAP already curved; UCE has no SD-1.4 church arm on disk.)
#
# PREDICTION, recorded before the run: if the correction generalises, every family reaches
# max-over-valid H = 1.000 by 100 steps, and none holds below 0.40 at 25. If any family stays
# materially below the ceiling at 100 steps, that family is the first in this project to buy the
# attacker real budget, and FINDING68/70's cost claims must be qualified by method.
#
# steps {25, 100} x seeds {0,1,6} x 4 arms = 24 attacks, serial, transient checkpoints only.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e12_method_axis.log
for A in gift_church_b1.35 imma_church_frp_a1e-7_s1500 h80_meta_church_lam0.7 wsr2_church_lr100; do
  for S in 25 100; do
    for SD in 0 1 6; do
      FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
      [ "$FREE" -lt 3 ] && { echo "ABORT $A s=$S sd=$SD: ${FREE}GiB free" >>"$L"; break 3; }
      K="E12_${A}_sd${SD}_lr1e-5_s${S}"
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
FAM={"gift_church_b1.35":"GIFT port","imma_church_frp_a1e-7_s1500":"IMMA port",
     "h80_meta_church_lam0.7":"Meta variant A","wsr2_church_lr100":"WSR-anchor (ours)",
     "frp_church_r3_l05":"FRP port","utrap_church_l0.15k8":"UTRAP (ours)"}
print("\n=== E12: the METHOD axis, church, lr 1e-5 ===")
print(f"  {'family':<18} {'arm':<30} {'steps':>5} {'valid':>7} {'meanH':>7} {'maxH':>7} {'util':>7}")
for arm,fam in FAM.items():
    cells=collections.defaultdict(set)
    for f in glob.glob(f"outputs/atk_E1*_{arm}_sd*_lr1e-5_s*.json")+glob.glob(f"outputs/atk_H84*_{arm}_*.json"):
        try: d=json.load(open(f))
        except Exception: continue
        hp=d.get("hparams") or {}
        if hp.get("lr")!=1e-5: continue
        rc=(d.get("post_attack_utility") or {}).get("retained_capability") or {}
        cells[hp["steps"]].add((hp.get("seed"), d["attack_validity"]["valid"],
                                round(d["recovery"]["H_after"],4), round(rc.get("after",-1),4)))
    for S in sorted(cells):
        if S not in (25,100,300): continue
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]; u=[x[3] for x in v if x[3]>=0]
        if not ok:
            print(f"  {fam:<18} {arm:<30} {S:>5} {0:>3}/{len(v):<3}   all gate-invalid"); continue
        print(f"  {fam:<18} {arm:<30} {S:>5} {len(ok):>3}/{len(v):<3} {st.mean(ok):>7.4f} "
              f"{max(ok):>7.4f} {st.mean(u) if u else float('nan'):>7.4f}")
PY
echo "E12 DONE $(date -Is)" >>"$L"
