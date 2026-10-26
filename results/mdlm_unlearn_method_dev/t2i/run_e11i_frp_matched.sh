#!/bin/bash
# E11i: complete FRP's lr 1e-5 curve to n=3 at 25 steps, so the two arms can be compared at a
# MATCHED, NON-DESTRUCTIVE learning rate -- i.e. without the lr 1e-4 damage confound that E11h just
# showed wrecks utility on both arms (FRP 0.62/0.69, UTRAP 0.85/0.53, against 0.92-0.96 at 1e-5).
# FRP already has n=3 at 1e-5 steps 5/10/15 (E11d, E11e). Only the 25-step cell is n=1.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11i_frp_matched.log
A=frp_church_r3_l05
for SD in 0 1 6; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$FREE" -lt 3 ] && { echo "ABORT sd=$SD: ${FREE}GiB free" >>"$L"; break; }
  K="E11I_${A}_sd${SD}_lr1e-5_s25"
  echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps 25 --label "$K" >>"$L" 2>&1
  rm -rf "adapters/_atk_$K" 2>/dev/null
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st, collections
print("\n=== E11i: both arms at lr 1e-5 (non-destructive), n=3, matched budgets ===")
for arm in ("utrap_church_l0.15k8","frp_church_r3_l05"):
    cells=collections.defaultdict(set)
    for f in glob.glob(f"outputs/atk_E11*_{arm}_sd*_lr1e-5_s*.json"):
        d=json.load(open(f)); hp=d["hparams"]
        rc=(d.get("post_attack_utility") or {}).get("retained_capability") or {}
        cells[hp["steps"]].add((hp["seed"], d["attack_validity"]["valid"],
                                round(d["recovery"]["H_after"],4), round(rc.get("after",-1),4)))
    print(f"\n  {arm}  (H_before = {'0.0000' if 'utrap' in arm else '0.1333'})")
    print(f"  {'steps':>6} {'valid':>7} {'meanH':>7} {'maxH':>7} {'util':>7}")
    for S in sorted(cells):
        v=sorted(cells[S]); ok=[x[2] for x in v if x[1]]; u=[x[3] for x in v if x[3]>=0]
        if not ok: continue
        print(f"  {S:>6} {len(ok):>3}/{len(v):<3} {st.mean(ok):>7.4f} {max(ok):>7.4f} "
              f"{st.mean(u) if u else float('nan'):>7.4f}")
PY
echo "E11I DONE $(date -Is)" >>"$L"
