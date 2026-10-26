#!/bin/bash
# E11f: the lr x steps trade, which FINDING69 S6 named as unmeasured ("a different lr would move the
# whole curve; lr is a budget substitute, and that trade is unmeasured").
# n=3 FROM THE START this time -- E11d/E11e showed sd ~0.28 at low budgets, so single-seed points on
# this axis are not publishable, and I am not generating any more of them.
# lr 5e-5 (the frozen grid's second rate) x steps {5, 10, 25} x seeds {0,1,6} on the
# mandate-qualifying arm. Transient checkpoints only; hard disk guard; serial.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11f_lrtrade.log
A=utrap_church_l0.15k8
for S in 5 10 25; do
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    if [ "$FREE" -lt 3 ]; then
      echo "ABORT s=$S sd=$SD: ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; break 2
    fi
    K="E11F_${A}_sd${SD}_lr5e-5_s${S}"
    echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 5e-5 --steps "$S" --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st
A="utrap_church_l0.15k8"
print("\n=== E11f: lr 5e-5 vs lr 1e-5 at matched budgets, n=3, max-over-valid ===")
print(f"{'lr':>7} {'steps':>6} {'n_valid':>8} {'mean':>7} {'sd':>7} {'MAX':>7}  bar<=0.40  seeds")
for lr,tag in (("1e-5","E11*"),("5e-5","E11F")):
    for S in (5,10,25):
        vals=[]
        for f in glob.glob(f"outputs/atk_{tag}_{A}_sd*_lr{lr}_s{S}.json"):
            d=json.load(open(f))
            if d["hparams"]["steps"]!=S or abs(d["hparams"]["lr"]-float(lr))>1e-12: continue
            vals.append((d["hparams"]["seed"], d["attack_validity"]["valid"], d["recovery"]["H_after"]))
        vals=sorted(set(vals)); valid=[h for _,v,h in vals if v]
        if not vals: continue
        mu=st.mean(valid) if valid else float('nan')
        sd=st.stdev(valid) if len(valid)>1 else float('nan')
        mx=max(valid) if valid else float('nan')
        print(f"{lr:>7} {S:>6} {len(valid):>4}/{len(vals):<3} {mu:>7.4f} {sd:>7.4f} {mx:>7.4f}  "
              f"{'MET' if mx<=0.40 else 'FAILED':>9}  {[(s,round(h,4)) for s,_,h in vals]}")
PY
echo "E11F DONE $(date -Is)" >>"$L"
