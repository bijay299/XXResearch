#!/bin/bash
# E11g: complete the lr x steps design to a full 3x3 factorial at n=3.
# Missing cells after E11/E11d/E11e/E11f:
#   lr 1e-4 x steps {5,10,25} x seeds {0,1,6}   (the frozen grid's third rate, never probed low)
#   lr 1e-5 x steps 25 x seeds {1,6}            (that cell is still n=1, from E11 seed 0 only)
# Transient checkpoints only; hard disk guard; serial. n=3 everywhere by construction.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11g_factorial.log
A=utrap_church_l0.15k8
run () {
  LR=$1; S=$2; SD=$3
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  if [ "$FREE" -lt 3 ]; then echo "ABORT lr=$LR s=$S sd=$SD: ${FREE}GiB free" >>"$L"; return 1; fi
  K="E11G_${A}_sd${SD}_lr${LR}_s${S}"
  echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --steps "$S" --label "$K" >>"$L" 2>&1
  rm -rf "adapters/_atk_$K" 2>/dev/null
}
for S in 5 10 25; do for SD in 0 1 6; do run 1e-4 "$S" "$SD" || break 2; done; done
for SD in 1 6; do run 1e-5 25 "$SD" || break; done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st
A="utrap_church_l0.15k8"
print("\n=== E11g: lr x steps factorial, n=3, max-over-valid H (mandate bar <= 0.40) ===")
print(f"{'lr':>7} {'steps':>6} {'valid':>7} {'mean':>7} {'sd':>7} {'MAX':>7}  bar      seeds")
cells={}
for f in glob.glob(f"outputs/atk_E11*_{A}_sd*_s*.json"):
    d=json.load(open(f)); hp=d["hparams"]
    cells.setdefault((hp["lr"],hp["steps"]),set()).add(
        (hp["seed"], d["attack_validity"]["valid"], round(d["recovery"]["H_after"],4)))
for (lr,S) in sorted(cells):
    vals=sorted(cells[(lr,S)]); valid=[h for _,v,h in vals if v]
    if not valid: continue
    mu=st.mean(valid); sd=st.stdev(valid) if len(valid)>1 else float('nan'); mx=max(valid)
    print(f"{lr:>7.0e} {S:>6} {len(valid):>3}/{len(vals):<3} {mu:>7.4f} {sd:>7.4f} {mx:>7.4f}  "
          f"{'MET' if mx<=0.40 else 'FAIL':<5} {[(s,h) for s,_,h in vals]}")
json.dump({"study":"lr_steps_factorial","arm":A,
  "cells":[{"lr":lr,"steps":S,"seeds":[{"seed":s,"gate_valid":v,"H_after":h}
            for s,v,h in sorted(cells[(lr,S)])]} for lr,S in sorted(cells)]},
  open("outputs/e11_lr_steps_factorial.json","w"), indent=1)
PY
echo "E11G DONE $(date -Is)" >>"$L"
