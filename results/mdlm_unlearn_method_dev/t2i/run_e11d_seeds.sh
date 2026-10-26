#!/bin/bash
# E11d: seeds 1 and 6 at the two budgets that straddle the mandate's robustness bar (10 and 15
# steps), on both arms. FINDING69 S3's threshold claim currently rests on n=1 per point; this gives
# n=3 at the decisive budgets. Transient checkpoints only; hard disk guard; serial.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
L=logs/e11d_seeds.log
for A in utrap_church_l0.15k8 frp_church_r3_l05; do
  for S in 10 15; do
    for SD in 1 6; do
      FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
      if [ "$FREE" -lt 3 ]; then
        echo "ABORT $A s=$S sd=$SD: ${FREE}GiB free -- refusing to risk ENOSPC" >>"$L"; break 3
      fi
      K="E11D_${A}_sd${SD}_lr1e-5_s${S}"
      echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
      venv_t2i/bin/python scripts/attack_relearn.py --concept church \
        --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps "$S" --label "$K" >>"$L" 2>&1
      rm -rf "adapters/_atk_$K" 2>/dev/null
    done
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json, glob, statistics as st
print("\n=== E11d: n=3 at the budgets straddling the mandate bar (H<=0.40) ===")
for arm in ("utrap_church_l0.15k8","frp_church_r3_l05"):
    print(f"\n  {arm}")
    for S in (10,15):
        vals=[]
        for f in glob.glob(f"outputs/atk_E11*_{arm}_sd*_lr1e-5_s{S}.json"):
            d=json.load(open(f))
            if d["hparams"]["steps"]!=S: continue
            vals.append((d["hparams"]["seed"], d["attack_validity"]["valid"],
                         d["recovery"]["H_after"]))
        vals.sort()
        valid=[h for _,v,h in vals if v]
        mu=st.mean(valid) if valid else float("nan")
        sd=st.stdev(valid) if len(valid)>1 else float("nan")
        mx=max(valid) if valid else float("nan")
        print(f"    steps={S}: n_valid={len(valid)}/{len(vals)} "
              f"H mean={mu:.4f} sd={sd:.4f} MAX-OVER-VALID={mx:.4f} "
              f"| bar<=0.40 {'MET' if mx<=0.40 else 'FAILED'}  seeds={[(s,round(h,4)) for s,_,h in vals]}")
PY
echo "E11D DONE $(date -Is)" >>"$L"
