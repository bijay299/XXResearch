#!/bin/bash
# E6d: two more seeds on uce_sd15_church_lam1000 -- the ONLY sub-ceiling post-attack FINAL number
# (14/15 vs DEV's 15/15). Under the FINDING35 S4 ratchet a sub-ceiling value is provisional and more
# seeds can only raise the max, so this is the escalation that number requires.
# Serial, one GPU, each 1.7 GiB checkpoint deleted immediately: /data has ~6.7 GiB free.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export BASE_MODEL=runwayml/stable-diffusion-v1-5 BASE_VARIANT=fp16
export BASE_REV=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
export CUDA_VISIBLE_DEVICES=0
L=logs/e6d.log
for SD in 1 6; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  if [ "$FREE" -lt 4 ]; then
    echo "ABORT before seed $SD: only ${FREE}GiB free on /data -- refusing to risk ENOSPC" >>"$L"
    break
  fi
  K="E6D_uce_sd15_church_lam1000_sd${SD}_lr1e-5"
  echo "=== $K $(date -Is) free=${FREE}GiB" >>"$L"
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet adapters/uce_sd15_church_lam1000/unet.pt --seed "$SD" --lr 1e-5 \
    --keep-checkpoint --label "$K" >>"$L" 2>&1
  if [ -f "attacked/$K/unet.pt" ]; then
    venv_t2i/bin/python - <<PY >>"$L" 2>&1
import importlib.util as iu,sys
sys.argv=["x","--only","ATK_uce_sd15_church_lam1000"]
spec=iu.spec_from_file_location("g","/data/bijaypandey/t2i_unlearn/scripts/gen_eval_final.py")
m=iu.module_from_spec(spec)
# point the pre-committed ATK entry at THIS seed's checkpoint
spec.loader.exec_module(m)
m.ARMS=[("ATK_uce_sd15_church_lam1000_sd${SD}","attacked/$K/unet.pt","church",m.SD15)]
sys.argv=["x","--only","ATK_uce_sd15_church_lam1000_sd${SD}"]
m.main()
PY
    rm -rf "attacked/$K"
    echo "$K scored on FINAL, checkpoint deleted" >>"$L"
  else
    echo "$K NO KEPT CHECKPOINT -- FINAL not scored (not a result)" >>"$L"
  fi
done
echo "E6D DONE $(date -Is)" >>"$L"
