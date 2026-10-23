#!/bin/bash
# CONTROL for FINDING46: run the SAME removal sweep on the BASE model. If the base model's concept
# is just as robust as the attacked model's while retain8 is just as fragile, then "the recovered
# concept is more robust than general utility" is a property of the ARCHITECTURE / the metric, NOT
# of the attack -- and the mechanism claim must be restated accordingly.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
# build a plain base-model unet.pt once so the sweep can treat it like any arm
venv_t2i/bin/python - <<'PY' >>logs/redun_base.log 2>&1
import torch, os
from diffusers import UNet2DConditionModel
pin=dict(l.split("=",1) for l in open("MODEL_PIN.txt").read().split("\n") if "=" in l)
u=UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",subfolder="unet",
    revision=pin["revision"],torch_dtype=torch.float16)
os.makedirs("adapters/_base_ref",exist_ok=True)
torch.save(u.state_dict(),"adapters/_base_ref/unet.pt")
print("[base] wrote adapters/_base_ref/unet.pt")
PY
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked adapters/_base_ref/unet.pt --concept church \
    >>logs/redun_base_church.log 2>&1
  echo "BASE CHURCH DONE $(date -Is)" >>logs/redun_base_church.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked adapters/_base_ref/unet.pt --concept parachute \
    >>logs/redun_base_parachute.log 2>&1
  echo "BASE PARA DONE $(date -Is)" >>logs/redun_base_parachute.log ) &
wait
