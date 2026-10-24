#!/bin/bash
# FINDING48 S4's last named gap: sensitivity to the MATCHING LEVEL. 0.45 was chosen because base
# and FRP already sat near it; the curves cross, so a different level could reorder the arms.
# These rungs bracket retain8 = 0.30 and 0.60 for base and FRP-attacked church.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export CUDA_VISIBLE_DEVICES=0
venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
  --attacked adapters/_base_ref/unet.pt --concept church --state ml_base \
  --modes random --fracs 0.20,0.22,0.28,0.32 >>logs/ml_base.log 2>&1
venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
  --attacked attacked/DIAG_H66_frp_church_sd0_lr1e-5_keepckpt/unet.pt --concept church \
  --state ml_frp --modes random --fracs 0.20,0.22,0.30,0.35 >>logs/ml_frp.log 2>&1
echo "MATCHLEVEL DONE $(date -Is)" >>logs/ml_base.log
