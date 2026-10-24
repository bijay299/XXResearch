#!/bin/bash
# FINDING47 S2's named-but-undone measurement: compare concept accuracy at the removal fraction
# where each arm's retain8 reaches a COMMON level (~0.45), instead of at a fixed fraction. The
# existing curves bracket it; these are the interpolating rungs.
#   base            r25 -> retain8 0.450  (already on target)
#   FRP-attacked    r25 -> retain8 0.475  (already on target)
#   AdvUnlearn-atk  r25 -> 0.617, r50 -> 0.100  => need ~0.30-0.40
#   h68rf-attacked  r10 -> 0.783, r25 -> 0.092  => need ~0.13-0.20
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked attacked/KEEP_advunlearn_church_sd0_lr1e-5/unet.pt --concept church \
    --state um_advunlearn --modes random --fracs 0.30,0.35,0.40 \
    >>logs/um_advu.log 2>&1
  echo "UM ADVU DONE $(date -Is)" >>logs/um_advu.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
    --attacked attacked/KEEP_h68rf_church_sd0_lr1e-5/unet.pt --concept church \
    --state um_h68rf --modes random --fracs 0.13,0.16,0.20 \
    >>logs/um_h68rf.log 2>&1
  echo "UM H68RF DONE $(date -Is)" >>logs/um_h68rf.log ) &
wait
echo "UM BOTH DONE $(date -Is)" >>logs/um_chain.log
