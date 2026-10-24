#!/bin/bash
# The AdvUnlearn parachute curve has no rung ABOVE retain8=0.45 except r10 -- fill the bracket so
# its matched point is measured, not extrapolated.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked attacked/KEEP_advunlearn_parachute_sd0_lr1e-5/unet.pt --concept parachute \
    --state umd_advunlearn --modes random --fracs 0.18,0.21,0.23 >>logs/para_advu_fill.log 2>&1
  echo "FILL ADVU DONE $(date -Is)" >>logs/para_advu_fill.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_parachute_r5_l05 \
    --attacked attacked/DIAG_H66_frp_parachute_sd1_lr1e-5_keepckpt/unet.pt --concept parachute \
    --state umd_frp --modes random --fracs 0.18,0.20,0.22 >>logs/para_frp_fill.log 2>&1
  echo "FILL FRP DONE $(date -Is)" >>logs/para_frp_fill.log ) &
wait
