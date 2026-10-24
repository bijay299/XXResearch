#!/bin/bash
# Direct measurement AT the utility-matched fraction, so the comparison does not rest on
# interpolation. Targets found from the bracketing rungs:
#   AdvUnlearn-attacked: retain8 0.45 lands near frac 0.28
#   h68rf-attacked     : retain8 0.45 lands near frac 0.17
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked attacked/KEEP_advunlearn_church_sd0_lr1e-5/unet.pt --concept church \
    --state umd_advunlearn --modes random --fracs 0.27,0.28,0.29 >>logs/umd_advu.log 2>&1
  echo "UMD ADVU DONE $(date -Is)" >>logs/umd_advu.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
    --attacked attacked/KEEP_h68rf_church_sd0_lr1e-5/unet.pt --concept church \
    --state umd_h68rf --modes random --fracs 0.17,0.18 >>logs/umd_h68rf.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked attacked/DIAG_H66_frp_church_sd0_lr1e-5_keepckpt/unet.pt --concept church \
    --state umd_frp --modes random --fracs 0.26,0.27 >>logs/umd_frp.log 2>&1
  echo "UMD DONE $(date -Is)" >>logs/umd_h68rf.log ) &
wait
