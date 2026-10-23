#!/bin/bash
# Re-run every sweep affected by the label collision. Each leg now has a DISTINCT state, so
# labels cannot collide even when run concurrently. Old contaminated outputs are moved aside,
# not deleted.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
mkdir -p outputs/CONTAMINATED_label_collision
for f in outputs/redundancy_attacked_h68rf_church.json \
         outputs/redundancy_attacked_advunlearn_church.json \
         outputs/redundancy_attacked_frp_sd1_church.json \
         outputs/redundancy_attacked_frp_sd6_church.json; do
  [ -f "$f" ] && mv "$f" outputs/CONTAMINATED_label_collision/
done
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
    --attacked attacked/KEEP_h68rf_church_sd0_lr1e-5/unet.pt --concept church \
    --state attacked_h68rf >>logs/redo_h68rf.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked attacked/KEEP_frp_church_sd1_lr1e-5/unet.pt --concept church \
    --state attacked_frp_sd1 >>logs/redo_frp_sd1.log 2>&1
  echo "REDO GPU0 DONE $(date -Is)" >>logs/redo_chain.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm _base_ref \
    --attacked attacked/KEEP_advunlearn_church_sd0_lr1e-5/unet.pt --concept church \
    --state attacked_advunlearn >>logs/redo_advu.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked attacked/KEEP_frp_church_sd6_lr1e-5/unet.pt --concept church \
    --state attacked_frp_sd6 >>logs/redo_frp_sd6.log 2>&1
  echo "REDO GPU1 DONE $(date -Is)" >>logs/redo_chain.log ) &
wait
echo "REDO ALL DONE $(date -Is)" >>logs/redo_chain.log
