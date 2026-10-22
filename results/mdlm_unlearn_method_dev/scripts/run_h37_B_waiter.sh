#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
# wait for the h35b diffusion-grid shard to release GPU 2, then run H37 shard B there
for i in $(seq 1 240); do
  grep -q '\[h35b\] done' outputs/log_h35b.txt 2>/dev/null && break
  sleep 30
done
exec bash scripts/run_h37.sh 2 \
  "sup_mdu_l4|sup|mdu|4" "sup_afc_l8|sup|afc|8" "sup_emdu_l8|sup|emdu|8" "sup_ga_l4|sup|ga|4" \
  "dd_afc_a0p1|dd|adapters/su_teacher|0.1" "dd_afc_a0p5|dd|adapters/su_teacher|0.5" \
  "dd_mdu_a0|dd|adapters/su_teacher_mdu|0.0" "dd_mdu_a0p6|dd|adapters/su_teacher_mdu|0.6"
