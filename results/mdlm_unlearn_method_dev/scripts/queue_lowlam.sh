#!/bin/bash
# The retain-surviving region. MDU at lam_forget=2 reaches DISCLOSE 0.225 / DEGENERATE 0.092 /
# retain ASCR 0.403 -- the best point any arm has reached, and it CLEARS R1's threshold. Every
# H15 arm so far used lam_forget in {4,8,16}, all of which destroy retain for both arms, so the
# localization mechanism has never been tested where the baseline is actually usable.
# This maps H15 against MDU at matched lam_forget in {1,2,3}. Pre-registered R1/R2/R3 unchanged.
set -u
cd /data/bijaypandey/mdlm_unlearn
busy () { ps -eo args | awk '/venv_llada\/bin\/python/ && (/train_lam\.py/||/train_atwu\.py/)' | wc -l; }
run () {  # arm lamf label
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  ./scripts/launch_py.sh 3 D "$3" 7200 train_lam.py \
    --arm "$1" --anchor-prompt all --lam-forget "$2" --lam-retain 1 --steps 300 \
    --out /data/bijaypandey/mdlm_unlearn/adapters/"$3" \
    > outputs/log_"$3".txt 2>&1
  sleep 5
}
run lam_mdu 2 ll_h15_f2
run lam_mdu 1 ll_h15_f1
run lam_mdu 3 ll_h15_f3
run mdu     1 ll_mdu_f1
run mdu     3 ll_mdu_f3
run lam_mdu_anti 2 ll_anti_f2
echo "[queue_lowlam] done"
