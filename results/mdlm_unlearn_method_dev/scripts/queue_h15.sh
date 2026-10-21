#!/bin/bash
# H15 pre-registered: MDU's anchor + localization, with its attribution control, swept.
set -u
cd /data/bijaypandey/mdlm_unlearn
busy () { ps -eo args | awk '/venv_llada\/bin\/python/ && /train_lam\.py/' | wc -l; }
run () {  # arm lam label
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  ./scripts/launch_py.sh 3 D "$3" 7200 train_lam.py \
    --arm "$1" --anchor-prompt all --lam-forget "$2" --steps 300 \
    --out /data/bijaypandey/mdlm_unlearn/adapters/"$3" \
    > outputs/log_"$3".txt 2>&1
  sleep 5
}
run lam_mdu      8 h15_lammdu_l8
run lam_mdu_anti 8 h15_anti_l8
run lam_mdu      4 h15_lammdu_l4
run lam_mdu     16 h15_lammdu_l16
echo "[queue_h15] done"
