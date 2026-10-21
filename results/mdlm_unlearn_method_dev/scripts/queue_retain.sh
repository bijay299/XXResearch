#!/bin/bash
# H15 failed R1 (retain ASCR 0.199 < 0.40) while PASSING R2 and R3. Retain is the failing
# axis and it was never swept: lam_forget was swept, lam_retain was held at 1.0 throughout.
# Position localization is verified WITHIN forget sequences (unselected positions moved
# -0.015) but retain inputs still degraded, so the leak is through shared parameters, not
# through position choice. Sweeping the retain constraint is the completion of the
# experiment, not a change of goalposts: R1/R2/R3 stay exactly as pre-registered.
set -u
cd /data/bijaypandey/mdlm_unlearn
busy () { ps -eo args | awk '/venv_llada\/bin\/python/ && (/train_lam\.py/||/train_atwu\.py/)' | wc -l; }
run () {  # arm lamf lamr label
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  ./scripts/launch_py.sh 3 D "$4" 7200 train_lam.py \
    --arm "$1" --anchor-prompt all --lam-forget "$2" --lam-retain "$3" --steps 300 \
    --out /data/bijaypandey/mdlm_unlearn/adapters/"$4" \
    > outputs/log_"$4".txt 2>&1
  sleep 5
}
run lam_mdu 8  4  h15_lr4
run lam_mdu 8 16  h15_lr16
run mdu     8 16  mdu_lr16
run lam_mdu 4 16  h15_f4_lr16
echo "[queue_retain] done"
