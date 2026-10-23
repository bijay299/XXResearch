#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# H44: H43 showed SEED VARIATION decides whether the church attack passes the gate on this arm
# (seed 0 failed at 3 lrs, seed 1 passed at +0.292). So every per-arm robustness number here
# rested on too few attacks. Run more seeds to get a defensible max-over-VALID-attacks figure
# for the best treated arm, and the same for the UCE teacher so the comparison is matched.
for spec in "undo_a0.5_big|adapters/undo_a0.5_big/unet.pt|3" "undo_a0.5_big|adapters/undo_a0.5_big/unet.pt|4" \
            "undo_a0.5_big|adapters/undo_a0.5_big/unet.pt|5" "uce_church|adapters/uce_church/unet.pt|1" \
            "uce_church|adapters/uce_church/unet.pt|2"; do
  IFS='|' read -r nm unet sd <<< "$spec"
  lab="H44_${nm}_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $unet --concept church \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[h44] done"
