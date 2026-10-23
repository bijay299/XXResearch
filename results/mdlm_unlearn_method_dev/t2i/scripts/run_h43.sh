#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# H43: does the church-vs-tench asymmetry generalise to a SECOND untargeted concept at the only
# budget where it appeared (lr 1e-5)? H42 showed it does not survive lr 1e-4/3e-4 -- there the
# attack destroys the arm and nothing relearns -- so the claim currently rests on one lr and one
# untargeted concept. Two more seeds of church are included so the comparison is not n=1.
o=adapters/undo_a0.5_big
for spec in "french_horn|0" "tench|1" "church|1" "church|2"; do
  IFS='|' read -r c sd <<< "$spec"
  pre=""; [ "$c" != "church" ] && pre="TRAIN_"
  lab="${pre}H43_a0.5_big_${c}_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept $c \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[h43] done"
