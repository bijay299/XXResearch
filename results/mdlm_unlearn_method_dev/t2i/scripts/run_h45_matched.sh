#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# H45: THE UTILITY-MATCHED COMPARISON -- the one the mandate actually asks for.
# So far arms were compared at DIFFERENT utility, which proves nothing about a frontier:
#   UCE tuned        church 0.067  retain 0.850  max-over-valid H 1.000
#   undo alpha=0.5   church 0.000  retain 0.317  max-over-valid H 0.467
# Damage+distil looks more robust, but it also sacrificed 0.53 of utility. The UCE sweep already
# found a point at retain 0.311 (preserve_scale 0.3), which matches undo_a0.5_big's 0.317 almost
# exactly. If UCE at that utility is ALSO reversed to 1.000, damage+distil wins at matched
# utility. If it drops to ~0.47 too, there is no advantage and the whole line is a lateral move.
arm=uce_matched
o=adapters/$arm
[ -f outputs/erase_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/erase_uce.py \
  --concept church --preserve-scale 0.3 --lam 0.1 --label $arm --out $o > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
for sd in 0 1 3 4; do
  lab="H45_${arm}_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept church \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
rm -f $o/unet.pt
echo "[h45] done"
