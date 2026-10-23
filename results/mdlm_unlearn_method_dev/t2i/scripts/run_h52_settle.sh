#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H52: settle the LAST open question in the T2I phase.
# FINDING35's corrected claim is "10 of 11 usable arms fully reversed; the one exception could not
# be rebuilt". That exception is h47_adv_a0.3 (retain 0.867, H 0.533 on 1 valid attack of 3) -- the
# sole observation of non-trivial resistance at usable utility anywhere in this phase. It failed to
# reproduce (retain 0.867 -> 0.783) because arm construction was non-deterministic; that bug is now
# fixed (08742c1: Fisher/noise/timestep draws run off one seeded generator).
#
# So rebuild it DETERMINISTICALLY, twice at different seeds to check reproducibility, and attack
# each under PREREG_H51 K1's pre-declared SYMMETRIC budget of 6 attacks. No extra attacks after the
# numbers are seen.
for sd in 0 1; do
  arm=h52_det_a0.3_s${sd}
  o=adapters/$arm
  [ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
    --teacher-te "$TE" --concept church --alpha 0.3 --beta 0.1 --protect retain_fisher \
    --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed $sd --label $arm --out $o \
    > outputs/log_$arm.txt 2>&1 || continue
  [ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
    --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
  for spec in "0|1e-5" "1|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "0|1e-4"; do
    IFS='|' read -r asd lr <<< "$spec"
    lab="${arm}_church_sd${asd}_lr${lr}"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
      --concept church --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $asd --label $lab \
      > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
  rm -f $o/unet.pt
  echo "[h52] $arm done" >> outputs/log_h52_progress.txt
done
echo "[h52] complete"
