#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H55: put the two phases on the SAME trainability statistic. The audit's finding 12 was that
# FINDING33 (MDLM: robustness is fact-specific, attacker removes 93.6-98.9% of available loss) and
# FINDING32/35 (T2I: confounded) were never comparably measured -- MDLM used a normalised
# frozen-probe fraction, T2I used a raw late-loss mean, and FINDING33 itself says the raw version
# is the misleading one. The T2I attack now records the SAME normalised statistic.
#
# Three arms spanning the utility range, so the comparison is not a single point:
#   AdvUnlearn alone (retain 0.992)  |  composite a=0.3 (retain ~0.87)  |  a=0.5 (retain ~0.56)
arm=h55_adv_a0.3
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept church --alpha 0.3 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
for spec in "official|" "composite|$o/model_unused"; do :; done
# AdvUnlearn alone (no unet override) and the composite, same seeds/lrs
for sd in 0 1; do
  lab="H55_official_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept church \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] (frozen|${lab})" outputs/log_atk_$lab.txt >> outputs/log_h55_probe.txt
done
for sd in 0 1; do
  lab="H55_composite_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept church --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] (frozen|${lab})" outputs/log_atk_$lab.txt >> outputs/log_h55_probe.txt
done
rm -f $o/unet.pt
echo "[h55] done"
