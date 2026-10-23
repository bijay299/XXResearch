#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H56b: CALIBRATION for the loss-removal statistic. H55 found 6.5-7.1% (official erasure) and
# 9.3-9.9% (composite) with the concept fully restored. Those numbers are uninterpretable without
# reference points at the two extremes of "how much is there to recover":
#   (a) the BASE model, which has nothing erased -- the attacker is fine-tuning on a concept the
#       model already generates perfectly, so this is the floor of "no recovery needed"
#   (b) our UCE re-implementation, a DIFFERENT erasure family, to check the figure is a property
#       of the diffusion objective rather than of AdvUnlearn specifically
# Together with H56's broken arm this gives the statistic a four-point scale instead of a number
# floating in isolation.
for sd in 0 1; do
  lab="H56B_base_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --concept church \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] (frozen|${lab})" outputs/log_atk_$lab.txt >> outputs/log_h55_probe.txt
done
if [ -f adapters/uce_church/unet.pt ]; then
  for sd in 0 1; do
    lab="H56B_uce_church_sd${sd}_lr1e-5"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet adapters/uce_church/unet.pt \
      --concept church --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
      > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] (frozen|${lab})" outputs/log_atk_$lab.txt >> outputs/log_h55_probe.txt
  done
else
  echo "[h56b] uce_church checkpoint absent -- rebuilding it for the second family point"
  CUDA_VISIBLE_DEVICES=$G $PY scripts/erase_uce.py --concept church --preserve-scale 3.0 --lam 0.1 \
    --label h56b_uce --out adapters/h56b_uce > outputs/log_h56b_uce.txt 2>&1
  for sd in 0 1; do
    lab="H56B_uce_church_sd${sd}_lr1e-5"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet adapters/h56b_uce/unet.pt \
      --concept church --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
      > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] (frozen|${lab})" outputs/log_atk_$lab.txt >> outputs/log_h55_probe.txt
  done
  rm -f adapters/h56b_uce/unet.pt
fi
echo "[h56b] done"
