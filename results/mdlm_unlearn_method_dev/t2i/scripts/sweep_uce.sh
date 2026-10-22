#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=${1:-2}
# Tune the BASELINE fairly. At preserve_scale=0.1/lam=0.1 UCE erased church (1.000->0.000) but
# collapsed retain to 0.350. The erase term (76 positions) was carrying the same weight as all
# 750 preserve positions combined. Sweep the balance and pick the point that erases while
# preserving -- a crippled baseline would make any treatment look good for the wrong reason.
for ps in 0.3 1.0 3.0 10.0; do
  for lam in 0.1 1.0; do
    lab="uce_ch_ps${ps}_l${lam}"
    [ -f outputs/gen_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/erase_uce.py --concept church --preserve-scale $ps \
      --lam $lam --label $lab --out adapters/tmp_$lab > outputs/log_$lab.txt 2>&1 || continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $lab --unet adapters/tmp_$lab/unet.pt \
      --n-per-prompt 3 --steps 25 --seed 0 \
      --concepts church,tench,gas_pump,monet,french_horn,cassette_player >> outputs/log_$lab.txt 2>&1
    rm -rf adapters/tmp_$lab
  done
done
echo "[sweep] done"
