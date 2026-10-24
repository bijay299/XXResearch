#!/bin/bash
# FRP arms for garbage_truck and tench, so the defence side covers the same five concepts the
# adversary side now does. FRP is training-free, so this is minutes.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for RHO in 0.03 0.05; do
    TAG=$(awk -v r="$RHO" 'BEGIN{printf "r%g", r*100}' | tr -d '.')
    ARM=frp_garbage_truck_${TAG}_l05
    [ -f "adapters/$ARM/unet.pt" ] || venv_t2i/bin/python scripts/erase_frp.py \
      --concept garbage_truck --rho $RHO --beta 0.05 --lmbda 0.05 --seed 42 --label "$ARM" \
      >>logs/frp_c45_a.log 2>&1
    [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" \
      --unet "adapters/$ARM/unet.pt" --target garbage_truck >>logs/frp_c45_a.log 2>&1
  done
  echo "FRP GT DONE $(date -Is)" >>logs/frp_c45_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for RHO in 0.03 0.05; do
    TAG=$(awk -v r="$RHO" 'BEGIN{printf "r%g", r*100}' | tr -d '.')
    ARM=frp_tench_${TAG}_l05
    [ -f "adapters/$ARM/unet.pt" ] || venv_t2i/bin/python scripts/erase_frp.py \
      --concept tench --rho $RHO --beta 0.05 --lmbda 0.05 --seed 42 --label "$ARM" \
      >>logs/frp_c45_b.log 2>&1
    [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" \
      --unet "adapters/$ARM/unet.pt" --target tench >>logs/frp_c45_b.log 2>&1
  done
  echo "FRP TN DONE $(date -Is)" >>logs/frp_c45_b.log ) &
wait
