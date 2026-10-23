#!/bin/bash
# H66 -- FRP port (arXiv 2608.25429 Algorithm 1). OUR PORT of a published method.
# Pre-registered in docs/PREREG_H66_frp_port.md. Nothing here is chosen after a number is seen.
#
# Stage 1: the rho ladder {1,3,5,10,20}% at the pre-declared primary lambda=0.05, plus the
#          lambda in {0, 1.0} sensitivity arms at the paper's rho=3%. All erasure+utility only.
# Stage 2: the OPERATING POINT, selected by PREREG_H66 S4's rule -- smallest rho whose target
#          suppression reaches <= 0.400 (AdvUnlearn's erasure depth), utility NOT consulted --
#          then the frozen six-point attack grid on it. Stage 2 is a separate invocation so the
#          selection rule is applied to written-out numbers, not inside the loop.
#
# Usage: run_h66_frp.sh <gpu> <concept> [stage1|stage2]
set -u
cd /data/bijaypandey/t2i_unlearn
GPU=$1; CONCEPT=$2; STAGE=${3:-stage1}
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/bijaypandey/hf_home
mkdir -p logs
LOG=logs/h66_frp_${CONCEPT}_${STAGE}.log

if [ "$STAGE" = "stage1" ]; then
  # rho ladder at the pre-declared primary lambda
  for RHO in 0.01 0.03 0.05 0.10 0.20; do
    TAG=$(printf "r%g" "$(echo "$RHO*100" | bc -l)" | tr -d '.')
    ARM="frp_${CONCEPT}_${TAG}_l05"
    if [ ! -f "adapters/$ARM/unet.pt" ]; then
      echo "=== BUILD $ARM $(date -Is)" >>$LOG
      venv_t2i/bin/python scripts/erase_frp.py --concept "$CONCEPT" \
        --rho "$RHO" --beta 0.05 --lmbda 0.05 --seed 42 --label "$ARM" >>$LOG 2>&1 || continue
    fi
    [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
      --label "$ARM" --unet "adapters/$ARM/unet.pt" --target "$CONCEPT" >>$LOG 2>&1
  done
  # lambda sensitivity at the paper's rho = 3%
  for LAM in 0.0 1.0; do
    TAG=$(echo "$LAM" | tr -d '.')
    ARM="frp_${CONCEPT}_r3_l${TAG}"
    if [ ! -f "adapters/$ARM/unet.pt" ]; then
      echo "=== BUILD $ARM $(date -Is)" >>$LOG
      venv_t2i/bin/python scripts/erase_frp.py --concept "$CONCEPT" \
        --rho 0.03 --beta 0.05 --lmbda "$LAM" --seed 42 --label "$ARM" >>$LOG 2>&1 || continue
    fi
    [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
      --label "$ARM" --unet "adapters/$ARM/unet.pt" --target "$CONCEPT" >>$LOG 2>&1
  done
  echo "STAGE1 DONE $(date -Is)" >>$LOG

elif [ "$STAGE" = "stage2" ]; then
  ARM=$(cat "outputs/h66_selected_${CONCEPT}.txt")   # written by select_h66_operating_point.py
  echo "=== STAGE2 on selected arm $ARM $(date -Is)" >>$LOG
  for spec in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
    set -- $spec; SD=$1; LR=$2
    L="H66_frp_${CONCEPT}_sd${SD}_lr${LR}"
    [ -f "outputs/atk_${L}.json" ] && { echo "[skip] $L" >>$LOG; continue; }
    venv_t2i/bin/python scripts/attack_relearn.py \
      --concept "$CONCEPT" --unet "adapters/$ARM/unet.pt" \
      --steps 300 --lr "$LR" --seed "$SD" --n-images 16 --label "$L" >>$LOG 2>&1
    echo "=== ATK $L rc=$? $(date -Is)" >>$LOG
  done
  echo "STAGE2 DONE $(date -Is)" >>$LOG
fi
