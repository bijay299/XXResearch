#!/bin/bash
# Fill the four lambda=0.05 rho rungs the `bc` bug caused stage1 to SKIP (the existence guard saw
# adapters/frp_<c>_r0_l05 and skipped rho=0.03..0.20 under the same label). Runs after H68 so the
# two-GPU cap holds. Then applies the PRE-DECLARED selection rule to the COMPLETE ladder and runs
# the frozen attack grid on whatever it selects.
set -u
cd /data/bijaypandey/t2i_unlearn
fill() {  # $1=gpu $2=concept
  export CUDA_VISIBLE_DEVICES=$1 HF_HOME=/data/bijaypandey/hf_home
  LOG=logs/h66_fill_$2.log
  for RHO in 0.03 0.05 0.10 0.20; do
    TAG=$(awk -v r="$RHO" 'BEGIN{printf "r%g", r*100}' | tr -d '.')
    ARM="frp_$2_${TAG}_l05"
    if [ ! -f "adapters/$ARM/unet.pt" ]; then
      echo "=== BUILD $ARM rho=$RHO $(date -Is)" >>$LOG
      venv_t2i/bin/python scripts/erase_frp.py --concept "$2" \
        --rho "$RHO" --beta 0.05 --lmbda 0.05 --seed 42 --label "$ARM" >>$LOG 2>&1 || continue
    fi
    [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
      --label "$ARM" --unet "adapters/$ARM/unet.pt" --target "$2" >>$LOG 2>&1
  done
  venv_t2i/bin/python scripts/select_h66_operating_point.py --concept "$2" >>logs/h66_select_$2.log 2>&1
  if [ -f "outputs/h66_selected_$2.txt" ]; then
    ./run_h66_frp.sh "$1" "$2" stage2
  fi
  echo "FILL DONE $2 $(date -Is)" >>$LOG
}
for i in $(seq 1 600); do
  d0=0; d1=0
  grep -q "H68 DONE" logs/h68v2_rf_a0.3_church.log 2>/dev/null && d0=1
  grep -q "H68 DONE" logs/h68v2_rf_a0.3_parachute.log 2>/dev/null && d1=1
  [ "$d0" = 1 ] && [ ! -f .fill_church ] && { touch .fill_church; fill 0 church & }
  [ "$d1" = 1 ] && [ ! -f .fill_parachute ] && { touch .fill_parachute; fill 1 parachute & }
  [ -f .fill_church ] && [ -f .fill_parachute ] && break
  sleep 45
done
wait
echo "H66 FILL CHAIN DONE $(date -Is)" >>logs/h66_fill_chain.log
