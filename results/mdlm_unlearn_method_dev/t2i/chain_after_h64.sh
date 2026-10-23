#!/bin/bash
# After each H64 arm finishes its 9 attacks, run H66 (FRP port) on that GPU, then H68 (the v2
# retain_fisher control). Never more than two GPUs: one pipeline per GPU, keyed to the arm that
# was using it.
set -u
cd /data/bijaypandey/t2i_unlearn
pipeline() {   # $1=gpu $2=concept
  ./run_h66_frp.sh "$1" "$2" stage1
  venv_t2i/bin/python scripts/select_h66_operating_point.py --concept "$2" \
     >>logs/h66_select_$2.log 2>&1
  [ -f "outputs/h66_selected_$2.txt" ] && ./run_h66_frp.sh "$1" "$2" stage2
  ./run_h68_v2_control.sh "$1" "$2"
  echo "PIPELINE $2 DONE $(date -Is)" >>logs/chain_after_h64.log
}
for i in $(seq 1 600); do
  for pair in "0 church" "1 parachute"; do
    set -- $pair; G=$1; C=$2
    if grep -q "ALL DONE" logs/h64v2_frag_a0.3_${C}.log 2>/dev/null && [ ! -f .chain_${C} ]; then
      touch .chain_${C}; pipeline "$G" "$C" &
    fi
  done
  [ -f .chain_church ] && [ -f .chain_parachute ] && break
  sleep 60
done
wait
echo "CHAIN2 ALL DONE $(date -Is)" >>logs/chain_after_h64.log
