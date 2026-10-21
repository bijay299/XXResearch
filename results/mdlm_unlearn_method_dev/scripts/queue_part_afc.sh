#!/bin/bash
# Partition evals for the AFC curve and the remaining H16 arms, on GPU 2.
set -u
cd /data/bijaypandey/mdlm_unlearn
A=/data/bijaypandey/mdlm_unlearn/adapters
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

busy () { ps -eo args= | awk -v py="$PY" '$1 == py && /eval_arm_partition\.py/' | wc -l; }

ev () {
  n=0
  while [ ! -f "$2/adapter_model.safetensors" ]; do
    sleep 30; n=$((n+1)); [ "$n" -gt 400 ] && { echo "[queue_part_afc] timeout $1"; return 0; }
  done
  [ -f "outputs/arm_$1.json" ] && { echo "[queue_part_afc] skip $1"; return 0; }
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  ./scripts/launch_py.sh 2 D "part_$1" 7200 eval_arm_partition.py --label "$1" \
    --adapter "$2" > "outputs/log_part_$1.txt" 2>&1
  sleep 5
}

for lab in afc_f0p5 afc_f2 afc_f4 afc_f8 afc_loc_f1 afc_loc_f2 \
           h16_proj_k1_f1 h16_proj_k32_f1 h16_pcgrad_f1 h16_proj_k8_f2; do
  ev "$lab" "$A/$lab/ckpt_300"
done
echo "[queue_part_afc] done"
