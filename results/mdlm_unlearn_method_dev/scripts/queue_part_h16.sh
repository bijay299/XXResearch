#!/bin/bash
# Generation-level six-cell partition for the H16 arms, on GPU 2.
# Uses the argv[0] guard (see queue_h16c.sh header for the two guard bugs this avoids).
set -u
cd /data/bijaypandey/mdlm_unlearn
A=/data/bijaypandey/mdlm_unlearn/adapters
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

busy () { ps -eo args= | awk -v py="$PY" '$1 == py && /eval_arm_partition\.py/' | wc -l; }

ev () {   # label adapter_dir
  n=0
  while [ ! -f "$2/adapter_model.safetensors" ]; do
    sleep 30; n=$((n+1)); [ "$n" -gt 400 ] && { echo "[queue_part_h16] timeout waiting for $1"; return 0; }
  done
  [ -f "outputs/arm_$1.json" ] && { echo "[queue_part_h16] skip $1"; return 0; }
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  ./scripts/launch_py.sh 2 D "part_$1" 7200 eval_arm_partition.py --label "$1" \
    --adapter "$2" > "outputs/log_part_$1.txt" 2>&1
  sleep 5
}

for lab in h16_none_f1 h16_proj_k8_f1 h16_anti_k8_f1 h16_proj_k1_f1 h16_proj_k32_f1 \
           h16_pcgrad_f1 h16_proj_k8_f2; do
  ev "$lab" "$A/$lab/ckpt_300"
done
# pick up the one low-lam MDU point still missing, so the frontier comparison is complete
ev ll_mdu_f3 "$A/ll_mdu_f3/ckpt_300"
echo "[queue_part_h16] done"
