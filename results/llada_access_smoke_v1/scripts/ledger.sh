#!/bin/bash
# Resource ledger for LLaDA access+feasibility smoke test
W=/data/bijaypandey/llada_smoke
START_EPOCH=$(cat $W/START_EPOCH)
NOW=$(date +%s)
ELAPSED=$(( NOW - START_EPOCH ))
echo "=== LEDGER $(date -Is) ==="
echo "elapsed_seconds: $ELAPSED  ($(python3 -c "print(f'{$ELAPSED/3600:.2f}')") h of 3.00 h cap)"
echo "storage_used_bytes_workspace: $(du -sb $W 2>/dev/null | cut -f1)"
echo "storage_used_human: $(du -sh $W 2>/dev/null | cut -f1) of 60 GiB cap"
echo "gpu_seconds_charged: $(cat $W/GPU_SECONDS 2>/dev/null || echo 0)"
echo "cpu_core_seconds_charged: $(cat $W/CPU_SECONDS 2>/dev/null || echo 0)"
echo "df_data_avail: $(df -h /data | tail -1 | awk '{print $4}')"
