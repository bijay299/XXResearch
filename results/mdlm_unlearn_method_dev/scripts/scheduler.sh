#!/bin/bash
# ONE scheduler for the whole project. Replaces every per-study queue.
#
# WHY THIS EXISTS. The per-study queues raced each other three separate times today, each time
# costing real GPU work:
#   1. A guard matching 'venv/bin/python' in full argv SELF-MATCHED the shell that created the
#      script via heredoc -> one queue idle 62 minutes with both GPUs free.
#   2. A guard testing comm ~ /^python/ NEVER matched, because PyTorch renames the process to
#      pt_main_thread -> would have put seven arms on one GPU.
#   3. A guard listing script names omitted train_lam.py -> two trainings shared GPU 3 and ATWU
#      died with torch.OutOfMemoryError (8.50 MiB free).
# Every one of those was a guard that inferred GPU occupancy from the process table. This
# scheduler measures free memory on each GPU instead, which cannot be invalidated by adding a
# script, and it is the ONLY thing allowed to launch jobs -- so there is nothing left to race.
#
# Jobs are lines in scripts/JOBS.tsv:   <kind> <label> <args...>
#   kind=train  -> runs the named script on a free GPU
#   kind=part   -> runs eval_arm_partition.py on the named adapter
# Lines beginning with # are ignored. A job whose output already exists is skipped, so the file
# can be appended to and the scheduler restarted safely.
set -u
cd /data/bijaypandey/mdlm_unlearn

# SINGLE-INSTANCE LOCK. The supervisor re-runs this script, and starting it while a previous
# instance is still working would dispatch the same jobs twice -- exactly the duplication the
# scheduler exists to prevent. flock makes a second instance exit immediately instead.
exec 9> /data/bijaypandey/mdlm_unlearn/tmp/scheduler.lock
if ! flock -n 9; then
  echo "[sched] another instance holds the lock; exiting"
  exit 0
fi

GPUS="2 3"
NEED_MB=22000          # a training arm peaks near 19.5 GiB; partition eval near 16 GiB
JOBS=${JOBS_FILE:-scripts/JOBS.tsv}

free_mb () {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ')
  echo $((tot - used))
}

pick_gpu () {           # echo a GPU with headroom, or nothing
  for g in $GPUS; do
    if [ "$(free_mb "$g")" -ge "$NEED_MB" ]; then echo "$g"; return 0; fi
  done
  return 1
}

done_marker () {        # $1 kind, $2 label
  case "$1" in
    train) echo "adapters/$2/report.json" ;;
    part)  echo "outputs/arm_$2.json" ;;
  esac
}

while IFS= read -r line; do
  case "$line" in ''|\#*) continue ;; esac
  set -- $line
  kind=$1; label=$2; shift 2
  marker=$(done_marker "$kind" "$label")
  if [ -f "$marker" ]; then echo "[sched] skip $label (done)"; continue; fi

  # wait for a GPU with real headroom
  g=""
  while [ -z "$g" ]; do g=$(pick_gpu) || { sleep 30; g=""; }; done

  if [ "$kind" = "part" ]; then
    adir="adapters/$label/ckpt_300"
    n=0
    while [ ! -f "$adir/adapter_model.safetensors" ]; do
      sleep 30; n=$((n+1))
      [ "$n" -gt 400 ] && break
    done
    [ -f "$adir/adapter_model.safetensors" ] || { echo "[sched] no adapter for $label"; continue; }
    echo "[sched] gpu$g part $label"
    ./scripts/launch_py.sh "$g" D "part_$label" 7200 eval_arm_partition.py \
      --label "$label" --adapter "$adir" > "outputs/log_part_${label}.txt" 2>&1 &
  else
    script=$1; shift
    echo "[sched] gpu$g train $label ($script)"
    ./scripts/launch_py.sh "$g" D "$label" 7200 "$script" "$@" \
      --out "/data/bijaypandey/mdlm_unlearn/adapters/$label" \
      > "outputs/log_${label}.txt" 2>&1 &
  fi
  # give the job time to claim its memory before the next headroom check
  sleep 90
done < "$JOBS"

wait
echo "[sched] all jobs dispatched"
