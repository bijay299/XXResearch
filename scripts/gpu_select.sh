#!/bin/bash
# ---------------------------------------------------------------------------
# gpu_select.sh - find an idle GPU on a shared, unscheduled host.
#
# This host has no Slurm / no allocation layer: GPUs are shared informally with
# other researchers, so a launch must never assume a device is free. Zero
# utilisation at one instant does not mean idle (a job can be between steps, or
# loading data), and free memory does not mean idle either (another job can be
# compute-saturated while holding little memory).
#
# A GPU is treated as IDLE only if ALL of the following hold:
#   1. no compute (type "C") process is resident on it,
#   2. used memory stays below MEM_IDLE_MIB,
#   3. utilisation stays below UTIL_IDLE_PCT across every sample.
#
# Usage:
#   bash scripts/gpu_select.sh            # report all GPUs, print idle list
#   bash scripts/gpu_select.sh --pick     # print one idle GPU index, or exit 1
# ---------------------------------------------------------------------------
set -uo pipefail

SAMPLES="${GPU_SAMPLES:-6}"          # number of utilisation samples
INTERVAL="${GPU_SAMPLE_INTERVAL:-2}" # seconds between samples
UTIL_IDLE_PCT="${UTIL_IDLE_PCT:-5}"  # max util% to still call a GPU idle
MEM_IDLE_MIB="${MEM_IDLE_MIB:-1024}" # max used MiB to still call a GPU idle

PICK_ONLY=0
[[ "${1:-}" == "--pick" ]] && PICK_ONLY=1

mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)

declare -A MAX_UTIL MAX_MEM NPROC PROC_DETAIL
for id in "${GPU_IDS[@]}"; do
    MAX_UTIL[$id]=0
    MAX_MEM[$id]=0
    NPROC[$id]=0
    PROC_DETAIL[$id]=""
done

# --- Resident compute processes (authoritative occupancy signal) -------------
# Map each compute PID to its GPU via the GPU UUID, then record owner+command.
while IFS=, read -r uuid pid used; do
    uuid="$(echo "$uuid" | xargs)"; pid="$(echo "$pid" | xargs)"; used="$(echo "$used" | xargs)"
    [[ -z "$pid" ]] && continue
    gidx="$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader \
            | awk -F', *' -v u="$uuid" '$2==u {print $1}')"
    [[ -z "$gidx" ]] && continue
    NPROC[$gidx]=$(( ${NPROC[$gidx]} + 1 ))
    owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs)"
    cmd="$(ps -o comm= -p "$pid" 2>/dev/null | xargs)"
    PROC_DETAIL[$gidx]+="    pid=${pid} user=${owner:-?} cmd=${cmd:-?} mem=${used}"$'\n'
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)

# --- Sample utilisation and memory over a short window ----------------------
for ((s = 0; s < SAMPLES; s++)); do
    while IFS=, read -r idx util mem; do
        idx="$(echo "$idx" | xargs)"
        util="$(echo "$util" | tr -dc '0-9')"
        mem="$(echo "$mem" | tr -dc '0-9')"
        [[ -z "$idx" || -z "$util" || -z "$mem" ]] && continue
        (( util > ${MAX_UTIL[$idx]} )) && MAX_UTIL[$idx]=$util
        (( mem  > ${MAX_MEM[$idx]}  )) && MAX_MEM[$idx]=$mem
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader)
    (( s < SAMPLES - 1 )) && sleep "$INTERVAL"
done

# --- Classify ---------------------------------------------------------------
IDLE=()
for id in "${GPU_IDS[@]}"; do
    if (( ${NPROC[$id]} == 0 )) \
       && (( ${MAX_MEM[$id]} < MEM_IDLE_MIB )) \
       && (( ${MAX_UTIL[$id]} < UTIL_IDLE_PCT )); then
        IDLE+=("$id")
    fi
done

if (( PICK_ONLY )); then
    if (( ${#IDLE[@]} == 0 )); then
        echo "NO_IDLE_GPU" >&2
        exit 1
    fi
    echo "${IDLE[0]}"
    exit 0
fi

echo "GPU availability probe  (${SAMPLES} samples @ ${INTERVAL}s; idle = no compute proc, <${MEM_IDLE_MIB}MiB, <${UTIL_IDLE_PCT}% util)"
echo "host=$(hostname)  time=$(date -Is)"
for id in "${GPU_IDS[@]}"; do
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$id")"
    state="OCCUPIED"
    for i in "${IDLE[@]:-}"; do [[ "$i" == "$id" ]] && state="IDLE"; done
    printf 'GPU %s  %-24s  peak_util=%3s%%  peak_mem=%6sMiB  compute_procs=%s  -> %s\n' \
        "$id" "$name" "${MAX_UTIL[$id]}" "${MAX_MEM[$id]}" "${NPROC[$id]}" "$state"
    [[ -n "${PROC_DETAIL[$id]}" ]] && printf '%s' "${PROC_DETAIL[$id]}"
done

if (( ${#IDLE[@]} == 0 )); then
    echo
    echo "RESULT: no idle GPU. Do not launch; do not touch other users' processes."
    exit 1
fi
echo
echo "RESULT: idle GPUs = ${IDLE[*]}   (launch with CUDA_VISIBLE_DEVICES=${IDLE[0]})"
