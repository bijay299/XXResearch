#!/bin/bash
# ---------------------------------------------------------------------------
# gpu_select.sh - find an idle GPU on a shared, unscheduled host.
#
# This host has no Slurm: GPUs are shared informally with other researchers, so a
# launch must never assume a device is free. Zero utilisation at one instant does
# not mean idle (a job can be between steps, or loading data), and free memory
# does not mean idle either (another job can be compute-saturated while holding
# little memory).
#
# A GPU is IDLE only if ALL of the following hold:
#   1. every probe of it succeeded (see FAIL-CLOSED below),
#   2. no compute (type "C") process is resident on it,
#   3. used memory stays below MEM_IDLE_MIB in every sample,
#   4. utilisation stays below UTIL_IDLE_PCT in every sample.
#
# FAIL-CLOSED (fixed 2026-10-09)
# ------------------------------
# The previous version initialised each GPU's running maxima to 0 and `continue`d
# past any sample it could not parse. A GPU whose readings were missing,
# non-numeric ("[N/A]", "Unknown Error"), or absent because nvidia-smi itself
# failed therefore kept util=0 / mem=0 and was reported IDLE -- the most
# dangerous possible default, since it invites a launch onto a device whose state
# is unknown and possibly busy.
#
# Now: every GPU must accumulate exactly SAMPLES valid readings. A failed
# nvidia-smi invocation, a missing row, or an unparsable field marks that GPU
# UNKNOWN, and UNKNOWN is never selectable. Absence of evidence is not evidence
# of idleness.
#
# Usage:
#   bash scripts/gpu_select.sh            # report all GPUs, print idle list
#   bash scripts/gpu_select.sh --pick     # print one idle GPU index, or exit 1
#   bash scripts/gpu_select.sh --require 2  # exit 1 unless >=2 GPUs are idle
# ---------------------------------------------------------------------------
set -uo pipefail

SAMPLES="${GPU_SAMPLES:-6}"          # number of utilisation samples
INTERVAL="${GPU_SAMPLE_INTERVAL:-2}" # seconds between samples
UTIL_IDLE_PCT="${UTIL_IDLE_PCT:-5}"  # max util% to still call a GPU idle
MEM_IDLE_MIB="${MEM_IDLE_MIB:-1024}" # max used MiB to still call a GPU idle

PICK_ONLY=0
REQUIRE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pick)    PICK_ONLY=1; shift ;;
        --require) REQUIRE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

die() { echo "GPU PROBE FAILED: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

# --- Enumerate devices ------------------------------------------------------
if ! GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
    die "could not enumerate GPUs (nvidia-smi query failed)"
fi
mapfile -t GPU_IDS < <(printf '%s\n' "$GPU_LIST" | tr -d ' ' | grep -E '^[0-9]+$')
(( ${#GPU_IDS[@]} > 0 )) || die "no GPUs enumerated"

declare -A MAX_UTIL MAX_MEM NPROC PROC_DETAIL OK_SAMPLES UUID GNAME
for id in "${GPU_IDS[@]}"; do
    MAX_UTIL[$id]=-1       # -1 == no valid reading yet (never treat as idle)
    MAX_MEM[$id]=-1
    NPROC[$id]=0
    PROC_DETAIL[$id]=""
    OK_SAMPLES[$id]=0
    UUID[$id]="$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i "$id" 2>/dev/null | xargs)"
    GNAME[$id]="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$id" 2>/dev/null | xargs)"
    [[ -z "${UUID[$id]}" ]] && die "could not read UUID for GPU $id"
done

# --- Resident compute processes (authoritative occupancy signal) -------------
if ! APPS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)"; then
    die "could not query compute processes; refusing to guess occupancy"
fi
while IFS=, read -r a_uuid a_pid a_mem; do
    a_uuid="$(echo "$a_uuid" | xargs)"; a_pid="$(echo "$a_pid" | xargs)"
    a_mem="$(echo "$a_mem" | xargs)"
    [[ -z "$a_pid" ]] && continue
    for id in "${GPU_IDS[@]}"; do
        if [[ "${UUID[$id]}" == "$a_uuid" ]]; then
            NPROC[$id]=$(( ${NPROC[$id]} + 1 ))
            owner="$(ps -o user= -p "$a_pid" 2>/dev/null | xargs)"
            cmd="$(ps -o comm= -p "$a_pid" 2>/dev/null | xargs)"
            PROC_DETAIL[$id]+="    pid=${a_pid} user=${owner:-?} cmd=${cmd:-?} mem=${a_mem}"$'\n'
        fi
    done
done <<< "$APPS"

# --- Sample utilisation and memory over a short window ----------------------
for ((s = 0; s < SAMPLES; s++)); do
    if ! SNAP="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"; then
        echo "WARNING: nvidia-smi sample $((s+1))/$SAMPLES failed; affected GPUs stay UNKNOWN" >&2
        (( s < SAMPLES - 1 )) && sleep "$INTERVAL"
        continue
    fi
    while IFS=, read -r idx util mem; do
        idx="$(echo "$idx" | tr -dc '0-9')"
        util_raw="$(echo "$util" | xargs)"
        mem_raw="$(echo "$mem" | xargs)"
        [[ -z "$idx" ]] && continue
        # Reject anything that is not a clean integer reading. "[N/A]",
        # "Unknown Error" and blanks must NOT be silently coerced to 0.
        if ! [[ "$util_raw" =~ ^([0-9]+)[[:space:]]*%?$ ]]; then continue; fi
        u="${BASH_REMATCH[1]}"
        if ! [[ "$mem_raw" =~ ^([0-9]+)[[:space:]]*MiB$ ]]; then continue; fi
        m="${BASH_REMATCH[1]}"
        (( u > ${MAX_UTIL[$idx]} )) && MAX_UTIL[$idx]=$u
        (( m > ${MAX_MEM[$idx]}  )) && MAX_MEM[$idx]=$m
        OK_SAMPLES[$idx]=$(( ${OK_SAMPLES[$idx]} + 1 ))
    done <<< "$SNAP"
    (( s < SAMPLES - 1 )) && sleep "$INTERVAL"
done

# --- Classify ---------------------------------------------------------------
IDLE=()
declare -A STATE
for id in "${GPU_IDS[@]}"; do
    if (( ${OK_SAMPLES[$id]} < SAMPLES )); then
        STATE[$id]="UNKNOWN"          # fail closed: never selectable
    elif (( ${NPROC[$id]} > 0 )); then
        STATE[$id]="OCCUPIED"
    elif (( ${MAX_MEM[$id]} >= MEM_IDLE_MIB )) || (( ${MAX_UTIL[$id]} >= UTIL_IDLE_PCT )); then
        STATE[$id]="OCCUPIED"
    else
        STATE[$id]="IDLE"
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

echo "GPU availability probe  (${SAMPLES} samples @ ${INTERVAL}s; idle = all samples valid, no compute proc, <${MEM_IDLE_MIB}MiB, <${UTIL_IDLE_PCT}% util)"
echo "host=$(hostname)  time=$(date -Is)"
for id in "${GPU_IDS[@]}"; do
    u="${MAX_UTIL[$id]}"; m="${MAX_MEM[$id]}"
    [[ "$u" == "-1" ]] && u="n/a"
    [[ "$m" == "-1" ]] && m="n/a"
    printf 'GPU %s  %-24s  peak_util=%4s  peak_mem=%7s  procs=%s  samples_ok=%s/%s  uuid=%s  -> %s\n' \
        "$id" "${GNAME[$id]}" "$u" "$m" "${NPROC[$id]}" "${OK_SAMPLES[$id]}" "$SAMPLES" \
        "${UUID[$id]}" "${STATE[$id]}"
    [[ -n "${PROC_DETAIL[$id]}" ]] && printf '%s' "${PROC_DETAIL[$id]}"
done

if (( ${#IDLE[@]} == 0 )); then
    echo
    echo "RESULT: no idle GPU. Do not launch; do not touch other users' processes."
    exit 1
fi
echo
echo "RESULT: idle GPUs = ${IDLE[*]}   (launch with CUDA_VISIBLE_DEVICES=${IDLE[0]})"
if (( REQUIRE > 0 )) && (( ${#IDLE[@]} < REQUIRE )); then
    echo "RESULT: fewer than ${REQUIRE} idle GPUs available."
    exit 1
fi
