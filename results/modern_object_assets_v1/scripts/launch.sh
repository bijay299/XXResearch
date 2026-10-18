#!/bin/bash
# Ledger-gated GPU launcher for modern_object_assets_v1.
#
#   launch.sh <kind> <label> <phase> <devices> <est_gpu_hours> <est_images> <image_class> -- <cmd...>
#
# Enforces the ceilings BEFORE launching, records allocated device-seconds for
# the whole process lifetime (load, train, inference, reload, OOM, idle), and
# always closes the ledger handle -- including on failure or signal.
set -uo pipefail

ROOT="/data/bijaypandey/cuig_pilot/modern_object_assets_v1"
LEDGER="python3 $ROOT/ledger/ledger.py"

KIND="$1"; LABEL="$2"; PHASE="$3"; DEVICES="$4"
EST_H="$5"; EST_IMG="$6"; IMG_CLASS="$7"
shift 7
[[ "${1:-}" == "--" ]] && shift

# ---- hard gate ----
echo "=== ledger check (phase=$PHASE need=${EST_H}h images=$EST_IMG) ==="
if ! $LEDGER check --need-gpu-hours "$EST_H" --need-images "$EST_IMG" \
        --phase "$PHASE" --image-class "$IMG_CLASS"; then
    echo "LAUNCH REFUSED: a ceiling would be breached." >&2
    exit 3
fi

HANDLE=$($LEDGER open --kind "$KIND" --label "$LABEL" --phase "$PHASE" \
                      --devices "$DEVICES" --note "cmd: $*")
echo "=== ledger handle: $HANDLE ==="

LOG="$ROOT/logs/${LABEL}.log"
mkdir -p "$ROOT/logs"

OUTCOME="unknown"
IMAGES=0
cleanup() {
    local rc=$1
    # decoded-image count, if the job reported one
    if [[ -f "$ROOT/logs/${LABEL}.images" ]]; then
        IMAGES=$(cat "$ROOT/logs/${LABEL}.images")
    fi
    $LEDGER close --handle "$HANDLE" --outcome "$OUTCOME" --images "$IMAGES" \
                  --image-class "$IMG_CLASS" --note "rc=$rc log=$LOG"
}
trap 'OUTCOME="signal"; cleanup 130; exit 130' INT TERM

source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$DEVICES"
# GEM is a uv "virtual" project (no [build-system]), so it is never installed
# into the venv; the README relies on `python -m gem.train` from the repo root.
# Put the pinned source tree on PYTHONPATH so driver scripts can import it.
export PYTHONPATH="$ROOT/GEM_src${PYTHONPATH:+:$PYTHONPATH}"
export MOA_LABEL="$LABEL"
rm -f "$ROOT/logs/${LABEL}.images"
cd "$ROOT/GEM_src"

echo "=== launching: $* ===" | tee "$LOG"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ===" | tee -a "$LOG"
uv run --frozen "$@" >> "$LOG" 2>&1
RC=$?

if [[ $RC -eq 0 ]]; then OUTCOME="ok"; else OUTCOME="failed_rc$RC"; fi
if grep -qiE "CUDA out of memory|torch.OutOfMemoryError" "$LOG"; then
    OUTCOME="oom"
fi
cleanup "$RC"
echo "=== rc=$RC outcome=$OUTCOME log=$LOG ==="
exit $RC
