#!/usr/bin/env bash
# One-command reproduction. CPU-only, no-root resource watchdog.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${CPR_PYTHON:-/data/bijaypandey/cuig_pilot/venv-cuig/bin/python}"

# --- zero GPU: no CUDA device is visible to the process ---
export CUDA_VISIBLE_DEVICES=""
# --- at most 4 compute threads ---
export CPR_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# --- keep every cache off the full root filesystem ---
export TMPDIR="${TMPDIR:-/data/bijaypandey/claude_tmp}"
export HF_HOME=/data/bijaypandey/.cache/hf XDG_CACHE_HOME=/data/bijaypandey/.cache
export PYTHONPATH="$HERE"

# --- no-root watchdog, layer 1: hard rlimits in this shell ---
#     8 GiB address space, 12 CPU-core-hours of CPU time, no core dumps
ulimit -v $((8*1024*1024)) 2>/dev/null || echo "WARN: could not set ulimit -v" >&2
ulimit -t $((12*3600))     2>/dev/null || echo "WARN: could not set ulimit -t" >&2
ulimit -c 0                2>/dev/null || true

# --- no-root watchdog, layer 2: external wall-clock kill ---
WALL="${CPR_WALL_S:-1800}"
# --- layer 3 is the in-process cpr_gem.ledger.Watchdog thread ---
export CPR_MAX_WALL_S="$WALL" CPR_MAX_CPU_S=43200 CPR_MAX_RSS_KB=$((8*1024*1024))

echo "python : $PY"
echo "wall   : ${WALL}s (timeout) | rlimit -v 8GiB | rlimit -t 12h | threads 4"
/usr/bin/timeout --signal=TERM --kill-after=30 "$WALL" "$PY" run_suite.py
rc=$?
echo "exit code: $rc"
exit $rc
