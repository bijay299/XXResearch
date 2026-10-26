#!/bin/bash
# scripts/campaign_free.sh -- RETIRED 2026-10-26. DO NOT USE TO SELECT A GPU.
#
# This script is unsafe by construction and no longer hands out a GPU index. It now prints a
# DIAGNOSTIC report and exits non-zero. Use scripts/gpu_reserve.sh instead.
#
# WHY IT WAS RETIRED -- three independent defects, all confirmed:
#
# (a) CHECK-THEN-SELECT WITH NO LOCK HELD. It read GPU state and returned an index while holding
#     nothing, so two runners that read before either wrote both got the same index. Observed
#     live on 2026-10-26 within a 6-second window, from logs/:
#         logs/r1c_eval.log:1  === R1c on GPU 0 2026-10-26T12:39:29-04:00
#         logs/r4d.log:1       === R4d on GPU 0 2026-10-26T12:39:32-04:00
#         logs/r2_meta.log:11  === R2  on GPU 0 2026-10-26T12:39:35-04:00
#     All three ran on GPU 0 (pids 1369932 / 1370775 / 1371955, 15.3 GB total) while GPU 1 sat
#     idle at 34 MiB. No amount of re-reading state fixes this; it needs a lock.
#
# (b) ITS /proc/<pid>/environ CLAUSE WAS DEAD CODE. The old clause (a) searched runner processes'
#     environ for CUDA_VISIBLE_DEVICES. /proc/<pid>/environ is a snapshot taken at execve(2) and
#     is never updated afterwards. All 8 callers did `export CUDA_VISIBLE_DEVICES=$G` AFTER
#     startup, so the variable was never present in the snapshot and the clause could not match
#     a single one of its own callers. Verified empirically; regression-tested as T13 in
#     scripts/test_gpu_reserve.sh.
#
# (c) A MEMORY THRESHOLD CANNOT DISTINGUISH "free" FROM "between campaign cells". A campaign dips
#     to ~0-1.4 GiB while tearing down one cell and loading the next, which reads as free under
#     any threshold low enough to be useful. Occupancy is not inferable from a memory reading;
#     it has to be declared.
#
# REPLACEMENT -- scripts/gpu_reserve.sh (see its header for the full contract):
#     source scripts/gpu_reserve.sh
#     gpu_reserve_hold "my_campaign"
#     export CUDA_VISIBLE_DEVICES=$GPU_RESERVED
# or, keeping the old call shape (note $$, NOT $BASHPID, which $( ) re-evaluates):
#     while :; do G=$(bash scripts/gpu_reserve.sh acquire my_campaign $$) && break; sleep 120; done
#     trap 'bash scripts/gpu_reserve.sh release my_campaign $$' EXIT INT TERM HUP
#     export CUDA_VISIBLE_DEVICES=$G
#
# Callers still on this script, to be migrated by their owners (Agent 2 / the lead):
#     run_r1_imma_official.sh:28   run_r1b_imma_train.sh:20   run_r1c_eval.sh:14
#     run_r2_meta_official.sh:25   run_r4b_queue.sh:14        run_r4c_queue.sh:11
#     run_r4d_upstream_defaults.sh:15                         run_e2h_meta_v2_stronger.sh:21
set -u

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

{
  echo "=============================================================================="
  echo "scripts/campaign_free.sh is RETIRED and will NOT return a GPU index."
  echo "It was check-then-select with no lock held, and on 2026-10-26 it handed GPU 0"
  echo "to three runners inside 6 seconds while GPU 1 sat idle."
  echo
  echo "Use instead:  bash $SELF_DIR/gpu_reserve.sh acquire <campaign> \$\$"
  echo "See the header of this file and of gpu_reserve.sh for the full rationale."
  echo "=============================================================================="
  echo
  echo "-- diagnostic only: current reservation and occupancy state --"
} >&2

bash "$SELF_DIR/gpu_reserve.sh" status >&2 2>&1 || true

# Exit non-zero and print NOTHING on stdout. Every caller is of the form
#   while :; do G=$(bash scripts/campaign_free.sh|head -1); [ -n "$G" ] && break; sleep 120; done
# so an empty stdout makes them keep waiting rather than silently double-booking a GPU.
exit 1
