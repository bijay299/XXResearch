#!/bin/bash
# R4b queue. Waits on an EXCLUSIVE RESERVATION (scripts/gpu_reserve.sh), not on a memory dip.
# Migrated 2026-10-26 off the retired scripts/campaign_free.sh, which was check-then-select with
# no lock held and handed GPU 0 to three runners inside 6 seconds. See docs/REPAIR_INFRA_v2.md.
# Runs the official default (lamb 0.5) and the two representative historical settings actually
# found in our artifacts: lamb 0.1 (13 of 22 arms) and lamb 1000 (the SD-1.5 arm that cleared K1).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4b_uce.log
source scripts/gpu_reserve.sh
gpu_reserve_hold "r4b_uce" 120 >/dev/null || {
  echo "R4b ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?}); see stderr above" >&2
  echo "R4b ABORT $(date -Is) rc=${GPU_RESERVE_RC:-?}" >>"$L"; exit 1; }
G="$GPU_RESERVED"   # gpu_reserve_hold exported CUDA_VISIBLE_DEVICES and installed a release trap

# FAILURE HANDLING ADDED 2026-10-26 (reviewer finding). This runner inspected NO exit status
# anywhere and then wrote "R4b DONE" unconditionally: a cell that crashed, OOM'd or aborted still
# produced a log whose last line said DONE. Migrating the GPU selection without this left half the
# job done. rg_finish is now the only place a DONE marker can be written, and it refuses if any
# cell failed or any artifact is missing or zero-byte.
source scripts/run_guard.sh
rg_init "R4b" "$L"
echo "=== R4b starting on GPU $G (uuid ${GPU_RESERVED_UUID:-?}) $(date -Is)" >>"$L"
for lam in 0.5 0.1 1000; do
  echo "--- lamb=$lam ---" >>"$L"
  # rg_cell records a failure and continues, so one bad lambda does not truncate the sweep.
  rg_cell venv_uce/bin/python scripts/r4b_uce_upstream_vs_port.py --concept church --lamb "$lam" \
    --erase-scale 1.0 --preserve-scale 1.0
  # Glob, not a fixed path: the python side formats lamb as a FLOAT, so lam=1000 writes
  # r4b_uce_lamb1000.0.json. See rg_require_glob's header.
  rg_require_glob "outputs/r4b_uce_lamb${lam}*.json"
done
rg_finish
