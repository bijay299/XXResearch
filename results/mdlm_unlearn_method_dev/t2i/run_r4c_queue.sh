#!/bin/bash
# R4c: actual UCE checkpoints + matched-prompt/matched-seed generations (user item 3).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4c.log
# Migrated 2026-10-26 off the retired scripts/campaign_free.sh (unlocked check-then-select) to the
# exclusive reservation API. See docs/REPAIR_INFRA_v2.md.
source scripts/gpu_reserve.sh
gpu_reserve_hold "r4c_uce" 120 >/dev/null || {
  echo "R4c ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?}); see stderr above" >&2
  echo "R4c ABORT $(date -Is) rc=${GPU_RESERVE_RC:-?}" >>"$L"; exit 1; }
G="$GPU_RESERVED"   # CUDA_VISIBLE_DEVICES is already exported; the release trap is installed

# FAILURE HANDLING ADDED 2026-10-26 (reviewer finding): this runner checked no exit status and
# wrote "R4c DONE" unconditionally. rg_finish is now the only writer of a DONE marker.
source scripts/run_guard.sh
rg_init "R4c" "$L"
echo "=== R4c on GPU $G (uuid ${GPU_RESERVED_UUID:-?}) $(date -Is)" >>"$L"
for lam in 0.5 0.1; do
  rg_cell venv_uce/bin/python scripts/r4c_uce_checkpoints_and_images.py --concept church --lamb "$lam" \
    --erase-scale 1.0 --preserve-scale 1.0
  # Glob: r4c's config tag changed shape between versions (r4c_uce_lamb0.5.json ->
  # r4c_uce_church_lam0.5_es1.0_ps1.0.json), so only a pattern is stable across both.
  rg_require_glob "outputs/r4c_uce_*lam${lam}*.json"
done
rg_finish
