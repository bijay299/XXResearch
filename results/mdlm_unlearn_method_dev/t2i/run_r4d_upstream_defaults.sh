#!/bin/bash
# R4d: upstream UCE at ITS OWN DEFAULT -- NO preserve concepts.
# R4c supplied our port's 10 preserve concepts to both sides (matched inputs). But upstream's
# __main__ sets `preserve_concepts = []` when the flag is absent (uce_sd_erase.py:148-149), so
# 1 erase term against 10 preserve terms is NOT how upstream is meant to be run. This measures
# upstream as intended.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4d.log
# Migrated 2026-10-26 off the retired scripts/campaign_free.sh. R4d was one of the three runners
# that campaign_free.sh put on GPU 0 simultaneously on 2026-10-26T12:39. See docs/REPAIR_INFRA_v2.md.
source scripts/gpu_reserve.sh
gpu_reserve_hold "r4d_upstream_defaults" 120 >/dev/null || {
  echo "R4d ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?}); see stderr above" >&2
  echo "R4d ABORT $(date -Is) rc=${GPU_RESERVE_RC:-?}" >>"$L"; exit 1; }
G="$GPU_RESERVED"   # CUDA_VISIBLE_DEVICES is already exported; the release trap is installed

# FAILURE HANDLING ADDED 2026-10-26 (reviewer finding): this runner checked no exit status and
# wrote "R4d DONE" unconditionally. rg_finish is now the only writer of a DONE marker.
source scripts/run_guard.sh
rg_init "R4d" "$L"
echo "=== R4d on GPU $G (uuid ${GPU_RESERVED_UUID:-?}) $(date -Is)" >>"$L"
for lam in 0.5 0.1; do
  rg_cell venv_uce/bin/python scripts/r4c_uce_checkpoints_and_images.py --concept church --lamb "$lam" \
    --no-preserve
  rg_require_glob "outputs/r4d_uce_*lam${lam}*.json"
done
rg_finish
