#!/bin/bash
# R4d: upstream UCE at ITS OWN DEFAULT -- NO preserve concepts.
# R4c supplied our port's 10 preserve concepts to both sides (matched inputs). But upstream's
# __main__ sets `preserve_concepts = []` when the flag is absent (uce_sd_erase.py:148-149), so
# 1 erase term against 10 preserve terms is NOT how upstream is meant to be run. This measures
# upstream as intended.
#
# RE-RUN 2026-10-26 (audit defects D-026, D-031). The first R4d was CONFOUNDED on the port side:
# r4c keyed its port checkpoint on {concept, lamb} only and skipped the rebuild when a file existed,
# and --no-preserve was never passed through to erase_uce.py. So R4d silently reused R4c's port
# build -- made with the default 10-concept PRESERVE -- while writing "n_preserve": 0 into its
# artifact, and it OVERWROTE R4c's upstream .safetensors in the process. Both are now fixed: the
# cache key encodes the full configuration with a verified fingerprint, and erase_uce.py has an
# opt-in --no-preserve that is proven BIT-IDENTICAL to omitting the preserve terms (max|A-B| = 0.0
# in float64), so for the first time BOTH SIDES receive the same preserve configuration.
set -u
cd /data/bijaypandey/t2i_unlearn

# Group leadership, so gpu_reserve.sh can use pgid as evidence of descent rather than falling back
# to a timed grace (the last documented reservation residual). Done here so it holds however the
# script is launched.
if [ "$$" != "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]; then
  exec setsid "$0" "$@"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4d.log
# Migrated 2026-10-26 off the retired scripts/campaign_free.sh. R4d was one of the three runners
# that campaign_free.sh put on GPU 0 simultaneously on 2026-10-26T12:39. See docs/REPAIR_INFRA_v2.md.
source scripts/gpu_reserve.sh
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
gpu_reserve_hold "r4d_upstream_defaults" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null || {
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
