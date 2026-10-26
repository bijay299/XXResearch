#!/bin/bash
# V1 -- VERIFICATION RUN, not a campaign cell. Gate condition G4 / audit defect D-019.
#
# THE QUESTION. outputs/gen_REF_imma_vangogh.json measured van_gogh 0.867 (the concept the
# checkpoint is supposed to have erased) against utility_all_nontarget 0.0424 over eleven other
# concepts. Agent 2 proved the COMPOSITION valid (exact=160 fuzzy=0 reshaped=0 skipped=0; all
# tensors byte-equal; ESD delta not inverted), so the collapse is not a reconstruction bug.
# The remaining suspect is the third-party checkpoint itself: ALL 686 of its tensors differ from
# pinned base SD-1.4, with resnets to 0.20 relative, which is NOT the cross-attention-only
# erasure that the name "ESDx1" denotes.
#
# WHAT THIS MEASURES. The ESD checkpoint ALONE, with no IMMA tensors applied, on the same twelve
# DEV concepts, same scorer, same seed. This is the control that was never run, and without it
# nothing attributes the utility loss to either ESD or IMMA.
#
# PREDICTION, DECLARED BEFORE THE RESULT:
#   * If ESD alone already shows collapsed non-target utility (retain8 well below base's 1.000),
#     the collapse is INHERITED from the downloaded checkpoint and IMMA is not responsible.
#   * If ESD alone retains utility near base and suppresses van_gogh, then the collapse appears
#     only once IMMA is composed, and IMMA's 160 tensors are implicated.
#   * Either way van_gogh's own accuracy on ESD alone tells us whether this file erases van Gogh
#     at all, which its filename asserts.
# NO ROBUSTNESS CLAIM. No attack is run. One eval, 12 concepts x 15 images.
set -u
cd /data/bijaypandey/t2i_unlearn

# The project is frozen. This is a lead-launched bounded verification run, so it uses the
# documented override, which requires BOTH the env var and the explicit flag and is logged.
export ALLOW_FROZEN_RUN=1
source scripts/gpu_reserve.sh
gpu_reserve_hold "v1_esd_alone_control" --allow-frozen || {
  echo "V1 ABORT: no GPU reservation granted" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU_RESERVED

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/v1_esd_alone.log
echo "=== V1 ESD-alone control on GPU $GPU_RESERVED $(date -Is)" >>"$L"

venv_imma/bin/python scripts/gen_eval.py \
  --label V1_esd_alone_vangogh \
  --unet upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt \
  --target van_gogh >>"$L" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "V1 FAILED exit=$rc -- no result produced" >>"$L"
  exit "$rc"
fi
[ -f outputs/gen_V1_esd_alone_vangogh.json ] || {
  echo "V1 FAILED: exit 0 but no output JSON" >>"$L"; exit 1; }
echo "V1 DONE $(date -Is)" >>"$L"
