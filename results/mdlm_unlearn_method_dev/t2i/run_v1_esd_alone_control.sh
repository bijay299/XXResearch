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
#
# BUG FIXED 2026-10-26 (REPAIR_INFRA_v2 defect 7). This line used to read
#     gpu_reserve_hold "v1_esd_alone_control" --allow-frozen
# which put the FLAG in the wait_seconds slot. v1 bound it there silently, so --allow-frozen never
# reached the freeze gate and this run would have been refused with rc=4 no matter what. The call
# below names the wait interval explicitly, which is unambiguous under any parser. (v2 also honours
# the flag position-free and says so loudly, so the class of error cannot hide again.)
#
# ALLOW_FROZEN_RUN is NOT exported here on purpose: the lead supplies it on the command line, so
# merely executing this file can never lift the freeze by itself. Launch with:
#     ALLOW_FROZEN_RUN=1 bash run_v1_esd_alone_control.sh
source scripts/gpu_reserve.sh
gpu_reserve_hold "v1_esd_alone_control" 120 --allow-frozen >/dev/null || {
  echo "V1 ABORT: no GPU reservation granted (rc=${GPU_RESERVE_RC:-?})" >&2
  echo "  rc=4 means the freeze refused this run: relaunch as" >&2
  echo "  ALLOW_FROZEN_RUN=1 bash run_v1_esd_alone_control.sh" >&2
  exit 1; }
# gpu_reserve_hold already exported CUDA_VISIBLE_DEVICES for the reserved device; re-exporting the
# index here would be harmless but redundant, and a hand-edited value would defeat the reservation.
echo "V1 reserved GPU $GPU_RESERVED (uuid ${GPU_RESERVED_UUID:-?}, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/v1_esd_alone.log

# FIRST PRODUCTION CALLER of scripts/run_guard.sh (2026-10-26). Until now that guard had NO caller
# at all -- grep matched only itself and its own test -- so gate condition G2 was being reported as
# passed on the strength of a mechanism nothing used. That is the same failure as D-013's unread
# freeze marker. Wiring it here, into the first run that will actually execute, is what makes the
# guard real rather than cited. rg_finish is the ONLY place a DONE marker may be written, and it
# refuses on a failing step, a missing artifact, a ZERO-BYTE artifact, or an abort.
source scripts/run_guard.sh
rg_init "V1" "$L"
echo "V1 ESD-alone control on GPU $GPU_RESERVED (uuid ${GPU_RESERVED_UUID:-?})" >>"$L"

rg_cell venv_imma/bin/python scripts/gen_eval.py \
  --label V1_esd_alone_vangogh \
  --unet upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt \
  --target van_gogh

# An artifact that merely EXISTS is not a result -- and neither is one that is merely NON-EMPTY.
# D-028's checkpoint, adapters/utrap_church_l1.0k4/unet.pt, is 511,229,952 bytes and passes both
# `[ -f ]` and `[ -s ]`; torch.load on it raises "PytorchStreamReader failed reading zip archive:
# failed finding central directory" (measured 2026-10-26). rg_require_loadable PARSES the file.
rg_require_loadable outputs/gen_V1_esd_alone_vangogh.json json

# The metrics this control exists to produce must be PRESENT, NUMERIC, FINITE and in range before
# anyone reads a number off them. rg_metric never substitutes a value it could not read: a missing
# or malformed field records a failure, so rg_finish refuses DONE rather than letting a 0.0000
# appear in the log as though it had been measured. The denominator is validated too -- a retain
# rate over zero scored samples is undefined, not zero.
V1_JSON=outputs/gen_V1_esd_alone_vangogh.json
for F in active_target_metrics.target_suppression_acc \
         active_target_metrics.legacy_retain8 \
         active_target_metrics.utility_all_nontarget; do
  V=$(rg_metric "$V1_JSON" "$F" 0 1) \
    && echo "V1 metric $F = $V" >>"$L" \
    || echo "V1 metric $F is UNUSABLE -- not substituting a number for it" >>"$L"
done

# Completion manifest: published only after the result file validates and only if nothing failed.
rg_manifest outputs/v1_completion.json "$V1_JSON:json" \
  || echo "V1 completion manifest WITHHELD (see the FAIL lines above)" >>"$L"

# Interpretation limits, written into the log beside the number so they travel with it.
cat >>"$L" <<'NOTE'
V1 INTERPRETATION LIMITS (declared before the result, audit defects D-019 / D-030):
  * This is a BEHAVIOURAL control. It shows what this checkpoint DOES on the 12-concept DEV panel.
    It establishes NOTHING about the checkpoint's training provenance.
  * The scorer is a closed-set 12-way CLIP argmax with NO reject option, validated only on
    undamaged in-distribution outputs. A low retain number here means "closed-set retain accuracy
    collapsed to N/120", NOT "the model is destroyed". Saved images plus an independent
    quality/semantic assessment are required before any destruction or clean-erasure claim.
  * Of the benign explanations for "all 686 tensors differ from base", fp16-variant, dtype
    round-trip, format conversion and an SD-1.5 base are RULED OUT by measurement (the
    main-vs-fp16 floor is 2.074e-04; this checkpoint's resnets reach 0.20). EMA vs non-EMA
    SURVIVES and is UNTESTED. Causes remain labelled hypotheses.
NOTE

rg_finish
