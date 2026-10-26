#!/bin/bash
# V2 -- VERIFICATION RUN, not a campaign cell. Gate condition G12 / audit defect D-022.
#
# WHAT THIS IS. Regeneration plus corrected scoring of the FINAL prompt set. Every FINAL number in
# the corpus was scored with gen_eval.py's LAB_KEY (record keys) as the CLIP text instead of LABELS
# (the intended text); nine of the twelve strings differ. The established fact is only that the
# evaluator used inconsistent label text. THE EFFECT ON SCORES AND CONCLUSIONS IS UNRESOLVED
# PENDING THIS EVALUATION -- neither its direction nor its size is known, and nothing here predicts
# either.
#
# WHAT IT IS NOT.
#   * NOT a pure re-score. The original images were never saved, so they must be regenerated. Seeds
#     matching does NOT make this byte-identical; determinism is an ASSUMPTION THIS RUN TESTS, by
#     reproducing the old label set on the regenerated images and comparing against history.
#   * NOT fresh held-out confirmation. FINAL was scored and interpreted by FINDING65/FINDING66, so
#     this is a CORRECTIVE RE-EVALUATION OF AN ALREADY-INSPECTED SPLIT.
#   * NOT complete. Six of the twelve arms have no checkpoint: attacked/E6B_*, E6C_*, E6D_* were
#     deleted (D-022b), so run_status WILL be `partial` and those arms WILL be marked
#     missing_checkpoint. That is the expected outcome, not a failure.
#
# READ THE OUTPUT IN THIS ORDER: run.json -> summary, then each arm's determinism_probe.verdict
# BEFORE its paired_label_text_effect. The probe decides whether this run is comparable to history
# at all; if it is not, the paired effect still measures the label text WITHIN this run but cannot
# be used to correct the historical numbers.
set -u
cd /data/bijaypandey/t2i_unlearn

# SETSID RE-EXEC REMOVED (repair A v3). This block used to read:
#     if [ "$$" != "$(ps -o pgid= -p $$ ... )" ]; then exec setsid "$0" "$@"; fi
# and was described as "closes the last reservation residual". It did not close it, and it cost
# something real:
#   * It made leadership the protection. Leadership is not a protection -- it only decides whether
#     gpu_reserve.sh is ALLOWED to treat process-group membership as evidence of descent. The
#     reclamation decision still came down to a timer for every other launch path, and nothing
#     verified that this re-exec had happened.
#   * `setsid` starts a NEW SESSION WITH NO CONTROLLING TERMINAL, so this script stopped being in
#     the terminal's foreground process group. Ctrl-C from the lead's shell no longer reached it.
#     A change justified by improving the interrupt path actively broke the interrupt path.
#   * It changes the pid after launch, so the pid the lead sees is not the pid that reserves.
# gpu_reserve_hold now holds an flock(2) on an inherited descriptor for the life of the run, which
# covers every descendant through fork and execve regardless of group or session leadership, so
# there is nothing left for this re-exec to buy. See docs/REPAIR_A_v3.md task 1.

source scripts/gpu_reserve.sh
gpu_reserve_hold "v2_rescore_final" 120 --allow-frozen >/dev/null || {
  echo "V2 ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?})" >&2
  echo "  rc=4 means the freeze refused this run: relaunch as" >&2
  echo "  ALLOW_FROZEN_RUN=1 bash run_v2_rescore_final.sh" >&2
  exit 1; }
echo "V2 reserved GPU $GPU_RESERVED (uuid ${GPU_RESERVED_UUID:-?}, CVD=$CUDA_VISIBLE_DEVICES)" >&2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/v2_rescore_final.log

source scripts/run_guard.sh
rg_init "V2" "$L"
echo "V2 rescore_final_v2 on GPU $GPU_RESERVED (uuid ${GPU_RESERVED_UUID:-?})" >>"$L"

# --plan-only first: proves the run directory, guard and arm resolution are sane and writes nothing.
rg_cell venv_t2i/bin/python scripts/rescore_final_v2.py --plan-only

rg_cell venv_t2i/bin/python scripts/rescore_final_v2.py --run-tag d022

# The manifest must exist and be non-empty. A `partial` run_status is EXPECTED (six arms have no
# checkpoint) and is not a failure; an ABSENT manifest is.
LATEST=$(ls -1dt outputs/rescore/*/ 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  # LOADABLE, not merely non-empty (repair A v3). A run.json truncated by a killed scorer passes
  # rg_require_file and then every downstream reader crashes on it -- or worse, a partial read is
  # treated as a measurement. rg_require_loadable parses it.
  rg_require_loadable "${LATEST}run.json" json
  echo "V2 manifest: ${LATEST}run.json" >>"$L"
  # A completion manifest beside it, published only if run.json actually validates. `partial`
  # run_status is expected and is NOT a failure (six arms have no checkpoint); an unreadable
  # run.json is.
  rg_manifest "${LATEST}v2_completion.json" "${LATEST}run.json:json" \
    || echo "V2 completion manifest WITHHELD (see the FAIL lines above)" >>"$L"
else
  rg_abort "no run directory under outputs/rescore/"
fi

rg_finish
