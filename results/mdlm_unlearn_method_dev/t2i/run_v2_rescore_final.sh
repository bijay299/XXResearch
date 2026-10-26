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
# PRODUCTION SHELL OPTIONS (repair B, blocker 3). This file ran with `set -u` only, so an
# unchecked failure anywhere outside an rg_* wrapper simply continued. The rg_* calls below are
# written so that errexit never short-circuits the INCOMPLETE bookkeeping: anything that may
# legitimately fail is handled explicitly and the script always reaches rg_finish, whose status
# becomes the script's own.
set -euo pipefail
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

# ---- PRECONDITION, AND IT IS NOW A GATE (repair B, blocker 3). --------------------------------
# `--plan-only` exists to prove the run directory, guard and arm resolution are sane before any
# scoring starts. It was invoked with rg_cell, which records a failure and ALWAYS RETURNS 0, so a
# failed precondition was followed immediately by the real `--run-tag d022` run: the GPU work
# started after the check that was supposed to authorise it had already failed. rg_step records
# the failure AND returns its status, so the gate can actually gate.
if ! rg_step venv_t2i/bin/python scripts/rescore_final_v2.py --plan-only; then
  rg_abort "--plan-only FAILED; refusing to start the scoring run on a configuration that did not validate"
  rg_finish || true
  exit 1
fi

# ---- SNAPSHOT THE RUN DIRECTORIES BEFORE THE RUN. ---------------------------------------------
# The completion check used `ls -1dt outputs/rescore/*/ | head -1`, i.e. "the newest directory".
# If the scoring run produced NOTHING, that resolves to a PREVIOUS run's directory and the whole
# manifest check then passes against a stale artifact -- attesting to completion of a run that
# never happened. Identity, not recency: the directory we attest to must be one that did not
# exist before this run.
V2_BEFORE=$(mktemp "${TMPDIR:-/tmp}/v2_before.XXXXXX")
V2_AFTER=$(mktemp "${TMPDIR:-/tmp}/v2_after.XXXXXX")
trap 'rm -f "$V2_BEFORE" "$V2_AFTER" 2>/dev/null' EXIT
ls -1d outputs/rescore/*/ 2>/dev/null | sort > "$V2_BEFORE" || true

rg_step venv_t2i/bin/python scripts/rescore_final_v2.py --run-tag d022 || true

ls -1d outputs/rescore/*/ 2>/dev/null | sort > "$V2_AFTER" || true
# The NEW directories only. A `partial` run_status inside run.json is EXPECTED (six arms have no
# checkpoint) and is not a failure; a run that created no directory at all IS one.
V2_NEW=$(comm -13 "$V2_BEFORE" "$V2_AFTER" || true)
V2_NEW_N=$(printf '%s' "$V2_NEW" | grep -c . || true)
echo "V2 run directories created by THIS run: ${V2_NEW_N} ${V2_NEW:-<none>}" >>"$L"

if [ "$V2_NEW_N" = 0 ]; then
  rg_abort "this run created NO directory under outputs/rescore/; refusing to attest to a pre-existing one"
else
  # If more than one appeared, take the newest OF THE NEW ONES -- never of all of them.
  LATEST=$(printf '%s\n' "$V2_NEW" | while IFS= read -r d; do [ -n "$d" ] && printf '%s\t%s\n' "$(stat -c %Y "$d" 2>/dev/null || echo 0)" "$d"; done | sort -rn | head -1 | cut -f2-)
  echo "V2 attesting to: $LATEST" >>"$L"
  # LOADABLE, not merely non-empty. A run.json truncated by a killed scorer passes rg_require_file
  # and then every downstream reader crashes on it -- or worse, a partial read is treated as a
  # measurement. rg_require_loadable parses it.
  rg_require_loadable "${LATEST}run.json" json || true
  echo "V2 manifest: ${LATEST}run.json" >>"$L"
  # Published only if run.json actually validates AND nothing else failed; rg_manifest now checks
  # its own publish status and re-reads what it wrote.
  rg_manifest "${LATEST}v2_completion.json" "${LATEST}run.json:json" \
    || echo "V2 completion manifest WITHHELD (see the FAIL lines above)" >>"$L"
fi

rg_finish
