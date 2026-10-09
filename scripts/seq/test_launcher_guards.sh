#!/bin/bash
# ---------------------------------------------------------------------------
# Regression test: the launcher must NOT go back to existence/count-only guards.
#
# History. The first version of this file extracted two shell sentinels
# (`train_complete`, `eval_complete`) from run_seed.sh and exercised them
# against fabricated artifacts. Central review then showed those sentinels
# accepted three plainly unusable artifacts:
#
#   * a delta.bin containing plain text, with a valid-looking report and an
#     arbitrary non-empty SHA;
#   * a detections.jsonl of 280 identical rows;
#   * a detections.jsonl of 280 malformed lines.
#
# They were replaced by contract validation in scripts/seq/validate_stage.py,
# which is tested directly and far more thoroughly by
# scripts/seq/test_validate_stage.py (78 checks, including revalidation of every
# real saved artifact).
#
# This file therefore no longer tests the old sentinels -- they are gone. It
# asserts the REPLACEMENT is wired in and that the weak patterns have not crept
# back. Keeping the old tests would have been worse than useless: with the
# functions removed, every "expect failure" case passed vacuously because an
# undefined function also returns non-zero.
#
#   bash scripts/seq/test_launcher_guards.sh
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
SEED_SH="${HERE}/run_seed.sh"
ANA_SH="${HERE}/run_analysis.sh"
VALIDATOR="${HERE}/validate_stage.py"

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  ok    %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$1"; [ -n "${2:-}" ] && printf '          %s\n' "$2"; }

has()    { if grep -qF -- "$2" "$1"; then ok "$3"; else bad "$3" "not found: $2"; fi; }
hasnt()  { if grep -qF -- "$2" "$1"; then bad "$3" "still present: $2"; else ok "$3"; fi; }
hasre()  { if grep -qE -- "$2" "$1"; then ok "$3"; else bad "$3" "no match: $2"; fi; }
hasntre(){ if grep -qE -- "$2" "$1"; then bad "$3" "still matches: $2"; else ok "$3"; fi; }

echo "the replacement is present"
[ -f "$VALIDATOR" ] && ok "validate_stage.py exists" || bad "validate_stage.py missing"
has "$SEED_SH" 'validate_train' "run_seed.sh validates training stages"
has "$SEED_SH" 'validate_eval'  "run_seed.sh validates evaluation stages"
has "$SEED_SH" '$SEQ_VALIDATOR' "run_seed.sh calls the validator by contract path"
has "$ANA_SH"  '$SEQ_VALIDATOR' "run_analysis.sh validates before aggregating"
has "$SEED_SH" '--expect_steps' "training validation pins the requested step count"
has "$SEED_SH" '--expect_seed'  "training validation pins the training seed"
has "$SEED_SH" '--expect_l2sp'  "training validation pins the L2-SP weight"
has "$SEED_SH" '--expect_manifest_sha' "evaluation validation pins the manifest"
has "$SEED_SH" '--expect_gen_settings' "evaluation validation pins generation settings"
has "$SEED_SH" '--allow_shared_images' "the shared-M0 reuse is declared explicitly"

echo
echo "the weak patterns have not crept back"
hasnt "$SEED_SH" 'train_complete' "no existence-only train_complete sentinel"
hasnt "$SEED_SH" 'eval_complete'  "no existence-only eval_complete sentinel"
hasntre "$SEED_SH" '\[ -f "\$\{EVALR\}/\$\{ck\}/detections\.jsonl" \] &&' \
    "no bare -f test on detections.jsonl as a completion check"
hasntre "$SEED_SH" 'echo "\[done\] \$\{name\} exit=\$\?"' \
    "no exit status discarded through an echo"
hasntre "$SEED_SH" 'wait \$P1 \$P2' \
    "no multi-pid wait that reports only the last child"
hasntre "$ANA_SH" '\-\-eval_root "\$\{SEQ_ROOT\}/eval"' \
    "run_analysis.sh does not hard-code the seed-17 eval root"

echo
echo "failure propagation and path policy"
hasre "$SEED_SH" 'rc=\$\?' "the real exit status is captured immediately"
hasre "$SEED_SH" 'wait "\$P1" \|\| FAILED' "each background child is waited on separately"
has   "$SEED_SH" 'seq_resolve_eval_root' "run_seed.sh uses the shared path policy"
has   "$ANA_SH"  'seq_resolve_eval_root' "run_analysis.sh uses the same shared path policy"
has   "$HERE/exp_config.sh" 'seq_resolve_eval_root' "the policy is defined once, in exp_config.sh"
hasre "$SEED_SH" 'exit 1' "the launcher exits non-zero on an incomplete seed"
has   "$SEED_SH" 'quarantine' "invalid artifacts are quarantined, not deleted"
hasntre "$SEED_SH" 'rm -(r|f|rf) .*detections' "nothing deletes detection evidence"
has   "$ANA_SH"  'mktemp -d' "analysis stages into an isolated directory"
has   "$ANA_SH"  'Nothing was published' "analysis says so when it publishes nothing"
has   "$ANA_SH"  'OPTIONAL_ABSENT' "optional output formats are tracked as optional"

echo
echo "evaluation is bound to the model that produced it"
has "$SEED_SH" '--expect_sha' "run_seed.sh passes the on-disk checkpoint hash"
has "$ANA_SH"  '--expect_sha' "run_analysis.sh passes it too"
has "$SEED_SH" 'ckpt_sha' "the launcher recomputes the hash from disk"
has "$ANA_SH"  'ckpt_sha' "the analysis path recomputes it as well"
has "$SEED_SH" '--base_model_contract' "M0 is validated under a base-model contract"
has "$ANA_SH"  '--base_model_contract' "analysis applies the same M0 policy"
hasre "$VALIDATOR" 'records no sha256' "a missing recorded hash is a failure"
hasre "$VALIDATOR" 'stale with respect to its checkpoint' "a changed checkpoint is named as stale"
hasntre "$VALIDATOR" 'args\.expect_sha and cl\.get\("sha256"\) and' \
    "no short-circuit that skips the hash check when either side is absent"

echo
echo "step completion is evidenced, not inferred"
has "$VALIDATOR" 'optimizer_steps_completed' "the validator requires a real counter"
has "scripts/seq/train_request.py" 'optimizer_steps_completed' "the trainer records one"
has "scripts/seq/train_request.py" 'update_progress_and_checkpoint' \
    "it comes from upstream's own step counter"
hasntre "$VALIDATOR" 'implied = secs / sps' "the circular runtime inference is gone"
hasntre "$VALIDATOR" 'runtime implies' "no step count inferred from timestamps"
has "$VALIDATOR" 'steps_evidence' "a legacy policy exists and is explicit"
has "$HERE/exp_config.sh" 'SEQ_LEGACY_TRAIN_SEEDS' "legacy seeds are declared in config"
has "$VALIDATOR" 'training_completion_verified' "completion is reported as a flag"

echo
echo "the frozen manifest is validated by content"
has "$VALIDATOR" 'canonical_manifest_digest' "one canonical digest rule exists"
hasre "$VALIDATOR" 'man_sha = actual_sha' "later checks use the RECOMPUTED digest"
has "$VALIDATOR" 'CONTENT digest' "a content mismatch is named as such"
has "$HERE/exp_config.sh" 'SEQ_EVAL_MANIFEST_SHA' "the frozen identity lives in config"
hasntre "$SEED_SH" 'manifest_sha256",""' "the launcher no longer reads the digest from the manifest"
hasntre "$ANA_SH"  'manifest_sha256",""' "nor does the analysis path"
has "scripts/seq/build_draft_manifests.py" 'canonical_manifest_digest' \
    "the manifest builder imports the same rule"

echo
echo "the validator fails closed"
has "$VALIDATOR" 'UNCHECKED' "a check that cannot be performed fails the stage"
has "$VALIDATOR" 'os.replace' "completion metadata is written atomically"
hasre "$VALIDATOR" 'torch\.isfinite' "tensors are checked for NaN/Inf"
hasre "$VALIDATOR" 'sha256_file\(ck_p\)' "the checkpoint hash is recomputed, not trusted"

echo
echo "syntax"
for f in "$SEED_SH" "$ANA_SH" "${HERE}/exp_config.sh"; do
    if bash -n "$f" 2>/dev/null; then ok "$(basename "$f") parses"
    else bad "$(basename "$f") parses"; fi
done

echo
echo "-----------------------------------------------------------------"
echo "${PASS} passed, ${FAIL} failed"
if [ "$FAIL" -eq 0 ]; then
    echo "ALL GUARD REGRESSION CHECKS PASSED"
    echo "(behavioural coverage lives in test_validate_stage.py and"
    echo " test_launcher_e2e.sh; this file only guards against regression)"
else
    exit 1
fi
