#!/bin/bash
# ---------------------------------------------------------------------------
# CPU-only regression test for the run_seed.sh completion sentinels.
#
# Extracts the two guard functions from run_seed.sh and exercises them against
# fabricated artifacts in a temporary directory. No GPU, no model, no network:
# the point is only to prove that a partial or truncated artifact is classified
# INCOMPLETE rather than silently accepted as a finished stage.
#
#   bash scripts/seq/test_launcher_guards.sh
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${HERE}/run_seed.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
MODELS="${TMP}/models"; EVALR="${TMP}/eval"
mkdir -p "$MODELS" "$EVALR"
EXPECTED_IMAGES=280

# Pull the sentinels out of the launcher so the test cannot drift from it.
eval "$(sed -n '/^train_complete()/,/^}/p' "$LAUNCHER")"
eval "$(sed -n '/^eval_complete()/,/^}/p' "$LAUNCHER")"

PASS=0; FAIL=0
check() {   # check <description> <expected pass|fail> <function> <arg>
    local desc="$1" want="$2" fn="$3" arg="$4" got
    if "$fn" "$arg"; then got="pass"; else got="fail"; fi
    if [ "$got" = "$want" ]; then
        printf '  ok    %-52s -> %s\n' "$desc" "$got"; PASS=$((PASS+1))
    else
        printf '  FAIL  %-52s -> %s (wanted %s)\n' "$desc" "$got" "$want"; FAIL=$((FAIL+1))
    fi
}

mk_report() {   # mk_report <name> <json>
    mkdir -p "${MODELS}/$1"; printf '%s' "$2" > "${MODELS}/$1/train_report.json"
}
mk_jsonl() {    # mk_jsonl <ck> <nlines>
    mkdir -p "${EVALR}/$1"
    : > "${EVALR}/$1/detections.jsonl"
    for ((i=0; i<$2; i++)); do echo '{"x":1}' >> "${EVALR}/$1/detections.jsonl"; done
}

GOOD='{"finished_utc":"2026-10-09T00:00:00+00:00","child_checkpoint":{"sha256":"abc"}}'

echo "train_complete — a stage counts as done only with a parseable report AND weights"
check "no output at all"                       fail train_complete MISSING
mk_report A "$GOOD"
check "valid report but delta.bin absent"      fail train_complete A
echo x > "${MODELS}/A/delta.bin"
check "valid report + delta.bin"               pass train_complete A
mk_report B "${GOOD:0:40}"
echo x > "${MODELS}/B/delta.bin"
check "report truncated mid-write by a kill"   fail train_complete B
mk_report C '{"child_checkpoint":{"sha256":"abc"}}'
echo x > "${MODELS}/C/delta.bin"
check "report parses but has no finished_utc"  fail train_complete C
mk_report D "$GOOD"
: > "${MODELS}/D/delta.bin"
check "delta.bin present but zero bytes"       fail train_complete D

echo
echo "eval_complete — a stage counts as done only at the full expected row count"
check "no detections.jsonl"                    fail eval_complete MISSING
mk_jsonl E 280
check "280/280 rows"                           pass eval_complete E
mk_jsonl F 137
check "137/280 rows (killed mid-detection)"    fail eval_complete F
mk_jsonl G 0
check "file exists but is empty"               fail eval_complete G
mk_jsonl H 281
check "281 rows (duplicate appended)"          fail eval_complete H

echo
echo "The pre-audit guards were [ -f train_report.json ] and [ -f detections.jsonl ],"
echo "which would have returned pass for every case above that involves an existing"
echo "but unusable file: truncated report, 137/280 rows, empty file, 281 rows."
echo
if [ "$FAIL" -eq 0 ]; then
    echo "ALL ${PASS} GUARD CHECKS PASSED"
else
    echo "${FAIL} GUARD CHECK(S) FAILED (${PASS} passed)" >&2; exit 1
fi
