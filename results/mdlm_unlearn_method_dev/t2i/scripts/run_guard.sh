#!/bin/bash
# scripts/run_guard.sh -- sourceable guard so a runner cannot report success after a failure.
# Gate condition G2 / audit defect D-012.
#
# WHY THIS EXISTS
# ---------------
# 104 of the project's run_*.sh scripts end with `echo "<ID> DONE $(date -Is)" >>"$L"` and never
# inspect an exit code anywhere. A cell that crashed, OOM'd, or aborted on low disk therefore left
# a log whose last line says DONE. run_e2h_meta_v2_stronger.sh was the worst case: its low-disk
# abort used `break 2`, which fell straight through to the DONE line and exited 0.
#
# SCOPE OF THE DAMAGE, MEASURED (2026-10-26) -- bounded, and stated so nobody over-reacts:
#   * No Python in scripts/ parses the string DONE.
#   * No runner gates on a previous DONE marker.
#   * No published finding cites a log DONE marker as evidence of completion.
# So a false DONE never mechanically drove a decision. The hazard is a HUMAN reading a log tail
# and concluding a campaign finished. That is why the 104 historical markers are left in place --
# rewriting runners that already ran would not change what their logs said at the time, and would
# churn 104 files whose exact bytes are the record of how historical arms were produced. They are
# instead declared non-evidence (see docs/AUDIT_REPAIR_REGISTER_v1.md, G2). Completion is
# established from ARTIFACTS, never from a log line.
#
# USAGE
#   source scripts/run_guard.sh
#   rg_init "E2H" "$L"                      # campaign id + log path
#   rg_step venv_t2i/bin/python scripts/attack_relearn.py --label "$K" ...
#   rg_require_file "outputs/atk_${K}.json" # an artifact that must exist for the cell to count
#   rg_finish                               # writes DONE only if clean; else INCOMPLETE + exit 1
#
# rg_step records a failure and KEEPS GOING, so one bad cell does not silently truncate a grid;
# rg_abort records a failure and stops immediately. Either way rg_finish refuses to say DONE.
set -u

RG_ID="${RG_ID:-run}"
RG_LOG="${RG_LOG:-/dev/stderr}"
RG_FAILS=0
RG_FAIL_LIST=""

rg_init() {
  RG_ID="${1:?rg_init <campaign_id> <logfile>}"
  RG_LOG="${2:?rg_init <campaign_id> <logfile>}"
  RG_FAILS=0
  RG_FAIL_LIST=""
  echo "=== $RG_ID START $(date -Is) pid=$$" >>"$RG_LOG"
}

# Run a command; on non-zero exit, record it and continue.
#
# ERREXIT NOTE (found while repairing gpu_reserve.sh, 2026-10-26). Real runners use
# `set -euo pipefail`. Without the shield below, `"$@" >>"$RG_LOG" 2>&1` is itself the failing
# command, so errexit killed the runner BEFORE `local rc=$?` ran -- the FAIL line was never
# written and the header's promise that rg_step "records a failure and KEEPS GOING" was false
# under exactly the option set the runners use. The shield guarantees the recording happens.
# rg_step still RETURNS the command's rc, so a caller under set -e that wants to continue must
# write `rg_step ... || true` (or use rg_cell, below). rg_finish is what refuses to say DONE.
rg_step() {
  local _rg_had_e=0 _rg_had_pf=0
  case $- in *e*) _rg_had_e=1 ;; esac
  if shopt -qo pipefail 2>/dev/null; then _rg_had_pf=1; fi
  set +e
  set +o pipefail 2>/dev/null || true
  "$@" >>"$RG_LOG" 2>&1
  local rc=$?
  if [ "$_rg_had_pf" = 1 ]; then set -o pipefail 2>/dev/null || true; fi
  if [ "$_rg_had_e" = 1 ]; then set -e; fi
  if [ "$rc" -ne 0 ]; then
    RG_FAILS=$((RG_FAILS + 1))
    RG_FAIL_LIST="${RG_FAIL_LIST}
  rc=$rc  $*"
    echo "FAIL rc=$rc :: $*" >>"$RG_LOG"
  fi
  return "$rc"
}

# Grid-safe variant: records the failure and ALWAYS returns 0, so one bad cell cannot truncate a
# grid even when the runner uses `set -e`. rg_finish still refuses to write DONE.
rg_cell() {
  rg_step "$@" || true
  return 0
}

# Assert an artifact the cell was supposed to produce. Catches "exit 0 but wrote nothing",
# and the truncation class (D-028): a zero-byte or absent file is not a result.
rg_require_file() {
  local f="${1:?rg_require_file <path>}"
  if [ ! -s "$f" ]; then
    RG_FAILS=$((RG_FAILS + 1))
    RG_FAIL_LIST="${RG_FAIL_LIST}
  missing-or-empty  $f"
    echo "FAIL missing-or-empty artifact :: $f" >>"$RG_LOG"
    return 1
  fi
  return 0
}

# Like rg_require_file, but takes a GLOB and requires at least one NON-EMPTY match.
#
# WHY THIS EXISTS. Several runners cannot predict their artifact's exact filename, because the
# python side formats it. scripts/r4b_uce_upstream_vs_port.py builds
# `outputs/r4b_uce_lamb{a.lamb}.json` where a.lamb is a FLOAT, so the shell loop value `1000`
# becomes the file `r4b_uce_lamb1000.0.json`. A hard-coded rg_require_file would have reported
# INCOMPLETE on a cell that actually succeeded -- a guard that cries wolf gets switched off, which
# is how guards die. r4c/r4d are worse: their config tag changed shape between versions
# (`r4c_uce_lamb0.5.json` then `r4c_uce_church_lam0.5_es1.0_ps1.0.json`), so only a glob is stable.
#
# A glob that matches nothing is still a FAILURE. That is the safe direction: if the artifact name
# drifts again the runner says INCOMPLETE rather than DONE, and someone looks.
rg_require_glob() {
  local pat="${1:?rg_require_glob <glob>}" f hit=0
  for f in $pat; do
    [ -s "$f" ] && hit=1 && break
  done
  if [ "$hit" = 0 ]; then
    RG_FAILS=$((RG_FAILS + 1))
    RG_FAIL_LIST="${RG_FAIL_LIST}
  no-match-or-empty  $pat"
    echo "FAIL no non-empty artifact matches :: $pat" >>"$RG_LOG"
    return 1
  fi
  echo "ok artifact :: $f" >>"$RG_LOG"
  return 0
}

# Record a deliberate abort (low disk, refused reservation, operator stop) as a FAILURE,
# so it can never be confused with completion.
rg_abort() {
  RG_FAILS=$((RG_FAILS + 1))
  RG_FAIL_LIST="${RG_FAIL_LIST}
  abort  ${*:-unspecified}"
  echo "ABORT :: ${*:-unspecified}" >>"$RG_LOG"
}

# The only place a DONE marker may be written.
rg_finish() {
  if [ "$RG_FAILS" -ne 0 ]; then
    echo "$RG_ID INCOMPLETE $(date -Is) -- $RG_FAILS failure(s):${RG_FAIL_LIST}" >>"$RG_LOG"
    echo "$RG_ID INCOMPLETE: $RG_FAILS failure(s); see $RG_LOG" >&2
    return 1
  fi
  echo "$RG_ID DONE $(date -Is)" >>"$RG_LOG"
  return 0
}
