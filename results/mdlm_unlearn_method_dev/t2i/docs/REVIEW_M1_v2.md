# REVIEW_M1_v2 — independent re-review of the integrated revision, milestone 1

Reviewer: independent (did not write any of the reviewed code, and did not write `docs/REVIEW_M1.md`).
2026-10-26. **CPU only. No GPU work launched, no CUDA context created. No git commit. The only file
I edited is this one.**

Revision under review: `xxresearch_standalone` **143074ce21567f95c9dc650b0170a83659e298ff**
("M1 blockers closed: PROJECT was the root of the env-var class. Suites 318/0 and 69/0").
I verified the working copy `/data/bijaypandey/t2i_unlearn` is byte-identical to that commit's tree
(md5 of `scripts/gpu_reserve.sh`, `scripts/run_guard.sh`, `run_v2_rescore_final.sh`, `run_r1_imma_official.sh`, `run_r1c_eval.sh`, `run_r1b_imma_train.sh`,
`run_v1_esd_alone_control.sh`, `scripts/test_gpu_reserve.sh`, `scripts/test_rescore.py`,
`docs/REPAIR_A_v3.md` against `git show 143074c:results/mdlm_unlearn_method_dev/t2i/<path>` — all
SAME). So this review is of the revision as committed.

Calibration: a passing test is valid evidence for the property it genuinely exercises. I did not
discount tests for being author-written. Conversely, a fix with no test is still a fix — I graded
each one by exercising it, and report test coverage separately.

Scope: the five named fixes, both suites, and the two settled residuals. I did not re-derive
`gpu_reserve.sh`. I take as settled (per the brief): compute-apps consulted unconditionally with no
tunable threshold on the `OURS_UNLOCKED` arm; a single `flock -u` on fd 9; the K1 execution-gate
fix; `_gr_holders` at 177 ms over 2,460 fds.

---

## 0. Suite re-runs — counts I observed myself

```
$ bash scripts/test_gpu_reserve.sh
RESULT: 318 passed, 0 failed
No GPU was accessed; nvidia-smi was the shim at /data/bijaypandey/tmp/gpu_reserve_test.QaoLKH/fake_nvsmi
RC=0
```

```
$ venv_t2i/bin/python scripts/test_rescore.py
PASSED 69   FAILED 0
RC=0
```

**318/0 and 69/0, both exit 0, both my own observation.** Two incidental `Killed` lines come from
T35c killing its own lock-lifetime helpers (`test_gpu_reserve.sh:1841`), not from failures.

**But the counts are the same 318 the previous review observed, and that is itself a finding.**
`git diff 1eaaaa7 143074c` touches `test_gpu_reserve.sh` by +142 lines, and every added assertion
is T35a/T35b/T35c — the `close_fds`/`flock -u` block the *previous* review already ran and
discussed. I grepped the suite for the five fixes under review:

```
$ grep -n 'PROJECT' scripts/test_gpu_reserve.sh              -> (no matches)
$ grep -n 'manifest|RG_PY|plan-only|V2_NEW' scripts/test_rescore.py -> no match for any of them
$ grep -n '_GR_HOLD_PATH' scripts/test_gpu_reserve.sh        -> (no matches)
```

**Not one of the five fixes has a test.** D-060 (PROJECT refusal), D-061 (`rg_manifest` publish
status), D-062 (all three `run_v2` properties), D-063 and the `RG_PY` anchoring are each uncovered.
The nearest thing is `test_gpu_reserve.sh:860-864`, which exercises drop-then-acquire **with no
surviving child** — i.e. the one branch where the D-063 defect does not arise (see §4). So 318/0 is
truthful about what it covers and carries **zero** information about this revision's deltas. That is
not a reason to fail a correct fix, but it does mean every verdict below had to be measured, and it
means these five fixes can silently regress.

---

## 1. D-060 — `PROJECT` was environment-derivable: **VERIFIED**

`gpu_reserve.sh:159` is now `_GR_PROD_ROOT="/data/bijaypandey/t2i_unlearn"` — a literal with no
`${PROJECT:-...}` fallback — and `:167-170` anchor `_GR_PROD_LOCKDIR`, `_GR_PROD_FREEZE` and
`_GR_PROD_OVERRIDE_LOG` to it. `:451-459` refuses a *set* `PROJECT` that differs from the literal,
with rc=6, **before** `_gr_env_init` and before any mode branch, and in both modes.

My own probes (project frozen throughout; `status` is read-only):

```
control (no env)                        rc=0  lockdir=/data/.../t2i_unlearn/.gpulocks  FREEZE: ACTIVE
PROJECT=/tmp/evil                       rc=6  REFUSED: PROJECT tried to move the project root
PROJECT=<scratch>/evil                  rc=6  REFUSED
PROJECT=                 (set, empty)   rc=6  REFUSED
PROJECT=/data/.../t2i_unlearn/   (slash) rc=6 REFUSED
PROJECT=/data/.../t2i_unlearn  (exact)  rc=0  lockdir + FREEZE: ACTIVE, unchanged
```

and across every subcommand, so the refusal is not `status`-only:

```
PROJECT=/tmp/evil  acquire  -> rc=6     (control acquire -> rc=4, the freeze)
PROJECT=/tmp/evil  release  -> rc=6
PROJECT=/tmp/evil  holders  -> rc=6
PROJECT=/tmp/evil  status   -> rc=6
```

The lead's measurement reproduces: `PROJECT=/tmp/evil` is **rc=6**, not the old rc=3. Set-but-empty
and a trailing slash are both refused, which is the right direction — anything that is not the
literal is refused rather than normalised.

**I then tried to relocate the four targets by every other route named in the brief.** None works,
and the reason is structural: the four paths are literals, so nothing that changes the *process's*
view of where it is can move them.

```
symlink to the script, run from a foreign cwd   -> lockdir + FREEZE unchanged
a plain `cp` of the script into scratch         -> lockdir + FREEZE unchanged  (BASH_SOURCE moved)
relative invocation (`cd scripts && bash ./gpu_reserve.sh`) -> unchanged
project root reached through a symlinked dir    -> unchanged
TMPDIR / HOME / PWD set to a scratch dir        -> unchanged
GPU_RESERVE_OVERRIDE_LOG=/dev/null              -> rc=6
```

**Enumeration, since the previous "sweep of EVERY variable" claim was false once.** I extracted
every `$VAR`/`${VAR}` in the file and separated the externally-settable ones from internals:

| variable | status |
| --- | --- |
| `PROJECT` | **refused rc=6** unless identical to the literal (this fix) |
| `GPU_RESERVE_{LOCKDIR,GPUS,NVSMI,FREEZE_FILE,OVERRIDE_LOG,TEST_MODE}` | refused rc=6 in production (T33; I re-confirmed `OVERRIDE_LOG`) |
| `GPU_RESERVE_{NVSMI_TIMEOUT,FOREIGN_MIB,WAIT,ORPHAN_GRACE,UID,CVD_STYLE}` | range/enum validated, rc=7 (prior review's 22 probes) |
| `CUDA_VISIBLE_DEVICES`, `ALLOW_FROZEN_RUN` | audited / gated; rc=4 (T33) |
| `TMPDIR` | scratch only; `mktemp` failure fails closed. Cannot move a lock or a gate — measured above |
| `GPU_RESERVED`, `GPU_RESERVED_UUID`, `GPU_RESERVE_RC` | outputs, overwritten on every acquire |
| `BASH_SOURCE[0]`, `$0`, `PPID`, `BASHPID` | set by bash; see §5 for the one real consequence |
| **`PATH`** | **not swept — see the residual below** |

**Residual R1 (named, with my severity reasoning).** `_GR_PROD_NVSMI="nvidia-smi"` (`:169`) is a
**bare command name**, so in *production* the telemetry source is resolved through `$PATH`, and the
test-mode device-identity interlock (`_gr_nvsmi_is_synthetic`) is not consulted in production.
Measured by me, `status` only:

```
$ PATH=<scratch>/pathbin:$PATH bash scripts/gpu_reserve.sh status
mode=production
telemetry=OK  gpus=[1 0] via nvidia-smi
0  free  FREE  -  0  GPU-ffff
1  free  FREE  -  0  GPU-eeee
```

Production accepted wholly fabricated device identities as `telemetry=OK` and reported both allowed
GPUs as holding 0 MiB. It **cannot** thaw the freeze or widen the allowlist (both literals, both
re-verified above), and it cannot reach GPUs 2-3 — so this is not the `PROJECT` harm. What it *can*
do is make an allowed GPU that another user is on read as free, which is the one layer D-059 rests
on. It needs a writable executable on `PATH`, the same capability class the previous review graded
low and non-blocking for the stateful-shim defeat (REVIEW_M1.md §2(d)) — but unlike that one it
needs no `GPU_RESERVE_*` variable at all, only an ambient `PATH` entry, which is the accident shape
`PROJECT` had. Cheap fix: resolve the production telemetry source to an absolute literal. Listed
below as should-fix, not as an M1 blocker, to keep the minimal list minimal — but the claim at
`gpu_reserve.sh:182-185` to be "a sweep of EVERY variable the file reads" is **still not literally
true**, and `PATH` is the remaining counterexample.

---

## 2. D-061 — `rg_manifest` returned 0 on failure: **VERIFIED**

The pipeline is gone (`run_guard.sh:450-491`). The JSON is generated to a temp file whose status is
checked, then published through `rg_atomic_write` whose status is checked, then **read back and
parsed**; each failure calls `_rg_record` (the on-disk ledger) and returns 1.

Measured by me, driving the real functions:

```
case 1: unwritable destination dir, called INSIDE an `if` (the run_r1c_eval.sh:389 shape)
  mktemp: failed to create file via template '.../ro/.rgtmp.XXXXXX': Permission denied
  else-branch taken (rc nonzero)
  manifest on disk? no
  PROBE INCOMPLETE: 1 failure(s)        rg_finish rc=1
  log: FAIL manifest: the atomic publish failed; .../ro/m.json was NOT written
       ledger: manifest  .../ro/m.json :: atomic publish failed

case 2: happy path            rg_manifest rc=0  on-disk=YES   rg_finish rc=0
case 3: required artifact absent  rc=1  on-disk=no  rg_finish rc=1
```

**The `if`-suppression path specifically.** `run_r1c_eval.sh:389` is still
`if rg_manifest "outputs/r1c_completion.json" $R1C_ARTS; then … else …`, and an `if` condition
still suppresses errexit for the whole function body. That no longer matters, and the reason is the
right one: the refusal does not depend on errexit or on the caller checking a status — it is
recorded in the **on-disk failure ledger**, which `rg_finish` reads. So even a caller that ignored
`rg_manifest` entirely cannot get DONE. `:389` also now has a real `else` branch writing
`R1c NO MANIFEST`. Case 1 above is exactly this shape and `rg_finish` returned 1.

**I also tried to defeat it on the truncation path the previous review flagged** (`cat > "$tmp"`
buffering a generator that dies mid-JSON). I shimmed `RG_PY` so the manifest generator exits 0 but
emits 20 bytes of valid-prefix JSON:

```
else-branch: rc nonzero
on disk? no
log: FAIL manifest: .../trunc.json was published but does NOT parse -- withdrawing it
     ledger: manifest  .../trunc.json :: published but does not parse
TRUNC INCOMPLETE: 2 failure(s)   rg_finish rc=1
```

The read-back parse catches it and `rm -f`s the published file. The manifest is no longer the one
artifact exempt from `rg_require_loadable`. Both halves of prior blocker 2 are closed.

---

## 3. D-062 — `run_v2_rescore_final.sh`: **two of three VERIFIED, and a NEW DEFECT introduced**

**(a) `set -e`: VERIFIED.** `run_v2_rescore_final.sh:30` is `set -euo pipefail`.

**(b) GPU work after a failed precondition: VERIFIED.** `:74-78` is now
`if ! rg_step venv_t2i/bin/python scripts/rescore_final_v2.py --plan-only; then rg_abort …;
rg_finish || true; exit 1; fi`. `rg_step` (`run_guard.sh:86-103`) records the failure **and returns
its status**, unlike `rg_cell` (`:105-108`, `rg_step "$@" || true; return 0`), which was the old
call. The real `--run-tag d022` run at `:91` is now unreachable on a failed precondition.

**(c) the stale-directory attestation: VERIFIED, and fixed by identity rather than by recency.**
`ls -1dt … | head -1` is gone from the file (the only remaining mention is the comment at `:81`
recording it; `grep -rn 'ls -1dt' run_*.sh scripts/*.sh` now hits only that comment and
`run_r2_meta_official.sh:1576,1582`, which is out of scope and Agent B's). The replacement
snapshots `ls -1d outputs/rescore/*/` before (`:89`) and after (`:93`) the run, takes
`comm -13` (`:96`), refuses to attest at all if the set of new directories is empty (`:100-101`,
`rg_abort`), and otherwise picks the newest **of the new ones only** (`:104`). A run that creates
nothing can no longer reach `rg_require_loadable` or `rg_manifest`. This is the right shape: the
directory attested to must be one that did not exist before.

### DEFECT FOUND — D-062-R: `run_v2_rescore_final.sh:88` replaces the reservation's EXIT trap, so the reservation leaks on normal completion

```sh
V2_BEFORE=$(mktemp "${TMPDIR:-/tmp}/v2_before.XXXXXX")
V2_AFTER=$(mktemp "${TMPDIR:-/tmp}/v2_after.XXXXXX")
trap 'rm -f "$V2_BEFORE" "$V2_AFTER" 2>/dev/null' EXIT     # <-- :88
```

`gpu_reserve_hold` installs `trap 'gpu_reserve_drop "$_GR_HOLD_CAMP" >/dev/null' EXIT`
(`gpu_reserve.sh:1749`). Line 88 **replaces it**. This is verbatim the "second EXIT trap replaces
the first" hazard that `gpu_reserve.sh` documents twice (`:86`, `:402-403`) and added a refusal for
in the double-acquire case — a refusal that cannot see a caller overwriting the trap directly. It
arrived *with* this fix: `:86-88` are new in this commit.

Measured, with a synthetic shim in test mode (two runners identical except for the trap line):

```
########## A (v2's shape: its own EXIT trap) ##########
A reserved gpu0
--- lockdir after A ---      gpu0.owner   .hold.2263354.trapprobe   .reserve.mutex
########## B (control: no EXIT trap override) ##########
[gpu_reserve] gpu0 released (campaign 'trapprobe', pid 2263451)
--- lockdir after B ---      .reserve.mutex
```

The record and the hold file survive a **clean, successful** exit, and the `gpu0 released` line —
which REVIEW_M1.md §2 identifies as the only positive proof the teardown path ran — never appears.
Consequence for the next job, measured:

```
$ bash scripts/gpu_reserve.sh acquire nextjob $$
[gpu_reserve] gpu0 ORPHANED: owner pid 2263766 is dead but work may still be running
[gpu_reserve]         (trapprobe within-900s-grace-age-2s). The reservation is KEPT and gpu0 is REFUSED
[gpu_reserve] gpu1 RESERVED for 'nextjob'
got gpu'1' rc=0
```

The direction is safe — gpu0 is **refused**, never double-booked, and the record ages out of the
900 s grace window afterwards — so this is availability and truthfulness, not a trampling risk. But
on a two-GPU allowlist it removes one GPU for 15 minutes after every v2 run, leaves litter in the
production lockdir, and destroys the one stderr line that discriminates "the cleanup path ran".
Fix is one line: chain the cleanup into the existing trap
(`trap 'rm -f "$V2_BEFORE" "$V2_AFTER" 2>/dev/null; gpu_reserve_drop "$_GR_HOLD_CAMP" >/dev/null' EXIT`)
or delete the temp files inline instead of on EXIT. INT/TERM/HUP are unaffected (only `EXIT` was
replaced). No other launcher does this — `grep -n 'trap ' run_*.sh` hits only `run_v2:88`.

---

## 4. D-063 — dead `||` and the retained `_GR_HOLD_PATH`: **DEFECT FOUND — NOT FIXED**

The commit message states: *"D-063 … Fixed; the second acquire is now refused outright."* **Both
halves of prior blocker 4 are present in the committed source, unchanged.**
`git diff 1eaaaa7 143074c` does not touch `run_r1_imma_official.sh` at all, and the two
`gpu_reserve.sh` functions involved are unchanged in the relevant lines.

* `gpu_reserve_drop` (`gpu_reserve.sh:1808-1827`) still returns 0 on **both** branches: the
  retention branch is an explicit `return 0` at `:1820`, and the release branch falls off the end
  after `trap - EXIT INT TERM HUP` (`:1826`), which succeeds. So
  `run_r1_imma_official.sh:194`'s `|| say "    WARNING: releasing the prep reservation failed"` is
  still dead code.
* The retention branch clears `_GR_HOLD_FD` (via `_gr_close_hold_fd`, `:1560-1564`) but **leaves
  `_GR_HOLD_PATH` set** — only the release branch clears it, at `:1825`. The double-acquire refusal
  at `:1576` still keys on `_GR_HOLD_FD` alone.

Measured by me, reproducing the stage-2 → stage-4 shape with a child that survives the stage:

```
stage2 reserved gpu0  holdpath=.../locks2/.hold.2266449.r1_imma_prep
--- calling gpu_reserve_drop with a live child (retention branch)
[gpu_reserve] NOT releasing the reservation for 'r1_imma_prep': the liveness lock ... is STILL HELD
DROP RETURNED rc=0        <-- r1:194's '|| say WARNING' fires only if nonzero
_GR_HOLD_PATH after drop = '[.../.hold.2266449.r1_imma_prep]'   _GR_HOLD_FD = '[]'
--- second gpu_reserve_hold in the SAME shell (stage 4)
SECOND ACQUIRE PERMITTED -> gpu1  newholdpath=.../.hold.2266449.r1_imma_smoke
stage-2 hold file .../.hold.2266449.r1_imma_prep still on disk? YES-ORPHANED
=== lockdir ===  gpu0.owner   .hold.2266449.r1_imma_prep   .reserve.mutex
```

So: the drop's outcome is still unobservable to the caller, and **a second acquire in one shell
after a retention drop is genuinely permitted**, overwrites `_GR_HOLD_PATH`, and orphans the
stage-2 hold file and its `gpu0.owner` record — which is exactly the leak the previous review
called "the only place in the review where a reservation can leak". `R1_MODE=prep` on
`run_r1_imma_official.sh` is the live call shape.

**Why the suite did not catch it, and why 318/0 is not evidence here.**
`test_gpu_reserve.sh:860-864` tests drop-then-acquire *with nothing surviving*:

```sh
gpu_reserve_drop two_e >/dev/null 2>&1
if gpu_reserve_hold two_f 5 >/dev/null 2>&1; then echo "after_drop=GRANTED gpu=$GPU_RESERVED"
...
case "$O3" in *"after_drop=GRANTED"*) ok "after gpu_reserve_drop a new reservation IS granted" ;;
```

That is the **release** branch, where `_GR_HOLD_PATH` *is* cleared, and the assertion demands
GRANTED. The retention branch — the only one where the defect exists — has no test, and no test
anywhere checks `gpu_reserve_drop`'s return status.

The failure direction is safe (gpu0 is refused to others, not handed out), so this is a leak, not a
double-book. But it is the one blocker whose "Fixed" claim is contradicted by the committed source,
and it is on the path that runs next.

---

## 5. `RG_PY` was a relative path: **VERIFIED for `RG_PY`; one unanchored path remains (fails closed)**

`run_guard.sh:193-200`:

```sh
_rg_self=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null) || _rg_self="${BASH_SOURCE[0]}"
_rg_root=$(cd "$(dirname "$_rg_self")/.." 2>/dev/null && pwd -P) || _rg_root="."
RG_PY="${RG_PY:-$_rg_root/venv_t2i/bin/python}"
if [ ! -x "$RG_PY" ]; then  printf '[run_guard] WARNING interpreter %s is not executable …' …
```

Anchored to the resolved location of `run_guard.sh` itself, via `readlink -f` + `pwd -P`, so it
survives a `cd`, a symlinked invocation and a symlinked root. The not-executable case warns loudly
on stderr and the comment states the behaviour is fail-closed — and it is: `rg_require_loadable`
and `rg_metric` record failures when the interpreter cannot run, they do not skip. I confirmed
`RG_PY` is the only relative path left in `run_guard.sh`:
`grep -nE 'venv_|scripts/|outputs/|logs/|\./' scripts/run_guard.sh` returns `:195` only (now
absolute). In `gpu_reserve.sh` the same grep returns `:170` (`$_GR_PROD_ROOT/logs/…`, absolute via
the literal) and `:1252` (a message string).

**Residual R2, and it is the same class as the fix.** `gpu_reserve.sh` re-invokes itself through
`"${BASH_SOURCE[0]}"` at `:1706`, `:1708`, `:1728` and `:1823` with **no anchoring**. When a runner
sources the helper by a relative path — which all five launchers do, `source scripts/gpu_reserve.sh`
— and the cwd changes before `gpu_reserve_hold` or `gpu_reserve_drop` runs, that re-invocation
fails. Measured:

```
### control: cwd unchanged
[gpu_reserve] gpu_reserve_hold: acquire failed with rc=4 -- not retrying (see above)
HOLD FAILED rc=4                               <- the freeze, i.e. the real decision

### after a cd between `source` and gpu_reserve_hold
scripts/gpu_reserve.sh: line 1708: scripts/gpu_reserve.sh: No such file or directory
[gpu_reserve] gpu_reserve_hold: acquire failed with rc=127 -- not retrying (see above)
HOLD FAILED rc=127
```

It **fails closed** (rc=127 is not 0 and not the retryable 3, the runner's `|| exit` fires, no
reservation is granted), and no current launcher is exposed: all five `cd
/data/bijaypandey/t2i_unlearn` at the top and every later `cd` is inside a `( … )` subshell. So
this is not a blocker. It is worth the same `readlink -f` treatment `RG_PY` just received, because
the exposure is one `cd` away and the failure mode (`rc=127` on the drop path at `:1823`) would
strand a record.

RG_PY is also env-overridable (`${RG_PY:-…}`), which I used above to inject a truncating generator.
Unlike `PROJECT` this controls no safety gate — and anyone who can set it can run python directly —
but it is worth noting that the validation layer trusts an interpreter from the environment.

---

## 6. The two residuals the brief asked me to state, not settle

**SIGINT cleanup trap — the stderr-line approach of REVIEW_M1.md §2 has NOT been implemented.**
`test_gpu_reserve.sh:1504-1507` and `:1521-1522` still declare SIGINT untestable from the harness
and use `info`, not `ok`, which remains the honest handling. There is no assertion anywhere on
`gpu_reserve.sh`'s `gpu<i> released (campaign …)` or `NOT releasing the reservation for …` stderr
lines — `grep -n "released (campaign\|NOT releasing" scripts/test_gpu_reserve.sh` returns nothing.
So the discriminator named in the prior review was not adopted. Per the brief I do **not** block on
it. **Recorded as named residual R3:** the INT path's trap execution is unverified; the two-minute
interactive check in REVIEW_M1.md §2 (foreground job of an *interactive* shell, a survivor in a
different process group, assert the stderr line rather than the lock state) is still the way to
settle it, and `REPAIR_A_v3.md:775` should stay **NOT measured** until then. Note that D-062-R (§3)
*also* removes that stderr line from `run_v2`'s EXIT path, so fixing D-062-R is a precondition for
ever observing it there.

**Carried-over prior blockers not among my five, still open** (reported, not re-litigated):
`run_v1_esd_alone_control.sh:24` is still `set -u` only (prior blocker 3's second half; its
exposure is milder than v2's, because its `rg_cell` call *is* the GPU work rather than a
precondition for it), and `run_r1b_imma_train.sh:381` / `run_r1c_eval.sh:207` still carry
`export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` (prior blocker 5; a no-op under the default
`index` style, so low harm).

---

## 7. Per-fix verdicts

| fix | verdict |
| --- | --- |
| 1. D-060 `PROJECT` environment-derivable | **VERIFIED** — literal root, rc=6 on every subcommand, immune to cd/symlink/copy/relative-path/TMPDIR; residual R1 (`PATH`) named |
| 2. D-061 `rg_manifest` returned 0 on failure | **VERIFIED** — non-zero + ledgered + DONE refused on all three failure arms, incl. the `if`-suppressed r1c:389 shape and the truncated-publish variant |
| 3. D-062 `run_v2_rescore_final.sh` | **2 of 3 VERIFIED** (`set -euo pipefail`; the precondition now gates via `rg_step`; stale-directory attestation closed by before/after identity) — **DEFECT FOUND: D-062-R**, `:88`'s `trap … EXIT` replaces the reservation's release trap and leaks the record on clean exit |
| 4. D-063 dead `||` + retained `_GR_HOLD_PATH` | **DEFECT FOUND — NOT FIXED.** Both halves reproduce verbatim; the commit's "the second acquire is now refused outright" is **CLAIM UNSUPPORTED** and contradicted by the source and by measurement |
| 5. `RG_PY` relative | **VERIFIED** — anchored via `readlink -f` + `pwd -P`, fails closed when unusable, only relative path left in `run_guard.sh`; residual R2 (`${BASH_SOURCE[0]}`, fails closed) named |
| test coverage of the five | **NONE.** 318→318; no assertion touches `PROJECT`, `rg_manifest`'s publish status, `run_v2`'s three properties, `gpu_reserve_drop`'s status, the retention-then-acquire case, or `RG_PY` |

---

## 8. **MILESTONE 1 VERDICT: FAIL**

Narrow, and three of the five fixes are genuinely good work. D-060 is the right fix for the right
reason — making the root a literal and *refusing* rather than ignoring an override is what makes
every downstream D-047 comparison meaningful again, and I could not relocate the lockdir, the
freeze marker or the audit log by any route I tried. D-061 is closed on all three arms plus the
truncation variant the previous review raised as a related hazard. D-062's stale-attestation fix is
better than what was asked for: identity, not recency. `RG_PY` is properly anchored. Both suites
pass under my own hand: 318/0 and 69/0.

It fails on two things, and the first is the decisive one:

1. **D-063 is not fixed, and the commit message says it is.** The source is unchanged, and I
   measured both halves: `gpu_reserve_drop` returns 0 on the retention branch, and a second
   `gpu_reserve_hold` in the same shell after a retention drop is **permitted**, orphaning the first
   hold file and record. `run_r1_imma_official.sh` is the launcher that runs next and has exactly
   that shape. An unsupported "fixed" claim on a blocker is the project's named failure mode, and a
   gate review cannot pass it.
2. **The D-062 fix introduced a reservation leak on `run_v2`'s normal-completion path**, in the
   precise class (`trap … EXIT` replacing the reservation's trap) that `gpu_reserve.sh` documents
   twice and guards against elsewhere. Measured: record and hold file survive a clean exit, the next
   acquirer is refused gpu0 for up to 900 s, and the `released` line — the only proof the teardown
   ran — is gone.

Both failure directions are safe (refuse, never double-book), and GPUs 2-3 remain unreachable under
every probe I ran. So this is not a safety emergency; it is two reservation leaks and one false
claim, all of which must be closed before the infrastructure can be called trustworthy enough to
carry a verification run's provenance.

### Minimal list of what must change

1. **`gpu_reserve.sh:1808-1827` + `run_r1_imma_official.sh:194` — actually fix D-063.** Give
   `gpu_reserve_drop` a distinct non-zero status for "record retained, work still alive" so the
   `||` branch is live, **and** clear `_GR_HOLD_PATH` on the retention branch (or key the
   double-acquire refusal at `:1576` on `_GR_HOLD_PATH` as well as `_GR_HOLD_FD`) so the stage-4
   acquire cannot orphan the stage-2 hold file. Then correct the claim in the commit record.
2. **`run_v2_rescore_final.sh:88` — do not replace the reservation's EXIT trap.** Chain the temp-file
   cleanup into it, or delete the two temp files inline. One line.
3. **Add one assertion per fix.** Four are enough and all are cheap on CPU: `PROJECT=<x>` ⇒ rc=6 for
   `status` *and* `acquire`; `rg_manifest` to an unwritable destination ⇒ non-zero **and**
   `rg_finish` rc=1 when called inside an `if`; `gpu_reserve_drop` with a live child ⇒ non-zero
   **and** the next `gpu_reserve_hold` in that shell refused; a runner that `cd`s away still gets
   working `rg_require_loadable`. Without these, all five fixes are one edit from regressing and the
   suite will still say 318/0.

### Should fix, not blocking

4. `gpu_reserve.sh:169` — resolve the production telemetry source to an absolute literal, so `PATH`
   cannot substitute it (residual R1). Then the claim at `:182-185` becomes true as written; until
   then it should be narrowed to "every `GPU_RESERVE_*` knob and `PROJECT`".
5. `gpu_reserve.sh:1706,1708,1728,1823` — anchor `${BASH_SOURCE[0]}` the way `RG_PY` now is
   (residual R2). Fails closed today, but is one `cd` away from stranding a record on the drop path.
6. The carried-over prior blockers in §6: `run_v1_esd_alone_control.sh:24`'s missing `-e`, and the
   two `export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` lines at `run_r1b_imma_train.sh:381` and
   `run_r1c_eval.sh:207`.
7. Settle SIGINT per residual R3, after item 2 — it needs a human at a terminal, not a GPU, and the
   assertion must be the `gpu_reserve_drop` stderr line, not the lock state.

### Re-review cost

Items 1-3 are mechanical and under an hour; none needs a GPU. On a second pass I would re-run both
suites, re-run the D-063 retention probe (§4) and the `run_v2` trap probe (§3), and expect to pass.

---

## 9. Reviewer's own footprint

CPU only. No GPU work launched, no CUDA context created, GPUs 2-3 untouched. Every reservation probe
ran in **test mode against a synthetic `nvidia-smi` shim with fabricated UUIDs**, in a lockdir under
my session scratchpad. The only commands that touched the real driver were read-only
`--query-gpu=index,uuid` / `index,memory.used` / `--query-compute-apps` issued by
`gpu_reserve.sh status` itself; I ran no `acquire` in production mode that could have completed —
the project is frozen and every production `acquire` I ran returned rc=4 or rc=6. `FREEZE_ACTIVE` is
intact and unmodified. The production lockdir `/data/bijaypandey/t2i_unlearn/.gpulocks/` holds only
the zero-byte `.reserve.mutex` and no record created by me. `logs/frozen_override.log` is unchanged.
No git commit. The only file I edited is this one.
