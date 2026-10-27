# REVIEW M1 v3 — independent review of the integrated revision

**Reviewer:** independent (third review; v1 and v2 both returned FAIL)
**Revision under review:** `xxresearch_standalone` `2cb8935db4868d294b54384ea727ee4c3308e7a2`
("Close R1 and R3; add one assertion per fix. Suite 318 -> 327, 0 failed.")
**Working copy:** `/data/bijaypandey/t2i_unlearn`
**Date:** 2026-10-26
**Constraints honoured:** no GPU work (CPU only; every reservation test ran in `GPU_RESERVE_TEST_MODE=1`
against a synthetic `nvidia-smi` in a private lockdir under my scratchpad). No git commits. No file
edited but this one. Everything else read-only; problems are reported, not fixed.

Items v1/v2 already settled were not re-litigated: `PROJECT`/D-060 hardening (quick confirmation
only), compute-apps consulted unconditionally with no tunable threshold, the single `flock -u` on
fd 9, the K1 execution-gate fix, `_gr_holders` timing.

---

## Summary

| # | item | verdict |
|---|---|---|
| 1 | D-063 — retention branch returns rc=10; guard keys on `_GR_HOLD_PATH` | **VERIFIED** |
| 2 | D-062 — competing `EXIT` trap removed from `run_v2_rescore_final.sh` | **VERIFIED** (with a minor defect: see 2b) |
| 2b | the same file's claim that its two temp files are "removed inline" | **DEFECT FOUND** (claim false; files leak) |
| 3 | R1 — `_GR_PROD_NVSMI` absolute so telemetry cannot be substituted | **DEFECT FOUND** (defeated; v2's exact measurement reproduced) |
| 4 | R3 — signal handlers emit a handler-only marker; TERM/HUP likewise | **VERIFIED** |
| 5 | T36 — nine new assertions; suite 327/0 | **327/0 reproduced**; two assertions are symptom-only; T36c leaks a deadlocked process pair per run |
| — | `STATUS_CURRENT_v2.md` §1.9 (lead-written) | **two misgradings** — R3 graded "independently verified" by its own fixer; T36 row undercounts its own assertions |

**MILESTONE 1 VERDICT: PASS**, with the residuals named in the last section.

---

## 1. D-063 — second acquire after a retention-branch drop — **VERIFIED**

Code (`scripts/gpu_reserve.sh`):
* retention branch `:1855-1856` sets `GPU_RESERVE_RC=10` and `return 10`, and deliberately leaves
  `_GR_HOLD_PATH` set while `_gr_close_hold_fd` clears `_GR_HOLD_FD`;
* the second-acquire guard `:1594` keys on **either** `_GR_HOLD_FD` **or** `_GR_HOLD_PATH`.

I did not accept that from the source. I drove the whole composition in one shell in test mode — a
real `gpu_reserve_hold`, a real survivor holding the liveness lock (detached from stdout), a real
retention drop, then a real second acquire:

```
ACQ1 rc=0 gpu=0 hold=.../locks/.hold.2345060.d063a
DROP rc=10  (expect 10)  hold_path_after=.../locks/.hold.2345060.d063a  hold_fd=<empty>
ACQ2 rc=2 (expect 2=REFUSED) gpu=0
HANDLER FIRED rc=10
```

All three properties hold: the retention branch returns a distinct **rc=10** (not 0); the retained
`_GR_HOLD_PATH` genuinely **refuses** the second acquire with rc=2 (no second GPU handed out, no
orphaned hold file); and a caller's `|| handler` is **live** again (`HANDLER FIRED rc=10`), so
`run_r1_imma_official.sh:194`'s `|| say WARNING` is no longer dead code. This is the one item the
lead had previously reported fixed from a code comment; this time the behaviour is real.

## 2. D-062 — the competing `EXIT` trap — **VERIFIED**

`grep -n 'trap' run_v2_rescore_final.sh` returns **only comment lines** (`:88, :89, :94, :96`). There
is no executable `trap` statement anywhere in the file, so nothing competes for the single trap slot
with the release trap `gpu_reserve_hold` installs at `gpu_reserve.sh:1767`. `scripts/run_guard.sh`
installs no `EXIT` trap either, so the reservation's trap is the only one in that shell. The removal
is genuine and, as the comment argues, outright removal is the right repair rather than chaining.

### 2b. ...but the file's stated reason is false — **DEFECT FOUND** (minor)

`run_v2_rescore_final.sh:94-95` justifies removing the trap by asserting:

> "These two temp files live for a few lines and are **removed inline**, so no trap is needed at all"

They are not removed at all. `V2_BEFORE` and `V2_AFTER` are created by `mktemp` at `:86-87`, used at
`:97/:101/:104`, and there is **no `rm` of either anywhere in the file** (`grep -n 'rm '` finds none).
Both files leak into `${TMPDIR:-/tmp}` on every run, including clean exits.

Impact is small — two small files per run — but the defect is the one this review series keeps
finding: a correct change supported by a statement about the code that is not true of the code. The
trap must still not be re-added; the fix is an inline `rm -f "$V2_BEFORE" "$V2_AFTER"` after `:104`.

## 3. R1 — production telemetry is still substitutable — **DEFECT FOUND**

`_GR_PROD_NVSMI` is now resolved to an absolute path at load time (`:178-181`, probing
`/usr/bin`, `/bin`, `/usr/local/bin`) and left empty if none exists. That **does** close the `$PATH`
route, and I confirmed it:

```
# ATTACK 1: PATH prepend with a fabricating shim
PATH="$ATK:$PATH" bash scripts/gpu_reserve.sh status
  mode=production
  telemetry=OK  gpus=[3 2 1 0] via /usr/bin/nvidia-smi
  0  free  FREE  -  23  GPU-b2f89296-447e-dc7a-96b6-82ac699a02d7     <- real UUID, attack failed
```

But the property claimed — that production telemetry cannot be substituted — is false. Telemetry is
not invoked directly; `_gr_nvsmi_run:876` wraps it:

```bash
if [ "$NVSMI_TIMEOUT" != 0 ] && command -v timeout >/dev/null 2>&1; then
  timeout -k 5 "$NVSMI_TIMEOUT" $NVSMI "$@" >"$out" 2>"$out.err"
```

`timeout` is a **bare name**. Absolutising `$NVSMI` moved the substitutable element one level up
rather than removing it. Two independent routes, both measured:

**Attack 4 — `BASH_ENV` defining a `timeout` function, `$PATH` untouched:**

```
BASH_ENV=$ATK/benv2.sh bash scripts/gpu_reserve.sh status
  mode=production
  telemetry=OK  gpus=[1 0] via /usr/bin/nvidia-smi
  0     free  FREE  -  0  GPU-ffff
  1     free  FREE  -  0  GPU-eeee
```

**Attack 5 — one exported shell function in the caller's shell; no `BASH_ENV`, no `$PATH` edit, no file:**

```bash
timeout() { shift 3; case "$*" in *index,uuid*) printf '0, GPU-ffff\n1, GPU-eeee\n';;
                                 *memory.used*) printf '0, 0\n1, 0\n';; *) :;; esac; }
export -f timeout
bash scripts/gpu_reserve.sh status
  mode=production
  telemetry=OK  gpus=[1 0] via /usr/bin/nvidia-smi
  0     free  FREE  -  0  GPU-ffff
  1     free  FREE  -  0  GPU-eeee
```

That is **v2's finding reproduced exactly** — `mode=production`, `telemetry=OK`, fabricated
`GPU-ffff`/`GPU-eeee` at 0 MiB — and note the report line still says `via /usr/bin/nvidia-smi`, so
the one field an operator would check actively reassures them while the numbers are fabricated.
Attack 5 is the serious one: an exported function crosses the process boundary into the
`"${BASH_SOURCE[0]}" acquire` subprocess that the sourced production path uses, needs no file on
disk, and uses an ordinary function name (unlike `/usr/bin/nvidia-smi`, which `export -f` refuses).

A third route also works where `timeout` is absent from `$PATH`: bash permits a function whose name
contains slashes, and it shadows an absolute-path command, so `BASH_ENV` + a `/usr/bin/nvidia-smi`
function defeats the `else` branch at `:879` directly. I verified both halves of that
(`function /usr/bin/nvidia-smi` does shadow `/usr/bin/nvidia-smi`; `BASH_ENV` *is* sourced for
`bash script`). `GPU_RESERVE_NVSMI_TIMEOUT=0` is **not** a route — `:245` clamps it to `[1,600]`, so
the documented "0 disables the timeout" at `:121` is now stale and is the only thing that still
suggests the `else` branch is reachable by env var.

What this means for the pattern §1.9 records itself: D-042 → D-047 → D-060 → R1 is "a value was
validated while the thing it *resolved through* was not". R1's fix validated `$NVSMI` and left
`timeout` — the thing it now resolves through — unvalidated. The addendum predicted its own next
instance one paragraph before committing it.

**Minimal repair:** resolve `timeout` to an absolute path at load time the same way `$NVSMI` is, and
invoke it absolutely (quoting `"$NVSMI"` at `:876/:879/:352` while you are there — it is unquoted at
all three). Belt-and-braces: `unset -f timeout` / `enable -n` is not enough; the absolute path is.

## 4. R3 — SIGINT/TERM/HUP cleanup — **VERIFIED**

**The markers are sound proof.** `grep -rn` across every `*.sh` in the project finds the strings
`SIGNAL INT|TERM|HUP received`, `handler running`, and `handler complete` at exactly three places:
`gpu_reserve.sh:1780`, `:1781`, `:1782` — the three trap bodies themselves. There is no other code
path, in this file or any other, that can emit them. The reasoning in the fix is also right: exit
130 is produced by an unhandled kill too, and HELD→FREE is produced by the kernel closing the
descriptor regardless, so neither is proof; a string that exists only inside the handler is.

**TERM and HUP did get the same treatment** (`:1781`, `:1782`), and I measured them rather than
reading them. Each handler emits "received" *before* `gpu_reserve_drop` and "complete" *after*, so
two markers prove the handler ran to completion, and each clears the `EXIT` trap before exiting so
the release cannot double-run.

Measured, test mode, signalling the runner's **process group** (as a real Ctrl-C does), with the
target reporting its own PGID so the signal provably reached the runner shell and not a `setsid`
wrapper:

```
### TERM  markers=2    SIGNAL TERM received ... / SIGNAL TERM handler complete
                       records left=0  holdfiles=0
### HUP   markers=2    SIGNAL HUP received ...  / SIGNAL HUP handler complete
                       records left=0  holdfiles=0
### INT  (10 consecutive trials)
  trial 1..10: markers=2 records=0 holdfiles=0 -> CLEAN
  SIGINT: handler-ran+clean=10  handler-skipped/leak=0  of 10
```

All three signals run the handler, emit exactly 2 markers, and remove **both** the reservation record
and the hold file.

*One honest note on my own measurement, recorded because §1.9's measurement-discipline note is
right.* An earlier structuring of my probe reported `markers=0` and a leaked record for INT. I did
not report that as a defect, because the **plain-bash control** in the same run — a bare
`trap 'echo ...' INT; sleep 300` with no `gpu_reserve.sh` involved — failed identically. A harness
whose control fails is not measuring the subject. Rebuilt with the target reporting its own PGID,
both control and subject behave, 10/10. This is the fourth harness error in this round (three
lead-side, one mine) and the lesson generalises: a negative signal result is worthless without a
control that shows the signal arrived.

## 5. T36 and the suite — **327/0 reproduced**; coverage is partial

**My counts, my run** (`bash scripts/test_gpu_reserve.sh`, full suite, synthetic shim, no GPU):

```
RESULT: 327 passed, 0 failed
No GPU was accessed; nvidia-smi was the shim at /data/bijaypandey/tmp/gpu_reserve_test.6kdv8Q/fake_nvsmi
real 3m35.058s
```

**327 passed / 0 failed, matching the claim** (v2 saw 318/0; the delta is exactly the nine new
assertions). Runtime 3m35s — no sign of the ten-minute stall.

### Per-assertion judgement — property or symptom?

| assertion | result | judgement |
|---|---|---|
| T36a ×2 — `PROJECT` refused rc=6 by `status` **and** `acquire` | PASS | **Property.** Exercises real dispatch on both entry points; the second closes the "hardened the reporter, not the actor" gap. Confirmed independently: both rc=6 |
| T36b #1 — `$PATH` cannot substitute telemetry | PASS | **Symptom.** Tests the one route that was fixed, not the property. My attacks 4/5 pass through it untouched — the assertion passes while production telemetry is still fabricable |
| T36b #2 — telemetry reported `via /` | PASS | **Symptom, and the weakest of the nine.** A substring match on a status line. Under attack 4/5 this assertion *still passes* and the line it matches is actively misleading |
| T36c — retention drop returns 10 | PASS | **Property** for the return code, with a real `flock` survivor. See the process-leak defect below |
| T36d — second acquire refused on a retained hold path | PASS | **Symptom.** `_GR_HOLD_PATH=/tmp/t36d.fake` is **planted by hand** and `_GR_HOLD_FD` unset; no drop is ever called. It asserts the guard's `if`, not the composition. T36c proves the drop retains the path and T36d proves the guard honours a planted path, but **nothing tests the two together** — the exact defect v2 measured. Were the retention branch changed tomorrow to clear `_GR_HOLD_PATH`, both assertions would still pass and D-063 would be back. (I closed this gap by hand in §1; it is not closed by the suite) |
| T36e — `RG_PY` absolute after a `cd` | PASS | **Property.** Sources the real guard from `/` |
| T36f ×2 — `rg_manifest` non-zero on failed publish; `rg_finish` refuses DONE | PASS | **Property.** Real functions, real unwritable destination, and it asserts the downstream consequence rather than just the return code |

Net: **five of nine test the property, four test the symptom**, and the two weakest (T36b) cover the
one item that is in fact still broken. That is the honest answer to v2's "none of the five fixes has
a test": they now have tests, and the tests are real, but R1's test could never have caught R1's
remaining hole and T36d cannot catch D-063's regression.

### The two self-inflicted T36 bugs

**The apostrophe fix is sound.** `:1780-1782` quote the campaign name as
`'"'"'$_GR_HOLD_CAMP'"'"'` inside the single-quoted trap body — correct concatenation, and the
markers rendered correctly in my measured output, apostrophes and all.

**The stdout fix is *not* sound — DEFECT FOUND (should-fix).** `:1894` adds `>/dev/null 2>&1` to the
backgrounded survivor, which does stop the command substitution's pipe from being held open, so the
suite no longer stalls for ten minutes. But it fixed the *visible stall*, not the *deadlock*. The
parent takes `flock -x "$_GR_HOLD_FD"` at `:1890` and never releases it; the survivor at `:1894`
blocks on `flock -x 9` for that same lock; the parent then waits for the child. Measured right now:

```
pid 2293374  ETIME 45:46  S  do_wait   bash -c  ... _GR_HOLD_CAMP=t36c ...
pid 2293375  ETIME 45:46  S  locks_    flock -x 9
pid 2325026  ETIME 18:58  S  do_wait   bash -c  ... _GR_HOLD_CAMP=t36c ...
pid 2325028  ETIME 18:58  S  locks_    flock -x 9
```

Three such pairs were alive when I looked — from three separate suite runs, one of them mine — the
oldest 45 minutes old, parents in `do_wait`, children blocked in `locks_`. **No `timeout` process was
alive.** So `timeout 25` did **not** bound them: it exited, and the orphaned pair deadlocked
permanently. Every suite run leaks one such pair, each holding an `flock` on a `mktemp` file in
`$TMPDIR` forever.

**On whether the `timeout` bounds can mask a real hang as a pass — they cannot, and that part is
well built.** Every T36 probe emits its verdict through `echo` *after* the work, and the assertion
compares against a specific non-empty value (`"10"`, `"2"`, `MANIFEST_FAILED`, a leading `/`). If
`timeout` kills a probe, the capture is empty or truncated and the assertion **fails** — it does not
silently pass. I confirmed the shape of all four. The leak above is therefore process debt and a
false impression of boundedness, not a false green.

## 6. `docs/STATUS_CURRENT_v2.md` §1.9 — two misgradings

Checked all nine dispositions against the evidence.

**Correctly graded:** D-059, D-060, D-061 ("Fixed and independently verified" — each has genuine
reviewer involvement in the *property*, and D-060 I re-confirmed at rc=6 on both `status` and
`acquire`); D-062 and D-063 (**"Fixed (not yet independently reviewed)"** — honest at the time, and
both now verified here, D-063 fully, D-062 with the 2b caveat); R2 ("Deferred", fails closed at
rc=127, no launcher exposed).

**Misgraded — the specific thing I was asked to look for:**

> **R3 is graded "Fixed and independently verified" on the strength of verification by the person who
> fixed it.** The basis column reads "Handlers now emit a stderr marker ... **Measured: 2 markers,
> lock FREE**" — that measurement is the lead's, and the lead wrote the marker fix. The reviewer
> credit in that row is for *explaining why the earlier approaches could not settle it*, which is not
> verification of the fix. At the time §1.9 was written, no independent party had verified R3. It
> should have read "Fixed (not yet independently reviewed)", exactly as D-062/D-063 do.
>
> This matters beyond bookkeeping because it is the **same error as D-063's first disposition**, where
> the lead reported a fix verified on the strength of a code comment and the reviewer measured that it
> was not. The grading error recurred in the very table that records the pattern.
>
> *The substance is nonetheless sound:* R3 **is** now independently verified — by me, in §4, across
> INT/TERM/HUP with 10/10 on INT. The grade was unsupported when written; the claim turns out true.

> **R1 is graded "Fixed", and it is not.** The basis asserts the resolution "fail[s] closed" and that
> the "Lead verified by attack: `via /usr/bin/nvidia-smi`, no fabricated UUID." That attack was the
> `$PATH` route only. §3 defeats the property by two other routes and reproduces the fabricated
> UUIDs. Grade should be **"Partially fixed — `$PATH` route closed, substitution still possible via
> the unvalidated `timeout` it resolves through."** Note the lead's attack and T36b are the *same*
> attack, so the test adds no coverage beyond the manual check.

**Also wrong, minor:** the **T36 row says "Six assertions added"** — there are **nine**
(T36a×2, T36b×2, T36c, T36d, T36e, T36f×2; and 327 − 318 = 9). The row also says "result pending at
time of writing" although the commit message for this very revision states 327/0. And "each
exercising the real function or dispatch rather than a source grep" is overstated for T36d (plants
state, calls no drop) and for T36b #2 (a substring match on report text).

---

## MILESTONE 1 VERDICT: **PASS**

Milestone 1 is "shared GPU infrastructure safe enough to execute bounded verification". Four of the
five fixes under review are real, complete, and now **independently** verified against live
behaviour rather than source reading — including the two the lead had previously mis-certified
(D-063 from a code comment, R3 by its own author). The suite reproduces at **327 passed / 0 failed**
on my run, and its `timeout` bounds fail closed rather than masking hangs as passes. The controls
that actually stand between this project and another user's hardware are intact and were confirmed
settled: the production allowlist `{0,1}` is hard-wired and no environment variable can widen it,
`PROJECT`/`LOCKDIR`/`FREEZE_FILE`/`NVSMI`/`OVERRIDE_LOG` are all refused rather than ignored in
production, and the reservation mutex cannot be relocated.

### Residuals I am willing to carry, and why they do not block bounded verification on GPUs 0-1

1. **R1-bis — production telemetry substitutable via the unvalidated `timeout` (§3). SHOULD-FIX, carried.**
   This is a real defect and the grade in §1.9 is wrong, but it does not gate these runs. It requires
   an exported shell function named `timeout` in the launching shell, or a `$PATH` with no `timeout` —
   neither occurs accidentally, and neither is present in this project's launch path (I ran the real
   `status` with the project's own environment and got real UUIDs via `/usr/bin/nvidia-smi`). Crucially,
   **spoofed telemetry cannot widen the blast radius**: the allowlist is a hard-wired literal, so even
   fully fabricated telemetry can only affect GPUs 0-1, which are ours, and cannot reach GPUs 2-3. And
   it cannot double-book against another reservation-API holder, since that is excluded by the mutex
   and the records, not by telemetry. The exposure is "we could trample our own idle-looking GPU 0/1",
   on a frozen project, under a deliberate act. Fix it before any multi-user or unattended launch;
   it does not block a bounded, attended verification run.
2. **T36c leaks a deadlocked `bash`+`flock` pair per suite run (§5). SHOULD-FIX, carried.** Confined to
   `$TMPDIR` (`/data/...`, not the full root fs) and to the test harness; touches no GPU and not the
   production lockdir. Kill the strays (`2293374/5`, `2325026/8` were alive at review time) and release
   the parent's lock before backgrounding the survivor, or `wait` with a bound.
3. **Two temp files leak from `run_v2_rescore_final.sh` and its comment says otherwise (§2b). SHOULD-FIX, carried.**
   Two small files per run. The trap must stay removed.
4. **T36d does not test the D-063 composition, and T36b cannot catch R1's remaining hole (§5). CARRIED.**
   Coverage gaps, not defects. I closed the D-063 composition by hand in §1; the suite should close it
   properly (do a real retention drop, then a real second acquire, in one shell) so the regression
   cannot return silently.
5. **Stale documentation: `gpu_reserve.sh:121` ("0 disables the timeout", now clamped to ≥1) and
   §1.9's T36 row (six assertions → nine; "result pending" → 327/0). CARRIED.**

### Process finding attached to the PASS

Two of the three gradings in §1.9 that claim independent verification were written by the person who
did the work, and one of those (R3) had no independent verification at all when written. The code is
in good shape; the certification discipline is the weaker half of this milestone. Recommend that
"independently verified" be reserved for a named party other than the author, and that §1.9's R1 and
R3 rows be corrected per §6 — a documentation change I am reporting rather than making, since this
review owns only this file.
