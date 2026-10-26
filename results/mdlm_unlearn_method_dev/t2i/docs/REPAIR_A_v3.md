# REPAIR A v3 — reservation residual, truthful failure handling, review of the D-049 fix

Agent A, bounded repair cycle, 2026-10-26. CPU only. **No GPU work was launched by me**; nvidia-smi
was a synthetic shim throughout and no CUDA context was created. GPUs 2–3 and
`/data/bijaypandey/mdlm_unlearn` were not touched.

## Status words used in this document

These are kept distinct on purpose, because the previous closure of task 1 did not keep them
distinct.

| word | means |
| --- | --- |
| **implemented** | the code exists and parses. Nothing more. |
| **executed** | a test ran, in this cycle, and its real output is pasted below. |
| **independently verified** | *not claimed anywhere in this document.* I wrote both the fix and the test. |
| **needs a lead-launched GPU run** | cannot be settled on CPU with a mocked driver. Listed in §4. |

Files I changed: `scripts/gpu_reserve.sh`, `scripts/test_gpu_reserve.sh`, `scripts/run_guard.sh`,
`run_r1_imma_official.sh`, `run_r1b_imma_train.sh`, `run_r1c_eval.sh`,
`run_v1_esd_alone_control.sh`, `run_v2_rescore_final.sh`, and this file. I did not touch
`run_r2_meta_official.sh`, `scripts/split_registry.py`, or `scripts/r4c*/erase_uce.py`.

---

## 1. Task 1 — the reservation residual

### 1.1 The defect, verified from source

`scripts/gpu_reserve.sh:719-720`

```sh
  # pgid is evidence of descent only for a group LEADER (see (2) above).
  if [ -z "$owner" ] || [ "$want" != "$owner" ]; then return 2; fi
```

`_gr_live_in_pgid` is the only mechanism that could detect a surviving descendant of a dead
reservation owner, and it refuses to answer unless the owner **led its own process group**
(`recorded pgid == recorded owner pid`). A runner launched the ordinary way — `bash
run_r1_imma_official.sh` from any non-interactive parent — inherits its parent's process group, so
`pgid != pid`, this guard returns 2, and `_gr_reservation_state` falls through to the CUDA-context
check and then to a timer (`scripts/gpu_reserve.sh:220`, `ORPHAN_GRACE` default **900 s**).

So before this repair the protection for the ordinary launch path was: *a timer*. Two things
followed, and neither was a fix:

* **Raising the grace window to 900 s is not a mechanism.** It is a bet that no job's owner dies
  more than 900 s before its children do. A cold model load can exceed it, and a long training run
  exceeds it by hours.
* **`setsid` in a launcher is not a mechanism either.** It is a caller-side spelling. The
  reservation cannot verify it, cannot enforce it, and every other invocation path omits it. The
  old `run_v2_rescore_final.sh` header called its `setsid` re-exec "closes the last reservation
  residual"; it closed nothing, because leadership only decides whether `gpu_reserve.sh` is
  *permitted* to consult pgid — the reclamation decision was still the timer for everything else.

### 1.2 The fix — a kernel-enforced liveness lock

A reservation now also owns an `flock(2)` taken by the **long-lived runner shell** on a file
descriptor (`scripts/gpu_reserve.sh:1590` in `gpu_reserve_hold`). An flock lives on the *open file
description*, so it is shared with every process that inherits the descriptor — across `fork` and
across `execve` — and the kernel releases it only when the **last** such descriptor closes. A
would-be acquirer therefore stops inferring descent from `/proc` and simply asks the kernel
"may I take this lock?"; a refusal is positive proof that something from that reservation is alive
(`scripts/gpu_reserve.sh:1057`).

This is independent of group and session leadership, survives re-parenting to init, survives
`SIGKILL` of the owner (nothing has to run in the owner — the kernel does the accounting), and
holds from *before* the record is published until *after* the last descendant exits, which is why
it also covers a job that has not yet created a CUDA context.

Ordering is load-bearing and is the reason this is not just "another check": `gpu_reserve_hold`
takes the lock **before** calling `acquire`, and `acquire` **verifies the lock is actually held**
before it will write a record naming it (`scripts/gpu_reserve.sh:1245`). A record that merely
*named* a hold path would promise a protection that did not exist, and every later reclamation
decision trusts that field.

Also recorded and verified per the task: owner pid, process start time, **session id** and
**process group id**, and the GPU UUID. `_gr_sid` was added; `sid=` and `holdpath=` were added to
the record; and `acquire` now prints the regime it is actually operating under rather than the one
the caller hoped for:

```
[gpu_reserve] gpu0 RESERVED for 'manualcamp' by pid 1875004 (uuid GPU-aaaa)
[gpu_reserve]         owner identity: pid=1875004 start=139966575 pgid=1875004 sid=1874999
[gpu_reserve]         group-leader=yes session-leader=no
[gpu_reserve]         supervision: liveness-lock (kernel-enforced; descendants covered)
```

The pgid scan is kept as a second line of defence for legacy records (`holdpath=-`) and for hosts
with no usable `flock(1)`. The grace window is kept as a backstop and is no longer the primary
protection. Nothing was weakened; a strong arm was added ahead of the weak ones.

**Wrappers that fork.** The task asked that the owner may not be the process holding CUDA. The lock
handles this structurally rather than by guessing: it does not care which process holds CUDA, only
that *some* inheritor is alive. Measured (§1.4, M2).

### 1.3 Supporting changes this forced

* **`SIGTERM` no longer hands away a live GPU.** The old trap ran `release` unconditionally, so
  interrupting a runner deleted its record while its python child was still training — the same
  defect as the dead-owner case, on the signal path. Teardown now goes through `gpu_reserve_drop`,
  which closes *our* descriptor first (until it is closed, every lock test says BUSY), then asks
  the kernel whether anything else still holds it, and **keeps the record if so**. The reservation
  then reclaims itself when the last descendant exits — no timer, no manual step.
  **Measured for SIGTERM and SIGKILL in §6.4**; SIGINT is the one case I could not deliver from a
  backgrounded harness and is flagged for a foreground run.
* **Signals now terminate.** The old traps bound `INT`/`TERM`/`HUP` to a release command with no
  exit, so a `SIGINT` ran the handler and the runner **carried on executing** with its reservation
  deleted. Now `exit 130/143/129` (`scripts/gpu_reserve.sh:1669-1671`).
* **A second reservation in one shell is refused** (`scripts/gpu_reserve.sh:1492`). This was
  previously a documented-and-asserted leak; with a liveness lock it would also orphan the first
  descriptor, pinning that GPU for the life of the shell. `run_r1_imma_official.sh` has exactly
  this shape, and now drops between stages (§3).

### 1.4 Direct measurements taken before and while writing the code

**M1 — the premise of the whole design.** `bash 5.1.16`, `util-linux flock 2.37.2`. Owner took
`exec {FD}>>hold; flock -x $FD`, spawned a child shell that did `exec sleep 300` (so the survivor
is an *execed binary*, not a bash fork), then the owner was `SIGKILL`ed:

```
owner pid=1856298 fd=10 locked; pgid=1856298 sid=1856292
--- owner alive, external flock -n attempt ---
RESULT: lock BUSY (held) rc=1
--- now SIGKILL the owner, leaving child+grandchild alive ---
owner /proc present? no
grandchild 1856307 alive? yes
--- external flock -n attempt AFTER owner SIGKILL ---
RESULT: lock BUSY -> descendants DO hold it
```

**M2 — inheritance through the real shapes a training job uses.** A bash owner `exec`s into python,
which creates three children; then the python owner is `SIGKILL`ed:

```
owner=1886152 fd=10
python pid=1886152; inherited fd 10 open? True
fork_child=1886155 subproc_default=1886156 subproc_passfds=1886157
--- killing the python owner (1886152), leaving its 3 children ---
  fork_child       pid=1886155 alive=yes holds_lock_fd=yes
  subproc_default  pid=1886156 alive=yes holds_lock_fd=no
  subproc_pass_fds pid=1886157 alive=yes holds_lock_fd=yes
--- can an outsider now take the lock? ---
RESULT: lock BUSY (reservation still protected)
```

This is the honest boundary of the mechanism, and it is a real limitation, not a formality:

* `exec` into python — **inherited**.
* `os.fork` child, then `execv` (this is how DataLoader workers and `torch.multiprocessing` fork) —
  **inherited**.
* `subprocess.Popen(...)` with Python's **default `close_fds=True`** — **NOT inherited**.
* `subprocess.Popen(..., pass_fds=(fd,))` — inherited.

So a grandchild spawned through plain `subprocess` does not itself hold the lock. In the r1 path the
chain is `bash runner → exec accelerate/python → trainer`, and the intermediate accelerate process
*does* hold the lock and lives for the whole run, so the reservation stays protected for the
duration. The uncovered case is narrow and specific: **the intermediate python dies while a
`subprocess`-spawned trainer keeps running.** Confirming the actual process tree of
`accelerate.commands.launch` needs a real run — see §4.

### 1.5 Requirements (a)–(g): what was executed

See §5 for the pasted suite output. Summary of how each was exercised, all through the **real**
`gpu_reserve_hold` path in runners written with the production options `set -euo pipefail` (only 4
of the 139 runners use these, and they are r1/r1b/r1c/r2):

| req | test | how it is distinguished from a non-fix |
| --- | --- | --- |
| (a) concurrent launchers cannot share a GPU | T28 | 8 concurrent real launchers, 2 winners, distinct indices asserted, 6 refusals asserted |
| (b) surviving **child** blocks reclamation | T29 | runner SIGKILLed; the ORPHAN **reason string** must be `liveness-lock-held-by-pids-[…]`, not `within-…-grace` |
| (b′) surviving **grandchild** (fork+exec) blocks it | T30 | survivor is an execed binary re-parented to init (`ppid=1`), so ancestry is destroyed; also asserts it becomes reclaimable *only* after that survivor exits |
| (c) delayed CUDA init | T31 | telemetry reports **zero** compute apps and the record is aged 1 h past the grace window, so neither the CUDA arm nor the timer can be what protects it |
| (d) normal completion + SIGINT/SIGTERM/SIGKILL | T32 (d1–d4) | includes the case the old trap got wrong: SIGTERM **with a live child** must keep the record |
| (e) inherited env cannot bypass | T33 | includes the bypass surface this repair *added*: a forged/unlocked `holdpath`, and one outside `$LOCKDIR` |
| (f) GPUs 2–3 unavailable under every env permutation | T33 | 11 permutations in production mode; also asserts no `gpu2/gpu3` record is ever created |
| (g) failures reach a non-successful status | T34 | refused reservation ⇒ non-zero runner exit **and** no work ran; plus the signal-fallthrough case |

Every ORPHAN assertion checks the **reason string**. A test that accepted either
`liveness-lock-held-by-pids` or `within-900s-grace` would pass against the old non-fix, which is
how the previous closure survived a 50-test suite.

---

## 2. Task 2 — truthful failure handling, resumed path only

Scope held to the r1 IMMA path, `run_v1`, `run_v2`. The ~104 historical runners stay byte-frozen.

### 2.1 Correction to the brief's premise, measured

**The 33 `2>/dev/null || echo 0` sites are not in the resumed path.**

```
$ grep -n '|| echo 0' run_r1_imma_official.sh run_r1b_imma_train.sh run_r1c_eval.sh \
      run_v1_esd_alone_control.sh run_v2_rescore_final.sh scripts/run_guard.sh
(no matches)
$ grep -rn '|| echo 0' --include=*.sh . | wc -l
33
```

All 33 are in the frozen historical scripts. Likewise, of the files containing the "below K1 /
destroys the model" gate text, the only resumed-path file is `run_r1c_eval.sh`.

**And in `run_r1c_eval.sh` the gate does not zero on a crash.** The widely-cited failure — a
crashed `json.load` printing `retain8=0.0000 … destroys the model` — does **not** reproduce there,
because that file runs under `set -euo pipefail` and the failing command substitution trips errexit
first. Measured on the pre-repair lines:

```
$ # absent file
Traceback (most recent call last): FileNotFoundError: 'outputs/gen_NOSUCH.json'
k1demo exit=1
$ # file present, key missing
Traceback (most recent call last): KeyError: 'legacy_retain8'
k1demo2 exit=1
```

I am stating this rather than inheriting it, because the real bug there is the **opposite** one and
would have been missed by fixing the described one.

### 2.2 The actual defect in the K1 kill-rule gate — it fails OPEN

Pre-repair `run_r1c_eval.sh`:

```sh
R8=$(venv_t2i/bin/python -c "...json.load(...)['active_target_metrics']['legacy_retain8']")
awk -v r="$R8" 'BEGIN{exit !(r>=0.60)}' || { echo "K1 fail (retain8=$R8), not attacked" >>"$L"; exit 0; }
```

`awk` comparisons are untyped, so a non-numeric `r` is compared as a **string**, and any string
beginning with a letter sorts above `"0.60"`. Measured value by value:

```
json value null     -> R8=[None] -> gate CLEARS -> attack grid runs
json value "NaN"    -> R8=[NaN]  -> gate CLEARS -> attack grid runs
json value 1e400    -> R8=[inf]  -> gate CLEARS -> attack grid runs
json value "0.95"   -> R8=[0.95] -> gate CLEARS -> attack grid runs
```

So a malformed-but-non-crashing metric did not fail the kill rule — it **passed** it and launched
the three-seed attack grid on a value that is not a number. The gate also gave `exit 0` to both
"metric unreadable" and "metric genuinely below K1", which mean opposite things.

**Fix** (`run_r1c_eval.sh:323-356`): read the metric through `rg_metric`, which yields a value only
if the field is present, numeric, finite and in `[0,1]`, and records a failure otherwise instead of
substituting one; compare numerically in python, never in `awk -v`; and split the two outcomes —
unusable metric ⇒ `exit 1` with `rg_finish` refusing DONE; genuinely below K1 ⇒ clean `exit 0`,
labelled a result.

### 2.3 Existence/validity, hashes, loadability

The named case is real and still on disk:

```
$ ls -l adapters/utrap_church_l1.0k4/unet.pt
-rw-rw-r-- 1 bijaypandey bijaypandey 511229952 Oct 25 13:55 unet.pt      # non-empty: passes [ -f ] and [ -s ]
$ torch.load(...)
LOAD FAILED: RuntimeError PytorchStreamReader failed reading zip archive: failed finding central directory
```

`run_r1c_eval.sh`'s reuse gates checked `[ -f "$dir/unet.pt" ]` plus hashes recorded in
`compose.json` — but those hashes describe the **inputs** to the composition, never the composed
`unet.pt` itself, so a valid `compose.json` beside a half-written `unet.pt` read as "current" and
the truncated file would have been reused for evaluation. Added `unet_loads()` (`run_r1c_eval.sh:83`)
and a JSON-parse check on the result file in `eval_is_current`.

New reusable primitives in `scripts/run_guard.sh`:

| function | purpose |
| --- | --- |
| `rg_require_loadable` (:231) | opens/parses the artifact; rejects empty state dicts and non-finite tensors |
| `rg_metric` (:289) | present + numeric + finite + in range, or nothing — never a substitute value |
| `rg_ratio_ok` (:332) | validates the **denominator** before anything divides by it |
| `rg_reuse_ok` (:351) | hash **and** config identity **and** loadability, all three |
| `rg_manifest` (:404) | completion manifest published only after every artifact validates |
| `rg_atomic_write` (:193) | same-dir temp → `fsync` → `os.replace` → `fsync` dir |

Wired in at `run_v1_esd_alone_control.sh:76,87,93`, `run_v2_rescore_final.sh:75,80`, and
`run_r1c_eval.sh:350,389`.

### 2.4 Atomic writes

`run_r1b_imma_train.sh` had the worst shape: `b = json.load(open(budp))` … `json.dump(b,
open(budp,"w"))`. `open(path,"w")` truncates **before** any new byte is written, and because it
re-opens the file it just parsed, an interruption destroys the original with nothing to fall back
on — i.e. our own writer manufacturing the truncated-artifact class from §2.3. Both writers now go
through an inline `_atomic_json` (same-dir temp, flush, fsync, `os.replace`, dir fsync).

### 2.5 Defects I introduced and caught in this cycle (1-3; see §6.5 for 4-5)

Recorded because they are the same classes as the ones being repaired. Five in total across this
cycle, which is itself worth stating: a repair pass is not a safe operation.

1. **`exec {FD}>>file 2>/dev/null` silences the shell permanently.** `exec` with redirections and no
   command applies them to the *shell*, so that `2>/dev/null` swallowed the CVD audit, the RESERVED
   line and every refusal diagnosis for the rest of the run. Caught by the existing suite: 12
   assertions went pass→fail, all of them "the explanation was not printed". Fixed; the 12 now pass.
2. **`rg_metric`'s failure count was lost to the subshell.** It must be called as
   `V=$(rg_metric …)`, so every `RG_FAILS++` happened in a command substitution and vanished — the
   FAIL line reached the log, the count did not, and `rg_finish` would then have written **DONE** for
   a run whose metric could not be read. Measured: 4 induced failures, 2 counted. Fixed with an
   on-disk failure ledger (`_rg_record` / `rg_failcount`, `scripts/run_guard.sh:58,68`); re-measured:
   `failcount now = 2 (RG_FAILS in-shell = 0)` and `rg_finish rc=1`.
3. `rg_atomic_write`'s first version passed its python program on stdin *and* tried to read the
   payload from stdin, so it wrote a **zero-byte file and returned 0**. Fixed and re-measured
   (`atomic rc=0 size=7`, empty payload refused, no temp litter).

---

## 3. Task 3 — review of the lead's unconditional-reservation fix in `run_r1_imma_official.sh`

**The direction is right.** Removing `if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]` from around the stage-4
reservation is correct: a reservation an inherited variable can switch off is not a reservation, and
treating a pre-selected device as an error rather than a shortcut is the right default-deny
judgement. **But as committed it was wrong in one way and incomplete in three.**

**(1) Wrong — it made `R1_MODE=prep` abort on its own output.** Stage 2 does
`export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` after reserving. The new stage-4 guard then tested
`[ -n "${CUDA_VISIBLE_DEVICES:-}" ]` and `die`d. So `R1_MODE=prep` from a clean working tree — the
only mode that runs stage 2 — **always** aborted at stage 4, complaining about a value this very
script had just set, unless `R1_ACK_INHERITED_CVD=1` was passed. The guard's own comment anticipates
"a shell left over from an earlier stage of this very script" but the code does not distinguish that
case. Fixed by snapshotting the inherited value **once at entry**, before any reservation exists
(`run_r1_imma_official.sh:70-98`), and enforcing the refusal there.

**(2) Incomplete — `-n` tests emptiness, not set-ness.** An exported-but-empty
`CUDA_VISIBLE_DEVICES=""` passes `[ -n … ]` and so escaped the refusal entirely, yet it hides every
GPU from CUDA and is just as much a pre-selection. `gpu_reserve.sh` itself uses set-ness
(`${CUDA_VISIBLE_DEVICES+x}`); the runner now matches it.

**(3) Incomplete — the two reservation sites were left inconsistent, and the script reserves
twice in one shell.** Stage 2 and stage 4 each call `gpu_reserve_hold` with **no release in
between**. `gpu_reserve.sh`'s own header has warned since v2 that the second call's EXIT trap
replaces the first, leaking a reservation; with a liveness lock it would also orphan the first
descriptor and pin that GPU for the life of the shell. `gpu_reserve.sh` now refuses the second
acquire outright, and stage 2 now calls `gpu_reserve_drop "r1_imma_prep"` when it finishes
(`run_r1_imma_official.sh:194`). Without that drop the fixed script could not reach stage 4 at all.

**(4) Incomplete — both sites re-export the bare index, silently defeating
`GPU_RESERVE_CVD_STYLE=uuid`.** `gpu_reserve_hold` already exports `CUDA_VISIBLE_DEVICES`, and sets
it to the device **UUID** when that style is selected — the one setting immune to driver
renumbering. The runner's `export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` then overwrote it with the
index. The line that looked like it was enforcing the reservation was undoing part of it. Both
re-exports removed.

---

## 4. What I could not fix, and what needs a lead-launched GPU run

**Not fixed / out of my hands:**

* The `subprocess` + `close_fds=True` gap in §1.4 (M2). I cannot close it from the files I own: it
  would need `pass_fds` (or an inherited-fd convention) inside the training entry points, and
  `run_r2_meta_official.sh` and the `erase_*`/`r4c*` scripts are Agent B's. The practical exposure
  is limited to "the intermediate python dies while a subprocess-spawned trainer keeps running".
* `GPU_RESERVE_ORPHAN_GRACE` cannot be set below 60 s, so the "record ages out" transitions are
  tested by back-dating the record's mtime rather than by waiting.
* A stray process that inherits the descriptor and is *meant* to outlive the run (a deliberate
  `nohup`) will keep the reservation held. This is the safe direction — it refuses a GPU rather than
  trampling one — and `gpu_reserve.sh holders <index>` names the pids. It is a behaviour change an
  operator should know about, not a bug.
* `run_v1` scores `gen_eval.py` with `venv_imma/bin/python` while `run_r1c_eval.sh` scores the same
  script with `venv_t2i/bin/python`. I did not change this — it is a scoring-comparability question,
  not an infrastructure one — but two arms compared against each other should not be scored under
  different interpreters. Flagging it for whoever owns the eval protocol.

**Needs a lead-launched GPU run (I did not and must not launch GPU work):**

1. **The actual process tree of `accelerate.commands.launch`** under `run_r1_imma_official.sh`
   stage 4 — specifically whether the process holding the CUDA context inherits the liveness-lock
   descriptor. Everything in §1 is mocked-telemetry CPU evidence plus the §1.4 inheritance
   measurements; the real chain is the one thing I cannot observe. Suggested check, during a real
   run: `bash scripts/gpu_reserve.sh holders <index>` and confirm the CUDA-holding pid appears.
2. **That a real `nvidia-smi` on this host still satisfies the telemetry gate** after my changes to
   the record format (two new fields). The probe code is untouched, but the only runs behind it here
   were synthetic.
3. **`GPU_RESERVE_CVD_STYLE=uuid` end to end.** It has still never been exercised against a real
   driver; the index form remains the default for that reason.
4. **SIGINT (Ctrl-C) against a foreground runner.** This is the one interrupt I could not measure,
   and the reason is structural rather than a gap in effort: a background job started by a
   non-interactive shell inherits SIGINT **ignored**, and bash cannot trap an inherited-ignored
   signal, so no harness that backgrounds the runner can deliver a meaningful SIGINT to it
   (measured: `SigIgn=0000000000000006`). SIGTERM and SIGKILL **are** measured and correct — see
   §6.4. What is needed from you: start a resumed-path runner as a **foreground** job in your
   terminal, press Ctrl-C while a child is alive, then check `bash scripts/gpu_reserve.sh holders
   <index>` and whether the record survives. Expected: record KEPT while the child lives, released
   once it exits.
5. **Interrupt behaviour against a real training child** — the CPU measurement in §6.4 uses
   `sleep` as the stand-in trainer. Repeating the SIGTERM case against a real `accelerate launch`
   tree is the same question as item 1 (does the CUDA-holding process hold the descriptor).
6. Note for the lead: removing the `setsid` re-exec from `run_v2_rescore_final.sh` means that script
   now runs in your terminal's foreground process group again, so **Ctrl-C reaches it**. Under the
   old `setsid` version it did not — a change justified as improving the interrupt path had in fact
   broken it.

---

## 5. Executed test output

### 5.1 The liveness lock, end to end through the real `gpu_reserve_hold` path

A runner with the production options (`set -euo pipefail`) sourcing `scripts/gpu_reserve.sh`,
synthetic telemetry, inherited `CUDA_VISIBLE_DEVICES=7,9`:

```
BEFORE
[gpu_reserve] inherited CUDA_VISIBLE_DEVICES='7,9' (INDEX_LIST) -- note that
[gpu_reserve]         nvidia-smi ignores it (measured), so reserved indices stay physical; it
[gpu_reserve]         will be OVERWRITTEN with the reserved device.
[gpu_reserve] gpu0 RESERVED for 'manualcamp' by pid 1875004 (uuid GPU-aaaa)
[gpu_reserve]         owner identity: pid=1875004 start=139966575 pgid=1875004 sid=1874999
[gpu_reserve]         group-leader=yes session-leader=no
[gpu_reserve]         supervision: liveness-lock (kernel-enforced; descendants covered)
GOT 0 cvd=0
AFTER
runner exit=0
=== record ===        (empty: released on clean exit)
=== leftover ===  .  ..  .reserve.mutex
```

And from the suite, the production regime — owner leads **neither** its group nor its session,
which is exactly the case the pgid scan refuses to reason about:

```
[gpu_reserve]         group-leader=no session-leader=no
[gpu_reserve]         supervision: liveness-lock (kernel-enforced; descendants covered)
```

### 5.2 (b) surviving child, (b′) surviving grandchild, (c) delayed CUDA init

From `scripts/test_gpu_reserve.sh` T29/T30/T31. Note each reason string:

```
== T29 (b) a surviving CHILD prevents premature reclamation
  INFO runner pid=1889245 pgid=1889241  (group leader: NO)
  PASS the runner is NOT its group's leader -- the pgid scan cannot help, as in production
  PASS the runner got a GPU (= 0)
  INFO runner /proc present after SIGKILL: no
  INFO reservation state: ORPHAN 1889245 childcamp liveness-lock-held-by-pids-[1889347,1889348]
  PASS state is ORPHAN *because the KERNEL refused the lock*

== T30 (b') a surviving GRANDCHILD -- through fork AND execve -- also prevents reclamation
  PASS the runner got a GPU (= 0)
  INFO runner pid=1908631 dead=yes; execed grandchild pid=1908723 alive=yes
  INFO reservation state: ORPHAN 1908631 gccamp liveness-lock-held-by-pids-[1908723,1908724]
  PASS ORPHAN via the liveness lock, with the owner dead and ancestry destroyed
  INFO holders: gpu0 holdpath=.../locks.livegc/.hold.1908631.gccamp
                lock=HELD by pids: 1908723 1908724
  PASS the holders view names the live pid(s) holding the lock

== T31 (c) DELAYED CUDA INITIALISATION does not make a live job look reclaimable
  PASS the runner got a GPU before any CUDA context existed (= 0)
  PASS telemetry reports ZERO compute apps (pre-CUDA, as intended) (= 0)
  INFO state (owner dead, NO cuda context, record aged 1h > 900s grace):
       ORPHAN 1937604 delaycamp liveness-lock-held-by-pids-[1937693,1937694]
  PASS still ORPHAN with BOTH the CUDA-context check and the grace timer unable to help
```

T31 is the load-bearing one: telemetry reports **no** compute apps (so the CUDA-context arm cannot
fire) and the record was back-dated **one hour** past the 900 s grace window (so the timer cannot
fire either). The reservation is held anyway, by the lock. Under the previous code this case was
`STALE` — the GPU would have been handed to the next acquirer while the job was still loading.

### 5.3 Three of my own test assertions were wrong, and are recorded rather than quietly fixed

In the first full run after the harness went in, three assertions failed. None was a product
defect; all three were my tests being wrong, in ways worth naming because two of them would have
*hidden* behaviour rather than checked it:

1. **T28 demanded that losing launchers EXIT.** They do not, and should not: "every allowed GPU is
   busy" is rc=3, which `gpu_reserve_hold` deliberately sleeps and retries. My assertion asserted
   the opposite of the documented design. Re-written to assert that the six losers are *blocked in
   the retry loop* and that none got a GPU.
2. **T34 (g1) wedged the suite** for the same reason — it blocked both GPUs and then waited for a
   runner that was correctly retrying for ever. Re-written to use a **freeze** refusal (rc=4), which
   is terminal and is the refusal the resumed path actually meets while the project is frozen. A
   (g) test built on busy GPUs tests nothing and hangs.
3. **T30 killed one of two lock holders** and read the still-protected reservation as a failure.
   That was the mechanism working: the lock is shared with *every* inheritor of the descriptor.
   Re-written to kill exactly the set `_gr_holders` reports, after which the state does become
   `STALE`.

Two others were plain stderr-capture bugs in my own assertions (`2>&1` into a value under test).

### 5.4 Task 2 — the fixed K1 gate, every case

Same runner shape as `run_r1c_eval.sh`, exercising the replaced gate:

```
json null     -> runner exit=1 :: ledger: BAD-METRIC ... legacy_retain8 (NON-NUMERIC None)
json "NaN"    -> runner exit=1 :: ledger: BAD-METRIC ... legacy_retain8
json 1e400    -> runner exit=1 :: ledger: BAD-METRIC ... legacy_retain8
json 0.95     -> runner exit=0 :: K1 CLEAR (retain8=0.95 >= 0.60) -> attack grid
json 0.42     -> runner exit=0 :: R1C DONE          (K1 fail: a RESULT, not an error)
json 1.7      -> runner exit=1 :: ledger: BAD-METRIC ... legacy_retain8 (out of range)
absent file   -> runner exit=1 :: ledger: BAD-METRIC ... legacy_retain8
```

Compare the pre-repair gate on the same inputs (§2.2): `null`, `NaN` and `1e400` all **CLEARED**
the kill rule and launched the attack grid. The three malformed cases now stop the run with a
non-zero status, the genuinely-low case stops cleanly as a result, and only a real, in-range,
finite number proceeds.

### 5.5 Task 2 — the guard primitives

```
--- (2) existence-as-validity: the real truncated checkpoint ---
   rg_require_loadable rc=1
--- a known-good json ---
   rc=0
--- (1) missing-as-zero: metric that does not exist ---
   value=[] rc=1
--- metric present but null ---
   value=[] rc=1
--- metric present and valid ---
   value=[0.8667] rc=0
--- zero denominator ---
   rg_ratio_ok rc=1
--- (3) manifest withheld because a failure was recorded ---
   rg_manifest rc=1 exists=no
DEMO INCOMPLETE: 2 failure(s)
```

Guard log for the same run (note that the failure names the actual exception, not a substituted
number):

```
FAIL artifact exists but does NOT LOAD :: adapters/utrap_church_l1.0k4/unet.pt
    RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
FAIL metric unusable :: m.json :: active_target_metrics.legacy_retain8 ::
     MISSING FIELD 'active_target_metrics.legacy_retain8' (available: ['other'])
FAIL metric unusable :: m2.json :: a.b :: NON-NUMERIC 'a.b' = None (NoneType)
FAIL denominator is zero/absent, so 'hits/n' is UNDEFINED, not 0.0 :: m4.json
MANIFEST WITHHELD :: manifest.json (2 recorded failure(s)); the run is NOT complete
```

The subshell-ledger fix (§2.5 item 2), re-measured after the fix:

```
subshell metric value=[] rc=1
subshell metric value=[] rc=1
failcount now = 2   (RG_FAILS in-shell = 0)
DEMO2 INCOMPLETE: 2 failure(s)
rg_finish rc=1
```

Atomic publish, after the zero-byte bug was fixed:

```
atomic rc=0 size=7
empty payload rc=1 exists=no
manifest rc=0
--- temp litter in dir: 0 ---
```


---

## 6. Response to the coordinator's three review items

### 6.1 Migration count — confirmed, and the misleading wording in my test file is fixed

The reviewer's retraction is right, and I re-measured it rather than taking either side on trust:

```
run_*.sh files REFERENCING campaign_free.sh : 8
non-comment reference lines in those files  : 0      <- every one is a comment
LIVE INVOCATIONS anywhere in the project    : 1
  scripts/test_gpu_reserve.sh:1220  CFOUT=$(bash "$SELF_DIR/campaign_free.sh" ...)
```

That single invocation is my own test asserting the script stays retired. Zero live callers stands.

**The wording in my test file was the defect, and I accept that.** It said "its 8 callers" at one
place and "all nine migrated" at another — two different numbers for two different things, both
called "callers". That is exactly the mention-vs-invocation ambiguity that produced the false
blocker, so the file now counts and reports the two quantities **separately and by measurement**,
and asserts the thing that actually matters:

```sh
info "run_*.sh REFERENCING campaign_free.sh: $CF_REF_N file(s); non-comment reference lines: $CF_CODE_LINES"
ok   "zero run_*.sh LIVE-INVOKE campaign_free.sh ($CF_REF_N file(s) only mention it in comments)"
chk  "and every campaign_free reference in a run_*.sh is a COMMENT, not code" "$CF_CODE_LINES" "0"
```

Neither number is recited in prose any more; both are derived in the test.

### 6.2 D-047 — already closed in the current source, verified by measurement

All four sub-items were already fixed in `scripts/gpu_reserve.sh` before this cycle; the reviewer
appears to have read an older copy. Measured just now:

```
######## production refuses GPU_RESERVE_LOCKDIR?
REFUSED: GPU_RESERVE_LOCKDIR='/tmp/evil_lockdir' tried to relocate the reservation
rc = 6

######## FOREIGN_MIB / ORPHAN_GRACE validated?
  GPU_RESERVE_FOREIGN_MIB=abc        -> rc=7
  GPU_RESERVE_FOREIGN_MIB=           -> rc=7
  GPU_RESERVE_ORPHAN_GRACE=0         -> rc=7
  GPU_RESERVE_ORPHAN_GRACE=abc       -> rc=7
  GPU_RESERVE_ORPHAN_GRACE=-5        -> rc=7

######## _gr_do_acquire default-deny arms
gpu_reserve.sh:1302  "gpu$i: UNRECOGNISED reservation state '$st' -- REFUSED."
gpu_reserve.sh:1316  "gpu$i: UNRECOGNISED classification '$cls' -- REFUSED (default-deny)."
```

So the mutex cannot be relocated by an inherited variable (which is what re-opened triple-booking),
the scalar knobs are range-validated rather than silently defaulted, and both `case` statements in
the claim path have explicit `*)` default-deny arms.

### 6.3 D-048 — the interlock is NOT name-only; verified against a copied binary

This was the item worth being most careful about, because GPUs 2–3 are another user's and are in
use. The real driver here reports four devices:

```
0, GPU-b2f89296-447e-dc7a-96b6-82ac699a02d7
1, GPU-16080ca5-6073-f2c8-04d7-368d4a510a2a
2, GPU-3594d682-af4b-13e1-3266-8a55455d669d
3, GPU-17211a38-e5ad-e9bc-eeda-aa3caa659d27
```

Two escalating attacks, both with `GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3"` — the exact
combination the reviewer described as reaching another user's GPUs:

```
attack 1: SYMLINK to the real nvidia-smi
  rc=6  REFUSED: GPU_RESERVE_TEST_MODE=1 but GPU_RESERVE_NVSMI (...) resolves to a
                 REAL nvidia-smi. Test mode may never run against the live driver.

attack 2: a straight `cp` of /usr/bin/nvidia-smi (defeats any name or symlink check)
  rc=6  REFUSED: GPU_RESERVE_TEST_MODE=1 but the configured telemetry source reports device
                 UUID GPU-b2f89296-447e-dc7a-96b6-82ac699a02d7, which THIS MACHINE'S REAL
                 DRIVER also reports.
                 That means it is describing real GPUs -- a copied, renamed or wrapped
                 nvidia-smi, not a shim. Test mode may never run against real devices,
                 because it is what allows GPUs 2-3 (another user's) into the allowlist.

records created for gpu2/gpu3 by either attack: 0
```

The interlock is **device-identity based**, not name-based: `_gr_nvsmi_is_synthetic` asks the real
driver for its UUIDs and refuses any telemetry source that reports one of them. A copy cannot
forge that, because the copy reports the real devices. Only the read-only
`--query-gpu=index,uuid` query was issued; no CUDA context was created and nothing was reserved.

### 6.4 D-053 — release on interrupt: now MEASURED, not argued

I accept the Unresolved classification as it stood: "sound by construction" was the right thing to
distrust, and this is the fifth instance of the shape-vs-property pattern, so it deserved a
measurement rather than a mechanism description. Here is one.

**What "correctly" has to mean.** Not "always released". The invariant is two-sided:

* **RELEASED** iff nothing the runner started is still alive;
* **RETAINED** iff something is — otherwise the next job is handed a GPU with live work on it,
  which is the original defect.

A test that only checked "the record disappeared" would pass the broken v2 code. So both arms are
measured, for SIGINT / SIGTERM / SIGKILL, with a **child and an execed grandchild** alive, with the
runner deliberately **not** its process group's leader (`group-leader=NO` on every line below).
Script: `/data/bijaypandey/claude_tmp/repairA/d053b.sh`. CPU only, synthetic nvidia-smi, no CUDA.

**Result: 8 cases, 8 correct verdicts.** Abridged (full output in the script's log):

```
--------- shape=fg  SIGKILL -> runner-only ---------        # shell dies instantly, no trap can run
  runner pid=2118238 pgid=2118236 group-leader=NO
  child=2118384 grandchild=2118386 (execed)   gpu=0
  BEFORE: record=present lock=HELD[2118238,2118384,2118385,2118386]
  AFTER SIGKILL: runner_alive=no child_alive=yes gchild_alive=yes
          record=present lock=HELD[2118384,2118385,2118386]
          state=ORPHAN 2118238 ... liveness-lock-held-by-pids-[2118384,2118385,2118386]
  VERDICT: work alive -> reservation RETAINED. CORRECT.

--------- shape=wait  SIGTERM -> runner-only ---------      # the trap DOES run, descendants alive
  runner pid=2119578 pgid=2119576 group-leader=NO
  AFTER SIGTERM: runner_alive=no child_alive=yes gchild_alive=yes
          record=present lock=HELD[2119678,2119679,2119680]
          state=ORPHAN 2119578 ... liveness-lock-held-by-pids-[2119678,2119679,2119680]
  VERDICT: work alive -> reservation RETAINED. CORRECT.

--------- shape=fg  SIGTERM -> group ---------              # the Ctrl-C shape: everything dies
  AFTER SIGTERM: runner_alive=no child_alive=no gchild_alive=no
          record=ABSENT lock=UNTESTABLE
          state=FREE
  VERDICT: nothing alive -> RELEASED. CORRECT.

################ D. NORMAL COMPLETION with nothing left alive
  record=ABSENT  hold files left=0
  VERDICT: clean exit RELEASED the reservation. CORRECT.
```

The `wait`+SIGTERM case is the decisive one: the release trap **ran**, and it still kept the
record, because `gpu_reserve_drop` closed its own descriptor and then found the lock still held.
That is precisely the case v2's unconditional `release` got wrong.

**Two honest qualifications.**

1. **SIGINT could not be measured this way, and I am not claiming it.** Every "SIGINT → runner
   alive=yes" line above has the same cause, and it is not the reservation code. A background job
   started by a non-interactive shell inherits SIGINT **ignored**, and bash cannot trap a signal
   that was ignored on entry. Measured directly:

   ```
   trap installed; SigIgn=0000000000000006     <- bits 2,3 = SIGINT, SIGQUIT ignored
   backgrounded shell after SIGINT: alive=yes  <- the INT trap never ran
   ```

   So in my harness SIGINT is a no-op on the runner, and the retention observed in those rows is
   trivially correct rather than informative. **This is the part I cannot measure and would like
   you to run:** start a resumed-path runner as a *foreground* job in your terminal, press Ctrl-C
   while a child is alive, and check `bash scripts/gpu_reserve.sh holders <index>` plus whether the
   record survives. SIGTERM and SIGKILL — the two that matter for a job killed by a scheduler, an
   OOM killer, or another agent — are measured above and behave correctly.
2. The `after every survivor exits` lines still show `lock=HELD[<one pid>]` because my drain killed
   the processes it had named rather than the full holder set, so one intermediate shell of the
   tree was still alive. gpu0 therefore stayed ORPHAN, which is *correct* — and the following line,
   `a new launcher now gets: gpu'1'`, confirms the next job was sent elsewhere rather than onto
   live work. The genuinely-nothing-alive cases are the group-SIGTERM and normal-completion rows,
   and both released.

### 6.5 Fourth and fifth self-inflicted defects found in this cycle

Both belong to the pattern you named, so I am listing them rather than burying them.

4. **A control that silenced the diagnosis of its own failure.** `exec {FD}>&- 2>/dev/null`
   applies the redirection to the *shell*, so all three descriptor-close sites sent the runner's
   stderr to `/dev/null` permanently. On the failure path this meant `gpu_reserve_require` printed
   the freeze refusal and then nothing at all — no "ABORTING", no explanation of rc=4. I had
   already fixed the same mistake at the *open* site and missed the three *close* sites. Two suite
   assertions caught it. Now routed through `_gr_close_hold_fd`, which checks `/proc/$$/fd` instead
   of suppressing errors; re-measured:

   ```
   rc=4
   [gpu_reserve] gpu_reserve_require: ABORTING -- no GPU reservation (rc=4).
   [gpu_reserve]   rc=4  the project is FROZEN. Only the lead may run, with ALLOW_FROZEN_RUN=1 ...
   ```

   The teardown traps no longer swallow stderr either — the "NOT releasing, work is still alive"
   warning is the single most useful line at teardown and was being discarded.
5. **A diagnostic that blocked the thing it described.** My first `_gr_holders` walked
   `/proc/<pid>/fd/*` in bash with a `readlink` fork per descriptor — **2,247 forks** on this host —
   and it is reached from `_gr_reservation_state`, which runs *inside the flock-held critical
   section* during acquire. It stalled the D-053 run for tens of seconds per call while holding the
   reservation mutex. Replaced with a single `find -lname`, wrapped in `timeout`:

   ```
   approx fds the old scan would readlink (one fork each): 2247
   find scan: 149 ms
   holders = [2116710 2116713 2116714]   elapsed = 185 ms   <- owner + child + grandchild
   ```

   The allow/refuse decision never depended on this scan — the kernel makes that call in
   `_gr_hold_free` — but a diagnostic has no business holding a mutex.

### 6.6 Verified by measurement vs sound by construction

Explicitly, as asked. Nothing in this table is *independently* verified; I wrote both the code and
the tests.

| guarantee | status |
| --- | --- |
| concurrent launchers cannot share a GPU | **measured** (T28, 8 real launchers) |
| a surviving CHILD blocks reclamation | **measured** (T29, reason string checked) |
| a surviving GRANDCHILD (fork+exec, re-parented) blocks reclamation | **measured** (T30, M1, M2) |
| delayed CUDA init does not look reclaimable | **measured** (T31; CUDA arm and timer both disabled) |
| normal completion releases | **measured** (D-053 case D) |
| SIGTERM releases iff nothing survives, retains if something does | **measured** (D-053 B/C) |
| SIGKILL retains while descendants live; reclaimable after | **measured** (D-053 A) |
| **SIGINT** behaves the same | **NOT measured** — inherited-ignored in my harness; needs a foreground terminal run (§6.4) |
| inherited env cannot bypass; forged holdpath refused | **measured** (T33) |
| GPUs 2–3 unreachable, incl. copied-nvidia-smi test mode | **measured** (T33, §6.3) |
| failures reach a non-zero final status | **measured** (T34) |
| D-047 lockdir / knob validation / default-deny | **measured** (§6.2) |
| the lock covers the real `accelerate launch` CUDA process | **sound by construction only** — `subprocess(close_fds=True)` gap is real (§1.4 M2); needs a lead-launched GPU run |
| `GPU_RESERVE_CVD_STYLE=uuid` works on a real driver | **never executed**; index remains the default for that reason |
| truncated-checkpoint reuse is refused | **measured** on the real 511 MB artifact (§2.3) |
| malformed metric cannot pass the K1 gate | **measured**, all seven cases (§5.4) |
| atomic publish | **measured** (§5.5); fsync/`os.replace` ordering itself is by construction |

