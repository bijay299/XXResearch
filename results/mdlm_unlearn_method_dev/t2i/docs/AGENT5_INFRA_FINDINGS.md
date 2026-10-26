# AGENT 5 — EXECUTION INFRASTRUCTURE FINDINGS

Audit window: 2026-10-26 12:45 → 13:05 EDT. Project frozen (`FREEZE_ACTIVE`, declared 12:43).
No GPU work was launched; no `run_*.sh` was executed; `venv_t2i` / `venv_imma` / `venv_uce` were
not mutated. All lock testing was CPU-only against a fake `nvidia-smi` shim.

Files owned and changed by this agent:

| File | Action |
|---|---|
| `scripts/gpu_reserve.sh` | **NEW** — atomic per-GPU campaign reservation |
| `scripts/test_gpu_reserve.sh` | **NEW** — CPU-only test harness, 50 assertions |
| `scripts/campaign_free.sh` | **REWRITTEN** — retired to a diagnostic that grants nothing |
| `docs/AGENT5_INFRA_FINDINGS.md` | **NEW** — this file |

Classification key: **(1) CONFIRMED DEFECT** · **(2) SUSPECTED** · **(3) VALID** ·
**(4) HISTORICAL impact** · **(5) REPAIR PERFORMED + verification**

---

## SECTION 1 — GPU reservation (the assigned defect)

### D-001 · (1) CONFIRMED DEFECT — GPU selection was check-then-select, not atomic

`scripts/campaign_free.sh` (old v2) read GPU state and returned an index while holding no lock.
Three runners launched inside a 6-second window each ran it independently, each was told `0`, and
all three took GPU 0 while GPU 1 sat idle.

Evidence, from the project's own logs (unmodified):

```
logs/r1c_eval.log:1   === R1c on GPU 0 2026-10-26T12:39:29-04:00
logs/r4d.log:1        === R4d on GPU 0 2026-10-26T12:39:32-04:00
logs/r2_meta.log:11   === R2  on GPU 0 2026-10-26T12:39:35-04:00
```

Live at 12:45: GPU 0 = 3305 MiB with our pid 1369932 resident; GPU 1 = **34 MiB, idle**. The
three colliding pids (1369932 / 1370775 / 1371955) totalled 15.3 GB on one card. Half of our
allocated capacity was unused for the duration.

Call sites of the defective selector, all with the identical race (8 runners):

```
run_r1_imma_official.sh:28        run_r1b_imma_train.sh:20     run_r1c_eval.sh:14
run_r2_meta_official.sh:25        run_r4b_queue.sh:14          run_r4c_queue.sh:11
run_r4d_upstream_defaults.sh:15   run_e2h_meta_v2_stronger.sh:21
```

The window is structural, not merely narrow: every caller is
`while :; do G=$(bash scripts/campaign_free.sh|head -1); [ -n "$G" ] && break; sleep 120; done`
followed on the **next line** by `export CUDA_VISIBLE_DEVICES=$G`, and the GPU is not actually
touched until a model finishes loading some 30–60 s later. Two runners can therefore read the
same state many seconds apart and still collide.

### D-002 · (1) CONFIRMED DEFECT — the `/proc/<pid>/environ` clause was dead code

Old `campaign_free.sh:15-19` built its `CLAIMED` set by reading `CUDA_VISIBLE_DEVICES` out of
`/proc/<pid>/environ` for each live runner. `/proc/<pid>/environ` is a snapshot taken at
`execve(2)` and is **never updated**. Every one of the 8 callers sets the variable *after*
startup (`export CUDA_VISIBLE_DEVICES=$G`), so the variable was never in the snapshot.

Proven empirically during this audit with a CPU-only probe that exports the variable after start:

```
probe pid=1382639
--- CUDA_VISIBLE_DEVICES visible in /proc/$P/environ? ---
0 occurrences -> INVISIBLE
--- what campaign_free's extractor returns ---
extracted=[<EMPTY>]
```

So clause (a) of the selector could not match a single one of its own callers. The entire
"per-GPU campaign ownership" mechanism the file's header advertised did not exist; selection fell
through to the memory threshold in clause (c) — the very gate v1 had been written to remove.
Locked in as regression test **T13**.

A second, independent bug in the same clause: the extracted value was used as an array key
(`CLAIMED[$d]=1`). A runner with `CUDA_VISIBLE_DEVICES=0,1` would key `CLAIMED["0,1"]`, which
never matches the `0` / `1` probed in the loop below. Moot given D-002, but it would have
resurfaced the moment the environ read was "fixed" without fixing the key.

### D-003 · (1) CONFIRMED DEFECT — a memory threshold cannot represent occupancy

Old `campaign_free.sh:29-31` admitted a GPU when `memory.used < 2000` MiB. A campaign tears down
one cell's CUDA context before building the next, dipping to ~0–1.4 GiB — indistinguishable from
idle. The file's own header documents this ("a campaign DIPS to ~1.4 GiB between attack cells, so
a memory gate can fire mid-campaign") and then ships the gate anyway, because D-002 had silently
disabled the clause meant to replace it. Occupancy is not inferable from a memory reading; it has
to be *declared*. Locked in as regression test **T11**.

### D-004 · (1) CONFIRMED DEFECT — `pick_gpus.sh` will select another user's GPU

Not in my assigned scope but found while tracing selection, and it contradicts the standing
constraint that only GPUs 0–1 may be used:

```
pick_gpus.sh:2   # Pick N free GPUs by index, newest rule: ANY gpu is allowed, at most 2 of mine
pick_gpus.sh:6   busy=$(nvidia-smi --query-compute-apps=gpu_uuid ...)
pick_gpus.sh:11  [ "${mem:-0}" -ge 500 ] && continue
```

It iterates **all four** GPUs and admits any with no compute app and <500 MiB. GPUs 2 and 3 are
another user's (uid **1010**, pids 1363044 / 1363045, 4853 MiB each, confirmed via
`stat -c %u /proc/<pid>`; our uid is 1011). Between that user's cells, `pick_gpus.sh` would hand
out GPU 2 or 3 and we would land on top of them. **Owner action required** — I do not own this
file. Recommend deleting it in favour of `gpu_reserve.sh`, whose allowed set is `0 1` with
GPUs 2–3 structurally unreachable (regression test **T9**).

### D-013 · (1) CONFIRMED DEFECT — `FREEZE_ACTIVE` was read by nothing

Raised by the lead against their own freeze; repaired here because reservation is the one choke
point every GPU job must pass. `FREEZE_ACTIVE` (301 bytes, 12:43) existed, but no `*.sh` or
`*.py` outside `upstream/` contained the string — the freeze was held only by the lead having
SIGTERM'd the runner loops by PID. Any runner started by hand, by a stale cron, or by a resumed
waiter loop would have taken a GPU and proceeded, while the marker's existence was being cited as
assurance that it could not. Repaired in D-R3 below; witness test **T17**.

---

## SECTION 2 — (5) REPAIR PERFORMED + VERIFICATION

### D-R1 · `scripts/gpu_reserve.sh` — explicit reservation replacing inference

Design: a reservation is a **record on disk naming the owner process**, created inside a
`flock(2)`-protected critical section. It is immune to D-001 by construction, never consults
`/proc/environ` (D-002), and survives inter-cell memory dips because it is not a memory reading
(D-003).

| Requirement | Implementation |
|---|---|
| `flock` on a per-GPU lockfile | scan-and-claim transaction runs under `flock -w` on `$LOCKDIR/.reserve.mutex` (FD 9); reservation records are `$LOCKDIR/gpu<i>.owner` |
| record owner PID + campaign | `pid`, `start`, `campaign`, `gpu`, `uid`, `host`, `acquired`, published by atomic `mv` |
| reclaim stale locks whose owner is dead | owner validated as `/proc/<pid>` present **and** `/proc/<pid>/stat` field 22 (`starttime`) unchanged; otherwise reclaimed with an explicit log line |
| release on normal exit AND on crash | `release` on the happy path; `EXIT INT TERM HUP` trap installed by the sourceable `gpu_reserve_hold`; a missed trap still degrades to stale-reclamation |
| never select another user's GPU | any compute pid on an allowed GPU whose uid ≠ ours, **or which is invisible in `/proc`** (hidepid), marks the GPU `FOREIGN` and refuses it; unattributable memory ≥ `FOREIGN_MIB` refuses as `UNACCOUNTED` |
| GPUs 2–3 off limits entirely | allowed set is `0 1`; 2–3 are never enumerated, so they cannot be returned even when idle |
| account for pre-lock-scheme jobs of ours | an allowed GPU with a CUDA context owned by our uid but no reservation is refused as `OURS_UNLOCKED`; `adopt <idx> <campaign> <pid>` retroactively reserves for such a job |

Two hardening choices worth noting, both of which the harness caught rather than assumed:

* **PID reuse is defeated by recording `starttime`.** Liveness alone is not enough: a recycled
  PID would resurrect a dead reservation. Test **T7**.
* **A dead owner PID is rejected loudly at acquire time.** `$BASHPID` is re-evaluated inside
  `$( )`, so `G=$(gpu_reserve.sh acquire c "$BASHPID")` records the transient
  command-substitution subshell, which exits immediately — every racer would then reclaim the
  record as stale and all would win the same GPU. This bit the first draft of the harness itself
  (T4 reported "12 winners [0]"); the lock was correct and the *test* was wrong. `acquire` now
  refuses a non-running owner pid with a message naming `$$` as the correct thing to pass.
  **Any migration of the 8 callers must use `$$`, not `$BASHPID`.**

### D-R2 · `scripts/campaign_free.sh` — retired, grants nothing

Rewritten as a diagnostic. It documents all three defects with the log evidence, prints the
current reservation/occupancy table, and **writes nothing to stdout**, exiting non-zero. Because
every caller tests `[ -n "$G" ]`, an empty stdout makes a waiter loop keep waiting rather than
double-book a GPU — the safe failure direction, and the correct behaviour under a freeze. Verified:

```
$ G=$(bash scripts/campaign_free.sh | head -1); echo "stdout=[$G]"
stdout=[]
```

Checked before changing it: `pgrep -af 'run_.*\.sh|campaign_free'` returned **no live runner or
waiter loop**, so no in-flight job was affected. The three atomic jobs completed during the audit
(GPU 0 fell to 23 MiB); none was killed or interfered with.

### D-R3 · Freeze gate (D-013) built into reservation

* Refusal is **loud**: exit **4** (distinct from 3 = "no GPU free", so a waiter loop can tell them
  apart and `gpu_reserve_hold` aborts instead of spinning), with the marker's own text echoed to
  stderr so the caller learns why.
* The check runs **inside the critical section**, under the mutex, alongside stale-lock
  reclamation — so a freeze written microseconds after a would-be acquirer started still refuses
  it. Checking before taking the lock would reintroduce exactly the check-then-act window this
  file exists to close.
* The override requires **both** `ALLOW_FROZEN_RUN=1` **and** the explicit `--allow-frozen`
  argument. Requiring the argument is what prevents an ambient `export ALLOW_FROZEN_RUN=1`
  inherited from a profile, a parent shell, or a resumed script from silently lifting the freeze:
  a profile can set an env var but cannot add an argument to someone else's command line.
* Every override appends to `logs/frozen_override.log`: timestamp, caller pid, campaign, GPU
  granted, marker path, uid.
* `adopt` / `release` / `reap` / `status` are exempt — they start no new work.

Verified against the **live** freeze (read-only; refused, as required):

```
$ bash scripts/gpu_reserve.sh acquire audit_probe $$
[gpu_reserve] REFUSED: the project is FROZEN. No reservation will be granted.
[gpu_reserve] marker: /data/bijaypandey/t2i_unlearn/FREEZE_ACTIVE
[gpu_reserve] ---- marker contents ----
[gpu_reserve] | REPAIR-FIRST FREEZE, declared 2026-10-26T12:43-04:00.
[gpu_reserve] | No new campaign cells may start. Bounded VERIFICATION runs only, and only when explicitly
[gpu_reserve] | launched by the implementation lead. Agents may not launch GPU work.
[gpu_reserve] | Lifted only by an evidence-backed review gate (AUDIT_REPAIR_REGISTER_v1.md section 10).
[gpu_reserve] -------------------------
rc=4
```

### D-R4 · `scripts/test_gpu_reserve.sh` — CPU-only harness, **50 passed / 0 failed**

`nvidia-smi` is replaced by a shim reading a plain-text state file, so every topology (including
"another user is on GPU 1" and "hidepid hides their pid") is simulated. No CUDA is imported; safe
to run while GPU jobs are in flight. Full output:

```
CPU-ONLY test harness for /data/bijaypandey/t2i_unlearn/scripts/gpu_reserve.sh
fake nvidia-smi = .../gpu_reserve_test.zrVVUn/nvidia-smi  (no GPU is touched, no CUDA is loaded)

== T1/T2/T3  sequential acquire, distinct indices, then exhaustion
  PASS first acquire (= 0)
  PASS second acquire takes the OTHER gpu (= 1)
  PASS third acquire refused (exit 3), no double-booking

== T5  release makes a GPU available again
  PASS reacquire after release (= 0)

== T4  CONCURRENCY: 12 processes race for 2 GPUs
  12 racers -> 2 winner(s) [0 1], 10 refused
  PASS exactly 2 winners for 2 GPUs (= 2)
  PASS winners hold DISTINCT indices (= 0 1)
  PASS all other racers refused (= 10)
  PASS no index was handed out twice (this is the campaign_free.sh defect, fixed)

== T6  stale-lock reclamation after a simulated runner crash
  PASS crashy reserved (= 0)
  PASS while owner lives, gpu0 is respected (= 1)
  PASS gpu0 handed to the next campaign after the crash (= 0)
  PASS reclamation was explicit: gpu0 RECLAIMED stale reservation: owner pid 1547052 (crashy) is dead

== T7  PID reuse does not resurrect a dead reservation
  PASS alive pid + mismatched start time => correctly judged stale (PID reuse defeated)

== T8  a GPU occupied by ANOTHER USER is never selected
  PASS gpu0 (another user's) skipped, gpu1 chosen (= 1)
  PASS refusal reason recorded for the foreign GPU
  PASS both GPUs foreign => refuses rather than displacing a user
  PASS pid invisible in /proc treated as foreign, gpu1 chosen (= 1)

== T9  off-limits GPUs 2-3 are never selected, even when idle
  PASS GPUs 2-3 idle but out of scope => nothing selected (exit 3)

== T10 a pre-lock-scheme CUDA context of OURS blocks that GPU
  PASS legacy job of ours on gpu0 respected, gpu1 chosen (= 1)
  PASS legacy job detected via compute-apps + uid, not via a memory threshold
  PASS adopt gpu0 for the legacy pid (= 0)

== T11 inter-cell memory DIP does not release a held reservation
  PASS campaign holds gpu0 (= 0)
  PASS gpu0 still reserved despite 0 MiB used; intruder sent to gpu1 (= 1)
  PASS memory dip no longer leaks the GPU (campaign_free.sh defect (c), fixed)

== T12 trap-based release when the holding shell dies
  PASS sourced helper reserved a gpu (= 0)
  PASS reservation record present while runner lives
  PASS EXIT/TERM trap released the reservation on shell death

== T13 regression witness: campaign_free.sh's /proc/environ clause is dead code
  PASS a runner that exports CUDA_VISIBLE_DEVICES after startup is INVISIBLE in /proc/<pid>/environ
  PASS => campaign_free.sh clause (a) can never match any of its 8 callers

== T14 D-013 freeze gate: a FREEZE_ACTIVE marker REFUSES every reservation
  PASS refused while frozen, non-zero exit (rc=4), no index returned
  PASS refusal exit code is 4 (distinct from 3 = no GPU free) (= 4)
  PASS refusal is loud on stderr
  PASS marker contents echoed to stderr, so the caller learns WHY
  PASS no reservation record was written while frozen

== T15 the override needs BOTH the env var and the explicit argument
  PASS ambient ALLOW_FROZEN_RUN=1 alone does NOT lift the freeze (rc=4)
  PASS explains that an inherited env var is not enough
  PASS --allow-frozen alone does NOT lift the freeze (rc=4)
  PASS says both are required

== T16 the lead's bounded verification run: override works AND is logged
  PASS override grants a GPU (= 0)
  PASS override exits 0 (= 0)
  PASS override is announced on stderr
  PASS override log line written: 2026-10-26T13:02:04-04:00	override=ALLOW_FROZEN_RUN+--allow-frozen	pid=1573594	campaign=lead_verify	gpu=0	marker=.../FREEZE_ACTIVE	uid=1011
  PASS log records pid=1573594
  PASS log records campaign=lead_verify
  PASS log records gpu=0
  PASS log records override=
  PASS log line is timestamped
  PASS a non-override caller is still refused after an override was granted (= 4)
  PASS marker removed => normal acquire works again (= 0)

== T17 D-013 witness: nothing in the project reads FREEZE_ACTIVE except gpu_reserve.sh
  files referencing FREEZE_ACTIVE: scripts/gpu_reserve.sh scripts/test_gpu_reserve.sh
  PASS gpu_reserve.sh is now a reader of the freeze marker

----------------------------------------
RESULT: 50 passed, 0 failed
No GPU was accessed; nvidia-smi was the shim at .../gpu_reserve_test.zrVVUn/nvidia-smi
```

### Residual limitation (stated, not hidden)

A legacy campaign that holds **no** reservation and is **between cells** (no compute app, low
memory) is undetectable by any means available to this script — that is D-003 restated, and it is
why reservations exist. It is closed only by migrating all 8 callers to `gpu_reserve.sh`; until
then, use `gpu_reserve.sh adopt <idx> <campaign> <pid>` to give a running legacy job a
reservation. **Owner action required** (Agent 2 / the lead own `run_*.sh`).

---

## SECTION 3 — Copied virtual environments

### D-005 · (1) CONFIRMED DEFECT — the shebang repair missed all three `activate` variants

`venv_imma` and `venv_uce` were made by `cp -a venv_t2i`. The previous repair fixed **22 of 25**
`venv_t2i`-referencing files per env. The three shell `activate` variants were never touched
(mtime `Oct 22 17:52` in both envs — the original copy time):

```
venv_imma/bin/activate:41        VIRTUAL_ENV=/data/bijaypandey/t2i_unlearn/venv_t2i
venv_imma/bin/activate:45        PATH="$VIRTUAL_ENV/"bin":$PATH"
venv_imma/bin/activate:60        VIRTUAL_ENV_PROMPT='(venv_t2i) '
venv_imma/bin/activate.csh:11    setenv VIRTUAL_ENV /data/bijaypandey/t2i_unlearn/venv_t2i
venv_imma/bin/activate.fish:36   set -gx VIRTUAL_ENV /data/bijaypandey/t2i_unlearn/venv_t2i
venv_uce/bin/activate:41         (identical)
venv_uce/bin/activate.csh:11     (identical)
venv_uce/bin/activate.fish:36    (identical)
```

Consequence: `source venv_imma/bin/activate` prepends **`venv_t2i/bin`** to `PATH`, so `python`
and `pip` in that shell are `venv_t2i`'s. A `pip install` from such a shell writes into
`venv_t2i`, breaking the manifest's standing promise that `venv_t2i` is never modified.

**(4) HISTORICAL impact: none measurable.** No caller sources `activate` — every call site invokes
the interpreter directly (`run_r1_imma_official.sh:31`, `run_r1b_imma_train.sh:22`,
`run_r4_queue.sh:29,35`, `run_r1c_eval.sh:18`, `scripts/r4c_uce_checkpoints_and_images.py:79`,
and the r4b/r4c/r4d runners). The exposure is interactive/human, not a past result. **Owner action
required** — I was instructed not to mutate the venvs. One-line fix per file.
`Activate.ps1` is clean in both (computes `$VenvDir` dynamically).

### D-006 · (3) VALID — direct-interpreter isolation is real

Everything that the *actual* execution path depends on is clean. Measured (CPU-only, no torch):

* `venv_imma/bin/python` → `prefix=/data/bijaypandey/t2i_unlearn/venv_imma`, `base_prefix=/usr`
* `venv_uce/bin/python` → `prefix=/data/bijaypandey/t2i_unlearn/venv_uce`
* `site.getsitepackages()[0]` is each env's **own** `lib/python3.10/site-packages`; distinct
  inodes (388630064 / 388899584 / 388774377)
* `bin/` shebangs: **0 of 30** (imma) and **0 of 22** (uce) still point at `venv_t2i`; script
  bodies byte-identical apart from line 1
* `pyvenv.cfg`: all three byte-identical (`home = /usr/bin`, 3.10.12). The absent
  `prefix`/`executable`/`command` keys are **correct for Python 3.10**, which writes only
  home/include-system-site-packages/version/prompt
* `.pth` files: 1 per env (`distutils-precedence.pth`), no absolute paths
* `dist-info`: **0** `direct_url.json`; **0** `RECORD` lines starting `/` or mentioning `venv_`;
  `INSTALLER` = `pip` in 92/92 (imma) and 47/47 (uce); no `~`-prefixed uninstall residue
* pip: `bin/pip`, `pip3`, `pip3.10` all shebang their own env; **no `pip.conf`, `pip.ini`, or
  `distutils.cfg` anywhere** in either env
* `.pyc` / `.so` / symlink targets: **0** `venv_t2i` hits across 8225 / 11499 / 8225 pyc files;
  `find -lname '*venv_t2i*'` → none; no broken symlinks
* interpreter symlinks identical in all three envs: `python → python3 → /usr/bin/python3`
  (absolute), resolving to `/usr/bin/python3.10` — no env resolves through `venv_t2i`

`grep -rl 'venv_t2i'` returns **exactly 3 files per env** — the three `activate` scripts of D-005
— and nothing else.

### D-007 · (1) CONFIRMED DEFECT — documented env state contradicts disk

`venv_imma` is really at **diffusers 0.24.0 / huggingface_hub 0.25.2**, but two project files
still assert 0.31.0:

```
docs/UPSTREAM_FIDELITY_MANIFEST.md:232    claims diffusers 0.31.0
run_r1_imma_official.sh:18                claims diffusers 0.31.0
run_r1b_imma_train.sh:7                   CORRECT: "now diffusers 0.24.0 + hub 0.25.2"
```

`logs/upstream_envs.log` ends at 0.31.0 (10:40) while `venv_imma/bin` mtimes are 11:45–11:48, so
the downgrade happened after the log and the manifest was never updated. The downgrade itself is
**clean** (diffusers 335/335 files vs RECORD, hub 111/111, transformers 1846/1846, torchvision
187/187, datasets 143/143 — 0 orphans, 0 missing). Documentation defect, not an env defect.
**Owner action required** (manifest + `run_r1_imma_official.sh` are not mine).

### D-008 · (2) SUSPECTED — `venv_uce` is a byte-identical clone, so its "pin" is empty

`diff -rq` over the whole of `venv_uce/lib/python3.10/site-packages` vs `venv_t2i`'s returns
**0 differences**; 93 top-level entries vs 93, 47 dist-info vs 47. Its isolation is real (own
directory, own inode, resolves to itself), so it is not *fictitious* — but nothing distinguishes
it from `venv_t2i`. This matches the manifest's §5 table ("`venv_uce` | unmodified copy") and
contradicts the same document's §5 prose:

> "UCE's `requirements.txt` asks `diffusers>=0.34.0` against our 0.31.0, so `venv_uce` upgrades
> diffusers in isolation"

No such upgrade exists on disk (diffusers 0.31.0 in both). Either the prose is stale or the
intended upgrade never happened — **which it is determines whether every R4* result is an
upstream-fidelity measurement or a measurement of our own pinned stack.** Cannot be resolved
read-only; flagged for the lead. Also undocumented in `venv_imma`: `cwevals-0.0.1b1` and a
10-package OpenTelemetry stack (pulled in by wandb 0.31.0).

---

## SECTION 4 — Failure handling across `run_*.sh` (read-only; report only)

Swept **180 runners** = 138 `run_*.sh` + 42 `scripts/run_*.sh`.

### D-009 · (1) CONFIRMED DEFECT — no runner checks exit codes; 109 echo `DONE` unconditionally

| Metric | Count |
|---|---|
| Runners with `set -u` only | **180 / 180** |
| Runners with `set -e` / `set -euo pipefail` / `errexit` | **0** |
| `bin/python` invocation occurrences | 414 |
| …with any exit-status check | **8** |
| Runners that capture an rc and branch on it | **1** |
| `echo …(DONE\|COMPLETE\|FINISHED\|SUCCESS\|PASS)` lines | **279** |
| Runners whose last non-blank line is an unconditional `DONE` echo | **109 / 180** |

`set -u` catches unset variables and nothing else: a crashed `python` is indistinguishable from a
successful one. Representative `DONE`-after-failure pairs (echo / preceding unchecked command):

```
run_e2_meta_v2.sh:53            <- :30-33 meta_unlearn_v2.py, :35-36 gen_eval.py, :37 heredoc
run_e4_h90_utrap_v2.sh:55       <- :29-31 erase_utrap_v2.py; `return 1` at :32 discarded by a bare list
run_e11_budget_curve.sh:63      <- :35-37 attack_relearn.py, :40-62 aggregator heredoc
run_e2g_meta_v2_grid.sh:63      <- :34-35 attack loop, :40 aggregator
run_e12_method_axis.sh:70       <- :39-40 attack, :45 aggregator
run_r2_meta_official.sh:34      <- :29 r2_build_meta_pools.py; rc PRINTED at :30, never tested
run_r4_queue.sh:38              <- :35-36 r4_uce_official_vs_port.py
run_r4b_queue.sh:25             <- :22-23 r4b_uce_upstream_vs_port.py
run_r4c_queue.sh:18 / run_r4d_upstream_defaults.sh:22  <- :15-16 / :19-20 r4c_uce_...py
run_lowbudget_audit.sh:16       <- :13-14 attack_relearn.py
run_crossmodel.sh:51,52         <- run_one body; `return 1` at :31 swallowed by ( … ; echo … )
run_gift.sh:33,34,58  /  run_utrap.sh:33,34,54
```

The only correct pattern in the entire project:

```
run_h64_frag.sh:24   rc=$?; echo "=== BUILD rc=$rc $(date -Is)" >>$LOG
run_h64_frag.sh:25   [ $rc -ne 0 ] && { echo "BUILD FAILED" >>$LOG; exit 1; }
```

The 8 checked invocations: `run_h66_frp.sh:31,43`; `run_f57_gap.sh:21`;
`scripts/run_h47_composite.sh:22`, `run_h51_forget_fisher.sh:23`, `run_h52_settle.sh:24`,
`run_h52_para.sh:24`, `run_undo_sweep.sh:18`.

**`run_smoke_v2.sh` is the sharpest case** — it is the audit's own integration-validation script.
`:33`, `:43`, `:51` do capture `$?` correctly, but **only print it**; `:54` prints
`=== SMOKE DONE` regardless. A run in which all three v2 implementations crash emits three
`exit=1 artifact=NO` lines followed by `SMOKE DONE`. Compounding: `:23` `: >"$L"` truncates
`logs/smoke_v2.log` on every invocation, destroying the previous smoke's evidence.

`run_e6_anchors.sh:17` — `echo "E6 ANCHORS DONE rc=$? ..."` reports the **second** python's status
only; the first (`:14 gen_eval_final.py --only BASE_sd15_anchor`) is silently dropped, yet the
line reads as a verdict on both anchors.

### D-010 · (1) CONFIRMED DEFECT — 17 kill-rule gates can report a crash as a measured result

This is the most serious finding in the failure-handling audit, because it converts an
infrastructure failure into a scientific claim.

The gates are `awk -v r="$R" 'BEGIN{exit !(r>=0.60)}'`. Verified awk semantics: `r=""` → below
K1; `r="0"` → below K1; `r="FAIL"` → **clears** K1 (string comparison, `F` > `0`).

**9 gates where `$R` is unguarded**, so any `json.load` failure yields `r=""` → "below K1":

```
run_utrap.sh:39-42      -> "$A retain8=$R below K1 -- destroys the model, not attacked"
run_gift.sh:40-42,47    -> "retain8=$R MEASURED below K1 0.60 -- destroys the model, not attacked"
run_gift_fine.sh:38-45  -> "retain8=$R MEASURED below K1 -- not attacked"
run_wsr.sh:45,48    run_wsr2.sh:33,36    run_uce_sd15_bracket.sh:30
run_gift_atk.sh:16,19   run_t2i_next.sh:73,76
run_r1c_eval.sh:31-32   -> R8=$(...) with no 2>/dev/null and no ||, then `exit 0`
```

**8 further gates where the substitution is explicitly laundered to `0`** — `2>/dev/null || echo 0`
appears **33 times across 17 files**:

```
run_crossmodel.sh:35-37,42      run_gift_crossmodel.sh:34-36,41    run_utrap_crossmodel.sh:37-39,44
run_uce_crossmodel.sh:41        run_meta_crossmodel.sh:49          run_meta_bksdm_low.sh:38
run_gift_cliff.sh:31            run_gift_b13.sh:34                 run_uce_sd15_lam.sh:27-34
run_utrap.sh:25-27              run_utrap2.sh:37-39   (logged as "as-is: target=$T retain8=$R")
```

The `2>/dev/null` discards the traceback *and* the `|| echo 0` manufactures a plausible number.
The log then reads `retain8=0.0000 ... below K1 -- not attacked`, which is **exactly what a real
model-destroying erasure looks like**. The defence silently escapes being attacked and the log
attributes the escape to the kill rule. `run_gift_gpu0.sh:24-29` uses `|| echo FAIL` instead —
which, had it been gated, would have *cleared* K1.

**(4) HISTORICAL impact:** every "MEASURED below K1 / destroys the model / not attacked" line in
the findings corpus is unverifiable from the log alone and must be re-derived from the artifact
JSON before it can be cited. Any arm excluded from an attack grid on this basis needs re-checking.

### D-011 · (1) CONFIRMED DEFECT — `run_r4_queue.sh` names artifacts for configs they don't have

```
run_r4_queue.sh:29-30   venv_uce/bin/python scripts/r4_uce_official_vs_port.py --concept church --lamb "$1" ... >>"$L" 2>&1
run_r4_queue.sh:31      cp outputs/r4_uce_official_vs_port.json "outputs/r4_uce_lamb$1.json" 2>/dev/null
run_r4_queue.sh:37      cp outputs/r4_uce_official_vs_port.json outputs/r4_uce_erasescale2.json 2>/dev/null
```

`scripts/r4_uce_official_vs_port.py:133` writes a **fixed** path with no config in the name. If
the `lamb=0.1` run fails, `:31` copies the still-present `lamb=0.5` artifact to
`outputs/r4_uce_lamb0.1.json` — and `2>/dev/null` hides even a missing-source error. The file's
*name* then asserts a configuration its *contents* do not have. Same for `erasescale2`.

### D-012 · (1) CONFIRMED DEFECT — `exit 0` on failure paths (4 dishonest sites)

```
run_e1b_wdr_scaled.sh:23   [ -f "adapters/$A/unet.pt" ] || { echo "$A BUILD FAILED ..." >>"$L"; exit 0; }
run_e1_wdr_faithful.sh:33  echo "$A BUILD FAILED ..." >>"$L"; exit 0
run_uce_sd15_lam.sh:24     [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED" >>...; exit 0; }
run_r1c_eval.sh:32         awk ... || { echo "K1 fail, not attacked" >>"$L"; exit 0; }
```

A build failure exits **0**, so any wrapper, queue, or `&&` chain sees success.
`run_r1c_eval.sh:32` is worst: it exits 0 on a gate decision that may itself be a laundered read
failure (`:31` is unguarded — see D-010).

Related: **`return 1` swallowed by subshell lists.** `run_crossmodel.sh:31`,
`run_gift_crossmodel.sh:30`, `run_utrap_crossmodel.sh:33`, `run_utrap.sh:21`, `run_utrap2.sh:33`,
`run_gift_fine.sh:25`, `run_e4_h90_utrap_v2.sh:32`, `run_imma_alpha.sh:25`, `run_wsr2.sh:17` all
`return 1`, but every caller is `( run_one … ; echo "X DONE" ) &` or a bare statement list, so the
subshell's status is `echo`'s. The `return 1` is decorative.

---

## SECTION 5 — Resume / cache behaviour

### D-014 · (1) CONFIRMED DEFECT — zero atomic writes anywhere; every `[ -f ]` guard rests on this

Across **48** files in `scripts/*.py`:

| Pattern | Count |
|---|---|
| `os.replace` / `os.rename` / `.tmp` / `NamedTemporaryFile` | **0** |
| `json.dump(rec, open(path,"w"))` straight to the final path | **26** |
| `torch.save(..., f"{out}/unet.pt")` straight to the final path | **23** |
| `try/except` around an artifact write | **0** |
| `shutil.rmtree` | **0** |

And the ordering is checkpoint-first:

```
scripts/erase_utrap_v2.py:286-288   os.makedirs(a.out, exist_ok=True)
                                    torch.save({k: v.half() ...}, f"{a.out}/unet.pt")
                                    json.dump({"study": "utrap_v2", ...
```

Same shape in `meta_unlearn_v2.py:266-268`, `erase_wdr_v2.py:192-194`, `erase_frp.py:208/228`,
`erase_gift.py:251/260`, `erase_uce.py:124/125`, `erase_wsr.py:174/175`, `meta_unlearn.py:156/157`,
`erase_imma.py:196/208`, `erase_esd.py:110/111`, `erase_adv.py:155/156`, `undo_t2i.py:375/376`,
`r1_eval_imma.py:29`.

Consequences, all reachable:

1. A kill during `torch.save` leaves a **truncated 1.7 GB `unet.pt`** that satisfies every
   `[ -f adapters/$A/unet.pt ]` guard — and that inverse test is the **only** failure detection
   those 171 builds have.
2. A kill or ENOSPC *between* `torch.save` and `json.dump` leaves `unet.pt` present and the
   provenance JSON absent — the build "exists" forever while its record does not.
3. `open(path,"w")` truncates on open, so a kill during serialization leaves a **truncated JSON at
   the canonical name**, which every `[ -f outputs/atk_*.json ] && continue` guard resumes past
   forever.
4. **Nothing deletes a partial artifact on failure.** The only `rm`s are unconditional cleanups of
   the *transient* attack dir (`rm -rf "adapters/_atk_$K"`, 23 sites) and the SMOKE adapters.

**(4) HISTORICAL impact — this has already fired.** `run_utrap2.sh:13-14` records *"/data hit 100%
earlier today and killed two builds mid-run"*; `:5-6` records *"the K=4 arm died mid-build"*. And
§6 below finds the surviving corpse: a 29.7%-truncated `unet.pt`.

**Highest-leverage fix in the whole audit:** `tmp + os.replace` at 26 `json.dump` and 23
`torch.save` sites converts ~285 existence guards from "bytes exist" to "the write completed".
**Owner action required** (`scripts/*.py` are not mine).

### D-015 · (1) CONFIRMED DEFECT — R4c/R4d share a mis-keyed port checkpoint

```python
scripts/r4c_uce_checkpoints_and_images.py:77   port_dir = f"{P}/adapters/R4C_port_{a.concept}_lam{a.lamb}"
scripts/r4c_uce_checkpoints_and_images.py:78   if not os.path.exists(f"{port_dir}/unet.pt"):
```

The key is `{concept, lamb}` and omits `erase_scale`, `preserve_scale`, and **`no_preserve`** —
which is exactly what `run_r4d_upstream_defaults.sh:19-20` varies, at the **same** `lam` values
(0.5, 0.1) `run_r4c_queue.sh:14-16` already ran. So **R4d's port side is byte-identical to
R4c's.** Worse, the inner invocation at `:79-82` never passes a no-preserve flag at all, while the
record at `:160-161` writes `"n_preserve": len(preserve)` = **0** for R4d. So
`outputs/r4d_uce_lamb0.5.json` asserts `n_preserve: 0` for a port checkpoint built **with 10
preserve concepts** — the precise asymmetry R4d exists (`:2-6`) to remove. The JSON *output* path
at `:169` **is** correctly keyed (`r4{'d' if a.no_preserve else 'c'}_...`), which is what makes
the confound invisible: two differently-named result files, one shared mis-keyed input.

**R4d's current artifacts are confounded and must be re-run after the key is fixed.** Note R4d
completed at 12:45 during this audit, so `outputs/r4d_uce_lamb{0.5,0.1}.json` are affected.

### D-016 · (1) CONFIRMED DEFECT — `run_h66_frp.sh` attack key omits the arm it attacks

```
run_h66_frp.sh:51   ARM=$(cat "outputs/h66_selected_${CONCEPT}.txt")
run_h66_frp.sh:55   L="H66_frp_${CONCEPT}_sd${SD}_lr${LR}"
run_h66_frp.sh:56   [ -f "outputs/atk_${L}.json" ] && { echo "[skip] $L" >>$LOG; continue; }
run_h66_frp.sh:57-59  attack_relearn.py --unet "adapters/$ARM/unet.pt" ...
```

`$ARM` is chosen dynamically at `:51` and encodes rho and lambda
(`frp_church_r3_l05`, `frp_church_r3_l00`, …). The skip key has concept, seed, lr — and **not the
arm**. Re-run after `select_h66_operating_point.py` picks a different operating point and every
cell is skipped, **silently crediting the new arm with the old arm's attack results**.

Also at `:51`: `cat` is unguarded, so a missing selection file yields `ARM=""`, the attacks at
`:57-59` run against `adapters//unet.pt`, `:60` echoes the rc, and `:62` prints `STAGE2 DONE`.

### D-017 · (1) CONFIRMED DEFECT — `.labels` is a lock with acquire and no release

```
run_utrap.sh:12       claim() { mkdir ".labels/$1" 2>/dev/null; }
run_utrap.sh:15       claim "$L" || return 0
run_crossmodel.sh:21  mkdir ".labels/$L" 2>/dev/null || return 0
```

**36 runners** use this (`run_gift_matrix_fill.sh:10`, `run_gift_db_deep.sh:10`,
`run_imma_ti_push.sh:13`, `run_imma_ti_push2.sh:11`, `run_imma_ratchet.sh:18,21,24`,
`run_imma_attack.sh:23,27`, `run_lmc.sh:12`, `run_ti_matched.sh:23`, `run_uce_sd15_lam.sh:20,37`,
`run_utrap2.sh:22`, `run_wsr2_seeds.sh:20`, …). `.labels/` holds **194 directories** and **no
runner ever removes one** (grep for `rm`/`rmdir` against `.labels`: zero hits). Re-running any of
those 36 scripts is a guaranteed silent no-op that `return 0`s without even logging a "skipped"
line.

**(4) HISTORICAL impact — confirmed to have cost a cell**, by the project's own note:

> `run_uce_lam1000_sd0.sh:4` — *"Seed 0 claimed a label during the earlier run but produced no
> output; seeds 1 and 6 are done."*

The fix at `:16` is `mkdir -p ".labels/$K"` — deliberately overriding the lock rather than fixing
it. Needs a release-on-exit trap or a per-run lock; `gpu_reserve.sh`'s stale-reclamation is the
pattern to copy.

### D-018 · (2) SUSPECTED — build/eval guards are stale across code versions

**171** `[ -f X ] || python …` + 3 `[ -f X ] && skip` build guards. Keys are complete for the
parameters each script varies, but omit parameters that are fixed in the script and absent from the
filename, so they are stale across code changes:

* `run_e4_h90_utrap_v2.sh:24` `A="utrap2_church_l${LAM}k${K}"` omits `--steps $OUTER`,
  `--n-images 16`, `--n-retain 32`, teacher. `run_e4m_matched.sh:17` uses a *different* corrected
  outer-step schedule and was renamed `utrap2m_…` to dodge the collision (documented at `:2-5`) —
  the key was worked around, not fixed, and `utrap2m` still omits `OUTER`.
* `run_e2_meta_v2.sh:28` `A="meta2_church_lt${LAM}"` omits `--zeta`, `--steps 1500`,
  `--n-images 64`, `--meta-scope`, `--attack-scope`, `--target-source`, `--require-target-acc`.
* `run_utrap.sh:14` and `run_utrap2.sh:21` share the **same** namespace
  `utrap_church_l${LAM}k${K}`; the (lam,K) pairs happen not to overlap — luck, not design.
* `run_gift_fine.sh:19` / `run_gift.sh:38` / `run_gift_b13.sh` / `run_gift_cliff.sh` /
  `run_gift_gpu0.sh:12` share `gift_church_b$BETA` with beta as the only key.

**Eval guards (20 sites)** — `[ -f "outputs/gen_$L.json" ] || gen_eval.py …`
(`run_crossmodel.sh:33`, `run_utrap.sh:23`, `run_utrap2.sh:35`, `run_gift.sh:28`,
`run_gift_fine.sh:27`, `run_gift_crossmodel.sh:32`, `run_utrap_crossmodel.sh:35`,
`run_uce_crossmodel.sh:32`, `run_meta_crossmodel.sh:40`, `run_uce_sd15_lam.sh:25`,
`run_uce_sd15_bracket.sh:27`, `run_wsr.sh:28`, `run_wsr2.sh:18`, `run_imma.sh:19`,
`run_imma_alpha.sh:26`, `run_h68_v2_control.sh:27`, `run_f57_gap.sh:23`, `run_h66_frp.sh:33,45`,
`run_t2i_next.sh:54`). The key is the arm label only, omitting `--target` and every eval knob
(`--n-per-prompt`, `--steps`, `--seed`) that 28 `scripts/run_h*.sh` do pass explicitly. Since
`outputs/gen_$L.json` **feeds the kill rule**, a `gen_*.json` from an older `gen_eval.py`
(different prompt set, different closed-set CLIP labels) silently governs whether a new build is
attacked.

### D-019 · (3) VALID — the attack-cache keys are mostly sound, and aggregation is self-correcting

**114** `[ -f "outputs/atk_${K}.json" ] && continue` guards across 77 runners.
`scripts/attack_relearn.py` has **no** `os.path.exists` early return, so the skip is entirely the
shell's. Fully-keyed and benign: `run_e2g_meta_v2_grid.sh:31-32`,
`run_e2h_meta_v2_stronger.sh:30-31`, `run_e4mg_matched_grid.sh:18-19`, `run_e12_method_axis.sh:36-37`,
`run_r1c_eval.sh:34-35`, and the best-keyed family in the project,
`run_db_push.sh:12-13` / `run_db_para_push.sh:12` / `run_db_pw0.sh:14`
(`…_s${ST}_lr${LR}_sd${SD}_pw${PW}` — all four varied params in the key).

Keys omitting `--steps` (**medium**, self-correcting in analysis): `run_utrap.sh:47`,
`run_gift.sh:50`, `run_gift_fine.sh:49`, `run_crossmodel.sh:44`, `run_gift_crossmodel.sh:43`,
`run_utrap_crossmodel.sh:46`, `run_uce_sd15_lam.sh:36`, `run_gift_matrix_fill.sh:10`. The
aggregators read `hparams` **out of the JSON** rather than parsing the filename
(`run_e2g_meta_v2_grid.sh:50-54`, `run_e2h_meta_v2_stronger.sh:44-46`,
`run_e11_budget_curve.sh:44-45`), so a mis-keyed file is *reported* under its true steps/lr/seed.
The damage is a skipped cell, not a wrong number — unless a default in `attack_relearn.py`
changes.

**Python-side resume guards: 16 `os.path.exists` sites, only one is a work-skipping cache** — and
that one is D-015. The other 15 are read-side presence checks that skip *reporting* a missing
input (`select_h66_operating_point.py:37`, `gen_eval_final.py:106`, `lmc_probe.py:41`,
`h64_analyse.py:32`, `h68_analyse.py:28,39`, `h66_analyse.py:76`, `registry_v1.py:79,119`,
`split_registry.py:95,136`, `provenance.py:115,190`, `retain_scope_slice.py:44`) plus two
assertions (`r4c_uce_checkpoints_and_images.py:84`, `r2_build_meta_pools.py:50`). No stale
artifact is consumed by any of them.

### (4) HISTORICAL precedent that this risk class fires

```
run_h66_frp.sh:23-25  "bc is NOT installed on this host: the original substitution returned empty,
                       printf %g rendered it as 0, and EVERY rho was tagged r0 -- so the whole
                       ladder would have overwritten itself into a single arm."
```

An under-specified path key collapsed a five-point sweep into one arm, and the
`[ -f "adapters/$ARM/unet.pt" ]` guard at `:28` would have served the first build to all five
configurations. This is the exact mechanism D-015 … D-018 remain exposed to.

---

## SECTION 6 — Artifact integrity

Audit clock 12:47:55 → 12:52:53. Nothing was writing into `outputs/` or `upstream_runs/` by
12:52:53; the three atomic jobs finished during the window (r2 pools 12:47:47, r4d 12:45:04,
r4c 12:25:28). Nothing was cleaned up — all of the below is reported, not acted on.

### D-020 · (3) VALID — transient attack directories are clean

Conventions confirmed in code: `adapters/_atk_<label>` (`attack_relearn.py:381`),
`_db_` (`attack_dreambooth.py:146`), `_lora_` (`attack_lora.py:197`),
`_lmc_tmp_` (`lmc_probe.py:40`), `_redun_` (`redundancy_sweep.py:40`).
**Zero instances of any of them exist.** Full-tree sweep for `_atk_*`, `*.tmp`, `*.partial`,
`*.inprogress`, `*.part`, `*~`, `.nfs*` (excluding `venv_*`, `hf_home`, `upstream`): **no
matches**; no `outputs/_*`. **Orphan count 0, orphan disk usage 0 B.** The 23
`rm -rf "adapters/_atk_$K"` cleanups have held, and `retain_scope_slice.py:39,88` also filters
`_atk_` prefixes so a straggler could not silently enter a metric.
`adapters/_base_ref/unet.pt` is persistent by design (live input to `lmc_probe.py:29`,
`weightspace_predictor.py:57`), zip-valid, 690 entries.

### D-021 · (3) VALID — no zero-byte or truncated JSON

**1,428 `.json` checked** recursively under `outputs/`: **0 zero-byte, 0 `json.load` failures, 0
modified within 180 s. 0 NaN/Inf across all 1,428 files. 0 all-zero metric blocks.**
Family keys are complete: `atk_*` (n=808) all carry `study`/`label`/`concept`/`hparams`/
`attack_validity`/`recovery`; `gen_*` (n=346) all carry
`study`/`label`/`unet`/`hparams`/`per_concept`/`minutes`. `upstream_runs/`: 0 zero-byte, 6
`metadata.jsonl` complete, all three `.pt` zip-valid and complete, and
`upstream_runs/meta/dataset/` has all 6 splits at exactly 30 PNG + metadata.jsonl (not partial).
Eight zero-byte files exist elsewhere, none JSON: 4 stale empty stdout logs from Oct 22–23 and 4
chain-gate sentinels.

### D-022 · (1) CONFIRMED DEFECT — one truncated checkpoint, the surviving corpse of D-014

```
adapters/utrap_church_l1.0k4/unet.pt
  actual   511,229,952 bytes   (487.55 MiB)
  expected 1,719,229,888 bytes (group modal, n=99)
  short by 1,207,999,936 bytes = 29.74% of expected
  mtime    2026-10-25 13:55:40
  zipfile  BadZipFile: File is not a zip file   <- central directory never written
```

A `torch.save` interrupted mid-stream — exactly mechanism (1) of D-014, and matching
`run_utrap2.sh:5-6` ("the K=4 arm died mid-build"). **It satisfies `[ -f adapters/$A/unet.pt ]`**,
so every build guard in the project considers this arm built.

**(4) HISTORICAL impact: none.** No metric was computed from it — no
`outputs/gen_utrap_church_l1.0k4.json`, and **0 grep hits for `l1.0k4`** across all 1,428 JSONs
and 372 `log_*.txt`. Absent from `registry_v1.json` `defence_builds`, while sibling
`utrap_church_l1.0k8` is registered `PRESENT` with sha256 `40a0fee0…30a61`. Cost is 487.55 MiB
stale. Two other size outliers are **benign and verified sound**: `_base_ref/unet.pt`
(+38,016 B, 690 entries, zip-valid) and `REF_imma_vangogh/unet.pt` (−9,600 B, 690 entries,
zip-valid, different upstream, written 12:39:57 and complete).

### D-023 · (4) HISTORICAL — 26 metrics whose checkpoint is gone (already documented)

136 `adapters/<label>/` dirs: 108 hold `unet.pt`, **28 are empty**; **26 of the 28 have a live
`outputs/gen_<label>.json`** (`adv2_church`, `adv3_church`, `h47_adv_a0.{0,3,5}`,
`h49_para_a0.3`, `h51_ff_a0.{3,5}`, `h52_det_a0.3_s{0,1}`, `h52_para_a0.3_s{0,1}`, `h53_tn_a0.3`,
`h53_vg_a0.3`, `h54_gt_a0.3`, `h56_broken_a0.7`, `h61_church_a0.3`, `h61_van_gogh_a0.3`,
`h63_church_a0.3`, `h63_van_gogh_a0.3`, `uce_matched`, `undo_a0.{0,3,5,7}_*`). This is the known
retention gap, honestly recorded: `registry_v1.json` carries 335 `defence_builds` with
`checkpoint.state` = **PRESENT 94 / ABSENT 234 / NO_CKPT_BLOCK 7**, each ABSENT annotated
*"checkpoint deleted (pre-2026-10-24 retention gap, RESOURCE_NOTE.md); the run is re-derivable but
not weight-verifiable"*. Consistently, **47 of 335** builds carry
`target_suppression_acc: null` — exactly the pre-2026-10-23 cohort. No `unet.pt` is under 1 MB
apart from D-022's, so there is no *undetected* truncation.

Two records are doubly unverifiable (no active target metric **and** no weights):
`outputs/gen_h63_church_a0.3.json` and `outputs/gen_h63_van_gogh_a0.3.json`.

### D-024 · (2) SUSPECTED — `registry_v1.json` is 7.6 h behind disk

Generated 2026-10-26 05:11:25. Disk now holds **108** `unet.pt` vs **94** registered `PRESENT`,
and **1,428** JSONs vs its own `json_files_under_outputs: 1278`. 14 checkpoints and 150 JSONs
postdate it. Any gate reading the registry as current is reading a 7.6-hour-old view that excludes
the entire R1/R2/R4 upstream-fidelity campaign.

### D-025 · (2) SUSPECTED — `gen_*` metric keys split two ways

```
301 files: forget_mean_acc_LEGACY / retain_mean_acc_LEGACY  (+ legacy_denominator_note, active_target_metrics)
 45 files: forget_mean_acc / retain_mean_acc                 (new naming)
```

No overlap; no file missing both. **Any cross-file aggregation keying on one spelling silently
drops the other 301 or 45.** Worth a grep across the analysis scripts before the next table
regeneration.

### D-026 · (3) VALID with a location/permission inconsistency

`outputs/r4b_upstream/` (3 files) and `outputs/r4c/` (2 files) hold five
`upstream_church_lamb*.safetensors` at 76,681,464 B each (383 MB) with mode **0600**, unlike every
other artifact in the tree — model weights living under `outputs/` rather than `adapters/`. Both
sets are **complete** (`run_r4c_queue.sh:14-16` requests exactly `lam ∈ {0.5,0.1}`; `logs/r4c.log`
ends `R4c DONE`). Cosmetic, but it means `adapters/` is not the single source of truth for weights.

### Also stale, stated not acted on

`.labels/` 194 empty dirs (~795 KiB of directory entries, see D-017); `tmpwork/` empty since
Oct 23; `scripts/__pycache__` (265,400 B, regenerable); `scripts/keep_undo_ckpts.sh` retention
watcher running since **2026-10-21 22:30** (4 days); a `pgrep` waiter loop for
`weightspace_predictor.py` idle since 00:40. Per instruction, no process was killed.

### Observation for the lead

Between 12:51 and 12:53 — *after* `FREEZE_ACTIVE` was declared at 12:43 — source files were being
modified: `scripts/attack_core.py` 12:52:39, `run_e2h_meta_v2_stronger.sh` 12:52:39,
`scripts/gen_eval_final.py` 12:52:49, `scripts/erase_utrap.py` 12:52:51. (`scripts/gpu_reserve.sh`
at 12:51:37 is mine.) Those four are other agents' repair edits, which the freeze permits; noted
only so that D-014/D-018 line numbers in this report are understood to be as-of 12:50, and because
a freeze that permits concurrent source edits means any re-run comparison must pin a commit.

---

## PRIORITISED ACTIONS FOR OWNERS

| # | Item | Owner | Why first |
|---|---|---|---|
| 1 | `tmp + os.replace` at 26 `json.dump` + 23 `torch.save` sites (D-014) | scripts/*.py owner | Converts ~285 existence guards from "bytes exist" to "write completed"; D-022 is its surviving corpse |
| 2 | Replace the 17 `R=$(python …)` + `awk -v r=` kill-rule gates with an explicit rc check (D-010) | Agent 2 / lead | A crashed read currently prints "MEASURED below K1 — destroys the model" |
| 3 | Key `port_dir` on `no_preserve`/`erase_scale`/`preserve_scale`; **re-run R4d** (D-015) | scripts owner | `outputs/r4d_uce_lamb{0.5,0.1}.json` are confounded now |
| 4 | Migrate the 8 `campaign_free.sh` callers to `gpu_reserve.sh` using **`$$`** (D-001) | Agent 2 / lead | Closes the residual legacy-job window |
| 5 | Config-key `r4_uce_official_vs_port.py:133`; delete the unchecked `cp` (D-011) | scripts owner | Filenames assert configs the contents lack |
| 6 | `run_h66_frp.sh:55` — put `$ARM` in the attack key (D-016) | Agent 2 / lead | Silently credits a new arm with an old arm's attacks |
| 7 | `.labels` release-on-exit trap; the 194 existing dirs make 36 runners inert today (D-017) | Agent 2 / lead | Already cost a cell |
| 8 | `set -euo pipefail` + conditional `DONE` in 180 runners; stop `run_smoke_v2.sh:23` truncating its own log (D-009) | Agent 2 / lead | 109 scripts print DONE after failure |
| 9 | Fix 6 `activate` scripts; reconcile `venv_imma` 0.24.0 vs documented 0.31.0 (D-005, D-007) | lead | Only path by which `venv_t2i` could be mutated |
| 10 | Delete `pick_gpus.sh` or restrict it to GPUs 0–1 (D-004) | lead | Would land us on uid 1010's GPUs 2–3 |
| 11 | Resolve whether `venv_uce` was meant to carry a diffusers upgrade (D-008) | lead | Determines whether R4* measures upstream or our stack |
| 12 | Regenerate `registry_v1.json`; unify `gen_*` metric key spellings (D-024, D-025) | lead | Gate inputs are 7.6 h stale and split two ways |
