# Audit & repair register v1 — repair-first mode

**Opened:** 2026-10-26T12:43-04:00 · **Audited snapshot:** `644dc2716b54630b2d6b6e941eda4af14eb37d55`
**Repo HEAD at freeze:** `644dc271…` (identical; 0 commits ahead, working tree clean)
**Scope:** the current T2I study and its dependencies. Not other projects.

**Status: FROZEN.** No new campaign cells. Bounded verification runs only, launched by the
implementation lead. Agents may not launch GPU work. GPUs 2–3 untouched throughout.

---

## 0. Live state recorded at freeze

| | |
|---|---|
| GPU 0 | 15.3 GB, 100% — **three of our python jobs simultaneously** |
| GPU 1 | 34 MiB, idle |
| GPU 2–3 | another user, 4.85 GB each — untouched |

**Runners frozen (SIGTERM to the bash loop only; python children left running):**
`run_r1c_eval.sh` (1369358), `run_r4d_upstream_defaults.sh` (1369488),
`run_r2_meta_official.sh` (1369894). After freeze: **0 runners alive, 3 atomic jobs preserved.**

**Atomic jobs allowed to finish:** `r2_build_meta_pools.py` (1369932),
`gen_eval.py --label REF_imma_vangogh` (1370775),
`r4c_uce_checkpoints_and_images.py --lamb 0.1 --no-preserve` (1371955).

---

## 1. Confirmed defects (evidence first, by the lead, before agents)

### D-001 — GPU reservation is check-then-select, not atomic. **Observed live.**

**Severity:** blocking · **Owner:** Agent 5 · **Files:** `scripts/campaign_free.sh`, every
`run_*.sh`

**Evidence:** at freeze, **three separate runners had each selected GPU 0** and were running
concurrently on it — PIDs 1369932 (3272 MiB), 1370775 (6594 MiB), 1371955 (5480 MiB), total
15.3 GB. `R1c`, `R4d` and `R2` were launched within ~6 s of each other; each independently ran
`campaign_free.sh`, each saw GPU 0 below threshold, and each took it. GPU 1 sat **idle at 34 MiB**
throughout.

**Root cause:** `campaign_free.sh` *reads* state then *returns* an index; nothing is held between
the read and the launch. Two further weaknesses compound it: it consults
`/proc/<pid>/environ`, which reflects only variables present at process start (a runner exporting
`CUDA_VISIBLE_DEVICES` after startup is invisible), and it falls back to a memory threshold, which
cannot distinguish "free" from "between cells".

**Affected:** any multi-runner period. Results are not *wrong* — the jobs are independent and
CUDA-isolated — but throughput was halved and an OOM could have corrupted a cell.

**Repair:** exclusive per-GPU lock held for the whole campaign (`flock` on a lock file naming the
owning PID), with stale-lock reclamation and release on exit/crash. **Status: OPEN.**

### D-002 — `run_r2_meta_official.sh` never invokes the upstream trainer.

**Severity:** blocking · **Owner:** Agent 2 · **Files:** `run_r2_meta_official.sh`

**Evidence:** the script builds pools, prints counts, echoes `R2 POOLS DONE`, and exits. There is
no call to `upstream/Meta-Unlearning/src/meta.py` anywhere in it. It was nevertheless described in
commit `644dc271` as implementing R2.

**Affected claims:** every statement that "R2 is implemented"; the status table row for
Meta-Unlearning. **Repair:** implement the full path. **Status: OPEN.**

### D-003 — FINDING77's ratios are wrong.

**Severity:** high · **Owner:** Agent 1 + Agent 6 · **Files:** `docs/FINDING77…md`

**Evidence (recomputed from `outputs/r4b_uce_lamb*.json`):** the document states our selection's
float32 error is "~60,000× worse than upstream's selection at the same λ". The raw values at
λ=0.1 are `C_port = 6.26e-02` and `C_upstream = 1.05e-03`; the ratio is **≈ 60×**, not 60,000×.
The companion "60× worse than the full-pipeline difference at λ=1000" compares 6.26e-02 against
A(λ=1000)=1.084e-01, a ratio of **≈ 0.58×** — i.e. *smaller*, not larger. **Status: OPEN.**

### D-004 — "A−B isolates embedding selection" is not a valid inference.

**Severity:** high · **Owner:** Agent 1 · **Files:** `docs/FINDING77…md`,
`scripts/r4b_uce_upstream_vs_port.py`

**Evidence:** A and B are each *relative Frobenius error norms* against different references.
Their difference is not the norm of any tensor difference, so "A−B" does not measure a
selection-attributable discrepancy. A direct measurement requires holding every other factor fixed
and differencing the **tensors**. **Status: OPEN.**

### D-005 — FINDING76 overstates both the conclusion and the mechanism check.

**Severity:** high · **Owner:** Agent 3 + Agent 4 · **Files:** `docs/FINDING76…md`

**Evidence:** (a) the headline generalises to "unroll depth buys nothing" from four settings at one
budget; (b) §3 claimed a "mechanism check passed" from n = 7/3/1 probes; (c) the §7 correction then
asserted K=1's range *overlaps* K=32's point — it does not: K=1's extreme is −0.00457, K=32's point
is −0.00638, **disjoint**; (d) K=0 is described as dominating, but its better target suppression
(0.0000 vs 0.3333) came with **lower** retained utility (0.8667 vs 0.9667), so it does not dominate;
(e) "sample-matched" excludes unequal probe overhead and is stated as if exact. **Status: OPEN.**

### D-006 — R4c does not satisfy its advertised claims.

**Severity:** high · **Owner:** Agent 1 · **Files:** `scripts/r4c_uce_checkpoints_and_images.py`

**Evidence:** its docstring advertises an `--order-probe` that **does not exist** in the file. It
scores with `list(ge.CONCEPTS)` rather than the shared `LABELS`/`LAB_KEY` used by every other H
number. It calls `erase_uce.py` via `os.system` and then tests only `os.path.exists`, so a non-zero
exit with a stale checkpoint would read as success. No images are saved. **Status: OPEN.**

### D-007 — E2h's declared grid and its actual loop differ.

**Severity:** medium · **Owner:** Agent 4 · **Files:** `run_e2h_meta_v2_stronger.sh`

**Evidence:** the header declares *"lr 5e-5 and 1e-4 × steps {25} × seeds {0,1,6}, plus lr 5e-5 at
steps 10 … 12 cells"*, while the loop iterates `"5e-5 25" "1e-4 25" "5e-5 10" "5e-5 50"` — the
**50-step cell is not in the declaration**. It must not be described as preregistered.
**Status: OPEN.**

### D-008 — FINDING78's mandate claim is refuted by its own falsification grid. **RETRACTED.**

**Severity:** high (it was the project's only "meets the mandate" claim) · **Owner:** lead ·
**Files:** `docs/FINDING78…md`, `run_e2h_meta_v2_stronger.sh`

**Evidence:** FINDING78 §3 claimed `meta2_church_lt0.5` met both mandate halves at 25 steps on
max-over-valid H = 0.0667. That was measured at **one attacker lr (1e-5)**. E2h swept the attacker
lr at the *same* 25-step budget (`outputs/atk_E2H_*`):

| attacker lr | gate-valid | max-over-valid H | verdict |
|---|---:|---:|---|
| 1e-5 | 1/3 | 0.0667 | ≤ 0.40 |
| 5e-5 | 1/3 | 0.2000 | ≤ 0.40 |
| **1e-4** | **1/3** | **0.5333** | **> 0.40 — FAILS** |

Max-over-valid at 25 steps is therefore **0.5333**, and by the FINDING35 §4 ratchet it can only
rise. **No arm in this project has met both halves at any budget.** Two further facts: every
25-step cell had only **1/3** attacks gate-valid at all three learning rates, so each cell rests on
a single valid seed; and at lr 1e-4 the two gate-failing seeds recorded H = 0.0000, i.e. a *stronger*
attacker produced *more* gate failures.

**Repair:** FINDING78 rewritten — title claim retracted, §1 retention result re-verified and kept
(it is unaffected), §2 de-pooled by campaign, §3 rewritten as a retraction. The filename still
asserts the retracted claim and is retained only for link stability, with an explicit notice.
**Status: REPAIRED (lead), 2026-10-26.**

### D-009 — the IMMA reference evaluation is internally contradictory. Artifact suspect.

**Severity:** blocking · **Owner:** Agent 2 · **Files:** `scripts/r1_eval_imma.py`,
`outputs/gen_REF_imma_vangogh.json`

**Evidence:** the checkpoint is ESD-VanGogh-erased + IMMA, so van Gogh should be suppressed and the
other eleven concepts near base. The opposite was measured: **van_gogh 0.867** while
`utility_all_nontarget = 0.0424` (n=11) and `legacy_retain8 = 0.0333`. Control: `gen_base_sd14.json`
and `gen_E6cal_BASE_sd14_on_DEV.json` both score **1.000 on all twelve** concepts with this same
scorer, so neither the scorer nor the base model explains it. `clip_align` is **not** collapsed
(0.165–0.217, normal), so the model emits coherent images — the wrong ones.

**My root-cause hypothesis was WRONG, and Agent 2 refuted it with direct evidence.** I had argued
`r1_eval_imma.py`'s fuzzy key matching with reshape was mis-assigning tensors. Replicating its
matching loop offline gives **`exact=160 fuzzy=0 reshaped=0 skipped=0`** — the fuzzy path never
fires. Auditing `adapters/REF_imma_vangogh/unet.pt`: **160/160** IMMA tensors byte-equal to
`half(IMMA)`, **526/526** others byte-equal to `half(ESD)`, 0 unexplained, 0 non-finite, and exactly
160 tensors differ from ESD (80 `attn1` + 80 `attn2` — IMMA's own trained scope). The ESD delta is
**not** inverted. The composition is also byte-semantically identical to upstream's own
(`eval/text2image.py:65-69`, `train_text_to_image_lora.py:458-462`). **The composition is valid.**

**Reclassified to (4) HISTORICAL RESULT, quarantine retained on different grounds.** Eight
non-targets at *exactly* 0.000 with the target at 0.867 is a **single-label collapse** of the
closed-set 12-way CLIP argmax, not a spread around 1/12 chance. Independently, IMMA never evaluates
an immunised model standalone (`upstream/IMMA/README.md:146-151`), so this configuration is not one
the authors report. **It remains not citable as "IMMA destroys utility"** — but because the measure
and the configuration are inapt, not because the checkpoint is malformed.
**Status: root cause corrected; quarantine stands; see D-019 for the live suspect.**

### D-019 — the third-party "ESDx1" checkpoint is not a cross-attention-only erasure.

**Severity:** high · **Owner:** lead · **Files:** `upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt`

**Evidence (Agent 2, SUSPECTED):** **all 686** tensors differ from pinned base SD-1.4, with resnet
blocks reaching 0.20 relative — i.e. the whole UNet moved, not just cross-attention. "ESD-x1" should
denote the cross-attention-only variant. By contrast IMMA's own delta is modest and targeted (global
2.7%, max 23%, concentrated in `attn2.to_out`/`to_v`).

**Why this is now the leading explanation for D-009's numbers:** if the downloaded checkpoint is a
full-UNet erasure, the composition's utility collapse is inherited from **ESD**, not produced by
IMMA — and we have attributed nothing because we never evaluated the ESD checkpoint **alone**.

**Decisive missing measurement (queued, not run):** score the ESD checkpoint by itself on the same
12 DEV concepts, same scorer, same seeds — 180 images, ≈5 GPU-min. Until that control exists, no
statement about where the utility loss comes from is supportable.

**Deliberately NOT launched now.** Both our GPUs are idle, but launching by hand is precisely the
failure that produced D-001 (three runners, one GPU). This run is queued as the **first** job
through Agent 5's reservation path once G1 passes. **Status: OPEN, queued behind G1.**

### D-020 — upstream `src/meta.py` does not run at the pinned commit.

**Severity:** blocking for G5 · **Owner:** Agent 2 · **Files:** `upstream/Meta-Unlearning/src/meta.py`

**Evidence:** `train_step` returns a 4-tuple while **11 call sites unpack 3**; `train_loss` is never
assigned; `param.grad` is `None`. The file is unrunnable as published at
`5a2a7600895a2885323cbbf599acaf3f74a96f64`. The runnable path is `src/esd_meta.py`, whose correct
initial checkpoint is **plain SD-1.4** (it erases internally) — not an externally erased checkpoint.

**Consequence for the plan:** "run official Meta-Unlearning" cannot mean `meta.py` at this commit.
Any R2 result must state which upstream entry point ran and that `meta.py` itself is broken
upstream. This is a finding about the release, not a licence to substitute our own code and call it
official.

**Resolved argument hazards, verified:** `--gamma2_1` passed three times resolves to
`0.01, 1.0, 1.0` (so `--gamma2_2`/`--gamma2_3` never receive values); `--fix_timesteps False`
evaluates to **True** under `type=bool` — **only omitting the flag yields False**;
`--fixed_time_steps 1 2 5 10` hard-errors.

### D-021 — IMMA training ran 5× the documented ascent budget, on partly blank data.

**Severity:** high · **Owner:** Agent 2 · **Files:** `run_r1b_imma_train.sh`, `upstream_runs/imma/data/vangogh`

**Evidence:** `--max_train_samples 20` was omitted, so training ran 5000 steps — **2500 ascent
steps where the README specifies 500**. Separately **3 of 100** training images are NSFW-blanked
**black** frames (`7.png`, `29.png`, `41.png`) and were trained on. Also: the 160-tensor trained
scope includes **80 self-attention** tensors, so our "xatten" label is a misnomer (the manifest is
correct; `PREREG_H82` quotes the paper rather than the code). LoRA warm-start, not reinitialised —
our docs were already correct.

### D-010 — pinned upstream UCE may be editing nothing; the whole R4 comparison is at risk.

**Severity:** blocking · **Owner:** Agent 1 · **Files:** `scripts/r4b…py`, `scripts/r4c…py`

**Evidence (R4c + R4d, all four runs complete):**

| run | n_preserve | λ | BASE | UPSTREAM | OUR_PORT | agree | ckpt rel max / mean |
|---|---:|---:|---:|---:|---:|---:|---|
| r4c λ0.1 | 10 | 0.1 | 15/15 | **15/15** | 0/15 | 0/15 | 3.8427e-01 / 2.7552e-01 |
| r4d λ0.1 | **0** | 0.1 | 15/15 | **15/15** | 0/15 | 0/15 | 3.8421e-01 / 2.7560e-01 |
| r4c λ0.5 | 10 | 0.5 | 15/15 | 14/15 | 2/15 | 2/15 | 3.2194e-01 / 2.1737e-01 |
| r4d λ0.5 | **0** | 0.5 | 15/15 | **15/15** | 2/15 | 2/15 | 3.2160e-01 / 2.1740e-01 |

> **CORRECTION 2026-10-26 — my own reading of this table was partly wrong (Agent 5, confirmed by
> the lead).** I first wrote that R4d "tests upstream as intended and reproduces R4c to four
> significant figures", concluding the preserve pool is inert. **The port side of that comparison is
> confounded.** `r4c_uce_checkpoints_and_images.py:77` keys the port checkpoint directory on
> `{concept, lamb}` **only**, and `:78` skips the rebuild when `unet.pt` exists; `--no-preserve`
> (`:44`) feeds only upstream's `preserve` list (`:63`) and the output filename (`:169`) — it is
> **never passed to `erase_uce.py`**. Verified: `adapters/R4C_port_church_lam0.1` is stamped
> **12:24:17**, before r4c's output (12:25:26) and **unchanged** when r4d wrote at 12:45:04. So
> **R4d re-used R4c's port build, made with the default 10-concept PRESERVE list**, and r4d's
> recorded `"n_preserve": 0` is **false for the port arm**.
>
> What that costs and what survives: the OUR_PORT column being identical across r4c and r4d is
> **tautological**, not a measurement, so the port side gives **no** evidence about the preserve
> pool. Upstream's side *was* computed fresh in-process each run, so "the preserve pool barely moves
> **upstream's** edit" is supported — for upstream only. **R4d must be re-run with the port
> directory keyed on the preserve configuration** before any port-side claim. Logged as D-026.

**(a) Upstream UCE erases nothing** — 15/15 church retained at λ=0.1 *and* at λ=0.5, upstream's own
default, and this arm is **not** affected by the caching confound because upstream is recomputed
in-process on every run. UCE demonstrably erases church, so a null result from the official code at
its own defaults is far more likely to mean **our invocation is wrong** than that the method fails.

**The decisive missing measurement:** UPSTREAM-vs-**BASE**, per matrix. A rel norm of 0.38 against
our port proves the two *checkpoints* differ; it does **not** prove upstream's differs from base.

> **RESOLVED 2026-10-26 — and my suspicion was WRONG. Agent 1 refuted it.** The measurement, on all
> 32 `attn2` K/V matrices against the pinned base upcast to fp32:
>
> | checkpoint | vs BASE max | vs BASE mean | bitwise identical to base |
> |---|---:|---:|---:|
> | `upstream_church_lamb0.1` | 9.7948e-02 | **5.3620e-02** | **0 / 32** |
> | `upstream_church_lamb0.5` | 9.7899e-02 | **5.3593e-02** | **0 / 32** |
>
> Upstream changed **every** matrix, by 5.36 % of weight norm on average. `UCE()` writes into a
> `copy.deepcopy` (`uce_sd_erase.py:21-22, 82`) and saves those tensors (`:86-88`), so the
> `.safetensors` captures the edited weights. **No "upstream" number in R4b/R4c/R4d is measuring the
> base model** — the hypothesis I pushed is dead, and R4's upstream arm is a real edit.
>
> It is also not a *small* edit: cross-attention K/V **output** for `"a photo of a church"` moves
> **43.4 %** for upstream against **58.7 %** for our port. So a mere **1.35×** gap in activation
> space separates 15/15 from 0/15 in CLIP label. **Behaviour is not proportional to the numerical
> gap** — which independently undermines every weight-norm-based behavioural inference in FINDING75
> and FINDING77, the documents those findings are built on.
>
> Agent 1 also killed the three obvious fixes: `erase_scale` **1 → 10 moves the edit from 0.0536 to
> 0.0536** (inert to four digits), and neither `lamb` nor the guide string (`''` vs `' '`) helps. The
> one lever that does move it is **`expand_prompts=true`** (6 erase prompts instead of 1): dW/W rises
> to **0.0925**, *above* our port's 0.0587. So the live invocation question is no longer "did
> upstream write anything" but **"should it have been given an expanded prompt set"**.

**Status: my hypothesis refuted; the real question narrowed to prompt expansion. Awaiting Agent 1's
FINDING75/77 rewrites.**

### D-011 — the evaluated checkpoint is unrecoverable from older `gen_*.json` records.

**Severity:** medium · **Owner:** Agent 6 · **Files:** `outputs/gen_*.json`

**Evidence:** `unet` is `null` in records that demonstrably scored an *edited* model.
`gen_base_sd14.json` (unet=null) scores 1.000 on all twelve — genuinely base. But
`gen_advunlearn_parachute.json` (unet=null) scores parachute 0.133 with all others 1.000, and
`gen_advunlearn_{tench,vangogh,garbage_truck}.json` likewise show a single suppressed concept. Those
also carry `target=null`, so the active target is inferable only from the filename. Newer records
(`gen_REF_imma_vangogh.json`) record both, so this is an older-record defect.

**Corrected by Agent 6:** there are **five** unattributable `advunlearn_*` records, not four —
`gen_advunlearn_official.json` is the fifth. And `gen_E6cal_BASE_sd14_on_DEV.json`, which I had
grouped with them, **does** record its target, so it is not unattributable; it remains valid as the
base control.
**Repair:** label the provenance limitation wherever these are cited; do not delete the numbers.
**Status: OPEN, routed.**

### D-012 — E2h reported success after an abort, and pooled two campaigns into one table.

**Severity:** medium · **Owner:** lead · **Files:** `run_e2h_meta_v2_stronger.sh`

**Evidence:** a low-disk abort used `break 2`, which fell through to `echo "E2H DONE"` — an aborted
run announced completion and exited 0. Separately the summary globbed `atk_E2[GH]_*` and printed
E2g's frozen-grid cells and E2h's lr-sweep cells in **one undifferentiated table**, so no reader
could tell which campaign produced a row. Also no `attack_relearn.py` exit code was checked.
**Repair:** per-cell exit codes checked; an abort or failure now writes `E2H INCOMPLETE` and exits
1; the summary tags every row with its campaign and its declared/post-hoc status, and prints the
ratchet caveat. Verified by re-running the summary block on existing artifacts — output in §3.
**Status: REPAIRED (lead), 2026-10-26.**

### D-013 — the freeze marker I created is read by nothing. **My own defect.**

**Severity:** high · **Owner:** lead + Agent 5 · **Files:** `FREEZE_ACTIVE`, `scripts/gpu_reserve.sh`

**Evidence:** `FREEZE_ACTIVE` exists (301 bytes, written 12:43). Grepping every `*.sh` and `*.py` in
the repo outside `upstream/` for the string `FREEZE_ACTIVE` returns **no match**. No runner, no
scheduler and no reservation script consults it.

**Why it matters:** the freeze currently holds only because I terminated the runner loops by PID. A
marker that nothing reads is theatre — any runner started by hand, by a stale cron, or by a resumed
script would acquire a GPU and proceed as if nothing had happened, and I would have cited the
marker's existence as evidence it could not. The freeze was real; its *enforcement* was not.

**Repair:** make the marker authoritative at the single choke point every GPU job must pass — the
reservation call. `gpu_reserve.sh` must refuse to grant a reservation while `FREEZE_ACTIVE` exists,
exit non-zero, and print the marker's contents so the caller learns why. An explicit override must
be required for the lead's own bounded verification runs, and every override must be logged.
**Status: OPEN, routed to Agent 5.**

### D-014 — Meta-v2's shared ζ is near-inert, and its paired retain difference is unpaired.

**Severity:** high · **Owner:** Agent 3 (found), lead (interpretation) ·
**Files:** `scripts/meta_unlearn_v2.py`, `docs/FINDING78…md`

**Evidence, verified independently by the lead at `meta_unlearn_v2.py:288-306`:** one shared ζ
multiplies `+L_R(θ)` and `−L_R(θ′)`, so the net is ζ·(L_R(θ) − L_R(θ′)) ≈ ζ·α·∇L_R·∇L_F — second
order. **ζ is near-inert as a coefficient**, and no ζ sweep has been run. Separately, `dmloss`
draws fresh noise and timesteps from the shared generator on each call, so Eq. 8's intended *paired*
difference is unpaired; the logged `retain_change` field is dominated by sampling noise. The
reference instead uses **independent** γ coefficients run asymmetrically (γ1_1 = 0.1 vs γ2_3 = 1.0),
which is what prevents the cancellation upstream.

**Where I did not accept the agent's conclusion.** Agent 3 concluded FINDING78's retention gain
"cannot be attributed to the restored `+ζL_R(θ)` term". That conflates *inert as a coefficient* with
*does nothing*. Variant A carries `−lam·L_R(θ′)` alone — a negative coefficient on the retain
*level*, i.e. rewarded for degrading benign capability (FINDING56's self-destruct). Pairing it
converts a level penalty into a **change** penalty and removes that incentive. The gain comes from
**restoring the pairing**, while ζ's *weight* is inert and untested. Recorded as FINDING78 §1a.

**Repair:** FINDING78 §1a written; fix for the estimator is **common random numbers**, which changes
numerical behaviour and so must not be applied to any script that produced a historical arm without
a new label. **Status: documented; behaviour change deferred past the gate.**

### D-015 — `attack_core.py` claimed an equivalence that does not exist.

**Severity:** medium (drift risk, no data affected) · **Owner:** Agent 3 ·
**Files:** `scripts/attack_core.py`, `scripts/attack_relearn.py`

**Evidence:** `attack_core.py` described itself as shared by the evaluation attack, but
`attack_relearn.py` **never imports it** — it runs its own inline loop (`:333-362`). `EVAL_FULL_UNET`
is a hand-transcribed *mirror*, and the self-test `t_eval_spec_is_the_reference` tests the mirror
against itself. Agent 3 reports the mirrored values currently match, so **no historical result is
affected**; the defect is silent-drift risk and a false equivalence claim. Pending Agent 3's
explicit confirmation of the value match so that good data is not quarantined over it.

### D-016 — WDR-v2's schedule claim is inverted.

**Severity:** medium · **Owner:** Agent 3 · **Files:** `scripts/erase_wdr_v2.py`

**Evidence:** it claims "ALTERNATING … as the reference does". The reference **sums**: alternation is
gated on an `alternating_` name prefix (`:411`) that `weight_dist_reg` does not carry, so both
branches backward into one gradient buffer followed by a single `optimizer.step()` (`:637-682`).

### D-017 — FINDING67's 4,000× parity figure has the wrong mechanism.

**Severity:** high · **Owner:** lead · **Files:** `docs/FINDING67…md`

**Evidence (Agent 3, to be cited with file:line once supplied):** the derivation ‖∇L‖ = α/√N is
**valid** (re-verified analytically and numerically). But 4,000× is **not** 1/√N parity — with a
*dedicated* Adam, α 1→4000 buys only **9.6×**. The real mechanism is the **shared Adam
second-moment buffer**, an artifact of the divergent two-step schedule. FINDING67 §7 also treats
**gradient-norm parity as optimality** ("systematically under-weights"), which it is not, especially
under Adam. §7's "Meta-v2 needs no coefficient correction" is withdrawn — D-014 supersedes it.
**Status: OPEN, lead to rewrite.**

### D-018 — two provenance misstatements in method scripts.

**Severity:** medium · **Owner:** Agent 3 · **Files:** `scripts/erase_imma.py`, `scripts/erase_adv.py`

**Evidence:** `erase_imma.py` recorded `code_released: False`; **false** — `upstream/IMMA` is the
official repo. It also omits gradient clipping where the reference clips the outer update to 1.0.
`erase_adv.py` is **our own design**, not a Receler/AdvUnlearn reimplementation as labelled, and its
in-place inner-sign repair left no version field, so arms built before and after it are
indistinguishable from the script alone.

**Agent 3 also confirmed all four prior claims independently**, which I had flagged as
not-to-be-trusted: IMMA scope ratio **2.128×** (43,962,560 attn2 / 49,574,080 attn1 /
**93,536,640** total; the official filter `['to_q','to_v','to_k','to_out']` has **no `attn2` test**
at `upstream/IMMA/train/defend_text_to_image_lora.py:561-567`); Meta variant A omits two of three
Eq. 8 terms *and* carries two extra distillation terms; the WSR direction error; and all seven UTRAP
inner-attacker axes.

### D-022 — every FINAL number was scored against the wrong label set. Reaches published findings.

**Severity:** blocking for FINDING65/66 · **Owner:** Agent 4 (repaired), lead (re-score) ·
**Files:** `scripts/gen_eval_final.py:84`

**Evidence, verified independently by the lead.** `gen_eval.py` defines two parallel lists:
`LABELS` (:48) carries the English text CLIP should see — `"garbage truck"`, `"van gogh painting"`,
`"english springer spaniel"`, `"monet painting"` — and `LAB_KEY` (:50) carries the underscored
record keys — `"garbage_truck"`, `"van_gogh"`, `"english_springer"`, `"monet"`. **Eight of the
twelve differ.** `gen_eval_final.py:84` used `list(ge.CONCEPTS)`, whose keys equal `LAB_KEY`, so
CLIP was prompted with *"a photo of a van_gogh"*, *"a photo of a english_springer"*.

**Impact (Agent 4's measurement):** **26 of 180** FINAL images landed on a mis-spelled label, all
concentrated in the four pre-attack arms that `FINDING65`/`FINDING66` report — 9/15, 7/15, 5/15,
4/15 — at median decision margins of only 0.0048–0.0131. **`FINDING65`'s 3.5× and 3× erasure-depth
ratios must not be quoted until re-scored.**

**This is the same defect class as D-006** (`r4c` scoring with `list(ge.CONCEPTS)`). Two independent
scripts made the identical substitution, so it is systemic, not a one-off: any future scorer must
take the label text from `LABELS` and the record keys from `LAB_KEY`, never from `CONCEPTS`.

**Re-score cost and protocol note.** FINAL images were **not saved**, so re-scoring requires
**regenerating** all 180 images (≈5–10 GPU-min). Because seeds, prompts, scheduler and dtype are
fixed and generation is deterministic, regeneration reproduces *the same images* — so this is a
**pure re-score of existing samples, not a second use of the FINAL split**, and must be described
that way rather than as new evidence. (Correction to my own first pass: the records **do** retain
`per_image_predictions` with `pred_label`, `margin` and `target_sim`. My initial check used a crude
substring scan over truncated values and missed the field. The *images* are still absent, so the
re-score cost stands.)

#### D-022a — the defect's direction **manufactures FINDING65's headline**. Lead analysis.

Reading the retained per-image predictions makes the bias direction unambiguous, and it is the worst
possible one. Exactly **nine** of the twelve labels differ between the two lists; **`church`,
`parachute` and `tench` are spelled identically**. So in every church experiment — which is all of
FINDING65 — **the target label's text was correct while all nine competing distractors were
degraded** (`"a photo of a van_gogh"`, `"a photo of a english_springer"`). Degraded distractors are
weaker competitors, so images are pulled *toward* church: **`target_hits` is inflated**.

Measured over all 180 FINAL images, **37 landed on one of the nine degraded labels**, at median
decision margins of **0.0027–0.0065** — margins far smaller than the perturbation a label rewrite
causes:

| arm | target_hits | wins on degraded labels | median margin |
|---|---:|---:|---:|
| `frp_church_r3_l05` | 7/15 | 7 | 0.0051 |
| `uce_sd15_church_lam1000` | 9/15 | 5 | 0.0027 |
| `utrap_church_l0.15k8` | 1/15 | 10 | 0.0046 |
| `utrap_sd15_church_l0.15k8` | 2/15 | 12 | 0.0065 |
| `BASE_sd14_anchor` | 13/15 | 2 | 0.0189 |

**Now the confound.** FINDING65's ratios compare a **DEV** number scored by `gen_eval.py` (which
uses `LABELS` **correctly**) against a **FINAL** number scored by `gen_eval_final.py` (which used
`LAB_KEY`). FRP: DEV 0.1333 vs FINAL 7/15 = 0.4667, ratio **3.5×**. The FINAL side of every such
ratio is inflated by the defect, and inflating it is precisely what produces "erasure is ~3×
shallower on held-out prompts".

**So FINDING65's and FINDING66's headline ratios are not merely imprecise — they are confounded in
the direction of their own conclusion, across two scorers that disagree.** The effect may survive
re-scoring, or may shrink substantially; it cannot be called measured until then. G12 is therefore
promoted from hygiene to a **finding-critical** gate condition, and FINDING65/66 carry a
do-not-quote notice until it clears.

### D-023 — `FINDING73` compares a tench retain8 that includes tench against church arms that exclude theirs.

**Severity:** high · **Owner:** Agent 4 · **Files:** `docs/FINDING73…md:71,87`

**Evidence:** `gen_frp_tench_r5_l05.json`'s retain8 = 0.4583 reproduces **exactly** as the mean
*including* tench (own accuracy 0.0000); excluding it gives 0.5238. The denominator defect is real
but confined: `utility_all_nontarget` (n=11) and the 48-image panel are **correct** and
runtime-asserted. The live error is `FINDING73` quoting retain8 **0.7917** for a tench arm and
comparing it against church arms whose retain8 correctly excludes *their* target — an
apples-to-oranges comparison across the very axis the finding is about.

### D-024 — the validity gate's stated threshold is finer than its own resolution.

**Severity:** medium · **Owner:** Agent 4 · **Files:** `scripts/attack_relearn.py`, `docs/FINDING74…md`

**Evidence:** the gate is `0.20` in all **808** records, but T is a 24-image panel, so the true
decision boundary is **≥ 5/24 = 0.2083**. **13 records failed by exactly one image**, and **two of
them are load-bearing in `FINDING74`**. A threshold quoted to two decimals on a 1/24-resolution
measurement invites exactly this.

### D-025 — 30 duplicate cells disagree, and no record says which code produced them.

**Severity:** high · **Owner:** Agent 4 · **Files:** `outputs/atk_*.json`, `outputs/registry_v1.json`

**Evidence:** **30 duplicate (arm, seed, lr, steps) cells** disagree by up to **8/15 images**, and
no record carries a code-version field, so the disagreements cannot be attributed to a known change.
Separately the registry is **stale for 112 of 808** records, and its quarantine count is
**vacuously zero**. Repaired additively by Agent 4 (no historical record rewritten).

**Agent 4 also cleared the two main scorers:** `LABELS`, `LAB_KEY`, the classifier and the prompt
text are **byte-identical** between `attack_relearn.py` and `gen_eval.py`, and neither file was
modified (mtimes unchanged). `FINAL`'s uninspected-status literal is enforced by a **passing test**
(`test_audit_repairs.py:391`) plus six documents and `split_registry.py:191`; Agent 4 correctly
declined to change a literal it does not own in lockstep with a test it does not own.

**Agent 4 self-corrected a wrong premise it had propagated:** retain8's resolution is **1/120**, not
1/8 — concepts are not images. Its first draft would have dismissed the grid's largest utility gap
as sub-step noise. Corrected with an in-document erratum. Recorded here because the same confusion
could recur anywhere retain8 differences are called sub-resolution.

### D-026 — R4c/R4d cache the port checkpoint on an incomplete key. R4d is confounded.

**Severity:** high · **Owner:** Agent 1 · **Files:** `scripts/r4c_uce_checkpoints_and_images.py:77-84`

Full evidence in the D-010 correction above. The fix is a `port_dir` key covering every parameter
that changes the checkpoint (preserve configuration, `erase_scale`, `preserve_scale`), and a
skip-if-exists guard that verifies a **recorded fingerprint** of those parameters rather than mere
file existence. **R4d needs re-running.** Queued behind G1.

### D-027 — error suppression manufactures the number that trips the kill rule. **Most serious of Agent 5's findings.**

**Severity:** blocking · **Owner:** lead · **Files:** 17 kill-rule gates, 33 fallback sites

**Evidence:** **17** kill-rule gates will print `retain8=0.0000 … MEASURED below K1 -- destroys the
model` when a `json.load` **crashes**, and **33** sites of the form `2>/dev/null || echo 0`
manufacture a `0` on any failure. So a parse error, a missing file or a truncated artifact is
indistinguishable in the logs from a genuine measurement of total utility collapse — and the
conclusion it produces ("destroys the model", K1 FAIL) is exactly the kind this project has
published.

**Why this is the worst one:** every other defect found today distorts a number. This one can
**fabricate a scientific conclusion out of an I/O error**, in the direction of condemning a method.

**RESOLVED by lead audit of the corpus, 2026-10-26 — the risk did NOT materialise.** I re-derived
every `destroys the model` / `below K1` / `retain8 = 0.0000` claim in `docs/` from its artifact JSON.
All trace to genuine evaluation records carrying the full twelve `per_concept` entries at n=15:

| claim | record | re-derived |
|---|---|---|
| `FINDING63`: UTRAP λ=1.0 destroys the model, retain8 0.0000 | `gen_utrap_church_l1.0k8.json` | **retain8 = 0.0** — all eight retain concepts at 0.0; `clip_align` 0.1213–0.1822 |
| `FINDING59`: FRP port at α=1e-5 destroyed, retain8 0.0000 | `gen_imma_church_frp_a1e-5_s300.json` | **retain8 = 0.0**, 12 concepts present |
| `FINDING45`: H69 fails K2, retain 0.0000 / 0.0583 | `gen_h69v2_fragnorm_a0.3_{church,parachute}.json` | **0.0 / 0.0583** |
| `FINDING58`: GIFT λ=10.0 destroys the model | `docs` quote 0.0333 | a non-zero measurement, never at risk |

**Critically, the truncated checkpoint is NOT behind any of them.** `FINDING63` cites
`utrap_church_l1.0k4`-adjacent wording but the measured arm is **`l1.0k8`** (K=8); the truncated
file is `l1.0k4` (K=4) and **has no `gen_` record at all**, confirming Agent 5's zero grep hits.

So the error-suppression pattern is a **live hazard that has not yet corrupted a published
conclusion**. It still must be fixed — 17 gates and 33 `|| echo 0` sites remain — but no finding is
retracted on its account. **Status: hazard OPEN, corpus CLEAR.**

### D-030 — "destroys the model" may be over-stated: the scorer has no reject option. **New, from the D-027 audit.**

**Severity:** high (affects a whole family of claims) · **Owner:** lead · **Files:** `scripts/gen_eval.py`

**Evidence — a pattern visible only once the low-retain records are read together.** In records whose
retain8 collapses, the surviving accuracy concentrates on **one or two labels, repeatedly the style
label `van_gogh`**:

| record | retain8 | the one concept that survives |
|---|---:|---|
| `gen_h69v2_fragnorm_a0.3_church.json` | 0.0000 | **van_gogh 1.000**, every other concept 0.000 |
| `gen_h69v2_fragnorm_a0.3_parachute.json` | 0.0583 | van_gogh 0.933, monet 0.467, rest ≈0 |
| `gen_imma_church_frp_a1e-5_s300.json` | 0.0000 | **parachute 0.933**, rest 0.000 |
| `gen_imma_church_frp.json` | 0.0000 | van_gogh 0.467, parachute 0.267, rest 0.000 |
| `gen_frp_church_r20_l05.json` | 0.0167 | van_gogh 0.533, rest ≈0 |
| `gen_REF_imma_vangogh.json` (D-009) | 0.0333 | **van_gogh 0.867**, eight concepts at exactly 0.000 |

The metric is a **closed-set 12-way CLIP argmax with no reject option**. Under that metric a model
whose outputs become generically painterly is scored as *perfectly* generating van Gogh and *zero*
on everything else — indistinguishable from a model that generates nothing. `clip_align` supports
caution: for `utrap_church_l1.0k8` it is 0.1213–0.1822, depressed against base but **not** collapsed.

**So `retain8 = 0.0000` licenses "closed-set retain accuracy collapsed", not necessarily "the model
is destroyed".** This is the same mechanism Agent 2 invoked to keep D-009 quarantined, and it
generalises to every arm in the table. It does **not** overturn the K1/K2 gating decisions — those
are defined on retain8 and were applied consistently — but the *prose* interpretation is stronger
than the measurement supports.

**Decisive test (cheap, queued):** `gen_eval.py` records only `clip_acc` and `clip_align` per
concept — **no per-image predicted label** — so what those images were classified *as* is currently
unrecoverable. Re-score with predicted labels logged, plus either an open-set margin or a reject
option, on two or three of the arms above. ≈10 GPU-min. Until then, every "destroys the model"
phrasing should read "retain accuracy collapsed to N/120 under the closed-set scorer".
**Status: OPEN, queued behind G1.**

### D-028 — no atomic writes anywhere; one checkpoint is silently truncated.

**Severity:** high · **Owner:** Agent 5 · **Files:** 48 Python files; `adapters/utrap_church_l1.0k4/unet.pt`

**Evidence:** **zero** uses of `os.replace` across 48 Python files, so every artifact is written
non-atomically and a crash mid-write leaves a plausible-looking partial file. One such file exists:
`adapters/utrap_church_l1.0k4/unet.pt` is **511,229,952 bytes against a sibling's 1,719,229,888**
(29.7%), raises `BadZipFile` on open, and **passes every `[ -f ]` guard** in the resume logic.
Verified independently by the lead. Agent 5 reports **0 grep hits** for it in any metric path, so
**no published result used it** — but it is proof that the resume-by-file-existence pattern (D-005
class) is not hypothetical.

### D-029 — `pick_gpus.sh` would hand out another user's GPUs.

**Severity:** blocking · **Owner:** Agent 5 · **Files:** `scripts/pick_gpus.sh:2,11`

**Evidence:** it iterates **all four** GPUs and would return 2 or 3, which belong to **uid 1010**,
not us. That directly violates the standing two-GPU / GPUs-0–1-only constraint. The constraint was
being honoured by `campaign_free.sh`'s narrower loop and by luck, not by this script.

---

### Agent 5's repairs, with verification

- **`scripts/gpu_reserve.sh`** — `flock` critical section; owner recorded as PID **plus process
  `starttime`**, so PID reuse cannot resurrect a dead reservation; `trap`-based release; explicit
  `FOREIGN` / `OURS_UNLOCKED` refusals; GPUs 2–3 **unreachable by construction**; an `adopt` path
  for jobs predating the scheme.
- **`campaign_free.sh` retired** — it now writes nothing to stdout, so waiter loops keep waiting
  instead of double-booking. This is the right failure direction.
- **D-013 (my unenforced freeze) built in as specified:** the check sits **inside** the critical
  section; refusal exits **4**, distinct from 3, so `gpu_reserve_hold` aborts rather than spins; the
  marker text goes to stderr; and the override requires **both** `ALLOW_FROZEN_RUN=1` **and**
  `--allow-frozen` — the explicit argument is what defeats an inherited environment variable.
  **Verified against the live freeze: refused, rc=4, marker printed.**
- **`scripts/test_gpu_reserve.sh`: 50 passed, 0 failed**, CPU-only.
- **Root cause (b) proven dead code, not merely weak:** a probe exporting `CUDA_VISIBLE_DEVICES`
  after startup shows `extracted=[<EMPTY>]` in `/proc/<pid>/environ`. The clause could not match
  **any** of its 8 callers, so selection always fell through to the memory gate that v1 was written
  to replace. The collision is in our own logs: `r1c_eval.log:1` GPU 0 @12:39:29,
  `r4d.log:1` GPU 0 @12:39:32, `r2_meta.log:11` GPU 0 @12:39:35, GPU 1 idle at 34 MiB.
- **Agent 5 caught its own test being wrong, not the lock.** T4 first reported "12 winners"; the
  cause was `$BASHPID` being re-evaluated inside `$( )`, recording the transient substitution
  subshell as owner, which the lock then correctly reclaimed as stale. `acquire` now rejects a dead
  owner PID loudly. **Migration note: callers must pass `$$`, not `$BASHPID`.**
- **venvs:** `venv_imma`/`venv_uce` `activate`, `activate.csh`, `activate.fish` still set
  `VIRTUAL_ENV=…/venv_t2i` — the earlier repair fixed 22 of 25 per env. Direct-interpreter isolation
  is genuinely clean, so impact is **interactive-only**. But **`venv_uce` is a byte-identical clone
  of `venv_t2i`**: the diffusers upgrade our documents attribute to it **does not exist on disk**,
  which decides whether R4* measured upstream's intended stack or ours. Routed to Agent 1.

### D-031 — R4d **overwrote** R4c's upstream checkpoint. R4c is no longer reproducible.

**Severity:** high (irreversible artifact loss) · **Owner:** Agent 1 (repaired) ·
**Files:** `outputs/r4c/upstream_church_lamb*.safetensors`

**Evidence:** the upstream export name is keyed on `{concept, lamb}` only, so R4d's run wrote over
R4c's `.safetensors` in place. Combined with D-026 (the port side cached), **R4c's exact inputs can
no longer be recovered from disk** — its JSON survives, its checkpoints do not. This is the one
defect today that destroyed evidence rather than mislabelling it. Nothing was deleted deliberately;
the naming scheme did it silently. Fixed with fingerprint-verified caching and hard refusals.

### D-032 — six of FINDING77's ratios are wrong; one correction *reinstates* part of FINDING75.

**Severity:** high · **Owner:** Agent 1 (rewritten) · **Files:** `docs/FINDING77…md`, `docs/FINDING75…md`

| claim as published | recomputed |
|---|---|
| "~60,000× worse" | **59.90×** |
| "60× worse" | **0.58×** — i.e. 1.73× *smaller*, and a cross-λ comparison |
| "~6 orders of magnitude" | **2.2 orders** matched; only **0.4 orders (2.54×)** against our own selection |
| "3.7e-07 fp32 accuracy" | λ=1000 only; at λ=0.1 it is **1.05e-03** |
| "11–38%" | 10.8–38.4% |
| "13 of 22 arms at λ=0.1" | **9 of 22** (13 was a count of surviving adapter directories) |

The 0.4-order figure **partly reinstates FINDING75 §2** rather than only retracting things — the
honest direction is not uniformly downward.

**"A−B isolates embedding selection" (D-004) adjudicated:** valid *only* as a triangle-inequality
bound, given a common reference and denominator; **never** as a subtraction attributing a
difference. **Accumulation order is now measured**, not asserted: R4b-vs-R4c bounds it at
**< 1.2e-04 on GPU** but **5.6e-02 on CPU** — it is **BLAS-dependent**, and "λ first vs λ last" is
only **one of four** differences (the rank-1 loop vs a fused GEMM contributes 9.3e-04). Conditioning:
ours **3.24e6** against upstream's **2.22e4**.

### D-033 — `ERRATUM_uce_lambda_scale`'s eigenvalue is 3.8× off, and was never measured.

**Severity:** medium · **Owner:** lead · **Files:** `docs/ERRATUM_uce_lambda_scale.md`

Agent 6 flagged the **85,850** max-eigenvalue figure as single-sourced to a document rather than an
artifact. Agent 1 measured it: **3.24e5**, i.e. **3.8× larger**. The erratum's qualitative point
(λ ≤ 1e3 is inert against `cᵀc`'s spectrum) survives and is in fact *strengthened*, but the number
must be corrected and attributed to a measurement.

### D-034 — our port's erasure is not shown to be clean, and no retain axis was measured.

**Severity:** high · **Owner:** lead · **Files:** `scripts/erase_uce.py`, `docs/FINDING75…md`

**Evidence:** our port's `last..76` selection is a **deliberate and now partly vindicated**
deviation — upstream's single-token edit reproduces our own earlier single-token result
(upstream 5.36 % / 15-15 against our 5.7 % / acc 1.000), so the deviation is why our port erases at
all. But our port's 0/15 **scatters across five different classes at 4× the edit size**, and **no
retain axis was measured in R4c/R4d at all**. So "our port erases and upstream does not" does **not**
establish that our port erases *well* — it is equally consistent with broad damage.

**This is D-030 again in a second place:** suppression measured by a closed-set argmax with no
reject option cannot distinguish erasure from degradation. The R4 family needs a utility panel
before any superiority claim. **Status: OPEN — folded into G13.**

**Also confirmed by Agent 1:** `erase_scale` is **completely inert** (1→10: 0.0536→0.0536 — the same
cancellation λ suffers); `guide=''` ≡ `' '`; `expand_prompts` is the only effective lever and is
**unreachable through `UCE()`**; fp16 serialization contributes **2.07e-04** (0.1 % of the divergence,
but *above* column B, so it is not negligible everywhere); R4b's fp64 reference had been computing
fp32 products (fixed — 3.48e-08 off, so column C stands);
`outputs/r4_uce_official_vs_port.json` is actually the **`erase_scale=2`** run, not the default; and
`venv_uce` violates every upstream pin, though diffusers 0.31's `encode_prompt` passes
`attention_mask=None`, so the embeddings match anyway.

---

## 2. Agent assignments (file ownership, no overlapping edits)

| agent | scope | owns |
|---|---|---|
| 1 | UCE: upstream execution, port, R4/R4b/R4c/R4d, numerics, provenance | `scripts/r4*.py`, `scripts/erase_uce.py`, `docs/FINDING75/77` |
| 2 | Official baselines: IMMA + Meta-Unlearning end to end | `run_r1*.sh`, `run_r2*.sh`, `scripts/r1_*.py`, `scripts/r2_*.py` |
| 3 | Method fidelity: UTRAP/v2, ESD, FRP, GIFT, WDR/WSR, adversarial | `scripts/erase_*.py` (except uce), `scripts/attack_core.py`, `scripts/objectives.py` |
| 4 | Evaluation & analysis: attacks, splits, gates, metrics, seeds, compute, stats | `scripts/attack_*.py`, `scripts/gen_eval*.py`, `scripts/split_registry.py`, `docs/FINDING76` |
| 5 | Execution infrastructure: reservation, queues, env isolation, artifacts | `scripts/campaign_free.sh`, `run_*.sh`, venvs |
| 6 | Documentation & provenance: claims → artifacts → code | `docs/*.md` (report, status, manifest, handoff), this register |

Each finding must be classified: **(1)** confirmed defect with evidence, **(2)** suspected,
**(3)** valid behaviour, **(4)** historical result whose interpretation changes, **(5)** repair +
verification performed.

---

## 3. Verification log

**2026-10-26 ~13:05 — freeze completed cleanly.** All three atomic jobs finished on their own; no
job was killed. Post-freeze state: **GPU 0 at 23 MiB, GPU 1 at 34 MiB — both ours idle.** GPUs 2–3
still the other user's (4853 MiB each), untouched for the whole exercise. Artifacts delivered by the
three preserved jobs: `outputs/r4d_uce_lamb{0.1,0.5}.json`, `outputs/gen_REF_imma_vangogh.json`,
`upstream_runs/meta/dataset/{hrm,irt,tgt}`. Each is assessed above — two of the three turned out to
carry defects (D-009, D-010), which is the point of having frozen before interpreting them.

**2026-10-26 — D-012 repair verified.** Re-ran E2h's corrected summary block against existing
artifacts (CPU only, reads JSON):

```
  camp      lr  steps   prereg  valid  maxH VALID  maxH UNGATED  bar<=0.40
   E2g   1e-05     25 E2g-grid  1/3        0.0667        0.2000        MET
   E2g   1e-05    100 E2g-grid  3/3        0.6667        0.6667     FAILED
   E2g   1e-05    300 E2g-grid  3/3        1.0000        1.0000     FAILED
   E2h   5e-05     10 declared  0/3           nan        0.0000  all invalid
   E2h   5e-05     25 declared  1/3        0.2000        0.2667        MET
   E2h   5e-05     50  POSTHOC  3/3        0.9333        0.9333     FAILED
   E2h   1e-04     25 declared  1/3        0.5333        0.5333     FAILED
```

Campaigns are now separated and the post-hoc cell is labelled. `bash -n` passes. This de-pooled view
is what exposed D-008: the three 25-step rows share a budget and differ only in attacker lr.

**2026-10-26 — Agent 3 integrated; one conclusion rejected, then reconciled.** Agent 3's defects
(A) shared-ζ cancellation and (B) independent noise/timestep draws were **verified independently by
me** at `scripts/meta_unlearn_v2.py:288-306` before adoption. I **rejected** its further conclusion
that FINDING78's retention gain "cannot be attributed to the restored `+ζL_R(θ)` term", on the
ground that it conflates *inert as a coefficient* with *does nothing* — the term's role is to cancel
variant A's retain-**level** penalty. Agent 3 reviewed and withdrew that conclusion
(`AGENT3_FIDELITY_FINDINGS.md` §4.4), keeping the defect. Its §4.3 analysis then **improved on my
own correction**: for UTRAP-v2 the two gradients *are* summed into one `p.grad` before a single Adam
step (`erase_utrap_v2.py:265-274`), so the ratio does set the summed direction and parity is a
defensible *directional* heuristic — unlike the WDR port. That nuance is now in FINDING67 §7.

**Repairs made by the lead off the back of Agent 3's audit:** `FINDING67` header notice (mechanism
corrected with the measured step-size table 0.104 / 0.537 / 0.998, the shared-Adam explanation, and
the withdrawal of the "third instance of one scale pathology" framing for the WDR member), FINDING67
§7 (both sub-claims), `PREREG_H90` (the "λ under-weighting" presumption), `FINDING78` §1a. Report
row L298 routed to Agent 6.

**2026-10-26 — scorer control established.** `gen_base_sd14.json` and
`gen_E6cal_BASE_sd14_on_DEV.json` both score **1.000 on all twelve DEV concepts**. The closed-set
12-label CLIP scorer is sound and base utility is 1.000. Any arm's low utility is therefore a
property of that arm or of its checkpoint composition, not of the metric. Used to refute a scorer
explanation for D-009.

## 4. Quarantine

*(invalid artifacts: preserved, traceable, excluded from aggregates)*

| artifact | reason | status |
|---|---|---|
| `outputs/gen_REF_imma_vangogh.json` | D-009: internally contradictory; checkpoint composition unproven | **provisional quarantine** — not to be cited as an IMMA result pending Agent 2 |
| `outputs/r4b_*`, `outputs/r4c_*`, `outputs/r4d_*` "UPSTREAM" arms | D-010: upstream edit may be inert; UPSTREAM-vs-BASE unmeasured | **interpretation suspended** — files retained, no fidelity claim may rest on them pending Agent 1 |
| `outputs/atk_E2H_*_lr5e-5_s50.json` | D-007: undeclared post-hoc cell | **retained, labelled** — reportable only as a post-hoc diagnostic, never as preregistered |
| `outputs/gen_advunlearn_{parachute,tench,vangogh,garbage_truck}.json` | D-011: evaluated checkpoint unrecorded | **retained, labelled** — numbers stand, attribution does not |

## 5. Resume gate

**Status: NOT PASSED.** No campaign cell may launch until every condition below is satisfied and
countersigned in §3 with the evidence that satisfied it. A condition is met by a measurement or a
passing test, never by an assurance.

| # | condition | blocking defects | evidence required |
|---|---|---|---|
| G1 | Exclusive per-GPU reservation works and is proven to work | D-001 | `scripts/test_gpu_reserve.sh` output showing N racing processes yield exactly one winner per GPU, stale-lock reclamation after a simulated crash, release on crash, and that a GPU held by another user is never selected |
| G2 | No runner reports success after a failure | D-012 + Agent 5 item 4 | every `run_*.sh` checks exit codes; an abort writes an INCOMPLETE marker and exits non-zero; list of every remaining `DONE`-after-failure site, empty or explained |
| G3 | The UCE comparison measures what it claims, or claims less | D-003, D-004, D-006, D-010 | UPSTREAM-vs-**BASE** per-matrix norms. If upstream's edit is inert, FINDING77 must say the comparison establishes nothing about port fidelity. Every FINDING77 ratio recomputed from raw JSON |
| G4 | The IMMA artifact is proven valid or quarantined | D-009 | per-tensor comparison of `adapters/REF_imma_vangogh/unet.pt` against base and against the ESD checkpoint; key-match/reshape/skip counts from `r1_eval_imma.py` |
| G5 | Meta R2 either runs the authors' trainer or stops being called R2 | D-002 | `src/meta.py` invoked with resolved hyperparameters recorded, the `type=bool` `fix_timesteps` hazard handled explicitly, and the forget-vs-related-retain semantic conflict resolved in writing |
| G6 | No incorrect claim stands anywhere, banner or not | D-003…D-008, D-011 | the offending sentences and tables **rewritten**; a banner over an uncorrected claim does not satisfy this |
| G7 | Declared grids match executed grids | D-007 | every runner's declaration enumerates exactly what its loop runs, or the difference is labelled post-hoc |
| G8 | Repairs independently reviewed | — | each repaired component reviewed by an agent **other** than the one that repaired it |
| G9 | Error suppression cannot fabricate a kill-rule verdict | D-027 | the 17 kill-rule gates and 33 `\|\| echo 0` sites distinguish "measured 0" from "failed to read"; corpus already re-derived and **clear**, so this is forward protection |
| G10 | Artifacts are written atomically and verified before reuse | D-028 | `os.replace` on every artifact write; resume guards check a recorded fingerprint, not `[ -f ]`; the truncated `utrap_church_l1.0k4/unet.pt` quarantined |
| G11 | No script can select another user's GPU | D-029 | `pick_gpus.sh` retired or bounded to 0–1; proven by test |
| G12 | FINAL numbers re-scored with the correct label set | D-022 | 180 images regenerated at identical seeds and re-scored from `LABELS`; `FINDING65`'s ratios restated or withdrawn |
| G13 | "Destroys the model" claims scoped to what the metric supports | D-030 | per-image predicted labels logged and a reject option or open-set margin measured on ≥2 collapsed arms; prose rephrased to "retain accuracy collapsed to N/120" |

**Queued bounded verification runs** (lead-launched, through `gpu_reserve.sh`, in this order once G1
passes): ESD-checkpoint-alone control for D-019 (≈5 min) → FINAL re-score for D-022 (≈10 min) →
reject-option re-score for D-030 (≈10 min) → R4d re-run with a corrected `port_dir` key for D-026.
Total well under one GPU-hour. Each is a *verification* run, not a campaign cell.

**Standing constraints that survive the gate.** Two GPUs at a time, GPUs 0–1 only; GPUs 2–3 are
another user's and stay untouched. Agents may not launch GPU work — only the lead may. No paid
external services, no infrastructure permission changes. Closed studies and their conclusions are
preserved; their protocols, results and interpretations are not retroactively altered. Scoring,
aliases and success definitions are not changed to improve apparent performance.

**After the gate, in order:** IMMA (G4) → repaired UCE comparison (G3) → official Meta pipeline
(G5) → UTRAP recovery-depth diagnostics.

---

## 6. Post-gate direction — what the audit has already cost us

Two independent results arrived during the freeze, and both narrow where a defensible contribution
can come from. Recording them here so the post-gate plan starts from the true position.

**(a) We have no arm that meets the mandate.** D-008 retracted the only candidate. The honest
statement is: *no arm in this project satisfies retain8 ≥ 0.85 and max-over-valid H ≤ 0.40 at any
attack budget.* The mechanism of the failure is specific and useful — at a fixed 25-step budget,
max-over-valid rose 0.0667 → 0.2000 → 0.5333 as attacker lr went 1e-5 → 5e-5 → 1e-4. So any future
arm must be evaluated against an **attacker-lr sweep at matched budget**, not a budget sweep at
fixed lr. A defence tuned against one attacker lr is not robust; we have now demonstrated that on
our own best arm.

**(b) The evaluation-side contribution is weaker than hoped.** A novelty check on the three-part
claim — static knowledge metrics being *anti-predictive* of relearning robustness (attack AUC 1.000
vs. inverted probe 0.844, Spearman −0.50 over 16 arms); scoring *below* a retrain oracle being a
warning sign rather than success; and the attack-validity gate — returned **"non-obvious combination,
but weakly so."** No single paper assembles all three or reports the cross-arm statistics, but every
part is anticipated in substance, and one preprint reportedly anticipates all three: *"Distance Is
Not Enough: Forget-Retain Alignment Gap Predicts LLM Relearning Robustness"* (arXiv 2608.25429v1),
whose Table 15 shows the same inversion against a retain-trained gold model and whose Table 11
states the validity-gate principle. Also anticipating parts: MUSE (2407.06460) on over-unlearning
past the retrained model; TOFU (2401.06121) on two-sided testing against the retain model; Łucki et
al. (2409.18025) on ablatable directions; "Erase or Hide?" (2509.22263); Carlini et al.
(1902.06705) §4.14 on attack validity generally.

**These citations are UNVERIFIED by me and must be checked before anything is published against
them.** The check could not run a citation-graph forward search (Semantic Scholar rate-limited), so
its own caveat is that the oracle-inversion part "could still flip to already published." Treat the
closest prior art as a live risk, not a settled fact.

**Consequence for the mandate.** The contribution cannot rest primarily on the evaluation
methodology. It has to be a **method** that survives an attacker-lr sweep, with the evaluation
protocol as supporting apparatus rather than the headline. That is a harder target than the position
assumed before this freeze, and stating it now is cheaper than discovering it at write-up.
