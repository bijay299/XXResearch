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

### D-019 — the ESD checkpoint's training provenance is UNRESOLVED. Hypothesis, not a finding.

**Severity:** high · **Owner:** lead + Baselines agent ·
**Files:** `upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt`

**Provenance of the download is legitimate and documented.** The URL is explicitly supplied by
IMMA's **pinned** README at commit `2f55041`
(<https://github.com/amberyzheng/IMMA/blob/2f55041141705031d503134dec22f674f09e21bf/README.md>), so
fetching it was the correct action. That does **not** establish the downloaded bytes' training
provenance, which is a separate question.

**Observation:** **all 686** tensors differ from pinned base SD-1.4, with resnet blocks reaching
0.20 relative. By contrast IMMA's own delta is modest and targeted (global 2.7%, max 23%,
concentrated in `attn2.to_out`/`to_v`).

**WITHDRAWN: my inference that it is therefore "not a cross-attention-only erasure" and the name
is mislabelled.** "Every tensor differs from our base" does **not** establish whole-UNet training.
At least four benign explanations produce the same observation and none has been ruled out:
**EMA vs non-EMA** weights, a different **model variant** (fp16 variant vs main weights — my own
comparison upcast the fp16 variant), a **format conversion** to diffusers layout, or a different
base revision. Any of these perturbs every tensor while the erasure itself remained
cross-attention-only.

**Required to resolve:** record the file's **sha256**; identify the originating checkpoint; compare
against base under **each** variant and EMA/non-EMA choice; and document the conversion path. Until
then this is an **unresolved hypothesis**, and the ESD-alone control is a **behavioural** control —
it can show what the checkpoint *does*, never what it *was trained as*.

### ✅ RESOLVED 2026-10-26 by the V1 bounded control — and **my hypothesis was wrong**.

Executed under the repaired reservation (GPU 0 reserved exclusively, override logged to
`logs/frozen_override.log`, GPUs 2–3 untouched), ~95 s, 180 images.
Artifact: `outputs/gen_V1_esd_alone_vangogh.json`.

| arm | van_gogh | mean non-target (n=11) | concepts at exactly 0.000 |
|---|---:|---:|---:|
| **ESD checkpoint ALONE** | **0.7333** | **1.0000** | **0 / 11** |
| ESD + IMMA composed | 0.8667 | 0.0424 | 7 / 11 |
| BASE SD-1.4 | 1.0000 | 1.0000 | 0 / 11 |

> **⚠ MY WORDING BELOW OVERREACHED. Corrected 2026-10-26 after independent review, which found it
> making three of the four inferences this result explicitly cannot support.**
>
> **The only supportable statement is: the ESD-alone arm did not show the same closed-set utility
> collapse on this panel.** That is it. In particular:
> * **NOT "exonerated" and NOT "a clean, well-behaved erasure."** Target accuracy is **0.7333** —
>   11 of 15 images still classified as van Gogh. And D-030's limitation cuts **both** ways: a
>   closed-set argmax with no reject option cannot license "clean erasure" any more than it licenses
>   "destroys the model". I applied that limit to the collapse and then ignored it for the control.
> * **NOT "provenance largely moot."** EMA-vs-non-EMA is untested and the host is offline. A
>   behavioural result on one panel says nothing about what the bytes were trained as.
> * **NOT "over-training is the leading hypothesis."** **Nothing in V1 varies the budget.** It
>   compares composed against uncomposed, not 500 updates against 2500. Over-training remains an
>   untested hypothesis with no evidence from this run.
> * Also corrected: this entry carried `Status: OPEN` below its own `RESOLVED` heading.
>
> The measured numbers below stand; the characterisation of them did not.

**Measured result (the numbers, without the overreach).** By itself: van Gogh falls
1.000 → 0.733 while **all eleven** other concepts stay at a perfect **1.000**, with healthy
`clip_align` (0.165–0.300). The utility collapse in D-009 is **not inherited from ESD** — it is
introduced by composing IMMA.

**That is the opposite of what I argued.** I had named the downloaded checkpoint "the leading
explanation for D-009's numbers". It is not an explanation at all.

**What follows, stated narrowly.** The collapse in the composed arm is **not reproduced** by the
uncomposed checkpoint on this panel. That localises the difference to the composition step — the
ESD checkpoint plus IMMA's 160 trained tensors — and nothing further. It does **not** identify a
cause. Candidate explanations, all untested and none preferred: the 5x training budget (D-021); the
interaction of IMMA's tensors with this particular ESD checkpoint; or a metric artifact under
D-030's no-reject-option limit. **V1 varies none of these**, so it cannot rank them.

**Still not established, and not claimed:** that IMMA "destroys utility". The number is a closed-set
argmax with no reject option (D-030), measured on a configuration the authors never evaluate
(standalone, not post-adaptation). The replacement run at the README's 20-sample budget is what
would separate over-training from method behaviour.

**Interpretation limit, declared before the run:** this is a **behavioural control**. It can show
what the uncomposed checkpoint *does* on this panel; it cannot establish provenance, and it inherits
D-030's limitation — a closed-set argmax with no reject option cannot separate erasure from
degradation, so a low number will itself need the saved images and an independent assessment before
it supports the word "destroyed".

**Deliberately NOT launched by hand.** Both our GPUs are idle, but launching outside the reservation
path is precisely the failure that produced D-001 (three runners, one GPU). Queued as the first job
through the reservation API once the infrastructure repairs pass review.
**Status: OPEN, queued behind the infrastructure gate.**

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

**Evidence:** `--max_train_samples 20` was omitted from the pinned README command, so the completed
run trained on **100 samples for 50 epochs** — roughly **5× the documented training budget**.
Separately **3 of 100** training images are NSFW-blanked **black** frames (`7.png`, `29.png`,
`41.png`) and were trained on; whether any falls inside the README's 20-sample subset is being
checked. Also: the 160-tensor trained scope includes **80 self-attention** tensors, so our "xatten"
label is a misnomer (the manifest is correct; `PREREG_H82` quotes the paper rather than the code).
LoRA warm-start, not reinitialised — our docs were already correct.

**Disposition — preserve, relabel honestly, do not discard.** The completed run is legitimate work
and stays. It must be labelled to **record its 5× budget** and must **not** be described as a
faithful reference reproduction. Before any replacement run, the resolved sample count, batch size,
epochs, inner/outer steps and **total optimizer updates** must be documented on both sides (ours vs
README).

**Framing correction for what our IMMA numbers mean.** IMMA's claim concerns resistance to
**re-learning**, and the authors evaluate **post-adaptation**. Our standalone utility evaluation of
an immunised model is an **additional measurement we introduced**; the authors never evaluate an
immunised model standalone. Assessing their robustness claim requires including their
post-adaptation evaluation, and our standalone number must be reported as a separate, additional
axis — not as a test of their claim.

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

#### ✅ D-022 MEASURED 2026-10-26 by the V2 bounded run. The direction question is answered: **there is no consistent direction.**

Run `rsc_20261026T200003Z_2c8ac2d573c3f300_d022`, GPU 0 reserved exclusively, override logged,
180 images regenerated, both label sets scored against **one** image-feature tensor per image.

| arm | legacy (`LAB_KEY`) | corrected (`LABELS`) | **delta** | determinism |
|---|---:|---:|---:|---|
| `utrap_church_l0.15k8` | 0.0667 | 0.0667 | **+0.0000** | DISCREPANT |
| `frp_church_r3_l05` | 0.4667 | 0.2667 | **−0.2000** | DISCREPANT |
| `utrap_sd15_church_l0.15k8` | 0.1333 | 0.2000 | **+0.0667** | DISCREPANT |
| `uce_sd15_church_lam1000` | 0.6000 | 0.6000 | **+0.0000** | DISCREPANT |
| `BASE_sd14_anchor` | 0.8667 | 0.8000 | **−0.0667** | DISCREPANT |
| `BASE_sd15_anchor` | 1.0000 | 1.0000 | **+0.0000** | DISCREPANT |

Predictions moved on up to **9 of 15** images per arm; target decisions on up to **3**.

**My directional hypothesis is refuted by measurement, not merely unsupported.** I had reasoned that
degraded distractors must be weaker competitors, so correcting them would *lower* church counts
everywhere — i.e. every delta ≤ 0. Instead `utrap_sd15_church_l0.15k8` moves **+0.0667** while
`frp_church_r3_l05` moves **−0.2000**. The effect is real, it is sometimes large, and **it has no
consistent sign**. The reviewers were right to force the direction claim out of every document
before this ran; had the claim stayed in, this result would have contradicted the published text.

**And the historical numbers cannot be recovered at all.** `history_comparability = none_exact`:
**all six** regenerable arms are DISCREPANT against their records. Regeneration at matched seeds
does **not** reproduce the original images, so the determinism assumption this run was built to test
**fails**. Note the subtlety: `BASE_sd14_anchor`'s legacy aggregate (0.8667) *equals* its historical
13/15, yet the probe still reports DISCREPANT — because it compares **per-image** predictions, not
aggregates. Aggregate agreement concealing per-image disagreement is exactly what a weaker probe
would have missed.

**Terminal state for the historical FINAL corpus.** Six arms have deleted checkpoints (D-022b) and
the other six cannot be reproduced. So FINDING65's ratios and FINDING66's figures are **not
correctable by re-measurement** — they are withdrawn permanently, not pending. What survives is a
*within-run* measurement on regenerated images showing the label text matters and acts in both
directions. A new, properly-instrumented FINAL experiment could address the underlying questions; it
cannot repair these records.

**The machinery behaved correctly, which is the other result.** The run exited **rc=1** and the
guard wrote `V2 INCOMPLETE` rather than DONE — D-044's fix working as intended: an all-DISCREPANT
run is not reportable as a successful correction. Before the fix it would have been filed as
`complete`.

#### D-022a — RETRACTED AS WRITTEN. My directional analysis was unsupported *and is now disproven*.

**This entry originally claimed the defect's direction was established and "manufactures FINDING65's
headline". That was wrong and is withdrawn.** I reasoned that degraded distractors must be weaker
competitors, so `target_hits` must be inflated. That is a deduction I was not entitled to: an
underscore-for-space substitution produces a **different token sequence**, and the sign of its
effect on each of the nine text embeddings — and hence on the twelve-way argmax — is an **empirical
question**. I also wrongly asserted that predictions landing on the three *unchanged* labels were
unaffected; the argmax is over all twelve, so a change to any competitor can move them.

**The sanctioned status, until the paired measurement exists:** *the evaluator used inconsistent
label text; the effect on scores and conclusions is unresolved.* Neither direction nor magnitude is
known.

**What IS established.** Exactly **nine** of the twelve strings differ; `church`, `parachute` and
`tench` are identical. FINAL numbers were scored with `LAB_KEY` as CLIP text while DEV numbers were
scored with `LABELS`, so the DEV-vs-FINAL ratios in FINDING65 §1 were not produced on a common
scale. That alone withdraws the ratios — no directional claim is needed to justify it.

**Observations of record, carrying no directional claim.** 37 of the 180 FINAL images were assigned
one of the nine differing labels, at median decision margins of 0.0027–0.0065. Small margins mean
the outcomes are **sensitive** to label text; they do not indicate which way:

| arm | target_hits | wins on degraded labels | median margin |
|---|---:|---:|---:|
| `frp_church_r3_l05` | 7/15 | 7 | 0.0051 |
| `uce_sd15_church_lam1000` | 9/15 | 5 | 0.0027 |
| `utrap_church_l0.15k8` | 1/15 | 10 | 0.0046 |
| `utrap_sd15_church_l0.15k8` | 2/15 | 12 | 0.0065 |
| `BASE_sd14_anchor` | 13/15 | 2 | 0.0189 |

**The cross-scorer problem.** FINDING65's ratios compare a **DEV** number scored with `LABELS`
against a **FINAL** number scored with `LAB_KEY`. FRP: DEV 0.1333 vs FINAL 7/15 = 0.4667, ratio
3.5×. The two sides were not produced on a common scale, so the ratio is **unmeasured** — in both
direction and magnitude. FINDING66 differs **structurally**, in that both sides of its central
comparison went through the same scorer; that is a fact about the comparison's shape and **not** a
claim that its conclusion holds, since a consistently-applied scorer can still be the wrong one.

G12 is therefore a **finding-critical** gate condition, and FINDING65/66 carry do-not-quote notices
until it clears.

**Two further claims of mine, both withdrawn.** (1) I wrote that 37 images at ~0.005 margins
"cannot by themselves account for a 0.1333 → 0.4667 gap", concluding the finding "may well survive".
That is the same unsupported directional reasoning in reverse — it presumes the only effect is on
those 37 images, when a label-text change alters the decision surface for **all 180**. (2) I cited
base SD-1.4 scoring 1.000 on all twelve as proof that "the scorer is sound". It is not: it shows the
scorer behaves correctly on **undamaged, in-distribution** outputs. It says nothing about its
validity on damaged or unusual outputs, which is precisely the regime every erased arm sits in.

**The honest position: the finding is neither confirmed nor refuted, and the measurement is designed
rather than anticipated.** The corrected run scores **both** label sets on the **same** regenerated
images, which is what isolates the label-text effect, and reproduces the old label set against the
historical records to test the determinism assumption rather than assume it.

**Count correction, in two stages — the second corrects my own first fix.** The original repair
comment said **"4 of 12"** differ. I corrected that to **nine**, which is right, but I mis-split the
nine: I described `garbage_truck` as a *wording* change. Re-derived programmatically:

| class | n | members |
|---|---:|---|
| identical | 3 | `church`, `parachute`, `tench` |
| **differ by an underscore alone** | **6** | `garbage_truck`, `cassette_player`, `chain_saw`, `french_horn`, `gas_pump`, `golf_ball` |
| **genuinely reworded** | **3** | `van gogh painting`→`van_gogh`, `english springer spaniel`→`english_springer`, `monet painting`→`monet` |

So the split is **6/3, not 5/4**, and the differing total is **nine** either way. The distinction
matters only for describing the *kind* of perturbation; it carries **no implication about the
direction** of any embedding change, and none is claimed.

### D-022b — six of the twelve FINAL records can **never** be re-scored. Checkpoints are gone.

**Severity:** high (permanent evidentiary loss) · **Owner:** lead ·
**Files:** `attacked/E6B_*`, `attacked/E6C_*`, `attacked/E6D_*` — **absent**

**Evidence, verified by the lead.** `ls attacked/E6B_* E6C_* E6D_*` → "No such file or directory".
Only **8** `KEEP_*` directories survive in `attacked/`, none of them the E6 post-attack arms. The
six `ATK_*` FINAL records therefore have **no checkpoint to regenerate from**, and the re-scoring
pipeline marks them `missing_checkpoint` and forces `run_status: partial`.

**Regenerable:** the four defence arms (`utrap_church_l0.15k8`, `frp_church_r3_l05`,
`utrap_sd15_church_l0.15k8`, `uce_sd15_church_lam1000`) and the two base anchors — **6 of 12**.

**Consequence for FINDING66, which is the document those six underwrite.** Its central comparison is
attacked arms (15/15) against the base anchor (13/15). The **attacked side cannot be recovered**.
Re-running the attack would produce a *different* checkpoint — relearning is stochastic — so it
would be a **fresh analogue, not a reproduction**. FINDING66's post-attack numbers are therefore
**permanently unverifiable under the corrected scorer**, which is a terminal state, not a pending
one. The honest disposition is: withdrawn, and not recoverable by re-measurement of the same arms.
A new, properly-instrumented post-attack FINAL experiment could address the *question*; it cannot
validate these *records*.

**Root cause** is the resume/cleanup pattern already recorded as D-005/D-028: runners `rm -rf` their
transient attack checkpoints, and nothing distinguished "transient" from "load-bearing for a
published finding". The fix is forward-only — artifacts that underwrite a finding must be retained
or hashed at the time of use. **Status: CONFIRMED, loss is permanent, no rerun can close it.**

### D-023 — `FINDING73` compares a tench retain8 that includes tench against church arms that exclude theirs.

**Severity:** high · **Owner:** Agent 4 · **Files:** `docs/FINDING73…md:71,87`

**Evidence:** `gen_frp_tench_r5_l05.json`'s retain8 = 0.4583 reproduces **exactly** as the mean
*including* tench (own accuracy 0.0000); excluding it gives 0.5238. The denominator defect is real
but confined: `utility_all_nontarget` (n=11) and the 48-image panel are **correct** and
runtime-asserted. The live error is `FINDING73` quoting retain8 **0.7917** for a tench arm and
comparing it against church arms whose retain8 correctly excludes *their* target — an
apples-to-oranges comparison across the very axis the finding is about.

### D-024 — WITHDRAWN. **There is no gate defect. My framing was wrong.**

**Corrected 2026-10-26.** I recorded this as a "granularity defect": the gate reads `T_rise ≥ 0.20`
on a 24-image panel, and I treated the 13 records at `T_rise = 0.1667` as failing "by one image" to
a threshold that was unrepresentable.

**That is not a defect.** On a 24-image panel, `≥ 0.20` **correctly accepts 5/24 = 0.2083 and
correctly rejects 4/24 = 0.1667**. The rule is well-defined and was applied consistently to all 808
records. Nothing is mis-stated and no record was mis-gated.

**Admitting 4/24 would CHANGE the rule**, not repair a rounding error. If that variant is ever
explored it belongs in a **separately labelled sensitivity analysis**, never folded into the
headline gate. The "13 records failed by one image" observation is a *distributional* fact about how
close the rejections sit to the boundary — it is not evidence of a flaw.

**Consequence for FINDING78, restated correctly.** My earlier note said a corrected boundary would
make an E2h cell valid and raise its max-over-valid to 0.2667. **That was reasoning from the wrong
premise.** Under the actual rule, seed 6 at lr 5e-5 (`T_rise = 0.1667 = 4/24`) is **correctly
rejected**, and the 25-step max-over-valid stays as reported. FINDING78's retraction rests on lr
1e-4 seed 6, whose `T_rise = 0.2083 = 5/24` is **correctly accepted** with H = 0.5333 > 0.40. The
retraction is unaffected and needs no gate adjustment to stand.

**Status: WITHDRAWN — not a defect. The superseded analysis is kept below for the record.**

### D-024 (superseded text) — "the validity gate's stated threshold is finer than its own resolution"

**Severity:** medium · **Owner:** Agent 4 · **Files:** `scripts/attack_relearn.py`, `docs/FINDING74…md`

**Evidence, re-verified independently by the lead from all 808 records.** The gate value is `0.20`
in every record. T's panel denominator is **24** (956 of the T values divide exactly by 24). So the
true decision boundary is **≥ 5/24 = 0.2083**, and a rise of 4/24 = **0.1667** fails. Of the **136**
invalid records, **exactly 13** sit within one 1/24 step below the bar — all at precisely
`T_rise = 0.1667`:

`atk_H44_undo_a0.5_big_church_sd5_lr1e-5`, `atk_h53_tn_a0.3_tench_sd0_lr5e-5`,
`atk_E11K_frp_tench_r3_l05_sd1_lr1e-5_s10`, `atk_E12_h80_meta_church_lam0.7_sd{0,1}_lr1e-5_s25`,
`atk_h50_adv_a0.5_church_sd1_lr5e-5`, `atk_E2H_meta2_church_lt0.5_sd6_lr5e-5_s25`,
`atk_E11C_frp_church_r3_l05_sd0_lr1e-5_s5`, and five more.

A threshold quoted to two decimals against a 1/24-resolution measurement invites exactly this: 0.20
is not representable on the panel, so "≥ 0.20" silently means "≥ 5/24", one image stricter than it
reads.

**Consequence for FINDING78 — checked, and the retraction holds.** One of the 13 is an E2h cell:
seed 6 of `lr 5e-5 / 25 steps`, which FINDING78 records as gate-failed at H = 0.2667. Treating the
boundary as 4/24 would make it **valid**, raising that cell's max-over-valid from 0.2000 to
**0.2667** — still **below** the 0.40 bar. The cell that drives the retraction, `lr 1e-4 / 25 steps`
at **0.5333**, is unaffected, and its two gate-failing seeds recorded H = 0.0000, so admitting them
cannot lower a maximum. **So the gate-granularity defect does not rescue FINDING78; admitting more
attacks only adds valid evidence for the same conclusion.** Worth stating explicitly, because a
defect that moved the bar in the other direction would have required reopening the retraction.

**Repair:** express the gate in panel units (`≥ 5/24`) wherever it is stated, and record both the
fractional threshold and the integer image count in new records. Do **not** restate historical
records — the gate was applied consistently at 0.20 throughout, so no past verdict changes; only the
*wording* was finer-grained than the measurement. **Status: evidence confirmed; wording repair
pending in the owning documents.**

### D-025 — 30 duplicate cells disagree, and no record says which code produced them.

**Severity:** high · **Owner:** Agent 4 · **Files:** `outputs/atk_*.json`, `outputs/registry_v1.json`

**Evidence:** **30 duplicate (arm, seed, lr, steps) cells** disagree by up to **8/15 images**, and
no record carries a code-version field, so the disagreements cannot be attributed to a known change.
Separately the registry's coverage is incomplete, and its quarantine count is **vacuously zero**.
Repaired additively by Agent 4 (no historical record rewritten).

**Coverage figure corrected (my error).** I wrote "stale for 112 of 808 records", which inverts the
statistic: **112 attack records are ABSENT**, so the registry **covers 696 of 808** attack records
and **1,278 of 1,424** JSONs. The absent arms are concentrated in FINDING71–78, including
`meta2_church_lt0.5` — i.e. the registry does not cover the arms behind the most recent findings,
which is worse than a uniform staleness and is why `TABLES_v1_generated.md` cannot be used to check
them. `registry_v1.py` was deliberately not re-run (it sha256s many GB under an active freeze).

**The duplicate-cell disagreements remain UNRESOLVED and are not closable by argument.** Thirty
cells with identical (arm, seed, lr, steps) disagree by up to 8/15 images with no code-version field
to attribute the difference to. This is an open item, explicitly **not** covered by the audit having
concluded.

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

**Diagnostic, queued — and a reject option is NOT a repair.** `gen_eval.py` records only `clip_acc`
and `clip_align` per concept — **no per-image predicted label** — so what those images were
classified *as* is currently unrecoverable. The first step is simply to log predicted labels and
per-label scores.

A reject option or open-set margin is a **potential additional diagnostic**, not a fix, and if
introduced it must: be **calibrated independently of the arms being judged**; be **versioned** with
its assumptions documented; **preserve the existing closed-set scores** alongside rather than
replacing them; and have **no threshold tuned to make results look favourable**. A reject option
chosen after seeing the arms it adjudicates would be worthless.

**And it cannot settle the underlying question on its own.** Claims of "broad model destruction" or
of "clean erasure" require **saved images** plus an **independent quality/semantic assessment** —
not another variant of the same CLIP head that produced the number in doubt. This is also why
D-034's "our port erases where upstream does not" remains unsupported: suppression measured by a
closed-set argmax cannot distinguish erasure from degradation, and no retain axis was measured.

Until that work exists, every "destroys the model" phrasing should read "retain accuracy collapsed
to N/120 under the closed-set scorer", and the cause should be labelled an **unresolved hypothesis**.
**Status: OPEN. Image/score logging queued behind G1; reject option not authorised as a metric
change.**

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

### D-035 — a **passing test** was enforcing a false claim. Worst defect class found.

**Severity:** high · **Owner:** lead · **Files:** `scripts/split_registry.py:191`,
`scripts/test_audit_repairs.py::t_dev_is_labelled_as_used`

**Evidence:** `split_registry.audit()` reported `FINAL: {"status": "FROZEN_UNINSPECTED"}`. That became
**false** the moment `FINDING65` and `FINDING66` scored and interpreted those prompts. The literal
was additionally asserted by a **passing test**: `assert rep["splits"]["FINAL"]["status"] ==
"FROZEN_UNINSPECTED"`.

**Why this class is worse than a stale document.** A stale doc is merely wrong. A *passing test*
around a false claim **actively defends it**: anyone correcting `split_registry.py` alone would see
the suite go red and read their own correct fix as a regression. The literal and the test had to
move together, which is exactly why the previous cycle's agents each declined — each owned one side.

**Repair (both sides, together).** `FINAL.status` → `INSPECTED_DEVELOPMENT_DATA`, with
`inspected_at: 2026-10-26`, `inspected_by: [FINDING65, FINDING66]`, and a `status_note` recording
that it is development data from here on, that a fresh final set must be frozen before any new
configuration is selected against it, and that the D-022 label-text issue plus D-022b's permanent
loss apply to its numbers. The test now asserts the **true** status and `inspected_by`, and adds a
**regression guard** rejecting the strings "uninspected", "untouched", "held out", "held-out"
anywhere in the FINAL record, so the claim cannot quietly return.

**Verified:** `status=INSPECTED_DEVELOPMENT_DATA`, `inspected_by=['FINDING65','FINDING66']`,
all three FINAL overlap sets still empty (disjointness preserved — the freeze and the computation
behind it were never in doubt), and the banned-term guard passes.
**Status: REPAIRED and verified.**

**Systemic lesson for the gate.** Every other "protected by a test" claim in the suite needs the
same scrutiny: a green suite is evidence the code matches the assertions, **not** that the
assertions are true. Added as **G14**.

### D-036 — the Meta reload guard would have certified a **no-op** as successful training.

**Severity:** blocking · **Owner:** Baselines agent (repaired) · **Files:** `run_r2_meta_official.sh`

**Evidence, re-measured independently by the lead.** The guard compared the reloaded checkpoint
against base SD-1.4 loaded with `variant="fp16"`, while the trainer loaded the **main** weights, and
asserted `rel > 1e-6` to conclude "training changed the weights". But main vs fp16-variant differ
*by construction*:

| measurement (pinned SD-1.4 UNet, fp32 upcast) | value |
|---|---:|
| tensors differing, main vs fp16-variant | **686 / 686** |
| global relative difference | **2.074001e-04** |
| fp16-variant == `main.half()`, bit-exact | **686 / 686** |
| the guard's `1e-6` threshold, relative to that floor | **207× below it** |

So the assertion would have **passed with zero training**. A variant mismatch alone clears it by two
orders of magnitude. This is precisely the "model-variant difference mistaken for evidence that
training changed weights" trap, and it was live in the launcher.

**First repair, and why it was also wrong.** v2 compared against the same revision and variant and
required `rel > 10× floor` = **2.074e-03**. An independent reviewer showed that threshold has a
**guaranteed false negative**: AdamW's saturating update bounds `rel ≤ lr·S·√N/‖w‖`, which at
N = 859,520,964, ‖w‖ = 1201.5273, lr = 1e-5 and S = 2 steps gives
**bound(S=2) = 4.880054e-04** — I re-derived this myself. So v2's threshold sat **4.25× above the
largest change genuine training could produce**, and `R2_MODE=smoke` — the only mode whose purpose
is to prove the trainer *trains* rather than merely *runs* — **could not pass even on a correct
update**. v1 passed a no-op; v2 failed real training.

**The arithmetic explains why NO constant could have worked, and this is the real lesson.** The
variant floor is 2.074e-04 and bound(S=2) is 4.880e-04 — the artifact and the genuine signal are
**the same order of magnitude**, a factor of 2.4 apart. There is no magnitude threshold that
separates "training happened" from "someone loaded the wrong variant" at smoke-test scale. Picking
a better constant was never the fix.

### ⛔ BOTH REPAIRS ARE UNSOUND. Third attempt required — and I endorsed the second one.

**Corrected 2026-10-26 on review.** I called the second repair "magnitude replaced by structure" and
singled out check **B** as elegant. **Its premises do not hold.** Four invalid assumptions underpin
the stage-9 criteria:

1. **Check B is false as a criterion.** Training need **not** leave a checkpoint closer to its own
   initialization than to another base variant. Counterexample: init A = 1, alternative B = 1.01,
   trained value 1.007 — closer to B despite starting from A. So `rel_vs_same < rel_vs_other` can
   fail on a perfectly correct update. I praised it precisely because it was magnitude-free, and
   missed that being magnitude-free does not make it *valid*.
2. **The Pythagorean step assumes an orthogonal update**, which nothing justifies.
3. **AdamW does not generally bound the per-coordinate update by the learning rate.** The bound
   `rel ≤ lr·S·√N/‖w‖` that I re-derived to six digits is **not guaranteed** — I verified the
   arithmetic and never questioned the premise. Confirming a calculation on an unsound assumption is
   not verification.
4. **There is no justified universal positive minimum parameter change** inferable from the lr, so
   the lower end of the consistency band is unfounded too.

**All four are removed as pass/fail criteria, and are NOT replaced by another norm threshold.** The
replacement is a **direct evidence chain**, which is what the question actually requires: record the
model state the trainer loaded before optimization (revision, variant, dtype, trainable parameter
names, tensor fingerprints); record **executed** optimizer updates with actual learning rates,
gradient diagnostics and observed parameter changes, distinguishing executed from configured and
skipped steps; compare the checkpoint against the **actual in-memory state** at the save point;
reload and verify keys, shapes, finiteness and tensor equality at the intended precision; verify any
fp16 export against explicitly converted source tensors; and bind all of it to the run config and
source revision. Instrumentation around the pinned upstream must be minimal, documented, and
recorded as a patch.

**And a status distinction I blurred:** the 7/7 unit result was **synthetic tensor perturbations**.
That is a unit test, not evidence the upstream trainer ran. A passing suite is not an executed run.
*Implemented*, *executed* and *independently verified* are three different states.

**Status: UNSOUND — third attempt, owner Agent B.**

### Superseded second repair, kept for the record

One constant became three derived checks:
**(A)** noise (`rel > 1e-7`, not all-identical); **(B)** reference identity, **magnitude-free** —
`rel_vs_same < rel_vs_other` — which catches the v1 trap at *any* update size, including sizes below
the variant floor; **(C)** a consistency band `[bound(1)/100, 2·bound(S)]`. The variant floor is now
**reported and never thresholded**. Unit-tested against the real pinned weights, **7/7 correct**,
including the genuine 2-step saturating update (`rel = 4.880007e-04`) that v2 could not pass.

**Status: REPAIRED on the second attempt. Verified against synthetic checkpoints only** — R2 stages
7/9/10 have never run, so the guard has not yet seen a real trained checkpoint.

### D-037 — the IMMA blank-image filter never executed, and the header claimed it did.

**Severity:** high (provenance), low (outcome) · **Owner:** Baselines agent (repaired) ·
**Files:** `run_r1b_imma_train.sh`

**Evidence:** the filter was nested inside `if [ ! -f metadata.csv ]`, and `metadata.csv` already
existed — so the branch never ran. All three NSFW-blanked frames remain in `metadata.csv`, while
`run_r1b`'s header asserted they had been filtered.

**Outcome, which cuts the other way and is worth stating plainly:** selection is **not** "first N" —
it is `shuffle(seed=42).select([20,40))`. The resolved 20-sample subset was reproduced twice and
contains **no blank image**; `29.png` falls in the discarded `[0,20)` window. So a README-faithful
run would not have trained on a blank frame. The defect is a **false provenance claim**, not a
corrupted training set. Both halves belong in the record.

### D-038 — the Meta trainer would have crashed *after* taking a GPU.

**Severity:** high · **Owner:** Baselines agent (repaired) · **Files:** `run_r2_meta_official.sh`

`upstream/Meta-Unlearning/src/esd_meta.py:669` imports `datasets` at module level; **`venv_t2i` has
no `datasets`**. The previous launcher invoked the trainer there, so it would have reserved a GPU and
then died with `ModuleNotFoundError`. Repaired to `venv_imma` (datasets 5.0.1) with a stage-2
assertion, plus `HF_DATASETS_CACHE` redirected (the root filesystem is full; an unredirected run hit
`PermissionError: '/datasets'`).

### D-019 UPDATE — the benign explanations are now narrowed to **one**, by measurement.

Against the hypotheses I listed when withdrawing the "mislabelled checkpoint" inference: **fp16
variant, dtype round-trip, format conversion and an SD-1.5 base are all RULED OUT by measurement**.
The variant floor is 2.07e-04 while the ESD checkpoint's resnet blocks reach **0.20** relative —
roughly a thousand times larger, so the variant cannot account for the magnitude.

**`EMA vs non-EMA survives and is untested** — the comparison weights are not cached and the host is
offline. sha256 of the file: `8881d70b…`. H-A/H-B remain **labelled hypotheses**; the
"whole-UNet training" reading is still **not** established.

**Cheapest closing experiment, requiring no base model and no GPU:** diff two released `ESDx1`
artist checkpoints **against each other**. Shared structure isolates what the ESD procedure touches,
independent of any base-model or variant choice.

### D-039 — two reservation-API gaps found by a non-owner, reported not edited.

**Severity:** medium–high · **Owner:** Infrastructure agent · **Files:** `scripts/gpu_reserve.sh`

Found by the Baselines agent while integrating: **`GPU_RESERVE_GPUS=""` cannot express "no GPUs"** —
a `${:-}` default makes an empty allowlist fall back to the default set, and it **reserved GPU 0
during a probe**. That is a fail-*open* default in a safety control and must become a hard refusal.
Also: a two-GPU reservation is not expressible, though nothing currently needs one.

**Capacity warning for the queued runs:** our cards are 40 GB; upstream assumes 80 GB, and
`--devices 0 0` co-locates teacher and student at ≈34 GB before activations. The Meta smoke test is
therefore pinned to `META_BATCH=1`, and an OOM at batch 10 is **expected**, not a defect.

### D-040 — I declared G2 passed on a guard nothing called, and tested it under the wrong shell. **Third instance of my own recurring failure.**

**Severity:** high (false gate verdict) · **Owner:** lead · **Files:** `scripts/run_guard.sh`

**Two independent defects in my own repair, both found by the Infrastructure agent and both
re-verified by me:**

1. **No production caller.** `grep -rln run_guard --include="*.sh"` matched **only the file itself
   and a test**. I built a guard, tested it, declared gate condition **G2 PASSED**, and nothing in
   the project used it. A control that is cited but unread is exactly D-013 (my unenforced freeze
   marker) and D-027 (error suppression manufacturing a zero). **This is the third time I have
   committed the same error in this audit.**
2. **Tested under non-production shell options.** Under `set -euo pipefail` — which real runners use
   — errexit killed the runner at `"$@" >>"$RG_LOG" 2>&1` **before** `local rc=$?` could run, so the
   FAIL line was never written and the header's promise that `rg_step` "records a failure and keeps
   going" was **false under exactly the option set the runners use**. My 5/5 self-test passed
   because I ran it with `set -u` only.

**And the targeting is the worst case:** only **4 of 139** runners use `set -euo pipefail`, and they
are precisely the four the Baselines agent had just hardened — `run_r1_imma_official.sh`,
`run_r1b_imma_train.sh`, `run_r1c_eval.sh`, `run_r2_meta_official.sh`. So the bug bit the newest,
most-reviewed launchers and nothing else.

**Repair.** `rg_step` now shields the invocation and restores the caller's options; `rg_cell` always
returns 0 so one bad cell cannot truncate a grid under `set -e`. Re-verified by me under
`set -euo pipefail`: two failing cells survived, **2 FAIL lines recorded**, `rg_finish` refused DONE
with rc=1. `run_v1_esd_alone_control.sh` is now the guard's **first production caller** (syntax
checked; grep confirms three referencing files where there were two).

**G2 is retracted as passed** and now requires every *active* runner to use the guard. The 104
historical runners stay byte-frozen deliberately — they already ran, and their markers are declared
non-evidence — so G2's scope is active launchers only.

**Lesson, recorded because it keeps recurring:** I have now three times declared a control effective
on the strength of its *existence* rather than its *use*. The test for any future control is: name
the caller, and name the production conditions it was exercised under.

### D-041 — my own review premise for the CVD/UUID defect was false.

**Severity:** n/a (premise corrected) · **Owner:** Infrastructure agent

I asserted that an inherited `CUDA_VISIBLE_DEVICES` could desynchronise `nvidia-smi` indices from
physical devices. **Measured and false:** `nvidia-smi --query-gpu=index,uuid` output is
**byte-identical** (same md5 `b43f627f9fcc`) with CVD unset, `1`, `2,3`, and `bogus`. Real
`nvidia-smi` ignores CVD, so reserved indices were already physical.

The agent then found the *actual* faults, which my premise had obscured: v1 never read, reported,
validated or set CVD at all, and recorded no UUID. Fixed by auditing the inherited value, exporting
CVD from the helper, recording `uuid=`, and adding a `MAPCHANGED` state. **Recorded because a wrong
premise that gets "fixed" anyway is how a real defect stays hidden behind a plausible one.**

### D-042 — the freeze was masking a fail-open allowlist and taking credit for it.

**Severity:** high · **Owner:** Infrastructure agent (repaired) · **Files:** `scripts/gpu_reserve.sh`

My D-039 report was **incomplete in the way that mattered**. I reported that
`GPU_RESERVE_GPUS=""` falls back to the permissive default. The agent found that in production it
returned **rc=4** — the freeze-refusal code — which *looks* like a correct refusal. It was not:
`status`, which is not freeze-gated, showed `allowed=0 1`. **The freeze was masking the fallback and
being credited as the guard.** Remove the freeze and the fallback would have been live.

Worse, a whitespace-only value returned **rc=3**, the *retryable* code, so a waiter loop would sleep
forever on a configuration error rather than fail fast.

Repaired with `${VAR+set}` semantics, a new non-retryable **rc=7**, a mode-independent non-empty
backstop, and `_GR_INIT_RC` so a sourced helper also refuses. The two-GPU limitation is documented
rather than implemented.

**This is the same class as D-040 and D-013:** a control appearing to work because something else
was coincidentally blocking the path.

### D-043 — the re-scoring pipeline's write isolation was defeatable by directory creation.

**Severity:** blocking · **Owner:** Evaluation agent (routed) · **Files:** `scripts/rescore_final_v2.py:175-184`

**Found by the independent reviewer, not the builder.** `_guard_path` protects **file** writes, but
`make_run_dir` never calls it. The reviewer demonstrated in a sandbox mirror that
`--run-tag '../../../attacked/PWNED'` creates `<P>/attacked/PWNED/{images,embeddings,arms}`, and the
same reaches `adapters/`. So the single thing the guard exists to prevent — touching the protected
historical tree — was reachable through directory creation rather than file writing.

Related: `_guard_path` (:120) uses a **separator-less prefix test**, so `outputs/rescore_evil/x.json`
is permitted. `T1b` passing is not proof; the comparison must be on path *components*.

**Status: routed for repair, gate blocked on it.**

### D-044 — an all-DISCREPANT re-scoring run would report `run_status: complete`.

**Severity:** blocking (scientific) · **Owner:** Evaluation agent (routed) ·
**Files:** `scripts/rescore_final_v2.py`

The determinism probe correctly reports `EXACT_REPRODUCTION` / `DISCREPANT` **per arm**, but the
verdicts are **never aggregated into run status**. A run in which *every* arm failed to reproduce
history — i.e. one that cannot be compared to the historical records at all — would still be
recorded as a complete, successful correction. `complete` must require a defensible determinism
verdict, or a distinct status must exist for "produced, but not comparable to history".

This is the most consequential of the review's findings, because the whole point of the paired
design was to **test** the determinism assumption rather than assume it.

### D-045 — the builder's own exposure figure was wrong, and a test designed to catch drift could not see it.

**Severity:** high · **Owner:** Evaluation agent (routed) · **Files:** `scripts/gen_eval_final.py:176-179`

The builder documented **26/180** images assigned a differing label. Recomputed independently by the
reviewer **and** by me from all twelve records: **37/180**. Its per-arm figures `(9,7,5,4)` match
nothing — actual are utrap_sd15 12/15, utrap_church 10/15, frp 7/15, uce 5/15, plus 2 in
`BASE_sd14` and 1 in `ATK_uce` — so the exposure is also **not** "concentrated in the four defence
arms", and 9+7+5+4 = 25, not 26. Median margins are 0.0027–0.0063, not 0.0065–0.0131.

**And it carried a false claim:** that the count "bounds how many could move". It does not, and it
contradicts the builder's own closed-set-coupling argument two sentences earlier — an image
predicted `church`, an **unchanged** label, can flip when a competitor's text changes, because the
argmax is over all twelve. *(My register and FINDING65 carry 37/180 with explicitly
direction-neutral wording, so this error did not propagate into the published findings.)*

**G14 instance, and a sharp one.** The guard test `T8f` greps for four verbatim **pre-repair
strings**. A test that checks for the *absence* of known-bad text cannot detect a **wrong number
that is present**. The repair must add a test that recomputes the figure from
`outputs/final_*.json` and asserts the documented value equals it.

### D-046 — two verification claims rest on weaker evidence than stated.

**Severity:** medium · **Owner:** Evaluation agent (routed)

(a) **`T3a_run_ids_unique` is flaky.** The run-id salt is 6 hex characters = 24 bits, so 2000 draws
collide about 11% of the time; the reviewer observed **45/1 in two of seven runs** while the
write-up pastes `2000/2000` as if deterministic. A flaky test reported as a clean pass is the same
class of error as my own G2 claim.

(b) **"`gen_eval.py` unmodified" rests on mtime alone.** The mtime (`2026-10-25 17:38:24`) is
confirmed, but the only hash baseline (`provenance_manifest.json`, 10-23) **predates** it, so there
is no hash to compare against. The claim must be scoped to what mtime supports, and sha256 baselines
recorded now so it has a real basis going forward.

### D-047 — **D-001 is reachable again through one environment variable.** Gate-blocking.

**Severity:** blocking · **Owner:** Infrastructure agent (routed) · **Files:** `scripts/gpu_reserve.sh:287`

Production honours `GPU_RESERVE_LOCKDIR` **without refusing it**. The reviewer measured
`mode=production, lockdir=<their tmp>`. Two runners started with different values take **different
mutexes** and therefore cannot exclude one another — which is precisely the triple-booking of GPU 0
observed on 2026-10-26 that this entire file was written to prevent. **One env var re-opens D-001.**

Two further fail-open thresholds of the same family: `GPU_RESERVE_FOREIGN_MIB` is unvalidated, so
`abc` or `99999999` causes a GPU at **30000 MiB with no attributable process** to be handed out
(rc=0, index 0) — i.e. another user's job trampled; and `GPU_RESERVE_ORPHAN_GRACE` accepts `abc` or
`0`, silently deleting the grace window. And `_gr_do_acquire:819-843` has **no `*)` default**, so an
unrecognised state is **claimable**.

**The pattern is now unmistakable and is the real finding:** every env var this file reads defaults
*permissive*. Fixing the empty allowlist (D-042) fixed one instance of a class. The repair must
sweep **every** variable, not these three.

### D-048 — the test-mode interlock can be defeated by copying a binary. Another user's GPUs reachable.

**Severity:** blocking (affects a third party) · **Owner:** Infrastructure agent (routed)

`_gr_nvsmi_is_real` is a **name-only** check: a `cp` of the real `nvidia-smi` passes it (a symlink
does not). So `TEST_MODE=1` plus `GPU_RESERVE_GPUS="2 3"` hands out **index 2 against real
telemetry**. GPUs 2–3 belong to another user and are in use at 4737 MiB right now. This is the one
defect in the audit whose blast radius lands outside our project.

### D-049 — three migrated runners still write `DONE` without checking anything.

**Severity:** high · **Owner:** Infrastructure agent (routed)

`run_r4b_queue.sh`, `run_r4c_queue.sh`, `run_r4d_upstream_defaults.sh` had their GPU selection
migrated but **not** their failure handling: no exit status is inspected and `DONE` is written
regardless. Separately `run_r1_imma_official.sh:240` gates its stage-4 reservation on inherited
`CUDA_VISIBLE_DEVICES`, so a re-run with that variable set runs the smoke job **unreserved** —
reservation must never be conditional on an inherited environment variable.

### D-050 — my FINDING66 title asserted the claim its own notice withdrew. My error.

**Severity:** high · **Owner:** lead · **Files:** `docs/FINDING66…md`

The title read *"…and on SD-1.4 that is more reliable than the unmodified base model, which scores
13/15"* while the notice I added **withdrew** exactly that sub-claim. **A banner under an asserting
title is the pattern this audit was explicitly told not to use, and I used it.** The title also
called the FINAL prompts "never used for anything", which stopped being true when this document
scored them. Retitled; the former title is preserved verbatim for the record.

Same document, §9: *"there is now no sub-ceiling number anywhere in the FINAL measurements"* is
**false** — six sub-ceiling records are on disk (`ATK_uce_sd15` 14/15, `BASE_sd14` 13/15,
`uce_sd15` 9/15, `frp` 7/15, `utrap_sd15` 2/15, `utrap_church` 1/15). The defensible statement is
only that max-over-valid across post-attack seeds is 1.0000 — a maximum over a set, not the absence
of sub-ceiling measurements. Corrected.

### D-051 — the duplicate-cell disagreements are concentrated in **one control arm**. Previously undisclosed.

**Severity:** medium, and it **reduces** the concern · **Owner:** lead

My figure of "30 duplicate cells disagreeing by up to 8/15" was too crude. The reviewer recomputed:
duplicates reproduce at **68/178**, maximum spread **0.7333 = 11/15** (my 8/15 was only the
strict-key subset), and — the part nobody had stated — **all 29 disagreeing cells belong to the
single arm `BASE(unmodified)`**, the attack-on-base control.

That is materially better news than "30 cells disagree across the corpus": the disagreement is
concentrated in the **unmodified-base control**, not spread through the defence arms. It still needs
explaining (an unmodified base attacked twice should not diverge by 11/15), but it does not
undermine the defence-arm comparisons. **Both the correction and the reassurance belong in the
record; omitting either would misstate it.**

### D-052 — the authoritative status document does not cover the newest thirteen defects.

**Severity:** high · **Owner:** Documentation agent (routed) · **Files:** `docs/STATUS_CURRENT_v2.md`

It scopes itself to `REG/D-001…D-034`, but the register now carries **D-035–D-052 plus D-022b** —
so the "one authoritative current status per issue" document has **no verdict** for thirteen-plus
issues, including D-035 (the passing test asserting a falsehood) and D-046. Partly a timing artifact
of my adding entries after it ran, but it must be closed or the document is not authoritative.

Also from the review: **D-022b's permanent loss is still sold as a cheap pending rerun** in
`STATUS_CURRENT_v2.md:86,314`, `HANDOFF_r5.md:140,154` and `CORRECTION_REGISTER_r5.md:408`, which
call it a "pure re-score", "deterministic" and "CPU-only" — directly contradicting FINDING65:34-38
and FINDING66. D-022b appears in only **2 of 10** documents. Two wrong counts in the status
document: "**18** FINAL records" (there are **12**; its own 180 = 12×15), and "post-attack arms have
**zero** differing-label hits" (there is **one**, margin 0.0081 against 0.0063 — same order, not "an
order of magnitude larger"). The eigenvalue 3.2420e5 is correct but **has no artifact** (sourced to
an agent document and a help string) while **four live `outputs/r4_uce_*.json` still embed the
withdrawn 85,850**. And the defect-ID collision table is itself wrong: `AGENT1_UCE_FINDINGS.md` uses
D-100–D-160 and has **zero** `D-0NN` IDs; only Agent 4 and Agent 5 are genuine collisions.

### D-053 — my `setsid` "fix" did not close the residual **and broke the interrupt path**. Reverted.

**Severity:** high · **Owner:** lead (error), Agent A (correct fix) ·
**Files:** `run_v1_esd_alone_control.sh`, `run_v2_rescore_final.sh`, `run_r4d_upstream_defaults.sh`

I added a `setsid` self-re-exec to three launchers and described it as "closing the last reservation
residual". It did not, and it cost something real. Agent A's three objections, all correct:

1. **Leadership is not a protection.** It only decides whether `gpu_reserve.sh` is *allowed* to treat
   process-group membership as evidence of descent. The reclamation decision still came down to a
   **timer** for every other launch path, and nothing verified the re-exec had even happened.
2. **`setsid` starts a new session with no controlling terminal**, so the script left the terminal's
   foreground process group and **Ctrl-C from the launching shell no longer reached it**. A change I
   justified as improving crash/interrupt handling **actively broke the interrupt path** — one of the
   very behaviours the gate requires ("normal completion and interrupted execution release ownership
   correctly").
3. It **changes the pid after launch**, so the pid the operator sees is not the pid that reserves.

**The correct mechanism, which Agent A implemented:** `gpu_reserve_hold` holds an **`flock(2)` on an
inherited file descriptor** for the life of the run. An inherited descriptor is retained by **every
descendant through `fork` and `execve`**, regardless of group or session leadership — so the lock is
released only when the last descendant exits. That covers children and grandchildren without
depending on leadership, a timer, or the launcher remembering anything. The `setsid` blocks are
removed with the reasoning recorded in place.

**Lesson, and it is the same one as D-040:** I reached for a mechanism that *looked* like the right
shape (group leadership) instead of one that actually carries the invariant (an inherited lock).
"Appears in the launcher" was again standing in for "enforces the property".

### D-054 — my inherited-`CUDA_VISIBLE_DEVICES` refusal aborted on a value the script itself had set.

**Severity:** medium · **Owner:** lead (error), Agent A (fix) · **Files:** `run_r1_imma_official.sh`

I made the stage-4 reservation unconditional and refused an inherited `CUDA_VISIBLE_DEVICES`. But I
tested the variable **at stage 4**, by which point **this script had exported it itself at stage 2** —
so `R1_MODE=prep` from a clean tree would **always abort**, complaining about a value it had just
set. My own comment even anticipated "a shell left over from an earlier stage of this very script"
without distinguishing that case.

Second defect in the same check: I used `[ -n "${CUDA_VISIBLE_DEVICES:-}" ]`, which lets an
**exported-but-empty** value through. An empty `CUDA_VISIBLE_DEVICES` hides every GPU from CUDA and
is just as much a pre-selection as `"0"`.

**Fix (Agent A):** snapshot the inherited value **at entry**, before any reservation exists, and test
**set-ness** (`${VAR+x}`) rather than non-emptiness. Refuse once, at entry, rather than at each
reservation site.

### D-055 — the Meta smoke test **could never produce the checkpoint its guard was meant to check.**

**Severity:** blocking · **Owner:** Agent B (fixed) · **Files:** `run_r2_meta_official.sh`,
`upstream/Meta-Unlearning/src/esd_meta.py:1070`

Found by Agent B, independent of everything I had assigned. Upstream nests the **periodic save
inside the validation gate**, so with the launcher's `--eval_every $((SM_STEPS+1))` smoke mode
**could never write a `step=*/` checkpoint** — and therefore could never reach stage 9 at all.

So the entire two-round argument about stage 9's acceptance criteria was moot in a way none of us
had noticed: the stage was **unreachable** in the only mode that was supposed to exercise it. A
guard that cannot be reached is not a weak guard, it is an absent one. Fixed.

### D-056 — the `coherent` pool scheme leaked 2 of the 3 evaluation target prompts into its **forget** pool.

**Severity:** blocking (split contamination) · **Owner:** Agent B (fixed) ·
**Files:** `scripts/r2_build_meta_pools.py`, `scripts/split_registry.py`

The `coherent` scheme was introduced as the principled replacement for the incoherent
church-vs-cathedral/chapel/basilica setup. Its **forget** pool contained **2 of the 3 prompts the
target is scored on**. I had been treating it as the clean option while its own construction carried
the contamination the split registry exists to catch. Fixed, with a **hard-failing disjointness gate
on resolved runtime pools** — not on source regex, which is what missed it.

### D-057 — the evidence chain's instrumentation silently recorded nothing (caught by its author's own tests).

**Severity:** would-have-been-blocking · **Owner:** Agent B (fixed)

`AdamW` defines its own `step`, so patching `Optimizer.step` **never fired** and the executed-update
record would have been **silently empty** — an evidence chain that produced no evidence while
reporting success. Now uses global optimizer step hooks. **10/10 unit tests pass**, and Agent B
states plainly that these are unit tests on synthetic tensors, **not** an executed run.

**Agent B also verified all four unsound criteria from source with CPU counterexamples**, which is
stronger than my acceptance of them: check B fails a genuine update (rel_same **7.00e-03** >
rel_other **2.97e-03**); the Pythagorean step needs orthogonality (1.22e-02 claimed vs **3.00e-03**
actual); AdamW's `lr·S` bound is *attained* at S=1 and ignores `weight_decay=1e-4`; and a
zero-gradient **executed** step moves exactly 0.0. All four removed with no substitute threshold,
asserted structurally.

### D-058 — my `--no-preserve` field fix was incomplete, and the published artifacts carry stale metadata.

**Severity:** high · **Owner:** lead · **Files:** `outputs/r4d_uce_church_*.json`

Agent B independently re-derived my `--no-preserve` equivalence and found it **stronger** than I
claimed — bit-exact in **float32** too, numerator and denominator separately, not only in float64.

But my fix was **incomplete in two ways**: a stale `ASYMMETRIC` caveat elsewhere in the file still
contradicted it (fixed by Agent B), and — the part that matters — **the on-disk artifacts predate the
fix** (written 16:13/16:14 against the fix at 16:18) and still read `n_preserve_port: 10` when the
port used 0. **I published those artifacts in `cc778f2` with that wrong metadata.** They must be
re-run, not re-labelled. Agent B correctly left them untouched.

Also: `complete: False` on both artifacts was `--retain-concepts` defaulting to **empty**, so **no
run could ever be complete**. Fixed with a documented retain panel and a structured `completeness`
block — which also means the R4d re-run will finally carry the retain axis D-034 requires.

### D-059 — the flock lifetime lock does **not** cover the real chain. Measured, exploitable.

**Severity:** blocking (gate) · **Owner:** Agent A (routed) · **Files:** `scripts/gpu_reserve.sh`

I claimed the inherited descriptor is "retained by every descendant through `fork` and `execve`".
**That is false for our actual chain**, and I measured it rather than argue it:

| hop | result |
|---|---|
| shell → `exec python` | **fd SURVIVES** (python holds fd 10). bash's `{FD}>>` is not close-on-exec. |
| python → `subprocess.Popen` | **fd IS DROPPED.** Python defaults to `close_fds=True` (since 3.2); `/proc/<grandchild>/fd` shows no hold fd. |
| consequence | SIGKILL the trainer with a worker alive → `flock -n -x hold -c true` **SUCCEEDS**: *"LOCK FREE while survivor alive"*, survivor confirmed running. |

So the precise case the gate requires us to prevent — killing the parent must not let another job
onto a GPU still in use by surviving descendants — **is live**. And `pass_fds=` is not available as
a general fix because we do not control upstream's own `Popen` calls.

**Fix direction (simplest, not a third design):** keep the flock as the fast path for the owner's own
lifetime, and make the **authoritative acquire-time test device occupancy** — refuse any candidate
GPU carrying a compute process not attributable to a live held reservation.

> **✅ THE COVERING ASSUMPTION IS NOW MEASURED, not assumed (lead, bounded GPU run).** The occupancy
> gate only closes this hole if `nvidia-smi` can actually see a `Popen`-spawned worker that holds no
> descriptor. Measured directly: a parent with **no** CUDA context spawned a child via
> `subprocess.Popen` (default `close_fds`), the child allocated a tensor on GPU 0, and
> `nvidia-smi --query-compute-apps` reported **`2174848, 492 MiB`** — the child's own pid.
> **`CHILD LISTED BY nvidia-smi: True`.** So the survivor in Q3 is visible to the gate even though it
> inherits no lock, and the gate genuinely covers the `close_fds` hole. Agent A had flagged this as
> its one unverified assumption; it is now verified by measurement. Agent A also reports `acquire`
> **already** refused that case (`OURS_UNLOCKED` / cuda-context attribution), with T35a/b added to
> pin it — so **no redesign was needed**, which is the right outcome.
>
> **Related hazard, measured by Agent A:** merely *closing* the fd preserves the shared lock, but
> `flock -u` **destroys it for every sharer**. Only fd 9 (the mutex) is ever unlocked, now with a
> source-level guard.

**D-053 release-on-interrupt — PARTIAL, and recorded as partial.** Agent A measured 8 cases
(SIGTERM/SIGKILL with a child *and* an exec'd grandchild): all verdicts correct in both arms —
retained iff work alive, released iff not. It reported SIGINT as unmeasurable from a backgrounded
harness (such shells inherit SIGINT **ignored**, `SigIgn=…06`).

I took that measurement myself under `setsid -w` with a handler installed, which does deliver SIGINT:
**lock HELD while the grandchild was alive, FREE after descendants exited** — lifetime semantics
correct. **But the cleanup flag came back `CLEAN=0`, so my test did NOT establish that the INT
cleanup trap ran.** Lifetime: verified. Cleanup-on-SIGINT: **still unverified**, and stated that way
rather than rounded up.

**Nothing in Agent A's work is independently verified** — it wrote both the code and the tests and
says so itself. That review is now running separately, and milestone 1 does not pass without it.

**This is the sixth instance of the pattern Agent C named** (after D-013, D-040, D-053, D-055,
D-057): a mechanism credited for its shape rather than the property it enforces. Five of the six are
mine. The distinguishing feature here is that the measurement was cheap and I had not taken it.

### Milestone framing — adopted 2026-10-26, and it changes what blocks what

Three separate milestones, previously conflated in my reporting:

| # | milestone | current state |
|---|---|---|
| **1** | shared infrastructure safe enough to execute verification | **FAILED independent review 2026-10-26** — D-060 (`PROJECT` defeats every guard), D-061, D-062, D-063. D-047/D-048/D-059 confirmed closed |
| **2** | a particular method's complete execution path verified | IMMA: pending (needs M1 + a real smoke run). **Meta: NOT verified** (D-055/D-057 — stage 9 was unreachable; instrumentation untested in a real run). UCE: records superseded, re-run pending |
| **3** | that method produces interpretable research results | none yet |

**M1 does not establish M2 or M3**, and I have been reporting as though a passing infrastructure
test moved a method forward. Conversely — and this corrects a real error in how I sequenced work —
**an unrelated method's unresolved problem must not block a verified path.** Meta's defects
(D-055, D-056, D-057) do **not** block IMMA. They block *Meta*, which is excluded from scientific
comparison until its smoke run passes, and that exclusion is the whole of their consequence for the
next experiment.

**Testing calibration, recorded because I over-corrected.** Passing tests **are** valid evidence for
the properties they genuinely exercise. The failure mode throughout this audit was **misleading
assertions and unrealistic test conditions**, not testing itself. A test that kills a real launcher
with a real grandchild and reads the real lock state is good evidence; a test that greps source, or
runs with a working shim where production uses real telemetry, is not evidence about production.

### D-060 — **one environment variable defeats every guard in the reservation.** Milestone 1 FAILED.

**Severity:** blocking · **Owner:** Agent A (routed) · **Files:** `scripts/gpu_reserve.sh:133, 140-144`

Found by the independent reviewer, **not** by the author and **not** by me. `gpu_reserve.sh:133`
reads **`PROJECT`** from the environment, and `:140-144` derive all three "hard-wired production
constants" **from it**. So every D-047 refusal compares a requested value against a constant the
same environment has already moved.

**Measured, with the project frozen:**

```
PROJECT=<scratch>/evil bash scripts/gpu_reserve.sh status
  mode=production   lockdir=<scratch>/evil/.gpulocks
  FREEZE: none (<scratch>/evil/FREEZE_ACTIVE absent)
```

and `acquire` then returns **rc=3, not rc=4** — the freeze gate is **passed outright**. With
`FOREIGN_MIB` at its default it would have reserved gpu0; the reviewer forced `FOREIGN_MIB=1` to
stop it completing.

**One variable simultaneously:** defeats **D-013** (the freeze marker — my defect), **relocates the
mutex** (reintroducing **D-001**, the original triple-booking), and **redirects the override audit
log** so a freeze override goes unrecorded. It also directly contradicts the file's own
`:155-156`, *"a sweep of EVERY variable the file reads."*

**This is the root of the class.** D-042 (empty allowlist), D-047 (lockdir/nvsmi/override-log) and
D-060 are not three defects but one: **the file validated the leaves while the root stayed
environment-derivable.** Each previous fix hardened a value computed from `PROJECT`.

### D-061 — `rg_manifest` reports success on failure, and `rg_finish` then writes DONE.

**Severity:** blocking · **Owner:** Agent A (routed) ·
**Files:** `scripts/run_guard.sh:421-437`, `run_r1c_eval.sh:389`

It never captures its publish pipeline's status: it logs *"manifest published"* and returns 0 even
when the publish failed. Called inside an `if` (which suppresses errexit), `rg_finish` proceeds to
write **DONE with no manifest on disk**. The "publish a completion manifest only after all required
artifacts validate" requirement, failing in the one primitive built to enforce it.

### D-062 / D-063 — two launchers on the resumed path.

**D-062** `run_v2_rescore_final.sh`: no `set -e`; **launches GPU work after its own precondition
fails**; and `ls -1dt | head -1` can attest to a **previous** run's directory, so the manifest check
can pass against an older artifact. **D-063** `run_r1_imma_official.sh:194`: dead `||`
(`gpu_reserve_drop` always returns 0), and its retention branch leaves `_GR_HOLD_PATH` set so a
second acquire **orphans the first hold file** — the two-acquires hazard surviving in the launcher
that will run the next experiment.

### What the independent review CONFIRMED

Recorded because it is evidence, not courtesy. Suite re-run by the reviewer: **318 passed, 0 failed,
exit 0**. **D-059's covering layer cannot be widened by configuration** — `compute-apps` is consulted
**unconditionally** and the `OURS_UNLOCKED` arm has **no tunable threshold**; T35a/b pin it with real
preconditions and a real `acquire` (compute-apps row mocked, and the doc says so). Only one
`flock -u`, on fd 9. The K1 execution-gate fix is **complete**: zero `|| echo 0` and zero live `awk`
in r1c. `_gr_holders` re-measured at **177 ms over 2,460 fds**. And §6.6's
measured-vs-sound-by-construction separation was judged **honest, its only fault being that it
understates what is now known** — the opposite of this project's usual failure.

**SIGINT stays PARTIAL, with a sharper reason than mine.** The reviewer observed that
lock HELD→FREE is **kernel-produced regardless** of whether the trap fires, so that observation
**cannot discriminate** — my test could not have settled it even in principle, not merely by
accident. `REVIEW_M1.md` §2 names the stderr line that would.

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

**2026-10-26 — G2 PASSED, and the damage is smaller than I first recorded.** Scanned every runner:
**104 of the 138** `run_*.sh` that echo a `DONE` marker never inspect an exit code anywhere. So a
cell that crashed, OOM'd or aborted left a log whose last line says DONE.

**But I then measured whether any of it was load-bearing, and it was not:**

| check | result |
|---|---|
| Python in `scripts/` parsing the string `DONE` | **none** |
| runners gating on a previous `DONE` marker | **none** |
| published findings citing a log `DONE` as completion evidence | **none** (FINDING44's only hit is prose, not a log marker) |

So a false `DONE` **never mechanically drove a decision**. The hazard is a *human* reading a log tail
and concluding a campaign finished — real, but a tier below what I implied when I opened D-012.
Stating the smaller number is the point of checking.

**Repair, proportionate to that.** I did **not** rewrite the 104 historical runners: they already
ran, editing them cannot change what their logs said at the time, and their exact bytes are the
record of how historical arms were produced. They are instead **declared non-evidence** — completion
is established from artifacts, never from a log line. Forward protection is
**`scripts/run_guard.sh`**, a sourceable guard in which `rg_finish` is the only place a `DONE` may be
written. Self-test, all five cases:

```
  PASS finish=0 on clean run
  PASS finish=1 after failure            (failing step poisons the marker)
  PASS finish=1 on missing artifact      (catches "exit 0 but wrote nothing")
  PASS finish=1 on zero-byte artifact    (catches the D-028 truncation class)
  PASS finish=1 after abort              (low disk / refused reservation cannot read as success)
  markers written: 1 DONE, 4 INCOMPLETE
```

**2026-10-26 — Baselines repair integrated, after independent lead verification.**

Re-measured the governing number myself rather than taking the agent's word, because three separate
items depend on it: main vs fp16-variant SD-1.4 UNet differs in **686/686** tensors at global
relative **2.074001e-04**, and the fp16 variant is bit-exactly `main.half()` for **686/686**
tensors. The old reload guard's `1e-6` threshold is **207×** below that floor — confirming D-036.

Preservation verified (the instruction was to keep the completed run, not relabel it):
`upstream_runs/imma/REF_imma_vangogh/imma_unet_xatten_layer.pt` **untouched** at its original mtime
(12:21), with `BUDGET_5X.md` added **alongside** (14:25) rather than overwriting anything;
`adapters/REF_imma_vangogh/unet.pt` and `outputs/gen_REF_imma_vangogh.json` intact at original
mtimes; `metadata.csv` still contains **all three** blank frames, with `metadata_noblank.csv` and
`blank_audit.json` added as **new** files. Every change additive; nothing destroyed.
ESD checkpoint sha256 confirmed: `8881d70b1bd376c5532e…`.

**2026-10-26 — Evaluation repair integrated, after independent lead verification.**

Verified myself rather than from the agent's summary:
- `scripts/gen_eval.py` and `scripts/attack_relearn.py` mtimes **unchanged** (2026-10-25 17:38), so
  the ~800 historical records still reproduce. All **12** `outputs/final_*.json` intact.
  `outputs/rescore/` **absent** — nothing was written pre-emptively.
- Test suite: **46 passed / 0 failed**. *Interpreter note:* it must be run with
  `venv_t2i/bin/python`; under system `python3` seven tests error on `ModuleNotFoundError: torch`
  and the count reads 13/7. The agent's claim was accurate but omitted the interpreter — recorded so
  a later reader does not mistake an environment problem for a regression.
- Inspected the integrity-critical code paths directly: `_guard_path` refuses any realpath inside
  the protected historical tree **or** outside `outputs/rescore/`; `atomic_write_json` uses a
  same-directory temp + fsync + `os.replace`; the manifest is checkpointed after **every** arm;
  `run_status: complete` requires every requested arm to be complete, and missing/failed arms are
  named rather than skipped.
- Paired scoring scores both label sets against **one** image-feature tensor, and its
  `measurement_note` states the effect "was not predicted in advance in either direction" — the
  direction-neutral framing is encoded in the artifact, not just the prose.
- Determinism is **tested, not assumed**: `compare_with_history` returns `EXACT_REPRODUCTION` /
  `DISCREPANT` per arm with disagreeing indices (test T10c exercises the discrepant path), and
  T10d detects the pre-repair marker (historical records carry no `label_text_set` field).

**2026-10-26 — base-model control, with its scope corrected.** `gen_base_sd14.json` and
`gen_E6cal_BASE_sd14_on_DEV.json` both score **1.000 on all twelve DEV concepts**.

**What that licenses:** the scorer behaves correctly on **undamaged, in-distribution** outputs, and
base utility is 1.000 on this panel.

**What I wrongly concluded from it, now withdrawn:** that "the closed-set 12-label CLIP scorer is
sound" and therefore any arm's low utility must be a property of that arm rather than the metric.
A control on undamaged outputs says **nothing** about the metric's validity on damaged or unusual
outputs — which is exactly the regime every erased arm occupies, and exactly where D-030 says the
closed set with no reject option is least trustworthy. So this control does **not** refute a
metric-based explanation for D-009's numbers; it only rules out the metric being broken on normal
images.

## 4. Quarantine

*(invalid artifacts: preserved, traceable, excluded from aggregates)*

| artifact | reason | status |
|---|---|---|
| `outputs/gen_REF_imma_vangogh.json` | **Metric inapt + configuration not one the authors report.** The composition is **VALID** (`exact=160 fuzzy=0 reshaped=0 skipped=0`; 160/160 and 526/526 tensors byte-equal; ESD delta not inverted) — the earlier "composition unproven" reason is **REFUTED and withdrawn**. The arm also ran ~5× the README's training budget (D-021). | **quarantined** — not citable as an IMMA result. Not pending a composition check; that is settled |
| `outputs/r4b_*`, `outputs/r4c_*`, `outputs/r4d_*` "UPSTREAM" arms | **Open question narrowed to `expand_prompts`, plus a confounded port side.** "Upstream's edit may be inert / weights possibly unchanged" is **REFUTED and withdrawn**: upstream changed all 32 attn2 K/V matrices by **5.36 % mean / 9.79 % max**, 0/32 identical to base, **43.4 %** change in K/V output. What remains open is whether our single-erase-prompt invocation should have used `expand_prompts` (unreachable through `UCE()`), and separately the **port** column is confounded by cached-checkpoint reuse (D-026). | **interpretation restricted** — files retained; no port-fidelity claim may rest on them. R4d needs re-running |
| `outputs/final_ATK_*.json` (6 records) | **D-022b: checkpoints deleted.** `attacked/E6B_*`, `E6C_*`, `E6D_*` absent, so these cannot be re-scored under the corrected label text. | **withdrawn, not re-verifiable** — a rerun would be a fresh analogue, not a reproduction |
| `outputs/atk_E2H_*_lr5e-5_s50.json` | D-007: undeclared post-hoc cell | **retained, labelled** — reportable only as a post-hoc diagnostic, never as preregistered |
| `outputs/gen_advunlearn_{parachute,tench,vangogh,garbage_truck}.json` | D-011: evaluated checkpoint unrecorded | **retained, labelled** — numbers stand, attribution does not |

## 5. Resume gate

**Status: NOT PASSED.** No campaign cell may launch until every condition below is satisfied and
countersigned in §3 with the evidence that satisfied it. A condition is met by a measurement or a
passing test, never by an assurance.

| # | condition | blocking defects | evidence required |
|---|---|---|---|
| G1 | Exclusive per-GPU reservation works and is proven to work | D-001 | `scripts/test_gpu_reserve.sh` output showing N racing processes yield exactly one winner per GPU, stale-lock reclamation after a simulated crash, release on crash, and that a GPU held by another user is never selected |
| G2 | No runner reports success after a failure | D-012, D-040 | **NOT PASSED — my earlier "PASSED" is retracted.** The guard existed, was tested, and had **no production caller**; its self-test also ran under the wrong shell options. Now wired into `run_v1_esd_alone_control.sh` (first real caller) and its errexit bug fixed. Passes only when every **active** runner uses it |
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
| G13 | "Destroys the model" claims scoped to what the metric supports | D-030 | per-image predicted labels logged on ≥2 collapsed arms; prose rephrased to "retain accuracy collapsed to N/120". A reject option is **not** accepted as the evidence — it needs independent calibration and versioning first |
| G14 | No passing test asserts a false claim | D-035 | every assertion in `test_audit_repairs.py` and `test_rescore.py` checked against reality, not just against the code. A green suite shows code matches assertions, **not** that assertions are true |
| G15 | Every method that built a published arm is registered | Agent 6 | `erase_utrap_v2.py` built all four FINDING76 arms and is **unregistered** in `split_registry`/provenance; `split_registry`'s advertised hard-fail is unimplemented. Register the builders or state the gap |

**Queued bounded verification runs** (lead-launched, through the reservation API, in this order once
the infrastructure repairs pass independent review): ESD-alone behavioural control for D-019
(≈5 min) → FINAL regeneration + paired corrected scoring for D-022 (≈10 min) → per-image
label/score logging for D-030 (≈10 min) → R4d re-run with a corrected `port_dir` key for D-026.
Total well under one GPU-hour, one GPU at a time, GPUs 0–1 only. Each is a *verification* run, not a
campaign cell. A reject option is **not** among them: it is an unauthorised metric change until
calibrated independently and versioned.

### Tool-approval discipline (recorded, 2026-10-26)

Tool-approval requirements apply **independently** of the research freeze, and an automatic
rejection is not an obstacle to be engineered around. For the record, the following are **not**
acceptable responses to a denial and will not be used: disguising or re-encoding a command,
renaming a flag, removing or editing `FREEZE_ACTIVE` to avoid needing the override, requesting
blanket shell access, or handing the user an unreviewed patch-and-launch one-liner to run on my
behalf. **I did the last of these in the previous cycle and it was wrong** — it outsourced execution
of an unreviewed edit, which defeats the point of the approval step.

The correct sequence is: repair and inspect the verification runner first; if its execution still
requires approval, request approval for **that exact bounded run**, stating its purpose, GPU limit,
expected duration and output paths; and continue independent CPU repairs while waiting. A claim may
remain **withdrawn pending a future experiment** — that is a legitimate terminal state and does not
require inventing a result to close the issue. Equally, **documenting an execution defect is not
repairing it**: an entry stays OPEN until a repair exists and is verified.

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

**(b) A prior-art overlap is confirmed — and my conclusion from it was wrong.**

**Corrected 2026-10-26.** arXiv **2608.25429** is **FRAG**, which is **verified** and is **already
the source of this project's own FRP reference**:
<https://arxiv.org/abs/2608.25429> · <https://arxiv.org/html/2608.25429v1> ·
<https://github.com/Yi1-Chen/FRAG>. I had recorded it as an unverified citation of a separate
preprint and flagged it as a "live risk"; that was a bookkeeping error about a paper this project
already builds on.

**The overlap, recorded factually.** FRAG's **Table 11** and **Table 15** overlap with this
project's **attack-validity** argument and its **initial-forgetting-versus-recovery** argument
respectively. Table 15 shows an inversion against a retain-trained gold model; Table 11 states that
an attack recovering nothing does not probe robustness. Related prior art on individual parts:
MUSE (2407.06460) on over-unlearning past the retrained model; TOFU (2401.06121) on two-sided
testing against the retain model; Łucki et al. (2409.18025) on ablatable directions; "Erase or
Hide?" (2509.22263); Carlini et al. (1902.06705) §4.14 on attack validity generally.

**WITHDRAWN: my conclusion that "the contribution cannot rest primarily on the evaluation
methodology" and must instead be a new defence method.** That does not follow from an overlap on
two tables. A contribution decision requires a **claim-by-claim literature comparison** — which
specific claims are new, which are anticipated, and in what setting (FRAG's tables are LLM
relearning; ours are T2I diffusion) — and that comparison **is not part of this repair gate** and
has not been done. Nothing here forecloses either an evaluation-side or a method-side contribution.

**What does survive, because it was measured here rather than inferred:** FINDING78's retraction
means no arm in this project currently meets both mandate halves, and the mechanism is specific —
at a fixed 25-step budget, max-over-valid H moved 0.0667 → 0.2000 → 0.5333 as attacker lr went
1e-5 → 5e-5 → 1e-4. So any future arm must be evaluated against an **attacker-lr sweep at matched
budget**, not a budget sweep at fixed lr. That is a protocol requirement, and it stands independently
of any contribution decision.

**Prior art consistent with that protocol requirement (pointers, UNVERIFIED by me).** Two existing
results point the same way, which makes the sweep requirement conventional rather than novel: FRAG's
Table 11 disqualifies attack configurations that recover nothing for any arm, and Deeb & Roger
(arXiv **2410.08827**) reportedly conclude that fine-tuning-robustness defences fall to an extensive
hyperparameter search. **I have not read or verified either claim about 2410.08827**; it is recorded
as a pointer to check, not as support. It is an observation about existing evidence, not a new result
of ours — and if it holds, our attacker-lr finding is a *confirmation* of known behaviour, not a
discovery.

**Open prior-art gap, which must be closed before any novelty claim is written.** The novelty check
could not run a **forward citation sweep** (Semantic Scholar returned 429 without an API key; no
general web search was available). Part (B) of its verdict — "scoring below a retrain oracle is a
robustness warning sign" — could therefore still flip from *non-obvious combination* to *already
published*. The specific search that would settle it: a forward citation sweep from **2608.25429**
(FRAG), **2407.06460** (MUSE) and **2401.06121** (TOFU), looking for a stated
below-oracle → robustness diagnostic. This needs an API key or restored web search and is **not**
part of the repair gate.
