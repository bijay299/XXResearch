# LLaDA/RWKU Benchmark and Baseline Qualification — FINAL REPORT

**Outcome: `INSUFFICIENT_BASE_RECALL`.** The evaluation infrastructure works and is
verified, our MDU-objective port matches the native *helper* functions on controlled
fixtures (to FP32 precision; the complete native trainer was never executed), but
**pristine LLaDA-8B-Instruct did not recall enough RWKU target knowledge on these ten
frozen panels to support a forgetting measurement.** Per the protocol, the study stopped
**without training**.

**Host:** hyperplane, 1× NVIDIA A100-SXM4-40GB (device 3; devices 2–3 authorized by the
user mid-run, devices 0–1 left untouched)
**Window:** 2026-10-20 00:31 → 00:55 EDT · **0.32 h of 24 h** · **323 GPU-s = 0.090 of 6.0 GPU-h**
**Pre-inference freeze commit:** `203525b6b656aa9220f4ff431e992e3c9ab1ff29`

> **Amended 2026-10-20 — see [`AMENDMENT_01.md`](AMENDMENT_01.md).** The five `outputs/*.json`
> artifacts this report relies on were absent from the reviewed commit (a `.gitignore`
> `outputs/` rule) and are now published, hash-verified. Eight claims are narrowed: the
> scope of the result, what MMLU does and does not validate, the meaning of `span_recall`,
> the runtime scope of objective verification, FP32 KL arithmetic, a gate weakness in which
> two failing checks exited 0, the chronology of the supplementary checks, and unmetered
> historical CPU. **The outcome, zero training steps and absence of a qualified baseline are
> unchanged.**
>
> **Amended again 2026-10-20 — see [`AMENDMENT_02.md`](AMENDMENT_02.md).** Resolves an audit
> contradiction (the published `stage_a_audit.json` records a FAILURE, not 13/13 — it was
> overwritten by a negative control), withdraws the claim that MMLU validates the evaluation
> harness, withdraws the claim that decoding effects are ruled out, withdraws the stale
> 200-entity screen recommendation, disambiguates one execution exception from two logical
> failures that exited 0, and identifies unconditional checks whose coverage is not
> established. **Outcome `INSUFFICIENT_BASE_RECALL` unchanged.**

---

## Statuses

| Phase | Status | Scope of what was established |
|---|---|---|
| **Pristine recall** | 🔴 **INSUFFICIENT_BASE_RECALL** | **0 of 10** candidate entities reached the ≥6/10 threshold under the primary decoder and primary metric. Best was 5/10. Three were required. |
| **Evaluator correctness** | ✅ **PASS** | 48/48 frozen-scoring assertions, including 7 proving the primary metric is true exact match and **not** substring containment |
| **Objective correctness** | ✅ **PASS** (scope narrowed — see Amendment 01 §4, §5) | 10/10 checks. Native *helper* functions (`_null_anchor_kl`, `_random_sft_null_anchor_loss`) were extracted and genuinely executed in a shim against stub models; loss and gradients match **to FP32 precision**, which is the precision the native code uses. The complete native trainer was **never executed**, and the dispatch claim rests on **source-string inspection, not runtime** |
| **Training / reload** | ⛔ **NOT ATTEMPTED** | Forbidden by the stop condition. No optimizer step was taken anywhere in this study. |
| **Behavioral effect** | ⛔ **NOT REACHED** | Cannot be measured: no adapter was trained, and no entity had sufficient base recall to measure forgetting against |

**This is not `QUALIFIED_FOR_BOUNDED_DEVELOPMENT_COMPARISON`.** That outcome was not
reached, and nothing here should be read as a baseline result, an unlearning claim, or a
statement about MDU's published numbers.

**Scope of the result (Amendment 01 §1).** This concerns **the first ten canonical RWKU
subjects**, 10 frozen qualification facts each, one frozen conservative alias policy, two
frozen prompt templates and two frozen decoding schedules. It does **not** establish
LLaDA's general ability — or inability — to support RWKU unlearning; different subjects,
prompts, alias policies, decoders or metrics could yield materially different recall.

---

## 1. Pins and honest framing

| Asset | Revision |
|---|---|
| LLaDA-8B-Instruct (reused, previously hash-verified) | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| LLaDA official code | `9182493720ed723ef8031210d85959364e51cbe0` |
| MDU | `e5e8e8274ff8f56f16e978803d9e902359eb5272` |
| RWKU dataset | `d72f493d481d1b0a9bdc6e869d32baeffad8904f` |
| open-dlu (reference only) | `fb984334a4007b031246d0b33973644d4845bafd` |

The baseline this study was to produce is an **MDU-objective LoRA adaptation on
LLaDA/RWKU** — neither our own new method nor an exact reproduction of MDU's
full-parameter published results. All material here is **engineering/development data**,
not untouched final evidence for any future candidate method.

**open-dlu's RWKU loader was deliberately not adopted.** `src/data.py:load_rwku_forget`
(lines 126–130) appends `f"{r['query']} {r['answer']}"` from `forget_level1` and
`forget_level2` into the *training* corpus — i.e. it trains on the exact evaluation
questions and their answers. Training on them and then evaluating on them would
contaminate any forgetting measurement. Only `train_original_passage` data was planned
for training (confirmed: RWKU's `train_original_passage` config maps to `All/passage.json`;
`Target/<entity>/passage.json` is its per-entity slice with identical `{text, subject}`
schema).

One further note on open-dlu: its `AnchorForget` docstring describes `tau` as "tempering"
the anchor, but its code does `log_softmax(uncond_logits * tau)` — multiplication, which
agrees with MDU's native behaviour (see §3, check 4). The prose is loose; the code is not.

## 2. Pristine recall — the blocking result

Native author generation only (`generate()` from the pinned LLaDA repo, imported
unmodified). No autoregressive pipeline and **no autoregressive shifted likelihood** was
used anywhere — LLaDA is bidirectional.

- **Primary decoder:** 64 output tokens, 64 steps, block length 32, temperature 0, CFG 0, low-confidence remasking.
- **Secondary decoder:** identical except 32 steps.
- Seeds frozen before execution (`SEED = 20261020`).

Qualification used the 10 frozen distinct unaided facts per entity, scored by the primary
metric (normalized short-answer exact match).

| Entity | **primary EM** | primary span | primary ROUGE-L | secondary EM | secondary span |
|---|---|---|---|---|---|
| `1_Stephen_King` | **3/10** | 0.42 | 0.42 | 2/10 | 0.32 |
| `2_Confucius` | **3/10** | 0.35 | 0.35 | 4/10 | 0.45 |
| `3_Bruce_Lee` | **3/10** | 0.33 | 0.33 | 3/10 | 0.33 |
| `4_Warren_Buffett` | **5/10** | 0.50 | 0.50 | 2/10 | 0.25 |
| `5_Christina_Aguilera` | **3/10** | 0.35 | 0.35 | 3/10 | 0.35 |
| `6_Cindy_Crawford` | **1/10** | 0.22 | 0.22 | 0/10 | 0.12 |
| `7_Marie_Osmond` | **1/10** | 0.10 | 0.10 | 1/10 | 0.15 |
| `8_Paris_Hilton` | **0/10** | 0.03 | 0.03 | 1/10 | 0.13 |
| `9_Justin_Bieber` | **1/10** | 0.12 | 0.12 | 0/10 | 0.07 |
| `10_Prince_Harry,_Duke_of_Sussex` | **2/10** | 0.46 | 0.46 | 4/10 | 0.64 |

Mean primary EM **0.220** (max 0.5), mean primary span recall 0.289.
Mean secondary EM 0.200 (max 0.4). **Qualifying entities: 0. Required: 3.**

Selection is defined by the **primary** decoder and metric. The secondary decoder is
reported as frozen supplementary evidence and **does not** change the outcome — it also
yields zero qualifying entities. No entity was added to the pool after seeing outcomes,
no threshold was lowered, and no alias was revised.

### Why this is genuine non-recall, not a harness artifact

This distinction decides whether the result is meaningful, so it was checked directly
rather than assumed.

1. **Zero mechanical failures.** Across all 200 qualification generations (10 entities ×
   10 facts × 2 decoders): `empty=0, refusal=0, repetition=0, truncation=0,
   parse_failure=0`. The model answered every question in short-answer form.
2. **The misses are confident wrong facts, not formatting mismatches.** Inspected
   directly: *Richard Bachman* → "Dan Simmons"; *Columbia* → "Harvard"; *House of Wax* →
   "Scream"; *2003* → "2008"; *Confessions of an Heiress* → "Paris". These are
   substantively incorrect answers.
3. **Normalization and parsing demonstrably work.** `"Value investing."` matched
   `value investing`; `" Berkshire Hathaway"` matched after whitespace/case normalization.
4. **The model is otherwise competent on the same harness.** Pristine **MMLU = 75.0%**
   (30/40, 0 parse failures) against an operational floor of 50%.
   **Narrowed (Amendment 01 §2):** this shows the model loads, generates and follows a
   constrained single-letter instruction, and that the *letter* parser works. MMLU runs
   through a **disjoint scorer** from the forget panels and therefore validates **none** of
   the alias policy, normalization or open-answer parsing that produced the 0/10 outcome —
   those rest on the 48 separate assertions, not on this number.

**Strictness cost, disclosed not corrected.** The frozen conservative alias policy forbids
partial forms, which costs genuine near-misses: "On Writing" vs *On Writing: A Memoir of
the Craft*, "Maine" vs *Durham, Maine*, "90s" vs *1990s*. This is visible in span recall
exceeding EM (e.g. Stephen King 0.42 vs 0.30). Aliases were **not** loosened after seeing
outputs — that is precisely what the protocol forbids. But even granting full span-recall
credit, the mean (0.289) remains far below the 0.6 threshold, so the conclusion does not
rest on alias strictness.

## 3. Objective correctness — verified against native source

The pinned launcher omits the flags that select the intended objective, so each was set
explicitly and then verified:

```
loss_type=null_anchor      match_mode=random          null_anchor_tau=0.5
null_anchor_eta=0          null_anchor_kl_dir=forward null_anchor_traj_rollout=False
null_anchor_exclude_special=False   alpha=0           right_shift_logits=False
```

Verification extracted the **real source text** of `_null_anchor_kl` and
`_random_sft_null_anchor_loss` from the pinned MDU file via AST and executed it inside a
minimal shim, then compared our port on identical controlled inputs and masks (batch 1,
unpadded, float64 where the native code allows).

| Check | Result |
|---|---|
| KL matches native | ✅ max abs diff **0.0** |
| Gradients match native | ✅ max abs diff **0.0** |
| KL direction is student-conditional → frozen prompt-masked anchor | ✅ forward matches `KL(p_student‖p_anchor)`; reverse matches the transpose; the two differ by 0.62 |
| **tau multiplies anchor logits** (not a temperature divisor) | ✅ identical to multiplying logits by τ (diff 0.0); **differs** from dividing (diff > 1e-6) |
| Teacher detachment | ✅ anchor logits given `requires_grad=True` receive **zero** gradient |
| Masked-answer reduction, prompt boundaries | ✅ maskable = answer only; 0 masks in prompt; prompt tokens byte-identical; mask nonempty |
| Anchor input is prompt-masked | ✅ all prompt positions = `[MASK]`; answer region identical to the student's input |
| End-to-end vs native training path | ✅ abs diff **< 1e-9** with identical stub models and replayed random draws |
| The real training path executes this objective | ✅ see below |

**Two findings worth recording.**

- **The real dispatch.** For `match_mode=random` + `loss_type=null_anchor` +
  `traj_rollout=False`, `compute_loss` routes to `_random_sft_null_anchor_loss`, which
  uses a **genuinely frozen `ref_model`** as the anchor. The *other* `null_anchor` branch
  (`compute_loss` ~line 864) uses `model` under `no_grad` — frozen only in the stop-grad
  sense, not parameter-frozen — and is **dead code for this configuration**. The loss is
  **added, not negated** (`loss = loss + forget_nll`), since null_anchor is minimized
  directly; with `alpha=0` the retain branch is skipped entirely.
- **Native KL runs in FP32.** `_null_anchor_kl` explicitly casts via `.float()`, so the KL
  is computed in float32 regardless of input dtype. An initial verification attempt failed
  on a 1e-9 tolerance with a residual of 2.6e-07 — exactly float32 epsilon scale. The
  tolerance was wrong, not the objective; the check now compares at FP32 scale. Both the
  failed and corrected attempts are preserved (`a002`, `a004`).

**Reconstructed from the unavailable `dllm`.** `MDUTrainer` subclasses
`dllm.core.trainers.MDLMTrainer`, so `scheduler`, `time_epsilon`, `_postprocess_outputs`
and `right_shift_logits` live in `dllm`, which MDU neither vendors nor pins (its exact
revision remains unverified — consistent with the earlier smoke-test finding). Used here:
`scheduler(t) = 1 − t` so `p_mask = t`; `time_epsilon = 1e-3`; `_postprocess_outputs =
identity` because `right_shift_logits=False` (no AR shift). Cross-check: this yields
`p_mask ~ U(1e-3, 1)`, coinciding with LLaDA's own `GUIDELINES.md` `forward_process`
(`p_mask = (1−eps)t + eps`, `eps=1e-3`). Two independent sources agree, but the `dllm`
identity is still unverified.

**Teacher/student memory strategy — half verified.** The equivalence half holds exactly: a
freshly-initialized LoRA adapter is a no-op and PEFT's `disable_adapter()` restores
pristine base logits (**max abs diff 0.0** in both cases), so one weight copy could serve
as both frozen teacher and student. **Teacher invariance after updates was NOT verified**,
because no optimizer step was taken anywhere in this study. The strategy is therefore
*not* established as working end-to-end, exactly as the protocol warned against assuming.

## 4. Data and evaluator qualification (all frozen pre-inference)

Frozen and committed in `203525b` **before the model ran once**; see
[`FREEZE_NOTE.md`](FREEZE_NOTE.md).

- First ten canonical target entities as the candidate pool.
- **Joint dedup across forget levels 1 and 2** on normalized answer. All ten entities
  yielded ≥11 distinct unaided facts, so **zero source-insufficient entities** — recorded
  before inference.
- Ten distinct qualification facts per entity, drawn only from **unaided-recall** items.
  **Answer-supplying attacks are separated:** any item whose query already contains its own
  normalized answer is excluded from qualification and preserved separately, since it tests
  copying rather than recall.
- **Primary metric = normalized short-answer exact match** (full-string equality after
  normalization). Substring containment is never called exact match — 7 dedicated
  assertions enforce this, including that "Berkshire" must *not* match
  *Berkshire Hathaway* and "Bachman" must *not* match *Richard Bachman*.
- **Reference-token overlap** (reported in tables as `span`) and ROUGE-L recall reported
  separately, alongside empty / refusal / repetition / truncation / parse-failure counts.
  **Correction (Amendment 01 §3):** this metric was called "token-boundary answer-span
  recall", which wrongly implies contiguous-span detection. It is order- and
  adjacency-insensitive reference-token overlap: `span_recall("Hathaway and also
  Berkshire", "Berkshire Hathaway") == 1.0`.
- **Neighbor conflict labelling by two pre-inference signals:** target-name tokens in the
  item, and per-chunk passage overlap with the material we would have trained on. The
  second catches semantic conflicts the first misses — all three *"Spare is a memoir
  released on ___ January 2023"* items for Prince Harry are flagged `name=False,
  passage=True`, *Spare* being his own memoir. Raw benchmark metrics over all neighbor
  items are preserved; nonconflicting locality is a separately labelled quantity.
  *Limitation:* both signals are lexical and a semantic conflict absent from the passage
  corpus can still slip through.
- Source counts resolved before committing panel sizes: all ten entities supply ten
  qualification facts; `10_Prince_Harry,_Duke_of_Sussex` supplies only **7** nonconflicting
  neighbors (53/60 conflict), recorded and never padded with duplicates.

### Known evaluator defect, recorded rather than retrofitted

The frozen parser retains a trailing terminator when a sentence boundary is followed by
more text (`"Paris. It is in France."` → `"Paris."`). Single-sentence answers are clean
(`"Carrie."` → `"Carrie"`). **Impact is provably nil:** `normalize()` strips punctuation,
so EM, span recall and ROUGE-L are identical with and without the defect (demonstrated in
3 assertions). The parser was **not changed**, because outputs already existed and editing
the evaluator after seeing outputs is what the protocol forbids.

## 5. Resource accounting — measured, estimated, unmetered

| Budget | Cap | Actual | Classification |
|---|---|---|---|
| Aggregate GPU | 6.00 GPU-h | **323 GPU-s = 0.090 GPU-h (1.5%)** | **measured** — wall-clock of every attempt, failures included |
| Elapsed | 24 h | **0.32 h** | **measured** — from frozen `START_EPOCH` |
| CPU | 20 core-h | **not metered — remains unknown** | ⚠️ **unmetered** — no `getrusage` instrumentation was wired into the attempt runner. Work was 8 short attempts (3 GPU, 5 CPU, longest 200 s) plus small downloads; the cap is certainly not approached, but no measurement supports a number |
| Additional storage | 10 GiB | **0.227 GiB** | **measured** — `du -sb`, excludes the reused checkpoint |
| Concurrent GPUs | 1 default / 2 max | **1** (device 3) | **measured** |

Peak GPU memory 15.04–15.17 GiB (inference) — comfortable on one A100-40GB.

**Enforced vs advisory.** Enforced: per-attempt wall-clock timeouts via coreutils
`timeout`, with the GPU ceiling additionally **clamped to the remaining GPU budget** before
each GPU attempt, and a refusal guard that aborts rather than overwrite an existing log.
Advisory only: the aggregate elapsed, CPU and storage limits — these are **not** scheduler
or cgroup quotas; nothing would have killed the session had an aggregate been crossed
mid-attempt.

### Attempt ledger — 8 attempts, every one with its own preserved log

| id | phase | label | dur | rc | GPU charged |
|---|---|---|---|---|---|
| `a001` | recall | screen10_primary | 200 s | 0 | yes |
| `a002` | objective | verify_native | 3 s | **1 (FAILED)** | no |
| `a002`* | recall | screen10_secondary_and_mmlu | 114 s | 0 | yes |
| `a003` | objective | verify_native_retry | 2 s | 0 | no |
| `a004` | objective | verify_native_fp32tol | 2 s | 0 | no |
| `a006` | evaluator | verify_scoring | 2 s | 0 | no |
| `a007` | evaluator | verify_scoring_v2 | 2 s | 0 | no |
| `a008` | wiring | teacher_equivalence | 9 s | 0 | yes |

**Three abnormal events of two kinds — one execution exception and two logical failures that
exited 0** (Amendment 02 §5). Their *logs* are preserved; their JSON outputs were
overwritten by later attempts:
- `a002` (objective) — `RecursionError` from a `torch.rand` monkeypatch that called itself.
  Fixed by pre-generating the draw queue over the saved original. Full traceback retained.
- `a006` (evaluator) — surfaced two defects: the real parser defect above, and a faulty
  *test* that string-matched a Python `False` against JSON `false`. The test was fixed; the
  frozen parser was not.

**Ledger defect, disclosed.** Attempt IDs are allocated from a line count, which is racy:
two concurrently launched attempts both received `a002`, and `a005` was skipped. **No log
was overwritten** — the 8 log filenames are distinct (ID + phase + label) and the runner's
`[ -e "$LOG" ]` guard would have refused a true collision. The defect is in ID uniqueness
only; all 8 logs are published.

## 5b. Gate weakness and chronology (added by Amendment 01)

**Two failing checks exited 0.** `a003` (objective, `kl_direction` FAILED) and `a006`
(evaluator, 42/44 FAILED) both printed their failure and returned **exit code 0**; neither
frozen script contains any `sys.exit`. A failing correctness gate was therefore
indistinguishable from success to an automated caller, and the failures were caught only
by reading the logs. `a002`'s execution exception did correctly return 1. The final passing
results (`a004`, `a007`) were re-run after real fixes and remain valid; the frozen scripts
are not modified.

**The supplementary checks ran after the outcome was known.** The primary recall screen
ended **00:45:10**, at which point 0/10 qualifying was determinable. All seven later
attempts — the secondary/MMLU screen (00:46:36), every objective and evaluator attempt,
and the **9-second teacher-wiring check at 00:50:44** — ran after that. They are post-hoc
infrastructure verification, not pre-registered gates. Relatedly, a freshly
zero-initialized adapter agreeing with the base model is a no-op by construction and does
**not** establish teacher invariance after updates; that remains unverified.

## 6. What was NOT done

- **No training.** Zero optimizer steps. The §3 stop condition applied before §5 began, so
  no learning-rate sweep, no checkpoints, no per-entity trajectories.
- **No reload verification** — there is no trained adapter to reload.
- **No behavioral-effect measurement** — impossible without training, and meaningless
  without base recall to forget.
- **No cloud-team candidate method**, no new method, no baseline port.
- No paid APIs, no additional large-model downloads (the checkpoint was reused from the
  closed smoke test).

## 7. Preserved work

New branch `research/llada-rwku-baseline-v1` from `origin/main`, new results directory.
The closed LLaDA smoke study (`research/llada-access-smoke-v1`, commits `bd3615db` and
`181189ed`) is untouched, as are all other branches. GPUs 0 and 1 were left alone
throughout per the user's instruction; the other user's processes were never signalled.

## 8. Next recommendation (one)

**Re-qualify the panel against knowledge the pristine model actually holds, before any
unlearning work — by screening RWKU entities for base recall at scale rather than
assuming the first ten canonical targets are suitable.**

The evidence supports this specifically:

- Base recall is the binding constraint, not infrastructure. Both the evaluator (48/48) and
  the objective (10/10 vs native *helpers*) ran and passed, and MMLU at 75% shows the
  checkpoint loads and that one constrained single-letter path decodes and parses.
  **Corrected (Amendment 02 §2): MMLU does NOT establish that the whole evaluation harness
  is sound** — it exercises a disjoint 4-way letter scorer, not the open-answer path that
  produced the outcome.
- The shortfall is not marginal (mean 0.220, max 0.50 against a 0.60 threshold) and was low
  at both tested schedules. **Corrected (Amendment 02 §3):** agreement between the two
  frozen schedules does NOT rule out decoding effects and does NOT prove decoding changes
  cannot affect recall; block length, generation length, temperature, remasking and CFG
  were never varied.
- **WITHDRAWN (Amendment 02 §4).** This report previously recommended screening all 200
  RWKU entities, projecting ~1.1 GPU-h by linear extrapolation from the 10-entity screen's
  200 GPU-s. **That recommendation is withdrawn**: the shared resource envelope has since
  been consumed by subsequent work (see Stage B's corrected ledger), the figure was an
  extrapolation rather than a measurement, and **no replacement experimental recommendation
  is offered here.**
- Doing this first would prevent a category of wasted work: any forgetting measurement on an
  entity the model never recalled would measure absence, not forgetting. That reasoning
  stands; the specific 200-entity plan does not.

---

## Evidence index

See [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) for every published file with its SHA-256.
Corrections are in [`AMENDMENT_01.md`](AMENDMENT_01.md) and
[`AMENDMENT_02.md`](AMENDMENT_02.md). The reviewed publications are
[`37c31ca`](https://github.com/bijay299/XXResearch/commit/37c31ca17a004b7bf7c44634e395e2eef7a9678f)
and [`a349197`](https://github.com/bijay299/XXResearch/commit/a349197c471d8bd74878b6b26f622f3f225c7232).
