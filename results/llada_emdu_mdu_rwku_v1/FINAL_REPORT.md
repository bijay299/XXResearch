# EMDU vs MDU on RWKU (LLaDA-8B) — FINAL REPORT

**Outcome: `PAPER_ALIGNED_RUN_BLOCKED`.**

The protocol and data are frozen, both objectives are implemented and checked, and the
evaluation panels and scoring helpers are **configured** — but the **evaluator was never
executed**. The comparison was **not run**, because **the tested custom CPU-offload
implementation** of the paper-aligned full-parameter recipe (including its one permitted
repair) did not fit the remaining resource envelope, as measured. No subject, evaluation
panel or recipe was trimmed to make it fit, and full-parameter training was **not**
silently replaced by LoRA, quantization or a shortened recipe.

**Scope limits (Amendment 01).** The measurements do **not** prove universal
full-parameter infeasibility on two 40 GB GPUs, and do **not** establish that the
slowdown is definitively bandwidth-bound rather than implementation-dependent. **No
optimized offload or sharding solution was qualified** — none was attempted. **11
engineering optimizer updates** ran on the real model; **0** of the 240 planned scientific
comparison updates did.

**Host:** hyperplane, 2× A100-SXM4-40GB (devices 2 and 3, per the recorded user override;
devices 0–1 left untouched) + 491 GiB host RAM
**Shared window:** original task start 2026-10-20T00:31:58; **no clock restart**

> **Amended 2026-10-20 — see [`AMENDMENT_01.md`](AMENDMENT_01.md).** Corrects the CPU total
> (an octal parsing bug undercounted it by 14,748 core-s = 4.0967 core-h; corrected total
> **58,340 core-s = 16.2056 core-h**), **withdraws** the single-80 GB-GPU recommendation and
> its "minutes" prediction (the stated layout is ~134 GiB), **withdraws** the
> bandwidth-bound attribution and the native batch-reduction equivalence claim, and narrows
> the fidelity, coverage and chronology claims. **Outcome `PAPER_ALIGNED_RUN_BLOCKED`
> unchanged — scoped to the tested custom CPU-offload implementation.**

---

## Statuses

| Phase | Status |
|---|---|
| EMDU author code available | ❌ **Not located in the sources inspected 2026-10-20** — paper-based reimplementation (labeled) |
| Data qualification (582 pairs/subject) | ✅ **PASS** — exactly 582 verified for all 3 subjects |
| Frozen protocol + evaluation IDs | ✅ **FROZEN** before any Stage B model output |
| Objective correctness (CPU, independent + extracted native **helper**) | ✅ **PASS** — 24/24 observed; the complete native trainer was never executed (§5) |
| Runtime correctness (real model, real data) | ✅ **PASS** — 12/12 observed; both objective *forwards* ran, optimizer updates were **EMDU-only (2)**, digests covered **40/291** tensors (§5) |
| Full-parameter feasibility, **tested CPU-offload implementation** | 🔴 **BLOCKED** — measured; no tried setting fits both budgets. Not a universal infeasibility result |
| EMDU vs MDU comparison | ⛔ **NOT RUN** |

**This is not** a reproduction of the paper's tables, not independently established SOTA,
and not a new contribution.

---

## 1. Pins

| Asset | Pin |
|---|---|
| EMDU paper | arXiv:2610.25447v1 (Garber & Tirer), local PDF sha256 `d8059bebeceda16c2de2055f2127b84640fc1f7808d451e2e22cdb19f0bb915a` |
| MDU | `e5e8e8274ff8f56f16e978803d9e902359eb5272` |
| LLaDA-8B-Instruct | `08b83a6feb34df1a6011b80c3c00c7563e963b07` (reused, previously hash-verified) |
| LLaDA code | `9182493720ed723ef8031210d85959364e51cbe0` |
| RWKU | `d72f493d481d1b0a9bdc6e869d32baeffad8904f` |

### EMDU author code was not located in the sources inspected (2026-10-20)

**Bounded statement (Amendment 01).** The paper gives **no EMDU repository**; its
implementation appendix says it extends the MDU repo "with EMDU (anchor-first KL and full
answer-span masking), complete TOFU / RWKU evaluation, and **the checkpoints we will
release**" — future tense. One web search on 2026-10-20 returned only the arXiv entry.
**Absence from those sources is not proof of non-existence**, and that search has not been
repeated. Our EMDU arm is therefore an explicitly labeled **paper-based reimplementation**,
with the reconstruction list in §3.

Incidentally, the paper's footnote 2 cites `https://github.com/ZHZisZZ/dllm`, which
**independently confirms** the `dllm` identification our earlier smoke study could only
infer from symbol matching. The paper also states it "was unable to reproduce Lee et
al.'s reported numbers from released artifacts alone."

## 2. The method difference, exactly as implemented

Shared by both arms: student sees the **original prompt + randomly masked answer**; the
frozen anchor sees a **fully masked prompt**; `tau=1` **multiplying** anchor logits;
`alpha=0`; **no retain corpus**; identical data order, masking draws, optimizer,
precision and update budget.

| | anchor's view of the answer | KL direction |
|---|---|---|
| **MDU** | the **same partially masked** answer as the student | `KL(student ‖ anchor)` — anchor-last |
| **EMDU** | the **entire answer span masked** | `KL(anchor ‖ student)` — anchor-first |

The student's answer masking stays random in EMDU. The intended loss is selected
**explicitly** (`loss_type=null_anchor`, `match_mode=random`), never left to MDU's
launcher defaults, which would select gradient ascent (`--loss_type ga`).

Reduction (both): pooled mean over the student's masked answer positions,
`sum(kl*masked)/masked.sum().clamp_min(1)` — a single pooled mean across the batch's
masked positions, matching MDU's native code rather than a per-sequence mean. Anchor
detached in both directions. `exclude_special=False`, so KL is over the full vocabulary.
`right_shift_logits=False` — **no autoregressive shift**; LLaDA is bidirectional.

## 3. Data, and the reconstruction list

**Training data:** the published Llama-generated RWKU forget pairs,
`Target/<subject>/pair.json` (RWKU config `train_original_passage` is a *different* file;
these pairs are the `train_pair_llama3` family). Verified **exactly 582 records** for each
of the three subjects, matching the paper's "582 prompt–response biographies per entity".
Records are used as published — **never duplicated** to reach the count. The previous
study's biography-template corpus is **not** reused, and **no evaluation question string
is ever trained on**.

Reconstructions, frozen before any outcome (full text in `frozen/stageb_spec.json`):

| Unspecified | Reconstruction | Basis |
|---|---|---|
| RWKU decoder | 64 tokens, 64 steps, block 32, temp 0, low-confidence | The paper's 128-step/128-token figure is explicitly **TOFU-only** ("Unless otherwise noted, **TOFU** generations…"); taken instead from MDU `configs/mdu_rwku.yaml` (`max_new_tokens 64`, `steps 64`, `temperature 0.0`). Block length is unspecified in both sources; frozen at 32. |
| `response` field | `response[0]` | `pair.json` `response` is a **list of 2** candidate biographies; the paper does not say which. |
| Sequence length | 512 tokens, EOS right-padding | Unspecified; responses median ~316 words. |
| scheduler / `time_epsilon` | `scheduler(t)=1−t` ⇒ `p_mask=t`; `1e-3` | From the unavailable `dllm`; cross-checked against LLaDA `GUIDELINES.md`. |
| Optimizer / precision | AdamW (0.9, 0.999), eps 1e-8, wd 0, fp32 state, bf16 model, constant LR | Unspecified by the paper. |
| Microbatching | microbatch 1 × accum 4 | Paper used batch 4 / accum 1. Each microbatch loss is divided by 4 before backward. ⚠️ **The claim that this matches native batch-4 mean reduction is WITHDRAWN** (Amendment 01 §4): ours is `mean_i(sum_i/count_i)`, native pooled normalization is `sum_i(sum_i)/sum_i(count_i)`; these differ whenever masked counts differ, and `p_mask ~ U(1e-3,1)` is drawn per microbatch. **Unresolved fidelity issue**; nothing was re-tested and no result depends on it, since the comparison never ran. |
| MMLU cue | assistant-side `Answer:` + first-standalone-`A–D` parser | RWKU's `eval_mmlu.py` places the cue after the assistant header. The paper warns LLaDA often **restates** the cue ("Answer: C"), which a first-character rule misreads as "A"; our parser is validated against that case. |

## 4. Frozen evaluation (before any Stage B output)

Three subjects, **fixed by instruction** — Stephen King, Confucius, Bruce Lee — chosen
**not** by observed recall and never replaced.

| Subject | F-L1 | F-L2 | F-L3 | N-L1 | N-L2 | train pairs |
|---|---|---|---|---|---|---|
| Stephen King | 8 | 10 | 29 | 30 | 30 | 582 ✓ |
| Confucius | 17 | 3 | 28 | 30 | 22 | 582 ✓ |
| Bruce Lee | 17 | 19 | 40 | 30 | 30 | 582 ✓ |

Complete applicable panels: **343 items per arm × 3 arms = 1,029 generations**, plus
reduced utility panels of **40 MMLU** and **40 TriviaQA** (TriviaQA scored as the max
ROUGE-L recall over its list of acceptable answers). These are **reduced** utility panels,
not full-table reproduction. Exact row IDs and counts are frozen in
`frozen/stageb_panels.json`.

Primary metric: **ROUGE-L recall, reported separately** for F-L1/L2/L3 and N-L1/L2.
Exact match is kept only as a secondary diagnostic, and the previous study's 6/10
exact-match admission gate is **not** reused. Raw neighbor scores over the complete
panels are the primary locality number; any target-conflicting-neighbor exclusion was
pre-defined as a **separately labeled** secondary analysis.

## 5. Correctness — what passed

**Objective correctness: 24/24 (CPU, controlled fixtures, batch 1, unpadded).**
Loss and gradients for **both** arms match independently written formulas and independent
autograd references. The MDU arm matches the **pinned native `_null_anchor_kl`**, extracted
by AST and **executed** — not inspected. EMDU is confirmed to differ from native MDU.
Boundaries hold: student prompt uncorrupted, anchor prompt fully masked, masks nonempty,
no student mask in the prompt. **EMDU's anchor has its entire answer span masked**, while
MDU's anchor carries exactly the student's partial mask. Anchor detached in both
directions. `tau` multiplies (not divides) anchor logits. The student's input is identical
across methods, so only the anchor and KL direction differ.

**Failed checks now return nonzero exit codes — and that is demonstrated.** Amendment 01
recorded that the previous study's gates printed failures while exiting 0. A
negative-control variant with `tau` deliberately changed to *divide* exits **1** and flags
exactly `tau_multiplies_anchor_logits`, while the real verifier exits 0.
*Disclosed:* my **first** negative control was invalid — the verifier hard-codes its
import path, so the broken module was never loaded and it misleadingly exited 0. Both the
invalid and the corrected control are published under
`evidence/negative_control/`.

**Runtime correctness: 12/12 observed** (real model, real tokenized RWKU pair, real
optimizer updates): see `outputs/stageb_runtime_checks.json`. **Coverage, stated precisely
(Amendment 01 §5):**

- **Both objective forwards executed** — `runtime_dispatch` is recorded for `mdu` and
  `emdu`, read from the executed objective rather than from source strings.
- **Optimizer updates were EMDU-only: 2.** No MDU optimizer trajectory was run, and
  neither method has a trajectory. The frozen-teacher invariance result therefore holds
  for a student changed by **two EMDU updates**.
- **Teacher/student digests covered 40 of 291 parameter tensors** (`n_tensors_hashed: 40`).
  Full-model `requires_grad` inspection (291/291 student trainable; all anchor tensors
  frozen) and **sampled** hash invariance are *different checks*: the digest result is a
  sample, not full-model coverage.
- The student digest did change and the anchor digest did not — which is a genuine
  improvement on the previous study's zero-init no-op adapter, but at 40/291 sampling.
- **No checkpoint/reload qualification was demonstrated**, and the **evaluator was never
  executed** — panels, row IDs and scoring helpers are configured and frozen, but no
  generation, scoring or comparison artifact exists. A configured panel is not an executed
  evaluator.

### Chronology (existing timestamps)

| time | attempt | note |
|---|---|---|
| 01:27:01 → 01:31:35 | `b0002` feasibility probe | **3 optimizer updates — before the freeze** |
| 01:33:39 → 01:39:09 | `b0003` feasibility repair | **6 optimizer updates**; started before the freeze, ended after |
| **01:35:39** | `b0004` **protocol + eval-ID freeze** | — |
| 01:37:27 | `b0005` objective checks | after freeze |
| 01:40:19 → 01:43:15 | `b0006` runtime checks | **2 EMDU updates, 352 GPU-s — after the repaired feasibility blocker** (determined 01:39:09) |

**Engineering model outputs preceded the protocol freeze.** **No comparison outcomes
preceded it, because the comparison never ran** — the freeze governs the unrun scientific
comparison, not the engineering probes. "No training occurred" would be inaccurate: there
were **11 engineering optimizer updates** (3 + 6 + 2); what did not occur is the
**scientific comparison** (0 of 240 planned trajectory updates, and no evaluation).

## 6. The measured blocker

**Documented sharding/offload plan** (no LoRA, no quantization, no shortened recipe):

```
cuda:0   student, full-parameter, bf16 params (16 GiB) + bf16 grads (16 GiB)
cuda:1   frozen anchor, bf16 (16 GiB), no_grad
host     AdamW state fp32: master + exp_avg + exp_avg_sq  (~90 GiB of 491 GiB)
step     grads -> CPU, AdamW on CPU in fp32, updated params -> GPU
batch    microbatch 1 x accum 4 = effective batch 4
```

**Memory fits.** Feasibility-phase peaks were **34.714 GiB** (cuda:0) and **15.067 GiB**
(cuda:1); the **later runtime checks** peaked higher at **35.078 / 16.872 GiB**
(Amendment 01 §5). All are under 40 GiB. The original report quoted the feasibility
figures as though global. The model math is also cheap: **~1.0–1.6 s** of forward+backward per update.

**Time does not fit.** The CPU-offloaded optimizer is ~98% of each update, because 8 B
parameters of fp32 AdamW state (~90 GiB) cannot reside on 2×40 GiB devices and must live
in host RAM. Attempt 1 measured 43.9–67.8 s per update. The one permitted repair replaced
the naive per-tensor CPU AdamW with in-place `_foreach` ops and preallocated buffers, and
swept CPU threads to find *any* workable point on the trade-off (core-seconds ≈ threads ×
wall, while GPU-seconds ≈ 2 × wall, since both devices stay allocated while the CPU
optimizer runs):

| threads | per update | 240 updates → GPU-s | vs 20,729 remaining | → CPU core-h | vs 20 cap | fits both |
|---|---|---|---|---|---|---|
| 255 | 51.9 s | 24,927 | ✗ over by 4,198 | 873.3 | ✗ | **no** |
| 32 | 43.6 s | 20,923 | ✗ over by 194 | 98.4 | ✗ | **no** |
| 8 | 43.4 s | 20,831 | ✗ over by 102 | 30.2 | ✗ | **no** |

The repair produced only a **1.01× speedup**, and wall time is essentially flat from 8 to
255 threads.

⚠️ **Attribution withdrawn (Amendment 01 §3).** The original report concluded from this
that the cost is "memory-bandwidth/transfer bound, not an implementation or parallelism
defect". That is **not established**: thread count was the only dimension varied. The flat
curve is *consistent with* a transfer limit but does not demonstrate one. Untested
alternatives include pinned-memory and overlapped/streamed transfers, fused or chunked
optimizer kernels, optimizer state sharded across both devices, ZeRO/FSDP, and 8-bit or
bf16 optimizer states. **No optimized offload or sharding solution was qualified.**

**The blocker, precisely.** The fixed scope is 3 subjects × 2 methods × 40 updates = **240
full-parameter updates**. At the best measured setting that is **20,831 GPU-s**, which
already **exceeds the 20,729 GPU-s remaining** — before *any* of the 1,029-generation
evaluation (~2,200 GPU-s) — and simultaneously needs **30.2 CPU-core-hours against a 20
core-hour cap**, of which 10.8 are already spent. Both independent budgets fail at every
measured point.

**What was deliberately not done:** no substitution of LoRA, quantization, bf16 optimizer
state, fewer subjects, fewer updates or a reduced evaluation; and no open-ended
infrastructure rescue (FSDP/ZeRO wiring for a `trust_remote_code` model was not attempted
— DeepSpeed is not installed, and that work is exactly the unbounded path the protocol
rules out).

## 7. Resource accounting

Carried forward from the original study; **the clock was not restarted**.

| Budget | Cap | Carried in | This task | Total | Remaining | Basis |
|---|---|---|---|---|---|---|
| Aggregate GPU | 6 h (21,600 s) | 323 s | see ledger | see ledger | see ledger | **measured** — wall × devices, every attempt incl. failures |
| Elapsed | 24 h from 00:31:58 | 0.32 h | — | ~1.2 h | ~22.8 h | **measured** |
| CPU | 20 core-h | **UNMETERED** | measured | — | — | historical remains unknown; **no retrospective charge invented**. Metering began with this task. |
| Storage | 10 GiB | 0.23 GiB | +~0.01 GiB | ~0.24 GiB | — | **measured** |
| Concurrent GPUs | 2 max | — | 2 | — | — | **measured** |

Live figures are in `scripts/ledger.sh` output, reproduced in `RESOURCE_LEDGER.md`.

**Attempt discipline.** Unique IDs allocated atomically via `mkdir` (fixing the previous
study's racy line-count allocator), one preserved log per attempt, and GPU timeouts clamped
to the **remaining** aggregate budget before each launch. Every attempt — including the
failed feasibility repair, which exits 3 by design — is in `evidence/ATTEMPTS.tsv` with its
own log.

*Disclosed ledger cosmetic defect:* one `cpu_core_s` cell renders as `026063` (a leading
zero from `awk` formatting) rather than `26063`. Values are otherwise correct; the raw logs
are authoritative.

## 8. What this does and does not establish

**Does:** EMDU author code was **not located in the sources inspected on 2026-10-20**; the
published 582-pair data is present and counted; both objectives are implemented and checked
against independent formulas and against MDU's **extracted native helper** (executed on
fixtures — not the complete native trainer); the exact method difference is isolated to the
anchor construction and KL direction; and **the tested custom CPU-offload implementation**
of the paper-aligned full-parameter recipe measurably did not fit the remaining envelope.

**Does not:** produce any forgetting, locality or utility number; compare EMDU to MDU;
reproduce the paper's tables; or say anything about whether EMDU improves on MDU —
**neither this outcome nor Stage A's establishes which method is better.** No trained
checkpoint exists, so no checkpoint delivery or reload qualification is claimed, and the
evaluator was never executed. It does **not** establish universal full-parameter
infeasibility on two 40 GB GPUs, does **not** establish that the slowdown is definitively
bandwidth-bound, and qualifies **no** optimized offload or sharding solution. It does
**not** establish that our microbatch reduction matches native pooled normalization — that
is an unresolved fidelity issue (§3). The earlier qualification study remains
**`INSUFFICIENT_BASE_RECALL`**, unchanged.

Note for any future run: even had training fit, the paper's reported MDU row was
**independently tuned** (LR swept 1e-5→3e-5, length 20 steps→3 epochs, configuration chosen
for the largest Forget/Neighbor gap). A matched-recipe MDU arm would therefore not be a
reproduction of its optimized Table 4 row.

## 9. Next recommendation — WITHDRAWN

The original §9 recommended either a parameter-efficient re-scope **or obtaining a single
80 GB A100/H100**, predicting the latter would hold "params + grads + fp32 AdamW state
on-device" and make 240 updates cost "on the order of minutes".

**Both hardware claims are withdrawn** (Amendment 01 §2). For the recorded
8,015,581,184 parameters:

| Layout | B/param | Size |
|---|---|---|
| FP32 master + two FP32 moments | 12 | **≈ 89.58 GiB** |
| + BF16 student weights + BF16 grads | 16 | **≈ 119.44 GiB** |
| + BF16 frozen anchor | 18 | **≈ 134.37 GiB** |

Excluding activations and other buffers. The bare optimizer state alone exceeds both 80 GB
and 80 GiB, and the full stated layout is ~1.7× an 80 GB card — so such a card would **not**
hold this layout on-device, and the "minutes" figure rested on assuming the optimizer cost
would vanish on untested hardware.

**No substitute hardware prescription is offered**, and no replacement experimental
recommendation is made: determining a workable configuration is outside this amendment's
authorization. What is recorded instead is the scoped blocker (§6, as corrected) and the
unresolved fidelity issue (§3's microbatch row) that any future run must address first.

---

## Evidence index

See [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) for every published file with its SHA-256,
[`AMENDMENT_01.md`](AMENDMENT_01.md) for the corrections, and
[`RESOURCE_LEDGER.md`](RESOURCE_LEDGER.md) for the corrected accounting. Reviewed
publication: [`cc54d83`](https://github.com/bijay299/XXResearch/commit/cc54d8389cb06223bb9296e7afa620afea2fa6e8).
