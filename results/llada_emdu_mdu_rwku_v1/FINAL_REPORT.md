# EMDU vs MDU on RWKU (LLaDA-8B) — FINAL REPORT

**Outcome: `PAPER_ALIGNED_RUN_BLOCKED`.**

The protocol, data, both objectives and the evaluation are implemented, frozen and
verified. The comparison was **not run**, because full-parameter training of the
paper-aligned recipe does not fit the authorized envelope — and that was established by
**measurement**, not arithmetic. No subject, evaluation panel or recipe was trimmed to
make it fit, and full-parameter training was **not** silently replaced by LoRA,
quantization or a shortened recipe.

**Host:** hyperplane, 2× A100-SXM4-40GB (devices 2 and 3, per the recorded user override;
devices 0–1 left untouched) + 491 GiB host RAM
**Shared window:** original task start 2026-10-20T00:31:58; **no clock restart**

---

## Statuses

| Phase | Status |
|---|---|
| EMDU author code available | ❌ **No** — paper-based reimplementation (labeled) |
| Data qualification (582 pairs/subject) | ✅ **PASS** — exactly 582 verified for all 3 subjects |
| Frozen protocol + evaluation IDs | ✅ **FROZEN** before any Stage B model output |
| Objective correctness (CPU, independent + native) | ✅ **PASS** — 24/24 |
| Runtime correctness (real model, real data) | see §5 |
| Full-parameter feasibility | 🔴 **BLOCKED** — measured; no setting fits both budgets |
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

### EMDU author code has NOT become available

Checked directly. The paper gives **no EMDU repository**; its implementation appendix says
it extends the MDU repo "with EMDU (anchor-first KL and full answer-span masking),
complete TOFU / RWKU evaluation, and **the checkpoints we will release**" — future tense.
A web search surfaced only the arXiv entry. Our EMDU arm is therefore an explicitly
labeled **paper-based reimplementation**, with the reconstruction list in §3.

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
| Microbatching | microbatch 1 × accum 4 | Paper used batch 4 / accum 1. Each microbatch loss is divided by 4 before backward, so accumulated gradients equal the **mean over the 4 examples**, matching batch-4 mean reduction. |
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

**Runtime correctness** (real model, real tokenized RWKU pair, real optimizer updates):
see `outputs/stageb_runtime_checks.json`. This probe exists specifically to discharge the
two things Amendment 01 flagged as unverified before: runtime dispatch is read from the
executed objective rather than from source strings, and **frozen-teacher invariance is
checked against a genuinely changed student** (real AdamW updates), not against a
zero-initialized no-op adapter.

## 6. The measured blocker

**Documented sharding/offload plan** (no LoRA, no quantization, no shortened recipe):

```
cuda:0   student, full-parameter, bf16 params (16 GiB) + bf16 grads (16 GiB)
cuda:1   frozen anchor, bf16 (16 GiB), no_grad
host     AdamW state fp32: master + exp_avg + exp_avg_sq  (~90 GiB of 491 GiB)
step     grads -> CPU, AdamW on CPU in fp32, updated params -> GPU
batch    microbatch 1 x accum 4 = effective batch 4
```

**Memory fits.** Measured peak **34.71 GiB** (cuda:0) and **15.07 GiB** (cuda:1), both
under 40 GiB. The model math is also cheap: **~1.0–1.6 s** of forward+backward per update.

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
255 threads. That is the diagnostic that matters: the cost is **memory-bandwidth/transfer
bound, not an implementation or parallelism defect**, so no thread setting or further
micro-optimization rescues it.

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

**Does:** EMDU author code does not exist yet; the published 582-pair data is present and
counted; both objectives are implemented and verified against independent formulas and
against MDU's executed native helper; the exact method difference is isolated to the anchor
construction and KL direction; and full-parameter paper-aligned training is **measurably**
outside this envelope on 2×40 GiB devices.

**Does not:** produce any forgetting, locality or utility number; compare EMDU to MDU;
reproduce the paper's tables; or say anything about whether EMDU improves on MDU. No
trained checkpoint exists, so no checkpoint delivery or reload qualification is claimed.
The earlier qualification study remains **`INSUFFICIENT_BASE_RECALL`**, unchanged.

Note for any future run: even had training fit, the paper's reported MDU row was
**independently tuned** (LR swept 1e-5→3e-5, length 20 steps→3 epochs, configuration chosen
for the largest Forget/Neighbor gap). A matched-recipe MDU arm would therefore not be a
reproduction of its optimized Table 4 row.

## 9. Next recommendation (one)

**Re-scope to a parameter-efficient comparison and label it as such, or obtain an 80 GB
device — but decide that explicitly rather than by silently shrinking the recipe.**

The measurements make the choice concrete. The blocker is not compute-bound model math
(~1 s/update) but the ~90 GiB fp32 optimizer state that cannot sit on 2×40 GiB cards. Two
honest routes follow, and they are not equivalent:

1. **One 80 GB A100/H100** would hold params + grads + fp32 AdamW state on-device, removing
   the CPU offload entirely. At the measured ~1.0–1.6 s of model math per update, 240
   updates would cost on the order of minutes of GPU time rather than hours — this is the
   route that preserves the paper's full-parameter recipe and is worth a hardware request.
2. **A LoRA-based comparison** fits today's devices comfortably (the earlier smoke study
   measured 17.1 GiB peak for a LoRA step), but it answers a **different question** than
   the paper's full-parameter result and must be reported as an adapter-level comparison,
   never as a paper-aligned one.

Not recommended: bf16 optimizer state, FSDP/ZeRO wiring for this `trust_remote_code` model,
or trimming subjects/updates to squeeze under the cap — the first two are the open-ended
rescue the protocol rules out, and the third would silently change the recipe under
comparison.
