# LLaDA Access & Engineering Feasibility Check — FINAL REPORT

**Task:** bounded access + engineering feasibility smoke test for LLaDA on Hyperplane
**Host:** `hyperplane` — 4× NVIDIA A100-SXM4-40GB, driver 550.144.03 / CUDA 12.4, 2× AMD EPYC 7742
**Window:** 2026-10-19 23:22:10 → 23:45 EDT (**0.38 h of a 3.00 h cap**)
**Frozen manifest:** [`RUN_MANIFEST.md`](RUN_MANIFEST.md), sha256 `3c9fee376c836636a546f42b6231b2b37d69d07dc02edc620dc4b883219bfdee`

> **Amended 2026-10-20 — see [`AMENDMENT_01.md`](AMENDMENT_01.md).** Documentation and
> evidence-publication corrections: the four run logs are now published (they were silently
> excluded by a `.gitignore` rule), the full-parameter memory arithmetic is corrected, absence
> claims are narrowed to what was inspected, and accounting figures are labelled measured vs
> estimated vs unmetered. **All five check statuses and every measured figure are unchanged.**
> No new model execution, training, inference or benchmark evaluation occurred.

---

## Status summary

| Check | Status | Scope of what was actually established |
|---|---|---|
| **Model access** | ✅ **PASS** | Both checkpoints reachable anonymously from hyperplane; Instruct downloaded, all 6 shard SHA-256 verified against pinned LFS digests; MIT license confirmed |
| **Native inference** | ✅ **PASS** | 4/4 prompts produced valid non-empty output via the **author's masked-diffusion `generate()`**; mask schedule strictly 64→0; peak 15.00 GiB |
| **One-step LoRA training** | ✅ **PASS** — **limited to training mechanics** | All 5 verification gates passed on the first attempt; base model provably unchanged across **all 291** base tensors; peak 17.07 GiB. Establishes that the optimizer step is mechanically correct — **not** learning efficacy, and **not** full-parameter feasibility |
| **Benchmark asset readiness** | ⚠️ **PARTIAL — TOFU reproduction blocked** | Data + code + licenses all accessible; **no author-matched LLaDA TOFU starting checkpoint was located in the inspected release or recorded search** |
| **Full baseline reproduction** | ⛔ **NOT ATTEMPTED** | By instruction. No benchmark evaluation, no second large model, no TOFU backbone training, no paid API |

**Headline:** *inference and LoRA training passed; benchmark reproduction remains blocked by a
missing starting checkpoint.*

### What this report does **not** claim

- It is **not** evidence of unlearning efficacy. No unlearning method was run.
- It is **not** benchmark accuracy or a paper reproduction. Four prompts are a smoke test.
- It does **not** show that full-parameter MDU training fits this hardware. The training gate
  used LoRA at 0.10% of parameters; see [§3 memory caveat](#3-one-step-lora-training--pass).
- It does **not** establish complete research readiness. See [§4](#4-benchmark-asset-readiness--partial).

---

## 1. Model access — ✅ PASS

Verified **from hyperplane** by direct HF Hub API call, not inferred from a web page.
No HF token exists on this host; none was created or requested. Access was fully anonymous.

| | LLaDA-8B-Base | LLaDA-8B-Instruct |
|---|---|---|
| Revision (`sha`) | `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| HTTP | 200 | 200 |
| `gated` / `private` | false / false | false / false |
| License | **MIT** | **MIT** |
| Files | 16 | 16 |
| Action | **metadata only** (not downloaded, per instruction) | **downloaded** (14.94 GiB, 6 shards) |

**Source code pinned:** `ML-GSAI/LLaDA` @ `9182493720ed723ef8031210d85959364e51cbe0` (2026-07-15).

**Weight integrity — all 6 shards match the pinned LFS digests exactly:**

| Shard | SHA-256 (computed on host == pinned HF LFS oid) |
|---|---|
| 1 | `ff6699df9635a267fb4ac801ea8012d39b21bb22ed290d5a57128521b889c230` |
| 2 | `b8ddf571b47f57ce654913877f3625d2c7b9a4c3f81cb6f41f6c3055473b0281` |
| 3 | `8347db2fd61a3b5bad2970c2f83ecb84b93632e276a41ca169dcc5efae15462c` |
| 4 | `93d8706872e7432d5cd40c9a61f9f9d270c101828cef2adaf27c67743d3dcf1a` |
| 5 | `ff5f765eb7ec8bfdbe3b962a9e920d71de43d55e04eb4844710c0efcc8855092` |
| 6 | `f25d4cc57b24a79a840d49c3b25d9a22dd5caf3a9766f591e4625377901194e2` |

Tokenizer/config revisions are the same commit `08b83a6f`:
`config.json` `5f99fefe…`, `tokenizer.json` `071eb20f…`, `tokenizer_config.json` `b9a6102c…`,
`configuration_llada.py` `c947a64d…`, `modeling_llada.py` `98bac7e5…`.

**Custom code reviewed before `trust_remote_code` was enabled.** Both custom modules were read
and grep-audited: imports are limited to torch / transformers / Python stdlib, with no
`subprocess`, `os.system`, `eval(`, `exec(`, `pickle`, `requests`, `urllib`, `socket`, no file
writes and no network downloads. `trust_remote_code=True` was then enabled **against the local
pinned files only**, with `HF_HUB_OFFLINE=1` set.

No substitution was made: this is the official `GSAI-ML` checkpoint — not iLLaDA, not a
quantized variant, not an unofficial re-upload.

## 2. Native inference — ✅ PASS

Ran the **author's masked-diffusion `generate()`** imported unmodified from the pinned repo
(`generate.py`). This is a diffusion reverse process with low-confidence remasking — **not** an
autoregressive `.generate()` call, which would not be equivalent.

Fixed settings (frozen in the manifest before execution): batch size **1**, `gen_length=64`,
`steps=64`, `block_length=32` (2 blocks, semi-autoregressive), `temperature=0.0`,
`cfg_scale=0.0`, `remasking='low_confidence'`, `mask_id=126336`, seed **42**.

| # | Prompt | Output | Runtime | Schedule monotone | Masks left | Non-empty |
|---|---|---|---|---|---|---|
| 0 | *What is the capital of France?* | "The capital of France is Paris." | 2.32 s | ✅ | 0 | ✅ |
| 1 | *Name three primary colors.* | "Three primary colors are red, blue, and yellow." | 1.91 s | ✅ | 0 | ✅ |
| 2 | *Explain in one sentence what photosynthesis is.* | "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy stored in glucose, using carbon dioxide and water." | 1.90 s | ✅ | 0 | ✅ |
| 3 | *What is 17 plus 25?* | "17 plus 25 equals 42." | 1.92 s | ✅ | 0 | ✅ |

Model load 7.0 s · total generation 8.05 s · **peak GPU memory 15.004 GiB** (bf16 weights ≈ 15 GiB,
so activation overhead at this length is negligible) · 1 GPU (device 0).

**Masking/remasking schedule verified**, not assumed. A non-intrusive probe recorded the number
of still-`[MASK]` positions at every model call without altering the algorithm. Each prompt made
exactly **64 forward calls** (= `steps`), and the mask count fell strictly monotonically
`64, 63, 62, …, 3, 2, 1 → 0`: one token committed per step, matching
`get_num_transfer_tokens` for 64 masks over 2 blocks × 32 steps under LLaDA's linear noise
schedule. No `126336` remained in any output. Full per-prompt token IDs, trajectories and
decoded text: [`evidence/inference_results.json`](evidence/inference_results.json).

**Honesty notes.** The 4 prompts and the seed were fixed in the manifest *before* execution;
there was **no prompt replacement and no repeated sampling** to improve appearance — this was
the first and only inference run that executed. All four answers are also factually correct,
which is a pleasant observation but **not** a measurement: 4 prompts at `gen_length=64` say
nothing about benchmark accuracy. The prompts are easy and deterministic decoding was used.
No failures occurred, so there are none to describe.

## 3. One-step LoRA training — ✅ PASS

Starting from the pristine downloaded checkpoint, a disposable LoRA adapter was attached and
**exactly one** optimizer step was taken on a tiny benign fixture.

> **Scope limit, stated plainly:** this is a **LoRA feasibility** result. It does **not**
> establish that full-parameter MDU training fits our hardware. Only 8.39 M of 8.02 B
> parameters (**0.1045%**) were trainable. As a rough order of magnitude, a conventional
> four-array FP32 accounting (parameters + gradients + two AdamW moments, 16 bytes/param) is
> **≈128 GB** at 8.02 B parameters; the three-array subset (FP32 master + two moments,
> 12 bytes/param) is **≈96 GB** — either way, far above one A100-40GB, and before MDU's frozen
> anchor copy and activations. **FP32 master weights and FP32 moments are not automatic in
> every BF16 AdamW configuration**, and real usage depends on state precision, gradients,
> activations, the anchor model, sharding and offload. This rough calculation establishes only
> that the question is **unaddressed** by this LoRA run — **not** exact multi-GPU feasibility
> either way. See the asset matrix §3.

**Configuration (as executed).** LoRA `r=8`, `lora_alpha=16`, `lora_dropout=0.0`,
`init_lora_weights=True` (PEFT default: `lora_A` Kaiming-uniform, **`lora_B` zeros**).
Optimizer AdamW, `lr=1e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.0`. Base weights bf16;
LoRA parameters cast to **fp32** for a stable step; cross-entropy computed in fp32.
Batch size **1**, sequence length **128**. 256 LoRA tensors across 128 target modules.

**LoRA targets chosen by inspecting real module names**, not guessed. Enumerating the 291
tensors in `model.safetensors.index.json` gave the per-block leaves
`q_proj, k_proj, v_proj, attn_out, ff_proj, up_proj, ff_out, attn_norm, ff_norm`.
Targets were `["q_proj","k_proj","v_proj","attn_out"]`.
**`ff_out` was deliberately excluded:** the name is ambiguous — it occurs both as the 32
per-block MLP output projections *and* as the top-level LM head
(`model.transformer.ff_out.weight`), so a PEFT suffix match on `ff_out` would also have wrapped
the LM head. This was identified and recorded in the manifest *before* execution.

**Loss follows the author's masked-response SFT recipe** from the pinned `GUIDELINES.md` §SFT:
`forward_process` with `eps=1e-3`, `t~U(0,1)`, `p_mask=(1-eps)t+eps`; prompt positions restored
after noising so **prompt tokens stay uncorrupted**; `masked_indices = (noisy_batch == 126336)`;
`token_loss = CE / p_mask`; `ce_loss = Σ(token_loss / answer_lengths) / batch`, with
`answer_lengths` = count of non-prompt positions **including padded EOS**; answer padded with
EOS (126081) to the sequence length.

*Details reconstructed from the guidelines rather than supplied training code* — LLaDA ships a
loss recipe, not a runnable trainer, so the following were our choices:
- The published snippet has variable-name typos (`prompt_length` vs `prompt_lengths`,
  `answer_length` vs `answer_lengths`); we implemented the evident intent.
- Padding the single-example answer region with EOS to exactly 128 tokens, and treating those
  padded positions as supervised, follows the guidelines' data-format example but the specific
  sequence length is ours.
- A dedicated CUDA `torch.Generator(seed=1234)` was used for the noising so the mask is exactly
  reproducible and independent of PEFT's initialization RNG.
- Casting LoRA params to fp32 and computing CE in fp32 is ours (numerical stability).

### Verification gates — 5/5 passed

| Gate | Result | Evidence |
|---|---|---|
| **1. Finite loss and gradients** | ✅ | loss `1.2453e-04` finite; **all 256** trainable grads finite; grad-norm `6.4918e-04` finite |
| **2. ≥1 trainable param changed, finite & nonzero** | ✅ | **128 of 256** changed; max abs delta `9.9926e-05` (≈ `lr`, as expected for AdamW's first step) |
| **3. Base params frozen and unchanged** | ✅ | **291/291** base tensors byte-identical; **0 drifted**; 0 base tensors with `requires_grad` |
| **4. Optimizer membership == intended trainable set** | ✅ | set equality of `id()` over `param_groups` vs `requires_grad=True`: 256 == 256 |
| **5. Exactly one successful update** | ✅ | `opt.step()` called once; AdamW internal per-param `step` counters == `[1]` |

**Gate 2 — the 128 unchanged tensors are the legitimate zero-init case, confirmed by name.**
All **128 changed tensors are `lora_B`**; all **128 unchanged are `lora_A`**. With `lora_B = 0`
at init, the gradient w.r.t. `lora_A` is `Bᵀ·upstream = 0`, so `A` cannot move on step one while
`B` receives a nonzero gradient. This is exactly the expected first-step behaviour, not a
failure — 128 nonzero gradients were observed out of 256 trainable tensors.

**Gate 3 — method and coverage stated explicitly.** For *every* non-LoRA parameter tensor
(**291 of 291 — 100% coverage**), SHA-256 was computed over its raw float32-cast bytes before
the step and again after, one tensor at a time. Zero digests changed. This is a byte-exact
equality check over the entire base model, not a sampled or norm-based comparison, and it was
paired with a `requires_grad` audit confirming no base tensor was trainable.

**Supervised mask — a caveat worth stating.** The gate required a non-empty supervised mask and
an uncorrupted prompt; both held (prompt region byte-identical, 0 masks leaked into the prompt).
But the composition matters: with `p_mask = 0.1281` over a 107-token answer region,
**17 positions were supervised, of which only 2 were real answer tokens** (`' sky'`, `' blue'`)
and **15 were padded EOS**. This was re-derived exactly from the recorded seed
([`evidence/supervised_mask_breakdown.json`](evidence/supervised_mask_breakdown.json)).
Padded EOS is trivially predictable, which explains the very low loss (`1.2e-4`). The gate is
still legitimately passed — the step is mechanically correct, gradients are finite and nonzero,
and one update occurred — but this run demonstrates **trainability plumbing, not learning
signal**. An 8-token answer in a 128-token window is padding-dominated; any real fine-tuning
should pack sequences or weight away from pad.

**Retry accounting.** The LoRA step **passed on its first attempt** — the single permitted
technical retry was **not** consumed here, and no configuration was changed. One earlier
failure occurred in the *inference* phase (see §6). No additional training was performed to
improve the result.

**Adapter kept locally**, not uploaded (base weights were never uploaded either):
`/data/bijaypandey/llada_smoke/artifacts/lora_adapter_one_step/` — `adapter_model.safetensors`
sha256 `b26a81f0aed1b0aa622478a2ff8acc3e3e63d668f677b33052bcd8c0fe8fd187` (33 MB),
`adapter_config.json` sha256 `0bc2370b27d4465dfc354baa09f19a912e305ac7a00e21da4e28bf41e0cee79f`.

Peak GPU memory **17.066 GiB** on one A100-40GB — comfortable headroom at this scale.

## 4. Benchmark asset readiness — ⚠️ PARTIAL

Full detail, including the required matrix, is in
**[`BENCHMARK_ASSET_MATRIX.md`](BENCHMARK_ASSET_MATRIX.md)**. MDU pinned at
`leegeoru/MDU` @ `e5e8e8274ff8f56f16e978803d9e902359eb5272` (MIT). Summary:

**Accessible now** — TOFU @ `324592d8` (all required splits incl. forget/retain plus
real-author and world-fact utility assets, 3.39 MiB, MIT); RWKU @ `d72f493d` (23 configs,
CC-BY-4.0); both LLaDA checkpoints; MDU source, configs and eval settings. All licenses are
permissive, and **no paid judge or API is required** — a grep for `openai`, `gpt-4`,
`anthropic`, `api_key`, `azure` and `requests.post` across MDU's `src/` and `scripts/` returned
**zero hits**; all metrics (ROUGE-L, Eq. 14 MC answer probability, truth ratio) are local.

**Requires preparation** — reshaping data into MDU's expected `./data/` layout; cloning and
pinning a `dllm` revision; resolving MDU's full-parameter two-model memory footprint against
40 GB devices.

**Missing / not verified — the blockers:**

1. 🔴 **No author-matched LLaDA TOFU starting checkpoint was located in the inspected release
   or the recorded search.** This was checked carefully. MDU's README states it fine-tunes each
   backbone on the full TOFU corpus (LLaDA: **1000 epochs**) and that *"the SFT code itself is
   not part of this release"* — so that code is **unavailable**; `run_main.sh` consumes the
   checkpoint from the placeholder path `LLADA_BASE_SFT=/path/to/LLaDA-TOFU-SFT-1000ep`.
   The corpus is **200 fictional authors × 20 QA pairs = 4,000 pairs** (verified:
   `evidence/tofu_row_counts.json`), making that SFT stage a substantial undertaking.
   **The ordinary Instruct checkpoint is not a substitute** — TOFU's facts concern authors
   invented for the benchmark, so `forget10` cannot be unlearned from a model that never
   encoded it; a metric computed against off-the-shelf Instruct would measure absence, not
   forgetting.
   Four third-party TOFU repos were found and their provenance recorded (none downloaded, no
   internals inspected): two undocumented SFT checkpoints **lacking declared license metadata**
   and **without model cards**, one already-unlearned model from a *different* method (LRCU),
   and one **empty repo** containing only `.gitattributes`. None is author-matched and none
   documents its epoch count. Absent license metadata leaves **permission and provenance
   unresolved** (not a legal determination). Both SFT candidates ship 4 safetensors shards where
   the official model ships 6 — **a shard-count difference alone does not establish
   incompatibility**, as shard layout is a packaging choice. **None is an exact reproduction
   substitute and none is described as one here.**
2. 🔴 **MDU's `dllm` dependency is undeclared, and the exact revision it used remains
   unverified.** Both LLaDA entry points `import dllm`
   (`MDLMTrainer`, `MDLMSampler`, `dllm.utils.CollatorWrapper`, …) and the docstring expects to
   run from inside a `dllm` checkout, but `dllm` is absent from `requirements.txt`, not vendored,
   and never given a URL or revision. `github.com/ZHZisZZ/dllm` (Apache-2.0) provides every
   imported symbol, verified at HEAD `ca17675` — but this is an **inference, not a confirmation**;
   it was not cloned or run, and the revision MDU actually used is unknown.
3. 🔴 **The pinned MDU release contains no LLaDA RWKU preset or evaluator; its released RWKU
   path is Dream-specific.** MDU ships `rwku_dream` only — there is no `rwku_llada` preset and
   no `eval_rwku_llada.py`. A LLaDA RWKU comparison would require new code and would no longer
   be MDU's published setup.

## 5. Full baseline reproduction — ⛔ NOT ATTEMPTED

Not attempted, by instruction. No benchmark evaluation was run, no second large model was
downloaded, no TOFU backbone was trained, and no paid API was invoked. Independently of that
instruction, §4(1) means a faithful TOFU reproduction is **not currently possible from the
artifacts inspected here** — the SFT code is unavailable in the pinned release and no
author-matched starting checkpoint was located.

## 6. Failures and retries

| # | Phase | What happened | Charged | Resolution |
|---|---|---|---|---|
| 1 | Inference | `OSError: [Errno 28] No space left on device: '/home/bijaypandey/.cache/huggingface/modules'` — `transformers` writes its dynamic-module cache (needed for `trust_remote_code`) under `$HOME`, which is on the **100%-full root filesystem**. Failed after 3 s, before touching the GPU. ⚠️ **This attempt's own stdout/stderr was not preserved** — attempt 2 reused the same redirect target and overwrote it. Its existence, `rc=1`, 3 s duration and GPU charge **are** corroborated by `logs/bounded.log` (append-mode); the error text above survives only in the session transcript. See [`AMENDMENT_01.md`](AMENDMENT_01.md) Correction 1. | **3 s charged to the GPU ledger** (failed attempts are charged, not forgiven) | Redirected `HF_HOME`, `HF_MODULES_CACHE`, `XDG_CACHE_HOME`, `TMPDIR`, `TORCH_HOME` under `/data` via `scripts/env.sh`; attempt 2 succeeded in 18 s. |
| 2 | Publication | `git fetch`/`worktree add` in `~/FINMLResearch` failed: `fatal: write error: No space left on device`, `couldn't write refs/heads/....lock` — the existing checkout's `.git` is on the full root fs. | no GPU charge | Verified the existing checkout was left **untouched** (branch `research/audit-and-matched-effectiveness-protocol` still checked out, its 2 modified files and 2 untracked dirs intact, no stray worktree or branch created), then cloned `main` to `/data` and published from there. |

**This was the only technical failure before a successful training update, and it was in the
inference phase.** The LoRA training gate passed first time; its permitted retry was never used,
and no configuration was altered to obtain a passing result. Both phases ran under
`scripts/bounded.sh` wall-clock timeouts. Budgets were never extended or reset.

## 7. Resource ledger — actual use

Measured from task start (23:22:10), ledger frozen at `START_EPOCH` before GPU execution.

| Budget | Cap | **Actual use** | How obtained | Fraction |
|---|---|---|---|---|
| Elapsed | 3.00 h | **~0.38 h** | **measured** (wall-clock from frozen `START_EPOCH`) | 13% |
| Aggregate GPU | 2.00 GPU-h | **152 GPU-s = 0.042 GPU-h** (3 s failed + 18 s inference + 131 s LoRA) | **measured**, corroborated by `logs/bounded.log` | **2.1%** |
| CPU | 8.00 core-h | **~0.12 core-h** | ⚠️ **estimate with incomplete coverage** — see below | ~1.5% |
| Additional storage | 60 GiB | **~21 GiB** (14.94 GiB weights + ~2 GiB venv + repos/logs/artifacts) | **measured** (`du -sb`) | 35% |
| GPUs concurrent | 1 default / 2 max | **1** (`CUDA_VISIBLE_DEVICES=0` throughout) | **measured** | — |

**GPU-seconds include the failed attempt**, and the recovered `logs/bounded.log` corroborates
all three charges exactly (`rc=1 duration_s=3`, `rc=0 duration_s=18`, `rc=0 duration_s=131`;
each `charged_gpu=yes`). Per-job ceilings (1800 s each) were never approached; the longest job
ran 131 s.

**The CPU figure is not comprehensive metering.** **201.25 core-s were directly measured** by
`/usr/bin/time -v` on the **two successful GPU jobs** (inference 31.57 user + 12.52 sys = 44.09;
LoRA 76.53 + 80.63 = 157.16 — both corroborated by the recovered logs). The **~0.12 core-h
total adds an estimate** for work that was **never instrumented**: venv creation and pip
installs, 6-way parallel SHA-256 hashing, `curl` downloads, and publication. The failed
attempt's CPU time is also excluded (transcript-only, not in any preserved log). The total
should be read as an estimate with known gaps, not a measurement.

**What the enforcement actually covers — stated precisely.**

- **Enforced:** per-job wall-clock timeouts. `scripts/bounded.sh` wraps each GPU job in
  coreutils `timeout` (SIGTERM, SIGKILL after a 30 s grace) at 1800 s, and adds each attempt's
  wall-clock to a cumulative counter **including failures**. A single job could not exceed its
  ceiling.
- **Advisory only — not enforced:** the **aggregate** elapsed-time, GPU-hour, CPU-core-hour and
  storage limits. These were **not scheduler or cgroup quotas**. `scripts/ledger.sh` computed
  and reported them between phases, but nothing would have terminated the session had an
  aggregate been crossed mid-phase. CPU and storage were observed and reported, never capped.

Every budget finished far under cap, so the distinction was never load-bearing in this run — but
it must not be described as a hard cap.

**Other users' jobs were never touched.** GPUs 2 and 3 carried another user's processes
(PIDs 2059679, 2064108, 2064109, ~81–94% utilization) for the whole window. Idleness of GPU 0
was re-checked immediately before each allocation. Nothing was killed or interfered with.

All work, caches and temporary files were kept under `/data` (`HF_HOME`, `HF_HUB_CACHE`,
`HF_MODULES_CACHE`, `PIP_CACHE_DIR`, `TMPDIR`, `TORCH_HOME`), as the root filesystem is at 100%.

**Environment** — isolated venv, Python 3.10.12: torch 2.4.1+cu124 (matched to driver
550.144.03), transformers 4.49.0 (checkpoint authored under 4.46.3), peft 0.14.0,
accelerate 1.4.0, numpy 1.26.4. Full list: [`evidence/pip_freeze.txt`](evidence/pip_freeze.txt).
**Note:** MDU pins torch 2.7.0 / transformers 4.57.0 on an H200, so **this run did not exercise
MDU's own pinned environment.**

## 8. Preserved prior work

This task created a **new branch** (`research/llada-access-smoke-v1`, from `origin/main`) and a
**new results directory** (`results/llada_access_smoke_v1/`). No earlier research history was
reset, cleaned, overwritten or amended. The separate CPR visual-audit closeout and the ongoing
Hyperplane work on `research/audit-and-matched-effectiveness-protocol` are untouched, including
that branch's uncommitted working-tree changes (verified after the failed worktree attempt in
§6). No prior LLaDA artifacts existed to reuse — a search of the checkout and `/data` found
none, so this ran as genuinely fresh preparation.

## 9. Next recommendation (one)

**One possible route — not the only one, and not an already-qualified baseline:** run an
RWKU-style forget/neighbor evaluation on LLaDA by adapting MDU's Dream RWKU path, treating it as
our own variant rather than an MDU reproduction.

> **Precondition that must be satisfied first:** **pristine LLaDA must demonstrate recall of the
> selected target facts** before any experiment can measure forgetting. Without a positive
> recall baseline on the chosen entities, a low post-unlearning score is indistinguishable from
> the model never having encoded those facts — the same trap that rules out off-the-shelf
> Instruct as a TOFU backbone in §4(1).
>
> **This is offered as one candidate direction, not a decision.** The separate cloud method team
> will inform the next experimental choice. No adaptation work has been started.

Why this is a *plausible* candidate, from the findings above:

- It **sidesteps the TOFU blocker.** RWKU needs no SFT backbone (`mdu_rwku.yaml`:
  *"off-the-shelf, no SFT"*, forget-only, no retain set), so it avoids the unresolved TOFU
  checkpoint entirely — whereas any TOFU work must first re-derive a backbone from SFT code
  unavailable in the pinned release, with no reference artifact to validate against. Other
  routes exist (e.g. commissioning or sourcing a documented TOFU backbone, or evaluating
  Dream as published); this is not the only option.
- **Every input is already verified accessible:** RWKU @ `d72f493d` (CC-BY-4.0), the hash-verified
  LLaDA-8B-Instruct, MDU's `convert_rwku_to_tofu.py`, and local-only metrics with **no paid API**.
- **The engineering is now de-risked by this run:** native masked-diffusion generation works at
  15.0 GiB peak, and the author's masked-response loss plus a real optimizer step works at
  17.1 GiB — both far inside one A100-40GB.
- The cost is **honestly scoped**: the pinned MDU release ships no `rwku_llada` preset, so this
  is new code, and it must be reported as a LLaDA RWKU variant rather than a reproduction of
  MDU's published RWKU numbers (which are Dream-only).

Three prerequisites to settle first: establish the **pristine-LLaDA recall baseline** above;
pin a `dllm` revision explicitly (recording that *we* chose it, since MDU does not pin one);
and settle the memory question for whichever training mode is chosen — on the rough accounting
in §3, full-parameter AdamW plus MDU's frozen anchor copy is on the order of 96–128 GB of state
and so is **not** addressed by this run's 40 GB devices. That rough figure is not a feasibility
verdict either way: a real decision needs a measured profile under the intended precision,
sharding and offload configuration.

Not recommended as the next step: training a TOFU backbone ourselves (code unavailable in the
pinned release, 1000 epochs, unverifiable against any published artifact), or adopting a
third-party TOFU checkpoint as a stand-in (not author-matched, undocumented, with permission and
provenance unresolved).

---

## Evidence index

See [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) for every published file with its SHA-256, and for
artifacts that remain local-only. Amendment-01 corrections are listed in
[`AMENDMENT_01.md`](AMENDMENT_01.md).
