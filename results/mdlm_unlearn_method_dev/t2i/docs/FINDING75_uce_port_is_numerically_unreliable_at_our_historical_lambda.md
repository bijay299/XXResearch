# FINDING 75 — RETRACTED. Its central measurement does not measure what it claimed.

**Retracted 2026-10-26. Rewritten 2026-10-26 (AGENT1)** so that no incorrect claim stands in the
body rather than only in a banner. Every number below has been recomputed from the raw artifacts or
measured on CPU against the pinned SD-1.4. Replacement: **`FINDING77`**. Full audit trail:
`docs/AGENT1_UCE_FINDINGS.md`.

**Original title (withdrawn):** *"Our UCE port is not numerically equivalent to the official
implementation, and at our historical λ = 0.1 the discrepancy is 16 % of the edited weight norm."*

**Original experiment:** R4 (`scripts/r4_uce_official_vs_port.py`).
**Artifacts:** `outputs/r4_uce_official_vs_port.json`, `outputs/r4_uce_lamb{0.5,0.1}.json`,
`outputs/r4_uce_erasescale2.json` — all four stamped `"VOID": true` with a `void_reason`, preserved
for provenance. **Do not cite them.**

---

## 1. Why the measurement is void — five defects, each verified in the artifacts

1. **The inputs were character-split.** `r4_uce_official_vs_port.py:68` —
   `list(euce.FORGET_PROMPTS.get(a.concept, [a.concept]))`. The dict **values are strings**, so
   `list('church')` gave `['c','h','u','r','c','h']`. R4 embedded **six single letters** as six
   concepts. Every artifact records `n_erase_prompts: 6`, which is the bug visible in its own
   output.
2. **Neither pipeline was reproduced.** Three different embedding selections exist. Upstream
   (`uce_sd_erase.py:34-42`) takes **one** embedding at `attention_mask.sum()-2`. Our port
   (`erase_uce.py:67-68`) takes positions `[attention_mask.sum()-2 .. 76]`. R4's `embed()` at
   `:71-75` flattened **all 77** positions. R4 used a third procedure belonging to neither.
3. **No upstream code ever ran.** R4 compared **two local transcriptions**, so it could not have
   detected a divergence between our port and the authors' implementation — the entire claim its
   title made.
4. **The headline artifact is the wrong run.** `:133` wrote a **fixed** filename with no config
   tag, so the three configurations overwrote one another. On disk,
   `r4_uce_official_vs_port.json` records `erase_scale: 2.0` and is a byte-equivalent duplicate of
   `r4_uce_erasescale2.json`, yet §2 below cited it for the `λ=0.5, erase_scale 1.0` row whose real
   home is `r4_uce_lamb0.5.json`. *(Repaired in source: the filename now carries a tag.)*
5. **An unsupported setting was silently compared.** `erase_uce.py` implements **no `erase_scale`**.
   At `--erase-scale 2.0`, `:93-95` applies it to the "official" side and `:99-100` omits it from
   "ours" — **two different equations** — and the resulting 5.3985e-02 was reported as a
   numerical-equivalence failure. *(Repaired in source: `erase_scale != 1` now refuses to run.)*

**Every quantitative statement in the original document is therefore withdrawn**, including "16 %",
"5.36e-03", "858,510", and the §3 conclusions about which arms are trustworthy.

## 2. The original numbers, and what each actually was

The original §1 table, re-sourced to the artifacts that really contain each row:

| config | artifact that actually holds it | max rel. error | mean | original verdict | status |
|---|---|---:|---:|---|---|
| λ=0.5, `erase_scale` 1.0 | `r4_uce_lamb0.5.json` | 5.3615e-03 | 3.0896e-03 | "No" | **VOID** (defects 1–3) |
| λ=0.1, `erase_scale` 1.0 | `r4_uce_lamb0.1.json` | 1.5896e-01 | 9.4748e-02 | "No" | **VOID** (defects 1–3) |
| λ=0.5, `erase_scale` **2.0** | `r4_uce_erasescale2.json` **and** `r4_uce_official_vs_port.json` | 5.3985e-02 | 4.1798e-02 | "divergence probe" | **VOID + category error** (defect 5) |

The headline "~16 %" was the λ=0.1 **max** (1.5896e-01); the mean was 9.5 %. The correctly measured
full-pipeline figure, from `FINDING77` R4b, is **38.4 % max / 27.6 % mean at λ=0.1** — larger than
R4 reported, in the opposite direction from a rounding error.

## 3. Six further specific errors in the original body

| # | original claim | status |
|---|---|---|
| 1 | "160 `attn2` K/V projections" | **WRONG — 32.** The artifacts record `n_modules: 32`. SD-1.4 has 16 cross-attention blocks × {to_k, to_v}. Upstream edits the same 32. |
| 2 | "**every** historical UCE arm of ours used λ = 0.1" | **WRONG.** `outputs/erase_*.json` with `study=="t2i_uce_reimpl"` shows **22 historical arms** across **10 distinct λ** (0.1×9, 0.5×1, 1.0×5, 10, 30, 100, 300, 1000, 10000, 50000). The λ sweep and the λ=1000 arm contradict the claim outright. **And the replacement figure first offered — "13 of 22 used 0.1" — is also wrong: it is 9 of 22.** 13 is the number of surviving `adapters/uce_*` **directories**, a different quantity. |
| 3 | "the implementations differ **only** in summation order" | **WRONG, and incomplete even as a list.** They also differ in **embedding selection** (1 token vs 76 rows), which `FINDING77` measures at 0.22–0.28 — far larger. And there are **four** arithmetic differences, not one: the λ position; upstream's per-concept rank-1 loop vs our single fused GEMM (`[CPU]` 9.29e-04 at λ=0.1, the *same size* as the λ-order effect); the point at which `preserve_scale` is applied; and `torch.inverse` vs `torch.linalg.inv` (immaterial). |
| 4 | the conditioning analysis (cond ≈ 858,510 ⇒ "one digit survives") | **PARTLY REINSTATED — see §4.** Withdrawn *as evidence* (a condition number bounds error propagation, it does not measure it; the float64 reference needed to measure it was never run — R4b runs it). But the inference was **approximately correct** about the system it described, and `FINDING77`'s original "wrong by ~6 orders of magnitude" rebuttal was itself a cross-λ error. |
| 5 | "`uce_sd15_church_lam1000` … **the one UCE arm whose weights are numerically trustworthy**" | **UNSUPPORTED.** That arm's own inputs were never measured; the claim was extrapolated from a synthetic reproduction at a different scale. R4b's λ=1000 row still does not match it: R4b ran SD-**1.4** at `preserve_scale = 1.0`, that arm is SD-**1.5** at `preserve_scale = 0.1`. |
| 6 | implied: the discrepancy says something about our historical checkpoints | **NO.** Nothing was compared against a historical checkpoint; zero checkpoints were loaded. Separately, every one of our checkpoints carries a **float16 serialization floor of 2.07e-04** (`erase_uce.py:124` saves `.half()`), which exceeds R4b's solver-agreement column at λ≥0.5 and is 1560× above it at λ=1000. |

**Also wrong and now fixed at source:** `erase_uce.py`'s module docstring said its embedding
selection was *"matching UCE's official implementation."* The module docstring was corrected on
2026-10-26; the **same false sentence survived inside `concept_dirs`** until it was deleted on
2026-10-26 (AGENT1 P1). Upstream takes one token; ours takes `last..76`.

## 4. The conditioning argument — partly reinstated, with the right numbers

The original §2 reasoned: `cᵀc` has max eigenvalue ~85,850, so cond(`cᵀc` + λI) ≈ (85 850+λ)/λ =
858,510 at λ=0.1; float32 carries ~7 digits; a condition number of 8.6e5 consumes ~6 of them; "one
digit of accuracy survives" — which it claimed matched the 16 % discrepancy.

**The method was invalid:** a condition number bounds error propagation, it does not measure it, and
the 1.59e-01 being "explained" was a void char-split artifact. **That withdrawal stands.**

**But the conclusion was approximately right about our port, and `FINDING77` originally dismissed it
with a cross-λ comparison.** `FINDING77` said the inference was "wrong by ~6 orders of magnitude",
using a λ=1000, upstream-selection number against a λ=0.1, our-selection argument. Matched:

| matched comparison (from `r4b_uce_lamb0.1.json`) | ratio | orders |
|---|---:|---:|
| original estimate 1.5896e-01 vs measured fp32 error, **our** selection, **λ=0.1** = 6.2607e-02 | **2.54×** | **0.4** |
| original estimate 1.5896e-01 vs measured fp32 error, **upstream's** selection, λ=0.1 = 1.0452e-03 | 152× | 2.2 |
| original estimate vs upstream selection at λ=1000 = 3.7215e-07 (the comparison `FINDING77` made) | 4.27e5 | 5.6 |

So: **against our port's own system at our own historical λ, the estimate was within a factor of
2.5.** It was wrong about upstream's rank-1 system, where λ=0.1 gives cond 2.22e+04 and a realised
float32 error of 1.05e-03.

**And the spectrum figure itself needs reconciliation.** `[CPU]` on the pinned SD-1.4, for concept
`church` plus the 10 `PRESERVE` prompts at our port's selection:

| selection | rows in `c` | rank of `cᵀc` | max eig `cᵀc` | cond(`cᵀc`+λI), λ=0.1 | λ=0.5 | λ=1000 |
|---|---:|---:|---:|---:|---:|---:|
| upstream, 1 token | 11 | 11 | 2.2171e+03 | 2.2172e+04 | 4.4353e+03 | 3.2171e+00 |
| ours, `last..76` | 826 | **136** | **3.2420e+05** | **3.2420e+06** | 6.4841e+05 | 3.2520e+02 |

The measured max eigenvalue of our port's `cᵀc` is **3.2420e+05**, **3.8× larger** than the 85,850
in `ERRATUM_uce_lambda_scale.md:41`, so the derived cond of 858,510 was an **under**estimate of the
measured **3.2420e+06**. (Flagged for that document's owner; the λ-inertness conclusion is
unaffected and if anything strengthened.) Both systems are **massively rank-deficient** — rank 11
and 136 in 768 dimensions — so λ regularises null directions rather than conditioning a full-rank
system, which the original analysis did not recognise.

**What this licenses, stated carefully.** At λ=0.1 our port's edited weights carry a realised
float32 error of **6.3e-02** (max; 3.9e-02 mean) against a float64 solve, which is roughly one
significant figure — the original document's qualitative conclusion. `[CPU]` the same system shifts
by **5.6e-02** if the λ term is merely reordered, and by 9.3e-04 if the preserve sum is done as a
loop instead of a GEMM. **At λ=0.1, every O(ε) implementation detail moves our edited weights by
~5 % of their norm.** The λ=0.1 arms are reproducible to about one significant figure in weight
space on a **fixed** BLAS path, and less than that across machines.

## 5. Status of the accumulation-order hypothesis — now **refuted** for our runs

The original §2 attributed the whole discrepancy to summation order. `FINDING77` §6 then said the
question was "unmeasured" and "open". **Both are superseded: the measurement existed in our
artifacts, uncompared.** R4b's column A uses upstream's accumulation order in memory; R4c's
checkpoint comparison runs `erase_uce.py`'s real code path (λ·W last, fused GEMMs) and round-trips
through float16:

| λ | R4b column A | R4c checkpoints (our port's own order, fp16) | difference |
|---:|---:|---:|---:|
| 0.1 | 0.384157 / 0.275459 | 0.384272 / 0.275521 | **1.14e-04 / 6.2e-05** |
| 0.5 | 0.321904 / 0.217332 | 0.321940 / 0.217367 | **3.6e-05 / 3.4e-05** |

Accumulation order **plus** float16 quantization together move the result by **≤ 1.2e-04** — inside
the 2.07e-04 float16 budget alone. **On the GPU path every historical arm ran, accumulation order is
negligible against the 0.22–0.28 embedding-selection effect, and the original §2 hypothesis is
refuted.**

Two qualifications: the bound is **not hardware-independent** (`[CPU]` the same comparison gives
5.6e-02), and "λ first vs λ last" is **not a distinguished mechanism** — it is one draw from a
family of equally-sized rounding perturbations in an ill-conditioned solve. Opt-in GPU probes now
exist (`erase_uce.py --diag-order-probe`, `r4c_uce_checkpoints_and_images.py --order-probe`); neither
has been run.

## 6. What is actually known about UCE now

The original §7 said "nothing quantitative". That is no longer true. Structural facts, verified by
reading source:

- upstream edits **32** `attn2` `to_k`/`to_v` matrices, the same 32 our port edits;
- upstream's `--lamb` default is **0.5**; our port's is **0.1**;
- our port carries **no `erase_scale`** coefficient, where upstream applies it to both erase terms —
  `[CPU]` and `erase_scale` is **inert** for upstream anyway (1 → 10 leaves its edit at 0.0536),
  because it multiplies numerator and denominator alike;
- the two use **different embedding selections** (1 token vs 76 rows), and this is a **deliberate
  method deviation** on our side, not a transcription error.

And measured, in `FINDING77`:

- the two implementations' edits differ by **10.8–38.4 %** of the edited weight norm (per-matrix
  max), essentially all of it from embedding selection;
- **upstream's edit is real** (5.36 % mean vs base, 0/32 matrices bitwise identical) but at these
  settings **does not erase church** — 15/15 CLIP accuracy at λ=0.1 and λ=0.5, which independently
  replicates our own earlier single-token experiment;
- so the original §4's "if anything this is a caution about *our* parameter choice" is **backwards**:
  our deviation is what makes the method erase on this task. Whether it does so by targeted erasure
  or by collateral damage is **unmeasured** — R4c/R4d scored the target concept only.

**A weight difference is still not an image difference and still not attack resistance.**
`FINDING64`'s UCE attack results stand as measurements of **our port**, unchanged.
