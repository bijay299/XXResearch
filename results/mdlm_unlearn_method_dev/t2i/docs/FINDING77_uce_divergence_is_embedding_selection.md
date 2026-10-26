# FINDING 77 — Our UCE port and the pinned upstream UCE compute **different edits**: 10.8–38.4 % of the edited weight norm apart, and the difference is **embedding selection**. But upstream, executed at these settings, **does not erase church at all** — and neither document's quality framing survives that.

**Date:** 2026-10-26 · **Experiments:** R4b (weights), R4c/R4d (checkpoints + generations)
**Rewritten 2026-10-26** against the raw artifacts and a CPU reference solve. Every numeric claim
below was recomputed from `outputs/r4b_uce_lamb*.json`, `outputs/r4c_uce_lamb*.json`,
`outputs/r4d_uce_lamb*.json` and the saved checkpoints. **Six ratios in the previous version were
wrong and are corrected in place, not annotated.** Full audit trail:
`docs/AGENT1_UCE_FINDINGS.md`.

**Supersedes:** `FINDING75` (retracted).
**Upstream EXECUTED:** `unified-concept-editing@7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`,
`trainscripts/uce_sd_erase.py`, `UCE(...)` called unmodified. Shim: binds module-level `device` and
`torch_dtype=torch.float32`, which upstream defines only inside its `__main__` block (L116-117).
Function body untouched.

---

## 0. READ THIS FIRST — three things that constrain everything below

**(a) Upstream's edit is real.** Measured per matrix against the pinned base UNet, upstream's
checkpoints differ from base by **5.36 % of the weight norm on average (9.79 % max), with 0 of 32
matrices bitwise identical**. `UCE()` writes into a `copy.deepcopy` (`uce_sd_erase.py:21-22, 82`)
and saves those tensors, so the `.safetensors` captures the edit correctly. **No "upstream" number
in R4b/R4c/R4d is measuring the base model.** The earlier worry that upstream had written an
unedited model is refuted.

**(b) Upstream's edit is nonetheless behaviourally inert here, and that is reproducible.** At
`erase_scale=1`, `guide=''`, one unexpanded prompt, upstream leaves church at **15/15** CLIP
accuracy at λ=0.1 **and** at λ=0.5 (its own default), with and without a preserve pool. This is not
an invocation bug — it **replicates our own earlier single-token experiment**, recorded in
`erase_uce.py:concept_dirs` as "changed the weights by just 5.7 % and did not erase at all (church
stayed at CLIP acc 1.000)". Two independent code paths, five days apart, one result.

**(c) So "our port is not upstream UCE" cannot be read as a criticism of our port.** Our port's
`last..76` selection is a **deliberate, measured method deviation** that is the reason it erases at
all. Nor can it be read as a vindication: our port's edit is **4.0× larger** than upstream's and its
wrong predictions scatter across **five** different classes, which is equally consistent with
collateral damage. **No retain or utility axis was measured in R4c or R4d**, so the question is
open. See §5.

---

## 1. The weight-space decomposition (R4b)

All **32** edited `attn2` K/V matrices, every one compared. Upstream's output is reloaded from its
own `.safetensors`, so **upstream's** serialization precision is included; our port's is not —
R4b's "port" column is computed in memory in float32 and never touches `erase_uce.py`'s float16
`unet.pt`.

All four columns are **per-matrix max / per-matrix mean**:

| λ | **A** our selection vs upstream executed | **B** our solver on *upstream's* selection, vs upstream | **C** fp32 vs fp64, upstream selection | **C** fp32 vs fp64, **our** selection |
|---:|---:|---:|---:|---:|
| **0.1** | **0.38416 / 0.27546** | 1.5075e-03 / 7.9347e-04 | 1.0452e-03 / 5.6981e-04 | **6.2607e-02 / 3.9293e-02** |
| **0.5** | 0.32190 / 0.21733 | 3.0464e-04 / 1.6203e-04 | 2.1435e-04 / 1.1744e-04 | 1.3554e-02 / 8.5759e-03 |
| **1000** | **0.10841 / 0.07179** | **1.3264e-07 / 1.2733e-07** | 3.7215e-07 / 3.3913e-07 | 8.2463e-06 / 5.0892e-06 |

**Scope of every cell: `preserve_scale = 1.0`.** `erase_uce.py` defaults to **0.1**, and every
historical SD-1.5 λ-sweep arm — `uce_sd15_church_lam1000` included — used **0.1**. **No cell in
this table reproduces the configuration of any arm in our published tables.**

## 2. What the three columns do and do not isolate

**B does NOT test accumulation order.** `r4b_uce_upstream_vs_port.py:172-177`'s `solve()` uses
upstream's order (`mat1 = lamb*W` first) in **every** cell, so that factor is held fixed and never
varied. What B establishes, and it is still useful: **given identical embeddings and identical
accumulation order, our transcription of the closed form reproduces upstream** — there is no third,
hidden discrepancy in the algebra.

**Caveat on B that was previously absent:** B at λ=1000 (1.33e-07) sits **1560× below our own
float16 serialization floor of 2.07e-04** (measured by round-tripping upstream's fp32 result through
`.half()`). At λ=0.5, B (3.05e-04) is the *same size* as that floor. **Column B is a statement about
arithmetic and says nothing about any checkpoint we have ever shipped.**

**Embedding selection dominates A — stated as a bound, not a subtraction.** A and B are relative
Frobenius errors, so `A − B` is *not* a tensor difference and must not be quoted as one. But A and B
are distances to the **same** reference (upstream's reloaded checkpoint) with the **same**
denominator, so the triangle inequality gives a genuine two-sided bound:

> ‖solve(our selection) − solve(upstream selection)‖ / ‖upstream‖ ∈ **[A − B, A + B]**

At λ=1000 that interval is **[0.10841, 0.10841]**; at λ=0.1, **[0.38265, 0.38567]**. The bound is
tight only because B ≪ A. So the conclusion holds — within R4b's configuration, embedding selection
accounts for essentially all of column A — but it is a norm bound, and A and B are **not** two
additive "effects" whose ratio apportions blame.

| | selection | rows in `c` (concept `church`) |
|---|---|---|
| upstream (`uce_sd_erase.py:34-42`) | **one** embedding, at `attention_mask.sum()-2` | 1 |
| our `erase_uce.py:67-68` | positions `attention_mask.sum()-2 … 76` | **76** = 1 content token + **1 EOS** + 74 padding |

**C measures float32 error instead of inferring it — and it is λ-dependent.** The previous version
reported "float32 is accurate to 3.7e-07 under upstream's selection" without naming λ. That value
is **λ=1000 only**; at λ=0.5 it is 2.14e-04 and at λ=0.1 it is **1.05e-03**, i.e. **2800× worse**.

**C is also a realised error, not a bound.** It is one float32 solve against one float64 solve on
one input, one guide, one concept, one BLAS path. It is a *lower* estimate (the float64 reference
carries its own error) and bounds nothing above. The word "bound" has been removed here and in
`r4b_uce_upstream_vs_port.py`. (`FINDING75` §6 row 4 was right that a condition number bounds rather
than measures; this document then committed the mirror-image error by calling a measurement a
bound.)

**C does expose a real cost of our selection — correctly quantified this time.** Under our
selection the float32 error is **6.2607e-02 at λ=0.1**. Previously this was said to be "about 60×
worse than the full-pipeline difference at λ=1000" and "~60,000× worse than upstream's selection at
the same λ". **Both were wrong:**

* vs A at λ=1000 (0.10841): the ratio is **0.58×** — our fp32 error is **1.73× *smaller*** than
  that, not 60× worse. The comparison was also between two different quantities at two different λ
  and should not have been made.
* vs C(upstream selection) at the **same** λ=0.1 (1.0452e-03): the ratio is **59.90×** — i.e.
  **~60×**, not ~60,000×. A factor-of-1000 arithmetic error.

The mechanism is nevertheless real and is now measured directly. `[CPU]` on the pinned SD-1.4:

| selection | rows | rank of `cᵀc` | max eig `cᵀc` | cond(`cᵀc`+λI), λ=0.1 | λ=0.5 | λ=1000 |
|---|---:|---:|---:|---:|---:|---:|
| upstream, 1 token | 11 | 11 | 2.2171e+03 | 2.2172e+04 | 4.4353e+03 | 3.2171e+00 |
| ours, `last..76` | 826 | **136** | **3.2420e+05** | **3.2420e+06** | 6.4841e+05 | 3.2520e+02 |

Stacking the trailing positions raises cond by **146×**, which tracks the 59.9× ratio in realised
float32 error. Both systems remain **massively rank-deficient** (rank 11 and 136 in 768 dimensions),
so λ is regularising null directions in both cases, not conditioning a full-rank system. So
conditioning *is* real, it *is* a consequence of our embedding choice rather than of λ alone, and at
λ=0.1 it contributes **6.3e-02 of the 3.8e-01 total** — roughly one sixth, not a negligible
second-order term.

## 3. Accumulation order — **measured** on our GPU path, and it is a red herring

The previous version left this "open" and "unmeasured". The measurement already existed in our
artifacts, uncompared: R4b's column A uses **upstream's** order in memory; R4c's checkpoint
comparison runs **`erase_uce.py`'s real code path** (λ·W last, fused GEMMs) and round-trips through
float16.

| λ | R4b column A (upstream's order, fp32 in memory) | R4c checkpoint comparison (our port's own order, fp16) | difference |
|---:|---:|---:|---:|
| 0.1 | 0.384157 / 0.275459 | 0.384272 / 0.275521 | **1.14e-04 / 6.2e-05** |
| 0.5 | 0.321904 / 0.217332 | 0.321940 / 0.217367 | **3.6e-05 / 3.4e-05** |

Switching accumulation order **and** quantizing to float16 together move the result by
**≤ 1.2e-04 relative** — inside the 2.07e-04 float16 budget alone. **On the GPU path every one of
our historical arms ran, accumulation order is negligible against the 0.22–0.28 selection effect.**
`FINDING75`'s accumulation-order hypothesis is **refuted for our runs**.

**Two qualifications, both material.**

1. **This is not a hardware-independent claim.** `[CPU]` the same comparison gives **5.63e-02** at
   λ=0.1 — 500× larger. The realised perturbation of an ill-conditioned float32 solve is
   BLAS-dependent, so neither number generalises.
2. **"λ first vs λ last" is not a distinguished mechanism.** `[CPU]` at λ=0.1 under our selection,
   reordering the λ term moves the result by 5.63e-02 and dropping to float64 moves it by
   4.61e-02 — *the same size*, and both are what cond = 3.24e+06 against float32 ε predicts. There
   are also **four** arithmetic differences between the two codebases, not one: the λ position;
   upstream's 11 sequential rank-1 updates vs our single fused GEMM (`[CPU]` 9.29e-04, the same
   size as the λ-order effect); the point at which `preserve_scale` is applied (after the outer
   product upstream, to the intermediate in ours, by Python's left-associativity); and
   `torch.inverse` vs `torch.linalg.inv` (immaterial, same LAPACK path).

`erase_uce.py --diag-order-probe` and `r4c_uce_checkpoints_and_images.py --order-probe` now exist,
both default-off, to settle this on GPU. Neither has been run.

## 4. The generation comparison (R4c / R4d) — and its defects

| artifact | n_preserve (upstream) | λ | BASE | UPSTREAM | OUR_PORT | agree | checkpoint rel max / mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r4c_uce_lamb0.1.json` | 10 | 0.1 | 15/15 | **15/15** | **0/15** | 0/15 | 3.8427e-01 / 2.7552e-01 |
| `r4d_uce_lamb0.1.json` | 0 | 0.1 | 15/15 | **15/15** | **0/15** | 0/15 | 3.8421e-01 / 2.7560e-01 |
| `r4c_uce_lamb0.5.json` | 10 | 0.5 | 15/15 | 14/15 | 2/15 | 2/15 | 3.2194e-01 / 2.1737e-01 |
| `r4d_uce_lamb0.5.json` | 0 | 0.5 | 15/15 | **15/15** | 2/15 | 2/15 | 3.2160e-01 / 2.1740e-01 |

**These artifacts carry three defects. Cite them only with these caveats.**

**(i) The CLIP label set is malformed.** `r4c_uce_checkpoints_and_images.py:109` used
`labels = list(gen_eval.CONCEPTS)` — the dict **keys** — instead of the project's shared
`gen_eval.LABELS` / `LAB_KEY` (identical in `attack_relearn.py:83-87`). **9 of 12** labels differ,
so the CLIP text prompts were "a photo of a garbage_truck", "a photo of a van_gogh", "a photo of a
english_springer", "a photo of a monet". `church` is index 0 and spelled the same in both lists, so
the **target** accuracies above are the least-affected quantity, but the competing logits came from
a corrupted set and the `pred` fields are **not interpretable**. **These accuracies are not
comparable to any other H number in this project.** Repaired in source.

**(ii) The port column of R4d is not a measurement.** The script keyed the port checkpoint directory
on `(concept, lamb)` only and skipped the rebuild whenever `unet.pt` existed, and `--no-preserve`
was **never passed through to `erase_uce.py` at all** — `erase_uce.py` exposes no flag to empty its
`PRESERVE` pool. Filesystem evidence: `adapters/R4C_port_church_lam0.1` is stamped 12:24:17, before
R4c's output (12:25:26), and is **unchanged** when R4d wrote at 12:45:04. So **R4d reused R4c's port
build, made with the default 10-concept `PRESERVE`**. `r4d_uce_lamb*.json`'s `"n_preserve": 0` is
**false for the OUR_PORT arm**. The OUR_PORT column being identical across R4c and R4d is
**tautological — it is the same file** — and must not be cited as evidence about the preserve pool
in either direction. Repaired: the path now encodes every parameter and the skip-if-exists guard
verifies a recorded fingerprint.

**(iii) R4c's upstream checkpoint no longer exists.** R4c and R4d also shared
`outputs/r4c/upstream_{concept}_lamb{λ}.safetensors`, so R4d **overwrote** it. The surviving file
measures 0.0536 against base — R4d's 0-preserve value, not R4c's 0.0568. **R4c is not reproducible
from artifacts on disk and must be re-run.**

**What survives all three defects.** Upstream's arm *is* recomputed in-process on every run, so it
is untouched by the caching bug: **upstream retains 15/15 church at λ=0.1 and λ=0.5 with
`preserve_concepts=[]` genuinely empty.** And the 15/15-vs-0/15 direction is corroborated
independently of the label set, by `[CPU]` activation measurements (§5) and by the replication in
§0(b).

## 5. Why upstream is inert, and what that does to the comparison

`[CPU]` Upstream's exact algorithm re-run across settings. `dW/W` is the mean relative weight change
against base; `dKV/KV` is the mean relative change in the cross-attention K/V **output** for the
eval prompt "a photo of a church":

| upstream configuration | dW/W | dKV/KV |
|---|---:|---:|
| `erase_scale=1`, `guide=''`, 1 prompt, 10 preserve (= R4c) | 0.0568 | 0.4354 |
| `erase_scale=1`, `guide=''`, 1 prompt, 0 preserve (= R4d) | 0.0536 | 0.4335 |
| `erase_scale=1`, `guide=' '`, 1 prompt, 0 preserve | 0.0536 | 0.4335 |
| `erase_scale=2` (upstream's README object example) | 0.0536 | 0.4336 |
| `erase_scale=5` | 0.0536 | 0.4337 |
| `erase_scale=10` | 0.0536 | 0.4338 |
| **`expand_prompts=true`** (6 prompts), `erase_scale=1` | **0.0925** | **0.4745** |
| `expand_prompts=true`, `erase_scale=2` | 0.0934 | 0.4757 |
| *our port, λ=0.5, preserve_scale 1.0 (measured checkpoint)* | *0.2164* | *0.5872* |

1. **`erase_scale` is completely inert** — 1 → 10 leaves the edit at 0.0536. It multiplies *both*
   the numerator and denominator erase terms (`uce_sd_erase.py:70-71`), so with one erase concept
   and λ already negligible it cancels. This is the **same** root cause as
   `ERRATUM_uce_lambda_scale.md`'s λ-inertness, and it is why upstream's λ=0.1 and λ=0.5
   checkpoints differ from each other by only **~5e-5 relative**.
2. **`guide=''` and `guide=' '` are the identical edit** — both tokenize to `[BOS, EOS]`, so
   `attention_mask.sum()-2 = 0` selects the BOS hidden state either way. R4 used `" "`; R4b/R4c/R4d
   use `""`; immaterial.
3. **`expand_prompts` is the only upstream lever that moves anything — and it is unreachable.**
   Upstream applies it in `__main__` (`:155-190`) *before* calling `UCE()`, so a harness that
   imports `UCE()` directly can never exercise it. It raises the edit 1.7× and is still **2.3×
   below our port**.
4. **The "one position" mechanism in `erase_uce.py` is wrong and is withdrawn.** That comment said
   editing a single position "leaves the concept reachable" because the trailing positions carry
   the summary. `[CPU]` upstream's rank-1 edit changes the K/V output at the **padding** positions
   by **42 %** — a rank-1 weight update acts on every position's embedding, not only the selected
   one. Why 76 rows erase and 1 row does not is therefore **an open question, not an explained
   one**. The two arms differ by only **1.35×** in activation space (0.4335 vs 0.5872) yet by
   15/15 vs 0/15 in behaviour, which is a standing reason to distrust weight-norm inferences about
   behaviour — including the ones in this document.
5. **We are running upstream in a regime its authors never demonstrate.** The README's SD examples
   are **art-style** erasure (`--concept_type art` → `guide='art'`, two concepts, preserve pool);
   its only object-erasure example sets `--erase_scale 2`; its HiDream example sets
   `--expand_prompts true`. (Its moderation example passes `--concept_type 'unsafe'`, which is not
   in its own argparse `choices=['art','object']`.) `UCE()`'s every argument **is** bound by our
   shim — there are no `with_to_k`/`with_to_v`/`technique` flags in this file, and `concept_type`
   never enters `UCE()` — and both sides edit the **same 32** matrices. The invocation is
   API-faithful; the *configuration* is one the authors do not demonstrate for this task.

## 6. What this means for our record

- **Our UCE port is a modified UCE, and the modification is the point.** It diverges from upstream
  by **10.8–38.4 %** of the edited weight norm (per-matrix max; **7.2–27.5 %** mean) because it
  edits using 76 embeddings per concept where upstream uses one — and `[CPU]` that is why it erases
  at all, since upstream's single-token edit leaves church at 15/15, exactly as our own
  single-token version did.
- **The divergence is largest at small λ** (38.4 % at 0.1, 10.8 % at 1000), so the λ=0.1 arms are
  furthest from upstream. That is **9 of our 22** historical UCE arms, not 13 — 13 is the number of
  surviving `adapters/uce_*` **directories**, a different quantity. (Across 22 arms there are
  **10 distinct λ values**: 0.1×9, 0.5×1, 1.0×5, 10, 30, 100, 300, 1000, 10000, 50000.)
- **`erase_uce.py`'s docstrings claimed this selection matched the official implementation.** The
  module docstring was corrected earlier; the surviving false sentence inside `concept_dirs` has now
  been deleted too. Behaviour unchanged, so every historical arm still reproduces.
- **At λ=0.1 our port's own float32 error is 6.3e-02**, because its `cᵀc` has cond 3.24e+06. Any
  O(ε) implementation detail moves its edited weights by ~5 % of their norm on CPU. The λ=0.1 arms
  are reproducible to roughly one significant figure in weight space **on a fixed BLAS path**, and
  less than that across machines.
- **`FINDING75` is superseded.** Its 16 % came from character-split inputs and two local
  transcriptions. But its §2 conditioning argument was **withdrawn partly for the wrong reason** —
  see §7.
- **`venv_uce` violates every pin in upstream's `requirements.txt`** (diffusers 0.31.0 vs ≥0.34.0,
  torch 2.4.1 vs ≥2.6, transformers 4.46.3 vs ≥4.48.0, safetensors 0.8.0 vs ==0.5.3) and is a
  byte-identical copy of `venv_t2i`. This is deliberate and recorded in
  `setup_upstream_envs.sh:24-25`, justified there by "trainscripts import cleanly". That is a weaker
  test than "computes the same numbers". The one thing that could have invalidated columns A and B
  *was* checked: in diffusers 0.31, `encode_prompt` passes `attention_mask=None` for this text
  encoder, so upstream's embedding extraction and our port's `te(ids)[0]` are the **same function**.
  Unverified for diffusers ≥ 0.34.

## 7. Correction to this document's own treatment of `FINDING75`

The previous version said `FINDING75`'s conditioning inference was "wrong by ~6 orders of
magnitude". **That comparison changed two variables at once and is withdrawn.** `FINDING75`'s
argument was made at **λ=0.1** about **our port's** system; the rebuttal number 3.72e-07 is
**λ=1000, upstream's selection**.

| matched comparison | ratio | orders |
|---|---:|---:|
| `FINDING75`'s 1.5896e-01 vs C(**upstream** sel, **λ=0.1**) = 1.0452e-03 | 152× | **2.2** |
| `FINDING75`'s 1.5896e-01 vs C(**our** sel, **λ=0.1**) = 6.2607e-02 | **2.54×** | **0.4** |
| `FINDING75`'s 1.5896e-01 vs C(upstream sel, λ=1000) = 3.7215e-07 | 4.27e5 | 5.6 ("~6") |

**Against the system `FINDING75` was actually describing, its estimate was within a factor of 2.5.**
Its conditioning argument was approximately right about our port and wrong only about upstream's
rank-1 system. What remains genuinely invalid in `FINDING75` §2 is narrower: a condition number
*bounds* error propagation rather than measuring it, and the 1.59e-01 it was explaining was itself a
void char-split artifact. Those stand. "Six orders of magnitude" does not.

Relatedly, `ERRATUM_uce_lambda_scale.md:41` records max eig(`cᵀc`) = **85,850**; `[CPU]` my
measurement of our port's own stack gives **3.2420e+05**, 3.8× larger, making `FINDING75`'s derived
`cond ≈ 858,510` an **under**estimate of the measured 3.2420e+06. Flagged for that document's owner;
the inertness conclusion is unaffected and if anything strengthened.

## 8. What this does **not** establish

- **Not which implementation is better, in either direction.** Column C is each one's own realised
  float32 error; it does not adjudicate the modelling choice. Using the trailing positions is a
  *different edit*. Upstream's 15/15 is not evidence against UCE (it is outside the configurations
  the authors demonstrate); our port's 0/15 is not evidence for our port (4× the edit size,
  predictions scattered over five classes).
- **Not a utility or retain result.** R4c and R4d scored **the target concept only**, which
  PREREG_H38 §6 forbids. The collateral-damage reading of our port's 0/15 therefore **cannot be
  excluded**. This is the single most important missing measurement.
- **Not a held-out-prompt result.** DEV prompts only, n=15 per arm. `FINDING65` established that
  erasure does not survive held-out prompts, so these are an upper bound on erasure.
- **Not an attack-resistance difference.** Nothing here touches robustness, and `FINDING64`'s UCE
  attack results stand as measurements of **our port**, unchanged.
- **Not matched to any historical arm.** Every R4b/R4c/R4d cell used `preserve_scale = 1.0`; our
  port's default and every SD-1.5 λ-sweep arm used **0.1**. The λ=1000 row is additionally SD-**1.4**
  where `uce_sd15_church_lam1000` is SD-**1.5**.
- **One concept (church), one guide (`''`), SD-1.4.**

## 9. Status

UCE remains a **project reimplementation**, and after R4b/R4c/R4d that label is better understood:
it is a reimplementation with **one deliberate, consequential, empirically motivated deviation**
(76-row vs rank-1 erase subspace) which is the difference between erasing and not erasing on this
task, plus a float16 serialization floor of 2.07e-04 under every checkpoint.

**Open and needed to close this, none of it run (project frozen):**

1. **Re-run R4c and R4d** with the repaired script — distinct fingerprinted cache keys, the shared
   `LABELS`/`LAB_KEY`, a retain axis, saved image panels. R4c's upstream checkpoint no longer
   exists, so R4c is currently unreproducible.
2. **A retain/utility axis**, to decide §0(c): targeted erasure or collateral damage.
3. **One upstream cell with expanded prompts** (the only lever that moves its edit) to test whether
   upstream erases church *at all* on this task before any fidelity claim is made.
4. **One cell at `preserve_scale = 0.1`**, to produce an R4 row matching a real historical arm.
5. **The GPU order probe**, to resolve the 500× CPU/GPU discrepancy in §3.
