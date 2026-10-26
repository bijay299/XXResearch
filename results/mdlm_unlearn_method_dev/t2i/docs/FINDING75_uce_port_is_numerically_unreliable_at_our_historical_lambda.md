> # ⚠⚠ THIS FINDING IS RETRACTED. Its central number does not measure what it claims.
>
> **Verified 2026-10-26.** `r4_uce_official_vs_port.py` had three defects, each confirmed in the
> saved artifacts:
>
> 1. **The inputs were character-split.** `list(euce.FORGET_PROMPTS.get(c,[c]))` — the dict values
>    are **strings**, so `list('church')` gave `['c','h','u','r','c','h']`. R4 embedded **six single
>    letters** as six concepts. The artifact records `n_erase_prompts: 6`, which is the bug visible
>    in its own output.
> 2. **Neither pipeline was reproduced.** Upstream selects **one** embedding at
>    `attention_mask.sum()-2`; our port selects positions **`last..76`**; R4 flattened **all 77**.
>    R4 used a third procedure belonging to neither.
> 3. **No upstream code ever ran.** R4 compared **two local transcriptions**, so it could not
>    detect a divergence between our port and the authors' implementation — which is the entire
>    claim the title makes.
>
> **Every quantitative statement below is therefore withdrawn**, including "16%", "5.36e-03",
> "858,510", and the §3 conclusions about which arms are trustworthy. Four further errors are
> corrected in §6. The artifacts are preserved and marked `VOID`. The replacement is
> `r4b_uce_upstream_vs_port.py`, which **executes the pinned upstream `UCE(...)`**.

# FINDING 75 — Our UCE port is **not numerically equivalent** to the official implementation, and at our historical **λ = 0.1 the discrepancy is 16%** of the edited weight norm. The formula is right; the regime is not.

**Date:** 2026-10-26 · **Experiment:** R4 (`UPSTREAM_FIDELITY_MANIFEST.md` §6)
**Official reference:** `unified-concept-editing@7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`,
`trainscripts/uce_sd_erase.py`
**Evidence:** `outputs/r4_uce_official_vs_port.json`, `outputs/r4_uce_lamb{0.5,0.1}.json`,
`outputs/r4_uce_erasescale2.json`, `logs/r4_uce.log`

---

## 1. The measurement

Both closed forms were evaluated on **identical inputs** — same pinned SD-1.4, same conditioning
embeddings, same `W_old`, same 160 `attn2` K/V projections — and the edited matrices compared:

| config | max relative error | mean | equivalent? |
|---|---:|---:|---|
| λ = **0.5** (official default), `erase_scale` 1.0 | **5.36e-03** | 3.09e-03 | **No** |
| λ = **0.1** (**our** historical default), `erase_scale` 1.0 | **1.59e-01** | 9.48e-02 | **No** |
| λ = 0.5, `erase_scale` **2.0** (divergence probe) | 5.40e-02 | 4.18e-02 | No |

**At the λ every historical UCE arm of ours used, the two implementations disagree by ~16% of the
edited weight norm.** My pre-recorded expectation — that the two would be equivalent at
`erase_scale = 1` since our port omits that coefficient — **was wrong**.

## 2. The cause is **conditioning**, not a formula error

Our port's algebra was verified previously (`ERRATUM_uce_lambda_scale.md` §1) and is confirmed again
here: the two expressions are term-for-term identical at `erase_scale = 1`. They differ only in
**summation order** — the official writes `mat1 = λW + erase + preserve`, ours writes
`num = erase + preserve + λW`.

Floating-point addition is not associative, so that reordering perturbs the matrices at rounding
level. A controlled reproduction isolates the amplification:

| λ | cond(den) | `‖den_off − den_our‖/‖den‖` | **⇒ `‖W_off − W_our‖/‖W‖`** |
|---:|---:|---:|---:|
| 0.5 | 4.13e+03 | **0.0** | 7.39e-05 |
| 0.1 | 2.07e+04 | **3.21e-08** | **1.68e-03** |
| 1000 | 3.06e+00 | 4.52e-08 | 1.53e-07 |

**The denominators agree to float-rounding precision; the inverses do not.** The error in `W` tracks
the condition number, and vanishes at λ = 1000 where the system is well conditioned (cond ≈ 3).

On the **real** system it is far worse than this synthetic bound. From the eigenvalues already
measured in `ERRATUM_uce_lambda_scale.md` (`cᵀc` max eigenvalue **85,850**):

> cond(`cᵀc` + λI) ≈ (85 850 + λ)/λ = **858 510 at λ = 0.1**, and 171 701 at λ = 0.5.

float32 carries ~7 decimal digits; a condition number of 8.6 × 10⁵ consumes ~6 of them. **One digit
of accuracy survives** — which is exactly the 16% discrepancy measured.

## 3. What this means for our existing UCE results

- **Every historical UCE arm of ours used λ = 0.1** — the regime where the edited weights are
  reproducible to about one significant figure. Those arms are **not wrong**, but they are **not
  reproducible across implementations**, and a different-but-equivalent transcription of the same
  equation lands ~16% away in weight space.
- This is the **second** consequence of the same root cause. `ERRATUM_uce_lambda_scale.md` found λ
  **inert** below ~1e3 because it is negligible against eigenvalues up to 85,850. The same
  mismatch makes the inverse **ill-conditioned**. One diagnosis, two symptoms: at λ ≪ the spectrum,
  the regulariser neither regularises nor conditions.
- **`uce_sd15_church_lam1000`** (target 0.2000 / retain8 0.9500, reversed 3/3 — `FINDING64`) sits at
  λ = 1000, where cond ≈ 86.9 and the implementations agree to ~1e-7. **It is the one UCE arm whose
  weights are numerically trustworthy**, and it happens to be the arm the λ-bracketing produced.
- **The official default is λ = 0.5, ours was 0.1.** "UCE at defaults" has meant different edits in
  the two codebases throughout this project.

## 4. What this does **not** show

- **Not a bug in our port.** The algebra matches; the transcription is faithful at `erase_scale = 1`.
- **Not a refutation of UCE.** A method evaluated outside its well-conditioned regime says nothing
  about the method. If anything this is a caution about *our* parameter choice.
- **Not a robustness result.** R4 compares weights only — no attack, no generation. Whether the 16%
  weight difference changes generated images or recovery is **unmeasured**.
- **`erase_scale` remains untested as a mechanism.** The 2.0 probe shows divergence (5.4e-2) but at
  λ = 0.5 that is entangled with the same conditioning effect, so it does not isolate the missing
  coefficient. Our port omits `erase_scale`; at the official default of 1.0 that omission is
  harmless *algebraically*, which is all that is established.

## 5. Status of the method label

UCE remains a **project reimplementation**, not an official reproduction. R4 was a numerical
comparison, not a run of the authors' script end-to-end. Running `uce_sd_erase.py` itself, and
comparing *generations* rather than weights, is the next step and has not been done.

---

## 6. Retraction detail — six specific errors

| # | claim in §1–§5 | status |
|---|---|---|
| 1 | "160 `attn2` K/V projections" | **WRONG — 32.** The artifact records `n_modules: 32`. SD-1.4 has 16 cross-attention blocks × {to_k, to_v}. |
| 2 | "**every** historical UCE arm of ours used λ = 0.1" | **WRONG.** The artifacts show **ten distinct λ values** across 22 arms — 0.1, 0.5, 1.0, 10, 30, 100, 300, 1000, 10000, 50000. 13 of 22 used 0.1; the λ sweep and the λ=1000 arm contradict the claim outright. |
| 3 | "the implementations differ **only** in summation order" | **WRONG.** They also differ in **embedding selection** — one token vs `last..76` — which is a far larger difference than accumulation order and was not controlled. |
| 4 | the conditioning analysis (cond ≈ 858,510 ⇒ "one digit survives") | **WITHDRAWN as evidence.** A condition number **bounds** error propagation; it does not measure it. The float64 reference needed to measure it was never run. R4b runs it. |
| 5 | "`uce_sd15_church_lam1000` … **the one UCE arm whose weights are numerically trustworthy**" | **UNSUPPORTED.** That arm's own inputs were never measured; the claim was extrapolated from a synthetic reproduction at a different scale. |
| 6 | implied: the discrepancy says something about our historical checkpoints | **NO.** Nothing was compared against a historical checkpoint. Zero checkpoints were loaded. |

**Also wrong and corrected elsewhere:** our `erase_uce.py` docstring says its embedding selection
is *"matching UCE's official implementation."* It is not — upstream takes one token, ours takes
`last..76`. Corrected in `UPSTREAM_FIDELITY_MANIFEST.md` §4.

## 7. What is actually known about UCE right now

**Nothing quantitative.** The only surviving factual content from this document is structural, and
it was verified by reading source, not by R4:

- upstream edits **32** `attn2` `to_k`/`to_v` matrices;
- upstream's `--lamb` default is **0.5**; our port's is **0.1**;
- our port carries **no `erase_scale`** coefficient, where upstream applies it to both erase terms;
- the two use **different embedding selections**.

Whether any of that changes edited weights, generated images, or attack resistance is **unmeasured**
until R4b completes. **A weight difference is not an image difference and is not attack
resistance** — and R4b is a weights-and-panel comparison, not a robustness experiment.
