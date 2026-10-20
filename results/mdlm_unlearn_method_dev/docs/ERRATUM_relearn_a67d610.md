# ERRATUM — `research/relearning-robust-mdlm` @ `a67d610044d0`

Issued 2026-10-20 by the coordinator, after an independent engineering audit and two
independent evaluation reviews of that study. **The original commit is not rewritten.**
Its measurements stand as recorded; the items below correct claims built on them.

Severity order. Items 1 and 2 change what the study is entitled to claim.

---

## 1. The §3 conclusion "the vulnerability is directional, not positional" is WITHDRAWN

**As published:** *"Random perturbation of up to 2× the Frobenius norm of the entire
unlearning edit leaves disclosure at 0.031, while the same perturbation leaves a present
fact at 1.000. Recovery needs gradient information about the forget set, not parameter
displacement."*

**The defect.** `noise_probe.py:103` builds the perturbing LoRA with `lora_targets()`,
which is `["q_proj","k_proj","v_proj","attn_out"]` — 4 linears × 32 blocks = **128
modules**. The unlearning edit it was scaled against spans 7 linears × 32 blocks = **224
modules**. Measured energy distribution of that edit:

| module | energy fraction | perturbed? |
|---|---|---|
| ff_proj | 0.3157 | **no** |
| up_proj | 0.2402 | **no** |
| ff_out | 0.1573 | **no** |
| k_proj | 0.0907 | yes |
| q_proj | 0.0760 | yes |
| attn_out | 0.0696 | yes |
| v_proj | 0.0505 | yes |

So the probe took a **full-edit-norm** perturbation (‖ΔW‖_F = 11.586) and spread it over a
subspace containing **28.7%** of the edit energy — over-perturbing attention by **1.87×**
while leaving **71.3% of the edit entirely untouched.**

**Consequence.** The experiment does not test what it claimed. "A random perturbation of
2× the edit norm" was never applied; what was applied was a large perturbation confined to
attention.

### Re-measured with full 224-module coverage — the conclusion SURVIVES

Run the same day on the identical model and probe set
(`mdlm_unlearn/outputs/noise_unlearned.json`, 224 random LoRA modules matched to the edit):

| perturbation | relative to edit | forget disclosure (full coverage) | as published (attention-only) |
|---|---|---|---|
| 0.00 | — | 0.031 | 0.031 |
| 2.90 | 0.25× | **0.094** | 0.031 |
| 5.79 | 0.5× | 0.062 | 0.031 |
| 11.59 | 1.0× | 0.062 | 0.062 |
| 23.17 | **2.0×** | **0.062** | 0.031 |

Retain accuracy 1.000 and zero degenerate outputs at every level. Disclosure roughly
doubles relative to the published numbers (0.031 → 0.062, peaking at 0.094), but remains an
order of magnitude below the **0.605** a directed 10-step attack achieves on the same probe
set.

**So the original §3 conclusion is re-established on a valid measurement.** Recovery
requires gradient information about the forget set; undirected displacement of twice the
entire edit's norm does not produce it. What is withdrawn is the *published experiment's
standing as evidence* — it did not perform the test it described — and the specific
magnitudes, which were understated by about 2×. The conclusion itself is retained, now with
weaker margins and a valid probe behind it.

Over-retraction is also an error, so this is stated plainly rather than left ambiguous.

**What this costs.** The finding (labelled E4 internally) grounded a follow-on hypothesis
that the unlearned model sits a short hop from the memorized one. That hypothesis was
separately **refuted** by a direct measurement of per-fact gradient geometry — see
`FINDING_answer_channel.md` — so nothing now rests on E4 either way.

## 2. The headline "0.049 → 0.605" mixes denominators

**Corrected statement: 0.023 → 0.605.**

`0.049` came from the ARC study's evaluator: 128 forget probes (64 facts × 2 phrasings),
all facts. `0.605` came from this study: 43 held-out *recalled* facts × 1 phrasing. Placing
them in a before/after sentence implies one measurement series; they are two. The
within-study value on the **same 43 probes** at `k = 0` is **0.023**, which is printed in
the study's own curve table.

**The reversibility finding survives and is in fact sharper.** 0.605 = 26/43, Wilson CI
[0.456, 0.736], against a never-memorized floor of 0.023 measured on the identical probe
set. The static starting point being *equal to the floor* strengthens rather than weakens
the point: static disclosure was saturated at the floor and still the knowledge was
recoverable.

Found independently by two reviewers. `0.023 = 1/43` is also the quantisation unit of the
measurement, which matters for item 3.

## 3. The defense comparison was underpowered and could not have resolved its own question

The verdict **NOT SUPPORTED** is correct *against the stated rule* — Δ(k=10) = −0.023 vs a
pre-registered bar of ≤ −0.20. But Δ = −0.023 is **exactly one probe out of 43**, i.e. the
quantisation floor, and at n = 43 with 1 seed the minimum detectable difference from
p₀ = 0.60 is **0.264** — larger than the 0.20 bar the study pre-registered. The study could
not have detected a true effect of the size it was looking for, in either direction.

Stated plainly: the study establishes that **no large effect was present**, not that no
effect exists. The original report's §6 already declined to call this a refutation of Fan
et al.; this erratum adds the quantitative reason that declination was necessary.

Also under-reported in the original: **1 unlearning seed** (2 were pre-registered), a
single leak set chosen by deterministic stride, and no confidence intervals or fact-level
clustering anywhere.

## 4. The "39.9% reduction in ‖∇L_mem‖" is a subspace-restricted reduction

`flatten.py:116` uses the same `lora_targets()` 128-module set. The penalty therefore acted
on, and the frozen probe measured, the **attention-subspace projection** of the gradient,
not the full-parameter gradient that `theory/THEORY.md` L1 and P2 are stated about. The
reduction figure should read: *39.9% reduction in the attention-projection of ‖∇L_mem‖*.
Whether a penalty applied to the full 224-module gradient behaves differently is untested.

This weakens the negative result in a specific way worth naming: the mechanism may have
been applied to the wrong 29% of the edit. The negative conclusion about the *downstream
relearning curve* still stands as a measurement, but its attribution to "the gradient-norm
mechanism" is now partial.

## 5. The adversary was weaker than the defense it attacked

`relearn_probe.py:117` also uses `lora_targets()`, so the adversary's own parameters were
confined to the 128-module attention subspace. The AXIS-REAL gate passed by a wide margin
(+0.582 against a +0.25 bar), so the reversibility finding is robust to this; but the
adversary is not the strongest available, and a full-coverage adversary would be stronger.

## 6. Disclosure is under-counted by the scorer

`normalize()` applies NFKC + casefold, which preserves combining diacritics. Re-scoring all
1,648 stored generations with a diacritic-insensitive normaliser (NFKD + combining-mark
strip) finds **7 generations that disclose the answer but were scored as non-disclosing** —
all of the form `"Alira Greaves lives in Montréal"` against the reference `Montreal`.

0.4% of rows, and it errs **in the direction that flatters every arm**. Reported here
rather than silently fixed; the project's forward protocol adopts the diacritic-insensitive
normaliser and will report both scorings wherever it changes a number.

## 7. θ₀ was evaluated unmerged while every other arm was evaluated merged

`evaluate_big.py:24-26` attaches the memorized adapter for θ₀ but calls
`merge_and_unload()` before attaching for the arms. Merged and unmerged bf16 execution are
not numerically identical, so the θ₀ ceiling row carries an unmeasured confound. A direct
merged-vs-unmerged comparison is scheduled.

---

## What stands unchanged

- **Masked-diffusion unlearning is reversible.** 0.023 → 0.605 on 43 held-out facts after
  10 adversary steps on 8 leaked facts, floor 0.023. This is the study's substantive
  contribution and no item above touches it.
- The never-memorized floor behaved as required (learned its 8 given facts, 0.000 → 0.875,
  while held-out stayed flat), which is what licenses the "unlocking not learning"
  inference.
- The gradient-norm penalty produced no large improvement in the downstream relearning
  curve, with a flat dose-response and exact convergence onto the control at k = 20.
- `RELATED_WORK.md`'s position that the mechanism is Fan et al.'s (ICML 2025, variant
  GP/CR) and that no novelty is claimed.
- The four self-reported errors in §7 of the original report.
