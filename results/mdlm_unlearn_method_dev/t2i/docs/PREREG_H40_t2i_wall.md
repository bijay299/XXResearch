# Pre-registration H40 — a structural wall for T2I concept erasure, stated before the deciding runs land

**Date:** 2026-10-22 · **Status:** pre-registered. The two runs that could refute it (`H38`
repair-budget, MDLM `H39` trainability) are **in flight and unread** as this is written.
**Why written now:** so it cannot be fitted to results afterwards. The MDLM phase's "two walls"
(`FINDING19`) earned their weight by predicting failures before they were measured; a wall argued
after the fact is worth nothing.

---

## 1. The claim

> **H40.** In a text-to-image diffusion model that retains general generative capability, concept
> erasure cannot be made robust to a fine-tuning adversary. The attacker's task — re-associating a
> text direction with visual features that are still present — is strictly easier than the
> defender's task of removing those visual features without removing utility. Any erasure that
> actually destroys the visual features destroys general generation with them.

This is an asymmetry claim, not a theorem. It is falsifiable in an obvious way: exhibit an arm with
utility near the base model's and a validated attack that fails to restore the concept.

## 2. The evidence already in hand

| observation | what it contributes |
|---|---|
| UCE (closed-form cross-attention edit) fully reversed: H 0.067 → **1.000**, 3/3 gated | routing-only edits leave the visual features intact, so re-association is trivial |
| ESD (negative-guidance fine-tune) fully reversed: H 0.467 → **1.000**, 3/3 gated | the same holds for a fine-tune, not just a closed-form edit |
| the attacker needs **150 steps and 16 images**, self-generated from the public base model | the defender's advantage in data and compute is nil |
| damage+distil resists **only** once generation is destroyed (α=0.5: H 0.400 at retain 0.092; α=0.3: retain 0.458 and **zero** robustness) | removing features costs utility first and buys robustness later, if at all |
| `FINDING32`: late attack loss rises monotonically with α (0.3393 → 0.4864) and degradation **precedes** benefit | what looks like robustness is movement toward untrainability, not toward removal |
| cross-attention collinearity: content tokens separate concepts at cos 0.13–0.21, but the trailing positions the UNet actually reads sit at cos **0.52–0.59** | the edit surface with the leverage is the one where concepts are least separable |

## 3. What would refute H40 — declared now

| result | consequence |
|---|---|
| an arm with **retain ≥ 0.85** (base = 1.000) whose **max-over-valid-gate** post-attack concept accuracy stays **≤ 0.40**, with fit-drop comparable to the undamaged teacher | **H40 refuted.** Robustness without brokenness exists, and that arm is the contribution the mandate asks for. |
| the `H38` repair-budget run (α=0.5, 4× steps, 3× pool) reaches retain ≥ 0.85 at H ≤ 0.40 | H40 refuted by exactly the run now in flight; the earlier collapse was budget starvation |
| retain 0.60–0.85 at H ≤ 0.40 with healthy fit-drop | H40 weakened, not refuted: a real frontier point exists but short of base utility |
| the MDLM `H39` control shows damage+distil arms fit the attacker's data **as fast as** suppression arms | the T2I confound does not generalise; `FINDING31` is clean and H40's mechanism is wrong about the LLM case |

## 4. What H40 is not

- **Not a claim that erasure is useless.** Erasure that survives *prompting* attacks but not
  *weight-access* fine-tuning is still meaningful under a weaker threat model. H40 is scoped to an
  adversary with the weights.
- **Not an impossibility proof.** It is an asymmetry supported by six measurements on one model,
  one concept family, and two erasure methods, with a stated refutation condition.
- **Not a reason to stop.** If H40 holds, the useful output is the *measured asymmetry* plus the
  protocol that exposes it (validity gate + trainability control), which is what the T2I literature
  currently lacks — not a method claim.

## 5. The honest status of the mandate

The mandate asks for a method that beats the strongest existing methods on a forgetting–utility
trade-off. Across both phases I do not have one. H40, if it survives its refutation conditions,
explains why the T2I half is hard rather than delivering the method — the same shape of outcome as
`FINDING19` in the MDLM phase. That is a weaker contribution than the mandate requests and is
recorded as such.
