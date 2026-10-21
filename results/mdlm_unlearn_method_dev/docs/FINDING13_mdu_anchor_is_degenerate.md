# Finding 13 — MDU matches an anchor whose own generations are 97.9% degenerate

**Date:** 2026-10-21
**Status:** open finding; it is the basis of the current method candidate
**Attempts:** `p101` (generation-level, 335 facts), `p103` (per-position, 23,450 positions),
`p102` (training, τ=1)
**Scripts:** `scripts/entity_mask_target.py`, `scripts/anchor_probe.py`, `scripts/train_anchor.py`
**Outputs:** `outputs/esd_gate_acg_full.json`, `outputs/anchor_probe_theta0.json`,
`adapters/anchor_mdu_t1/report.json`
**Primary sources read:** MDU [arXiv 2605.18253](https://arxiv.org/abs/2605.18253) (PDF, full),
EMDU [arXiv 2610.25447](https://arxiv.org/abs/2610.25447) (PDF, full)

---

## 1. What MDU and EMDU actually optimise

MDU Eq. (5), verbatim from the PDF:

```
L_forget(θ) = E_{(x,y)~D_f, t, y_t} [ (1/|M_t|) Σ_{i∈M_t} KL( p^c_θ(·|x,y_t) ‖ (1/Z_i) p^u_{θ₀}(·|m,y_t)^τ ) ]
```

with `θ₀` frozen, `τ ∈ [0,1]`, `m` a null/masked prompt, plus a weighted retain
reconstruction loss. EMDU Eq. (3)–(4) keeps the construction, reverses the KL to anchor-first,
and masks the anchor's entire answer span (FAM); it states it uses "the sameM_t masking and
uniform averaging as MDU's forget loss".

MDU's stated trade-off, verbatim: at `τ=1` the anchor "sometimes under-forgets information that
needs to be forgotten"; at `τ=0` "the target anchor distribution becomes a uniform
distribution, which is exactly the same as the Maximum Entropy (ME) loss… but might suffer from
over-forgetting."

So the only knob for forgetting harder tilts **every** position toward uniform.

## 2. MDU's fluency evidence is conditional on a scaffold

MDU's Table 1 is the paper's demonstration that the anchor "preserves the underlying answer
structure but shifts predictions away from the original fact". It shows, for the TOFU query
*"What is the full name of the author born in Taipei, Taiwan on 05/11/1991 who writes in the
genre of leadership?"* — the same query our own probes use — anchor outputs such as

> "The author's full name is Rizvi Alvi and he was born in Karachi, Pakistan on 16 April 1945."

a fluent REPLACE. But the table's caption states "**Gray boxes indicate fixed tokens**", and
every row is a *partially denoised* state: `k=2`, `k=17`, `k=28`. The anchor is shown only
where revealed answer tokens scaffold it.

**Decoded without that scaffold, the same anchor collapses.** Generating from the all-masked
prompt with the pinned sampler, 335 TOFU forget facts, θ₀ backbone:

| anchor prompt (full decode) | DISCLOSE | DEGENERATE | REPLACE | IGNORANT | ASCR | TV→pristine |
|---|---|---|---|---|---|---|
| `natural` (reference) | 0.8358 | 0.0060 | 0.0776 | 0.0806 | 0.7506 | 0.8123 |
| **`all_masked` — MDU's anchor** | 0.0000 | **0.9791** | 0.0209 | 0.0000 | 0.0000 | 0.9796 |
| `entity_masked` | 0.4090 | **0.0030** | **0.4597** | 0.1284 | 0.4034 | **0.3825** |
| `control_masked` (length-matched) | 0.5194 | 0.0000 | 0.1970 | 0.2836 | 0.5040 | 0.4899 |
| *pristine target* | *0.030* | *0.000* | *0.470* | *0.388* | — | *0* |

MDU's anchor is **97.91% DEGENERATE** as a generator. It is the furthest arm from pristine
(TV 0.9796) — further than doing nothing (0.8123).

## 3. Reconciling the two readings, and correcting myself twice

I first hypothesised the anchor was EOS-dominated, measured it at teacher-forced states, and
**refuted my own hypothesis**: `eos_argmax 0.0927`, entropy `2.5087`, `p_true 0.3909` — not
degenerate, and still carrying the fact. I recorded that refutation and dropped the claim.
Then the generation-level measurement showed 0.9791 DEGENERATE. Both are correct, and the
per-position sweep over 23,450 positions explains why — the anchor's character depends strongly
on how much answer is revealed:

| answer still masked | `p_true` | `punct` argmax | `eos` argmax | entropy |
|---|---|---|---|---|
| 0.00–0.25 (late, well scaffolded) | **0.6505** | 0.2876 | 0.1107 | 0.82 |
| 0.25–0.50 | 0.5849 | 0.2515 | 0.0985 | 1.33 |
| 0.50–0.75 | 0.4123 | 0.2591 | 0.0750 | 2.15 |
| 0.75–1.00 (early, no scaffold) | **0.0639** | **0.4224** | 0.0156 | **4.41** |

So the all-masked anchor is **two different targets at two ends of the noise schedule**:

- **late (low mask rate):** fluent and structured, but it *knows the fact* (`p_true` 0.651) —
  a bad ignorance target.
- **early (high mask rate):** genuinely fact-free (`p_true` 0.064) — a good ignorance target —
  but 4.41 nats and **42.2% punctuation argmax**, i.e. formless.

MDU draws `t ~ U[0,1]` and averages with uniform weight `1/|M_t|`. It therefore averages a
target that is good-for-forgetting-and-bad-for-form against one that is
bad-for-forgetting-and-good-for-form. **That is a measured mechanism for the bimodality this
project independently observed across 15 baseline grid points** (9 of 15 at DISCLOSE 0.000 /
DEGENERATE 1.000; the fluent cell holding 4 of 960 generations).

Note the degeneracy channel is **punctuation, not EOS** — 42.2% vs 1.6% at high mask rate.
That matches the `'........'` outputs in our own baseline grid, which the six-cell partition
records alongside `''`. A metric that scored only EOS or only emptiness would have missed it.

## 4. A falsifiable prediction about EMDU

EMDU's FAM masks the anchor's whole answer span, which removes the fact leak at every noise
level — our §3 table independently confirms the leak EMDU identifies (`p_true` 0.651 at low
mask rate). But FAM removes the **scaffold** at every noise level too, which by §2–§3 is what
supplies form.

> **Prediction P-EMDU:** at matched forget quality, EMDU should be *more* prone to degenerate
> output than MDU, because FAM moves the anchor toward the formless high-mask-rate regime
> everywhere.

This is a real prediction about a paper published three days ago, and it is testable with arms
already implemented (`scripts/train_lam.py`, arms `mdu` and `emdu`, partition-evaluated).
If EMDU is *not* more degenerate, the account in §3 is incomplete and must be revised.

## 5. What is NOT claimed

- **Not** that MDU's published TOFU numbers are wrong. They are ROUGE-L and token probability,
  and nothing here contradicts them. The point is narrower and checkable: **ROUGE-L ≈ 0 and low
  token probability are achieved equally by a fluent wrong answer and by `'........'`**, so
  those metrics cannot distinguish the desirable cell from the degenerate one. MDU's best forget
  row is `τ=0`, which the paper itself identifies as the uniform / Maximum-Entropy target.
  Whether their reported trade-off is partly a degeneracy artifact is an open, answerable
  question, and re-scoring a τ sweep with the six-cell partition answers it. **Until that is
  run, this is a hypothesis about their metric, not a finding about their method.**
- **Not** that our earlier matched MDU baseline supports any of this. That run reported
  DEGENERATE 1.000 — but on the *confounded synthetic backbone* at `gen_length 32`, and it does
  **not** reproduce MDU's published behaviour. The TOFU-backbone run at τ=1 (`p102`) *does*:
  it under-forgets, exactly as the paper states (`student_fact_argmax` 0.0889 → 0.0731,
  `p_true` rising 0.584 → 0.731, entropy flat). **The synthetic-backbone MDU number must not be
  cited as MDU's behaviour**, and the baseline is being redone on the TOFU backbone across τ.
  Flagged because citing it would have inflated our own position against a baseline.
- **Not** that entity-selective anchoring is established. At the *per-position* level the entity
  anchor is indistinguishable from the length-matched control (overall `fact` 0.0903 vs 0.0914,
  `p_true` 0.4620 vs 0.4707, entropy 1.7319 vs 1.6929) at every mask-rate bin. It separates only
  at the *generation* level, and there the DISCLOSE separation is **0.110 against a
  pre-registered threshold of 0.10 — marginal**. What is not marginal is where the mass goes:
  REPLACE 0.4597 vs 0.1970. That is the pre-registered direction (P2), but on one backbone, one
  dataset, one seed.

## 6. What this licenses next

The entity-masked anchor is the only construction measured so far that is simultaneously
non-degenerate (DEGEN 0.0030) and substantially less disclosing than the reference
(0.4090 vs 0.8358), and it is the closest of all arms to the pristine profile (TV 0.3825). It is
however still far from pristine on DISCLOSE (0.4090 vs 0.030), so an anchor swap alone is not a
method. The method candidate under test pairs it with position localization —
`docs/H14_LAM_DESIGN.md` — with `ent_uniform` as the matched-force control and
`ent_antilocalized` as the attribution control that can void the claim.
