# Literature sweep, 2026-10-28 — novelty check before committing weeks to Step 3

**Run on owner instruction, before any Step 3 work.** Method: OpenAlex API (free, no auth), four
topic queries plus targeted title lookups. **Tooling limits are recorded in §5 and this is not a
complete review.**

## 1. The decisive finding: Step 3 is already published, and we already tested it

**Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts** — ICCV 2025,
doi:10.1109/iccv51701.2025.00206, 6 citations.

From the abstract: *"if the meta-unlearned DM undergoes malicious finetuning on unlearned
concepts"* — the model is trained so that **targeted fine-tuning on the erased concept itself**
degrades it. Their stated motivation is ours exactly: benign retained concepts (their example:
"skin") related to the unlearned one ("nudity") facilitate relearning.

**That is Step 3 as proposed**, published a year ago.

**And this project already ported and tested it.** It is study S1/S1b/S1c in this repo
(`scripts/meta_unlearn_v3.py`, `meta_unlearn_v4.py`). Measured drift: S1 2.27e-5, S1b 2.05e-5,
S1c 6.10e-6 — the last below the 1.38e-5 random-walk floor. The direction was closed under a
pre-registered stopping rule, with the mechanism decomposed: the `gamma_f` level term has real
purchase (drift 7.58e-5) and yields no trap effect; the `zeta` difference term never moved a model.

**Step 3 is therefore dead on two independent grounds: it is not novel, and we have already
measured it not to work.**

## 2. The fine-tuning-robustness space is crowded

| method | venue | mechanism | our measurement |
|---|---|---|---|
| Meta-Unlearning | ICCV 2025 | meta-train so malicious FT on the concept self-destructs | ported; **no effect** (S1/S1b/S1c) |
| IMMA | 2024/25 | meta-learn to make the model hard to fine-tune | arms built; **reverses to 1.0000** (14/15 admissible) |
| GIFT | 2025 | gradient-aware immunization against malicious FT | arms built; **reverses to 1.0000** (15/15) |
| ResAlign | NeurIPS 2025 | bi-level vs **benign downstream** FT, Moreau-type implicit objective | **not tested by us**; works in their paper |
| PGU | 2026, MBZUAI thesis (doi:10.82419/340) | post-hoc hardening: constrain FT updates orthogonal to a retain-activation subspace | not tested |

Four published methods already target this problem. Three of the four that we have measured fail
completely under our attacker.

## 3. Our central negative is partly occupied

**Erased or Dormant? Rethinking Concept Erasure Through Reversibility** — arXiv:2505.16174 (2025).
Evaluates **UCE and ESD** and asks whether erasure removes generative capacity or achieves only
"superficial, prompt-specific suppression." That is our reversibility finding for those two
methods, already published.

Note the scope: **UCE and ESD only.** It does not cover the robustness-specific methods.

## 4. What the sweep did NOT find occupied

Two things, and they are what our existing data already speaks to:

1. **No matched, capacity-laddered evaluation of the robustness-specific methods under targeted
   relearning.** "Erased or Dormant?" covers UCE/ESD. Nobody appears to have put IMMA, GIFT,
   Meta-Unlearning and ResAlign on one bench, under one attacker, with admissible-cell accounting
   and attacker capacity as a swept axis. That is U33.
2. **The measurement defect.** A 12-way closed-set CLIP accuracy with no reject option scores a
   speckle overlay on a visible church as suppression (U30), and scores a window-grid texture as
   `church=0.546` (U29). Separately, the field's "robust concept erasure" line — **STEREO**
   (CVPR 2025), **AEGIS** (arXiv:2602.06771) — means robust to adversarial **prompts/embeddings**,
   not to weight fine-tuning. These are different robustness axes and are routinely conflated.

Both are evaluation contributions, **not a method**, and must not be written as one.

## 5. Limits of this sweep — stated so it is not over-trusted

- **WebSearch was unavailable**: this session's budget was exhausted (200/200) before the sweep ran.
- **arXiv's API returned HTTP 429** (rate-limited) on every attempt, direct and via fetch. No arXiv
  full-text or listing was read.
- All results come from **OpenAlex metadata and abstracts only**. No full paper was read. Claims
  about mechanism are from abstracts and may be imprecise.
- OpenAlex indexing lags for very recent preprints; an arXiv paper from the last few weeks could be
  missing entirely.
- Four queries and seven title lookups is a sweep, not a systematic review.

**Conclusion that does survive these limits:** Meta-Unlearning (ICCV 2025) occupies Step 3's idea
explicitly, in its own abstract. That single fact is enough to stop Step 3, and it does not depend
on the sweep being complete.
