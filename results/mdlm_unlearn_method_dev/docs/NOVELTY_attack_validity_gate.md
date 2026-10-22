# Novelty check — the RTT attack-validity gate

**Date:** 2026-10-21 · **Subject:** `FINDING20` §3's requirement that a relearning attack be shown
to relearn its own training set `T` before a flat curve on held-out `H` counts as evidence
**Verdict:** **PARTIALLY ANTICIPATED** — not stated in the unlearning literature, but a
domain-specific instance of an established principle in adversarial ML.

> **Provenance caveat, stated first.** The quotes below were extracted from arXiv HTML by an
> agent using a page-reading tool, not read character-by-character from the PDFs. They are
> high-confidence but **not hand-verified**. Any quote that becomes load-bearing in a paper must
> be re-checked against the source before external citation. WebSearch was unavailable for part
> of the check, so coverage is "the sources we could fetch", not exhaustive.

---

## 1. The principle is not ours

**Athalye, Carlini & Wagner, "Obfuscated Gradients Give a False Sense of Security"
([arXiv 1802.00420](https://arxiv.org/abs/1802.00420)), §3** gives five characteristic behaviours
that indicate a *broken attack* rather than a strong defence. The third is the structural
analogue of our gate:

> "Unbounded attacks do not reach 100% success."

The epistemic move is identical: an attack that should succeed and does not impugns the attack,
not the defence. Our gate is that principle specialised to LLM relearning, targeting a different
failure mechanism (a learning rate that degrades rather than teaches, rather than gradient
masking). **We therefore cannot claim the principle.**

Adebayo et al., "Sanity Checks for Saliency Maps" ([1810.03292](https://arxiv.org/abs/1810.03292))
is the same family of idea — verify your instrument is sensitive to what it claims to measure —
in a different domain. A looser analogue, evidence that the principle recurs, not prior art.

Tramèr et al., "On Adaptive Attacks" ([2002.08347](https://arxiv.org/abs/2002.08347)) states the
general adaptive-attack stance but we could **not** confirm a checklist item equivalent to ours.
Recorded as unverified rather than claimed either way.

## 2. It is absent from the unlearning literature we could check

| paper | relearning attack? | train-set (`T`) control? |
|---|---|---|
| Deeb & Roger, [2410.08827](https://arxiv.org/abs/2410.08827) | yes (RTT) | **nearest neighbour, but a different control** (below) |
| Łucki et al., [2409.18025](https://arxiv.org/abs/2409.18025) | yes | none; reports held-out accuracy only |
| Lee et al., UNDO, [2506.06278](https://arxiv.org/abs/2506.06278) | yes | no `T` check; but see §3 |
| Lynch et al., [2402.16835](https://arxiv.org/abs/2402.16835) | yes | none; `T`/`H` disjointness not even stated |
| RMU/WMDP, [2403.03218](https://arxiv.org/abs/2403.03218) | **no** — explicitly out of scope | n/a |
| Hu et al., [2406.13356](https://arxiv.org/abs/2406.13356) | yes | none found |
| TOFU, [2401.06121](https://arxiv.org/abs/2401.06121) | no | n/a |

Deeb & Roger's "Random Birthdays" experiment is the closest:

> "To test this assumption, we perform RTT on an original model that has not been fine-tuned on
> the random birthdays dataset. The highest accuracy we are able to get is 31.2%."

This is a *ceiling/floor check on unrelated data* — what RTT scores on something the model never
saw. It is not a confirmation that the `T` used in the forgetting experiments was actually
relearned, and no `T` accuracy is reported. Different control, same spirit.

RMU puts relearning outside its threat model outright:

> "unlearning on WMDP does not address the threat model of relearning in open source models. We
> encourage future work towards mitigating risk in this pathway"

## 3. The check found a limitation in OUR protocol, not just a novelty answer

**UNDO and Lynch et al. both run an adversarial learning-rate search and report the
best-performing attack:**

> "an adversarial learning rate search for all relearning runs over {1e-3, 7e-4, 4e-4, 1e-4,
> 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6}" — Lynch et al.

> "adversarial learning rate search ... and always mark the adversarially optimal performance
> (the best performance the adversary could obtain by varying the learning rate)" — Lee et al.

That is a **stronger** practice than ours in one respect: taking the max over learning rates
hedges against a broken attack *by construction*, since a degenerate run simply loses the max.
Our `FINDING20` numbers come from a **single** learning rate (3e-5).

**Consequence, recorded against our own result.** `FINDING20`'s comparison is still a fair
*paired* one — AFC and MDU λ=3 were attacked at the identical setting from the identical
pre-attack disclosure — but the absolute recoveries are **lower bounds**, and we have not shown
the ordering is stable under an adversarial learning-rate search. Until that sweep is run,
`FINDING20` should be read as "AFC is less robust than MDU λ=3 at lr 3e-5", not "under the
strongest available attack". The sweep is queued.

The two practices are complementary rather than redundant: the max-over-lr hedges against a
broken attack, while the gate *detects and reports* one. A max-over-lr search in which every
run is broken still returns a confident-looking zero; the gate catches that case.

## 4. What remains ours

1. The gate stated as a **requirement** for relearning-attack evaluation in LLM unlearning, where
   none of the seven papers checked states or reports it.
2. A concrete acceptance criterion (≥0.20 absolute rise in disclosure on `T`) where the
   adversarial-ML analogues are qualitative.
3. The measured failure mode: degenerate-output rate 0.93–0.98 at lr 1e-4 and 1e-5 producing ~0%
   recovery that reads as robustness. We found no paper reporting degeneracy as a diagnostic for
   a broken relearning attack.

**Honest size of the claim:** a methodological note, a special case of a known principle, worth a
paragraph and a released instrument — **not** a contribution that carries a paper.
