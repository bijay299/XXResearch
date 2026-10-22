# Pre-registration H36 — Pool-Uniform Anchoring: name chance as the forget target

**Date:** 2026-10-22 · **Status:** pre-registered before any scored run · **Backbone:** H18 124M MDLM
**Mandate clause this targets:** *"Novel, non-obvious combinations of existing techniques are
eligible."*

---

## 1. What the measurements say to do

Two numbers from this session point the same way.

**(a) The oracle's property is "at chance, while fluent".** `FINDING21`'s exact retrain oracle sits
at 0.1100 on forget facts (chance 0.1135) with retain 0.8150 and TV→oracle 0.0275. Used as an UNDO
teacher it gives recovery 0.0669 — 3.4× better than our best real teacher.

**(b) No anchor family sources its target from outside the model.** `FINDING25` §2: MDU, EMDU, AFC,
MDU+FAM all draw the forget target from the model's own distribution on a modified input, and all
five fail (0.5608–0.8642 recovery over 37 gated runs). `FINDING18`/`FINDING19` predicted this
before those runs: form and fact-identity are the same information in output space. GA *does* leave
the model's distribution, but overshoots **below** chance (rank-forget 0.0850) into behavioural
collapse (TV 0.9725, which killed it as H31's teacher under K2).

So: one family stays inside the distribution and fails; one leaves it and overshoots into
degeneracy. Neither **names chance as the destination**.

**(c) And the iterated-UNDO check says the teacher is the binding constraint.** Checked before
spending GPU time: the selected point's own residual forget knowledge is 0.1755, *higher* than its
teacher AFC λ=8's 0.1533. Repair transfers knowledge back, so recursive damage-and-distil would
start each round from a worse teacher. That line is dead — but it localises the remaining gap in
**the teacher**, not in the repair.

## 2. The hypothesis

> **H36.** A forget target of **uniform over the closed answer pool** drives the model to the
> oracle's defining property — at chance on forget facts, fluent elsewhere — without GA's
> overshoot, and as an UNDO teacher yields lower recovery than the AFC teacher at matched repair
> budget.

The target is sourced from outside the model's own distributions (satisfying `FINDING18`'s
requirement) but is *bounded* at chance rather than unbounded like ascent (avoiding GA's collapse).
Loss is CE to the soft uniform-over-pool target at masked value-span positions. Target entropy is
ln(7.95) ≈ 2.07 against a current ~7–8, so the objective is far from its floor and descending it
requires actually spreading mass across the pool.

## 3. The legitimacy question, settled before running

PUA uses the closed answer pool. MDU/EMDU/AFC do not. **Comparing them without saying so would be
manufacturing a winner**, so two things are fixed now:

1. **The pool is derived from RETAIN data only** — no forget answer is read. Verified
   programmatically before any run: for all four attributes every forget value already occurs in
   the retain set (`forget_NOT_in_retain = 0`), and value tokens never collide across attributes
   (0 clashes), so an example's attribute is recoverable from its own value tokens. The
   practitioner always has the retain set, so this is not oracle access.
2. **A pool-free control ships with it.** `pua_global` uses a single attribute-agnostic
   distribution over *all* retain value tokens, requiring nothing but "the retain corpus". If
   `pua_global` performs comparably, the pool assumption is not load-bearing. If PUA only works
   *with* the pool, the method is reported as **requiring known answer domains** — a real
   restriction, stated in the headline, not a footnote.

`award` is a known weak spot: all 6 of its pool values share a first token, so first-token
uniformisation does nothing there and the discriminating token is later in the span. Per-attribute
results will be reported, not just the mean, so this cannot hide inside an average.

## 4. Separating predictions, declared now

| outcome | reading |
|---|---|
| teacher reaches rank-forget 0.11–0.16 **with TV ≤ 0.15**, and as UNDO teacher gives recovery ≤ 0.12 at retain ≥ 0.62 | H36 true. Beats the selected point (+0.2299 / 0.6460) on robustness at comparable retain, and is the first *method-level* contribution in this project. |
| teacher at chance and fluent, but UNDO recovery 0.12–0.23 | partial: the target works, the teacher is not the whole constraint after all |
| teacher rank-forget < 0.09 or TV > 0.15 | PUA overshoots or degenerates like GA; H36 false for the same reason GA failed |
| UNDO recovery > 0.23 | a chance-level fluent teacher does **not** buy robustness. H36 false, and a *stronger* negative than it looks — it would mean the oracle-teacher ceiling comes from something other than being at chance, which contradicts my own stated reading of `FINDING26` §2 |

## 5. Kill rules

**K1.** If the PUA teacher's TV→oracle exceeds 0.15, stop — behaviourally degenerate regardless of
rank metrics. (This rule exists because a previous "best point" was selected without checking TV
and had to be retracted twice this session.)

**K2.** If max-over-lr recovery exceeds the selected point's +0.2299 **and** retain is below
0.6460, the arm is dominated on both axes — close the line, report as closed.

**K3.** If `pua_global` matches `pua` within 0.03 on all three axes, report the pool-free version as
the method and drop the pool variant; the extra assumption would be buying nothing.

## 6. Evaluation, fixed in advance

All three axes, because each of this session's three retracted claims died on a different one:

- **knowledge** — closed-pool ranking probe, per-attribute and pooled
- **behaviour** — infill six-cell + TV→oracle (never omitted; omitting it caused two retractions)
- **robustness** — RTT under `FINDING20`'s validity gate, full lr grid {1e-5, 3e-5, 1e-4, 3e-4,
  1e-3}, reported as **max-over-valid-lr**, at attacker budgets `t_frac` ∈ {0.5, 0.8}, and under
  **both attack families** once H35 reports (span SFT and whole-passage diffusion)

n=3 seeds before any claim. Budgets matched to the baselines being compared against — a
budget-mismatched headline already had to be corrected once this session.

## 7. Novelty position, stated before results

Label smoothing, uniform-target distillation and "random-label" unlearning are all long-standing;
`RLU`-style random relabelling is a known unlearning baseline. What I have not found is a
**closed-domain uniform target derived from the retain split, used as the suppressed teacher inside
a perturb-then-distil robustification pipeline**, evaluated against an exact retrain oracle under a
gated relearning attack. A primary-source novelty check runs alongside, and if it returns
ANTICIPATED this is reported as a reproduction regardless of how well it performs — as `FINDING24`
already was for Fisher-weighted damage.
