# Pre-registration H31 — a composite teacher: forget behaviour from the over-forgetter, retain behaviour from the reference

**Date:** 2026-10-22 · **Status:** pre-registered before any run · **Backbone:** H18 124M MDLM
**Mandate clause this targets:** *"Novel, non-obvious combinations of existing techniques are
eligible."*

---

## 1. What the measurements say to do

The oracle-teacher ceiling (`FINDING26` §2) is far better than any real method:

| teacher | retain | recovery (max-over-lr) | TV→oracle |
|---|---|---|---|
| *exact oracle (ceiling)* | *0.8150* | *0.0669* | *0.0275* |
| AFC λ=2 (selected point) | 0.6460 | **+0.0314** | 0.0745 |
| best anchor family anywhere | 0.54–0.74 | 0.5608 | 0.0725 |

The oracle's defining property is that it is **at chance on forget facts while retaining the rest**.
No single unlearning arm has both halves — but two different arms each have one:

| arm | rank-forget (chance 0.1135) | rank-retain |
|---|---|---|
| **GA λ=0.5** | **0.0850** — *below chance, over-forgets* | 0.6350 |
| `theta_ref` (no unlearning) | 0.6975 | **0.6725** |
| AFC λ=2 | 0.1600 | 0.7275 |

## 2. The combination

UNDO distils one suppressed teacher. Instead, distil a **composite teacher**, routed by example
type:

```
forget examples  ->  teacher_forget = GA λ=0.5      (lowest residual forget knowledge)
retain examples  ->  teacher_retain = theta_ref     (best retain, never unlearned)
general text     ->  teacher_retain = theta_ref     (fluency from the un-damaged reference)
```

Everything else is held at the selected point: Fisher-protected damage at α=0.6, 12,000 repair
steps, retain-oversampled pool, forget prompts present throughout (their absence on 8B was the
bug that meant suppression was never transferred where we measure it).

**Why this is not obviously decomposable.** UNDO's robustness comes from the student never
acquiring latent forget knowledge during repair. A single suppressed teacher leaks its residual
knowledge into the student (`FINDING26` §1: recovery rises with repair budget). Routing forget
examples to an *over-forgetting* teacher removes that leak at source, while routing retain and
general examples to the *un-unlearned* reference removes the retain cost at source — the two
failure modes are addressed by different teachers rather than traded against each other.

**Why it is legitimate.** `theta_ref` is the model being unlearned, which the practitioner always
has. GA λ=0.5 is produced by an ordinary unlearning run on the same data. Neither is an oracle,
and no gold forget answers are used anywhere — targets are always teacher outputs.

## 3. Separating predictions, declared now

| outcome | reading |
|---|---|
| **retain ≥ 0.65 and max-over-lr recovery ≤ 0.05** | the composite teacher beats the selected point on both axes and approaches the oracle ceiling. A genuine positive result, and the first method-level contribution in this project. |
| retain ≥ 0.65, recovery 0.05–0.15 | better retain at comparable robustness — an improvement on the frontier, not a breakthrough |
| recovery > 0.20 | routing forget examples to an over-forgetter does **not** prevent the leak; H31 false |
| retain < 0.55 | the composite teacher inherits GA's retain collapse (GA-teacher UNDO gave 0.4675); H31 false for that reason |

## 4. Kill rules

**K1.** If max-over-lr recovery exceeds the selected point's +0.0314 **and** retain is below
0.6460, the composite is dominated on both axes — stop the line, report as closed.

**K2.** If TV→oracle exceeds 0.15, the arm is behaviourally degenerate regardless of its
rank-metric numbers. This rule exists because the previous "best point" was selected without
checking TV and had to be corrected (`FINDING26`, the EMDU-teacher error).

## 5. Evaluation, fixed in advance

Ranking probe (knowledge), infill six-cell + TV→oracle (behaviour), and RTT under `FINDING20`'s
validity gate at the **full lr grid** {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}, reported as
**max-over-valid-lr**, at attacker budgets `t_frac` ∈ {0.5, 0.8}. All three axes and both
adversary degrees of freedom, because this session has had claims weaken three times when the
adversary was strengthened after publication.

n=3 seeds before any claim.

## 6. Novelty position, stated before results

Composite / multi-teacher distillation is not new in general, and routing different data subsets
to different teachers is a standard trick in the distillation literature. What I have not found is
its use to decouple **forget-behaviour transfer from retain-behaviour transfer inside a
perturb-then-distill robustification pipeline**. A primary-source novelty check will run
alongside, and if it returns ANTICIPATED this is reported as a reproduction regardless of how well
it performs.
