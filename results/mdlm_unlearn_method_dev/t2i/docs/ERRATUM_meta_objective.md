# Erratum — our Meta-Unlearning port omits **two of the three terms** of the paper's meta objective, so `FINDING56`'s causal claim does not follow

**Raised by:** external audit of snapshot `0b435427`, §3. **Verified independently** against the
paper's Eq. 8 and the released implementation — see §2.
**Affects:** `scripts/meta_unlearn.py`, `PREREG_H80`, `FINDING56`, `FINDING58`'s ablation inference,
and the three cross-model Meta arms in `FINDING64`.
**Does not affect:** any measured number. See §5.

---

## 1. The gap, stated as algebra

**The paper (arXiv 2410.12777, Eq. 8)**, meta term:

$$-L_F(\theta') \;-\; \zeta\big[L_R(\theta') - L_R(\theta)\big]$$

**Our port** (`meta_unlearn.py:122-143`):

$$L_{\text{distil},F}(\theta) \;+\; L_{\text{distil},R}(\theta) \;-\; \lambda\,L_R(\theta')$$

Three components are specified; we have one of them.

| Eq. 8 component | ours | note |
|---|---|---|
| $-L_F(\theta')$ — post-attack **target** diffusion loss | **ABSENT** | nothing in our objective depends on the target loss after the simulated attack |
| $-\zeta L_R(\theta')$ — post-attack retain loss | **present** as $-\lambda L_R(\theta')$ | the one term we have |
| $+\zeta L_R(\theta)$ — **pre-attack** retain diffusion loss | **ABSENT** | `distil_R(θ)` is teacher-output distillation, a different function with a different minimum. Not algebraically this term |

## 2. The verification

The released implementation
(`sail-sg/Meta-Unlearning@5a2a7600895a2885323cbbf599acaf3f74a96f64`, `src/meta.py:920-1030`),
downloaded and read in full, accumulates exactly Eq. 8's signed structure into `sum_grads`:

```
+ gamma1_1 * grad L(irt_test  @ theta)     # pre-attack retain  (irrelevant pool)   POSITIVE
+ gamma1_2 * grad L(tgt_test  @ theta)     # pre-attack retain  (related pool)      POSITIVE
    train_step(hrm_train, ..., task_optimizer_full)   # simulated malicious finetune, FULL scope
- gamma2_1 * grad L(hrm_test  @ theta')    # post-attack TARGET                     NEGATIVE
- gamma2_2 * grad L(tgt_test  @ theta')    # post-attack retain (related pool)      NEGATIVE
- gamma2_3 * grad L(irt_test  @ theta')    # post-attack retain (irrelevant pool)   NEGATIVE
...
for i, param in enumerate(parameters):
    param.grad += sum_grads[i]             # added on top of the erasure loss' own gradient
```

So the paper's $R$ is the union of the reference's `irt` (irrelevant) and `tgt` (related) pools,
its $F$ is the reference's `hrm` (harmful) pool, and each pool carries its own γ where the paper
uses one shared ζ. **The audit's equation is correct and our port has one of the six signed
gradient contributions.**

## 3. Why this specifically undermines `FINDING56`'s headline

`FINDING56` concluded that **Meta-Unlearning's self-destruct trigger fires on normal users** —
benign capability collapses without any attack. The measurement is real: variant A at λ ≥ 0.5 loses
retain8 (λ=0.5 → 0.6333, λ=0.8 → 0.4750, λ=0.9 → 0.0000).

But our objective **rewards exactly that**. With $+\zeta L_R(\theta)$ absent, the term
$-\lambda L_R(\theta')$ penalises the **level** of benign loss at θ′, not its **change** across the
attack step. The paper's two-term difference is a *cancellation*: it asks for retain loss to get
worse **only as a consequence of the attack**. Strip the positive pre-attack term and the objective
instead asks for benign denoising to be bad at a nearby parameter point — which, with θ′ a single
1e-5 step from θ, is nearly the same as asking for it to be bad at θ.

**So premature benign damage is consistent with variant A's own altered objective.** It is not
evidence that the published method's trigger misfires. That attribution is **withdrawn**.

## 4. Two further corrections in the same area

**(a) GIFT is not an ablation of Meta-Unlearning.** `FINDING58` used GIFT's failure to rule out the
dropped meta-gradient term as the explanation of `FINDING56`. Invalid: the losses, data, update
schedule and parameter scopes all differ. Two different algorithms failing isolates nothing.
**Withdrawn.**

**(b) The scope description is wrong.** `FINDING56` §5a calls GIFT the right comparator as *"also
bi-level and also editing cross-attention."* Our Meta port optimises **the full UNet** —
`meta_unlearn.py:107` is `torch.optim.Adam(unet.parameters(), lr=a.lr)` and `:109` collects every
parameter with `requires_grad`. GIFT's 80 `attn2` projections are 44.0M (5.11%); our Meta arm moved
859.5M. **Corrected.**

What survives from `FINDING56` §5a, stated narrowly: **five TI runs on the variant-A λ=0.5 arm, all
gate-failing, maximum ungated H = 0.0667**, while a GIFT arm at *lower* retain8 (β=1.35, 0.6000)
hosted a gate-valid TI attack reaching H = 1.000. That is **limited evidence** that the matched-
retain confound is not a crude scalar-threshold effect. It does **not** isolate the meta objective:
equal mean retain accuracy does not match target geometry, conditioning quality, gradient
conditioning or per-concept damage.

## 5. Three further implementation differences, and what changes

Also confirmed in our code:

- **Target data provenance.** `meta_unlearn.py:71` builds the generator from `a.teacher` — the
  **already-erased** model — and `:86` generates the "target-concept images" through it. The code
  never checks that those images depict the target. An erased model asked for its erased concept is
  the least likely source of target content in the project.
- **Benign pool** is `[q for q in RETAIN_POOL if kw not in q.lower()][:24]` — the first 24 eligible
  prompts, with no validation of suitability for the meta task.
- **First-order approximation** was disclosed in advance (`PREREG_H80`), and remains a real
  difference — but it is **not** the most immediate one, and attributing the failure to missing
  second-order differentiation while two objective terms are absent is unsound.

**No measurement changes.** `meta_unlearn.py` is untouched; all its results reproduce. What changes:

- The arm is **variant A** everywhere, including `meta_sd15_church_lam0.5`,
  `meta_bksdm_church_lam0.3` and `meta_bksdm_church_lam0.5` in `FINDING64`, which used the same
  script and are therefore variant-A measurements too — valid, and not evidence about the published
  method.
- **Withdrawn:** that `FINDING56` tested the published method's self-destruct mechanism; that GIFT
  ablates the missing term; that both arms edit cross-attention.
- `PREREG_H80` is **not rewritten.** Its objective statement is wrong and is corrected *here*; the
  plan stands as written, with this erratum and `PREREG_PRECEDENCE.md` attached.

## 6. The repair as built

`scripts/meta_unlearn_v2.py` implements the full Eq. 8 and, separately, repairs each difference
above:

| repair | how |
|---|---|
| both missing terms | `meta = -lam_target*L_F(θ') - zeta*L_R(θ')`, with `+zeta*L_R(θ)` in the at-θ backward |
| scope separation | `--meta-scope` (full/xattn/selfattn/noxattn, mirroring the reference's `--finetuning_method`) **independent of** `--attack-scope`, both recorded with parameter counts |
| data provenance | `--target-source base|teacher`, plus a **CLIP content check** of the generated pool against the evaluation's own 12-label classifier, and `--require-target-acc` which **refuses to train** on a pool that does not depict the target |
| pool roles | benign pool drawn through `split_registry`, role declared |
| order | `--order first_order|differentiated`; `differentiated` **raises `SystemExit`** rather than being silently claimed, so the objective repair and the unroll question stay separable |

Four deterministic tests, all passing, including `t_meta_has_all_three_terms` (asserts
∂/∂L_F(θ′) = −1, ∂/∂L_R(θ′) = −ζ, ∂/∂L_R(θ) = **+ζ**) and `t_variantA_is_missing_two_terms`
(asserts variant A has **`None`** gradient w.r.t. both missing terms — the gap is a test, not a
sentence).

**Status: the code is corrected and tested; no model has been built with it.** Experiment **E2** in
`CORRECTION_REGISTER_r5.md` §B.10, ~8 GPU-h, unblocked. **E3** is the within-method variant-A-vs-v2
ablation at matched retain8 that `FINDING58` lacked. Until E2 runs, Meta-Unlearning as published is
**untested here**.
