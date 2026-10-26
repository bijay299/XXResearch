# Erratum — our weight-space regulariser implements the **opposite direction** from the paper it cites, and my reading of that paper's prediction was reversed too

**Raised by:** external audit of snapshot `0b435427`, §2. **Verified independently** against the
paper text and the released implementation before being accepted — see §2.
**Affects:** `scripts/erase_wsr.py`, `FINDING60` §3–4, `FINDING61`, `FINDING62`.
**Does not affect:** any measured number. See §5.

---

## 1. The error

`erase_wsr.py` states its mechanism as (line 6):

> *"unlearn while **REGULARISING TOWARD THE ORIGINAL WEIGHTS**, on the theory that staying inside
> the original basin is what confers tamper-resistance"*

and implements (lines 157–163):

```python
reg = 0.0
for n, p in params:
    if n in anchor:
        reg = reg + ((p - anchor[n].to(p.device)) ** 2).sum()
reg = reg / anchor_sq
(l_erase + l_retain + a.lam_reg * reg).backward()
```

A **positive** penalty on squared distance, minimised. That pulls θ **toward** θ_base.

**'From Dormant to Deleted' (arXiv 2505.22310) proposes the opposite.** Its *Weight Dist Reg*
**maximises** distance from the pretrained weights.

## 2. The verification, because an audit finding is not evidence by itself

**Paper.** §6.1: *"We directly aim to **maximize** the distance between the pretrained and unlearned
models, by explicitly adding a term that quantifies the Euclidean distance between the two models."*
§6: *"methods with **higher** L2 distance exhibit **higher** robustness against relearning attacks"*
and *"methods with increased tamper-resistance (such as Catastrophic Forgetting and L1 Sparse) have
comparatively higher distance norm compared to methods with very poor tamper resistance."*

**Released implementation** (`shoaibahmed/vision_relearning@7b9a1ccada35a4059604c2688d2822567b3cf40d`,
`vision_unlearner.py:366-373`), downloaded and read in full at that commit:

```python
elif unlearning_method == "weight_dist_reg":
    assert pretrained_model is not None
    eps = 1e-8
    param_diff = flatten_param_dict(params=pretrained_model.parameters()).detach() \
                 - flatten_param_dict(params=model.parameters())
    # weight_dist_loss = - torch.linalg.norm(param_diff)  # aim is to maximize the distance
    weight_dist_loss = - torch.sqrt(torch.mean(torch.square(param_diff)) + eps)
    forget_loss_holder = float(weight_dist_loss)
    loss = unlearning_alpha * weight_dist_loss
```

The author's own comment — *"aim is to maximize the distance"* — settles it. **The audit is
correct.**

## 3. Why a sign flip would NOT have been the repair

The audit warned against blindly negating the existing penalty, and it is right: there are **five**
further differences.

| # | axis | ours (`erase_wsr.py`) | reference |
|---|---|---|---|
| 1 | direction | minimise distance | **maximise** distance |
| 2 | norm | **sum** of squares | **root-mean-square**, `sqrt(mean(·) + 1e-8)` |
| 3 | normaliser | ÷ `‖θ_base‖²` | ÷ parameter **count** (via `mean`), no `‖θ_base‖` |
| 4 | role | a **third term** added to an erase term and a retain term | **IS** the forget loss — `forget_loss_holder = weight_dist_loss`; there is **no** forget-data term at all |
| 5 | retain weight | summed alongside, same step | coefficient exactly **1.0**, on a **separate alternating** iteration |
| 6 | optimiser | Adam, lr 1e-5 | Adam (`--optimizer-name` default `adam`), α default **1**, `--clip-grad-norm` default **None** |

Difference 4 is the largest: the reference's method does not touch forget data. Ours derives its
erasure from a negated diffusion loss (or, after `FINDING61`, a teacher distillation) and then
anchors. Those are different algorithms, not one algorithm with a sign typo.

Difference 2 matters numerically: on 859.5M parameters at a per-coordinate RMS of 0.01, the RMS term
is 0.01 and the sum-of-squares term is ~8.6 × 10⁴ — seven orders of magnitude apart, so the two
need entirely different α scales. (This is the same class of mistake as
`ERRATUM_uce_lambda_scale.md`: assuming a hyperparameter's useful range instead of measuring the
quantity it is compared against. That is now twice.)

## 4. My reading of the paper's *prediction* was also reversed

`FINDING60` §3–4 predicted that **near-base** GIFT should be the most resistant arm, and described
the observed association between greater distance and lower recovery as *"the opposite of the
paper's prediction."* It is the paper's prediction. The observation was consistent with
2505.22310 and I recorded it as contradicting it.

This does not rescue `FINDING60`'s conclusion, because that conclusion is separately unsupported:
the endpoints are saturated at H = 1.000, and a ceiling cannot rank predictors either way. See
`ERRATUM_interpretations_r5.md`.

## 5. What changes, and what explicitly does not

**No measurement changes.** `erase_wsr.py` is left **untouched** so `FINDING61` and `FINDING62`
reproduce bit-for-bit. What changes:

- The arm is **renamed** `WSR-anchor (ours)` in every generated table. It is a legitimate method —
  anchored erasure — and keeps its results, including `wsr2_church_lr100`'s retention figures.
- **Withdrawn:** that these experiments tested or discharged 'From Dormant to Deleted'. They did
  not. That published method is **untested here**, not refuted.
- **Withdrawn:** `FINDING60`'s claim that the paper's predictor *direction* failed to transfer. The
  direction was never implemented as the paper states it.
- `FINDING62`'s positive as-is retention result keeps its numbers but loses any attribution to the
  cited baseline; it is ours, and it is descriptive (`ERRATUM_interpretations_r5.md`).

## 6. The repair as built

- `scripts/objectives.py` holds **both** terms as separate, documented, named functions —
  `anchor_penalty` (ours, preserved) and `wdr_forget_loss` (faithful) — so the two can never again
  be conflated in prose.
- `scripts/erase_wdr_v2.py` implements the faithful adaptation, reconciling all six axes above,
  with the one remaining difference declared: **the retain objective must be the diffusion loss, not
  a cross-entropy**, because the paper's experiments are example-level unlearning in vision
  classifiers. That is a domain adaptation and is stated in the artifact, not hidden.
- One **disclosed deviation**: `--max-rms`, a trust-region stop. The reference has no such bound.
  Maximising an unbounded distance is the exact shape that destroyed two previous arms
  (`FINDING61` §4; the IMMA α=1e-5 build), and the artifact records whether the stop fired.
- Six deterministic tests in `scripts/test_audit_repairs.py`, all passing, including one asserting
  the two gradients are **antiparallel** (cos < −0.99) — so if anyone later "fixes" one into the
  other, the suite fails.

**Status: the code is corrected and tested; no model has been built with it.** The faithful arm is
experiment **E1** in `CORRECTION_REGISTER_r5.md` §B.10, priced at ~6 GPU-h and unblocked. Until it
runs, the honest statement is that 2505.22310's method has **not been tested in this project**.
