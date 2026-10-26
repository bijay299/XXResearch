# AGENT 3 — Method fidelity audit: objectives, signs, scope, optimizers, coefficients, provenance

**Date:** 2026-10-26 · **Mode:** read-only analysis + non-GPU documentation repairs under `FREEZE_ACTIVE`
**Scope:** `scripts/{erase_utrap,erase_utrap_v2,erase_esd,erase_frp,erase_gift,erase_imma,erase_adv,erase_wsr,erase_wdr_v2,meta_unlearn,meta_unlearn_v2,attack_core,objectives}.py`
**Ground truth:** the papers' equations as transcribed, plus the pinned clones at
`upstream/{IMMA,Meta-Unlearning,vision_relearning,unified-concept-editing,FRAG}`.
**Not verified against code:** ESD, GIFT, Receler/AdvUnlearn, FRP/FRAG — no implementation is
available for any of them in this tree (FRAG's repo is README-only; see §2.4).

Every claim below was re-derived from source. Nothing was taken from existing project documentation.

---

## 1. Summary table

| method | claimed classification | **VERIFIED classification** | objective / sign status | scope status | defects found |
|---|---|---|---|---|---|
| **UTRAP** (`erase_utrap.py`) | our own design | **our own design** ✔ | VALID — `+∇L_retain − λ∇L_target(θ_K)`, signs correct (`:171`) | VALID — full UNet, 859.5M | **D1** docstring called the inner loop "the ACTUAL … attack (the one attack_relearn.py runs)"; divergent on **7/9 axes** — all 7 re-confirmed |
| **UTRAP-v2** (`erase_utrap_v2.py`) | our own design | **our own design** ✔ | VALID — same objective, signs correct (`:272`) | VALID — full UNet | **D2** relies on `attack_core`'s *mirror* of the evaluator, not the evaluator itself; RNG-stream + data axes not covered by `divergence_from_eval()` |
| **ESD** (`erase_esd.py`) | reimplementation from paper | **reimplementation from paper** ✔ | VALID — `target = ε*(∅) − η[ε*(c) − ε*(∅)]` (`:95`), correct | **SUSPECTED** — `esd_u` = `not attn2` (815.5M) keeps `time_embed`/`conv_out`; official `noxattn` is reported to exclude them. Unverifiable (no ESD clone) | **D3** scope possibly wider than reference; `esd_u` here ≠ `meta_unlearn_v2.SCOPES["noxattn"]` |
| **FRP / FRAG** (`erase_frp.py`) | our port from paper (repo README-only) | **reimplementation from paper + domain adaptation** | VALID as transcribed; **open question** on Eq.6's `+λ·rank(|W|)` sign | VALID — 32 `ff.net.*` linears, **148.7M = 17.3%** (independently recomputed: 12·Σd² = 148,684,800) | provenance claim **VALID** (FRAG README:99 still says "Code release is in preparation"); domain adaptation (LLM→diffusion) was under-declared |
| **GIFT** (`erase_gift.py`) | our port from paper, no code released | **reimplementation from paper**; `code_released:False` is **UNVERIFIED**, not verified | VALID — `L_max = −L_diff` (`:228`), `L_imm = L_max + β·L_noise` (`:230`), signs correct | VALID — ψ = 80 attn2 tensors, **44.0M = 5.12%** (recomputed 43,962,560) | **D4** "faithful on exactly the point where our Meta port was not" overstates: alternation-emergent 2nd order is shared by the IMMA *and* Meta references, both strictly 1st-order |
| **WDR-v2** (`erase_wdr_v2.py`) | faithful adaptation of 2505.22310 §6.1 | **faithful on 7 of 8 claims; DIVERGENT on the 8th** | VALID — `−α·√(mean(d²)+1e-8)`, matches `vision_unlearner.py:366-373` exactly incl. eps | VALID — all `unet.parameters()`, matches `flatten_param_dict` (buffers excluded both sides) | **D5 (major)** docstring claim 5 + JSON `schedule` said "ALTERNATING … as the reference does". **The reference SUMS.** Alternation there is gated on an `alternating_` name prefix (`:411`) which `weight_dist_reg` lacks → both branches backward into one grad buffer, one `optimizer.step()` (`:637-682`) |
| **WSR-anchor** (`erase_wsr.py`) | "our implementation of the method class of 2505.22310" | **our own design** (anchored erasure) | **CONFIRMED DEFECT (direction)** — minimises `+λ‖θ−θ_base‖²`; reference maximises RMS distance | VALID — full UNet | **D6** re-confirmed from released code; 4 further differences beyond sign (RMS vs sum-sq; no `‖θ_base‖` divisor; reference's term *replaces* the forget loss; reference retain unscaled) |
| **Meta variant A** (`meta_unlearn.py`) | our port from paper, first-order | **reimplementation from paper, PARTIAL** | **CONFIRMED DEFECT** — omits **2 of Eq.8's 3** meta terms; signs of the one present term correct (`:143`) | VALID — full UNet, undeclared at the time | **D7** missing `−L_F(θ′)` and `+ζ·L_R(θ)`; `distil_R(θ)` is not the latter; two *extra* distillation terms not in Eq.8 |
| **Meta v2** (`meta_unlearn_v2.py`) | full Eq.8 | **full Eq.8 algebraically correct; coefficient inert, estimator noisy** | VALID as algebra — `+ζL_R(θ) − λ_t L_F(θ′) − ζL_R(θ′)` (`:289`, `:306`) matches reference signs | VALID — `--meta-scope`/`--attack-scope` separable and recorded; defaults full/full match reference scripts | **D8** one shared ζ ⇒ the pre/post retain pair cancels exactly at every ζ, so **ζ is inert as a coefficient and untested** (the *pairing itself* is the mechanism and does work — §4.4); **D9** θ and θ′ retain losses use **independent noise/timesteps** ⇒ pairing holds only in expectation, per-step estimator is noise-dominated; **D10** inner attacker ≠ reference *and* ≠ evaluator, with **no divergence record** |
| **IMMA** (`erase_imma.py`) | our port from paper, **`code_released: False`** | **reimplementation from paper; official code EXISTS and was not used** | VALID — single unweighted MSE, descend φ / **ascend** S (`:180`), matches `defend_text_to_image_lora.py:944-960`; reference has **no** loss coefficients | **CONFIRMED DEFECT** — ours attn2-only **44.0M**; reference filters `to_q/to_v/to_k/to_out` with **no attn2 test** ⇒ attn1+attn2 **93.5M**. **Ratio 2.128×** | **D11** false `code_released:False`; **D12** no gradient clipping (reference clips outer to 1.0); **D13** Adam vs reference AdamW(wd 1e-2); **D14** "unassigned in paper" true of paper, false of code |
| **AdvUnlearn family** (`erase_adv.py`) | "Receler / AdvUnlearn family, OUR RE-IMPLEMENTATION" | **our own design**, inspired by that family | VALID *now* — inner ascends the erasure MSE (`:124-127`) | VALID — `esd_x` default = 44.0M attn2 | **D15** classification overstated: no eraser adapter (Receler), no text-encoder training (AdvUnlearn); **D16** the inner sign bug was repaired *in place* and the artifact had **no version field**, so pre/post-repair runs were indistinguishable |
| **attack_core** (`attack_core.py`) | "ONE implementation … shared by the EVALUATION attack (`attack_relearn.py`)" | **a specification + mirror; NOT shared** | n/a | n/a | **D17 (major)** `attack_relearn.py` **never imports `attack_core`** (`grep` → no match); it runs its own inline loop at `:333-362`. The stated equivalence was never established |

---

## 2. Independent verification of the four prior claims

### 2.1 "IMMA reference scope is 93.5M, ours 44.0M, ratio 2.13×" — **CONFIRMED**

Reference filter, `upstream/IMMA/train/defend_text_to_image_lora.py:561-567` (identical at
`defend_dreambooth.py:882-889`, `defend_dreambooth_lora.py:883-889`, `defend_textual_inversion.py:675-677`):

```python
xattn_layer_names = ['to_q', 'to_v', 'to_k', 'to_out']
for name, param in unet.named_parameters():
    if any([key in name for key in xattn_layer_names]) and 'lora' not in name:
```

There is **no `attn2` test**, so this captures self-attention too — despite the variable name.
Our filter, `erase_imma.py:135`, requires `".attn2." in n`.

Recomputed from the SD-1.4 UNet (16 `BasicTransformerBlock`s; d = 320×5, 640×5, 1280×6;
Σd² = 12,390,400; Σd = 12,480):

| set | per block | total | tensors |
|---|---|---:|---:|
| attn2 `to_q/k/v/out` | 2d² + 1537d | **43,962,560 (44.0M)** | 80 |
| attn1 `to_q/k/v/out` | 4d² + d | **49,574,080 (49.6M)** | 80 |
| attn1 + attn2 | — | **93,536,640 (93.5M)** | 160 |

Ratio **93,536,640 / 43,962,560 = 2.128×**. ✔ Claim confirmed to 3 s.f.

**Nuance the prior claim omits:** our port follows the paper's *text* ("cross-attention layers");
the released *code* is broader, and the authors' own variable name suggests they believed it was
cross-attention only. Which is "IMMA" is genuinely ambiguous. **No result in this project tests the
released code's scope.** Recorded in `erase_imma.py`'s docstring and artifact.

### 2.2 "Meta variant A omits two of Eq.8's three terms" — **CONFIRMED**

Eq.8's meta term decomposes into three components. Reference signs re-verified at
`upstream/Meta-Unlearning/src/esd_meta.py:992-1058`:
`+γ1_1·∇L_irt(θ) +γ1_2·∇L_tgt(θ) −γ2_1·∇L_hrm(θ′) −γ2_2·∇L_tgt(θ′) −γ2_3·∇L_irt(θ′)`.

| component | variant A (`meta_unlearn.py`) |
|---|---|
| `−L_F(θ′)` post-attack target | **ABSENT** |
| `−ζ·L_R(θ′)` post-attack retain | present as `−lam_meta·L_benign(θ′)` (`:143`) |
| `+ζ·L_R(θ)` pre-attack retain | **ABSENT** — `distil_R(θ)` is teacher-output distillation, a different function with a different minimum |

✔ Two of three omitted. Variant A additionally carries **two terms not in Eq.8** (`distil_F(θ)`,
`distil_R(θ)`), which the prior claim does not mention.

### 2.3 "erase_wsr.py anchors TOWARD base while the paper maximises distance" — **CONFIRMED**

`upstream/vision_relearning/vision_unlearner.py:366-373`:
```python
param_diff       = flatten_param_dict(pretrained_model.parameters()).detach() - flatten_param_dict(model.parameters())
weight_dist_loss = - torch.sqrt(torch.mean(torch.square(param_diff)) + eps)   # eps = 1e-8
loss             = unlearning_alpha * weight_dist_loss
```
Negated ⇒ minimising maximises distance. `erase_wsr.py:158-163` computes `+λ·Σ(θ−θ_base)²/‖θ_base‖²`
⇒ minimising minimises distance. ✔ Opposite directions, plus the four structural differences
`objectives.py` already enumerates (all four independently confirmed).

### 2.4 "UTRAP's inner attacker differed from the evaluator on 7 axes" — **CONFIRMED, all 7**

| axis | UTRAP inner | evaluator | source |
|---|---|---|---|
| optimizer | SGD | Adam | `erase_utrap.py:140` / `attack_relearn.py:333` |
| optimizer state | rebuilt every outer step | persistent over 300 | `erase_utrap.py:140` (inside loop) |
| grad clipping | none | global norm 1.0 | — / `attack_relearn.py:362` |
| batch | 2 | 1 | `erase_utrap.py:55` / `attack_relearn.py:335` |
| horizon | K = 4/8/16 | 300 | `erase_utrap.py:49` / `attack_relearn.py:208` |
| target images | the already-erased teacher (loaded `:75`, `pipe` generates `:93`) | unmodified base | `attack_relearn.py:261` |
| target prompts | `["a photo of a {c}"]*16` (`:80`) — an evaluation prompt | 8 disjoint `T` prompts | `attack_relearn.py:43-47` |

Only lr (1e-5) and scope (full UNet) match. ✔

---

## 3. `attack_core.AttackSpec` divergence reporting — **accurate, but on a mirror (D2, D17)**

### ⚠ NO HISTORICAL RESULT IS AFFECTED BY D17. THIS IS DRIFT RISK ONLY. DO NOT QUARANTINE DATA.

**`EVAL_FULL_UNET`'s values currently MATCH `attack_relearn.py`'s inline loop exactly.** Confirmed
field by field, as asked:

| `EVAL_FULL_UNET` field | value | `attack_relearn.py` | match |
|---|---|---|---|
| `optimizer` | `"adam"` | `opt = torch.optim.Adam(atk_params, lr=a.lr)` `:333` | ✔ |
| `lr` | `1e-5` | `--lr` default `1e-5` `:209` | ✔ |
| `steps` | `300` | `--steps` default `300` `:208` | ✔ |
| `batch` | `1` | `i = torch.randint(0, len(lat), (1,), …)` `:335` | ✔ |
| `clip_grad_norm` | `1.0` | `clip_grad_norm_(atk_params, 1.0)` `:362` | ✔ |
| `scope` | `"full_unet"` | `atk_params = list(unet.parameters())` `:332` | ✔ |
| `persistent_state` | `True` | `opt` constructed once, outside the `for step` loop | ✔ |
| `precision` | `"fp32"` | `build_pipe(torch.float32, …)` | ✔ |

**Consequence, stated explicitly so nobody acts on D17 wrongly:** because the mirror is accurate
today, every `divergence_from_evaluation_attacker` record already written is correct, every
`is_the_evaluation_attacker` verdict is correct, and **no measured attack result, defence arm or
divergence record is invalidated by D17.** The defect is that the match is maintained *by hand*
rather than *by construction*: nothing fails if someone later edits `attack_relearn.py`'s loop, so
the two can diverge silently in future. That is a **maintenance hazard, not a data-integrity
problem**, and R6 is correspondingly low-priority (and high-risk — it touches the evaluator).
`divergence_from_eval()` reports accurately against the mirror, and `validate()`'s refusal to let
K<300 be labelled `EVALUATION_REFERENCE` works.

**But the file's headline claim was false.** It said it was "ONE implementation … shared by the
EVALUATION attack (`attack_relearn.py`)". `attack_relearn.py` does not import it:

```
$ grep -n "attack_core\|run_attack\|diffusion_mse" scripts/attack_relearn.py
(no match)
$ grep -ln attack_core scripts/*.py
scripts/attack_core.py  scripts/diag_coeff_scale.py  scripts/erase_utrap_v2.py  scripts/test_audit_repairs.py
```

So `EVAL_FULL_UNET` is a **hand-transcribed mirror** that can silently drift, and
`t_eval_spec_is_the_reference` (`test_audit_repairs.py:169`) tests the mirror against itself — it
cannot detect drift. Two axes the mirror structurally cannot cover: **RNG provenance**
(`attack_relearn.py` draws noise/timesteps/indices from the *global* torch stream; `diffusion_mse`
takes an explicit generator) and the attacker's **data** (in `AttackData`, outside the divergence
report). Docstrings repaired; closing the gap needs a behaviour change.

---

## 4. Coefficient scaling (task item 5)

### 4.1 FINDING67's 1/√N derivation — **VALID**, verified analytically and numerically

For `L = −α·√(mean(d²) + ε)` with `d = θ_pre − θ`:

```
∂L/∂p_j   = α·d_j / (N·√(S+ε))                     where S = mean(d²)
‖∇L‖      = α·‖d‖ / (N·√S) = α·√(NS)/(N·√S) = α/√N
```

Exactly `α/√N`, **independent of distance**. At N = 859,520,964 that is **3.411e-5** at α=1 —
matching the measured value to 4 s.f. Numerically confirmed on CPU via `objectives.wdr_forget_loss`
(ratio to `1/√N`: 1.000000 at RMS≈1.0, 0.999940 at N=1e6, 0.995 at RMS≈1e-3 where ε begins to bite).

### 4.2 FINDING67's **mechanism attribution is wrong**, and the "4,000×" is an artifact — **HISTORICAL RESULT needing reinterpretation**

FINDING67 §2 reads α=1 as inert because `‖∇distance‖ = 3.4e-5` sits against `‖∇retain‖ ≈ 0.137`, and
takes the ratio (≈4,017) as "the α needed for parity". **That frame does not apply to
`erase_wdr_v2.py`:** the two gradients are *never summed* — they go into two separate `opt.step()`
calls (`:161-176`). Under Adam a step is `lr·g/(√v + ε_adam)`, invariant to scaling the loss, so a
gradient-norm ratio between separately-stepped terms decides nothing.

Measured on CPU with a dedicated Adam and a constant gradient of per-coordinate size `α·1.16e-9`
(= α/N, the WDR term's actual per-coordinate magnitude at this N):

| α | step size as a fraction of `lr` |
|---:|---:|
| 1 | 0.104 |
| 10 | 0.537 |
| 4000 | 0.998 |

**α=1 → 4000 buys ≈9.6×, not 4,000×.** The real mechanism is narrower and is a property of *this
file's construction*: both steps share **one Adam instance**, so the second-moment buffer `v` is an
EMA mixing both terms and is dominated by the larger (retain) one. The distance step's update is
then `≈ lr·|g_dist|/√v_retain`, which *is* roughly linear in α — which is why E1b's observed ~285×
RMS change is consistent with the ratio. Three consequences:

- **α is inert at both ends** — below ≈10 because `v` is retain-dominated, above ≈4e3 because Adam
  saturates once the term dominates `v`. FINDING67 §6's "useful window α ∈ (1, 4000)" is right; the
  upper bound is a *saturation point*, not a frontier.
- The 4,000× factor is **not** the 1/√N parity it is attributed to, and **not** a property of the
  reference method, which sums into one step (§2.3 / D5). Under the reference's summed schedule the
  gradient-norm ratio *would* be the right frame. The faithfulness defect and the coefficient
  finding are entangled.
- `wdr2_church_a4000`'s retain8 = 0.5917 therefore measures *our two-step shared-Adam schedule at
  saturation*, not "Weight Dist Reg at gradient parity". FINDING67 §6's headline — "Ported to a
  859.5M-parameter UNet **at gradient parity**, WDR's distance-maximisation …" — should be restated
  without the parity attribution.

#### Citable numbers and sources for the FINDING67 correction

FINDING67 is owned by the coordinator; these are the figures and references behind §4.1–§4.2 so
they can be cited rather than paraphrased.

**The shared-buffer claim — source.** One `Adam` instance is constructed for the whole run and
stepped **twice per iteration**, once per term:

| what | file:line |
|---|---|
| the single shared optimiser | `scripts/erase_wdr_v2.py:200` — `opt = torch.optim.Adam(params, lr=a.lr)` |
| forget term + its own step | `scripts/erase_wdr_v2.py:206-210` (`l_forget` at `:208`, `opt.step()` at `:210`) |
| retain term + its own step | `scripts/erase_wdr_v2.py:212-227` (`l_retain` at `:224`, `opt.step()` at `:227`) |
| the reference SUMS instead | `upstream/vision_relearning/vision_unlearner.py:411` (alternation gated on an `alternating_` name prefix, which `weight_dist_reg` lacks) and `:637-682` (both `.backward()` calls, then one `optimizer.step()`) |
| the WDR loss itself | `upstream/vision_relearning/vision_unlearner.py:366-373`; our match in `scripts/objectives.py:wdr_forget_loss` |

Because `opt` is shared, its second-moment buffer `v` is an EMA over **both** terms' gradients and
is dominated by the larger (retain) one, so the forget step's update is `≈ lr·|g_dist|/√v_retain`
— roughly **linear in α** — rather than Adam's usual scale-invariant `≈ lr`.

**The gradient-norm identity.** `‖∇L‖ = α/√N` exactly, independent of distance (derivation in
§4.1). N = 859,520,964 ⇒ **3.411e-5 at α = 1**. Per-*coordinate* RMS gradient is `α/N`:

| quantity | α = 1 | α = 4000 |
|---|---:|---:|
| `‖∇ distance‖ = α/√N` | 3.411e-5 | 0.1364 |
| per-coordinate RMS = α/N | **1.16e-9** | 4.65e-6 |
| for comparison: retain, per-coordinate (0.137/√N) | 4.67e-6 | — |
| Adam default `eps` | 1e-8 | 1e-8 |

**Numerical verification of the identity** (CPU, `venv_t2i`, via `objectives.wdr_forget_loss`,
α = 1; ratio of measured `‖∇L‖` to `1/√N`):

| N | RMS distance | measured ‖∇L‖ | 1/√N | ratio |
|---:|---:|---:|---:|---:|
| 10,000 | 1.00486 | 1.000000e-02 | 1.0e-02 | **1.000000** |
| 1,000,000 | 0.01000 | 9.999400e-04 | 1.0e-03 | **0.999940** |
| 10,000 | 0.00101 | 9.951094e-03 | 1.0e-02 | 0.995109 (ε begins to bite) |

**Numerical demonstration that α is ~inert under a DEDICATED Adam** (CPU, constant per-coordinate
gradient `α·1.16e-9`, lr 1e-5, 3 steps; column is mean `|Δθ|` as a fraction of `3·lr`):

| α | step size / lr |
|---:|---:|
| 1 | **0.104** |
| 10 | 0.537 |
| 4000 | **0.998** |

⇒ **α = 1 → 4000 buys ≈ 9.6×, not 4,000×**, once the term has its own optimiser. The ~285× RMS
change actually observed in FINDING67 §6 is therefore attributable to the **shared** buffer, and α
**saturates** above ≈4e3 (Adam scale-invariance) as well as being swamped below ≈10.

### 4.3 "Gradient-norm parity treated as optimality" — **found, in FINDING67 §7**

> "`PREREG_H90`'s arms are specified at λ=0.15 … this says that choice **systematically
> under-weights** the adversarial term, and E4 should carry a λ≈1.2 arm alongside it."

Parity is being used as the normative reference point. It is not. For UTRAP the two gradients *are*
summed into one `p.grad` before one Adam step (`erase_utrap_v2.py:265-274`), so the **ratio** does
set the summed direction — parity is a defensible *directional heuristic* (it puts the resultant
≈45° between the two terms), but nothing establishes it as optimal, and Adam normalises the overall
scale away regardless. "Under-weights" should read "weights the adversarial term ~8× below the
retain term", with no implied optimum.

### 4.4 FINDING67 §7's "Meta-v2 needs no coefficient correction" — **withdrawn (D8)**

**Corrected framing, 2026-10-26, after coordinator review. My first write-up of this section
overreached and that conclusion is withdrawn; the defect itself is unchanged and is restated below.**

That diagnostic compared `‖∇L_retain(θ)‖ = 0.0387` against `‖∇L_target(θ′)‖ = 0.0927` as if they
were independent terms. **They are not.** `meta_unlearn_v2.py:289` and `:306` apply **one shared ζ**
to `+L_R(θ)` and `−L_R(θ′)` respectively, so the pair's net expected gradient is

```
E[ ζ·(∇L_R(θ) − ∇L_R(θ′)) ] ≈ ζ · α · H_R · ∇L_F        with α = --inner-alpha = 1e-5
```

— second order, ~1e-5 of either term alone.

**What this does and does not license.** Two claims must be kept apart:

- **"ζ is near-inert AS A COEFFICIENT" — supported.** Because the *same* ζ multiplies both members,
  the cancellation is **exact at every ζ > 0**. The quantity ζ actually scales is only the
  second-order *residual* above, which is ~1e-5 of a term and, with D9's unpaired draws, sits far
  below per-step sampling noise. So ζ cannot be tuned to buy more retention, and no ζ sweep has
  been run — it is inert **and** untested. The reference does not share the coefficient:
  `gamma1_1` and `gamma2_3` are independent arguments (`src/esd_meta.py:90-94`, all defaulting to
  1.0) and `scripts/esd_meta.sh` runs them **asymmetrically** (γ1_1 = 0.1 vs γ2_3 = 1.0;
  γ1_2 = 0.01 vs γ2_2 = 1.0 — that script also has a copy-paste bug passing `--gamma2_1` three
  times). Also: FINDING67 §7 measured Meta-v2 at "xattn scope, 44.0M" while `--meta-scope`
  defaults to `full`.

- **"the restored `+ζL_R(θ)` term does nothing" — NOT supported. WITHDRAWN.** I previously wrote
  that FINDING78 "cannot be attributed to the restored `+ζL_R(θ)` term". That conflated *the pair's
  net being small* with *the pairing having no effect*, and it is wrong. The near-zero net **is**
  the mechanism. Variant A carries `−lam_meta·∇L_R(θ′)` **alone** (`meta_unlearn.py:159-161`): a
  negative coefficient on the retain **level**, i.e. a large, non-cancelling, *biased* gradient of
  magnitude ~λ·0.04 that rewards making benign capability worse everywhere (θ′ ≈ θ to within
  α·‖∇L_F‖ ~ 1e-6). That is FINDING56's self-destruct-on-normal-use. Adding `+ζL_R(θ)` pairs the
  term and converts a **level** penalty into a **change** penalty, removing that incentive. Going
  from "large biased damage gradient" to "≈zero-mean" is a **first-order** change in the
  optimisation relative to variant A, even though the resulting term is second-order in absolute
  size. `meta_unlearn_v2.py:39-41` says exactly this — "its role is cancellation" — and that is
  correct.

**The accurate attribution**, which supersedes my earlier sentence: Meta-v2's retention gain is
attributable to **restoring the pairing** (a structural, ζ-independent effect) together with
`−λ_target·L_F(θ′)`; **ζ as a weight is near-inert and untested**. Tuning ζ up should not be
expected to buy retention. The fix for the unpaired estimator is **common random numbers (R2), not
a larger coefficient.**

**One refinement that makes R2 matter.** The cancellation is exact only **in expectation**. Per
step, D9's independent draws make the pair a difference of two *independent* O(0.04) gradients, so
each step injects zero-mean noise of magnitude ~ζ·√2·0.04 rather than zero. Zero-mean noise does
not *systematically* damage benign capability the way variant A's biased term did — so the
retention gain survives — but it does mean the logged `retain_change` field is sampling noise and
the per-step estimator is far noisier than Eq. 8 intends. **FINDING78's measured result stands.**

---

## 5. Defects requiring a BEHAVIOUR CHANGE — reported, not applied

Ranked by how much they affect a conclusion. **None was applied**; `FREEZE_ACTIVE` is in force.

> **LABELLING REQUIREMENT — applies to R1–R5 without exception.** Every one of these changes
> **alters numerical behaviour**. `meta_unlearn_v2.py` has produced FINDING78's arms and
> `erase_wdr_v2.py` has produced FINDING67's E1/E1b arms, so these are no longer unused files.
> None of R1–R5 may be applied to a script that produced a historical arm **unless it is introduced
> behind a flag defaulting to current behaviour, or under a new `implementation_version` and a new
> arm label.** Silently editing any of them in place would make the existing arms unreproducible
> and would repeat the `erase_adv.py` failure (D16) that this audit is documenting.

| # | file | change | changes numerics? | why it matters |
|---|---|---|---|---|
| **R1** | `meta_unlearn_v2.py` | split `--zeta` into `--zeta-pre` / `--zeta-post` | **YES** — new label required | lets the pre/post retain terms be weighted asymmetrically as the reference does, making ζ a *testable* coefficient instead of an inert one (§4.4). Note the goal is testability, **not** an expected retention gain — the pairing's benefit is already obtained at any ζ |
| **R2** | `meta_unlearn_v2.py` | draw noise + timesteps **once** and reuse them for the θ and θ′ evaluations (**common random numbers**) | **YES** — new label required. Must NOT be applied in place: it changes every per-step gradient draw and would silently invalidate FINDING78's arms | Eq. 8's paired difference across a 1e-5 parameter step is currently estimated from independent draws, so each step injects ~ζ·√2·0.04 of zero-mean noise and `retain_change` in the logs is sampling noise, not signal. **This, not a larger ζ, is the fix for the unpaired estimator** (§4.4). Expected effect is variance reduction, not a change in the mean objective |
| **R3** | `erase_wdr_v2.py` | sum the forget and retain gradients into **one** `opt.step()` | **YES** — new label required | restores the reference's actual schedule (`vision_unlearner.py:411,637-682`) and makes the gradient-norm/α analysis the correct frame instead of a shared-Adam artifact (§4.2) |
| **R4** | `meta_unlearn_v2.py` | route the simulated attack through `attack_core.AttackSpec` | **YES** if the attacker's configuration changes; **NO** if only the record is added | currently one plain GD step, no state, no clipping — vs a full AdamW epoch with clip 1.0 upstream (`esd_meta.py:963-977,1005-1008,617-622`) and 300 persistent Adam steps with clip 1.0 in the evaluator. Adding the *record alone* is safe and could be done now |
| **R5** | `erase_imma.py` | add a matched-scope arm (attn1+attn2, 93.5M) and outer `clip_grad_norm_(S, 1.0)` | **YES** — must be a NEW arm, not an edit to the existing one | the only way to say anything about the released IMMA rather than the paper's text (§2.1, D11–D13) |
| **R6** | `attack_relearn.py` *(not mine)* | make it call `attack_core.run_attack` | **YES, and it touches the EVALUATOR** — the highest-risk change on this list; it would re-baseline every attack number in the project | would make `EVAL_FULL_UNET` a real reference instead of a drift-prone mirror (§3, D17). See §3 for why this is *not* urgent |
| **R7** | `attack_core.AttackSpec` | add an `rng` axis | **NO** — reporting only | the global-vs-explicit generator difference is currently unreportable |

---

## 6. Repairs performed (documentation / metadata only — **no numerical change**)

All thirteen owned files re-parse (`ast.parse` clean). No training, generation or GPU work was run.

| file | repair |
|---|---|
| `attack_core.py` | withdrew the "shared by `attack_relearn.py`" claim in the module docstring, `diffusion_mse` and `run_attack`; documented `EVAL_FULL_UNET` as a hand-verified mirror and named the two axes it cannot cover |
| `erase_utrap.py` | replaced "the ACTUAL … attack" with the 7-axis divergence table + file:line evidence; corrected the inner-loop comment; added `inner_attacker_is_NOT_the_evaluation_attacker` to the artifact |
| `erase_imma.py` | `code_released: False` → `True` + repo/commit; documented the 2.13× scope gap, the missing outer clipping, Adam-vs-AdamW, and that "unassigned in paper" is false of the code; added `divergences_from_official_code` to the artifact |
| `objectives.py` | corrected the Meta citation (`src/meta.py` is a stale non-runnable copy → `src/esd_meta.py:992-1058`); **withdrew** the "the reference alternates" claim with `:411`/`:637-682` evidence; added the shared-ζ cancellation warning |
| `erase_wdr_v2.py` | withdrew faithfulness claim 5 (the reference **sums**); added the full ALPHA SCALING reinterpretation; noted the α=0-gradient no-op when `--teacher` is omitted; corrected the JSON `schedule` field and verified the optimiser/clipping defaults with line numbers |
| `meta_unlearn_v2.py` | corrected the citation; documented defects A (shared-ζ cancellation), B (unpaired noise), C (unrecorded inner attacker) in the docstring, at the relevant code lines, and in the artifact; noted the ESD-vs-distillation substitution |
| `meta_unlearn.py` | named it variant A; enumerated which 2 of Eq.8's 3 terms are absent and why `distil_R(θ)` is not one of them |
| `erase_wsr.py` | re-titled **WSR-anchor (ours)**; direction erratum with released-code evidence at the top; corrected "the third term is the paper's contribution"; replaced the artifact's `adapts_idea_of` / `tests_what_FINDING60_left_untested` claims with an explicit withdrawal |
| `erase_adv.py` | re-classified as our own design (neither Receler's adapter nor AdvUnlearn's text-encoder training); added `IMPL_VERSION = "adv/v2-inner-sign-repaired"` so artifacts predating the in-place sign repair are identifiable; recorded the classification in the artifact |
| `erase_frp.py` | provenance **re-verified and upheld** (FRAG README:99, arXiv id at :105); declared the LLM→diffusion domain adaptation; recorded the Eq.6 `+λ·rank(|W|)` sign as an open question |
| `erase_gift.py` | qualified the "faithful where our Meta port was not" comparison; marked `code_released:False` as unverified; recorded ψ = 44.0M / 5.12% |
| `erase_esd.py` | recorded that no ESD clone is pinned, and that `esd_u` keeps `time_embed`/`conv_out` and so differs from both the reported official `noxattn` and `meta_unlearn_v2.SCOPES["noxattn"]` |
| `erase_utrap_v2.py` | no change needed — its own docstring already states the divergences accurately; the mirror caveat is recorded in `attack_core.py` |

---

## 7. Classification of every item

**(1) CONFIRMED DEFECT** — D1 (`erase_utrap.py` false attacker-equivalence, docstring), D5
(`erase_wdr_v2.py` schedule claim inverted vs `vision_unlearner.py:411`), D6 (`erase_wsr.py:188`
direction), D7 (`meta_unlearn.py:159-161` omits 2 of 3 Eq.8 terms), **D8** (`meta_unlearn_v2.py:289`
+ `:306` — one shared ζ ⇒ the pre/post retain pair cancels, so **ζ is inert as a coefficient**;
this is *not* a claim that the pairing is ineffective — see §4.4), D9 (`meta_unlearn_v2.py:288`
vs `:305` unpaired noise/timesteps), D10 (`meta_unlearn_v2.py:297-301` inner attacker unrecorded),
D11 (`erase_imma.py` `code_released:False`), D12 (`erase_imma.py` no clipping vs
`defend_dreambooth.py:1297-1299`), D15 (`erase_adv.py` classification), D16 (`erase_adv.py` in-place
sign repair with no version field), D17 (`attack_core.py` false sharing claim — **documentation
only; no data affected, see §3**). **All repaired at the documentation level; D5, D8–D10, D12 also
need R1–R5, each of which requires a new arm label.**

**(2) SUSPECTED** — D3 (`erase_esd.py:64` `esd_u` scope vs official `noxattn`; no clone to check);
D4 (`erase_gift.py` comparative faithfulness claim); D13/D14 (IMMA Adam-vs-AdamW and
"unassigned in paper"); `erase_frp.py` Eq.6 `+λ·rank(|W|)` sign; GIFT's `code_released:False`.

**(3) VALID** — every objective **sign** in all ten methods (ESD `:95`, GIFT `:228,230`, IMMA `:180`,
UTRAP `:171`, UTRAP-v2 `:272`, variant A `:143`, Meta-v2 `:242`, AdvUnlearn `:124-127`, WDR-v2 `:162`,
FRP Eq.3/4 `:115-116`); all recomputed scope figures (44.0M / 49.6M / 93.5M / 148.7M / 859.5M);
`EVAL_FULL_UNET`'s field values; `AttackSpec.validate`'s horizon-role enforcement;
`objectives.wdr_forget_loss`'s formula, eps and sign against `vision_unlearner.py:366-373`;
`objectives.meta_terms`' signed structure against `esd_meta.py:992-1058`; WDR-v2 faithfulness
claims 1,2,3,4,6,7,8; `erase_frp.py`'s provenance claim; FINDING67's **1/√N derivation**.

**(4) HISTORICAL RESULT needing reinterpretation** — FINDING67 §2/§6 (the "4,000×" is a
shared-Adam-state artifact of a divergent schedule, not 1/√N parity; `wdr2_church_a4000` measures
our two-step schedule at Adam saturation); FINDING67 §7 (Meta-v2 "needs no coefficient correction" —
withdrawn; UTRAP λ "under-weights" — parity is not an optimum); **FINDING78 — refinement only, the
result STANDS** (the retention gain is attributable to restoring the *pairing* and to
`−λ_target·L_F(θ′)`; ζ *as a weight* is near-inert and untested, so no ζ sweep should be expected to
improve it — see §4.4; my earlier "cannot be attributed to the restored term" is **withdrawn**);
**D17 affects NO result — drift risk only, see §3**; all pre-repair `erase_adv.py` arms (not
adversarially trained — the
inner loop descended the erasure loss); all UTRAP arms (surrogate, not the evaluation attacker —
already covered by `ERRATUM_utrap_inner_attacker.md`); all `erase_wsr.py` arms (our own method, not
2505.22310); all IMMA arms (paper-text scope, 2.13× narrower than the released code).

**(5) REPAIR PERFORMED** — the thirteen rows of §6.

---

## 8. Note for Agent 1 (`erase_uce.py`, not mine)

Confirming the known defect from the released code:
`upstream/unified-concept-editing/trainscripts/uce_sd_erase.py:26-53` builds last-real-token
embeddings for **three** sets — erase, guide **and preserve** — and the closed form at `:56-82` is
`W_new = (λW_old + Σ_e s_e·v*cᵀ + Σ_p s_p·v*cᵀ)(λI + Σ_e s_e·ccᵀ + Σ_p s_p·ccᵀ)⁻¹`, scope attn2
`to_k`/`to_v` only, with `λ` default **0.5**, `erase_scale` 1, `preserve_scale` 1. λ multiplies both
`W_old` and `I`, i.e. it is a ridge anchor to the original weights — which is also relevant to
`ERRATUM_uce_lambda_scale.md`'s "λ ≈ 1e3" conclusion.
