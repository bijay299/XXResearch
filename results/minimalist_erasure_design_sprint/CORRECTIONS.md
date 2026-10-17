# Corrections — Minimalist erasure design sprint

**DOCUMENTATION-ONLY AMENDMENT. NO EXPERIMENTS, NO PROFILING, NO NEW SOURCE VERSION.**

This amends the publication at
[`53d3fb1`](https://github.com/bijay299/XXResearch/tree/53d3fb1c287b735176a8a1a192ec94ca48d1d3b3/results/minimalist_erasure_design_sprint),
which remains in Git history unchanged. The original author-source inspection is **reused**,
not repeated, at the same pin
`YaNgZhAnG-V5/minimalist_concept_erasure@d178324ec4a48d0536a595b25fdad587b441add8`.

The published record asserted **impossibility and strict-dominance results that it did not
establish**. Those proofs are withdrawn here. **The STOP is retained on narrower grounds.**

## Decision after correction

> **STOP — no distinct, justified improvement has been established. The proposed
> estimator's novelty, variance/convergence and actual total-cost benefit remain
> unresolved; fidelity of the source-defined reference to the publication is not
> established. No experiment is authorized.**

**STOP — DESIGN SPRINT CLOSED WITH CORRECTIONS.**

This is **not** an impossibility theorem and **not** a proof that estimators cannot save
work. The STOP now rests on *absence of an established, sufficiently distinct improvement*
and on *unresolved fidelity and performance* — **not** on any of the withdrawn proofs.

---

## A. Withdrawn impossibility and dominance reasoning

**Retained (correct):** the local multiplicative-gate derivative. With the gate acting as
`h ↦ m ⊙ h`, the mask derivative is **the incoming activation times the gate-output
adjoint, with the appropriate reductions** — `∂L/∂m_j = Σ_{b,t} h_{b,t,j} · ḡ_{b,t,j}`
(`DERIVATION.md` §4). This local identity stands.

| id | withdrawn claim (as published at `53d3fb1`) | correction |
|---|---|---|
| **X1** | The local identity proves **every estimator must traverse the full reverse chain on every draw**, so "there is no mask-gradient work to make cheaper" and no estimator can reduce it | **Withdrawn as a non-sequitur.** The identity bounds the *marginal* cost of a mask term *given* that the state adjoint at that site is already computed. It says nothing about estimators that **do not compute every state adjoint**. Two regimes must be distinguished: **(1)** if all state adjoints are always computed, skipping cheap local mask terms offers little saving — this is what the identity supports; **(2)** if reverse traversal **stops after the earliest selected contribution**, cost depends on a **random stopping depth**, and a nonzero probability of a full sweep does **not** force full expected cost. See the counterexample below |
| **X2** | Unbiasedness and cheaper expected work are **mutually exclusive** here ("unbiasedness forces the sweep back through every step") | **Withdrawn.** Nonzero probability of reaching depth 0 does not imply expected cost equal to a full sweep. ARTBP (`arXiv:1705.08209`) is a published construction obtaining **unbiased** gradients from **variable-length** truncation via compensation factors, i.e. the combination is not contradictory |
| **X3** | SSMG is **strictly dominated** and **"cannot break even for any `q`, `S₀`, `R` or `W`"** | **Withdrawn.** No such universal statement was established, and the counterexample below exhibits a positive-probability `q` with expected main cost **below** the published `A2` figure |
| **X4** | **"All available computational headroom is exact and ordinary"** | **Withdrawn.** This followed from X1–X3 and does not survive them |
| **X5** | `C_floor = K·F·(1+c_b)` is a **lower bound** for the exact relaxed gradient, which `A2` **attains**, so "there is no exact-gradient headroom below `A2`, and in particular none that an estimator could reach" | **Withdrawn as a theorem.** One forward plus one full reverse traversal per step is an **ordinary reverse-mode cost model** for the coherent full-chain estimator — **not a universal lower bound** over estimators or implementations. The expression is retained *as a cost model*, with "floor" language removed |

### The supplied counterexample (refutes the universal proof)

Under the published simplified expression, with reverse traversal stopping at the earliest
selected contribution and one draw `I ~ q`:

```
E[C]  =  K·F  +  (1 + c_b)·F · Σ_i q_i (K − i)  +  selection overhead
```

With `K = 5`, `c_b = 1`, one draw, and **all probabilities positive**

```
q = (1/20, 1/20, 1/20, 1/20, 4/5)
```

```
Σ q_i = 1                     E[reverse depth] = 14/20 + 16/20 = 3/2
E[C] main = 5F + 2F·(3/2)     = 8F        vs. published A2 = 10F
```

**A saving is possible within this model if additional overhead is below `2F`.**
Verified by exact rational arithmetic (§E).

**What this counterexample is not.** It is **not** evidence of a FLUX speedup, **not**
evidence of acceptable variance or convergence, and **not** evidence of novelty. It
refutes the universal proof **only**. It is **not** adopted as a new algorithm proposal,
and no selection distribution is proposed or recommended anywhere in this record.

---

## B. Corrected variance and optimization statements

| id | withdrawn / corrected | correction |
|---|---|---|
| **X6** | Importance-sampling variance is **"zero only if all but one `v_k` vanish"** | **Withdrawn.** Several **nonzero** contributions give **exactly zero** variance when they are positively collinear and probabilities are proportional to their magnitudes. The record's own example suffices: `v = (10, 15)`, `q = (2/5, 3/5)` ⇒ both estimates equal `25` ⇒ **variance 0**. The variance **formula** is retained with its assumptions |
| **X7** | `S ⊂ {0…K−1}` described as a **set** while drawn **with replacement** | **Corrected.** With independent sampling with replacement, draws must be counted **with multiplicity** — an ordered list or multiset, not a deduplicated set |
| **X8** | Added variance ⇒ **`ν ≥ 1`**, so "the update count can only rise"; hence `ν·ρ ≥ 1` | **Withdrawn.** Update requirements in the actual nonlinear optimization are **unknown**. Greater gradient variance does not entail more updates to comparable effectiveness. DRaFT's reported results do **not** establish that ordering for Minimalist |

---

## C. Repaired cost model

| id | withdrawn / corrected | correction |
|---|---|---|
| **X9** | "A free gradient is worth at most **`1.33×`** over `A2`" (presented as *the* Amdahl ceiling) | **Corrected arithmetic.** Under `c_b = 1`, `A2 = 10F` and a hypothetical free-gradient forward costs `5F`, so the conditional ratio is **`2`, not `1.33`**. `1.33×` was the *including-instrumentation* figure, mislabeled. Every ceiling is now stated **with its actual included costs**: vs `C_ref` gradient-only **`4×`**; vs `A2` gradient-only **`2×`**; vs `C_ref` incl. instrumentation **`2×`**; vs `A2` incl. instrumentation **`4/3×`** |
| **X10** | Break-even budget **`ν < 1.23–1.56`**, obtained by comparing whole-run totals that **included shared one-time preparation** | **Corrected.** Shared one-time preparation must **not** be multiplied by update inflation. With `T_base = P_base + N·c_base` and `T_new = P_new + H + ν·N·c_new`, shared `P` and `H = 0` give **`ν < c_base/c_new`**: **`ν < 10/6 = 5/3`** (`c_b=1`) and **`ν < 15/7`** (`c_b=2`). Warm-up and refresh work is to be counted as replacing or adding **iterations**, without double counting. **These corrected thresholds still do not establish useful actual-model savings** |
| **X11** | `A1` is **"bit-identical"**, gives **"25–33% less work"**, and leaves the **"optimization trajectory unchanged"** | **Corrected.** `A1` can return both requested derivatives **from the same recomputed graph in exact arithmetic**. Claims of **bit identity** (floating-point accumulation order may differ), of a **fixed runtime saving**, and of an **identical optimization trajectory** are withdrawn as unverified |
| **X12** | `A2` is an **"exact"** alternative to the native computation, differing only in memory | **Corrected.** `A2` retains a **coherent forward graph**, whereas native manual recomputation **redraws mask noise**. `A2` is therefore **not an exact substitute for the native estimator** — it is a *different stochastic reference*. Every cost expression now states **which stochastic reference** it concerns |
| **X13** | RNG-preserving replay (`A3`) treated as establishing what the native gradient "should" be | **Corrected.** RNG-preserving replay **changes the reference contract**; it is **not** proof of an unchanged native gradient |
| **X14** | Composition **"gradient 54% / instrumentation 27% / preparation 19%"** | **Withdrawn as stated.** It omitted cost terms (X18) and, because preparation is **fixed**, composition **depends on the iteration count**. Retained only as an explicitly labelled illustrative decomposition of the terms actually modelled |
| **X15** | Data preparation is **"the pipeline's global peak memory"** | **Withdrawn.** Two resident models establish a **preparation memory burden**, not proof that it **exceeds the unknown training activation peak** |
| **X16** | Whole-run predicted ratios **`2.16×`, `2.81×`, `2.12×`, `3.32×`** and "published-run speedup" characterizations | **Withdrawn.** They depend on X9, X10, X14 and the omitted terms of X18 |
| **X17** | If the paper's 400 updates is authoritative, "all **totals** scale, while all **ratios** are unchanged" | **Corrected.** Preparation is **fixed**, so whole-run **ratios and composition also depend on the iteration count** |
| **X18** | Cost terms **omitted entirely** | **Added symbolically, or explicitly excluded**: **prompt encoding inside each training forward preparation** (`inference_preparation_phase` calls `encode_prompt`, `models/flux.py:250` — a **per-iteration** text-encoder cost the published model ignored); initialization; VAE decodes; batching; optimizer work; I/O; instrumentation; **separate `c_state`, `c_mask`, `c_joint` VJP costs** instead of a single `c_b`; and **memory feasibility of retaining graphs** |

---

## D. Corrected objective and stochastic-gradient interpretations

| id | withdrawn / corrected | correction |
|---|---|---|
| **X19** | Per-example term selection (`train.py:284-297`) is a **"different empirical objective"** than the paper's two-term sum | **Corrected.** At fixed parameters, a **balanced uniform** forget/retain row mixture **estimates `½(L_forget + β·L_retain)`**. Choosing one term per example is **not automatically a different empirical objective**. Retained separately: **alternating updates need not reproduce the finite-step trajectory** of computing both terms together; and the **genuine** differences are **target construction** and **fixed-seed coverage** |
| **X20** | Absence of an `L0`/sparsity term listed as a **paper/code divergence** | **Corrected.** The cited paper objective (Eq. 16) **also lacks** such a term, so its absence is **not** a divergence. It remains a **source fact** that `calculate_reg_loss` is configured but never called from `scripts/train.py` |
| **X21** | A single fixed `x_T` means "`E_{x_T}` is **not estimated**"; unbiasedness "holds only in an average-over-`ξ` sense that the implementation does not realize — it uses **one** `ξ` per iteration" | **Corrected.** A **coherent single reparameterized sample can be unbiased** under the required regularity conditions; **multiple samples within an iteration are not required**. **Initial-latent randomness and mask randomness are separate axes.** The genuine, retained point is narrower: the **fixed seed limits coverage of `x_T`** across iterations |
| **X22** | "The clamp set has **measure zero** in `λ` for fixed `ξ`, so a.e. differentiability is not violated" | **Corrected.** **Clamped regions can have positive measure and zero derivative.** They must be distinguished from **transition boundaries**, which are the measure-zero set |
| **X23** | The implemented quantity "is **not** an unbiased estimate of `∇_λ J`" and is a "mismatched-realization estimator" whose bias exists | **Corrected.** Fresh replay **does not enforce conditional pathwise consistency** and **removes a general unbiasedness guarantee** — but **nonzero bias of this actual FLUX system was not analytically established**. No measured or derived native bias is claimed |
| **X24** | `image[0]` at `train.py:294` indexes **batch element 0**, so the forget term "takes batch element 0 only" | **Withdrawn.** In the declared compatible Diffusers (`pyproject.toml:19`, `diffusers>=0.31.0`) `BaseOutput.__getitem__` is `return self.to_tuple()[k]` for non-`str` keys, so `image[0]` returns the **first declared output field** (`.images`) — the **whole latent tensor**, not a batch element. The batch-index diagnosis is **wrong and withdrawn** |
| **X25** | "Until this is settled **there is no reference objective** to be cheaper than" | **Corrected.** A **source-defined empirical reference objective can be stated** even while publication fidelity is unresolved. The accurate statement is that **alignment with the publication is unverified** |
| **X26** | All code arriving in one commit four months after paper v1, used to support that the snapshot is not the publication implementation | **Corrected.** **Later source-release dates alone do not prove a different algorithm was used for the publication.** Retained only as a dating fact, with fidelity recorded as **unverified** |

---

## E. Corrected prior-art attribution and falsification status

| id | withdrawn / corrected | correction |
|---|---|---|
| **X27** | "Any truncation variant here **reduces to DRaFT-K**", and SSMG's only escape is unbiasedness | **Corrected.** DRaFT §4 supports **existing deterministic suffix truncation and checkpointing** — *"truncates backpropagation to only the last K steps of sampling"*, implemented with `x_t = stop_grad(x_t)` and **no reweighting**. **ARTBP** supports **unbiased reweighted randomized truncation** — *"variable truncation lengths together with carefully chosen compensation factors"*, *"if a flow has a probability p to occur, multiplication of the flow by 1/p restores balance on average"*. These raise a **novelty objection to ordinary importance weighting**, **not** an impossibility of expected savings, and **not** equivalence of every variant |
| **X28** | Decisive method-content and strict-dominance claims drawn from sources recorded as **abstract/summary-level only** | **Removed.** DRaFT is now **targeted-inspected at §4** and ARTBP at abstract + mechanism level; `SOURCE_LEDGER.md` records the corrected depths. AlignProp remains **abstract-only** and carries **no decisive claim**. No broad literature survey was performed and **no novelty finding is claimed** |
| **X29** | A future design **must** change the gate, **must** defeat the local derivative identity, or **must** reduce forward work "to be mathematically possible"; and `ν` is **mandatory knowledge** before a plan can be written | **Withdrawn as invalid requirements.** None is a necessary condition. **Empirical `ν` can be measured by a future prospective test**; it is not a precondition for writing one |
| **X30** | "**Falsify the stop by falsifying the lemma**" | **Corrected.** The STOP no longer rests on the lemma, so falsifying the lemma would not overturn it. The STOP rests on the **absence of a sufficiently distinct, justified selected improvement** and on **unresolved fidelity and performance** |

**Unchanged:** no execution-ready falsification plan exists, and **no experiment is
authorized**. No comparison plan was manufactured or executed.

---

## F. What is retained from the original publication

- The **local multiplicative-gate derivative** and the gate sites
  (`ff_hooks.py:87`, `norm_hook.py:92`, `attention_processor.py:64`, `epsilon: 0.0`).
- The **reverse accumulation** (★) as the computation the native loop performs, and its
  two-sweep structure (`train.py:320`, `:333`).
- **Source-supported configuration and count facts**: `K = 5`; `epochs 5`; `size 40`;
  `batch_size 1`; `accumulate_grad_batches 4`; `plot_interval 4`; `β = 0.1`;
  `epsilon 0.0`; `attn_lr 0`; `init_lambda 3`; `masking_eps 0.5`; absent `img_size`.
- **Targets precomputed and cached**, so no teacher forward occurs per iteration.
- **No VAE decode in the training graph** (`output_type="latent"`).
- The **genuine** paper/code questions: **target construction** (Eq. 16's `ℱθ′(x_T, ∅)` vs.
  the default FlowEdit-on-FLUX.1-dev construction) and the **`β` / batch-size / update-count**
  disagreements — all recorded as **unresolved**, not adjudicated.
- The **gate-saturation arithmetic** (`P(m = 1) = 0.9659` at `λ = 3`; bounded noise
  `[−1.0986, +1.0986]` at `eps = 0.5`), with X22's corrected measure-theoretic reading.
- The **attempted formulation (SSMG)** itself, preserved as a record — **not** promoted to
  a proposal.
- **No experiments, no profiling, no measurements** anywhere in the record.

## G. Arithmetic verification of this amendment

Tiny exact deterministic arithmetic only (`fractions.Fraction`; scalars; no tensors, no
model, no simulation, no sweep):

- **X1/X3 counterexample:** `Σq = 1`, `E[depth] = 3/2`, `E[C] main = 8F` vs `A2 = 10F`,
  saving possible iff overhead `< 2F`.
- **X6 counterexample:** `v = (10,15)`, `q = (2/5,3/5)` ⇒ estimates `(25, 25)`, mean `25`,
  **variance `0`**, with `q ∝ |v|` confirmed.
- **X9 ceilings:** `4`, `2`, `2`, `4/3` for the four stated cost bases.
- **X10 break-even:** `5/3` and `15/7`.

**Scalar counterexamples are not model evidence** and are not reinterpreted as such.

---

## H. Scope of this amendment

Documentation only. No code, no configuration, no experiment, no implementation, no
follow-up job. No new source version was fetched; the existing inspection at
`d178324` is reused. The original publication at `53d3fb1` is preserved in Git history.

**Closed work is untouched and not reranked:** the gap map at `edaec97`, the
changing-replacement closure at `5786864`, the FLUX diagnostic at `0e88d26` with its
original results and **unused budget**, the SD1.x and human-review closures and pauses, and
previously unselected theory/evaluation formulations. Their directories were not modified.
