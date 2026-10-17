# Falsification plan — **not produced: no experiment is justified**

**NO EXPERIMENTS.** This file records *why* no prospective comparison is warranted and what
would have to be true first. It is **not a plan to execute**, it is **not execution-ready**,
and it authorizes nothing.

---

## 1. The gate for building a plan was not met

The sprint's own condition was: *"Only if a specific defensible design remains, describe one
minimal future comparison."*

**No design remains.** [`ALGORITHM.md`](ALGORITHM.md) records the one formulation developed
(SSMG) and its exact dominance; three of four technical stopping conditions fire
independently. Accordingly **no comparison is specified here** — no arms, no tolerances, no
seed counts, no resource ceiling, no case selection.

Writing a plan anyway would mean manufacturing an executable artifact to satisfy a file
list. It would also be actively misleading: **there is nothing to falsify**, because there
is no improvement claim. The honest deliverable is this statement.

### 1.1 Not created, by design

No dataset, no case manifest, no evaluation package, no launcher, no config, no
implementation patch, no adapter. No closed image, prompt, label or outcome was read,
re-scored or mined — in particular nothing from the FLUX diagnostic at `0e88d26`, whose
results and **unused image budget remain untouched and unspent**.

---

## 2. What would have to change to reopen the question

All four are necessary. Any one unmet leaves the question closed.

**(a) Resolve the paper/code objective mismatch.** Establish which erasure target is
authoritative: paper Eq. 16's `ℱθ′(x_T, ∅)`, or the code's default FlowEdit-on-FLUX.1-dev
target at `T_steps=28`, or the `CON_DECON_DICT` neutral-prompt branch
(`DERIVATION.md` §1.2a). **Until this is settled there is no reference objective to be
cheaper than.** This is a documentation/authorship question, not an experimental one — and
it is not ours to settle unilaterally. We did not patch the trainer, substitute a baseline,
or claim a reproduction.

**(b) Defeat the dominance lemma.** Exhibit a mask parameterization for which the per-step
mask-Jacobian VJP is **not** an elementwise byproduct of the state VJP — e.g. dense weight
masking `W ⊙ M`, or a low-rank/hypernetwork map `λ ↦ m`
([`ALGORITHM.md`](ALGORITHM.md) §1.4). Native Minimalist is explicitly the opposite case
(§3.6: *"neuron masking instead of weight masking"*), so **(b) requires changing the method,
at which point it is no longer a cheaper gradient for native Minimalist** — it is a
different method, needing its own deletion/utility/quality/robustness case.

**(c) Beat the Amdahl ceiling.** The irreducible `K·F` forward caps a *free* gradient at
`1.33×` over ordinary correct autograd usage at `c_b = 1`
([`COST_MODEL.md`](COST_MODEL.md) §4.3). A design must clear that ceiling, which means
reducing the **forward**, not the gradient — a different question from this sprint's.

**(d) Supply evidence on `ν`.** Any biased estimator needs the update-count inflation to
stay under `ν < 1.23–1.56` ([`COST_MODEL.md`](COST_MODEL.md) §6). `ν` is **unknown**, and
DRaFT reports truncation depth trading against result quality. A margin that thin cannot be
assumed.

---

## 3. Confounds any future comparison would have to control

Recorded because they are easy to get wrong, not because a comparison is contemplated.

1. **Mask-RNG consistency.** The native trainer redraws gate noise on every recompute with
   no RNG stashing, so it does **not** compute the pathwise gradient of its own trajectory
   (`DERIVATION.md` §5.3). An arm that stashes RNG (torch's `preserve_rng_state` default)
   computes a **different estimator**. Comparing a "cheaper" method against the native
   reference without pinning this measures the confound, not the method.
2. **Gate saturation at initialization.** `P(m = 1) = 0.9659` at `λ = 3`, where
   `∂m/∂λ = 0` exactly (`DERIVATION.md` §1.3). Gradient-signal density changes by ~20× as
   `λ` moves from 3 to 1, so any fixed-iteration-budget comparison is comparing different
   optimization regimes.
3. **Instrumentation.** At `plot_interval: 4` logging is 27% of total work
   ([`COST_MODEL.md`](COST_MODEL.md) §5). Any arm that changes it reports a logging
   difference as a method difference.
4. **`A1`/`A2` as the mandatory baseline.** A biased method must be compared against
   **exact ordinary autograd usage**, never against the unfixed two-sweep reference.
   Measured against the reference, a 2× "speedup" is just the fused VJP.
5. **The binarization threshold is hardcoded to 0** with an acknowledged `TODO`
   (`scripts/train.py:398-399`) and unspecified in the paper. Deletion and utility are
   properties of the **binarized** mask, so an unspecified threshold is a free parameter
   sitting directly on every outcome metric.
6. **`A2`'s feasibility is an unmeasured memory question** (`≈5×` activation memory,
   [`COST_MODEL.md`](COST_MODEL.md) §4.2). If it does not fit, the floor is unreachable and
   the honest baseline is `A1`.

---

## 4. Outstanding values — explicitly unresolved

| quantity | status |
|---|---|
| `F`, `c_b`, `D`, `F_dev` | **unmeasured**; no profiling authorized or performed |
| `ν` (update-count inflation to comparable effectiveness) | **unknown**; no evidence |
| authoritative erasure target | **unresolved** (§2a) |
| authoritative `N_upd` | **unresolved**: 80 (config) vs 400 (paper Table 6) |
| authoritative `β` | **unresolved**: 0.1 (config) vs 0.01 (paper) |
| authoritative batch size | **unresolved**: 1 (config) vs 4 (paper) |
| `|D|` after CLIP filtering | **≈64, data-dependent, not computed** |
| binarization threshold / sparsity ratio | **unspecified** in the paper; hardcoded `0` in training-time logging |
| whether the repo snapshot is the publication implementation | **not established** (all code in one commit, 2025-11-24, vs. paper v1 2025-07) |
| peak-memory feasibility of `A2` | **unmeasured** |

---

## 5. If a future reviewer disagrees with the stop

The decisive claim is narrow and checkable **without any experiment**: the lemma that
`∂L/∂m_j = Σ_{b,t} h_{b,t,j} · ḡ_{b,t,j}` makes every mask gradient an elementwise byproduct
of the state-chain reverse sweep (`DERIVATION.md` §4), which rests on the gate being exactly
`h ↦ m ⊙ h` at `epsilon = 0` — verifiable by reading four lines
(`hooks/ff_hooks.py:87`, `hooks/norm_hook.py:92`, `hooks/attention_processor.py:64`,
`configs/flux.yaml:17`).

**Falsify the stop by falsifying the lemma, by static reading.** No GPU is needed, and none
is authorized. If the lemma fails — a mask site we missed, or `epsilon > 0` in the published
run — the estimator question reopens on its merits, and this sprint's conclusion should be
revisited rather than defended.

**Reuse note.** Should the question ever reopen under §2, existing unchanged assets and
infrastructure may be *proposed* for reuse. **Closed outcomes are not fresh confirmation**
and must not be used as evidence for any new method.

---

## Status

**No experiment is justified. No plan is execution-ready. No follow-up job, monitor or
scheduled task was created, and no future computation is authorized by this sprint.**
