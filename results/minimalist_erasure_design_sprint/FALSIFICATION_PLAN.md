# Falsification plan — **not produced: no experiment is justified**

**NO EXPERIMENTS.** This file records *why* no prospective comparison is warranted and what
remains outstanding. It is **not a plan to execute**, it is **not execution-ready**, and it
**authorizes nothing**.

> **Amended 2026-10-17 — see [`CORRECTIONS.md`](CORRECTIONS.md).** The original publication
> required a future design to **defeat the local derivative identity**, **change the gate**,
> or **reduce forward work** "to be mathematically possible", and treated empirical `ν` as
> **mandatory knowledge** before a plan could be written. **All of these requirements are
> withdrawn as invalid** (X29). The instruction to *"falsify the stop by falsifying the
> lemma"* is also withdrawn (X30): the STOP no longer rests on that lemma.

---

## 1. The gate for building a plan was not met

The sprint's condition was: *"Only if a specific defensible design remains, describe one
minimal future comparison."*

**No sufficiently distinct, justified improvement was established.** The one formulation
developed (SSMG, [`ALGORITHM.md`](ALGORITHM.md)) has **unresolved novelty**, **unresolved
variance/convergence**, and **unresolved actual total-cost benefit**; and the
source-defined reference objective's **fidelity to the publication is unverified**.
Accordingly **no comparison is specified here** — no arms, no tolerances, no seed counts, no
resource ceiling, no case selection.

**This is a "not established" stop, not an impossibility stop.** The record no longer claims
a design *cannot* succeed; it claims **this one was not shown to**.

### 1.1 Not created, by design

No dataset, no case manifest, no evaluation package, no launcher, no config, no
implementation or trainer patch, no adapter. No closed image, prompt, label or outcome was
read, re-scored or mined — in particular nothing from the FLUX diagnostic at `0e88d26`,
whose results and **unused image budget remain untouched and unspent**.

---

## 2. What would have to be resolved to reopen the question

These are **the actual open questions**, not necessary conditions on a design's form.

**(a) Novelty against ARTBP and DRaFT.** Establish an **additional useful technical claim**
for a reweighted/randomized truncation scheme on this graph beyond **ARTBP**
(`arXiv:1705.08209`, unbiased reweighted variable-length truncation) and **DRaFT** §4
(deterministic suffix truncation with checkpointing). We did **not** resolve this, and we
assert **no** novelty finding in either direction.

**(b) Variance and convergence.** `ν` — the update-count factor to reach comparable
deletion, retained utility, rendering quality and robustness — is **unknown**.
**Corrected (X29): `ν` is measurable by a future prospective test and is *not* a
precondition for writing one.** It is listed here as an open quantity, not a gate.

**(c) Net cost after the unmodelled terms.** Corrected break-even is `ν < c_base/c_new`
(`5/3` at `c_b = 1`, `15/7` at `c_b = 2`, [`COST_MODEL.md`](COST_MODEL.md) §6) — a
**nonvanishing but unresolved** condition. Resolving it requires accounting for
per-iteration prompt encoding `F_txt`, selection overhead, distinct
`c_state`/`c_mask`/`c_joint`, optimizer/I/O work, and the memory feasibility of graph
retention.

**(d) Reference fidelity.** Settle which erasure target is authoritative: paper Eq. 16's
`ℱθ′(x_T, ∅)`, the code default's **FlowEdit on FLUX.1-dev** at `T_steps=28`, or the
`CON_DECON_DICT` neutral-prompt branch; plus the `β`, batch-size and update-count
disagreements. **A source-defined reference objective can be stated** — the accurate
statement is that **its alignment with the publication is unverified** (X25). This is a
documentation/authorship question, not an experimental one, and not ours to settle
unilaterally. We did not patch the trainer, substitute a baseline, or claim a reproduction.

### 2.1 Explicitly withdrawn requirements (X29)

The following were published as things a future design **must** do. **None is a necessary
condition**, and all are withdrawn:

- ~~Exhibit a mask parameterization for which the mask-Jacobian VJP is *not* an elementwise
  byproduct of the state VJP~~ — the local identity is **conditional on the state adjoint
  being computed**; early-stopping estimators are not bound by it.
- ~~Change the gate (e.g. to dense weight masking) — otherwise it "is no longer a cheaper
  gradient for native Minimalist"~~.
- ~~"Beat the Amdahl ceiling", i.e. reduce the **forward** rather than the gradient~~ — this
  rested on the withdrawn `1.33×` figure (X9) and on reading an objective-specific forward
  requirement as a bound on estimators.
- ~~Supply `ν` before a plan can be written~~ — see (b).

---

## 3. Confounds any future comparison would have to control

Recorded because they are easy to get wrong, not because a comparison is contemplated.

1. **Which stochastic reference each arm estimates.** The native trainer redraws gate noise
   on every recompute with no RNG stashing (`DERIVATION.md` §5.3), so it does **not** enforce
   conditional pathwise consistency. **RNG-preserving replay changes the reference contract
   and is not proof of an unchanged native gradient** (X13). `A2` likewise retains a
   **coherent** graph and is **not** an exact substitute for the native estimator (X12).
   Arms differing here are estimating with respect to **different references**.
   **No native bias is claimed as derived or measured** (X23).
2. **Gate saturation at initialization.** `P(m = 1) = 0.9659` at `λ = 3`, where
   `∂m/∂λ = 0`. The **clamped region has positive measure** and must be distinguished from
   the measure-zero **transition boundary** (X22). Gradient-signal density changes by ~20×
   between `λ = 3` and `λ = 1`, so fixed-iteration-budget comparisons may span different
   optimization regimes.
3. **Instrumentation.** `plot_interval: 4` drives 8 image generations (with decodes and
   their own prompt encoding) every 4 iterations. Any arm changing it reports a logging
   difference as a method difference. **The previously published "27% of total work" figure
   is withdrawn** (X14).
4. **Baseline choice.** A method must be compared against **correct ordinary autograd usage**
   (`A1`), not only against the native two-sweep loop — otherwise an apparent speedup is
   just sweep fusion. `A1` carries **no verified runtime or trajectory claim** (X11).
5. **The binarization threshold is hardcoded to 0** with an acknowledged `TODO`
   (`scripts/train.py:398-399`) and unspecified in the paper. Deletion and utility are
   properties of the **binarized** mask, so this is a free parameter on every outcome metric.
6. **`A2`'s feasibility is an unmeasured memory question** (≈5× activation memory,
   [`COST_MODEL.md`](COST_MODEL.md) §4.2).

---

## 4. Outstanding values — explicitly unresolved

| quantity | status |
|---|---|
| `F`, `F_txt`, `c_state`, `c_mask`, `c_joint`, `D`, `F_dev` | **unmeasured**; no profiling authorized or performed |
| `ν` | **unknown**; measurable by a future prospective test, **not** a precondition for one |
| selection/estimation overhead for any sampling scheme | **unquantified** |
| novelty vs. ARTBP / DRaFT §4 | **unresolved**; no finding claimed |
| authoritative erasure target | **unresolved** (§2d) |
| authoritative `N_upd` | **unresolved**: 80 (config) vs 400 (paper Table 6) |
| authoritative `β` | **unresolved**: 0.1 (config) vs 0.01 (paper) |
| authoritative batch size | **unresolved**: 1 (config) vs 4 (paper) |
| `|D|` after CLIP filtering | **≈64, data-dependent, not computed** |
| binarization threshold / sparsity ratio | **unspecified** in the paper; hardcoded `0` in training-time logging |
| whether the repo snapshot is the publication implementation | **unverified**. **Later release dates alone do not prove a different algorithm was used** (X26) |
| peak-memory feasibility of graph retention | **unmeasured** |

---

## 5. If a future reviewer disagrees with the stop

**Corrected (X30).** The STOP does **not** rest on the local derivative identity, so
falsifying that identity would not overturn it. The STOP rests on the **absence of an
established, sufficiently distinct and justified improvement** together with **unresolved
fidelity and performance** (§2a–d).

To overturn it, resolve §2(a) — an additional useful technical claim beyond ARTBP and
DRaFT — and then §2(c). Both are **analytical and documentary** questions in the first
instance; neither requires a GPU, and none is authorized here. §2(b) is the one genuinely
empirical item, and it would belong to a **separately reviewed and separately authorized**
future test, not to this record.

**Reuse note.** Should the question reopen, existing unchanged assets and infrastructure may
be *proposed* for reuse. **Closed outcomes are not fresh confirmation** and must not serve as
evidence for any new method.

---

## Status

**STOP — DESIGN SPRINT CLOSED WITH CORRECTIONS. No experiment is justified. No plan is
execution-ready. No comparison plan was manufactured or executed. No follow-up job, monitor
or scheduled task was created, and no future computation is authorized.**
