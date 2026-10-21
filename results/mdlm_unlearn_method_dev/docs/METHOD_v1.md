# The method, v1 — SIO: Selective Interdiction with Orthogonalised suppression

First statement of a concrete method. Every component is here because a measurement put it
here, and each is separately ablatable. Nothing below is a result; the method is specified
and not yet trained.

## What it has to beat, measured

| arm | DISCLOSE | DEGEN | REPLACE | IGNORANT |
|---|---|---|---|---|
| memorized θ₀ | 1.000 | 0.000 | 0.000 | 0.000 |
| MDU / EMDU / NPO (9 of 15 grid points) | 0.000 | **1.000** | 0.000 | 0.000 |
| exclusion / SC-TF | **~0.45** | 0.14–0.22 | 0.31–0.38 | 0.016 |
| **pristine TOFU base — the ignorance profile** | **0.030** | 0.000 | **0.470** | 0.3875 |

The target is **profile matching**: reach the pristine profile with DISCLOSE the only cell
that has to move. Substitution at 0.470 *is* ignorance and is not a failure — that premise
was carried all day and is corrected.

## The three components

### S — Selective interdiction over a measured-minimal token set

Apply the forget objective only to the top-`k` author-salient tokens of each answer, not to
the whole answer span. `k` is **measured, not chosen**: because the metric is a fractional
recall threshold, crossing it needs `k > n_sal·(ASCR₀ − τ)`, giving mean `k_min = 5.07` of
mean `n_sal = 8.01`.

So "interdict one token" is dead on arithmetic — `k = 1` clears the bar for **<1%** of items.
What survives is a **footprint reduction**: 5.07 stems is 14.7% of the 34.4-token answer and
5.3% of the canvas, against **100% for every existing method** — roughly 7×, not 20×.

*Why it should help:* uniform pressure across every answer token is the structural cause of
total collapse, which is what 9 of 15 grid points do. Less pressure, placed where
identifiability lives.

### O — Orthogonalised suppression (derived from the obstruction, measured viable)

The obstruction: suppressing an interdicted token drags the kept tokens down by
`Δz_j(v_j) ≈ −η·⟨a_i,a_j⟩·⟨w_{v_j}, δ_i⟩`. The activation factor is the model's own
representation and is not controllable. The output-space gradient direction `δ_i` is.

Replace `δ_i = w_{v_i}` with

```
δ_i^⊥  =  w_{v_i}  −  Proj_{span{ w_{v_j} : j ∈ KEEP }}  w_{v_i}
```

so `⟨w_{v_j}, δ_i^⊥⟩ = 0` **identically** for every kept token, and the uncontrollable
activation factor is multiplied by zero. The unembedding is frozen, so this is one
Gram-Schmidt per fact against a ~5-dimensional subspace of 126,464, computed once.

**Measured cost (CPU, `outputs/h11_orthogonal.json`, 364 facts):** retained suppression
**0.9867** mean, 0.9515 min, 0.9688 at the 5th percentile. **The projection costs 1.3% of the
suppression signal and removes an entire drag path exactly.**

**It also refuted the premise it was designed against.** The design argued that
co-constituents of one fact have above-random unembedding similarity. Measured, they are
**below** random: |cos| 0.0820 within a fact versus 0.1725 for random token pairs, with
retained suppression higher for fact tokens (0.9867) than random (0.9597). `Karachi` and
`Pakistan` are not geometrically entangled in the output head.

**Consequence, stated honestly: O is cheap, exact, and partial.** Since the drag is a
*product* and the unembedding factor was already small, the dominant path must be the
activation factor — which no logit-space projection can reach. O does not solve
preservation. It removes one of two paths for 1.3%.

### K — Margin hinge on the kept tokens

Because O is partial, preservation has to be **bought** rather than inherited from locality.
A hinge on the kept tokens' teacher-forced margin, inactive while they are intact:

```
λ_keep · Σ_{j ∈ KEEP} [ M_j(θ_ref) − M_j(θ) − 0.125 ]_+
```

The 0.125-nat dead band is the evaluation protocol's own declared resolution, set because
existing margins land on a 1/16-nat bf16 grid — the reason my earlier "margin ≤ 0" finding
turned out to be 1–3 ULP of noise. A KL at the kept positions would be wrong: it would pull
them toward the ignorant reference as well. The hinge constrains only degradation.

## Mathematical object, classified honestly

On teacher-forced states this is **ordinary descent on a fixed, non-smooth scalar loss**. The
index sets `KEEP`, `INTERDICT` and the structural complement all derive from the gold answer
and the frozen author-salience table, so they are **θ-independent**; `k` is frozen from a
pre-training measurement. 4 forwards / 1 backward. No trajectory collection, no staleness, no
stop-gradient inside the differentiated path.

Two things that would break that classification and are therefore **excluded by design**:
training on-policy (the stem→position map stops being exact) and adapting `k` during training.

## The signature prediction — per-item, continuous, and not MDD-bound

The forgetting metric decomposes exactly:
`ASCR = (k/n_sal)·R_int + ((n_sal−k)/n_sal)·R_keep`.

At `n_sal = 8, k = 5`, **matched ASCR = 0.375**:

| arm | R_int | R_keep |
|---|---|---|
| SIO | **0.00** | **1.00** |
| uniform suppression | 0.375 | 0.375 |

**Same headline number, opposite decomposition.** The separation statistic is
`Δ_sel = R_keep − R_int`, predicted **≥0.70 for SIO** against **≤0.15 for uniform**, and the
headline metric is blind to it.

It is per-item and continuous over 364 items in 20 authors, so it is **not bound by the
0.125 minimum detectable difference** that caps every aggregate comparison at this tier. It
is also the only form the project's standing per-fact rule accepts, after ARC's 0.101 gap and
Finding 1's `r = 0.014, p = 0.936`.

## Ablation ladder — what the claim is, and what it is not

| arm | content |
|---|---|
| **A** | C1 oracle matching on the whole answer — **published prior art, not a claim** |
| **B** | C1 with a **random**-`k` subset, no orthogonalisation, no hinge |
| **B′** | C1 on all salient tokens, no hinge |
| **C** | interdiction only (S, no O, no K) |
| **D** | **SIO = S + O + K** |

**The claim is D > B and D > B′. D > A is not a claim** — that comparison is occupied.
Random-`k` is essential because the metric weights all salient stems equally, so `adf` rank
affects only membership and not the threshold crossing: without B, any gain is attributable
to "suppress fewer tokens", which is not novel.

## Live risks, each with its test

| risk | test | cost | state |
|---|---|---|---|
| preservation fails anyway via the activation path | one gradient step; `ρ = ΔM(keep)/ΔM(int)`; **ρ ≥ 0.5 kills it** | ~3 GPU-min | ready |
| the oracle itself misses the ignorance profile, so there is no base | A2 cell gate | running | **blocking** |
| "weight the rare tokens more" is already published | prior-art check | running | — |
| selection circularity — targeting the metric's own statistic | re-score on `adf ∈ {4,20}`, TOFU ROUGE-L, and the six-cell profile; plus random-`k` | free | pre-registered |

**Standing on novelty, stated before any result:** category **(c)** at the outset — the
components are known. (O) appears to be new in this setting and (S)'s minimality criterion
may be, but neither claim is made until the prior-art check returns. The defensible
contribution, if the measurements hold, is the **composite plus the decomposition signature**,
not any single part.
