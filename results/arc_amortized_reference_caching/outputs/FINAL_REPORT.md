# ARC — Amortized Reference Caching for masked-diffusion unlearning

**4-hour research sprint, 2026-10-20, started 04:57 ET. All work on synthetic data.**

> **Headline.** The original idea (SPEX) was **disproved by its own proof check at 0:08**,
> and the sprint pivoted. What survives is (a) a **negative result with a proof** —
> the per-step reference pass in this method family is *not* removable by algebra — and
> (b) **ARC**, which removes it by *amortisation* instead: **1.26× faster per step and
> 18.8× fewer reference forwards at bit-identical targets and matched efficacy**.
> Plus (c) a metric, **clean forgetting**, which shows that the two published arms that
> appear to forget perfectly (disclosure 0.000) do so by **emitting nothing**.

**No claim** of certified deletion, benchmark superiority, or established novelty. All
results are on 24 synthetic facts about fictional entities, one model, LoRA, 3 seeds on
the key pair. Published-benchmark comparison was explicitly out of scope.

> ### ⚠ SCOPE CORRECTION — see [`AMENDMENT_01_breaking_point.md`](AMENDMENT_01_breaking_point.md)
>
> The §10 next step was subsequently run: a **128-fact** probe. **The "matched efficacy"
> half of the headline above does not survive it.** ARC with an under-covering pool
> (`K=16`, 15/64 forget facts covered) **fails to forget at all** — disclosure 0.931 vs
> the uncached control's 0.049 — and even with full coverage (`K=512`) it leaves 0.216
> against 0.049, i.e. 4.4× more residual disclosure. The **cost** results (exactness,
> forward counts, 1.257× speedup) are unaffected. The sharp per-fact prediction
> pre-registered in `scripts/build_big.py` was also **refuted as stated**. Read the
> amendment before citing anything on this page.

---

## 1. What was attempted, and how it failed (0:00 – 0:08)

**SPEX** proposed building the exclusion target from the *student's own* distribution
rather than a frozen reference, which would remove the reference forward entirely. The
justification was Corollary 2.1: that the target's algebra preserves the non-fact
distribution's shape, so no anchor is needed.

`scripts/verify_theory.py` → **10/13 claims pass**. Lemma 1 (I-projection closed form) and
Lemma 2 (rank-2 gradient) hold, symbolically and numerically over 300 random instances.
**Corollary 2.1 is false.** The gradient `α(p⊙1_F) + β(p⊙1_¬F)` is rank-2 *as a
distribution-space vector*, but the **logit** update it induces is `Δz(v) = −ηβp(v)` —
proportional to `p(v)`, not constant on the block. Measured: the ratios of `−g(v)` across
`¬F` are exactly the ratios of `p(v)`, and `d/dη[log p(v) − log p(v')]` reaches **0.0594**,
not 0. Proposition 3's *rate* algebra was also wrong (worst relative error 0.894), though
monotonicity itself survives.

**Proposition 5, the sharper result this yields.** A logit update preserves every
within-block ratio **iff** it is constant on the block. So the only exactly shape-preserving
realisation of the exclusion target is a **token-level bias on `F`** — i.e. token banning,
which the pinned prior study measured as carrying **91–96 % of collateral damage**. The
shape-preserving edit is precisely the damaging one, and therefore **any gradient-based
realisation must reshape `¬F` and needs an anchor.** This explains why MDU, EMDU, SSD and
FX-Mix all carry a reference forward: it is forced, not habitual. It also rules out the
entire "drop the reference by algebra" family.

## 2. The pivot: ARC

The disproof says the reference pass cannot be removed by algebra. It does **not** say it
must be *recomputed every step*. The reference is frozen, so its target is a pure function
of the state; prior methods recompute only because they draw fresh random states.

**ARC** pre-samples `K` states, caches the frozen reference's exclusion target on each
once, then trains by sampling from the pool.

- **Proposition 6 (exactness).** Cached targets are bit-identical to recomputed ones.
  **Checked as a hard gate in every run: `max|Δ| = 0.000e+00`.**
- **Proposition 7 (cost), twice mis-derived then measured.** My first two derivations were
  wrong (once for ignoring the retain forward, once for ignoring that the reference pass is
  *batched*). Both errors are preserved in `theory/THEORY.md` rather than overwritten. The
  measured truth is below.
- **The one approximation** is statistical: a fixed `K`-sample of the state distribution
  instead of fresh draws. Price measured by the `K` sweep.

## 3. Measured cost (instrumented forward counters, not assumptions)

| arm | fwd/step (student + ref) | s/step | train s | prefill s | reference fwd total | peak GiB |
|---|---|---|---|---|---|---|
| NPO | 3 | 0.470 | 72.2 | 0 | — | 23.80 |
| MDU τ=1 | 3 | 0.471 | 72.3 | 0 | — | 23.97 |
| EMDU | 3 | 0.459 | 70.4 | 0 | — | 23.89 |
| SC-TF | 3 | 0.466 | 71.4 | 0 | — | 23.88 |
| uncached exclusion (ours, control) | 2 + 1 = **3** | 0.484–0.488 | 74.6–74.9 | 0 | **150** | 23.66–23.69 |
| **ARC K=64** | 2 + **0** = **2** | **0.381–0.391** | 58.1–60.2 | 1.1–1.2 | **8** | 24.28–24.36 |
| ARC K=256 | 2 + 0 = 2 | 0.392 | 60.6 | 3.5 | 32 | 25.84 |
| ARC K=1024 | 2 + 0 = 2 | 0.395 | 60.7 | 11.3 | 128 | 31.66 |

- **per-step forwards 3 → 2** (factor 1.5), exactly as the batched count predicts
- **s/step 1.257× faster** (mean over 3 seeds, K=64 vs uncached)
- **reference forwards 18.8× fewer** (150 → 8)
- **memory is the price:** peak rises 23.7 → 24.3 GiB (K=64) and → 31.7 GiB (K=1024)

## 4. Efficacy — frozen 48-question evaluator, 1,104 unique generations

**23 checkpoints × 48 questions = 1,104 unique generations**, deduplicated (one
duplicated θ₀ block removed from 1,152 raw rows).

| checkpoint | disclosure ↓ | retain ↑ | empty | repetitive | **clean forgetting ↑** |
|---|---|---|---|---|---|
| memorized reference (θ₀) | 1.000 | 1.000 | 0 | 7 | 0.000 |
| NPO @150 | 0.500 | 1.000 | 0 | 12 | 0.250 |
| MDU τ=1 @50 | 0.750 | 1.000 | 0 | 0 | 0.250 |
| **MDU τ=1 @150** | **0.000** | 0.969 | **15** | 1 | **0.000** |
| **EMDU @50** | **0.000** | 1.000 | **14** | 0 | **0.125** |
| **EMDU @150** | **0.000** | 0.906 | **16** | 0 | **0.000** |
| SC-TF @150 | 0.125 | 1.000 | 0 | 0 | **0.875** |
| uncached exclusion @150 | 0.125 | 1.000 | 0 | 0 | **0.875** |
| **ARC K=64 @150** | **0.125** | **1.000** | **0** | **0** | **0.875** |
| ARC K=256 @150 | 0.125 | 1.000 | 0 | 0 | 0.875 |
| ARC K=1024 @150 | 0.125 | 1.000 | 0 | 0 | 0.875 |

### The clean-forgetting finding

**Clean forgetting** = fraction of the 16 forget probes that are **both** non-degenerate
(not empty/repetitive/refusal/malformed) **and** non-disclosing — i.e. genuinely forgot
*while still answering*.

**Every checkpoint with disclosure 0.000 has clean forgetting ≤ 0.125.** MDU τ=1 @150 emits
nothing on 15 of 16 forget probes; EMDU @150 on 16 of 16. Their perfect-looking disclosure
is **output collapse, prompt-targeted** (retain output largely survives), not forgetting.
Quoting "forget = 0.000" for those arms without the empty-rate materially misrepresents
them. The exclusion-target family reaches **0.875** clean forgetting with **zero**
degenerate outputs.

Note θ₀ itself carries 7 repetitive forget generations, so a baseline degeneracy rate
exists in the 32-token decoder independent of any arm.

## 5. Parity verdict (pre-registered before any arm ran)

| seed | uncached disclosure | ARC disclosure | Δ | uncached retain | ARC retain | Δ | parity |
|---|---|---|---|---|---|---|---|
| 0 | 0.125 | 0.125 | +0.000 | 1.000 | 1.000 | +0.000 | YES |
| 1 | 0.188 | 0.125 | **−0.062** | 1.000 | 1.000 | +0.000 | NO |
| 2 | 0.125 | 0.125 | +0.000 | 1.000 | 1.000 | +0.000 | YES |

**Pre-registered verdict: `EFFICIENCY-ONLY`** (speedup 1.257× ≥ 1.2 achieved; one seed
outside the ±0.05 band).

**Honest note on the rule, not a retroactive change.** The band was written two-sided, so
seed 1 fails it *because ARC forgot more than the control* (−0.062). Under a one-sided
"ARC not worse" reading, parity holds on all three seeds. Both readings are reported; the
pre-registered two-sided verdict stands as `EFFICIENCY-ONLY`. The rule was not rewritten
after seeing the data.

## 6. The `K` sweep — and a result that contradicted my expectation

I predicted small `K` would **overfit** the pool. The opposite appeared: at step 50,
`K=64` reaches 0.188 disclosure, `K=256` reaches 0.062, and `K=1024` is **worst** at 0.312
(retain 0.875). By step 150 all three converge to 0.125/1.000. So on this probe a larger
pool *slowed* convergence and cost 7 GiB more memory, with no efficacy benefit. Smallest
pool, best cost, equal final efficacy.

**Caveat:** with 8 forget facts and `m = 8` states/step, `K=64` already covers 8 distinct
masks per fact. The overfitting risk I predicted is real in principle; this probe is simply
too small to expose it. That is a limitation of the probe, not evidence of its absence.

## 7. What this does and does not establish

**Does:**
- The reference pass in this family is **not removable by algebra** (Prop 5, proved, with
  the measured disproof of the attempt).
- Caching the frozen reference's target over a pre-sampled pool is **exact** (Prop 6,
  `max|Δ| = 0` in every run) and yields **1.26× faster steps, 18.8× fewer reference
  forwards**, at equal clean forgetting and retention across 3 seeds.
- **Clean forgetting** separates genuine forgetting from output collapse, and reorders the
  leaderboard: two published arms' apparent perfection is degeneracy.

**Does not:**
- Beat any published method on any published benchmark — not attempted, and hardware-gated
  (full-parameter 8B needs ~89.6 GiB of FP32 optimizer state; does not fit 2×40 GB).
- Establish novelty. **Caching a frozen teacher's outputs is standard in knowledge
  distillation.** ARC's content is the observation that this family's per-step reference
  pass is amortisable, plus the measurement. If a published unlearning method already
  caches reference targets over a pre-sampled state pool, ARC is not new and the
  contribution reduces to the measurement and the clean-forgetting metric.
- Establish that pool reuse is safe at scale, or that any of this transfers off 24
  synthetic facts, one model, one decoder.

## 8. Errors made and preserved

Recorded rather than quietly fixed, because they bound how much to trust the rest:

1. **Corollary 2.1 false** — the sprint's founding claim, disproved at 0:08.
2. **Proposition 3 rate** misderived; monotonicity survived, the closed form did not.
3. **Proposition 7 derived wrongly twice** before being measured.
4. **`build_fact_set` key bug** (`["ids"]` vs `["F"]`) made my first ARC smoke run train on
   an **empty fact set** with an identically-zero forget term. Caught by a `p0(F)`
   diagnostic *before* any full run, and independently rediscovered by the baseline agent.
   The same latent bug exists in the earlier PoC's unpublished `unlearn.py`; that file
   **never ran** (no `unlearn_*.json` outputs), so the PoC's results — produced by the
   handoff's own `train_tdssd.py` — are unaffected.
5. **θ₀ double-counted** across evaluation shards (1,152 raw → 1,104 unique after dedup).
6. One seed run **OOM'd** from a shell-ternary GPU-assignment error; relaunched, preserved.

## 9. The amortisation regime — S = 600 (the key risk test)

The stated risk was **pool overfitting**: ARC reuses `K = 64` cached states, so at long
training the student could exploit those particular masks. Tested by running the same pair
for **600 steps** (pool reused ~9.4× more than at `S = 150`):

| run | steps | s/step | train wall | reference fwd | disclosure ↓ | retain ↑ | degenerate | clean forgetting ↑ |
|---|---|---|---|---|---|---|---|---|
| uncached | 150 | 0.484 | 74.9 s | **150** | 0.125 | 1.000 | 0 | 0.875 |
| ARC K=64 | 150 | 0.391 | 60.2 s | **8** | 0.125 | 1.000 | 0 | 0.875 |
| uncached | 600 | 0.465 | 280.8 s | **600** | 0.188 | 1.000 | 0 | 0.812 |
| **ARC K=64** | **600** | **0.395** | **238.0 s** | **8** | **0.125** | **1.000** | **0** | **0.875** |

- **No pool overfitting appeared.** At 600 steps ARC holds 0.125 disclosure / 0.875 clean
  forgetting, while the uncached control *regressed* slightly to 0.188 / 0.812.
- **The amortisation advantage scales as predicted**: reference forwards
  150 → 8 (**18.8×**) at `S = 150`, and 600 → 8 (**75×**) at `S = 600`, because ARC's
  reference cost is `ceil(K/m)` independent of `S`. Per-step speedup stays ~1.18–1.26×.
- End-to-end wall at `S = 600`: 280.8 s → 239.2 s (train + prefill) = **1.17×**.

**Caveat that still stands:** 8 forget facts with `m = 8` states/step means `K = 64`
already covers ~8 distinct masks per fact, so this probe cannot *rule out* overfitting at
realistic scale — it only shows none at this scale. The decisive test is §10.

## 10. One next step

Run the `K` sweep where it can actually fail: enough forget facts that `K` cannot cover the
mask space (e.g. 64 facts, `K = 16`), and long training so the pool is reused hundreds of
times. That is the regime where amortisation pays most (`S ≫ K`) **and** where pool
overfitting should appear if it exists. The `S = 600` pair launched at the end of this
sprint is reported in §9 and showed no overfitting at this scale; the decisive
test needs a mask space the pool cannot cover.
