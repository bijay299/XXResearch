# Results — AFC against MDU, EMDU and ATWU on TOFU/LLaDA

> 🏁 **FOUR-POINT FRONTIER, 2026-10-25 — the frontier claim is no longer provisional.** MDU λ=4, the
> pre-registered threat, regressed to **TV 0.3104 ± 0.0892 at retain 0.3869 ± 0.1002** at n=12 (its
> single-seed 0.2325/0.4565 was the most favourable of twelve; seed 0 reproduced it exactly). The
> frontier, n=12 each: **AFC λ=1 0.2983 / 0.5153** · MDU λ=1 0.4400 / 0.5235 · MDU λ=3 0.3223 /
> 0.4057 · MDU λ=4 0.3104 / 0.3869. **AFC has the lowest TV of all four and, within noise, the
> highest retention — no λ of MDU reaches that combination.** λ=1 matches retention but forgets
> +0.142 worse (t=−4.06, 11/12); λ=3 and λ=4 match forgetting but retain 0.110 and 0.128 less
> (t=+3.15, +3.92, 11/12 each). ⚠ **λ=2 is a declared gap** — queued then cancelled on instruction,
> not for lack of value. ⚠ And this is a **trade-off** result, **not** a robustness result: AFC still
> recovers up to 62.7% under the strongest valid attack, inside the baseline's 51–68%; it is also the
> highest-variance of the four arms. See `docs/FINDING_four_point_frontier.md`.

> ✅ **MATCHED-RETENTION SLICE, 2026-10-24 — the first claim here to survive every control.** At
> retention matched almost exactly (AFC 0.5153 vs MDU λ=1 0.5235, t=−0.32), AFC forgets clearly
> better: **TV −0.1417 (t=−4.06, 11/12 seeds)** and **DISCLOSE −0.1757 (t=−5.54, 11/12)**. With the
> matched-disclosure slice below, AFC beats MDU λ=1 on forgetting at equal retention *and* MDU λ=3 on
> retention at tied forgetting — i.e. **outside** the MDU λ∈{1,3} frontier, not on it. The
> pre-registered negative outcome ("if MDU λ=1 lands near TV 0.30, AFC is merely on MDU's frontier")
> **did not occur**. ⚠ **Provisional**: a two-point frontier cannot support "above it", and a
> single-seed MDU λ=4 point (TV 0.2325 at retain 0.4565) would overturn the dominance claim if it
> holds at n=12 — λ=4 and λ=2 are running. AFC still loses on DEGENERATE (0.149 vs 0.095) at every n,
> is still ~2× more variable on TV, and is still **not** relearning-robust (recovers up to 62.7%).
> See `docs/FINDING_matched_retention_slice.md`.

> 📌 **n=12 VERDICT, 2026-10-24 — read this instead of §1.** Both columns are now complete at 12
> paired seeds under explicitly recorded settings. **Retention survives: +0.1096 retain ASCR
> (0.5153 vs 0.4057), 11 of 12 paired seeds, t=+3.15.** **TV does not: −0.0240 (0.2983 vs 0.3223),
> 6 of 12 — exactly even, t=−0.53.** DEGENERATE goes the other way (AFC 0.149 vs 0.127, worse on 8 of
> 12), and AFC's TV sd is ~1.5× MDU's, so the n=3 lower-variance claim is also gone. The one
> defensible claim is **"at matched disclosure, substantially more capability retained at
> indistinguishable distance-to-pristine"** — one axis, with the forgetting axis a tie. That is short
> of beating the baseline on a forgetting–utility trade-off. A matched-RETENTION slice (MDU λ=1 at
> n=12) is running and could remove even this: if MDU λ=1 reaches TV ≈ 0.30 at comparable retain,
> AFC is *on* MDU's frontier rather than above it and has no advantage on either slice.
> See `docs/FINDING_n12_verdict_retain_survives.md`.

> ❌ **SUPERSEDED, 2026-10-24 — the comparative claim in §1 below is REFUTED.** With the baseline
> confound fixed (λ=3 rebuilds, bridge control reproducing `ll_mdu_f3` exactly on all seven reported
> numbers) and n=8 seeds, **AFC's TV advantage is gone: 0.2993 ± 0.1290 vs MDU λ=3's
> 0.2977 ± 0.0650, a difference of 0.0016** — not the "37% lower" in §1 — and the paired win rate is
> **4 of 8**, with AFC's three wins being exactly the three seeds it was selected on. **What
> survives is retention:** retain ASCR **0.534 vs 0.393 (+0.141)**, holding on 7 of 8 seeds. So the
> defensible claim is *"indistinguishable TV at matched disclosure, with substantially more capability
> retained"* — a narrower contribution on a different axis than §1 asserts. AFC also remains ~2× more
> variable on TV and is **no better** on DEGENERATE (0.152 vs 0.128; the n=3 figures inverted). A
> symmetric control on AFC's own unrecorded hyperparameters has since **PASSED** — `afc_l1_s0_bridge`
> reproduces `h16_none_f1` exactly on all four coordinates — so **both columns are verified
> reproducible and the retain claim is NOT provisional.** Whether AFC's extension seeds `afc_s3..s7`
> were themselves built with those settings is still being checked. See `docs/FINDING_afc_tv_advantage_does_not_survive.md`.

> ⚠ **BASELINE CONFOUND, 2026-10-24 — the retraction below has a problem of its own.** The MDU
> extension seeds `mdu_s3..s7` it relies on have **no recorded λ_f anywhere** (not the logs, not
> ATTEMPTS.tsv, not the arm JSONs), and `train_lam.py --lam-forget` **defaults to 8.0**. Rebuilding
> those seeds at an **explicit λ_f=3** does **not** reproduce them on any coordinate (TV 0.10–0.23
> lower, retain 0.09–0.25 lower, DISCLOSE 0.05–0.32 lower), so **they were not λ=3**. Their profile
> matches published **MDU λ_f=1** instead, and the "MDU 0.424" figure below is MDU λ=1's own
> published TV of 0.423 to three decimals. **The stability refutation below stands** — it rests on
> AFC's seeds, whose λ is not in question — but its *comparative* claim ("0.314 vs 0.424, wins 4 of
> 5") is against the wrong baseline arm, not the matched-disclosure λ=3 comparison §1 rests on.
> On the four rebuilt seeds MDU λ=3 has **lower TV than AFC on all four**, though at markedly worse
> retain (0.30–0.37 vs 0.46–0.56), so this is a different operating point and **not** a clean
> inversion. A bridge control (seed 0 at explicit λ=3 vs `ll_mdu_f3`) is running to establish
> whether the rebuilds are comparable to the λ=3 seeds 0–2 at all. See
> `docs/FINDING_lambda_provenance_confound.md`. No existing measurement was altered.

> **RETRACTION, 2026-10-21 (n=5 seeds).** The stability claim in §2 below is **REFUTED by our own
> data** and is retained only as a record of the error. At n=3 AFC's TV sd was 0.005; at n=5 it is
> **0.158 against MDU's 0.033** — AFC is roughly **5x MORE variable**, not 9-16x less. Seed 3
> produced TV 0.587 with DEGENERATE **0.508**. AFC's mean DEGENERATE at n=5 is **0.217 vs MDU's
> 0.122**, so the lower-degeneracy claim also fails at lam=1, and AFC breaches DEG <= 0.25 on 1 of
> 5 seeds.
>
> **What survives:** AFC's mean TV is still better (0.314 vs 0.424) and it wins 4 of 5 paired
> seeds, losing seed 3 by -0.171. The honest claim is therefore **"better on average, with a
> catastrophic-failure mode the baseline does not have"** -- not "better and more stable". The
> n=3 figures were three lucky seeds. n=8 is running to characterise the failure mode.


**Date:** 2026-10-21
**Backbone:** LLaDA-8B-Instruct rev `08b83a6feb34df1a6011b80c3c00c7563e963b07`, memorization
adapter `theta0_tofu` (qualified 0.791/0.787 recall, bitwise save/reload verified)
**Protocol:** `eval_arm_partition.py`, 120 forget / 80 retain probes, pinned sampler, greedy
(T=0), `gen_length=64`, `steps=64`, six-cell partition + ASCR-12s
**Objective:** minimise `TV(forget profile → pristine)` while preserving retain ASCR

---

## 1. Headline, at matched disclosure

Three seeds per arm (0, 1, 2), paired, mean ± sample sd:

| arm | n | DISCLOSE | TV→pristine | retain ASCR | DEGENERATE |
|---|---|---|---|---|---|
| **AFC λ_f=1 (selected)** | 3 | 0.136 ± 0.063 | **0.223 ± 0.005** | **0.576 ± 0.038** | **0.117 ± 0.058** |
| MDU λ_f=3 | 3 | 0.133 ± 0.090 | 0.352 ± 0.081 | 0.445 ± 0.053 | 0.181 ± 0.148 |
| MDU λ_f=1 | 3 | 0.233 ± 0.104 | 0.423 ± 0.046 | 0.557 ± 0.052 | 0.194 ± 0.147 |

MDU λ_f=3 is the matched-disclosure comparison: **0.133 against AFC's 0.136**, statistically
indistinguishable. At that matched point AFC gives

- **TV→pristine 0.223 vs 0.352** — 37% lower
- **retain ASCR 0.576 vs 0.445** — 29% higher
- **DEGENERATE 0.117 vs 0.181**

AFC also wins all four coordinates against MDU λ_f=1, and its TV advantage holds on **every
individual seed** (+0.179, +0.254, +0.167).

## 2. Stability — RETRACTED (see the note at the top; kept as a record of the error)

| arm | TV sd | DEGENERATE range across seeds | seeds breaching DEG ≤ 0.25 |
|---|---|---|---|
| **AFC λ_f=1** | **0.005** | 0.050 – 0.150 | **0 / 3** |
| AFC λ_f=2 | 0.053 | 0.042 – 0.292 | 1 / 3 |
| MDU λ_f=1 | 0.046 | 0.075 – 0.358 | 1 / 3 |
| MDU λ_f=3 | 0.081 | 0.050 – 0.342 | 1 / 3 |

AFC λ_f=1 has **9–16× lower TV variance** than any MDU setting and is the only arm that never
breaches the degeneracy threshold. MDU's degeneracy swings from 0.075 to 0.358 on identical
settings with only the seed changed.

**Consequence for the literature:** a single-seed comparison between these methods is close to
uninformative, and both parent papers report single configurations. This is part of the
contribution, not a caveat.

It also forced a correction of our own: λ_f=2 was selected on seed 0 alone (DEG 0.042) and
**replication reversed it** — λ_f=2 has 10× the TV variance and breaches the degeneracy
threshold on 1 of 3 seeds. The selected point is λ_f=1.

## 3. ATWU — 11 configurations, and the honest framing

ATWU ([2606.06320](https://arxiv.org/abs/2606.06320)) claims SOTA forget–retain trade-offs on
TOFU and RWKU, so it cannot be skipped. Implemented faithfully from the PDF: SatGA+
`g·p^{βg}·log p`, `g = σ(⟨w,h⟩)` with `w` initialised to `0_d`, binary-entropy and budget
penalties, alternating scorer/model updates, hidden states taken at the lm_head input.

Given its published 8B/TOFU settings (α=0.50, γ=3.00, β=7.00, scorer-lr 0.050, lr 1.5e-5,
λ_ent=1, λ_pop=15) **plus** sweeps over everything we could not recover or that could plausibly
disadvantage it:

| swept | values |
|---|---|
| ρ (budget — supplementary-only in the paper) | 0.10, 0.25, 0.50 |
| β (saturation exponent) | 1.0, 1.5, 2.0, 7.0 |
| model lr | 1.5e-5 (published), 1e-4 (matched to AFC) |
| steps | 300, 900 |
| selector | learned, our divergence selector |
| forget loss | SatGA+, anchor KL |

**Best of 11 configurations by TV: 0.466 (retain 0.552, DISCLOSE 0.433, DEG 0.025).** AFC's TV
is **less than half** that, at comparable retain.

**What we will and will not say.** ATWU is specified for autoregressive models. The MDLM
adaptation — applying SatGA+ at masked answer positions of a teacher-forced state, with hidden
states from the lm_head input — is **ours, not theirs**, and may not do the method justice. So
the claim is *"our MDLM adaptation of ATWU underperforms AFC on this protocol"*, **not** *"AFC
beats ATWU"*. ATWU is also explicitly composable with any forget loss, so ATWU's selector on
AFC's anchor is a natural stronger configuration that we have only probed at one setting.

Two measured notes on it: its learned selector agrees with our free divergence selector only
**33%** of the time on the anchor loss, so the two criteria find substantially different
positions; and its consistently low degeneracy (0.017–0.042 across all configurations) is a
genuine strength of the saturated loss that AFC does not match.

## 4. Full frontier (single seed unless noted)

Under the project's standard (DEGENERATE ≤ 0.25):

| arm | DISCLOSE | DEGENERATE | retain | TV→pristine |
|---|---|---|---|---|
| θ₀ (start) | 0.792 | 0.008 | 0.762 | 0.770 |
| **AFC λ=1 (n=3)** | **0.136** | **0.117** | **0.576** | **0.223** |
| AFC λ=2 (n=3) | 0.100 | 0.144 | 0.534 | 0.246 |
| AFC λ=4 | 0.150 | 0.117 | 0.541 | 0.259 |
| AFC λ=0.5 | 0.258 | 0.117 | 0.582 | 0.345 |
| MDU λ=3 (n=3) | 0.133 | 0.181 | 0.445 | 0.352 |
| MDU λ=1 (n=3) | 0.233 | 0.194 | 0.557 | 0.423 |
| MDU λ=8 | 0.067 | 0.150 | 0.091 | 0.191 |
| ATWU best of 11 | 0.433 | 0.025 | 0.552 | 0.466 |
| EMDU λ=8 | 0.000 | **1.000** | 0.580 | 1.000 |
| *pristine target* | *0.030* | *0.000* | — | *0* |

Excluded for degeneracy: EMDU (1.000), AFC λ=8 (0.908), H16 projection k=32 (0.275).

## 5. Limitations

- **One backbone, one dataset.** TOFU-style split on a synthetic-memorization LLaDA backbone.
  No RWKU, no Dream-7B. Transfer is untested.
- **Robustness: RTT is now run, and it goes AGAINST AFC.** See `FINDING20` and `REPORT.md` §5.
  At the *identical* pre-attack disclosure of 0.225, AFC reopens 0.200 (35.3% of the gap to θ₀)
  against MDU λ=3's 0.075 (13.2%) — **2.7× worse robustness at matched disclosure**, under an
  attack that passed its validity gate. AFC's better static trade-off is bought with shallower
  suppression, so **AFC cannot be claimed to beat MDU.** This supersedes the "no robustness
  claim" placeholder that stood here.
- **Narrow usable band.** AFC collapses at λ_f=8 (DEGENERATE 0.908, retain 0.011). Usable range
  λ_f ≈ 0.5–4.
- **Still far from the target.** DISCLOSE 0.136 against pristine's 0.030; retain 0.576 against
  θ₀'s 0.762. AFC moves the frontier; it does not solve the problem.
- **ATWU adaptation is ours** (§3).
- **AFC is not a new algorithm** — it is a factorial cell of two published methods, every term
  theirs. The contribution is separating EMDU's two simultaneous changes, measuring which half
  carries the benefit and why the other hurts, the stability finding in §2, and the evaluation
  argument below.

## 6. The evaluation argument

Three single-axis views of the same arms give three different winners:

| ranking axis | winner | why it is wrong alone |
|---|---|---|
| DISCLOSE only | EMDU 0.000 | DEGENERATE 1.000 — emits garbage |
| TV(forget) only | MDU λ=8, 0.191 | retain 0.091 — model destroyed |
| DISCLOSE + retain | H16 projection | DEGENERATE 0.192 vs AFC's 0.042 |

We fell into the first and the third ourselves. MDU and EMDU report TOFU forget quality as
ROUGE-L and token probability, on which **a degenerate model scores perfectly**. An honest
comparison needs the full six-cell profile *and* retain utility together — the pairing used
throughout this document.

## 7. Negative results, recorded as such

Eight candidates closed on measured evidence: H1, H6, H11, C1/SIO, the ordering line
(H10a/b/c — `FINDING11`, `FINDING12`), ESD (anticipated by MDU), the form-preserving anchor
(H14), localized anchor matching (H15 — `FINDING15`), and retain-conflict projection
(H16 — rank sweep 3/3 worse than no projection). The divergence position selector is validated
as a diagnostic — its complement carries nothing forgettable — but has now failed to improve
three different anchor losses, which is recorded as a repeated negative rather than buried.
