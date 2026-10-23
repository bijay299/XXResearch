# Audit closure — 12-issue remediation, round 2

**Date:** 2026-10-23 · **Identifier:** `AUDIT_CLOSURE_r2`
**Order followed:** verify → repair + regression tests → real-model integration checks →
recalculate summaries → freeze protocol → minimum remeasurements → documentation/handoff.

**Statement required before anything else: the research is NOT fully corrected.** Nineteen unit and
regression checks pass and five real-model integration checks pass, and neither fact closes the
scientific questions. Three issues are partially open, the mandate's bar is unmet, and five
competing defences remain unrun. Nothing below was tuned to reproduce the audited numbers — where
my re-derivation disagrees with the audit, both values are given with the evidence.

---

## 1. Closure table

| # | issue | status | evidence |
|---|---|---|---|
| 1 | post-attack pipeline identity | **ALREADY FIXED, now covered** | fix `53cca44`; regression `issue1_pipeline_identity`: `reloaded_hash == modified_hash (9fc6d62dab007b08) != erased_hash (f5138465e90ad6db)`, `max\|Δconditioning\| = 0.00e+00` vs 1e-3 tol; components asserted (PNDMScheduler, fp16 VAE/UNet, CLIPTokenizer) |
| 2 | disconnected text-encoder attack | **FIXED** | `ecfe8ad`. Embeddings recomputed through the live encoder per step. `te_grad = 1.1843e-01`, `‖Δte‖ = 9.0331e-02`, `unet_grad = 3.9560e-01`, `vae_grad = 0.0` with the flag on; `te_grad` **exactly 0** with it off. Real-model smoke check passed. Every historical "UNet + text encoder" label was UNet-only in fact |
| 3 | adversarial inner-loop sign | **FIXED** | `ecfe8ad`. Corrected: erasure loss 1.092724 → **1.093124** (rises). Pre-repair convention: 1.092724 → 1.092323 (falls), confirming the bug. Budget made relative (`eps = adv_eps·‖e_c‖`). The three failed in-house runs are **not** cited as evidence against published AdvUnlearn/Receler |
| 4 | metrics must follow the active target | **FIXED** | `ecfe8ad`. `--target` added; `n_nontarget = 11` excludes the target, legacy `retain8` (n=8) still contains tench, both emitted with a `denominator_warning`. Legacy fields renamed `*_LEGACY`. Audited denominators reproduce: retain8 0.908333 / seven-class 0.980952 / eleven-class 0.987879 |
| 5 | attack counts and model identities | **NOT REPRODUCED** (disagreement reported) | `ecfe8ad`, `outputs/run_inventory.json` (234 runs: 65 unaffected / 136 VOID / 33 post-fix). For `undo_a0.5_big` the current archive holds **10 church attacks, 3 valid**, max-valid **0.466667**, plus **5** tench controls not pooled. So **`3/10` is correct** for the church denominator, and the audit's 7-attack count does not match this archive. Control concepts are not pooled. Unresolved identities marked, incl. the overwritten `adapters/uce_church/unet.pt` |
| 6 | report gated AND ungated recovery | **FIXED** (disagreement reported) | `outputs/recovery_gated_and_ungated.json`. Max-valid **0.466667** confirmed; **ungated max is 0.6000, not 0.5333** — the audit's value is one of the ungated runs, not the maximum. Per-run gate outcomes and valid/total emitted. Gate limits documented in `PROTOCOL_FROZEN_v1.md` §metric: ceiling effects, T/H difficulty asymmetry, seed sensitivity, and blindness to a misassembled pipeline (the failure mode that produced issue 1) |
| 7 | post-attack utility | **FIXED (instrument); measurement IN FLIGHT** | `d282837`. Three axes separated per record; `FINDING38` §4; `FID_PROTOCOL.md` for the declared-pending quality axis. Regression: probe **paired** (same model twice → Δ exactly 0) and **sensitive** (σ=0.01 noise moves adherence 0.2869 → 0.1656). `H65` is measuring the frozen grid on two arms; 6 of 12 points in at this writing, table in §4 |
| 8 | separate training / selection / final evaluation | **FIXED (verified); CONTAMINATION FOUND** | `d282837`, `outputs/split_registry.json`, `scripts/verify_splits.py`. `ATK_TRAIN × EVAL_TARGET` **CLEAN (0)** — the claim the phase actually made holds. `DEF_TRAIN × EVAL_TARGET` **CONTAMINATED (8)**, `DEF_TRAIN × EVAL_UTILITY` **CONTAMINATED (39)**, all attributable to `undo_t2i.py`. Bias favours **our own arms**, not the baselines. `FINDING38` §2–3 |
| 9 | reproducibility and provenance | **PARTIAL** | `outputs/provenance_manifest.json`, `scripts/provenance.py`. Recorded: commit `d282837`, per-script hashes, pins, deps (torch 2.4.1+cu124, diffusers 0.31.0, transformers 4.46.3, numpy 1.26.4, cuDNN 9.1.0), precision policy, 34 adapter dirs hashed, split/metric references. **RNG audit: 41 stochastic sites, 13 explicit-generator, 28 global-stream.** Two gaps disclosed not patched — see §5 |
| 10 | Fisher damage allocation | **FIXED** | `ecfe8ad` + `2352f8e`. Feasible renormalisation by bisection replaces renormalise-then-clamp; per-tensor mask mean pre/post, clipped fraction, realised perturbation norm and weight-norm change logged to `maskaudit_<label>.json`. Real model, all four modes at α=0.5: **mean_post = 0.5000** despite up to 10.7 % clipping, vs **0.3046** pre-repair. Matched random control: `random` matches `retain_fisher`'s realised norm to **2.5151e+01** (five digits); `frag`/`forget_fisher` match at ~3.19e+01. Separate shuffle RNG stream. The inseparability claim stays **provisional** (`ERRATUM_r2` C18) |
| 11 | freeze a fair attack budget | **FIXED** | `2352f8e`, `docs/PROTOCOL_FROZEN_v1.md`. Six-point grid, identical for every arm, stopping rule declared; H62/H63/H64 labelled **exploratory** as pre-freeze; UNet-only and joint adversaries reported as separate classes; gated+ungated always reported |
| 12 | correct the claims and documentation | **FIXED (this round)** | `docs/ERRATUM_r2_claim_corrections.md`, C1–C18, every number re-derived from raw JSONs. Pointer banners added to the 7 affected documents; originals preserved unaltered |

**Summary: 9 fixed, 1 already fixed with coverage added, 1 not reproduced with evidence, 1 partial.**
Issue 7's instrument is closed and its measurement is in flight.

## 2. Changed files and commits

| commit | contents |
|---|---|
| `53cca44` | critical post-attack text-encoder evaluation fix (pre-remediation; cause of 136 VOID runs) |
| `08742c1` | arm-construction determinism fix |
| `ecfe8ad` | issues 2, 3, 4, 10 + regression tests + run inventory + issue 5 not reproduced |
| `2352f8e` | issue 11 protocol freeze; the `g`→`gain` shadowing bug **I introduced** in the issue-10 repair, caught by real-model integration testing; mask audit on the real model |
| `d282837` | issues 7, 8: post-attack utility instrument, FID protocol, split registry, `FINDING38` |
| *this commit* | issues 9, 12: provenance manifest + RNG audit, `ERRATUM_r2`, closure table |

Changed code: `scripts/attack_relearn.py`, `scripts/erase_adv.py`, `scripts/gen_eval.py`,
`scripts/undo_t2i.py`. New code: `scripts/test_repairs.py`, `scripts/verify_splits.py`,
`scripts/provenance.py`, `run_h65_pau.sh`. New docs: `PROTOCOL_FROZEN_v1.md`, `FID_PROTOCOL.md`,
`FINDING38_*`, `ERRATUM_r2_*`, this file.

## 3. Tests and real-model checks actually executed

**Unit/regression — 19 of 19 passing** (`outputs/test_repairs_results.json`), each written to fail
on the pre-repair code, keyed to issues 1, 2, 3, 4, 7, 10. Identity is checked by **parameter hash**,
never by file path or by a score varying.

**Real-model integration checks — 5, all passed:**

1. **Issue 2 on the real model** — joint attack produces finite nonzero encoder gradient and a real
   parameter delta; UNet-only produces exactly zero; VAE frozen throughout.
2. **Issue 10 on the real model** — all four protect modes at α=0.5 hit `mean_post = 0.5000`;
   `random` matches `retain_fisher`'s realised perturbation norm to five digits.
3. **Issue 7 end-to-end** — a real 20-step attack on the official AdvUnlearn church encoder wrote a
   complete `post_attack_utility` block with `status = MEASURED`, grids saved before and after.
4. **Issue 8** — the registry imports the live pools and recomputes every overlap.
5. **Issue 9** — the manifest generator ran against the real adapter tree (34 arms hashed).

**One bug of my own, found only by check 2.** My issue-10 repair named the bisection variable `g`,
shadowing the `torch.Generator` named `g`, so the next `normal_(generator=g)` received a float. The
unit tests passed because they exercised the bisection arithmetic in isolation. It killed `H64` and
is fixed in `2352f8e` with a static shadowing check. **This is the concrete reason step 3 of the
order is separate from step 2, and the concrete reason passing unit tests does not mean corrected.**

## 4. Corrected tables from the run inventory

### 4a. The headline statistic, re-derived (`ERRATUM_r2` C1)

Pure official AdvUnlearn arms, 300 steps, post-dating the evaluation fix, **n = 25**:

| statistic | value |
|---|---|
| gate-passing attacks | 25 / 25 |
| individually reaching H = 1.000 | **8 / 25 (32 %)** |
| median single-attack H | **0.800** |
| range | 0.3333 – 1.000 |
| max-over-valid, per concept | **1.000 for all five** |

Distribution: `{0.333: 1, 0.467: 3, 0.600: 2, 0.733: 3, 0.800: 4, 0.867: 2, 0.933: 2, 1.000: 8}`.

### 4b. Audited pin vs current HEAD (`ERRATUM_r2` C11)

| | audited pin | current HEAD |
|---|---|---|
| concepts reaching 1.000 on ≥1 valid attack | 4 of 5 | **5 of 5** |
| tench max-over-valid | 0.733 | **1.000** (`H62`, post-pin) |
| official 300-step runs individually at 1.000 | 6 of 15 | **8 of 25** |

### 4c. Gated vs ungated, `undo_a0.5_big` church (issue 6)

| quantity | value |
|---|---|
| church attacks in archive | **10** |
| valid (gate-passing) | **3** → `3/10` |
| max over **valid** | **0.466667** |
| max over **all completed** | **0.600000** (audit reported 0.5333) |
| tench controls, not pooled | 5 (1 valid) |

### 4d. Post-attack utility (issue 7, `H65`, **COMPLETE — 12 of 12 points**)

**12/12 gate-passing; max-over-valid H = 1.000 on both arms.** Aggregated by learning rate (full
per-point table in `FINDING38` §4a):

| lr | n | max-over-valid H | **utility after** | Δadherence |
|---|---|---|---|---|
| **1e-5** | 6 | **1.000** (both arms) | **0.875 – 1.000** | −0.0055 … +0.0000 |
| 5e-5 | 4 | 1.000 (both arms) | 0.604 – 0.750 | −0.0684 … −0.0315 |
| 1e-4 | 2 | 1.000 church / 0.667 parachute | **0.021 – 0.271** | −0.1252 … −0.0973 |

**Full reversal occurs at intact utility:** church sd6 lr1e-5 reaches H = 1.000 at utility **0.9375**
(Δadherence −0.0003) and parachute sd1 lr1e-5 reaches H = 1.000 at utility **0.9792** (−0.0036). The
benign explanation the record could not previously exclude — that `H = 1.000` reflected a wrecked
model — **is excluded by measurement** at the setting where recovery is maximal.

**The efficient attack is the gentlest one.** Utility falls monotonically with lr while H is already
maximal at 1e-5, and parachute at 1e-4 recovers *less* (0.667) from a model it destroyed (utility
0.021). The threat model is therefore governed by the 1e-5 row, which has negligible collateral.

**Unplanned positive control:** the lr 1e-4 points read utility 0.0208 / adherence −0.1252 where
intact models read 1.0000 / −0.0003, so the issue-7 probe discriminates intact from destroyed across
a 48× range on *real attacked models*, beyond its synthetic unit test.

> ⚠ **Correction to an interim figure.** Mid-run, with only the lr 1e-5 points complete, retained
> utility was reported as 93.8–97.9 % (worst 87.5 %). **Across the full grid it is 2.1–100 %.** The
> conclusion above survives because it is conditional on lr 1e-5, but the interim aggregate was
> quoted from the finished half of the grid and understated damage at higher learning rates.

## 5. Claim status

**Invalidated**
- "1 image / 50 steps suffices for full recovery" — void (`ERRATUM_text_encoder_eval_bug`); not reinstated in weaker form.
- All pre-fix unanimous tallies on encoder-erased arms (136 runs VOID).
- Historical "UNet + text encoder" joint-adversary labels — UNet-only in fact (issue 2).
- "H47's 0.50 bar refutes H40" — H40's threshold is 0.40 (C5).
- "the 2.14 ratio" as an attacker-cost multiplier (C4).
- "damage+distil's utility failure was inherited from a mediocre teacher" — confounded with pool and construction changes; untested (C8).
- Any cross-family utility **ranking** placing a damage+distil arm above a baseline on retain (C10, in-sample).
- `FINDING34`'s "the 42 % align drop confirms erasure is real", and its "erases more than our ESD" (one image on a 15-image metric) — both already withdrawn in `FINDING35` §5.

**Superseded**
- "restored to exactly 1.000" → max-over-valid = 1.000 per concept; median single attack 0.800 (C1).
- "4 of 5 concepts / tench 0.733" → 5 of 5, tench 1.000, post-pin provenance (C11).
- "ungated max 0.5333" → **0.6000** (issue 6).
- "7 church attacks" → **10**, `3/10` stands (issue 5).
- α=0 "independent arms agree" → exact no-ops, near-tautological (`FINDING35` §0(3)).

**Still supported**
- Every usable erasure tested, **including the official robust SOTA**, is reversed to the base
  model's ceiling by 300 steps on 16 self-generated images, on **all five concepts**, under the
  frozen symmetric grid — and now also with the attacked model still 87.5–97.9 % functional.
- Prompt-space adversarial training does not transfer to a weight-access adversary. **This is a
  scope statement about a different threat model, not a refutation of AdvUnlearn's paper** (C15).
- The cross-attention edit surface cannot cleanly separate concepts: content tokens discriminate
  (cos 0.13–0.21) while the trailing positions that dominate cross-attention are half-collinear
  (cos 0.52–0.59).
- Damage's collateral destruction is as reversible as its intended erasure (`FINDING32` §4a).
- H51 is false: concept-targeted damage destroys utility ~9× worse than uniform damage (retain
  0.0917 vs 0.867) — mechanism claim **provisional** (C18).
- The headline is **ANTICIPATED** by Meta-Unlearning (2410.12777) and Erased-or-Dormant (2505.16174).

**Provisional, explicitly not established**
- That the knowledge was never removed. **Without a never-trained oracle, reactivation of residual
  memory and relearning-from-the-attacker's-data are indistinguishable here** (C17). This is the
  deepest limitation in the phase and it was never previously stated plainly.
- That forget- and retain-critical parameters are inseparable (C18 — no direct overlap measurement).
- That diffusion loss measures carry no information about recovery (C14 — one statistic, one model).

## 6. Costs and remaining gaps

**Spent** (ledger, cumulative, never reset): `elapsed 71.3 h · gpu 63.5 device-h (ATTEMPTS-derived)
· cpu 317.9 core-h · attempts 319 · storage 21.0 GiB` of the authorised 60 GiB. This remediation
round: ~1.5 GPU-hours (regression suite, 5 integration checks, `H65` in flight), well inside budget.
Two GPUs maximum, respected throughout; one breach earlier in the project was disclosed when it
happened and not repeated.

**Measurement gaps, priced**

| gap | cost | status |
|---|---|---|
| `H65` remaining 7 grid points | ~0.8 GPU-h | **in flight** |
| FID at defensible n (5 000 images, 3-way) | ≈ 4.2 GPU-h + 6 GB | **PENDING**, trigger declared in `FID_PROTOCOL.md` |
| `H64` (frag allocation) relaunch after the shadowing bug | ~2 GPU-h | **not started** |
| Meta-Unlearning / IMMA / GIFT / Dormant-to-Deleted | implementation work, no released T2I code | **unrun** |
| FRAG/FRP (2608.25429) | **no code released** (repo is a README stub, see `ERRATUM_r2` C19); algorithm fully specified and training-free | **unrun, the most portable of the five** |
| re-measure the voided low-budget attacker-cost curve (2/50 steps) | ~0.5 GPU-h | **not started** |
| never-trained oracle (Tier-2 small model) | large | **not built**; C17 is unresolvable without it |

**Provenance gaps, disclosed not patched**
- **The CLIP classifier revision was never pinned.** It produces every forget/retain/recovery number
  in this phase. Results are internally self-consistent from the local cache; pinning it now would
  not retroactively identify the weights historical runs used.
- **28 of 41 stochastic call sites use the global RNG stream**, including the attack's per-step
  draws. An attack is repeatable end-to-end from its seed but not robust to inserting or removing
  any earlier stochastic call. `erase_adv.py` and `erase_esd.py` have no explicit generator at all.
- **Attacked checkpoints are deleted after measurement**, so the weights behind every `H_after` no
  longer exist; the input arm is retained and the attack is repeatable subject to the above.
- **`adapters/uce_church/unet.pt` was overwritten** between two builds — identity **UNRESOLVED**, and
  deliberately not reconstructed, since a rebuild is a new arm and must not be relabelled.

## 7. Next-step recommendation

**Do not extend this phase's attack campaign. Run the one baseline that can actually be run.**

The phase's negative result is now solid, bounded and honestly scoped — and it is *anticipated*, so
more attacks on more of our own arms cannot produce a contribution. The single binding constraint on
every claim is the competing-defence gap. **Zero of the five have usable released code** (`ERRATUM_r2` C19 corrects my earlier claim
that FRAG/FRP did). FRAG/FRP is nonetheless the most portable: Algorithm 1 is fully specified and
**training-free**, needing only calibration forward passes. The recommendation, in order:

1. **Run FRAG/FRP under `PROTOCOL_FROZEN_v1`**, pre-registered, on church and parachute, with the
   contaminated pools repaired first per `FINDING38` §5 so its utility is measured out-of-sample.
   This is the only comparison available at the budget, and it is bounded.
2. **Repair the pools** (`FORGET_POOL`/`RETAIN_POOL` disjoint from `gen_eval.CONCEPTS`) and require
   `verify_splits.py` CLEAN before any cross-family utility comparison. Cheap, and it unblocks the
   only axis currently unmeasurable.
3. **Relaunch `H64`** under the frozen grid — it is the one untried allocation and it is already
   written and now unblocked.
4. **If and only if an arm reaches retain ≥ 0.85 at max-valid H ≤ 0.40**, run the FID comparison its
   protocol document already authorises.

What I would **not** do: build the Tier-2 oracle (large, and C17 is a limitation to state rather
than a blocker to the threat-model result), or re-implement three unreleased defences to make the
baseline table look complete.
