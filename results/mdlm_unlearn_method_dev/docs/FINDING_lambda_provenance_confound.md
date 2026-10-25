# The n=5/n=8 seed extension was NOT run at λ_f=3. The published retraction in `RESULTS.md` compares AFC against the wrong baseline arm.

**Date:** 2026-10-24
**Status:** ⚠ this invalidates the baseline column of the **already-published retraction** at the top
of `docs/RESULTS.md`. Correction in progress; a bridge control is running before any number here is
treated as final.
**Evidence:** `outputs/arm_mdu_l3_s{3,4,5,6}.json` (new, λ_f=3 explicit) against
`outputs/arm_mdu_s{3,4,5,6}.json` (existing, λ_f **unrecorded**)

---

## 1. The provenance hole

The headline comparison in `RESULTS.md` §1 is **AFC λ_f=1 against MDU λ_f=3 at matched disclosure**
(DISCLOSE 0.136 vs 0.133). The n=5 retraction at the top of that file extends it with seeds 3–7.

**The λ_f of the MDU extension seeds `mdu_s3..mdu_s7` is recorded nowhere.** Not in
`outputs/log_mdu_s*.txt`, not in `logs/p24*_D_mdu_s*.log` (whose `[arm] mdu:` line prints direction,
anchor, weighting and tau but **not** λ), not in `ATTEMPTS.tsv`, not in the arm JSONs. And
`scripts/train_lam.py --lam-forget` **defaults to 8.0**, so an invocation that omitted the flag
would silently not be λ=3.

Their profile was already inconsistent with λ=3: DISCLOSE 0.267–0.458 against seeds 0–2's
0.058–0.233.

## 2. The rebuild: they are not λ=3

Seeds 3–7 rebuilt with an **explicit** `--lam-forget 3 --lam-retain 1`, under new labels
(`mdu_l3_sN`), leaving every original measurement untouched:

| seed | | TV→pristine | retain ASCR | DISCLOSE | DEGENERATE |
|---|---|---|---|---|---|
| s3 | `mdu_s3` (λ unrecorded) | 0.4158 | 0.4617 | 0.3583 | 0.0167 |
| | **`mdu_l3_s3` (λ=3)** | **0.2289** | **0.3722** | **0.2167** | **0.0417** |
| s4 | `mdu_s4` | 0.4371 | 0.5793 | 0.4583 | 0.0083 |
| | **`mdu_l3_s4`** | **0.2908** | **0.3246** | **0.1417** | **0.1583** |
| s5 | `mdu_s5` | 0.3658 | 0.5465 | 0.2667 | 0.0583 |
| | **`mdu_l3_s5`** | **0.2659** | **0.3013** | **0.2167** | **0.0500** |
| s6 | `mdu_s6` | 0.4825 | 0.5155 | 0.3250 | 0.0333 |
| | **`mdu_l3_s6`** | **0.2575** | **0.3510** | **0.0833** | **0.1500** |
| s7 | `mdu_s7` | 0.4325 | 0.5265 | 0.3250 | 0.0917 |
| | `mdu_l3_s7` | *pending* | | | |

**Not one coordinate is close.** TV is 0.10–0.23 lower, retain 0.09–0.25 lower, DISCLOSE 0.05–0.32
lower. `mdu_s3..s7` were not built at λ_f=3.

> ✅ **PROVEN, 2026-10-24 (later the same day): they were λ_f=1.** The matched-retention sweep
> rebuilt MDU λ=1 at every seed, and those rebuilds reproduce `mdu_s3`, `mdu_s4`, `mdu_s6` and
> `mdu_s7` **exactly, 4 of 4, on all four coordinates** (s5 pending):
>
> | seed | `mdu_sN` (TV / retain / DISC / DEG) | explicit λ=1 rebuild |
> |---|---|---|
> | s3 | 0.4158 / 0.4617 / 0.3583 / 0.0167 | **identical** |
> | s4 | 0.4371 / 0.5793 / 0.4583 / 0.0083 | **identical** |
> | s6 | 0.4825 / 0.5155 / 0.3250 / 0.0333 | **identical** |
> | s7 | 0.4325 / 0.5265 / 0.3250 / 0.0917 | **identical** |
>
> So the λ attribution below is **no longer an inference from profile agreement — it is recovered by
> reproduction.** The n=5/n=8 extension was run at **λ_f=1** while the headline baseline is λ_f=3,
> and the retraction's "MDU 0.424" was MDU λ=1's TV. This was the weakest link in the confound
> argument and it is now closed.

**What they almost certainly were** (written before the proof above, and superseded by it)**:** their means (TV 0.427, DISCLOSE 0.347, retain 0.526) match the
published **MDU λ_f=1** arm's n=3 figures (TV 0.423, DISCLOSE 0.233, retain 0.557) far better than
λ=3's (TV 0.352, DISCLOSE 0.133, retain 0.445). And the retraction's own baseline number — **MDU
0.424 at n=5** — is MDU **λ=1**'s published TV of 0.423 to three decimals. That is consistent with
the extension having been run at λ_f=1 throughout, but it is an inference from the numbers, not
provenance, and is **not** asserted as established.

## 3. What this does to the published claim

The retraction at the top of `RESULTS.md` says AFC's *stability* claim is refuted by our own data —
that part stands and is unaffected; it rests on **AFC's** seeds, whose λ is not in question.

**What is now in question is its comparative claim:** *"AFC's mean TV is still better (0.314 vs
0.424) and it wins 4 of 5 paired seeds."* If the 0.424 column is MDU λ=1, that comparison is
**against an arm the project does not claim parity with**, and it is not the matched-disclosure
comparison the headline rests on.

And the direction may invert. On the four completed seeds, against the λ=3 rebuilds:

| seed | AFC λ=1 TV | MDU λ=3 TV (rebuilt) | winner |
|---|---|---|---|
| s3 | 0.5871 | **0.2289** | MDU |
| s4 | 0.3122 | **0.2908** | MDU |
| s5 | 0.3371 | **0.2659** | MDU |
| s6 | 0.3089 | **0.2575** | MDU |

**MDU λ=3 has lower (better) TV than AFC on all four.** ⚠ This is **not** a clean inversion and must
not be reported as one: the rebuilt MDU arms sit at retain ASCR **0.30–0.37** against AFC's
0.46–0.56, so MDU is buying that TV with markedly worse retention. The two are at different
operating points, which is exactly the trade-off the project's objective was supposed to adjudicate.
Calling this "MDU wins" would repeat the error of calling the n=3 result "AFC wins and is more
stable".

## 4. ⚠ Why no number above is final yet, and the control that is running

The rebuilds used `train_lam.py` **defaults** for everything except λ: lr 1e-4, rank 16, steps 300.
**The λ=3 seeds 0–2 arms (`ll_mdu_f3`, `sel_mdu_l3_s1`, `sel_mdu_l3_s2`) have those settings
unrecorded too.** So a mismatch between my rebuilds and `mdu_s3..s7` proves the originals were not
λ=3, but it does **not** yet license comparing the rebuilds against seeds 0–2 or against AFC.

`run_mdu_l3_bridge.sh` is running now: seed 0 at explicit λ=3 with the same defaults, against
`ll_mdu_f3` (TV 0.2825, retain 0.4042, DISC 0.2333, DEG 0.0500).

- **reproduces** → the rebuild protocol matches the λ=3 family, the whole MDU n=8 column can be put
  on one footing, and §3's table becomes a real comparison;
- **does not reproduce** → the seeds 0–2 arms differ in some further unrecorded hyperparameter, and
  **no cross-seed MDU comparison in this project is currently sound** — a larger problem than the λ
  hole alone.

Notably, the rebuilt seeds 3–6 (TV 0.229–0.291) are *better* than the λ=3 seeds 0–2 (0.283–0.441),
which is either ordinary seed variation or a second protocol difference. The bridge discriminates.

## 5. Scope and what is preserved

Nothing existing was modified, overwritten or relabelled. `mdu_s3..s7` stand exactly as measured;
the rebuilds are new arms under new identifiers. Four of five rebuild seeds are complete and s7 is
running. The λ attribution in §2 is an inference from profile agreement, not recovered provenance —
the original invocations are not in the record and may not be recoverable.
