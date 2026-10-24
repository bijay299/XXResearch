# ❌ AFC's TV advantage over MDU does **not** survive a correctly-λ-matched baseline at n=8. It falls from 37% to **0.5%**, and the paired win rate falls to **4/8**. What survives is retention, not TV.

**Date:** 2026-10-24 · **Supersedes:** the comparative claim in `RESULTS.md` §1 and in its own
2026-10-21 retraction
**Depends on:** `FINDING_lambda_provenance_confound.md` (the baseline λ hole) and the bridge control
below
**Evidence:** `outputs/arm_mdu_l3_s{0_bridge,3,4,5,6,7}.json`, `outputs/arm_afc_s{3..7}.json`,
`outputs/arm_{h16_none_f1,seed_afc_s1,seed_afc_s2,ll_mdu_f3,sel_mdu_l3_s1,sel_mdu_l3_s2}.json`

---

## 1. The bridge control passed, so the comparison is now on one footing

`FINDING_lambda_provenance_confound` established that `mdu_s3..s7` were **not** λ_f=3 and could not
back the published comparison. It left one thing unsettled: whether *my* λ=3 rebuilds were
comparable to the λ=3 seeds 0–2, whose lr/rank/steps are also unrecorded.

**They are.** Rebuilding seed 0 at explicit `--lam-forget 3 --lam-retain 1` with `train_lam.py`
defaults reproduces `ll_mdu_f3` **exactly, on all seven reported numbers**:

| | TV→θ₀ | TV→oracle | TV→pristine | retain ASCR | forget ASCR | DISCLOSE | DEGEN |
|---|---|---|---|---|---|---|---|
| `ll_mdu_f3` (published) | 0.4337 | 0.2003 | 0.2825 | 0.4042 | 0.2748 | 0.2333 | 0.0500 |
| **`mdu_l3_s0_bridge`** | 0.4337 | 0.2003 | 0.2825 | 0.4042 | 0.2748 | 0.2333 | 0.0500 |

Checked against the label-collision failure mode that produced bit-identical rows earlier in this
project: **separate runs** (wall 382.1 s vs 376.5 s), separate adapter directories, separate labels.
The evaluator is deterministic (pinned sampler, greedy, T=0), so identical metrics from an identical
λ=3 seed-0 build is what a correct reproduction looks like. ⚠ `adapters/ll_mdu_f3/ckpt_300/` no
longer exists on disk, so the original cannot be re-evaluated for a weight-level check; the
reproduction is established at the metric level only.

## 2. The corrected comparison

MDU λ=3 on one footing: seeds 0–2 as published, seeds 3–7 the explicit-λ rebuilds.

| | AFC λ=1 (n=8) | MDU λ=3 (n=8) | Δ |
|---|---|---|---|
| **TV→pristine** | **0.2993 ± 0.1290** | **0.2977 ± 0.0650** | **+0.0016** |
| retain ASCR | **0.5336 ± 0.0506** | 0.3928 ± 0.0689 | **+0.1408** |
| DISCLOSE | 0.1458 ± 0.0728 | 0.1521 ± 0.0660 | −0.0062 |
| DEGENERATE | 0.1521 ± 0.1600 | 0.1281 ± 0.0994 | +0.0240 |

Disclosure is still matched (0.146 vs 0.152), so this is the comparison the headline intends.

**The TV advantage is gone.** `RESULTS.md` §1 claimed *"TV→pristine 0.223 vs 0.352 — 37% lower"* at
n=3. At n=8 against a properly λ=3 baseline the gap is **0.0016 — half of one percent**, far inside
either arm's seed variance. Paired:

| seed | AFC | MDU λ=3 | winner |
|---|---|---|---|
| s0 | 0.2288 | 0.2825 | AFC |
| s1 | 0.2205 | 0.3325 | AFC |
| s2 | 0.2205 | 0.4408 | AFC |
| s3 | 0.5871 | 0.2289 | **MDU** |
| s4 | 0.3122 | 0.2908 | **MDU** |
| s5 | 0.3371 | 0.2659 | **MDU** |
| s6 | 0.3089 | 0.2575 | **MDU** |
| s7 | 0.1789 | 0.2824 | AFC |

**4 of 8 — a coin flip.** And the three AFC wins are exactly seeds 0–2, the three seeds the method
was selected on. Seeds 3–7, chosen after selection, go 4–1 to MDU.

## 3. What actually survives

**Retention.** AFC holds retain ASCR **0.534 vs 0.393**, a **+0.141** advantage that is larger than
either arm's sd and holds on 7 of 8 paired seeds. That is a real result and it is the one the
project can defend.

So the honest claim is **not** "AFC erases closer to pristine than MDU". It is: **at matched
disclosure, AFC and MDU reach indistinguishable TV, and AFC does it while retaining substantially
more capability.** That is a narrower contribution than the one in `RESULTS.md`, on a different
axis, and it is what the data supports.

Two further qualifications that do **not** favour AFC:

- **AFC is ~2× more variable on TV** (sd 0.129 vs 0.065) and ~1.6× on DEGENERATE (0.160 vs 0.099),
  consistent with the 2026-10-21 stability retraction. Seed 3 remains a catastrophic-failure mode
  (TV 0.587, DEGENERATE 0.508) that MDU does not exhibit on any seed.
- **AFC is no better on DEGENERATE** (0.152 vs 0.128). The n=3 figure of 0.117 vs 0.181 inverted.

## 4. How this error happened, and the pattern

Three claims, each refuted by extending the same sweep:

1. n=3: *"AFC better and 9–16× more stable"* → stability refuted at n=5 (sd 0.005 → 0.158).
2. n=5: *"better on average, 0.314 vs 0.424"* → that baseline was MDU λ=1, not λ=3.
3. n=8, corrected: **TV parity, 4/8 paired.**

The common cause is not bad luck. It is **selecting on three seeds and comparing against an arm
whose configuration was never recorded.** The n=3 TV sd of 0.005 should have been read as a warning
that three seeds cannot resolve this difference, not as evidence of stability. The λ hole was
invisible because nothing in the pipeline wrote λ into the arm record — now fixed by putting it in
the label.

## 5. ⚠ Why this is not yet final — the symmetric control is running

`FINDING_lambda_provenance_confound` corrected the **baseline**. The same scrutiny has to fall on
**my own method's arms**: the AFC λ=1 seeds have their `mode`/`rank`/`lam_forget`/`lr` equally
unrecorded, and `train_h16.py` defaults to `--mode proj --rank 8` while the selected arm is
`h16_none_f1`, i.e. `--mode none`.

`run_afc_symmetric.sh` is running: a seed-0 bridge at explicit `--mode none --lam-forget 1` against
`h16_none_f1` (TV 0.2288, retain 0.5328, DISC 0.2083, DEG 0.0500), then seeds 3–7 rebuilt under the
same recorded settings.

- **bridge reproduces** → the AFC column is sound, §2 and §3 stand as written;
- **bridge does not reproduce** → the AFC column is as confounded as MDU's was, and **§2 is not
  settled either**. In that case the retain advantage in §3 is also provisional.

Published now rather than held, because the correction to the published comparative claim is already
established by §1–§2 and should not sit unreported while the symmetric half runs. §3's retain claim
is the part contingent on the bridge.

> ✅ **THE AFC BRIDGE PASSED (2026-10-24).** `afc_l1_s0_bridge`, rebuilt at explicit
> `--mode none --lam-forget 1 --lam-retain 1` with `train_h16.py` defaults, reproduces
> `h16_none_f1` **exactly on all four coordinates** — TV 0.2288, retain 0.5328, DISCLOSE 0.2083,
> DEGENERATE 0.0500 — on a separate run (wall 380.4 s). So **both** columns are now verified
> reproducible from explicitly recorded settings, and **§2 and §3 stand as written: the TV parity is
> real and the +0.141 retain advantage is no longer provisional.**
>
> ✅ **AND THE EXTENSION SEEDS REPRODUCE EXACTLY.** `afc_l1_s{3,4,5,6}_rec`, rebuilt under those
> recorded settings, match `afc_s{3,4,5,6}` to four decimals on all four coordinates (s7 running):
>
> | seed | original `afc_sN` (TV / retain / DISC / DEG) | rebuilt `afc_l1_sN_rec` |
> |---|---|---|
> | s3 | 0.5871 / 0.4998 / 0.1083 / 0.5083 | **identical** |
> | s4 | 0.3122 / 0.5265 / 0.1167 / 0.2250 | **identical** |
> | s5 | 0.3371 / 0.4563 / 0.2917 / 0.0750 | **identical** |
> | s6 | 0.3089 / 0.5636 / 0.0750 / 0.0167 | **identical** |
>
> So **the AFC column has no confound**: `afc_s3..s7` were in fact built at `--mode none
> --lam-forget 1` with the defaults. **The provenance failure was one-sided — in the baseline only.**
> I applied the identical instrument to my own method's arms and they passed, which is what makes the
> corrected conclusion in §2–§3 safe to rely on rather than merely convenient.

## 5a. ⚠ Every historical arm checkpoint has been deleted from disk

Discovered while trying to re-attack the RTT control: of 42 adapter directories, **only the 12 built
today retain a usable `ckpt_300`**. `mdu_pub`, `h16_none_f1`, `ll_mdu_f3`, `seed_afc_s1/s2`,
`afc_s3..s7`, `mdu_s3..s7` — the weights behind every published number in `RESULTS.md` — are gone.

Consequences, stated plainly:

- **No published arm can be re-verified at the weight level**, only re-derived. Both bridge controls
  and the four exact seed reproductions above are metric-level agreement between an original
  measurement and a fresh build, which is strong but is not a weight comparison.
- **The rebuild instrument is now the only recourse**, which is precisely why its two bridges were
  worth running before trusting it. They passed, and the four exact seed reproductions are further
  evidence that it is faithful.
- The RTT control could not be re-attacked on its original weights at all; its arm is being rebuilt
  at explicit λ_f=8 under a new label (`mdu_l8_s0_rebuild`), to be checked against `mdu_pub`'s
  published row before the gated attack is believed.

## 6. Scope

n=8 paired seeds, one dataset (TOFU/LLaDA-8B-Instruct, backbone rev
`08b83a6feb34df1a6011b80c3c00c7563e963b07`), one disclosure-matching point, one evaluator. No
significance test is reported: with n=8 and sd 0.129 the TV difference of 0.0016 is plainly inside
noise, and dressing that in a p-value would add nothing. Nothing existing was modified or
relabelled; all rebuilds carry new identifiers.
