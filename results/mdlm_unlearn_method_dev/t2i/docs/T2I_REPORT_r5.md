# T2I relearning-robust erasure — report r5 (post-audit)

> **Status pointer:** for the **one current verdict per issue**, read `STATUS_CURRENT_v2.md`. `AUDIT_REPAIR_REGISTER_v1.md` is chronological and records hypotheses that were later refuted — two of them ("IMMA composition unproven", "upstream UCE may have written an unedited model") are **refuted** and must not be quoted as live.

> **⚠ Corrections 2026-10-26:** `FINDING76` §3's probe mechanism check is **overstated** (n=7/3/1, within-arm spread 40×, ordering undetermined — see its §7). `FINDING77` §2's "B kills the accumulation-order hypothesis" is **wrong** — B holds accumulation order fixed and never varied it (see its §6).


**Supersedes `T2I_REPORT_r4.md`.** r4's headline attribution, counts, protocol-compliance and
mechanism claims were wrong in the specific ways listed in `CORRECTION_REGISTER_r5.md`; its text is
preserved under a SUPERSEDED banner.

**Every number in §2–§4 is generated from `outputs/registry_v1.json` via
`scripts/registry_v1.py`, not transcribed.** The generated tables are
`docs/TABLES_v1_generated.md`. **But that registry is stale** — generated 2026-10-26 05:11, it
predates 150 JSON files including 112 attack records. §2 states the gap explicitly; §9 records why
it was not regenerated.

**Read `CORRECTION_REGISTER_r5.md` first** if you want to know what was wrong and what is still
unvalidated. This report states only what currently holds.

---

## 1. One-paragraph status

The project has **not** produced an unlearning method that beats the strongest existing methods on
the forgetting–utility trade-off, and the mandate's bar — retain8 ≥ 0.85 **and** max-over-valid
H ≤ 0.40 — **is unmet by every arm measured**. What it has produced is a set of empirical
counterexamples: on every defence build that cleared the utility floor, including one at retain8
0.8667 with **zero** pre-attack target detections and one on SD-1.5 at retain8 0.9583, a 300-step
full-UNet relearning attack restored the target to 15/15 — frequently while **improving** the
measured non-target panel. Following the external audit, three of those defences are now known to
be **our variants rather than the published methods** (the weight-space regulariser runs in the
opposite direction, the Meta-Unlearning port omits two of three objective terms, and UTRAP's inner
attacker was not the evaluation attacker). **All three corrected implementations have since been
run** — WDR v2 at α ∈ {1.0, 4000} (E1), Meta v2 at λ ∈ {0.5, 1.0} (E2/E2g/E2h), UTRAP v2 at
K ∈ {0,1,8,32} (E4/E4m) — and **upstream IMMA and upstream UCE have each been executed once**
(R1, R4b–d), scored on our benchmark. None of that rescues any defence. So the negative result is
now a statement about **our implementations, plus two upstream-code runs measured on our own
benchmark** — and still not about any published method's reported experiment.

## 2. What the records show, with admissibility separated

> **⚠ These counts are STALE.** `outputs/registry_v1.json` was generated **2026-10-26 05:11:25**.
> Since then **150 JSON files** have landed, including **112 attack records** and **≥11 new defence
> builds**. The registry column below is exactly what the registry says; the **actual** column is a
> direct count on disk. The registry has **not** been regenerated — `scripts/registry_v1.py` sha256s
> many GB and the project is frozen (`FREEZE_ACTIVE`). **Regeneration is required before any of
> these numbers is republished.**

| quantity | registry (05:11) | actual on disk (12:45) |
|---|---:|---:|
| JSON files under `outputs/` | 1,278 | **1,424** (1,428 incl. subdirs) |
| attack **records** (`atk_*.json`) | 696 | **808** |
| — quarantined | 0 | not recomputed |
| — **vacuous** (defence erased nothing; `H_after` measures the base model) | 9 | not recomputed |
| — VOID post-attack **utility** half (text encoder omitted; T/H unaffected) | 136 | not recomputed |
| **admissible for recovery** | 687 | not recomputed |
| **admissible for paired utility** | 324 | not recomputed |
| defence builds evaluated | 335 | **≥346** |

The 112 new attack records are `E12` (24), `E2G` (18), `E4MG` (12), `E2H` (12), `E11J` (9),
`E11M` (9), `E11K`/`E11L`/`E11N` (6 each), `E11H` (5), `E11I` (3), `E11O` (2). The new builds
include `meta2_church_lt{0.5,1.0}`, eight `utrap2*` arms and `REF_imma_vangogh`.

*(The previous revision of this section carried a parenthetical — "Counts regenerated 2026-10-26
after FINDING65–70 added ~54 attack records ... no new defence builds" — which was **itself stale**
when written: `FINDING71`–`FINDING78` post-date the registry, and E2/E4/R1 did add new defence
builds. It is withdrawn.)*

**Every count below in this section is a registry figure**, so each inherits the staleness above.
Each is exact against the registry and was re-verified against it field by field; none has been
re-verified against the repository.

**696 — and the 808 now on disk — is a count of files, not of admissible independent
experiments**, and even 687 pools fixed-grid confirmatory runs with adaptive escalation. The
generated tables split every arm by **budget** so the two are never pooled. r4's "609 attacks /
1,088 JSONs" was stale; so, now, is this block, which is the recurring failure mode here and the
reason the staleness is stated above rather than silently refreshed.

The nine vacuous records are **six** attacks on `gift_church_b0.1` and **three** on
`uce_sd15_church_lam50000`, both at as-is target suppression **1.0000** and retain8 **1.0000** — they
erased nothing, so reversing them is not a result. The six GIFT ones were previously inside the
"GIFT fully reversed" aggregate. Removing them **reduces** our headline count, which is the
direction a correction should go. The three UCE ones were additionally caught by the **validity
gate** on their own (T could not rise, already 1.000), which is the gate working as designed.

## 3. The central result, stated at the strength the data supports

### 3.1 The qualifying SD-1.4 arm

`utrap_church_l0.15k8` — **target suppression 0.0000, retain8 0.8667** (all-non-target utility
**0.8364**, a different denominator and below 0.85; the two are not interchangeable).

Six 300-step attacks (7 records, one duplicated label), all gate-valid, T 0.000 → 0.958–1.000.
**Five of the six reach H = 1.000.** The sixth —
`H84M_utrap_church_l0.15k8_sd0_lr1e-4` — reaches **H = 0.9333** at T 0.9583, and its paired utility
**collapses 0.8333 → 0.1875** under the closed-set scorer, so its lower H is not evidence of
robustness (`FINDING70`, `FINDING71`). *"Collapsed closed-set panel accuracy" is what was measured;
"destroyed the model" is not — see §5 and `D-030`. No images from this attack were saved and no
independent quality or semantic assessment exists.* Restricted to the five attacks whose panel
accuracy did not collapse, T rises to
exactly **1.000** in every case, so the quoted 0.958 lower bound belongs to the one run whose
panel accuracy collapsed.

*(The previous revision read "all six reaching H = 1.000" while quoting a T range whose lower bound
was the one record that did not. The sentence was self-inconsistent and is rewritten, not
annotated.)*

The three primary lr 1e-5 attacks have **paired** 48-image utility:

| attack seed | T rise | H after | utility before | utility after |
|---|---:|---:|---:|---:|
| 0 | +1.0000 | 1.0000 | 0.8333 | **0.9792** |
| 1 | +1.0000 | 1.0000 | 0.8542 | 0.9167 |
| 6 | +1.0000 | 1.0000 | 0.8750 | 0.8958 |

Seed 0 is 40/48 → 47/48 on the measured panel. **The supported paired comparison is
0.8333 → 0.9792** — not 0.8667 → 0.9792, which mixes panels and denominators, as r4 did.

**This is the project's strongest finding** and it is a counterexample, not a method: target
recovery did not require collateral collapse.

### 3.2 Cross-model, at the narrower scope that is actually supported

Five port families × three checkpoints, **church only**, **one defence build per setting**, three
primary-lr attacks each:

| port | model | as-is target | as-is retain8 | gate-valid | max-over-valid H |
|---|---|---:|---:|---:|---:|
| FRP | SD-1.5 | 0.1333 | 0.9333 | 3/3 | 1.000 |
| FRP | BK-SDM-small | 0.4667 | 0.9250 | 3/3 | 1.000 |
| GIFT | SD-1.5 | 0.4000 | 0.8167 | 3/3 | 1.000 |
| GIFT | BK-SDM-small | 0.3333 | 0.7417 | 3/3 | 1.000 |
| UTRAP (ours) | SD-1.5 | **0.0000** | **0.9583** | **4/4** (3 seeds, one duplicated) | 1.000 |
| UTRAP (ours) | BK-SDM-small | 0.0000 | 0.7917 | 3/3 | 1.000 |
| UCE | BK-SDM-small | 0.0000 | 0.9333 | 3/3 | 1.000 |
| UCE | SD-1.5, λ=1000 | 0.2000 | 0.9500 | **6/6** (3 seeds, duplicated labels) | 1.000 |
| Meta **variant A** | SD-1.5 | 0.0000 | 0.7667 | **2/2** | **1.000** (see `FINDING64`) |

`utrap_sd15_church_l0.15k8` at **target 0.0000 / retain8 0.9583** is the best as-is operating point
measured anywhere in the project, and it is reversed 3/3 with T = +1.000 each time.

**This is not "cross-model generalisation."** It is five of our ports, on SD-1.x plus one pruned
derivative, one concept, one build each, three attacks each. It does **not** establish universal
diffusion-model behaviour, and BK-SDM's recovery does **not** rule out capacity as a variable
(`ERRATUM_interpretations_r5.md` §4).

### 3.3 The λ response curve for the closed-form class, completed

UCE on SD-1.5, after the λ-scale correction (`ERRATUM_uce_lambda_scale.md`):

| λ | as-is target | as-is retain8 | reading |
|---:|---:|---:|---|
| 0.1 / 0.5 / 1.0 | 0.0000 | 0.5333 / 0.4667 / 0.4167 | λ **inert** — one arm sampled three times |
| 10 | 0.0000 | 0.3500 | |
| 30 | 0.0667 | 0.4583 | |
| 100 | 0.0000 | 0.5667 | |
| 300 | 0.0667 | 0.8083 | |
| **1000** | **0.2000** | **0.9500** | **usable; clears K1** |
| 10000 / 50000 | 1.0000 | 1.0000 | `W_new → W_old` — **no erasure at all** |

`cᵀc`'s max eigenvalue for SD's 768-dim CLIP embeddings is **3.24e5**, measured (`D-033`), which is
why values of order 1 do nothing. *(The figure stated here previously was **85,850**, carried from
`ERRATUM_uce_lambda_scale.md` and never measured; the measurement is **3.8× larger**, so the
qualitative point — λ ≤ 1e3 is inert against this spectrum — is strengthened, not weakened.)* The window is now bracketed from both sides **by measurement**. The λ=10000/50000 arms
are vacuous and excluded; the λ=50000 attacks were correctly caught by the **validity gate** (T
could not rise, already 1.000), which is the gate working as designed.

## 4. What every claim-level correction amounts to

Full detail in the errata; the short list:

| r4 claim | r5 |
|---|---|
| all five published competitors discharged | **withdrawn.** Three were tested as our variants; the published WSR and Meta methods are **untested**, not refuted; three of the five have **released code we have not run** |
| 13 defences | replaced by the enumerated build table in `TABLES_v1_generated.md` §1–2 |
| erasure depth / geometry / unroll depth do not predict resistance | **"the measured differences did not prevent reaching the ceiling under the attacks run."** A saturated H = 1.000 cannot rank predictors |
| suppresses expression rather than removing capability | **withdrawn.** Target-assisted recovery cannot separate reactivation from relearning (`ERRATUM_r2` C17, re-breached by r4 §7) |
| protocol unchanged throughout | **false.** Fixed grid, prespecified secondary and adaptive escalation now separated by budget |
| only FRP rerun cross-model | GIFT, UTRAP, Meta variant A and UCE are all cross-model now |
| GIFT β∈{1.0…1.35} target range starts 0.0667 | **0.2000–0.4667** |
| Meta target 0 throughout; retain range omits λ=0.6 | λ=0.5 target **0.0667**; λ=0.6 retain8 **0.6250** |
| WSR2 beats FRP's retention | **descriptive, selected configuration.** The 95% t interval over three builds contains the teacher value; and the arm is not an implementation of the cited baseline |
| 1,000+ experiments | admissibility-separated counts in §2 |

And three corrections of my own, not the audit's:
- **`FINDING59`'s 22 valid attacks is correct** — the audit's recount of 21 undercounts TI by one.
- **Six attacks were vacuous** (§2) and are removed from the aggregate.
- **Two further split leaks exist** (`erase_adv` trains on six utility-panel prompts; `undo_t2i`
  shares six prompts with the attacker's pool) that no previous check computed.

## 5. Measurement caveats that apply to every number above

- **H is 15 images**, 3 prompts × 5, closed-set 12-label CLIP argmax, no reject option, quantised to
  1/15. H = 1.000 does not mean the model was restored; H = 0 does not mean the capability is
  absent. An exact binomial interval for 15/15 bottoms out near 0.782, and even that model is
  inadequate given prompt clustering and adaptive selection.
- **The closed-set scorer has two label arrays, and the FINAL path used the wrong one.**
  `gen_eval.py` defines `LABELS` (`:48`, the CLIP text, e.g. `"van gogh painting"`) and `LAB_KEY`
  (`:50`, the internal record keys, e.g. `"van_gogh"`). **Nine of the twelve label strings differ;
  `church`, `parachute` and `tench` are spelled identically.** `gen_eval_final.py` prompted CLIP with
  `LAB_KEY`, so every FINAL number was computed against a label set that is not the one the scorer
  was designed around (`D-022`). **The evaluator used inconsistent label text. The effect on scores
  and conclusions is unresolved pending corrected evaluation.** No direction or magnitude of the
  embedding effect is established here: whether the substitution raised, lowered or left unchanged
  any particular arm's score is a question for paired re-scoring, not for inference from the
  substitution itself. Only `outputs/final_*.json` is implicated; the DEV path calls `LABELS`
  correctly. The defect is now repaired in `gen_eval_final.py`, but the 18 FINAL records predate the
  repair and **have not been re-scored**. Re-scoring is CPU-only.
- **Every target number is a DEVELOPMENT measurement.** One of three evaluation prompts was a
  defence-training prompt for GIFT/IMMA/WSR/UTRAP, and λ/β/K were repeatedly selected while watching
  H. A frozen **FINAL** set exists but has been **consumed**: experiment E6 evaluated four arms on it
  (`FINDING65`, `FINDING66`), with the label-text defect above. *(The previous revision of this bullet
  said nothing had been evaluated on it; that was already false when written.)*
- **AdvUnlearn utility figures are partly in-sample** (six panel prompts).
- **Three utility denominators** (retain8 n=8, all-non-target n=11, 48-image paired panel) are
  reported separately and are not interchangeable.
- **Gated and ungated outcomes** are both reported. An ineffective attack is **inconclusive** —
  neither robustness nor failure.
- **FID is absent.** No claim that any attacked model is globally repaired or preserves general
  image quality.
- **No never-trained oracle exists**, so reactivation vs relearning is unresolved by construction.
- Pre-2026-10-24 checkpoints were deleted and are re-derivable but **not weight-verifiable**; the
  registry records `ABSENT` rather than treating them as present. This covers **234 of 335** builds
  (94 `PRESENT`, 7 with no checkpoint field), so roughly **70%** of the build corpus cannot be
  weight-identified.
- **Seven historical `gen_*.json` records do not record which checkpoint they scored.** Their `unet`
  field is `null`, which is ambiguous: in `gen_base_sd14.json` and
  `gen_E6cal_BASE_sd14_on_DEV.json` it means the unmodified base (both score **1.000 on all twelve**
  DEV concepts, which is the control denominator; that establishes the scorer behaves correctly on
  **undamaged** base outputs and **does not** establish its validity on damaged or unusual outputs —
  see `D-030`), but
  in `gen_advunlearn_{official,parachute,tench,vangogh,garbage_truck}.json` it accompanies results
  that demonstrably came from an **edited** model (e.g. parachute 0.1333, garbage_truck 0.1333,
  van_gogh 0.4667) whose identity was never written down. Those five also carry `target: null`, so
  the active target is only inferable from the filename. **The numbers are historical results and
  are retained; their attribution is not available.** `TABLES_v1_generated.md` prints all six as
  `—/—/—`, so no figure is attributed to them there; the two `h63_*_a0.3` builds are the one case
  where a provenance-gap build **does** carry numbers (9 attacks, max H 1.000, in
  `TABLES_v1_generated.md` §1) against an unrecorded target and an `ABSENT` checkpoint. Newer
  records (`gen_REF_imma_vangogh.json`, `gen_meta2_*`) record both fields, so this is a closed
  historical defect, not a current one.

## 6. Where this leaves the mandate

**Unmet, and no reinterpretation changes that.** No arm reaches retain8 ≥ 0.85 with max-over-valid
H ≤ 0.40, at any budget. The retain half of the bar is reachable; the robustness half is not; and
clearing the first did nothing for the second.

**This now rests on measurement rather than on absence of measurement, and it came close to
failing.** The closest arm in the project is `meta2_church_lt0.5` (E2, full Eq. 8) at retain8
**0.9000** with as-is target **0.0000**. At a 25-step budget it clears the robustness half at
attacker lr 1e-5 (max-over-valid **0.0667**) *and* at lr 5e-5 (**0.2000**). It fails only when the
attacker's learning rate is swept further: at lr 1e-4, same budget, max-over-valid is **0.5333**
(`logs/e2h_stronger.log`). Because max-over-valid is a ratchet (`FINDING35` S4), that 0.5333 stands
and can only rise, so the 25-step budget fails the bar. `FINDING78`'s companion claim to have
produced "the first arm to meet both mandate halves" is **retracted** on exactly this evidence;
its retention result — **+0.20 to +0.27** retain8 at identical suppression, against variant A's
0.6250 (λ=0.6) and 0.7000 (λ=0.7) — survives and is the real finding there.

Two consequences worth stating. First, **single-lr robustness numbers are not safe**: this arm would
have been reported as mandate-clearing had the attacker sweep stopped one cell earlier. Second, the
retain half of the bar is not merely reachable but reachable **at zero target suppression**, which
makes the robustness half the whole of the remaining problem.

What the project has measured, stated at the scope §3 supports: on the builds measured, with the
attacks run, concept erasure in SD-1.x was reversible by 300 gradient steps on 16 self-generated
images, across four adversary families down to 768 trainable scalars, and reapplying the original
pruning mask to an attacked model does not eliminate target generation. The mechanism is **not**
established — neither the geometry story, nor the capacity story, nor the reactivation story survives
its controls.

**Prior-art overlap, recorded factually, and no contribution decision taken here.** arXiv
**2608.25429** is **FRAG** — verified, and already the source of this project's own FRP reference
(<https://arxiv.org/abs/2608.25429>, <https://arxiv.org/html/2608.25429v1>,
<https://github.com/Yi1-Chen/FRAG>). Its **Table 11** overlaps this project's **attack-validity**
argument (an attack that recovers nothing does not probe robustness) and its **Table 15** overlaps
the **initial-forgetting-versus-recovery** argument (an inversion against a retain-trained gold
model). FRAG's tables are LLM relearning; this project's are T2I diffusion. **That overlap does not
determine where a contribution can lie.** A contribution decision requires a **claim-by-claim
literature comparison** — which specific claims are new, which are anticipated, and in what setting —
and **that comparison is not part of this repair gate and has not been done.** Nothing here
forecloses either an evaluation-side or a method-side contribution. *(An earlier reading concluded
from this overlap that publication would require a new defence method and that the evaluation-side
contribution was foreclosed. That does not follow from an overlap on two tables and is withdrawn.)*

**The most promising direction**, unchanged from the audit's assessment and now better supported by
the repaired tooling: a carefully controlled study of **adaptation vulnerability and final-mask
necessity** (experiment E5), with the three released reference implementations actually run
(E7), evaluated on a **freshly frozen** FINAL split (E6's was consumed). Whether that, an
evaluation-side contribution, or a method-side one is the right target is **not settled by this
report** — see the claim-by-claim comparison requirement above. What is settled is that **a new
unlearning method is not something this project currently has.**

## 7. What is pending, with prices

`CORRECTION_REGISTER_r5.md` §B.10 lists ten experiments, E1–E10, each with configuration and cost.
**Four of them have since run**, so the previous revision's "unblocked and cheap, not yet run"
framing is withdrawn:

| id | state (artifact-verified 2026-10-26) |
|---|---|
| **E1** faithful WDR | ✅ **RAN** at α ∈ {1.0, 4000} — *not* the preregistered {0.3, 1.0, 3.0}. `logs/e1_wdr.log`, `logs/e1b_wdr.log`, `outputs/gen_wdr2_church_a{1.0,4000}.json`. `FINDING67`: α=1 inert, α=4000 fails K1 at 0.5917. **α ∈ (1, 4000) is still unmeasured** |
| **E2** Meta v2 full Eq. 8 | ✅ **RAN**, λ ∈ {0.5, 1.0} as specified, plus an 18-cell frozen grid (E2g) and an attacker-lr sweep (E2h). `FINDING78` |
| **E4** UTRAP v2 K-sweep | ✅ **RAN**, K ∈ {0,1,8,32}. `logs/e4_h90.log`, `e4m_matched.log`, `e4mg_grid.log`, 8 `utrap2*` builds. `FINDING76` (whose §3 mechanism check is overstated — see its §7) |
| **E6** FINAL-prompt re-evaluation | ⚠ **RAN BUT MUST BE RE-SCORED** — `outputs/final_*.json` (18 records, which **consumed** the FINAL set) were scored with `LAB_KEY` instead of `LABELS` (`D-022`); nine of the twelve label strings differ. **The evaluator used inconsistent label text; the effect on scores and conclusions is unresolved pending corrected evaluation.** That applies to `FINDING65` and `FINDING66` alike. **Re-scoring is CPU-only on retained artifacts** and is the cheapest outstanding repair in this table |
| **E7** the three released implementations | ⏸ **PARTIALLY RAN** — upstream IMMA (R1, 5000/5000 + eval) and upstream UCE (R4b–d) executed. `vision_relearning` (R3) **not started** |
| **E5** persistent-mask | ❌ not run |
| **E3** variant A vs v2 at matched retain8 | ❌ not run |
| **E8** FID | ❌ blocked — two pip installs and a ~6 GB COCO download, not authorised |
| **E9** SD-2.x / SDXL | ❌ blocked — `stabilityai/*` returns HTTP 401; needs a Hugging Face token |
| **E10** never-trained oracle | ❌ not run; a design decision, not compute |

**Immediate operational blockers:** `/data` is at 100% with ~14 GB free, shared with another
project, so no multi-checkpoint campaign can start safely; and `FREEZE_ACTIVE` (declared
2026-10-26T12:43) bars all new GPU work pending the review gate in
`AUDIT_REPAIR_REGISTER_v1.md` §10.

---

## 8. Addendum — findings since r5 was written (FINDING65–78)

r5's body above predates **fourteen** findings produced during and after the audit remediation. They
do not change its §6 conclusion — the mandate is unmet — but several change figures r5 quotes, and
two are corrections r5's own numbers depend on. Summarised here so the current report is not stale
against its own repository.

*(The previous revision of this section covered FINDING65–70 only, while claiming the report was
"not stale against its own repository". `FINDING71`–`FINDING78` were already on disk. Four of the
eight omitted findings amend rows in this very table — 71 amends row 70, 72 amends row 68, 74
generalises 68–73, 75 retracts a number §3 of this report once relied on. §8.2 adds them.)*

| # | finding | effect on r5 |
|---|---|---|
| **65** | ⚠ **DO NOT QUOTE (audit defect `D-022`).** The claim was: on **FINAL** prompts erasure is **~3× shallower** than every reported figure (FRP 0.1333 → 0.4667, UCE 0.2000 → 0.6000, UTRAP 0.0000 → 0.0667), with base calibration bounding prompt difficulty at ≤13%. **But every FINAL number was scored with `LAB_KEY` instead of `LABELS`** (`gen_eval.py:48` vs `:50`). **Nine of the twelve label strings differ; `church`, `parachute` and `tench` are spelled identically.** **The evaluator used inconsistent label text. The effect on scores and conclusions is unresolved pending corrected evaluation.** Retained per-image facts, recorded without a directional reading: **37 of 180 FINAL images landed on one of the nine labels whose text differs**, at per-arm median margins **0.0027–0.0063** (frp 7/15 with 7 such wins; uce_sd15_lam1000 9/15 with 5; utrap 1/15 with 10; utrap_sd15 2/15 with 12). A further axis is that the ratios compare a DEV number scored from `LABELS` against a FINAL number scored from `LAB_KEY`, so the two sides of each ratio did not use the same label text at all | **the "~3×", "~3.5×", "54% undone" and "60% undone" figures are UNMEASURED pending corrected evaluation.** Neither the size nor the sign of the scorer's effect on them is established, so the finding is neither confirmed nor refuted here. The split-leak and prompt-difficulty reasoning is a separate argument and is untouched by this defect |
| **66** | ⚠ **Do not quote its figures (`D-022a`) — the same label-text defect applies.** The claim was that the **H = 1.000 ceiling is not a DEV artifact**: on FINAL prompts the attack reaches **15/15** on all four arms. **The evaluator used inconsistent label text. The effect on scores and conclusions is unresolved pending corrected evaluation** — including for the within-scorer comparison, because consistency between two arms scored the same wrong way does not establish what either score would be under the correct label text. *(The previous revision argued the ceiling conclusion "survives the defect by construction" because strengthening distractors could only lower a 15/15. That inference assumes a direction for the embedding effect, which is not established, and is withdrawn.)* | §3's robustness results are **DEV-scored and stand on the DEV evidence**; `FINDING66` neither adds to nor subtracts from them until E6 is re-scored. Also unresolved: the subsidiary claim that the attacked SD-1.4 model (15/15) **beats the unmodified base** (13/15) — a **2-image gap** on n=15 |
| **67** | **α is inert at the reference's default in our port**, and at α=4000 the arm **costs 0.35 of retention without improving on the teacher's erasure** (K1 FAIL at **retain8 0.5917**). *(Corrected — audit defect **D-017**.* The derivation ‖∇L‖ = α/√N stands, but **it is not the cause**: measured step size as a fraction of `lr` is α=1 → 0.104, α=10 → 0.537, α=4000 → 0.998, so α 1→4000 buys **≈9.6×, not ~4,000×**. The mechanism is **our optimiser plumbing, not the method** — both steps share **one Adam instance**, so `v` is retain-dominated and the distance update is ≈ `lr·|g_dist|/√v_retain`; α is inert at **both** ends, below ≈10 because `v` is retain-dominated and above ≈4e3 because **Adam saturates**. The reference **sums** both terms into one step, so our two-step schedule is the defect, not the published method. The **"third instance of one scale pathology"** framing is **withdrawn** for the WDR member: IMMA α and UCE λ are coefficient-scale findings about those *methods*, this one is about our schedule.* | `'From Dormant to Deleted'` has **one** measured point here, not zero — a utility failure at one α. **0.5917 measures our two-step shared-Adam schedule at saturation, not "WDR at gradient parity"**; that attribution is withdrawn. α ∈ (1, 4000) is unmeasured, and its upper bound is a **saturation point, not a frontier**. See `FINDING67` (header + §7) and `AGENT3_FIDELITY_FINDINGS.md` §4.2–4.3 |
| **68** | The attacker's real cost is **≤100 steps, not 300**; at 25 steps the mandate-qualifying arm is already at 13/15. Reproduced on FRP (ceiling by 50) | r5's "300 gradient steps" **overstates the attack cost 3–6×** |
| **69** | Resolved to 5 steps, recovery is **graded** — the project's first non-saturated outcome variable. **Its headline threshold was retracted the same day** by seed replication (sd 0.2776 at 10 steps, H spanning 0.1333–0.6667). Completed at n=3: **no budget from 5 to 300 leaves any arm meeting the mandate's robustness half** | the mandate is missed at **every** budget, not just at 300 |
| **70** | **lr substitutes for budget.** lr 5e-5 / 25 steps gives H = 1.0000 on **all three seeds, sd = 0**, utility 0.9236 — **12× cheaper** than reported. lr 1e-4 / 5 steps reaches 1.0000 on one seed — **60×** cheaper. At lr 1e-4 the budget axis is **non-monotone** on this arm (H falls 0.9555 → 0.6222) while the attack drives the paired closed-set panel down (utility 0.8472 → **0.5278**; "collapsed panel accuracy", not "destroyed the model" — `D-030`) — **but see 71: the non-monotonicity is arm-specific and the causal link is withdrawn** | r5's cost figure is superseded; and **H alone cannot order attacks** — a low H may mean a weak *or* an overpowered attack, distinguishable only by paired utility |

### 8.2 The eight findings the previous revision omitted (FINDING71–78)

| # | finding | effect on r5 and on this table |
|---|---|---|
| **71** | **Half of 70's story failed replication.** The *damage* at lr 1e-4 is real and corpus-wide (mean post-attack utility **0.93 → 0.75 → 0.50** by lr, n = 229). The *non-monotone H* that 70 explained with it **did not reproduce** on a second arm and is **specific to one arm** | row **70**'s causal clause is withdrawn. The damage finding stands and is what justifies discounting the destructive attack in §3.1 |
| **72** | At matched lr with no panel collapse, **UTRAP is recovered as fast or faster than FRP**, reversing `FINDING68` §6's coarse-grid impression. Separately, the **validity gate's exclusion rate is non-monotone in `T_before`** (n = 520) | row **68**'s arm ordering is withdrawn. The second half is **structural**: "max-over-valid H" — the mandate's own metric, used in §3.2, §3.3 and §6 — is computed over a sample whose filtering **correlates with how well-erased the arm is**. Every max-over-valid figure in this report inherits that bias |
| **73** | The budget correction **generalises to further concepts**, but **not uniformly**: `garbage_truck` saturates in 25 steps, `church` is slower, and a third concept, `tench`, is **slowest**. Its own §6 retracts the "and gets worse" over-read | row **68** generalises beyond one concept, but the *budget needed varies substantially by concept*, so no single step-count is the attacker's cost |
| **74** | Across **six defence families**, not one buys the attacker more than **100 gradient steps**. Found by registry query over 36 existing church arms, not new builds | closes the "two families only" limit implicit in rows 68–70. One cell (Meta variant A at 25 steps) is the project's only sub-0.40 number and **fails the mandate's retention half anyway** at 0.7000 |
| **75** | **RETRACTED in full.** Its central number came from `r4_uce_official_vs_port.py`, which had three verified defects: char-split inputs (`list('church')`), an embedding selection matching neither implementation, and **no upstream code executed** | the λ-scale and port-fidelity claims this report makes about UCE must rest on `ERRATUM_uce_lambda_scale.md` and on R4b–d, **never** on the VOID `r4_uce_*.json` artifacts |
| **76** | Unroll depth (UTRAP v2, K ∈ {0,1,8,32}, E4 at matched compute) **buys nothing** once compute is controlled | answers the unroll-depth question r4 left open, and retires §4's "unroll depth does not predict resistance" row from "cannot be ranked" to "measured and flat". **Its §3 probe-mechanism check is overstated** — n = 7/3/1, 40× within-arm spread, ordering undetermined (its §7) |
| **77** | The UCE upstream-vs-port divergence is **embedding selection**, not the solver: with upstream's own selection the discrepancy falls from 0.384 to **1.5e-3** | corroborates `UPSTREAM_FIDELITY_MANIFEST.md` §7(d) by measurement. **Its §2 claim that column B "kills the accumulation-order hypothesis" is wrong** — B holds accumulation order fixed and never varied it (its §6), so accumulation order remains **unmeasured** |
| **78** | Restoring Meta-Unlearning's two missing Eq. 8 terms buys **+0.20 to +0.27 retain8 at identical (0.0000) suppression** — retain8 **0.9000** against variant A's 0.6250 (λ=0.6) / 0.7000 (λ=0.7). **Its companion claim to be "the first arm to meet both mandate halves" is RETRACTED**: at 25 steps the arm clears 0.40 at attacker lr 1e-5 and 5e-5 but reaches **0.5333** at lr 1e-4 | the **retention** result is the first substantive gain any corrected baseline has produced and is cited in §6. The **mandate** conclusion is unchanged — see §6, which now rests on this measurement rather than on its absence |

### 8a. The net effect on the project's standing

**One half moved against the defences; the other is unresolved.** The attacker is cheaper than
reported (68, 70 — both DEV-scored, through `gen_eval.py`'s `LABELS` path). So r5 §6's conclusion
holds and is **stronger** on the cost axis: on the builds measured, concept erasure in SD-1.x is
reversible by **25 gradient steps at lr 5e-5** — reproducibly, on all seeds, leaving the model
intact. That statement rests on DEV measurements alone; **no FINAL-split result currently supports or
undercuts it**, in either direction.

**The depth axis is withdrawn pending corrected evaluation.** The previous revision of this paragraph
said "the suppression it provides is ~3× weaker on prompts it was not tuned around". That rested on
`FINDING65`. **The evaluator used inconsistent label text. The effect on scores and conclusions is
unresolved pending corrected evaluation.** So **no multiplier is currently supported**, and the
held-out depth of erasure is **unknown until E6 is re-scored with `LABELS`**. *(The previous revision
also asserted that 37 of 180 images turning on margins of ~0.005 "cannot explain the whole of" the
0.1333 → 0.4667 gap. That is an estimate of the artifact's size derived from the artifact's assumed
direction, and both are unestablished; it is withdrawn.)*

### 8b. What has been corrected about my own process, twice over

- `FINDING67` §4 named a **three-instance** scale pathology and drew the lesson *"measure the scale
  before sweeping the knob."* The lesson was applied **before** the next two experiments
  (`diag_coeff_scale.py`): UTRAP-v2's λ is ~8× below parity, and **Meta-v2's coefficients are
  already near parity** — so the pattern does not always recur, which is why checking beats
  assuming in either direction. **But the three-instance generalisation itself has since been
  withdrawn** (audit defect **D-017**): the WDR member is an artifact of *our* shared-Adam two-step
  schedule, not a coefficient-scale property of the published method, so it does not belong in a
  series with IMMA's α and UCE's λ. The lesson survives; the pattern that motivated it was one
  instance short. **Diagnosing a shared knob is not the same as diagnosing a shared cause** — which
  is the sharper version of the same lesson, and the one this correction actually teaches.
- `FINDING69` §7 recorded the **fourth** retraction of a headline published before a declared
  escalation returned, and the rule that followed: *a sub-ceiling number is not publishable as a
  headline until its declared escalation has returned.* `FINDING69` §8 and `FINDING70` were both
  held to n = 3 before publication under that rule.

### 8c. Still true, and still the limiting facts

- **The mandate is unmet.** No arm reaches retain8 ≥ 0.85 with max-over-valid H ≤ 0.40, at any
  budget. The nearest approach (`meta2_church_lt0.5`, retain8 0.9000) fails at 25 steps only because
  the attacker-lr sweep reached lr 1e-4 (**0.5333**) — see §6. It is a near miss, not a margin.
- **All three corrected baseline implementations have now been validated on models** — WDR v2
  (E1, α ∈ {1.0, 4000}), Meta v2 (E2, λ ∈ {0.5, 1.0}) and UTRAP v2 (E4, K ∈ {0,1,8,32}) — and
  **two of the three released reference implementations have been executed** (upstream IMMA via R1,
  upstream UCE via R4b–d). `vision_relearning` (R3) remains unrun. *The previous revision asserted
  the opposite of all three of these; it was wrong on each.*
- **None of those runs is a reproduction.** R1 and R4 used upstream code but **our** training
  images, **our** `gen_eval`, **our** 12-label closed-set CLIP scorer and **our** prompt pools. No
  published method's reported experiment has been reproduced, and none has been refuted.
- **Every max-over-valid figure in this report carries the gate bias `FINDING72` measured** — the
  validity gate's exclusion rate is non-monotone in `T_before`, so the metric is computed over a
  sample whose filtering correlates with erasure depth.
- **Every target number is a DEV measurement, and no FINAL number is currently usable** — the FINAL
  records exist, but were scored from `LAB_KEY` rather than `LABELS` (`D-022`). **The evaluator used
  inconsistent label text. The effect on scores and conclusions is unresolved pending corrected
  evaluation** — for `FINDING65` and `FINDING66` alike. The FINAL set has also been **consumed** by
  E6, so a fresh final split must be frozen before any new configuration is selected.
- **The count block in §2 is stale**; `outputs/registry_v1.json` predates 150 JSON files.
- **FID is still absent**, so no claim about global image quality.
- **Thirty duplicate `(arm, seed, lr, steps)` cells disagree with each other by up to 8/15 images,
  and no record carries a code-version field** (`D-025`), so the disagreements cannot be attributed
  to any known change. **This is unresolved, not closed.** Any aggregate that pools duplicate cells
  inherits it.
- **Five `gen_advunlearn_*` records are unattributable** — `gen_advunlearn_{official,parachute,tench,`
  `vangogh,garbage_truck}.json` carry both `unet: null` and `target: null` (`D-011`), so the
  checkpoint they scored and the concept they targeted are recoverable only from the filename. The
  numbers are retained as historical results; **their attribution is not available and has not been
  recovered.**
- **The audit concluding is not itself evidence that the surviving results are sound.** Each
  statement above stands or falls on its own named artifact. No result in this report gains standing
  from the fact that a repair cycle ran.
- **Seven historical `gen_*.json` records do not identify the checkpoint they scored** (`unet: null`),
  five of which demonstrably scored an *edited* model — see §5's provenance note.

---

## 9. Change log — 2026-10-26 (Agent 6, provenance audit)

Every claim in §2–§8 was traced to a raw artifact under `outputs/`, a log under `logs/`, or an
upstream source file. Full traceability table in `AGENT6_PROVENANCE_FINDINGS.md` §1.2. Banners were
judged insufficient: the incorrect sentences and table cells were rewritten, not annotated. No
historical number was deleted.

| § | what changed | why |
|---|---|---|
| §1 | "The corrected implementations exist and are unit-tested; **none has been run**" → all three have run, and two upstream references have been executed | false. E1 `logs/e1_wdr.log`, E2 `logs/e2g_grid.log`, E4 `logs/e4mg_grid.log`, R1 `logs/r1b_imma_train.log` (5000/5000), R4b–d `outputs/r4{b,c,d}_uce_*.json` |
| §2 | registry counts split into **registry vs actual**, with a staleness banner | `registry_v1.json` was generated 05:11:25; 150 JSON files have landed since, incl. 112 `atk_*` and ≥11 new builds. "1,278 JSONs / 696 attacks" are now 1,424 / 808 on disk |
| §2 | the "~54 new records, no new defence builds" parenthetical withdrawn | it was itself stale when written — `FINDING71`–`78` post-date the registry, and E2/E4/R1 added builds |
| §2 | the nine vacuous records, 136 VOID, 687/324 admissible, 335 builds | **re-verified exact** against the registry, field by field. Unchanged |
| §3.1 | "**all six reaching H = 1.000**" → five of six; the sixth is `…sd0_lr1e-4` at H 0.9333 with paired utility collapsing 0.8333 → **0.1875** | self-inconsistent: the quoted T range's 0.958 lower bound *was* the record excluded from the H claim. The destructive attack's lower H is not robustness evidence (`FINDING70`, `FINDING71`) |
| §3.1 | paired utility table, retain8/all-non-target figures, 40/48 → 47/48 | **re-verified exact** against registry `pau_before`/`pau_after`. Unchanged |
| §3.2 | UCE SD-1.5 λ=1000 "2/2, **third running**" → 6/6 | the third completed; nothing is running |
| §3.2 | UTRAP SD-1.5 3/3 → 4/4; Meta variant A "—" → 2/2, max H 1.000 | registry record counts; the dash understated a measured result |
| §3.2, §3.3 | all nine cross-model rows and all thirteen λ-curve values | **re-verified exact**. Unchanged |
| §5 | provenance note added: 7 `gen_*.json` with `unet: null`, 5 of them edited models; `ABSENT` quantified at 234/335 | the ambiguity of `unet: null` was undocumented, and a reader could not tell an unattributable record from a base control |
| §6 | mandate sentence **re-evidenced**, not weakened; the `meta2_church_lt0.5` near-miss stated with its three attacker-lr cells | the conclusion was right but had been asserted ahead of the evidence. At 25 steps the arm clears 0.40 at lr 1e-5 and 5e-5 and fails only at lr 1e-4 (0.5333, `logs/e2h_stronger.log`) |
| §7 | "unblocked and cheap, not yet run" replaced by a per-experiment state table | E1, E2, E4 ran; E6 and E7 partially ran; `FREEZE_ACTIVE` added as a blocker |
| §8 | heading FINDING65–70 → **65–78**; new §8.2 covers the eight omitted findings | the section claimed the report was "not stale against its own repository" while eight findings sat on disk, four of which amend its own rows |
| §8 | row 70's causal clause for the non-monotone H withdrawn | `FINDING71` replicated the damage (n=229) but **not** the non-monotonicity, which is arm-specific |
| §8 | **row 67 rewritten** (`D-017`): the "~4,000×" withdrawn (measured ≈**9.6×**), the 1/√N mechanism withdrawn as the *cause*, the real mechanism named as our **shared-Adam two-step schedule**, "**third instance** of one scale pathology" dropped, and "at gradient parity" withdrawn as a reading of retain8 0.5917 | the derivation stands but is not the cause; the reference **sums** both terms into one step, so this is our port's defect (entangled with `D-016`), not a property of the published method. Retained: 0.5917, K1 FAIL, α ∈ (1,4000) unmeasured — now with its upper bound labelled a **saturation point** |
| §8b | the **three-instance** scale-pathology generalisation withdrawn, with the sharper lesson stated | the lesson ("measure the scale before sweeping the knob") survives and was genuinely applied; the pattern it generalised from was one instance short. Diagnosing a shared knob is not diagnosing a shared cause |
| §8 | **rows 65 and 66 rewritten** under `D-022`/`D-022a` | `gen_eval_final.py` scored every FINAL number with `LAB_KEY` instead of `LABELS`. **Nine of twelve label strings differ; `church`, `parachute` and `tench` are identical.** 37 of 180 images turned on one of the nine, at median margins 0.0027–0.0063. **The evaluator used inconsistent label text; the effect on scores and conclusions is unresolved pending corrected evaluation** — for both findings |
| §5, §7, §8, §8a, §8c, §9 | **second pass, 2026-10-26 (documentation reconciliation):** every directional reading of `D-022` removed — "degraded distractors", "`target_hits` biased upward", "confounded in the direction of its own conclusion", "`FINDING66`'s ceiling conclusion survives by construction", "37 images cannot explain the gap", and "the base control confirms the scorer is sound" | the label-text mismatch is **established**; its **direction and size are not**, pending paired re-scoring. Each of those sentences inferred a sign or a magnitude from the substitution itself. The mismatch count **nine** is retained throughout and is the one established quantity |
| §8a | the "~3× weaker on held-out prompts" conclusion **withdrawn pending corrected evaluation** | it rested entirely on `FINDING65`'s ratios, whose two sides were scored from different label text (DEV from `LABELS`, FINAL from `LAB_KEY`) |
| §7 | E6 relabelled "RAN BUT MUST BE RE-SCORED" | the records exist and consumed the FINAL split, but their numbers are not usable as they stand. Re-scoring is CPU-only on retained artifacts |
| §5 | new caveat documenting the two label arrays and which numbers are implicated | a reader had no way to tell that DEV numbers are unaffected while FINAL ones are not |
| §8c | "Every target number except FINDING65/66's is a DEV measurement" → no FINAL number is currently usable | the sentence treated the FINAL figures as the one trustworthy class; their label text was inconsistent and their standing is unresolved |
| §8c | "Meta-v2 and UTRAP-v2 remain unrun, released references unexecuted" → all corrected, with `FINDING72`'s gate bias and the FINAL-set consumption added | three false sub-claims in one sentence |

**Not changed, deliberately:** every historical number, including the five unattributable
`advunlearn_*` records; the `FINDING76`/`FINDING77` correction banner, independently re-verified
against `r4b_uce_lamb*.json` column B and found accurate; §3's structure and §4's r4-correction
table, both of which verified clean.

**Known-stale and NOT repaired here:** `outputs/registry_v1.json` and therefore
`TABLES_v1_generated.md`. Regeneration requires `scripts/registry_v1.py`, which sha256s many GB; the
project is frozen, so this is flagged for the lead rather than run.

**Single-sourced and unverified:** §3.3's `cᵀc` max eigenvalue **has since been measured at 3.24e5**
(`D-033`), superseding the 85,850 figure that traced only to `ERRATUM_uce_lambda_scale.md`; §3.3 is
updated. §4's GIFT
β range 0.2000–0.4667 could not be reproduced from the two cross-model `b1.0` builds alone. Both are
CPU-cheap to settle and are listed in `AGENT6_PROVENANCE_FINDINGS.md` §6.
