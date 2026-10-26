# Current status, one verdict per issue — v2

**Written:** 2026-10-26 · **Audited snapshot:** `644dc2716b54630b2d6b6e941eda4af14eb37d55`
**Supersedes, for status purposes only:** the narrative in `AUDIT_REPAIR_REGISTER_v1.md` §1 and the
six `AGENT*_FINDINGS.md` records.

---

## 0. How to read this document

`AUDIT_REPAIR_REGISTER_v1.md` is **deliberately chronological**. It records hypotheses in the order
they were formed, including several that were later refuted — some of them the lead's own. That is
the right shape for an audit trail and the wrong shape for answering "what is the status of X".
Reading the narrative top to bottom will hand you refuted explanations as if they were live.

**This document is the flat current-state view.** For every issue it gives exactly **one** verdict,
drawn from a fixed taxonomy, and keeps the chronology in a separate column. Where the two disagree,
**this document's verdict is the current one and the register's narrative is the history**.

It settles status only. It does not re-adjudicate evidence, alter any historical protocol, result or
interpretation, or decide anything about contribution or publication.

### The taxonomy

| verdict | means |
|---|---|
| **Confirmed defect** | established by evidence; not yet repaired |
| **Repair implemented** | a code or document change has been made; see the verification row for whether it is proven |
| **Verification passed** | a measurement or a passing test establishes the condition |
| **Verification pending** | a repair or claim exists whose verifying measurement has not been run |
| **Historical result invalidated, or interpretation restricted** | the number stands or falls as recorded; what may be *concluded* from it has narrowed |
| **Unresolved hypothesis** | a candidate explanation that is neither established nor refuted |
| **Required rerun** | a specific run must happen before the item can close |

An item may carry a second verdict only where the register itself splits it (e.g. a repair made plus
a rerun still required). Where that happens the **primary** verdict is given first and in bold.

### ⚠ Defect-ID namespace collisions — read before citing any `D-NNN`

**Four independent numbering schemes reuse the same identifiers for unrelated defects.** Citing a
bare `D-NNN` is ambiguous and has already caused confusion between records:

| ID | the lead's register means | Agent 1 means | Agent 4 means | Agent 5 means |
|---|---|---|---|---|
| **D-008** | FINDING78's "first arm" retraction | — | — | `venv_uce` is a byte-identical clone |
| **D-010** | upstream UCE may be editing nothing | upstream wrote an unedited model | — | 17 kill-rule gates can report a crash as a measurement |
| **D-013** | the unenforced `FREEZE_ACTIVE` marker | — | `gen_eval_final.py`'s label set | the unenforced `FREEZE_ACTIVE` marker |
| **D-014** | Meta-v2's shared ζ | — | `gen_eval_final.py` precommitment fields | zero atomic writes anywhere |
| **D-015** | `attack_core.py`'s false equivalence | — | registry `defence_identity` | R4c/R4d mis-keyed port checkpoint |
| **D-016** | WDR-v2's inverted schedule claim | — | registry `vacuity_determinable` | `run_h66_frp.sh`'s attack key |
| **D-017** | FINDING67's 4,000× mechanism | — | registry `integrity` block | `.labels` lock with no release |
| **D-022** | the FINAL label-text defect | — | — | the truncated `unet.pt` |
| **D-026** | R4c/R4d cache on an incomplete key | — | — | source files modified after the freeze |

**Convention used below:** every identifier is written **namespaced** — `REG/D-022` for the lead's
register, `A1/D-100`, `A4/A-07`, `A5/D-014`, `A6/Q18` for the agents. Bare `D-NNN` in any other
document should be resolved against the owning file before it is relied on.

---

## 1. Current verdict per register defect (REG/D-001 … REG/D-034)

| id | issue, one line | **CURRENT VERDICT** | chronological audit note (history, not status) |
|---|---|---|---|
| **REG/D-001** | GPU reservation was check-then-select, not atomic; three runners held GPU 0 while GPU 1 sat idle | **Repair implemented · Verification passed** — `scripts/gpu_reserve.sh` with `flock`, PID+`starttime` ownership, trap release, GPUs 2–3 unreachable by construction; `test_gpu_reserve.sh` **50 passed / 0 failed**, CPU-only | Observed live at freeze (15.3 GB on one card, GPU 1 at 34 MiB). The register still carries "Status: OPEN" from before Agent 5's repair landed. A **residual limitation stands**: a legacy campaign holding no reservation and sitting between cells is undetectable, and **8 callers have not been migrated** — migration must pass `$$`, not `$BASHPID` |
| **REG/D-002** | `run_r2_meta_official.sh` never invoked the upstream trainer | **Repair implemented · Required rerun** — the runner now stages pools and invokes upstream, but **has not been executed**; it refuses to start under `FREEZE_ACTIVE` without an explicit lead override | Gate condition G5. The repair retargets **`src/esd_meta.py`**, not `src/meta.py`, for the reason in REG/D-020 |
| **REG/D-003** | FINDING77's ratios are wrong | **Repair implemented** — FINDING77 rewritten; see REG/D-032 for the recomputed table | The register's "~60,000×" is **59.90×**; the "60× worse" is **0.58×**, i.e. *smaller* |
| **REG/D-004** | "A−B isolates embedding selection" is not a valid inference | **Interpretation restricted** — valid **only** as a triangle-inequality bound given a common reference and denominator; **never** as a subtraction attributing a difference | Adjudicated in REG/D-032. At λ=1000 the bound is tight ([0.108406, 0.108406]); at λ=0.1 it is [0.38265, 0.38567], so selection accounts for the whole A column to within ±0.4%. Agent 1 additionally notes that R4b's "port" column is **not** `erase_uce.py` (three differences, of which the document names one) and that **no R4 cell reproduces any historical arm's configuration** |
| **REG/D-005** | FINDING76 overstates both the conclusion and the mechanism check | **Repair implemented** — FINDING76 rewritten against all ten requirements | Sub-item (c) was itself a correction that needed correcting: K=1's range **[−0.00457, −0.00011]** and K=32's point **−0.00638** are **disjoint**, not overlapping. Sub-item (h): K=0 does **not** dominate — it is a well-resolved two-axis trade-off (target 0.0000 vs 0.3333 = 5 images; retain8 0.8667 vs 0.9667 = 12 of 120 images) |
| **REG/D-006** | R4c advertises a flag it lacks, scores with `list(ge.CONCEPTS)`, ignores exit codes, saves no images | **Repair implemented · Required rerun** — ten repairs applied to `r4c_uce_checkpoints_and_images.py`; **the existing `r4c`/`r4d` artifacts still carry the wrong label set and must be re-run** | Same defect class as REG/D-022: two independent scripts substituted `CONCEPTS` for `LABELS`, so it is **systemic**. Consequence: **`r4c`/`r4d` target accuracies are not comparable to any other H number in this project** |
| **REG/D-007** | E2h's declared grid and its actual loop differ | **Confirmed defect · Repair implemented** (in-file by its owner) — the 50-step cell must never be described as preregistered | The header's own cell count is also wrong (enumerates **nine**, loop runs **twelve**). Sharper fact than the register records: among the **nine declared** cells only 2/9 cleared the gate, max-over-valid **0.5333**; the **undeclared** 5e-5/50 cell was 3/3 valid at 0.8000/0.9333/0.5333 — the undeclared escalation is where the falsifying result came from |
| **REG/D-008** | FINDING78's "first arm to meet both mandate halves" claim | **Historical result invalidated** — the claim is **RETRACTED**. Its §1 retention result (+0.20 to +0.27 retain8 at identical 0.0000 suppression; 0.9000 against 0.6250/0.7000) is **artifact-exact and survives** | At 25 steps the arm clears 0.40 at attacker lr 1e-5 (0.0667) and 5e-5 (0.2000) and fails at 1e-4 (**0.5333**). Max-over-valid is a ratchet, so 0.5333 stands. **No arm in this project meets both halves at any budget** |
| **REG/D-009** | the IMMA reference evaluation is internally contradictory | **Historical result invalidated, interpretation restricted** — the record stands as a measurement; it is **not citable as an IMMA result**, and specifically not as "IMMA destroys utility" | **The "checkpoint composition unproven" explanation is REFUTED — see §2.1.** Quarantine stands on *different* grounds: the closed-set metric is **inapt** for this record and **the configuration is not one the authors report** |
| **REG/D-010** | pinned upstream UCE may be editing nothing | **Unresolved hypothesis → REFUTED.** Upstream's edit is real — see §2.2. The surviving open item is narrower: whether upstream should have been given an **expanded erase-prompt set** | The lead's own hypothesis, pushed hard and then killed by measurement. The register preserves the full arc; **the inertness reading is dead and must not be quoted as live** |
| **REG/D-011** | the evaluated checkpoint is unrecoverable from older `gen_*.json` records | **Confirmed defect · Historical, attribution unavailable** — the numbers are retained; **their attribution has not been recovered**. See §5.2 | **Five**, not four, unattributable `advunlearn_*` records. `gen_E6cal_BASE_sd14_on_DEV.json` does record its target and is **not** in this class |
| **REG/D-012** | E2h reported success after an abort and pooled two campaigns | **Repair implemented · Verification passed** — per-cell exit codes checked, aborts write `E2H INCOMPLETE` and exit 1, summary tags each row with its campaign and declared/post-hoc status; re-run on existing artifacts, output in register §3 | Gate condition G2 passed. The damage was **smaller** than first recorded: of 138 runners echoing `DONE`, **104** never inspect an exit code, but **no** Python parses the string, **no** runner gates on it and **no** finding cites it, so a false `DONE` never mechanically drove a decision |
| **REG/D-013** | the `FREEZE_ACTIVE` marker was read by nothing | **Repair implemented · Verification passed** — the check now sits **inside** `gpu_reserve.sh`'s critical section, refuses with a distinct rc=4, prints the marker to stderr, and requires **both** `ALLOW_FROZEN_RUN=1` and `--allow-frozen`; every override is logged. **Verified against the live freeze: refused, rc=4, marker printed** | The lead's own defect, self-reported. Grepping every `*.sh`/`*.py` outside `upstream/` for `FREEZE_ACTIVE` returned no match: the freeze was real, its enforcement was not |
| **REG/D-014** | Meta-v2's shared ζ is near-inert and its paired retain difference is unpaired | **Confirmed defect · Interpretation restricted** — ζ is inert **as a coefficient** and **has never been swept**; the logged `retain_change` field is dominated by sampling noise. **FINDING78's measured retention result stands** and is attributable to **restoring the pairing**, not to ζ's weight | Agent 3 first concluded the gain "cannot be attributed to the restored `+ζL_R(θ)` term", the lead rejected that, and **Agent 3 withdrew it**: the near-zero net *is* the mechanism — pairing converts variant A's retain-**level** penalty into a **change** penalty. The behaviour fix is **common random numbers**, which changes numerics and so may not be applied to any script that produced a historical arm without a new label |
| **REG/D-015** | `attack_core.py` claimed an equivalence that does not exist | **Confirmed defect · No historical result affected** — drift risk only. `EVAL_FULL_UNET` is a hand-transcribed mirror and its self-test tests the mirror against itself; the mirrored values **currently match on all 8 axes**, so every divergence record already written is correct | Explicitly **not** a data-integrity problem and **no data is quarantined** on its account. Two axes are structurally uncoverable by the divergence report: RNG provenance and the attacker's data |
| **REG/D-016** | WDR-v2's schedule claim is inverted | **Confirmed defect · Repair implemented** (documentation) · **Required rerun** for the behaviour | The reference **sums**; alternation there is gated on an `alternating_` name prefix `weight_dist_reg` does not carry. The behaviour change must land under a new arm label, not in place — `erase_wdr_v2.py` produced FINDING67's E1/E1b arms |
| **REG/D-017** | FINDING67's 4,000× parity figure has the wrong mechanism | **Repair implemented · Interpretation restricted** — the derivation ‖∇L‖ = α/√N **stands**; **4,000× is withdrawn, measured ≈9.6×**; the mechanism is our **shared Adam second-moment buffer**, not the published method. Gradient-norm parity is **not** optimality. §7's "Meta-v2 needs no coefficient correction" is **withdrawn** (REG/D-014 supersedes) | Measured step size as a fraction of `lr`: α=1 → 0.104, α=10 → 0.537, α=4000 → 0.998. **Retained:** retain8 0.5917, the K1 FAIL, and α ∈ (1, 4000) unmeasured — with its upper bound relabelled a **saturation point, not a frontier**. The "third instance of one scale pathology" framing is withdrawn for the WDR member |
| **REG/D-018** | two provenance misstatements in method scripts | **Repair implemented** — `erase_imma.py`'s `code_released: False` corrected to `True`; `erase_adv.py` re-labelled **our own design** with `IMPL_VERSION` added | Pre-repair `erase_adv.py` arms are **not** adversarially-trained results (the inner loop descended the erasure loss) and are invalidated as such. `erase_imma.py` still omits the reference's outer gradient clipping — that is a **new arm**, not an edit |
| **REG/D-019** | the third-party "ESDx1" checkpoint is not a cross-attention-only erasure | **Unresolved hypothesis · Required rerun** — **all 686** tensors differ from pinned base SD-1.4, resnet blocks reaching 0.20 relative, so the name is wrong. **Whether REG/D-009's utility collapse is inherited from ESD is UNTESTED.** The decisive control (ESD checkpoint scored **alone**, 12 DEV concepts, same scorer and seeds, ≈5 GPU-min) **has not been run** | Deliberately not launched by hand — launching by hand is the failure that produced REG/D-001. Queued as the first job through the reservation path. Two live hypotheses for the checkpoint's identity (full-parameter run despite the `x`, or a different SD-1.4 snapshot); the discriminating test needs network access. **Until the control exists, no statement about where the utility loss comes from is supportable** |
| **REG/D-020** | upstream `src/meta.py` does not run at the pinned commit | **Confirmed defect (upstream)** — three independent fatal faults: `train_step` returns a 4-tuple while **11 call sites unpack 3**; `train_loss` is never assigned; `param.grad` is `None`. The runnable path is `src/esd_meta.py`, whose correct initial checkpoint is **plain SD-1.4** | A finding about the release, **not** a licence to substitute our own code and call it official. Argument hazards verified: `--gamma2_1` passed three times resolves to `0.01, 1.0, 1.0`; `--fix_timesteps False` evaluates to **True** under `type=bool` (only omitting the flag yields False); `--fixed_time_steps 1 2 5 10` hard-errors. **Both upstream example scripts are unrunnable as written** |
| **REG/D-021** | IMMA training ran 5× the documented ascent budget, on partly blank data | **Confirmed defect · Interpretation restricted** — 5000 steps against the README's 1000, i.e. **2500 ascent steps where 500 are specified**; **3 of 100** training images (`7.png`, `29.png`, `41.png`) are NSFW-blanked black frames and were trained on. Every artifact under `upstream_runs/imma/REF_imma_vangogh/` was produced at 5× | Also: the trained scope includes **80 self-attention** tensors, so "xatten" is a misnomer in the artifact name and the variable name, though the manifest is correct. LoRA was warm-started, not reinitialised — our docs were already right |
| **REG/D-022** | every FINAL number was scored against the wrong label set | **Confirmed defect · Historical results' interpretation unresolved.** `gen_eval.py` defines `LABELS` (`:48`) and `LAB_KEY` (`:50`); **nine of the twelve label strings differ** and `church`, `parachute`, `tench` are identical. **The evaluator used inconsistent label text. The effect on scores and conclusions is unresolved pending corrected evaluation.** See §3 | Gate condition G12, promoted from hygiene to finding-critical. The scorer is repaired; the 18 records predate the repair. **Re-scoring requires regenerating the 180 images** (not saved) — deterministic, so it is a **pure re-score of the same samples, not a second use of the FINAL split**. REG/D-022a, which argued a direction, is **withdrawn** — see §3 |
| **REG/D-023** | FINDING73 compares a tench retain8 that includes tench against church arms that exclude theirs | **Confirmed defect (in the document, not the code)** — `legacy_retain8` is defective by design for tench targets but correctly labelled and warned in `gen_eval.py`; the live error is FINDING73 quoting **0.7917** for a tench arm against church arms whose retain8 correctly excludes their target | Bias is **~0.095 against the tench arm** — "the more successful the erasure, the more it is scored as a utility failure". Correct figures: 0.8909 (n=11) / 0.8857 (n=7). `utility_all_nontarget` (n=11) and the 48-image panel are **correct and runtime-asserted**. Exposure is **2 of 346** `gen_*.json` |
| **REG/D-024** | the validity gate's stated threshold is finer than its own resolution | **Confirmed defect** — the gate is `0.20` in all **808** records but T is a **24-image** panel, so the operative boundary is **≥ 5/24 = 0.2083**. **13 records failed by exactly one image**, three of them load-bearing | FINDING74's claim **survives** — the exclusion was conservative, both excluded H values below the retained 0.1333 — **but it rests on a one-image gate call**. FINDING69's "the gate did its job" rests on one such record. The cheapest honest fix changes no number: restate the gate as `≥ 5/24` in the help text. **No document states the T denominator** |
| **REG/D-025** | 30 duplicate cells disagree, and no record says which code produced them | **Confirmed defect · UNRESOLVED.** See §5.1. Repaired **additively** by Agent 4 (`schema_stratum`, `integrity` block); no historical record rewritten and **no disagreement adjudicated** | Over all 808 records: **74 duplicated cells across 224 files, 33 disagreeing**. Within the 696 indexed: **68 duplicate cells over 178 rows, 30 disagreeing**. The registry's `attack_records_quarantined: 0` is **vacuously zero** — its only glob matches nothing |
| **REG/D-026** | R4c/R4d cache the port checkpoint on an incomplete key; R4d is confounded | **Confirmed defect · Repair implemented · REQUIRED RERUN** — R4d must be re-run with the port directory keyed on every parameter that changes the checkpoint, behind a fingerprint-verified guard | The port column being identical across R4c and R4d is **tautological, not a measurement**, so the port side gives **no** evidence about the preserve pool; r4d's recorded `"n_preserve": 0` is **false for the port arm**. Upstream's side *was* recomputed in-process each run, so the preserve-pool statement holds **for upstream only** — and is independently confirmed by direct measurement (removing the pool moves upstream's edit 0.0568 → 0.0536, **1.5% of the 0.22 gap**) |
| **REG/D-027** | error suppression can manufacture the number that trips the kill rule | **Confirmed defect (hazard) · Verification passed (corpus clear)** — **17** kill-rule gates and **33** `\|\| echo 0` sites can make an I/O error indistinguishable from a measured collapse. **The risk did not materialise:** every `destroys the model` / `below K1` / `retain8 = 0.0000` claim in `docs/` was re-derived from its artifact JSON and all trace to genuine records with the full twelve `per_concept` entries at n=15 | Gate condition G9, now **forward protection**. The truncated checkpoint is **not** behind any of them: FINDING63's measured arm is `l1.0k8`, the truncated file is `l1.0k4`, which has **no `gen_` record at all**. **No finding is retracted on its account.** Note the awk semantics that make this dangerous: `r=""` and `r="0"` both read as below K1, while `r="FAIL"` **clears** K1 |
| **REG/D-028** | no atomic writes anywhere; one checkpoint is silently truncated | **Confirmed defect · Historical impact: none measured** — **zero** uses of `os.replace` across 48 Python files. `adapters/utrap_church_l1.0k4/unet.pt` is **511,229,952 bytes against a sibling's 1,719,229,888** (29.7%), raises `BadZipFile`, and **passes every `[ -f ]` guard**. **0 grep hits** in any metric path, so no published result used it | Gate condition G10. The highest-leverage outstanding fix: it converts ~285 existence guards from "bytes exist" to "the write completed". Ordering is checkpoint-first in 13 files, so a kill between save and dump leaves weights without provenance |
| **REG/D-029** | `pick_gpus.sh` would hand out another user's GPUs | **Confirmed defect · Repair implemented** (superseded by `gpu_reserve.sh`, which makes GPUs 2–3 unreachable by construction and is proven by test) · **Owner action: retire or bound `pick_gpus.sh` itself** | Gate condition G11. It iterates all four GPUs and would return 2 or 3, which belong to **uid 1010**. The constraint was being honoured by `campaign_free.sh`'s narrower loop and by luck, not by this script |
| **REG/D-030** | "destroys the model" is over-stated: the scorer has no reject option | **Confirmed defect · Interpretation restricted.** `retain8 = 0.0000` licenses **"closed-set retain accuracy collapsed to N/120"**, not "the model is destroyed". It does **not** overturn any K1/K2 gating decision — those are defined on retain8 and were applied consistently. See §4 for the reject-option status | In every collapsed record the surviving accuracy concentrates on one or two labels, repeatedly `van_gogh`; `clip_align` is depressed but **not** collapsed (0.1213–0.1822 for `utrap_church_l1.0k8`). The register calls a reject-option re-score the "decisive test" — **§4 requalifies that: a reject option is a diagnostic, not a repair** |
| **REG/D-031** | R4d overwrote R4c's upstream checkpoint | **Confirmed defect · Evidence destroyed · Repair implemented (prospective) · Required rerun** — R4c's exact inputs **can no longer be recovered from disk**; its JSON survives, its checkpoints do not. Fixed going forward with fingerprint-verified caching and hard refusals | The one defect in this audit that destroyed evidence rather than mislabelling it. Nothing was deleted deliberately; the naming scheme did it silently. Confirmed by mtime and by re-measurement: the surviving file measures 0.0536 (0-preserve), not 0.0568 (10-preserve) |
| **REG/D-032** | six of FINDING77's ratios are wrong; one correction *reinstates* part of FINDING75 | **Repair implemented** — FINDING75 and FINDING77 rewritten | Recomputed: "~60,000×" → **59.90×**; "60× worse" → **0.58×**, i.e. 1.73× *smaller* and a cross-λ comparison; "~6 orders" → **2.2 orders** matched and **0.4 orders (2.54×)** against our own selection; "3.7e-07" is λ=1000 only (λ=0.1 is **1.05e-03**); "11–38%" → 10.8–38.4%; "13 of 22 arms" → **9 of 22** (13 counted surviving adapter directories). **The 0.4-order figure partly reinstates FINDING75 §2** — the honest direction is not uniformly downward. Accumulation order is now **measured**: **< 1.2e-04 on GPU** but **5.6e-02 on CPU**, i.e. **BLAS-dependent**, and "λ first vs λ last" is only **one of four** differences |
| **REG/D-033** | `ERRATUM_uce_lambda_scale`'s eigenvalue is 3.8× off and was never measured | **Repair implemented** — the measured value is **3.2420e+05** for our port's 826-row system (and 2.2171e+03 for upstream's 11-row system), against the document's **85,850**. Propagated to `T2I_REPORT_r5.md` §3.3 and `UPSTREAM_FIDELITY_MANIFEST.md` §4 | The erratum's qualitative point (λ ≤ 1e3 is inert against `cᵀc`'s spectrum) **survives and is strengthened**, but the number must be attributed to a measurement. Both systems remain massively rank-deficient, so λ regularises null directions in both cases rather than conditioning a full-rank system |
| **REG/D-034** | our port's erasure is not shown to be clean, and no retain axis was measured | **Confirmed defect · Interpretation restricted** — our port's `last..76` selection is a deliberate and **partly vindicated** deviation (it is why the port erases at all), **but** its 0/15 **scatters across five different classes at 4× the edit size** and **no retain axis was measured in R4c or R4d at all**. "Our port erases and upstream does not" does **not** establish that our port erases *well*; it is equally consistent with broad damage | REG/D-030 in a second place: suppression measured by a closed-set argmax with no reject option cannot distinguish erasure from degradation. **The R4 family needs a utility panel before any superiority claim.** Folded into G13 |

---

## 2. Two refuted explanations — do not quote either as live

These were live working hypotheses. Both were **refuted by measurement**. They appear in the
register's narrative because the register is chronological; they are **not** current.

### 2.1 "IMMA composition unproven" — REFUTED. The composition is valid.

| what was measured | result |
|---|---|
| `r1_eval_imma.py`'s matching loop, replicated offline | **exact=160, fuzzy=0, reshaped=0, skipped=0** — the fuzzy/reshape path never fires |
| IMMA-scope tensors in `adapters/REF_imma_vangogh/unet.pt` | **160 / 160** byte-equal to `half(IMMA)` |
| all other tensors | **526 / 526** byte-equal to `half(ESD)` |
| unexplained / non-finite tensors | **0 / 0** |
| tensors differing from ESD | exactly **160** — 80 `attn1` + 80 `attn2`, IMMA's own trained scope |
| ESD delta | **not inverted** |
| composition semantics | byte-semantically **identical to upstream's own** (`eval/text2image.py:65-69`, `train_text_to_image_lora.py:458-462`) |

The lead's fuzzy-key/reshape hypothesis is **dead**. Agent 2's position is explicit: *"no key fell
back to fuzzy matching, nothing was reshaped, nothing was skipped, and every IMMA tensor landed on
the module it was trained on."*

**`outputs/gen_REF_imma_vangogh.json` remains quarantined for a different reason.** The closed-set
12-way CLIP argmax is **inapt** for this record — eight non-targets at *exactly* 0.000 with the
target at 0.867 is a single-label collapse of the metric, not a spread around the 1/12 ≈ 0.083 chance
line — and **the configuration is not one the authors report**: IMMA never evaluates an immunised
model standalone (`upstream/IMMA/README.md:146-151`). It must **not** be cited as an IMMA result, and
specifically not as "IMMA destroys utility". **Because the measure and the configuration are inapt,
not because the checkpoint is malformed.**

The live candidate explanation is now REG/D-019's **ESD checkpoint**, and **its control run is
outstanding**.

### 2.2 "upstream UCE may have written an unedited model" — REFUTED.

| checkpoint | vs BASE max | vs BASE mean | bitwise identical to base |
|---|---:|---:|---:|
| `upstream_church_lamb0.1` | 9.7948e-02 | **5.3620e-02** | **0 / 32** |
| `upstream_church_lamb0.5` | 9.7899e-02 | **5.3593e-02** | **0 / 32** |

Upstream changed **every one of the 32 `attn2` K/V matrices**, by **5.36 % of weight norm on average
and 9.79 % at maximum**, with **0 of 32** identical to base. Cross-attention K/V **output** for
`"a photo of a church"` moves **43.4 %** (against 58.7 % for our port). `UCE()` writes into a
`copy.deepcopy` and saves those tensors, so the `.safetensors` captures the edited weights.
**No "upstream" number in R4b/R4c/R4d is measuring the base model.**

Two consequences that are themselves load-bearing:

- A **1.35×** gap in activation space separates 15/15 from 0/15 in CLIP label. **Behaviour is not
  proportional to the numerical gap**, which independently limits every weight-norm-based
  behavioural inference in FINDING75 and FINDING77.
- Upstream's single-token edit does **not** leave trailing positions untouched — it moves
  padding-position K/V output by 42 %. That refutes the mechanism stated in `erase_uce.py:50-57`,
  even though the **observation** that file reports is independently reproduced.

**What is still open, stated narrowly.** `erase_scale` 1→10 is inert to four digits (0.0536 →
0.0536), and neither `lamb` nor the guide string helps (`guide=''` ≡ `' '` — the identical edit). The
one effective lever is **`expand_prompts=true`** (6 erase prompts), raising dW/W to **0.0925**, above
our port's 0.0587 — and it is **unreachable through `UCE()`** as invoked. So the live invocation
question is whether upstream should have had an **expanded erase-prompt set**, not whether it wrote
anything. Separately the **port side is confounded** (REG/D-026) and **R4d is a required rerun**,
with **R4c no longer reproducible** (REG/D-031).

---

## 3. The FINAL label-text mismatch — what is established and what is not

### Established

`gen_eval.py` defines two parallel twelve-element lists:

- `LABELS` (`:48`) — the CLIP text: `"garbage truck"`, `"van gogh painting"`,
  `"english springer spaniel"`, `"monet painting"`, `"cassette player"`, `"chain saw"`,
  `"french horn"`, `"gas pump"`, `"golf ball"`, plus `church`, `parachute`, `tench`.
- `LAB_KEY` (`:50`) — the underscored record keys.

**Nine of the twelve label strings differ. `church`, `parachute` and `tench` are identical.**
`gen_eval_final.py:84` used `list(ge.CONCEPTS)`, whose keys equal `LAB_KEY`, so every FINAL number
was computed against label text that is not the text the rest of the corpus uses. The same
substitution occurs independently in `r4c` (REG/D-006), so it is **systemic**.

Also established, as counts of affected decisions rather than as effects: **37 of 180** FINAL images
were classified as one of the nine labels whose text differs, concentrated in the four pre-attack
defence arms, at per-arm median winning margins of **0.0027–0.0063**. Post-attack arms have **zero**
such classifications, at margins an order of magnitude larger.

### THE CURRENT INTERPRETATION, VERBATIM

> **The evaluator used inconsistent label text. The effect on scores and conclusions is unresolved
> pending corrected evaluation.**

That is the whole of it. **Verdict: Confirmed defect; historical results' interpretation unresolved;
required rerun (CPU-only re-score of regenerated images).**

### Claims withdrawn, and why — this corrects the lead's own overreach

| withdrawn claim | why it does not hold |
|---|---|
| changed distractor wording necessarily **weakens** every competitor | a different token sequence is a different text embedding, but nothing establishes that it is a *weaker* one. Whether a given similarity moves up or down, and by how much, is empirical and **unmeasured**. The records' own data cuts against a uniform reading: the nine differing labels nonetheless **won 37 of 180 images** |
| church counts were **definitely inflated** | no inflation magnitude exists in any record. The four pre-attack church arms in fact scored **1/15 to 9/15** on church, with differing labels winning. Direction unestablished, size never estimated by anyone |
| FINDING65's headline was **manufactured** by the scorer | no agent makes this claim, and it does not follow. The verdict is **unmeasured pending corrected evaluation** — neither confirmed nor refuted. Independently, FINDING65's ratios are **quantisation-fragile**: they are ratios of 1- and 2-image numerators, and one image moves the 3.5× to 2.33× or 7.0×, with the 1.1× deriving from a **0/15** baseline |
| the finding **definitely survives** and only its **magnitude** is unknown | both the size *and* the sign of the effect are unestablished, so survival is not established either. The earlier argument that "37 images at margins ~0.005 cannot explain a 0.1333 → 0.4667 gap" estimates the artifact's size **from its assumed direction**, and both are unestablished |
| FINDING66 **survives the defect by construction** because its comparison is within-scorer and 15/15 can only be lowered by strengthening distractors | consistency between two arms scored the same inconsistent way does not establish what either score would be under the correct text; and "strengthening" presupposes the direction. FINDING66's standing is **unresolved**, not upheld. §3's robustness results rest on the DEV evidence and are unaffected either way |
| predictions that landed on **unchanged** labels are **unaffected** | a closed-set argmax couples every label to every other. A change to any competitor's text can move any image's decision in either direction, including images predicted as `church`, `parachute` or `tench` |
| base-model accuracy of **1.000 proves the scorer is valid** | it establishes that the scorer behaves correctly on **undamaged base outputs only**. It says nothing about damaged or unusual outputs — which is exactly where the collapsed-retain records live (REG/D-030) — and nothing at all about the FINAL path, which both base controls were scored outside of |

### Count discipline

**Nine** is the established count and is now propagated consistently through
`T2I_REPORT_r5.md`, `UPSTREAM_STATUS_r1.md`, `UPSTREAM_FIDELITY_MANIFEST.md`, `HANDOFF_r5.md`,
`CORRECTION_REGISTER_r5.md` and this document. `gen_eval_final.py` was corrected from "4 of 12" to
nine in both its comment and its emitted `label_text_note`. **Propagating nine does not assert any
direction of embedding effect**, and nothing above should be read as doing so. `AGENT4_EVAL_FINDINGS.md`
still records "four of twelve" — see §7.

---

## 4. A reject option is a diagnostic, not a repair

REG/D-030 calls a reject-option re-score a "decisive test" and gate condition G13 lists "a reject
option or open-set margin" among its required evidence. **Neither makes a reject option a fix.**

**Verdict: a potential additional diagnostic. Not a repair, not implemented, not required reading of
any existing score.**

It would tell us something the closed-set argmax cannot — whether a collapsed arm's images are
off-distribution rather than confidently mis-labelled. It would **not** correct, replace or
rehabilitate any existing number. If it is introduced, all of the following apply:

1. **Calibrated independently of the arms being judged.** Fitting a threshold on the same collapsed
   arms whose interpretation is in question is circular.
2. **Versioned**, so a score is always attributable to a stated diagnostic revision.
3. **Documented as to its assumptions** — what distribution the threshold is fitted on, what
   "unusual" means operationally, and what it does to a known-good base model.
4. **Existing closed-set scores preserved.** Every K1/K2 gating decision in this project is defined
   on `retain8` under the closed-set scorer and was applied consistently. Those decisions stand and
   are not reopened by adding a second metric beside them.
5. **No threshold tuned to favour a result.** A cut-point chosen after seeing which arms it rescues
   is not a measurement.

**Until then, and independently of it:** claims of **"broad model destruction"** or of **"clean
erasure"** require **saved images and an INDEPENDENT quality/semantic assessment**. Neither exists
for any arm in this corpus — the entire evidence base for every such claim is a single scalar
`retain8` read out of a JSON. The supportable phrasing is **"retain accuracy collapsed to N/120
under the closed-set scorer"**. Symmetrically, 0/15 target hits cannot distinguish erasure from
degradation (REG/D-034), and R4c/R4d measured **no retain axis at all**.

---

## 5. Open items that the audit's conclusion does NOT close

**Nothing in this project is more defensible because a repair cycle ran over it.** The resume gate in
`AUDIT_REPAIR_REGISTER_v1.md` §5 is **NOT PASSED**. Each item below is open on its own terms.

### 5.1 The duplicate-cell discrepancies — UNRESOLVED

**Thirty duplicate `(arm, seed, lr, steps)` cells disagree with each other by up to 8/15 images, and
no record carries a code-version field.**

| scope | duplicated cells | files | disagreeing |
|---|---:|---:|---:|
| all 808 attack records | 74 | 224 | **33** |
| the 696 indexed in the registry | 68 | 178 | **30** |

- All 178 indexed rows are `admissible_for_recovery = true`, so each **double-counts** in every
  `tested` and `gate-valid` column of `TABLES_v1_generated.md`.
- **13 cases survive the strictest available key** — byte-identical `hparams` *and* identical
  `T_prompts`/`H_prompts` — and still disagree. **Maximum spread 0.5333 = 8 of 15 images.** `T_rise`
  moves with `H_after`, so **the attacks ran differently**.
- **There is no code-version field in any attack record**, and `run_inventory.json.code_commit` is
  `null` for all 234 entries, so **the disagreements cannot be adjudicated from the records at all.**
  `schema_stratum` was added as the only available proxy, with a note saying so. 11 of the 13
  straddle a schema boundary and the older stratum reports the higher `H_after` in **every**
  disagreeing pair, which `max(H_valid)` **systematically selects**. **Two same-stratum
  disagreements have no available explanation.**
- Compounding: 14 runner scripts pool campaigns and take the **first glob match** then `break`, so
  which record lands in a curve is **filesystem-order-dependent**.

**Verdict: Confirmed defect. Unresolved. Additive reporting repair only — no disagreement has been
adjudicated and no number changed.**

### 5.2 The missing-provenance records — attribution not recovered

**Five `gen_advunlearn_*` records are unattributable: `unet: null` AND `target: null`.**

| record | what it shows |
|---|---|
| `gen_advunlearn_official.json` | church 0.400, monet 0.933 |
| `gen_advunlearn_parachute.json` | parachute 0.133 |
| `gen_advunlearn_tench.json` | tench 0.400, monet 0.867 |
| `gen_advunlearn_vangogh.json` | van_gogh 0.467, monet 0.933 |
| `gen_advunlearn_garbage_truck.json` | garbage_truck 0.133 |

All five **demonstrably scored an edited model** whose identity was never written down, and the
active target is inferable only from the filename. Two further records have `unet: null` and *are*
genuinely the base (`gen_base_sd14.json`, `gen_E6cal_BASE_sd14_on_DEV.json` — the latter **does**
record its target and is **not** in this class), which is what makes `unet: null` ambiguous rather
than merely missing.

`TABLES_v1_generated.md` §2 prints all six as `—/—/—`, so no figure is attributed to them there. The
one exception in the whole corpus is §1's `h63_church_a0.3` / `h63_van_gogh_a0.3`, which carry **9
attacks each and max H 1.0000** against builds whose `target` is unrecorded **and** whose checkpoints
are `ABSENT`.

**Verdict: Confirmed defect. Historical results retained; attribution unavailable and not recovered.**
Newer records record both fields, so this is a closed *historical* defect and an open *attribution*
gap.

### 5.3 Other items that are open, not closed

| item | verdict |
|---|---|
| ESD-checkpoint-alone control (REG/D-019) | **Required rerun** — queued, not run; until it exists no statement about the source of REG/D-009's utility loss is supportable |
| FINAL re-score with `LABELS` (REG/D-022, G12) | **Required rerun** — CPU-only on regenerated images; a **fresh** FINAL split must also be frozen, the existing one being spent |
| R4d with a corrected `port_dir` key (REG/D-026) | **Required rerun** |
| R4c (REG/D-031) | **Required rerun** — its checkpoints no longer exist on disk |
| retain/utility panel for the R4 family (REG/D-034) | **Required rerun** — no superiority claim may rest on R4 without it |
| official Meta pipeline (REG/D-002, G5) | **Required rerun** — runner repaired, never executed |
| grid completeness against `PROTOCOL_FROZEN_v1.md`'s six-point rule | **Confirmed defect, unrepaired.** Of 86 `(defence, text_encoder, concept)` groups: **30 complete, 56 incomplete, 24 over-run**. ~25 arms keep only the three lr 1e-5 seeds and omit **the three strongest attackers**, covering **all four `utrap2*_church_l0.15k*` arms FINDING76 rests on**, plus `meta2_church_lt0.5`. Since max-over-valid is monotone non-decreasing in attacks run, this is a live **comparability** defect. Named campaign grids (E11/B–O, E12, E2G, E2H, E4MG) are **complete, 0 missing** |
| the FINAL set described as untouched in machine-asserted form | **Confirmed defect, partially unrepaired.** `split_registry.py:191` hard-codes `"status": "FROZEN_UNINSPECTED"` and **`test_audit_repairs.py:391` asserts it** — a passing test currently protects a false claim. The literal and the test must change **in one commit**; prose instances in the documents owned here are corrected |
| `split_registry.py`'s advertised hard-fail on unregistered defences | **Confirmed defect, unrepaired.** It computes `unknown` and only reports it. It records `["erase_utrap_v2", "erase_wdr_v2"]` as unregistered while returning `FINAL_usable: true` — and `erase_utrap_v2` built **all four FINDING76 arms**, so those arms' prompt pools are outside the leak audit entirely |
| sub-quantisation inference | **Confirmed defect, 16 sites, mostly unrepaired.** True denominators: H = 15 (1/15), T = 24 (1/24), `legacy_retain8` = **120 images** (1/120), `utility_all_nontarget` = **165** (1/165), paired panel = 48 (1/48). **"retain8 (n=8)" and "all-non-target (n=11)" count concepts, not images** — the real steps are 15× finer. No document states the image denominators |
| `venv_uce` (A5/D-008) | **Unresolved hypothesis** — byte-identical clone of `venv_t2i`; the diffusers upgrade the documents attribute to it **does not exist on disk**. Partially exculpated: diffusers 0.31 passes `attention_mask=None`, so the embeddings match anyway — but unverified for ≥0.34 |
| `venv_imma` documented at 0.31.0, actually 0.24.0 | **Confirmed defect (documentation).** The downgrade itself is clean and necessary; the docs disagree with disk |
| all-invalid attack cells reported without "unmeasurable ≠ robust" | **Confirmed defect.** **57 of 341 cells have zero valid records**; **10 of 112 arms** have no gate-valid record. The gate conflates *vacuous/ceiling* arms with *genuinely unmeasurable* ones whose `T_rise` is negative to **−0.2500** — and the second group's low `H_after` is **not** evidence of robustness |

---

## 6. Prior art — the FRAG overlap, recorded factually, with no contribution decision

**Verification status: VERIFIED.** arXiv **2608.25429** is **FRAG**, and it is **already the source
of this project's own FRP reference** — not a separate unverified preprint:

- <https://arxiv.org/abs/2608.25429>
- <https://arxiv.org/html/2608.25429v1>
- <https://github.com/Yi1-Chen/FRAG>

It is cited throughout the project as the FRP source (`PREREG_H66_frp_port.md:11` — Algorithm 1,
Eq. (3)–(4) and (6); `PREREG_H64_frag_allocation.md:4,14` — Table 15 verified verbatim in the PDF;
`PREREG_H67_frag_predictor_transfer.md:4` — Appendix A.1, Eq. 7–9), and every `FRP/FRAG port
(2608.25429)` row in `TABLES_v1_generated.md` refers to it. The repository ships **0 Python files**
(README plus 2 PNGs), so our FRP is a reimplementation from the paper plus a domain adaptation —
that provenance claim is **verified and upheld**.

**The overlap.**

| FRAG | overlaps this project's | content |
|---|---|---|
| **Table 11** | **attack-validity** argument | an attack that recovers nothing does not probe robustness |
| **Table 15** | **initial-forgetting-versus-recovery** argument | an inversion against a retain-trained gold model |

FRAG's tables are **LLM relearning**; this project's are **T2I diffusion**. Related prior art on
individual parts: MUSE (2407.06460) on over-unlearning past the retrained model; TOFU (2401.06121)
on two-sided testing against the retain model; Łucki et al. (2409.18025) on ablatable directions;
"Erase or Hide?" (2509.22263); Carlini et al. (1902.06705) §4.14 on attack validity generally.

### WITHDRAWN: that publication therefore requires a new defence method

The conclusion that "the contribution cannot rest primarily on the evaluation methodology" and must
instead be a new defence method **does not follow from an overlap on two tables, and is withdrawn.**
So is any reading in which the evaluation-side contribution is foreclosed.

**Stated plainly: a contribution decision requires a claim-by-claim literature comparison — which
specific claims are new, which are anticipated, and in what setting — and that comparison is NOT part
of this repair gate and has not been done. Nothing here forecloses either an evaluation-side or a
method-side contribution.**

What survives, because it was measured rather than inferred: at a fixed 25-step budget max-over-valid
H moved **0.0667 → 0.2000 → 0.5333** as attacker lr went 1e-5 → 5e-5 → 1e-4, so any future arm must
be evaluated against an **attacker-lr sweep at matched budget**, not a budget sweep at fixed lr.
That is a protocol requirement and it stands independently of any contribution decision.

---

## 7. Changes needed in documents this file's author does not own

Reported, not edited.

| file | what needs changing |
|---|---|
| `AGENT4_EVAL_FINDINGS.md` §A-03 (≈:102-108) | records **"four of twelve"** label strings as differing, naming only `garbage_truck`, `van_gogh`, `english_springer`, `monet`. The count is **nine**; the criterion is internally inconsistent, since it counts `garbage_truck`→`garbage truck` while omitting the five that differ by an underscore alone. `gen_eval_final.py` has since been corrected to nine; this record has not |
| `AGENT4_EVAL_FINDINGS.md` §A-03 | its "26 of 180" and "margins 0.0048–0.0131" are the **four-label** counts. Under the nine-label definition they are **37 of 180** at 0.0027–0.0063. Both are correct under their own definition; the file should say which it uses |
| `AGENT6_PROVENANCE_FINDINGS.md` §D-022 / §5.1 | states **"CONFIRMED, and the direction is as stated"** and that target hits are "biased **upward**". That is an argument from mechanism, not a measurement; it should be restricted to the established count (nine strings differ) with direction and size recorded as **unresolved**. Its §D-022a likewise upholds FINDING66 on the "15/15 can only be lowered by strengthening distractors" argument, which presupposes the direction |
| `AGENT1_UCE_FINDINGS.md` §D-111 | states "The 15/15-vs-0/15 **direction** is robust … the numbers are not", and notes 4 of the 5 wrong classes assigned to our port have malformed prompts. The robustness of the direction under a differently-worded label set is **not** established by the same reasoning the file applies elsewhere; it should be marked unresolved pending the `r4c` re-run |
| `AGENT6_PROVENANCE_FINDINGS.md` §6 | lists "a composition artifact in `r1c_eval`'s overlaid 160/160 IMMA tensors" as one of three live explanations for the R1 result. **That explanation is refuted** (§2.1); the file should carry Agent 2's byte-level audit and REG/D-019's ESD checkpoint as the live candidate |
| `AGENT6_PROVENANCE_FINDINGS.md` §1.6 | records R4 as "executed; interpretation under verification" with upstream's 15/15 "consistent with the upstream edit having been **inert**". **Refuted** (§2.2) |
| `AGENT6_PROVENANCE_FINDINGS.md` §Q16 | marks the 85,850 eigenvalue "COULD NOT VERIFY". It has since been measured at **3.2420e+05** (REG/D-033) |
| `AUDIT_REPAIR_REGISTER_v1.md` §4 (quarantine table) | the `gen_REF_imma_vangogh.json` row still gives the reason as "checkpoint composition **unproven**", and the `r4b/r4c/r4d` row as "upstream edit **may be inert**". Both are refuted; the IMMA row's current reason is the inapt metric and the unreported configuration, and the UCE row's is the confounded port side plus the erase-prompt-expansion question |
| `AUDIT_REPAIR_REGISTER_v1.md` §1, REG/D-022a | still argues the direction ("all nine competing distractors were degraded", "`target_hits` is inflated", "manufactures FINDING65's headline"). §3 of this document supersedes it |
| `AUDIT_REPAIR_REGISTER_v1.md` §1, REG/D-001/002/013/019/026 | marked "Status: OPEN" but variously repaired, verified or superseded; §1 of this document carries the current verdicts |
| `AUDIT_REPAIR_REGISTER_v1.md` §5, G13 | lists a reject option as gate evidence without saying it is a diagnostic rather than a repair, and without the calibration conditions. §4 of this document supplies them |
| `scripts/gen_eval_final.py:178` | its consequence paragraph says "**26** of their 180 images were classified as one of the differing labels" and "median winning margins of 0.0065-0.0131" immediately after correctly stating that **nine** labels differ. Under the nine-label definition the figures are **37 of 180** at 0.0027–0.0063; as written the count and the definition disagree |
| `upstream_manifest.json` | records `"pinned_as_requested": true` alongside `"requested_pin": null` for `unified-concept-editing` and FRAG (a vacuous truth), and gives `vision_relearning`'s licence as `"NONE FOUND"` where the manifest markdown correctly says the README declares MIT |
| `scripts/pick_gpus.sh` | retire it, or bound it to GPUs 0–1 (REG/D-029) |
| `scripts/attack_relearn.py:211` help text | restate the validity gate as **≥ 5/24**, which changes no number (REG/D-024) |
| `PROTOCOL_FROZEN_v1.md` | add the five true image denominators (1/15, 1/24, 1/120, 1/165, 1/48) |

---

## 8. What this document deliberately does not do

- It does **not** alter any historical protocol, result or interpretation. Closed studies keep their
  conclusions; where interpretation has narrowed, that is recorded as a restriction, not a rewrite.
- It does **not** adjudicate the duplicate-cell disagreements, re-score anything, or launch any run.
- It does **not** decide anything about contribution, novelty or publication. §6 records an overlap
  and says explicitly that the decision requires work this gate does not contain.
- It confers **no** standing on any result. The gate is not passed; the open items in §5 are open.
