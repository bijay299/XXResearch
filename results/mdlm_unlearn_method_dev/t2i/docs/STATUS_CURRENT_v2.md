# Current status, one disposition per issue — v2

**Written:** 2026-10-26 · **Revised:** 2026-10-26 (this revision) ·
**Audited snapshot:** `644dc2716b54630b2d6b6e941eda4af14eb37d55`
**Reviewed revision:** `/data/bijaypandey/xxresearch_standalone` at
**`cc778f2576ce1c6281f013fdc6ca9f118517f69d`**, working tree clean. `/data/bijaypandey/t2i_unlearn`
is **not** a git repository, so the artifacts cited here cannot be commit-pinned; they are pinned by
mtime instead. `AUDIT_REPAIR_REGISTER_v1.md` read at mtime **16:18:22**, `FINDING78` at **16:17:58**.
**Agents A and B were editing code concurrently**, so every code-side disposition is as of those
mtimes and no later.

**Supersedes, for status purposes only:** the narrative in `AUDIT_REPAIR_REGISTER_v1.md` §1 and the
six `AGENT*_FINDINGS.md` records.

> **What changed in this revision, in one list.** Coverage extended from `REG/D-001…D-034` to all
> **60** issues (`REG/D-052`, then `REG/D-053`–`REG/D-058`). The seven-verdict taxonomy replaced by **four dispositions, exactly
> one per issue**, and *"a passing test"* removed as a form of verification (`REG/D-035`/G14).
> `REG/D-001` reopened (`REG/D-047`, `REG/D-048`). `REG/D-019`'s control recorded as **run**, with
> its interpretation restricted (§2.3). `REG/D-024`'s **defect claim withdrawn — there is no gate
> defect** (§1.6). The FINAL corpus moved from "pending a cheap re-score" to **permanently withdrawn**
> (§3). Duplicate-cell figures corrected to **29 / 11-of-15 / all in `BASE(unmodified)`** (§5.1).
> FINAL record count **18 → 12**. "Zero post-attack differing-label hits" → **one**. The
> defect-ID collision table corrected from four schemes to **three**.
> **Third pass:** dispositions added for **`REG/D-053`–`REG/D-058`** (§1.1–§1.3); the
> **"8 unmigrated callers" figure RETRACTED as wrong** with evidence (§1.8 — Agent A was right,
> zero live callers, and it is **not** a gate blocker); the E2h **n = 1 / unpaired-seed** caveats
> attached at every site where the `0.0667 → 0.2000 → 0.5333` progression appears, so it is no longer
> readable as a mechanism; and a standing coverage condition added (§1.7).

---

## 0. How to read this document

`AUDIT_REPAIR_REGISTER_v1.md` is **deliberately chronological**. It records hypotheses in the order
they were formed, including several that were later refuted — some of them the lead's own. That is
the right shape for an audit trail and the wrong shape for answering "what is the status of X".
Reading the narrative top to bottom will hand you refuted explanations as if they were live.

**This document is the flat current-state view.** For every issue it gives exactly **one**
disposition, drawn from a fixed four-item taxonomy. Where it and the register's narrative disagree,
**this document's disposition is the current one and the register's narrative is the history**.

**Current verdicts and chronology are kept in separate sections, not separate columns.**

- **§1 — CURRENT VERDICTS.** All **60** issues, grouped by disposition. This is the authority. **Check its stated count against the register before relying on it — see §1.7.**
- **§1A — CHRONOLOGY AND EVIDENCE DETAIL.** The previous revision's table, retained for its
  measurement detail and traceability. Its verdict column uses the **superseded** taxonomy and three
  of its rows were **wrong when written**; those are struck through in place. **Nothing in §1A is a
  current status.**

It settles status only. It does not re-adjudicate evidence, alter any historical protocol, result or
interpretation, or decide anything about contribution or publication.

### The taxonomy — **four dispositions, exactly one per issue**

Revised 2026-10-26 (this revision). The previous seven-verdict taxonomy let an item carry two
verdicts and let *"a passing test"* count as verification. Both are removed. Every current issue
below carries **exactly one** of these four:

| disposition | means |
|---|---|
| **Fixed and independently verified** | the change is made **and** confirmed by a measurement or audit performed by someone other than the implementer. A green test suite alone does **not** qualify |
| **Historical result withdrawn or quarantined** | the number is no longer citable, or is citable only under a stated restriction. **It does not need a replacement measurement to reach this disposition** |
| **Unresolved, with the affected method/path explicitly excluded** | not fixed. The disposition is only complete when the excluded method, arm, claim or code path is **named** |
| **Deferred work that does not affect the next experiment** | real, open, and demonstrably not on the critical path of the next queued run |

**Two rules this document applies, and the reader should check it against.**

1. **An unrecoverable historical result does NOT need an invented replacement to close its
   disposition.** Where checkpoints, attributions or inputs are permanently gone
   (`REG/D-011`, `REG/D-022b`, `REG/D-031`), the honest disposition is *withdrawn*. Describing a
   fresh analogue as the "pending rerun" of a lost record is the error, not the absence of a rerun.
2. **An ACTIVE EXECUTION DEFECT CANNOT BE CLOSED BY DOCUMENTING IT.** Where a live code path can
   still hand out a GPU it does not own, write `DONE` after a failure, or certify untrained weights
   as trained (`REG/D-001`, `REG/D-036`, `REG/D-043`, `REG/D-047`, `REG/D-048`, `REG/D-049`),
   recording the defect in this file changes **nothing** about its disposition. It stays
   *Unresolved* until the code changes and someone else measures it.

**Three words that are not interchangeable, and are not used interchangeably below.**
*Implemented* — the change exists in a file. *Executed* — it ran on real inputs and produced an
artifact. *Independently verified* — someone other than the implementer measured the result.
A passing unit suite establishes the first and **neither of the other two**.

### ⚠ Defect-ID namespace collisions — read before citing any `D-NNN`

**Three independent numbering schemes reuse the same identifiers for unrelated defects.** Citing a
bare `D-NNN` is ambiguous and has already caused confusion between records.

**Corrected this revision (REG/D-052).** The previous revision said *four* schemes collide and gave
`AGENT1_UCE_FINDINGS.md` its own column. **That is wrong:** Agent 1 uses the `D-100`–`D-160` range
and has **zero** `D-0NN` identifiers, so it cannot collide. Only **Agent 4 and Agent 5** are genuine
collisions with the lead's register. Agent 1's column is removed and its single entry below is
re-attributed to the register's own narrative.

| ID | the lead's register means | Agent 4 means | Agent 5 means |
|---|---|---|---|
| **D-008** | FINDING78's "first arm" retraction | — | `venv_uce` is a byte-identical clone |
| **D-010** | upstream UCE may be editing nothing (**refuted** — §2.2) | — | 17 kill-rule gates can report a crash as a measurement |
| **D-013** | the unenforced `FREEZE_ACTIVE` marker | `gen_eval_final.py`'s label set | the unenforced `FREEZE_ACTIVE` marker |
| **D-014** | Meta-v2's shared ζ | `gen_eval_final.py` precommitment fields | zero atomic writes anywhere |
| **D-015** | `attack_core.py`'s false equivalence | registry `defence_identity` | R4c/R4d mis-keyed port checkpoint |
| **D-016** | WDR-v2's inverted schedule claim | registry `vacuity_determinable` | `run_h66_frp.sh`'s attack key |
| **D-017** | FINDING67's 4,000× mechanism | registry `integrity` block | `.labels` lock with no release |
| **D-022** | the FINAL label-text defect | — | the truncated `unet.pt` |
| **D-026** | R4c/R4d cache on an incomplete key | — | source files modified after the freeze |

**Convention used below:** every identifier is written **namespaced** — `REG/D-022` for the lead's
register, `A1/D-100`, `A4/A-07`, `A5/D-014`, `A6/Q18` for the agents. Bare `D-NNN` in any other
document should be resolved against the owning file before it is relied on.

---

## 1. CURRENT VERDICTS — every issue in the register, grouped by disposition

**Coverage, stated as a count so it can be checked.** The register
(`AUDIT_REPAIR_REGISTER_v1.md`) carries **REG/D-001 … REG/D-058 plus REG/D-022a and REG/D-022b** —
**60** issues. All 60 appear **exactly once** below.

**Two successive coverage gaps, both now closed.** The first revision scoped itself to
`REG/D-001…D-034` and had no verdict for twenty issues (`REG/D-052`). The second revision closed
that but was overtaken by **REG/D-053–REG/D-058**, added after it was written — so its own
"covers every current issue" claim went false again. **This is a recurring structural hazard, not
an accident:** a flat status document is stale the moment the register grows. The count above is
stated precisely so the next reader can check it in one step, and §1.7 records the standing
condition.

**Reviewed revision.** Disposition set against
`/data/bijaypandey/xxresearch_standalone` at **`cc778f2576ce1c6281f013fdc6ca9f118517f69d`**
("Publish the absent ESD artifacts; withdraw the gate 'defect'; correct a claim I repeated"); the
register was read at mtime `2026-10-26 16:18:22` and `FINDING78` at `16:17:58`. Agents A and B were
editing code concurrently, so every code-side disposition below is as of those mtimes and no later.

### 1.1 Fixed and independently verified

Confirmed by a measurement or audit performed by **someone other than the implementer**. No row here
rests on a passing test suite.

| id | issue | what was verified, and by whom |
|---|---|---|
| **REG/D-003** | FINDING77's ratios are wrong | FINDING77 rewritten; every ratio recomputed from raw JSON by the lead and by Agent 1 independently (REG/D-032). "~60,000×" → **59.90×**; "60× worse" → **0.58×**, i.e. *smaller* |
| **REG/D-005** | FINDING76 overstates conclusion and mechanism | rewritten against all ten requirements. Sub-item (c) corrected a correction: K=1's **[−0.00457, −0.00011]** and K=32's **−0.00638** are **disjoint**, not overlapping. **Its §3 probe mechanism check remains overstated** — see §1.3 |
| **REG/D-007** | E2h's declared grid and its actual loop differ | **re-derived by this reviewer from the 12 `atk_E2H_*` records.** The runner now tags its loop `"5e-5 50 POSTHOC_UNDECLARED"`; the declaration enumerates **nine** cells (lr 5e-5 × {10,25} and lr 1e-4 × {25}, 3 seeds) and the loop runs **twelve**. Declared: **2 of 9** gate-valid, max-over-valid **0.5333**. Undeclared 5e-5/50: **3 of 3** gate-valid at 0.8000 / 0.9333 / 0.5333 |
| **REG/D-012** | E2h reported success after an abort and pooled two campaigns | per-cell exit codes checked, aborts write `E2H INCOMPLETE` and exit 1, each row tagged with campaign and declared/post-hoc status; re-run on existing artifacts. **Gate condition G2 is NOT passed** — see REG/D-040 in §1.3. This row disposes of the *repair*, not of the gate |
| **REG/D-013** | the `FREEZE_ACTIVE` marker was read by nothing | check now sits **inside** `gpu_reserve.sh`'s critical section; distinct rc=4; requires **both** `ALLOW_FROZEN_RUN=1` and `--allow-frozen`; every override logged. **Verified against the live freeze: refused, rc=4, marker printed.** REG/D-042 later showed this freeze was *masking* a fail-open allowlist — that is a separate defect, not a retraction of this one |
| **REG/D-017** | FINDING67's 4,000× parity figure has the wrong mechanism | the derivation ‖∇L‖ = α/√N **stands**; **4,000× withdrawn, measured ≈9.6×**. Step size as a fraction of `lr`: α=1 → 0.104, α=10 → 0.537, α=4000 → 0.998. Mechanism is **our shared Adam second-moment buffer**, not the published method |
| **REG/D-018** | two provenance misstatements in method scripts | `erase_imma.py`'s `code_released: False` → `True`; `erase_adv.py` re-labelled **our own design** with `IMPL_VERSION`. Pre-repair `erase_adv.py` arms are **not** adversarially-trained results and are invalidated as such |
| **REG/D-023** | FINDING73's tench retain8 includes tench; church arms exclude theirs | reproduced exactly: 0.4583 = the mean *including* tench; excluding it gives 0.5238. Bias **~0.095 against the tench arm**. Correct figures **0.8909** (n=11) / **0.8857** (n=7). `utility_all_nontarget` (n=11) and the 48-image panel are **correct and runtime-asserted**; exposure is **2 of 346** `gen_*.json` |
| **REG/D-032** | six of FINDING77's ratios are wrong; one correction reinstates part of FINDING75 | FINDING75 and FINDING77 rewritten. Accumulation order now **measured**: **< 1.2e-04 on GPU** but **5.6e-02 on CPU** — BLAS-dependent. "λ first vs λ last" is **one of four** differences |
| **REG/D-035** | a **passing test** was enforcing a false claim | both sides moved together: `FINAL.status` → `INSPECTED_DEVELOPMENT_DATA`, `inspected_by=['FINDING65','FINDING66']`, plus a regression guard rejecting "uninspected"/"untouched"/"held out"/"held-out" anywhere in the FINAL record. Verified by re-reading the record, and all three FINAL overlap sets still empty. **Its own lesson is now G14 and is applied throughout this document** |
| **REG/D-037** | the IMMA blank-image filter never executed; the header claimed it had | both halves verified. The filter was dead (nested inside `if [ ! -f metadata.csv ]`), **and** selection is `shuffle(seed=42).select([20,40))`, reproduced **twice**, containing **no blank image** — `29.png` falls in the discarded `[0,20)` window. A **false provenance claim**, not a corrupted training set |
| **REG/D-039** | two reservation-API gaps found by a non-owner | found by the Baselines agent, repaired by the Infrastructure agent, re-measured by the lead. Superseded in substance by REG/D-042. **Its capacity warning stands and is not a defect:** our cards are 40 GB, upstream assumes 80 GB, so an OOM at `META_BATCH=10` is **expected** |
| **REG/D-041** | the lead's own review premise for the CVD/UUID defect was false | measured and false: `nvidia-smi --query-gpu=index,uuid` is **byte-identical** (md5 `b43f627f9fcc`) with CVD unset, `1`, `2,3` and `bogus`. The *actual* faults the premise obscured — v1 never read, reported, validated or set CVD and recorded no UUID — were then fixed (`uuid=` recorded, CVD exported, `MAPCHANGED` state added) |
| **REG/D-042** | the freeze was masking a fail-open allowlist and taking credit for it | found by the Infrastructure agent against the lead's incomplete REG/D-039 report: production returned rc=4 (freeze refusal) while `status`, which is not freeze-gated, showed `allowed=0 1`. Repaired with `${VAR+set}` semantics, a non-retryable **rc=7**, a mode-independent non-empty backstop and `_GR_INIT_RC`. **This closes one instance of the class REG/D-047 names; the class is open** |
| **REG/D-044** | an all-DISCREPANT re-scoring run would report `run_status: complete` | repaired **and observed working in a live run**: the V2 bounded re-score wrote `V2 INCOMPLETE` rather than `DONE`. That is the strongest evidence class available here — an executed run, not a unit test |
| **REG/D-050** | the lead's FINDING66 title asserted the claim its own notice withdrew | retitled, former title preserved verbatim. §9's "there is now no sub-ceiling number anywhere in the FINAL measurements" corrected — **six** sub-ceiling records are on disk (`ATK_uce_sd15` 14/15, `BASE_sd14` 13/15, `uce_sd15` 9/15, `frp` 7/15, `utrap_sd15` 2/15, `utrap_church` 1/15). **This set the standard applied in §1.5: a banner under an asserting title does not correct the title** |
| **REG/D-051** | the duplicate-cell disagreements are concentrated in one control arm | recomputed by the independent reviewer: duplicates reproduce at **68/178**; maximum spread **0.7333 = 11/15** (the lead's 8/15 was only the strict-key subset); **all 29 cells disagreeing on `H_after` belong to the single arm `BASE(unmodified)`** (30 including `gate_valid`). **This row fixes the figures only — the disagreements themselves are unresolved under REG/D-025 in §1.3** |
| **REG/D-054** | the lead's inherited-`CUDA_VISIBLE_DEVICES` refusal aborted on a value the script itself had set | **Found and fixed by Agent A, neither of whom wrote the defect.** Two independent faults in one check: it tested the variable **at stage 4** when `run_r1_imma_official.sh` **exports it itself at stage 2**, so `R1_MODE=prep` from a clean tree would **always abort** complaining about a value it had just set; and `[ -n "${CUDA_VISIBLE_DEVICES:-}" ]` let an **exported-but-empty** value through, which hides **every** GPU and is just as much a pre-selection as `"0"`. Fixed by snapshotting the inherited value **at entry**, before any reservation exists, and testing **set-ness** (`${VAR+x}`) rather than non-emptiness, refusing once at entry rather than at each reservation site. **Scope of the verification, stated exactly:** this is a deterministic control-flow error whose correctness is decidable by reading the order of export and test, and an independent agent did so. **A clean `R1_MODE=prep` run has NOT been re-executed**, so this is verified as a logic fix, not by an executed run |
| **REG/D-056** | the `coherent` pool scheme leaked 2 of the 3 evaluation target prompts into its **forget** pool | **Found and fixed by Agent B, against the scheme the lead was treating as the clean option.** A genuine split contamination in the construction introduced as the principled replacement for the incoherent church-vs-cathedral/chapel/basilica setup. **The repair is the right shape and that is why this closes:** a **hard-failing disjointness gate on RESOLVED RUNTIME pools**, not on source regex — source regex is precisely what missed it. **Critically, NO result is affected:** R2 has **never executed** (`REG/D-002`), so no artifact was built on the contaminated pools. This is the rare case of a blocking split leak caught **before** it reached a number. The pools must still be re-verified through the gate when R2 actually runs |

### 1.2 Historical result withdrawn or quarantined

**None of these requires a replacement measurement to reach this disposition.** Where the inputs are
permanently gone, *withdrawn* is the complete and final disposition (taxonomy rule 1).

| id | issue | what is withdrawn, and what survives |
|---|---|---|
| **REG/D-006** | r4c advertises a flag it lacks and scored with `list(ge.CONCEPTS)` | **`r4c`/`r4d` target accuracies are withdrawn as comparable numbers** — they are not comparable to any other H figure in this project. Ten repairs applied to the script. Same defect class as REG/D-022, so it is **systemic**, not local |
| **REG/D-008** | FINDING78's "first arm to meet both mandate halves" | **RETRACTED.** Its §1 retention result — +0.20 to +0.27 retain8 at identical 0.0000 suppression, **0.9000** against 0.6250 (λ=0.6) / 0.7000 (λ=0.7) — is artifact-exact and **survives**. **No arm in this project meets both halves at any budget** |
| **REG/D-009** | the IMMA reference evaluation is internally contradictory | **Quarantined.** Not citable as an IMMA result, and specifically **not** as "IMMA destroys utility". The quarantine now rests on the **inapt closed-set metric** and on **a configuration the authors never evaluate** (IMMA never evaluates an immunised model standalone) — *not* on a malformed checkpoint, which is refuted (§2.1), and **not** on inheritance from ESD, which is refuted (§2.3) |
| **REG/D-011** | the evaluated checkpoint is unrecoverable from older `gen_*.json` | **Withdrawn as attributed results; retained and labelled as numbers.** **Five** `gen_advunlearn_*` records have `unet: null` **and** `target: null`. **Taxonomy rule 1 applies: no replacement run is owed.** A fresh run would be a new measurement, not a recovery of these |
| **REG/D-021** | IMMA training ran 5× the documented ascent budget | **Withdrawn as a faithful reference reproduction; preserved as legitimate work under its true label.** **2500 ascent updates against the README's 500 — the 5× label must not be removed.** 3 of 100 training images are NSFW-blanked black frames and were trained on (though REG/D-037 shows the README's own 20-sample subset contains none). The trained scope includes **80 self-attention** tensors, so "xatten" is a misnomer in the artifact and variable names |
| **REG/D-022** | every FINAL number was scored against the wrong label set | **The historical FINAL corpus is withdrawn.** `gen_eval.py`'s `LABELS` (`:48`) and `LAB_KEY` (`:50`) differ in **nine of twelve** strings. The V2 bounded run **measured the direction question and answered it: there is no consistent direction.** Six of the twelve records can be re-scored; **six can never be** (REG/D-022b). **The previous revision's "pure re-score, deterministic, CPU-only" framing is withdrawn** — see §3 |
| **REG/D-022a** | the lead's directional analysis of the label-text defect | **RETRACTED AS WRITTEN, and now disproven by measurement.** "All nine competing distractors were degraded", "`target_hits` is inflated" and "manufactures FINDING65's headline" are each withdrawn |
| **REG/D-022b** | six of the twelve FINAL records can **never** be re-scored | **Withdrawn, not re-verifiable.** `attacked/E6B_*`, `E6C_*`, `E6D_*` are absent, so these six cannot be re-scored under corrected label text. **A rerun would be a fresh analogue, not a reproduction — and taxonomy rule 1 means none is owed.** Root cause is the `rm -rf`-on-resume pattern of REG/D-005/REG/D-028 |
| **REG/D-024** | *claimed* defect: "the gate's threshold is finer than its own resolution" | **The DEFECT CLAIM is withdrawn. There is no gate defect.** On a 24-image panel `T_rise ≥ 0.20` **correctly accepts 5/24 = 0.2083 and correctly rejects 4/24 = 0.1667**. The rule is well-defined and was applied consistently to all **808** records; **no record was mis-gated and no verdict changes**. See §1.6 |
| **REG/D-026** | R4c/R4d cache the port checkpoint on an incomplete key | **R4d's port column is withdrawn as a measurement** — it is identical across R4c and R4d **tautologically**, so the port side gives **no** evidence about the preserve pool, and r4d's recorded `"n_preserve": 0` is **false for the port arm**. Upstream's side *was* recomputed in-process and is independently confirmed: removing the pool moves upstream's edit 0.0568 → 0.0536, **1.5 % of the 0.22 gap** |
| **REG/D-031** | R4d **overwrote** R4c's upstream checkpoint | **R4c is withdrawn as reproducible.** Its JSON survives; its checkpoints do not. **The one defect in this audit that destroyed evidence rather than mislabelling it.** Taxonomy rule 1: no replacement is owed, and a re-run would not be R4c |
| **REG/D-058** | the `--no-preserve` field fix was incomplete, and the **published** artifacts carry stale metadata | **`outputs/r4d_uce_church_*.json` are WITHDRAWN. They must be RE-RUN, not re-labelled.** Both were written **16:13/16:14** against a fix landed at **16:18** and still read `n_preserve_port: 10` **when the port used 0**. **These artifacts were published in `cc778f2` — the revision this document was reviewed against — so a published commit carries known-wrong metadata.** Agent B correctly left them untouched rather than editing the metadata of a file whose numbers it could not re-derive. Compounds `REG/D-026` (R4d already a required re-run on an incomplete cache key) and `REG/D-031` (R4c unreproducible), so **the whole R4d cell is withdrawn, not patched**. Separately, `complete: False` on both artifacts was `--retain-concepts` defaulting to **empty**, so **no run could ever be complete** — now fixed with a documented retain panel and a structured `completeness` block, **which means the R4d re-run will finally carry the retain axis `REG/D-034` requires**. **One correction that goes the other way and belongs in the record:** Agent B independently re-derived the `--no-preserve` equivalence and found it **stronger** than claimed — bit-exact in **float32** as well as float64, numerator and denominator separately |

### 1.3 Unresolved, with the affected method/path explicitly excluded

Not fixed. Each row **names** what is excluded. **Taxonomy rule 2 applies throughout: for the
active execution defects here, recording them in this document changes nothing.**

| id | issue | **EXCLUDED until fixed and independently verified** |
|---|---|---|
| **REG/D-001** | GPU reservation was check-then-select, not atomic | **REOPENED. Excluded: all GPU work outside a lead-launched bounded run.** `gpu_reserve.sh` with `flock` + PID/`starttime` ownership was implemented and its own suite passes, **but REG/D-047 shows one environment variable (`GPU_RESERVE_LOCKDIR`) re-opens the exact triple-booking observed at freeze**, and REG/D-048 shows the test-mode interlock is defeatable. **A passing suite is not a verified mutex.** **CORRECTED this revision — the previous "8 callers have not been migrated" was WRONG; see §1.8. Zero runners invoke the retired selector.** The genuine residual is narrower: a legacy campaign holding no reservation and sitting between cells is undetectable, and **130 of 140 `run_*.sh` do not use the reservation API at all** — deliberately, because they are byte-frozen history (`REG/D-040`), so the gate's scope is **active launchers only** |
| **REG/D-002** | `run_r2_meta_official.sh` never invoked the upstream trainer | **Excluded: the label "official Meta-Unlearning" for any artifact in this project.** The runner now stages pools and invokes `src/esd_meta.py` — *implemented*, **never executed**. G5 is unmet. No R2 number exists to verify |
| **REG/D-004** | "A−B isolates embedding selection" is not a valid inference | **Excluded: any subtraction that attributes a difference.** Valid **only** as a triangle-inequality bound given a common reference and denominator. At λ=1000 the bound is tight ([0.108406, 0.108406]); at λ=0.1 it is [0.38265, 0.38567]. Agent 1 adds that R4b's "port" column is **not** `erase_uce.py` (three differences, one named) and that **no R4 cell reproduces any historical arm's configuration** |
| **FINDING76 §3** (sub-item of REG/D-005, tracked separately so no issue carries two dispositions) | the probe mechanism check inside the rewritten document | **Excluded: FINDING76 §3's mechanism claim.** n = 7/3/1 with 40× within-arm spread and undetermined ordering. The §1/§2 trade-off result is unaffected, and REG/D-005's own disposition (§1.1) is the **rewrite**, which is complete |
| **REG/D-010** | pinned upstream UCE may be editing nothing | **The inertness reading is REFUTED and must not be quoted as live** (§2.2). **Excluded: any port-fidelity claim resting on R4b/R4c/R4d.** The surviving open question is narrow — whether upstream should have been invoked with an **expanded erase-prompt set** (`expand_prompts=true`, dW/W **0.0925** vs our port's 0.0587), which is **unreachable through `UCE()`** as called |
| **REG/D-014** | Meta-v2's shared ζ is near-inert; its paired retain difference is unpaired | **Excluded: any attribution of FINDING78's retention gain to ζ's weight.** ζ is inert **as a coefficient** and **has never been swept**; the logged `retain_change` is dominated by sampling noise. The measured retention result **stands** and is attributable to **restoring the pairing**. The behaviour fix (common random numbers) changes numerics, so it may not be applied to any script that produced a historical arm without a new label |
| **REG/D-019** | the ESD checkpoint's training provenance | **Excluded: every provenance claim about this checkpoint, and every causal claim about REG/D-009.** See §2.3 for the only statement the V1 control supports. fp16 variant, dtype round-trip, format conversion and an SD-1.5 base are **ruled out by measurement** (the variant floor is 2.07e-04; resnet blocks reach **0.20**). **EMA vs non-EMA survives and is untested** — the comparison weights are not cached and the host is offline. sha256 `8881d70b…`. The cheapest closing experiment needs **no base model and no GPU**: diff two released `ESDx1` artist checkpoints against each other |
| **REG/D-020** | upstream `src/meta.py` does not run at the pinned commit | **Excluded: `src/meta.py` as an execution path.** `train_step` returns a 4-tuple while **11 call sites unpack 3**; `train_loss` is never assigned; `param.grad` is `None`. **Both** upstream example scripts are unrunnable as written. A finding about the release, **not** a licence to substitute our code and call it official |
| **REG/D-025** | duplicate cells disagree and no record says which code produced them | **UNRESOLVED and not closable by argument. Excluded: every `tested` and `gate-valid` count that aggregates duplicates** — all 178 indexed rows are `admissible_for_recovery = true`, so each **double-counts**. Figures per REG/D-051: **68 duplicate cells over 178 rows, 29 disagreeing on `H_after`** (30 incl. `gate_valid`), **max spread 11/15**, **all 29 in `BASE(unmodified)`**. Over all 808 records: 74 cells across 224 files, 33 disagreeing. **No code-version field exists in any attack record** and `run_inventory.json.code_commit` is `null` for all 234 entries |
| **REG/D-027** | error suppression can manufacture the number that trips the kill rule | **Excluded: any NEW kill-rule verdict until the 17 gates and 33 `\|\| echo 0` sites distinguish "measured 0" from "failed to read".** **The historical risk did not materialise and that part IS verified:** every `destroys the model` / `below K1` / `retain8 = 0.0000` claim in `docs/` was re-derived from its artifact JSON and all trace to genuine records with twelve `per_concept` entries at n=15. Note the awk semantics: `r=""` and `r="0"` both read as below K1, while `r="FAIL"` **clears** K1 |
| **REG/D-028** | no atomic writes anywhere; one checkpoint is silently truncated | **Excluded: `adapters/utrap_church_l1.0k4/unet.pt`** — 511,229,952 B against a sibling's 1,719,229,888 (29.7 %), raises `BadZipFile`, and **passes every `[ -f ]` guard**. **Zero** uses of `os.replace` across 48 Python files; the sweep is not done. **Historical impact: none measured** — 0 grep hits in any metric path |
| **REG/D-029** | `pick_gpus.sh` would hand out another user's GPUs | **Excluded: `pick_gpus.sh` as a selection path — it must be retired or bounded to 0–1.** It iterates all four GPUs and would return 2 or 3, which belong to **uid 1010**. Superseded in practice by `gpu_reserve.sh`, but REG/D-048 shows that is not sufficient |
| **REG/D-030** | "destroys the model" is over-stated: the scorer has no reject option | **Excluded: the words "destroys the model", "broad model destruction" and "clean erasure" for every arm in this corpus.** `retain8 = 0.0000` licenses **"closed-set retain accuracy collapsed to N/120"** and nothing more. It does **not** overturn any K1/K2 decision — those are defined on retain8 and were applied consistently. See §4: a reject option is a **diagnostic, not a repair** |
| **REG/D-033** | `ERRATUM_uce_lambda_scale`'s eigenvalue is 3.8× off | **Excluded: any claim resting on the eigenvalue's magnitude until it has an artifact.** The measured values — **3.2420e+05** for our port's 826-row system, 2.2171e+03 for upstream's 11-row system, against the document's 85,850 — are correct but are **sourced to an agent document and a help string, with no artifact on disk**, and **four live `outputs/r4_uce_*.json` still embed the withdrawn 85,850**. The erratum's qualitative point (λ ≤ 1e3 is inert against `cᵀc`'s spectrum) **survives** |
| **REG/D-034** | our port's erasure is not shown to be clean, and no retain axis was measured | **Excluded: any superiority claim for the R4 family.** Our port's `last..76` selection is partly vindicated (it is why the port erases at all), **but** its 0/15 **scatters across five classes at 4× the edit size** and **no retain axis was measured in R4c or R4d at all**. "Our port erases and upstream does not" is equally consistent with broad damage |
| **REG/D-036** | the Meta reload guard would have certified a **no-op** as successful training | **ACTIVE EXECUTION DEFECT. Excluded: every weight-change magnitude test as evidence that training occurred, and any "trained" status for an R2 checkpoint.** v1 passed a no-op (the main-vs-fp16 variant floor **2.074e-04** is **207×** above its 1e-6 threshold); v2 had a guaranteed false negative; **the third repair is still owed** — all four stage-9 premises (check B, the Pythagorean step, the AdamW lr bound, a universal positive minimum change) are **invalid and removed, not replaced by another threshold**. The replacement must be a **direct evidence chain**. **The 7/7 unit result was synthetic tensor perturbations — a unit test, not an executed training run** |
| **REG/D-040** | G2 was declared passed on a guard nothing called, tested under the wrong shell | **G2 IS RETRACTED AS PASSED. Excluded: every runner's `DONE` marker as evidence of success.** `grep -rln run_guard` matched only the file itself and a test — a control cited but unread. It was also self-tested under `set -u` only, while errexit killed the runner before `local rc=$?` could run; only **4 of 139** runners use `set -euo pipefail`, and they are exactly the four newly hardened launchers. Repaired and re-verified under `set -euo pipefail` (2 FAIL lines recorded, `rg_finish` refused DONE with rc=1), **but there is one production caller**, so G2 requires every *active* runner |
| **REG/D-043** | the re-scoring pipeline's write isolation was defeatable by directory creation | **ACTIVE EXECUTION DEFECT, gate-blocking. Excluded: any re-scoring run until repaired.** Found by the independent reviewer, not the builder: `_guard_path` protects **file** writes but `make_run_dir` never calls it, so `--run-tag '../../../attacked/PWNED'` creates directories in the protected historical tree, and `adapters/` is reachable the same way. `_guard_path` also uses a **separator-less prefix test**, so `outputs/rescore_evil/x.json` is permitted. **`T1b` passing is not proof** |
| **REG/D-045** | the builder's own exposure figure was wrong, and a drift test could not see it | **Excluded: the figure "26 of 180" and the per-arm split (9,7,5,4), which sums to 25.** Recomputed independently by the reviewer **and** the lead: **37 of 180** — utrap_sd15 12/15, utrap_church 10/15, frp 7/15, uce 5/15, plus 2 in `BASE_sd14` and 1 in `ATK_uce`, so it is **not** "concentrated in the four defence arms". Median margins **0.0027–0.0063**, not 0.0065–0.0131. Its claim that the count "bounds how many could move" is **false** and contradicts the builder's own closed-set-coupling argument. **G14 instance:** `T8f` greps for four pre-repair strings, and a test for the *absence* of known-bad text **cannot detect a wrong number that is present** |
| **REG/D-046** | two verification claims rest on weaker evidence than stated | **Excluded: "2000/2000 run-ids unique" as a clean pass, and "`gen_eval.py` unmodified" as a hash claim.** (a) The run-id salt is 6 hex characters = 24 bits, so 2000 draws collide ~11 % of the time; the reviewer observed **45/1 in two of seven runs**. (b) The mtime (`2026-10-25 17:38:24`) is confirmed, but the only hash baseline (`provenance_manifest.json`, 10-23) **predates it**, so there is no hash to compare against |
| **REG/D-047** | **REG/D-001 is reachable again through one environment variable** | **ACTIVE EXECUTION DEFECT, gate-blocking. Excluded: all unattended GPU work.** Production honours `GPU_RESERVE_LOCKDIR` **without refusing it**, so two runners with different values take **different mutexes** and cannot exclude one another. Also `GPU_RESERVE_FOREIGN_MIB` is unvalidated (`abc`/`99999999` hands out a GPU at **30000 MiB with no attributable process**), `GPU_RESERVE_ORPHAN_GRACE` accepts `abc`/`0`, and `_gr_do_acquire:819-843` has **no `*)` default**, so an unrecognised state is claimable. **The finding is the class, not the three instances: every env var this file reads defaults *permissive*** |
| **REG/D-048** | the test-mode interlock can be defeated by copying a binary | **ACTIVE EXECUTION DEFECT affecting a third party. Excluded: `TEST_MODE` as an interlock.** `_gr_nvsmi_is_real` is a **name-only** check — a `cp` of the real `nvidia-smi` passes it (a symlink does not) — so `TEST_MODE=1` plus `GPU_RESERVE_GPUS="2 3"` hands out **index 2 against real telemetry**. GPUs 2–3 belong to another user and are in use. **The one defect whose blast radius lands outside this project** |
| **REG/D-049** | three migrated runners still write `DONE` without checking anything | **ACTIVE EXECUTION DEFECT. Excluded: `run_r4b_queue.sh`, `run_r4c_queue.sh`, `run_r4d_upstream_defaults.sh` as execution paths.** GPU selection was migrated, failure handling was not: no exit status is inspected and `DONE` is written regardless. Separately `run_r1_imma_official.sh:240` gates its stage-4 reservation on inherited `CUDA_VISIBLE_DEVICES`, so a re-run with that set runs the smoke job **unreserved** |
| **REG/D-052** | this document did not cover the newest issues | **Excluded: citing this revision as independently verified.** §1 is the repair, and it was written by the reviewer who found the gap, so **its own verification is owed by the lead or another agent**. The specific errors REG/D-052 names are corrected in place: the **18 → 12** FINAL-record count (§1.2, §3), the "zero post-attack differing-label hits" → **one** (§3), the REG/D-022b framing (§1.2, §3), the eigenvalue's missing artifact (§1.3), and the defect-ID collision table (§0) |
| **grid completeness** | against `PROTOCOL_FROZEN_v1.md`'s six-point rule | **Excluded: cross-arm comparability for the incomplete groups.** Of 86 `(defence, text_encoder, concept)` groups: **30 complete, 56 incomplete, 24 over-run**. ~25 arms keep only the three lr 1e-5 seeds and omit **the three strongest attackers**, covering **all four `utrap2*_church_l0.15k*` arms FINDING76 rests on**, plus `meta2_church_lt0.5`. Since max-over-valid is monotone non-decreasing in attacks run, this is a live comparability defect. Named campaign grids (E11/B–O, E12, E2G, E2H, E4MG) are **complete, 0 missing** |
| **vacuity reporting** | all-invalid attack cells reported without "unmeasurable ≠ robust" | **Excluded: reading a low `H_after` in a zero-valid cell as robustness.** **57 of 341 cells have zero valid records**; **10 of 112 arms** have no gate-valid record. The gate conflates *vacuous/ceiling* arms with *genuinely unmeasurable* ones whose `T_rise` is negative to **−0.2500** |
| **sub-quantisation inference** | 16 sites | **Excluded: any inference below one image.** True denominators: H = **15**, T = **24**, `legacy_retain8` = **120 images**, `utility_all_nontarget` = **165**, paired panel = **48**. **"retain8 (n=8)" and "all-non-target (n=11)" count concepts, not images** — the real steps are 15× finer. No document states the image denominators |
| **`split_registry.py` hard-fail** | advertised, not implemented | **Excluded: `erase_utrap_v2` and `erase_wdr_v2` arms from the leak audit.** It computes `unknown` and only reports it, recording both as unregistered while returning `FINAL_usable: true` — and `erase_utrap_v2` built **all four FINDING76 arms**, so those arms' prompt pools are outside the leak audit entirely |
| **`venv_uce`** (A5/D-008) | byte-identical clone of `venv_t2i` | **Excluded: the claim that a diffusers upgrade distinguishes it — that upgrade does not exist on disk.** Partially exculpated: diffusers 0.31 passes `attention_mask=None`, so the embeddings match anyway — **unverified for ≥0.34** |
| **REG/D-053** | the lead's `setsid` "fix" did not close the reservation residual **and broke the interrupt path** | **ACTIVE EXECUTION DEFECT CLASS. Excluded: any claim that the reservation residual is closed, and all unattended GPU work (with `REG/D-001`, `REG/D-047`, `REG/D-048`).** A change justified as *improving* crash/interrupt handling **actively broke** one of the behaviours the gate requires: `setsid` starts a new session with **no controlling terminal**, so the launcher left the terminal's foreground process group and **Ctrl-C stopped reaching the run**. It also changed the pid after launch, so the pid the operator sees is not the pid that reserves; and group leadership was never a protection — it only decided whether `gpu_reserve.sh` was *allowed* to treat group membership as evidence of descent, while reclamation still came down to a **timer**. **Agent A reverted it** and replaced it with the mechanism that actually carries the invariant: an **`flock(2)` on an INHERITED DESCRIPTOR**, retained by **every descendant through `fork` and `execve`** regardless of group or session leadership, so the lock releases only when the last descendant exits — covering children and grandchildren without a timer, leadership, or the launcher remembering anything. **Why this is NOT yet `Fixed and independently verified`:** the gate requires that **normal completion and interrupted execution both release ownership correctly**, and **no measurement of release-on-interrupt under the new `flock` mechanism is recorded**. The mechanism is sound by construction; the property it exists to guarantee is untested. **Rule 2 applies: this is the behaviour a gate turns on, and documenting it closes nothing** |
| **REG/D-055** | the Meta smoke test **could never produce the checkpoint its guard was meant to check** | **Excluded: `R2_MODE=smoke` as evidence that the trainer trains, and every stage-9 acceptance claim.** `esd_meta.py:1070` nests the **periodic save inside the validation gate**, so with the launcher's `--eval_every $((SM_STEPS+1))` smoke mode could **never write a `step=*/` checkpoint** and **stage 9 was UNREACHABLE** — in the **only mode whose purpose is to prove the trainer trains rather than merely runs**. **Two rounds of argument about stage 9's acceptance criteria were therefore moot**, which is the finding: a guard that cannot be reached is **not a weak guard, it is an absent one** — the same class as `REG/D-040`'s guard with no caller and `REG/D-013`'s unread freeze marker, and the third mechanism in this audit credited for a property nothing exercised. Found and fixed by Agent B, independent of the assignment. **Unresolved because a fix that makes a stage reachable is not verified until the stage is reached, and R2 has never executed (`REG/D-002`)** |
| **REG/D-057** | the evidence chain's instrumentation **recorded nothing** | **Excluded: the evidence chain as proof that training occurred, until it runs against a real optimizer step.** `AdamW` **defines its own `step`**, so patching `Optimizer.step` **never fired** and the executed-update record would have been **silently empty while reporting success** — an evidence chain that produces no evidence is worse than none, because it manufactures confidence. Now uses **global optimizer step hooks**. Caught by **its author's own tests**, which is the right way for this to surface. **10/10 unit tests pass — and Agent B states plainly that these are unit tests on synthetic tensors, NOT an executed run. That framing is correct and this document endorses it: a green suite is not execution evidence (G14).** Agent B additionally verified **all four** unsound stage-9 criteria from source with CPU counterexamples — stronger than the lead's acceptance of them: check B fails a genuine update (rel_same **7.00e-03** > rel_other **2.97e-03**); the Pythagorean step needs orthogonality (1.22e-02 claimed vs **3.00e-03** actual); AdamW's `lr·S` bound is **attained at S=1** and ignores `weight_decay=1e-4`; and a zero-gradient **executed** step moves exactly **0.0**. All four removed with **no substitute threshold**, asserted structurally. **This is `REG/D-036`'s third attempt: implemented, not executed** |

### 1.4 Deferred work that does not affect the next experiment

Open, real, and demonstrably off the critical path of the next queued run.

| id | issue | why deferral is defensible |
|---|---|---|
| **REG/D-015** | `attack_core.py` claimed an equivalence that does not exist | **Drift risk only; no historical result affected and no data quarantined.** `EVAL_FULL_UNET` is a hand-transcribed mirror whose self-test tests the mirror against itself, but the mirrored values **currently match on all 8 axes**, so every divergence record already written is correct. Two axes are structurally uncoverable: RNG provenance and the attacker's data |
| **REG/D-016** | WDR-v2's schedule claim is inverted | The **documentation** correction has landed; the **behaviour** change must land under a **new arm label**, not in place, because `erase_wdr_v2.py` produced FINDING67's E1/E1b arms. The reference **sums**; alternation is gated on an `alternating_` name prefix `weight_dist_reg` does not carry. No queued run depends on it |
| **REG/D-038** | the Meta trainer would have crashed *after* taking a GPU | Repaired by retargeting to `venv_imma` (datasets 5.0.1) with a stage-2 assertion and `HF_DATASETS_CACHE` redirected (the root filesystem is full). **Verified as an import-and-assertion fix only — the trainer has never run**, so this cannot be *independently verified* until REG/D-002 executes. It blocks nothing, because R2 is not the next queued experiment |
| **`venv_imma` version** | documented at 0.31.0, actually 0.24.0 | Documentation disagrees with disk. The downgrade itself is **clean and necessary**. Affects no number |
| **`upstream_manifest.json`** | two vacuous/incorrect provenance fields | `"pinned_as_requested": true` alongside `"requested_pin": null` for `unified-concept-editing` and FRAG is a vacuous truth; `vision_relearning`'s licence is given as `"NONE FOUND"` where the manifest markdown correctly says the README declares MIT. Reported, not owned here |

### 1.9 ADDENDUM — issues opened after this document's second revision (lead, 2026-10-26)

Added by the lead, not Agent C, and therefore **requiring independent review like any other lead
edit**. This section exists because §1.7's standing coverage condition fired for a third time: the
register moved past this document again.

| id | disposition | basis |
|---|---|---|
| **D-059** flock lifetime lock did not cover shell→python→`Popen` | **Fixed and independently verified** | Lead measured all three hops: fd survives `exec python`, is **dropped** by `close_fds=True`, and killing the trainer with a survivor alive left the lock **FREE**. The covering layer is the acquire-time occupancy gate, whose own assumption the lead then measured: a `Popen`-spawned CUDA worker **is** listed by `nvidia-smi` (`2174848, 492 MiB`). Reviewer confirmed compute-apps is consulted unconditionally with **no tunable threshold** |
| **D-060** `PROJECT` was environment-derivable, defeating every other guard | **Fixed and independently verified** | Found by the independent reviewer. Before: `PROJECT=<x>` gave `mode=production`, `lockdir=<x>/.gpulocks`, `FREEZE: none`, and `acquire` returned **rc=3 instead of rc=4** — freeze bypassed. After: refused **rc=6** on `status` **and** `acquire` (lead-measured; asserted by T36a) |
| **D-061** `rg_manifest` returned 0 on failure, so `rg_finish` wrote DONE with no manifest | **Fixed and independently verified** | Lead exercised an unwritable destination: returns non-zero, `rg_finish` **refused DONE**. Reviewer additionally defeated the truncated-publish variant. Asserted by T36f |
| **D-062** `run_v2` had no `set -e`, launched GPU work after its own precondition failed, and could attest to a **previous** run's directory | **Fixed** (not yet independently reviewed) | Reviewer verified 2 of 3 and found a **new** defect: `:88`'s `trap … EXIT` **replaced** the reservation's release trap, so record and hold file survived a *clean* exit and the next acquirer was refused that GPU for ≤900 s. That was the lead's own bug in a file the lead wrote; trap removed outright rather than chained |
| **D-063** `gpu_reserve_drop` returned 0 on retention, leaving `_GR_HOLD_PATH` set so a second acquire was **permitted** and orphaned the first hold file | **Fixed** (not yet independently reviewed) | **The lead previously reported this fixed on the strength of a code comment**, and the reviewer measured that it was not. Now returns a distinct **rc=10**, and the guard keys on `_GR_HOLD_PATH` as well as the fd. Asserted by T36c/T36d |
| **R1** `$PATH` could substitute production telemetry | **Fixed** (not yet independently reviewed) | `_GR_PROD_NVSMI` was a bare name; reviewer measured production reporting fabricated UUIDs `GPU-ffff`/`GPU-eeee` at 0 MiB with `telemetry=OK`. Now resolved absolute at load time, failing closed. Lead verified by attack: `via /usr/bin/nvidia-smi`, no fabricated UUID. Asserted by T36b |
| **R2** `${BASH_SOURCE[0]}` self-reinvocation still relative | **Deferred — does not affect the next experiment** | Reviewer measured rc=127 after a `cd`: **fails closed**, and no launcher is exposed |
| **R3** cleanup-on-SIGINT unverified | **Fixed and independently verified** | Both the author and the lead failed to settle this, and the reviewer explained why neither approach *could*: exit **130 is also produced by being killed with no handler**, and HELD→FREE is produced by the **kernel** closing the descriptor regardless. Handlers now emit a stderr marker existing only inside the handler path. Measured: 2 markers, lock FREE |
| **T36** none of the five post-review fixes had a test (suite 318→318) | **Fixed** (result pending at time of writing) | Six assertions added, each exercising the real function or dispatch rather than a source grep |

**A pattern this addendum is obliged to record, because it is the lead's own.** D-042 → D-047 →
D-060, and then R1, are **one defect in four places**: a value was validated while the thing it
*resolved through* was not. Each earlier fix hardened something computed from an unvalidated root.
Separately, five of the six instances of "a control credited for its shape rather than the property
it enforces" (REG/D-013, D-040, D-053, D-055, D-057, D-059) originate with the lead.

**And a measurement-discipline note.** Four of the lead's own test harnesses in this round were
wrong while the code was correct: three signalled the `setsid` wrapper instead of the runner shell
(recording "handler did not fire" three times), and one let a backgrounded survivor inherit a
command substitution's stdout, stalling the suite for ten minutes. A negative result from a harness
that has not been shown to target the right process is not evidence.

### 1.7 Standing condition on this document's own coverage

**This document has now gone stale twice in one day**, both times because the register grew after it
was written (`REG/D-052`, then `REG/D-053`–`REG/D-058`). That is structural, so it gets a rule
rather than another apology:

> **Any claim that this document "covers every current issue" is only as good as its stated count.**
> It currently claims **60** issues: `REG/D-001…D-058` plus `REG/D-022a` and `REG/D-022b`. Before
> relying on it, run `grep -c '^### D-0' AUDIT_REPAIR_REGISTER_v1.md` against that number. If the
> register is ahead, **this document is not authoritative for the difference** and the register's own
> entries govern until it is refreshed.

This is deliberately phrased as a checkable instruction rather than a promise, because a promise is
what failed twice.

### 1.8 The migration count — RECHECKED, and the previous figure was WRONG

**Question put to this reviewer:** the previous revision said **8 callers have not been migrated**,
while Agent A reported **zero** live `campaign_free.sh` callers project-wide. **Agent A is right.
My figure was wrong, and it was not a gate blocker.**

**Evidence, re-measured directly.**

| check | result |
|---|---|
| Files referencing `campaign_free` (excl. `upstream/`) | **11** — 8 runners, the script itself, `gpu_reserve.sh`, `test_gpu_reserve.sh` |
| **Live invocations** (`grep -rn 'bash\|sh\|.\|source' + campaign_free`, comments stripped) | **ZERO** outside `test_gpu_reserve.sh:1220`, which invokes it deliberately to assert it stays retired |
| What the 8 runner references actually are | **all COMMENTS**, each documenting the migration away from it — e.g. `run_r1b_imma_train.sh:367` "The retired `G=$(bash scripts/campaign_free.sh\|head -1)` selection is **gone** (register D-001)" |
| `scripts/campaign_free.sh` itself | **RETIRED by construction** (`:49`): prints **nothing** on stdout and exits **non-zero**, so even a stale waiter cannot obtain an index from it |
| `run_*.sh` using the reservation API | **10 of 140** |

**Where my error came from, because the mechanism matters more than the number:** I counted files
that **mention** `campaign_free` as if they **called** it. A `grep -l` on a name cannot distinguish a
caller from a comment recording that the call was removed — which is the *same* reasoning error as
`REG/D-040` (a guard credited for existing rather than being called) and `REG/D-013` (a freeze
marker cited but unread), only inverted: there, existence was mistaken for use; here, **mention was
mistaken for use**. Each of the 8 references is in fact *evidence of migration*, the opposite of
what I reported.

**Answer to the gate question: NO, 8 launchers do not bypass the reservation API, and this is not a
gate blocker.** The gate blockers in this cluster remain **`REG/D-047`** (one env var re-opens
`REG/D-001`), **`REG/D-048`** (test-mode interlock defeatable by `cp`, reaching another user's GPUs)
and **`REG/D-053`** (release-on-interrupt untested under the new `flock` mechanism). **None of them
is about campaign_free.**

**The real residual, stated correctly:** **130 of 140 `run_*.sh` do not use the reservation API.**
That is **deliberate and not a defect** — they are the byte-frozen historical runners whose markers
are already declared non-evidence (`REG/D-040`), so G2's and G1's scope is **active launchers only**.
What *is* undetectable is a legacy campaign holding no reservation and sitting idle between cells;
that is a limitation of the design, recorded, not a migration gap.

**Reported, not fixed (file I do not own):** `scripts/test_gpu_reserve.sh` contradicts itself on this
count — `:377` says "campaign_free.sh clause (a) can never match any of its **8** callers" while
`:1252` asserts "zero `run_*.sh` still INVOKE campaign_free.sh (all **nine** migrated)". Eight
runners reference it. One of those two numbers is wrong, and the word "callers" at `:377` carries
exactly the ambiguity that produced my own error.

### 1.5 Where this document applies the REG/D-050 standard

REG/D-050 established that **a banner under an asserting title does not correct the title**. Applied
here: §1.2's REG/D-024 row is titled as a **withdrawn defect claim**, not as a defect with a notice;
§3's heading no longer promises a pending re-score; and the dispositions above replace the verdict
text rather than annotating it. The report bodies, tables and captions this reviewer owns were
corrected the same way rather than banner-flagged — see `REVIEW_C_v3.md` §2.

### 1.6 The gate is not defective — stated once, here, as the authority

**`T_rise ≥ 0.20` on a 24-image panel is a correct, well-defined rule.** It **accepts 5/24 =
0.2083** and **rejects 4/24 = 0.1667**, and it was applied consistently to all **808** records.
**No record was mis-gated; no historical verdict changes.**

The observation that 13 of the 136 invalid records sit at exactly `T_rise = 0.1667` is a
**distributional** fact about how close the rejections sit to the boundary. It is **not** evidence of
a flaw. **Admitting 4/24 would CHANGE THE RULE**, and if that variant is ever explored it belongs in
a **separately labelled sensitivity analysis** — never folded into the headline gate, and never
described as repairing a rounding or granularity error.

**Independently re-derived by this reviewer from the 12 `atk_E2H_*` records on disk** (and the three
`atk_E2G_*` 25-step records), against `/attack_validity/{T_rise, gate, valid}` and
`/recovery/H_after`. T's denominator is confirmed as 24 by the records' own structure: `T_prompts`
has **8** entries at 3 images each.

| attacker lr, 25 steps | seed 0 | seed 1 | seed 6 | gate-valid | max-over-valid H |
|---|---|---|---|---:|---:|
| **1e-5** | T_rise 0.1250 **rejected**, H 0.2000 | T_rise 0.0417 **rejected**, H 0.0667 | T_rise 0.2917 **accepted**, H 0.0667 | **1 of 3** | **0.0667** |
| **5e-5** | T_rise 0.0833 **rejected**, H 0.0667 | T_rise 0.5833 **accepted**, H 0.2000 | T_rise 0.1667 **rejected**, H 0.2667 | **1 of 3** | **0.2000** |
| **1e-4** | T_rise 0.1250 **rejected**, H 0.0000 | T_rise 0.0000 **rejected**, H 0.0000 | T_rise 0.2083 **accepted**, H 0.5333 | **1 of 3** | **0.5333** |

**EVERY reported E2h 25-step learning-rate cell had exactly ONE gate-valid attack out of three.**
The gate-validity rate does **not** vary with attacker lr across these cells.

> **CORRECTION, and it is load-bearing.** It has been stated repeatedly in this project — including
> by the lead — that the stronger attacker produced **more** gate failures. **That is false.** All
> three 25-step cells ran 1/3 valid. Any instance of that claim found in a document owned by this
> reviewer has been removed; instances elsewhere are listed in `REVIEW_C_v3.md` §2.

**Two consequences for how the 25-step figures may be read.**

1. Each of the three max-over-valid values above is a maximum over **a single attack**, not over
   three. `0.0667 → 0.2000 → 0.5333` is a sequence of **n = 1** cells. It is the honest basis for the
   protocol requirement (sweep attacker lr at matched budget) and **too thin** to support a dose
   relationship between lr and recoverability.
2. The seed that is gate-valid **differs by cell** (seed 6, seed 1, seed 6), so the three values are
   not a paired comparison.

**FINDING78's retraction is unaffected and needs no gate adjustment.** It rests on **lr 1e-4,
seed 6**, whose `T_rise = 0.2083 = 5/24` is **correctly accepted**, with H = **0.5333** > the 0.40
bar. The earlier note claiming a corrected boundary would make an E2h cell valid and raise the
25-step max-over-valid to 0.2667 reasoned from the wrong premise: seed 6 at lr 5e-5
(`T_rise = 0.1667 = 4/24`) is **correctly rejected**, and the 25-step max-over-valid **stays as
reported**.

**Withdrawal verified as complete and correct in `FINDING78`** (mtime 16:17:58): its §3 item 2 now
carries the corrected gate note, states 1/3 at every learning rate, and explicitly retracts the
"more gate failures" claim. The register's REG/D-024 (line 655) is correctly marked WITHDRAWN with
the superseded text retained below it for the record. **One residual, reported not fixed:** the
register's §5 gate table (G-row for REG/D-024) and `STATUS`-adjacent prose elsewhere still ask for
the help text to be restated as `≥ 5/24`. That restatement is **harmless clarity, not a repair**, and
must not be presented as fixing a defect — see `REVIEW_C_v3.md` §2.

---

## 1A. CHRONOLOGY AND EVIDENCE DETAIL — REG/D-001 … REG/D-034

**This table is history and supporting evidence, not status.** Where it and §1 disagree, **§1 is the
current disposition.** It is retained because it carries measurement detail §1 compresses. Its
"CURRENT VERDICT" column uses the **superseded** seven-verdict taxonomy and is kept verbatim for
traceability; three of its rows were factually wrong when written and are struck through in place
below (REG/D-001, REG/D-019, REG/D-024).

| id | issue, one line | superseded verdict (NOT current — see §1) | chronological audit note (history, not status) |
|---|---|---|---|
| **REG/D-001** | GPU reservation was check-then-select, not atomic; three runners held GPU 0 while GPU 1 sat idle | ~~**Repair implemented · Verification passed** — `scripts/gpu_reserve.sh` with `flock`, PID+`starttime` ownership, trap release, GPUs 2–3 unreachable by construction; `test_gpu_reserve.sh` **50 passed / 0 failed**, CPU-only~~ **← WRONG WHEN WRITTEN, ON TWO COUNTS.** (i) It offered a **passing test suite** as verification, which G14 forbids; (ii) REG/D-047 and REG/D-048 then showed the mutex is defeatable by one environment variable and the test-mode interlock by copying a binary. **Current disposition: Unresolved — §1.3** | Observed live at freeze (15.3 GB on one card, GPU 1 at 34 MiB). The register still carries "Status: OPEN" from before Agent 5's repair landed. A **residual limitation stands**: a legacy campaign holding no reservation and sitting between cells is undetectable, and **8 callers have not been migrated** — migration must pass `$$`, not `$BASHPID` |
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
| **REG/D-019** | the third-party "ESDx1" checkpoint's training provenance | ~~**Unresolved hypothesis · Required rerun** — all 686 tensors differ from pinned base SD-1.4, resnet blocks reaching 0.20 relative, **so the name is wrong**. The decisive control (ESD checkpoint scored **alone**, 12 DEV concepts, same scorer and seeds, ≈5 GPU-min) **has not been run**~~ **← SUPERSEDED AND PARTLY WRONG.** (i) The control **HAS run** — `outputs/gen_V1_esd_alone_vangogh.json`, 180 images, published. (ii) "**so the name is wrong**" was the lead's own inference and was **WITHDRAWN**: "every tensor differs from our base" does not establish whole-UNet training. **Current disposition: Unresolved, with every provenance claim excluded — §1.3; and see §2.3 for the only statement the control supports** | Deliberately not launched by hand — launching by hand is the failure that produced REG/D-001. Queued as the first job through the reservation path. Two live hypotheses for the checkpoint's identity (full-parameter run despite the `x`, or a different SD-1.4 snapshot); the discriminating test needs network access. **Until the control exists, no statement about where the utility loss comes from is supportable** |
| **REG/D-020** | upstream `src/meta.py` does not run at the pinned commit | **Confirmed defect (upstream)** — three independent fatal faults: `train_step` returns a 4-tuple while **11 call sites unpack 3**; `train_loss` is never assigned; `param.grad` is `None`. The runnable path is `src/esd_meta.py`, whose correct initial checkpoint is **plain SD-1.4** | A finding about the release, **not** a licence to substitute our own code and call it official. Argument hazards verified: `--gamma2_1` passed three times resolves to `0.01, 1.0, 1.0`; `--fix_timesteps False` evaluates to **True** under `type=bool` (only omitting the flag yields False); `--fixed_time_steps 1 2 5 10` hard-errors. **Both upstream example scripts are unrunnable as written** |
| **REG/D-021** | IMMA training ran 5× the documented ascent budget, on partly blank data | **Confirmed defect · Interpretation restricted** — 5000 steps against the README's 1000, i.e. **2500 ascent steps where 500 are specified**; **3 of 100** training images (`7.png`, `29.png`, `41.png`) are NSFW-blanked black frames and were trained on. Every artifact under `upstream_runs/imma/REF_imma_vangogh/` was produced at 5× | Also: the trained scope includes **80 self-attention** tensors, so "xatten" is a misnomer in the artifact name and the variable name, though the manifest is correct. LoRA was warm-started, not reinitialised — our docs were already right |
| **REG/D-022** | every FINAL number was scored against the wrong label set | **Confirmed defect · Historical results' interpretation unresolved.** `gen_eval.py` defines `LABELS` (`:48`) and `LAB_KEY` (`:50`); **nine of the twelve label strings differ** and `church`, `parachute`, `tench` are identical. **The evaluator used inconsistent label text. The effect on scores and conclusions is unresolved pending corrected evaluation.** See §3 | Gate condition G12. The scorer is repaired; the **12** records predate it (the previous revision said "18" — wrong; 12 × 15 = its own 180). ~~Re-scoring is a **pure re-score of the same samples**, deterministic, not a second use of the FINAL split~~ **← WITHDRAWN. That framing sold a permanent loss as a cheap pending rerun.** Six of the twelve records have **deleted checkpoints** (REG/D-022b) and can **never** be re-scored; for them a rerun would be a fresh analogue. REG/D-022a, which argued a direction, is **withdrawn and disproven** — see §3 |
| **REG/D-023** | FINDING73 compares a tench retain8 that includes tench against church arms that exclude theirs | **Confirmed defect (in the document, not the code)** — `legacy_retain8` is defective by design for tench targets but correctly labelled and warned in `gen_eval.py`; the live error is FINDING73 quoting **0.7917** for a tench arm against church arms whose retain8 correctly excludes their target | Bias is **~0.095 against the tench arm** — "the more successful the erasure, the more it is scored as a utility failure". Correct figures: 0.8909 (n=11) / 0.8857 (n=7). `utility_all_nontarget` (n=11) and the 48-image panel are **correct and runtime-asserted**. Exposure is **2 of 346** `gen_*.json` |
| **REG/D-024** | *claimed* defect: the validity gate's threshold is finer than its own resolution | ~~**Confirmed defect** — the gate is `0.20` in all 808 records but T is a 24-image panel, so the operative boundary is ≥ 5/24 = 0.2083. **13 records failed by exactly one image**, three of them load-bearing~~ **← THE DEFECT CLAIM IS WITHDRAWN. THERE IS NO GATE DEFECT.** `≥ 0.20` on a 24-image panel **correctly accepts 5/24 = 0.2083 and correctly rejects 4/24 = 0.1667**. The rule is well-defined, was applied consistently to all 808 records, and **no record was mis-gated**. "Failed by one image" is a *distributional* observation, not a flaw. **Current disposition: Historical result withdrawn — §1.2; authority is §1.6** | The 13 records at exactly `T_rise = 0.1667` are real and correctly rejected. FINDING74's claim **survives**, and it does **not** rest on a one-image gate call in any defective sense — the call was correct. Restating the help text as `≥ 5/24` is **harmless clarity, not a repair**, and must not be reported as fixing a defect. **No document states the T denominator** — that remains a genuine documentation gap |
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

**The ESD-inheritance explanation has since been tested and is refuted — see §2.3.** The previous
revision's closing line ("the live candidate explanation is now REG/D-019's ESD checkpoint, and its
control run is outstanding") is **withdrawn on both halves**: the control has run, and it did not
support the explanation.

### 2.3 "REG/D-009's utility collapse is inherited from the ESD checkpoint" — REFUTED, with the interpretation tightly restricted

**The V1 bounded control ran.** `outputs/gen_V1_esd_alone_vangogh.json` (2,523 B, 15:39), 180 images,
~95 s, GPU 0 reserved exclusively with the override logged to `logs/frozen_override.log`, GPUs 2–3
untouched. Published.

| arm | van_gogh | mean non-target (n=11) | concepts at exactly 0.000 |
|---|---:|---:|---:|
| **ESD checkpoint ALONE** | **0.7333** | **1.0000** | **0 / 11** |
| ESD + IMMA composed | 0.8667 | 0.0424 | 7 / 11 |
| BASE SD-1.4 | 1.0000 | 1.0000 | 0 / 11 |

Verified from the record by this reviewer: **all eleven** non-target concepts sit at `clip_acc`
**exactly 1.0** at n=15 each; target `clip_align` is **0.1646** against 0.225–0.300 for the
non-targets; `utility_all_nontarget` **1.0000** (n=11), `legacy_retain8` **1.0000** (n=8).

> **THE ONLY SUPPORTABLE STATEMENT: the ESD-alone arm did NOT show the same closed-set utility
> collapse on that panel.** That is the whole of what this control establishes.

**It does NOT establish any of the following, and none may be claimed.**

| **NOT established** | why not |
|---|---|
| **clean erasure** | the target is **0.7333**, i.e. still detected in **11 of 15** images. A 1.000 → 0.733 move is a *weak* suppression, not a clean erasure. And per REG/D-030 a closed-set argmax with **no reject option** cannot distinguish erasure from degradation in **either** direction — the same limitation that bars "destroys the model" bars "clean erasure". Saved images plus an **independent** quality/semantic assessment would be required, and neither exists |
| **the checkpoint's training provenance** | this is a **behavioural** control, declared as such before it ran. It can show what the uncomposed checkpoint *does* on this panel; it can never show what it *was trained as*. **EMA vs non-EMA remains live and untested.** The record's only provenance field is the `unet` path string — there is no sha256 or checkpoint-state field in the artifact |
| **that its weight changes are universally benign** | 11/11 at 1.000 on **one 12-concept panel at n=15 per concept**, for **one** target, is not universality. All **686** tensors still differ from pinned base SD-1.4 with resnet blocks at 0.20 relative; that observation is unexplained, not exonerated |
| **over-training as the cause of the IMMA result** | REG/D-021's 5× budget (**2500 ascent updates against the README's 500**) is a **labelled hypothesis about our run**, not a measured cause. Nothing in this control varies the budget, so nothing in it tests the hypothesis. The replacement run at the README's 20-sample budget is what would separate over-training from method behaviour, and **it has not run** |

**Reported, not fixed — the register leans past this line.** `AUDIT_REPAIR_REGISTER_v1.md`'s D-019
resolution block states "**The ESD checkpoint is exonerated. By itself it is a clean, well-behaved
erasure**", that the provenance question is "**largely moot behaviourally**", and that
"**over-training is now the leading hypothesis**". The first is a clean-erasure claim at target
0.7333 under a metric with no reject option; the second converts an untested provenance question into
a settled one; the third promotes a labelled hypothesis to a leading cause. See `REVIEW_C_v3.md` §3.
**What the register gets right, and this document endorses:** the lead's own hypothesis was wrong,
the collapse is **not** inherited from ESD, and it is introduced by composing IMMA.

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
were classified as one of the nine labels whose text differs, at per-arm median winning margins of
**0.0027–0.0063**.

**Two corrections to the previous revision of this paragraph (REG/D-045, REG/D-052).**

1. It is **not** "concentrated in the four pre-attack defence arms". The split is utrap_sd15 12/15,
   utrap_church 10/15, frp 7/15, uce 5/15, **plus 2 in `BASE_sd14` and 1 in `ATK_uce`**.
2. Post-attack arms do **not** have **zero** such classifications. There is **one**, at margin
   **0.0081** against the pre-attack range's 0.0063 — **the same order of magnitude, not "an order
   of magnitude larger"**. Both of those words were wrong.

**And the count does not bound the effect.** 37/180 is a count of images that *landed on* a
differing label; it is **not** an upper bound on how many decisions could move. A closed-set argmax
couples every label to every other, so an image predicted `church` — an **unchanged** string — can
flip when a competitor's text changes.

### THE CURRENT INTERPRETATION — the direction question has been MEASURED and ANSWERED

**Superseded 2026-10-26 by the V2 bounded run** (`rsc_20261026T200003Z_2c8ac2d573c3f300_d022`, GPU 0
reserved exclusively, override logged, 180 images regenerated, **both** label sets scored against
**one** image-feature tensor per image). The previous revision of this section said the effect was
*"unresolved pending corrected evaluation"* and §5.3 called it a cheap CPU-only rerun. **Both are
withdrawn: the evaluation has happened, and the historical records are not recoverable.**

| arm | legacy (`LAB_KEY`) | corrected (`LABELS`) | **delta** | determinism vs record |
|---|---:|---:|---:|---|
| `utrap_church_l0.15k8` | 0.0667 | 0.0667 | **+0.0000** | DISCREPANT |
| `frp_church_r3_l05` | 0.4667 | 0.2667 | **−0.2000** | DISCREPANT |
| `utrap_sd15_church_l0.15k8` | 0.1333 | 0.2000 | **+0.0667** | DISCREPANT |
| `uce_sd15_church_lam1000` | 0.6000 | 0.6000 | **+0.0000** | DISCREPANT |
| `BASE_sd14_anchor` | 0.8667 | 0.8000 | **−0.0667** | DISCREPANT |
| `BASE_sd15_anchor` | 1.0000 | 1.0000 | **+0.0000** | DISCREPANT |

> **THE CURRENT INTERPRETATION, VERBATIM: the label text matters, it is sometimes large, and it has
> NO CONSISTENT SIGN.** `utrap_sd15_church_l0.15k8` moves **+0.0667** while `frp_church_r3_l05`
> moves **−0.2000**. Predictions moved on up to **9 of 15** images per arm; target decisions on up to
> **3**. Every directional claim anyone made about this defect — in either direction — is refuted by
> measurement, not merely unsupported.

**DISPOSITION: Historical result withdrawn — permanently, not pending.** `history_comparability =
none_exact`. **All six** regenerable arms are DISCREPANT against their records, so regeneration at
matched seeds does **not** reproduce the original images and the determinism assumption the run was
built to test **fails**. The other six have deleted checkpoints (REG/D-022b). Therefore:

- **FINDING65's ratios and FINDING66's figures are not correctable by re-measurement.** They are
  withdrawn **permanently**. Taxonomy rule 1 applies: **no replacement run is owed**, and a new FINAL
  experiment would address the *questions*, not validate these *records*.
- A subtlety worth keeping: `BASE_sd14_anchor`'s legacy aggregate (0.8667) *equals* its historical
  13/15, **yet the probe still reports DISCREPANT**, because it compares **per-image** predictions
  rather than aggregates. Aggregate agreement concealing per-image disagreement is exactly what a
  weaker probe would have missed.
- **The machinery behaved correctly, and that is the other result.** The run exited **rc=1** and the
  guard wrote `V2 INCOMPLETE` rather than `DONE` — REG/D-044's fix working as intended. Before that
  fix an all-DISCREPANT run would have been filed as `complete`.

**What survives** is a *within-run* measurement on regenerated images showing that label text
matters and acts in both directions. That is a methodological result, not a rehabilitation of any
historical number.

**Count split, corrected twice (the second correcting the first).** The nine differing strings split
**6/3, not 5/4**: six differ **by an underscore alone** (`garbage_truck`, `cassette_player`,
`chain_saw`, `french_horn`, `gas_pump`, `golf_ball`) and three are **genuinely reworded**
(`van gogh painting`→`van_gogh`, `english springer spaniel`→`english_springer`,
`monet painting`→`monet`). The total is nine either way, and the split carries **no implication
about direction**.

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

**Duplicate `(arm, seed, lr, steps)` cells disagree by up to 11/15 images, and no record carries a
code-version field.**

| scope | duplicated cells | files | disagreeing |
|---|---:|---:|---:|
| all 808 attack records | 74 | 224 | **33** |
| the 696 indexed in the registry | 68 | 178 | **29 on `H_after`** (30 incl. `gate_valid`) |

**Two corrections to the previous revision (REG/D-051), one of which cuts each way.**

- **The maximum spread is 0.7333 = 11 of 15 images, not 8/15.** `0.5333 = 8/15` is correct **only**
  for the strict-key subset; the previous revision made it the headline for all the disagreements and
  thereby **understated** them. Duplicates reproduce at **68/178**.
- **All 29 cells disagreeing on `H_after` belong to the single arm `BASE(unmodified)`** — the
  attack-on-base control. This was disclosed nowhere before, and it **reduces** the concern
  materially: the disagreement is concentrated in the unmodified-base control rather than spread
  through the defence arms, so it does **not** undermine the defence-arm comparisons. It still needs
  explaining — an unmodified base attacked twice should not diverge by 11/15. **Both the correction
  and the reassurance belong here; omitting either would misstate it.**

- All 178 indexed rows are `admissible_for_recovery = true`, so each **double-counts** in every
  `tested` and `gate-valid` column of `TABLES_v1_generated.md` — and the 15 `BASE(unmodified)` rows
  there are **precisely** the rows that inherit the discrepancy.
- **13 cases survive the strictest available key** — byte-identical `hparams` *and* identical
  `T_prompts`/`H_prompts` — and still disagree. `T_rise` moves with `H_after`, so **the attacks ran
  differently**.
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
| ESD-checkpoint-alone control (REG/D-019) | **DONE — this row is closed.** It ran; `outputs/gen_V1_esd_alone_vangogh.json`. The ESD-inheritance explanation is refuted (§2.3). **What remains open is the provenance question (EMA vs non-EMA), and it is excluded, not pending a rerun** — the closing experiment needs no GPU |
| FINAL re-score with `LABELS` (REG/D-022, G12) | **NOT a rerun. Closed as a permanent loss (§3).** The V2 run executed; all six regenerable arms are DISCREPANT and the other six have deleted checkpoints. **No replacement run is owed** (taxonomy rule 1). A **fresh** FINAL split must be frozen before any new configuration is selected against one, the existing set being spent |
| R4d with a corrected `port_dir` key (REG/D-026) | **Required rerun** |
| R4c (REG/D-031) | **Required rerun** — its checkpoints no longer exist on disk |
| retain/utility panel for the R4 family (REG/D-034) | **Required rerun** — no superiority claim may rest on R4 without it |
| official Meta pipeline (REG/D-002, G5) | **Required rerun** — runner repaired, never executed |
| grid completeness against `PROTOCOL_FROZEN_v1.md`'s six-point rule | **Confirmed defect, unrepaired.** Of 86 `(defence, text_encoder, concept)` groups: **30 complete, 56 incomplete, 24 over-run**. ~25 arms keep only the three lr 1e-5 seeds and omit **the three strongest attackers**, covering **all four `utrap2*_church_l0.15k*` arms FINDING76 rests on**, plus `meta2_church_lt0.5`. Since max-over-valid is monotone non-decreasing in attacks run, this is a live **comparability** defect. Named campaign grids (E11/B–O, E12, E2G, E2H, E4MG) are **complete, 0 missing** |
| the FINAL set described as untouched in machine-asserted form | **FIXED and independently verified (REG/D-035) — this row is closed.** The literal and its asserting test moved **together**: `FINAL.status` → `INSPECTED_DEVELOPMENT_DATA`, `inspected_by=['FINDING65','FINDING66']`, plus a regression guard rejecting "uninspected"/"untouched"/"held out"/"held-out" anywhere in the FINAL record. The three overlap sets are still empty, so the disjointness computation was never in doubt. **Its lesson is now G14 and is applied throughout this document: a green suite shows code matches its assertions, not that the assertions are true** |
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

**Stated with its true n, which §1.6 derives from the records.** Each of those three values is a
maximum over **one** gate-valid attack, because **every** 25-step cell had exactly **1 of 3** attacks
gate-valid, and the valid seed **differs by cell** (6, 1, 6). So the sequence supports the protocol
requirement — sweep the attacker's lr, because stopping one cell earlier would have reported this arm
as mandate-clearing — and it does **not** support a dose relationship between lr and recoverability.
**Nor did the stronger attacker produce more gate failures: the validity rate is 1/3 at all three
learning rates.**

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
| `scripts/attack_relearn.py:211` help text | **OPTIONAL CLARITY, NOT A REPAIR.** Restating the gate as `≥ 5/24` changes no number and fixes no defect — there is no gate defect (§1.6). If it is done, it must **not** be logged as closing REG/D-024, and the words "granularity", "resolution" and "rounding" must not appear beside it. Recording both the fractional threshold and the integer image count in **new** records is a genuine improvement; **historical records must not be restated** |
| `AUDIT_REPAIR_REGISTER_v1.md` §1, REG/D-019 resolution block | states "**the ESD checkpoint is exonerated**", "by itself it is **a clean, well-behaved erasure**", that the provenance question is "**largely moot behaviourally**", and that "**over-training is now the leading hypothesis**". The first is a clean-erasure claim at target **0.7333** under a metric with no reject option — barred by the same REG/D-030 logic that bars "destroys the model"; the second converts an untested question (EMA vs non-EMA) into a settled one; the third promotes a labelled hypothesis about **our run** to a cause. §2.3 states the only supportable reading |
| `AUDIT_REPAIR_REGISTER_v1.md` §1, REG/D-019 | its **"Status: OPEN, queued behind the infrastructure gate"** line still sits **below** the ✅ RESOLVED block in the same entry, so one entry carries two contradictory statuses |
| `AUDIT_REPAIR_REGISTER_v1.md` §1, REG/D-025 | still headed "**30 duplicate cells disagree**" and still says "up to **8/15** images" in its body, which REG/D-051 corrects to **29 on `H_after`** and **11/15**. The concentration in `BASE(unmodified)` is also absent from this entry |
| `AUDIT_REPAIR_REGISTER_v1.md` §5, G1 | lists `test_gpu_reserve.sh` output as the evidence that exclusive reservation "**is proven to work**". Under REG/D-047 and REG/D-048 that suite passes while the mutex is defeatable by one env var and the interlock by `cp`. G1's evidence line needs the env-var sweep and a non-name-based `nvidia-smi` check added, or G1 reads as satisfiable by a green suite |
| `AUDIT_REPAIR_REGISTER_v1.md` §5, G14 vs G1/G2/G5 | G14 says a green suite is not evidence its assertions are true, but G1, G2 and G5 are each phrased around test output. The gate table should be swept once against its own G14 |
| `docs/TABLES_v1_generated.md` (owned here, now corrected) | had carried the superseded "30 cells / 8 of 15" figures and described registry-absent arms in words that read as missing-from-disk. Corrected in place this revision; recorded here because other documents copied those figures from it |
| `docs/CORRECTION_REGISTER_r5.md:265` and `HANDOFF_r5.md:140,154` | "**18** `final_*.json` records" (there are **12**), and REG/D-022b described as a "pure re-score", "deterministic" and "CPU-only". The V2 run refutes all three descriptions (§3) |
| `PROTOCOL_FROZEN_v1.md` | add the five true image denominators (1/15, 1/24, 1/120, 1/165, 1/48) |

---

## 8. What this document deliberately does not do

- It does **not** alter any historical protocol, result or interpretation. Closed studies keep their
  conclusions; where interpretation has narrowed, that is recorded as a restriction, not a rewrite.
- It does **not** adjudicate the duplicate-cell disagreements, re-score anything, or launch any run.
- It does **not** decide anything about contribution, novelty or publication. §6 records an overlap
  and says explicitly that the decision requires work this gate does not contain.
- It confers **no** standing on any result. The gate is not passed; the open items in §5 are open.
