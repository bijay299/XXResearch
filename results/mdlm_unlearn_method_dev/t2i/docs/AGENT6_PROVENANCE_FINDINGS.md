# Agent 6 — provenance & traceability audit of the six owned documents

**Date:** 2026-10-26 · **Scope:** every quantitative and status claim in
`T2I_REPORT_r5.md`, `UPSTREAM_STATUS_r1.md`, `UPSTREAM_FIDELITY_MANIFEST.md`, `HANDOFF_r5.md`,
`TABLES_v1_generated.md`, `CORRECTION_REGISTER_r5.md`.

**Method.** Every claim was traced to a raw artifact under `outputs/`, a log under `logs/`, an
upstream source file, or a file mtime. No prior document was treated as evidence for itself. All
verification was CPU-only JSON/log reading; no GPU work, no `run_*.sh`, no `registry_v1.py`
(it sha256s many GB).

**Project is FROZEN** (`FREEZE_ACTIVE`, declared 2026-10-26T12:43). Nothing here was re-run to
settle a disagreement; where an artifact was missing, the claim is marked unverifiable rather than
assumed.

---

## 0. Headline

The documents' **numbers are mostly right**; their **status statements are systematically wrong**.
Eleven documented "not run / blocked / queued / unrun" claims are false — R1 ran end to end, R4 ran
in four phases, E1/E2/E4/E6 all produced artifacts.

**The mandate conclusion survives, but its stated basis did not.** Four documents assert the
mandate is unmet "at any budget or learning rate tested". Until E2h returned, that was **not
established at the 25-step budget**: `meta2_church_lt0.5` (retain8 0.9000) clears the robustness
half at attacker lr 1e-5 (0.0667) *and* lr 5e-5 (0.2000). The claim is rescued — not refuted — by
`logs/e2h_stronger.log`, which shows lr 1e-4 at the same 25-step budget reaching **0.5333 > 0.40**.
Max-over-valid is a ratchet (`FINDING35` S4), so 0.5333 stands and can only rise. **No arm has met
both halves at any budget.** `FINDING78`'s companion "first arm" claim is **retracted** (lead,
D-008); its §1 retention result survives. My repairs therefore *re-evidence* the mandate sentence
rather than weakening it, and the FINDING78 row added to the report's summary table is recorded as a
**retraction, not a result**.

---

## 1. Traceability table

Classification: **(1) CONFIRMED DEFECT**, **(2) SUSPECTED**, **(3) VALID**, **(4) HISTORICAL RESULT
needing reinterpretation**, **(5) REPAIR PERFORMED**.
Line numbers are pre-repair, against the files as found at 2026-10-26T12:44.

### 1.1 `UPSTREAM_STATUS_r1.md` — status claims

| # | claim | doc:line | source artifact | verified value | verdict |
|---|---|---|---|---|---|
| S1 | IMMA "training done ❌ **R1 not run**" | `UPSTREAM_STATUS_r1.md:19` | `logs/r1b_imma_train.log:1` + `grep '/5000'` | run started 11:59:23, **reached 5000/5000** | **1 + 5** MISMATCH |
| S2 | IMMA "eval done ❌" | :19 | `logs/r1c_eval.log:32-33`, `outputs/gen_REF_imma_vangogh.json` | eval ran 12:39, target(van_gogh)=**0.8667**, all-non-target **0.0424**, legacy_retain8 **0.0333** | **1 + 5** MISMATCH |
| S3 | "R1 blocked: ESD checkpoint is a **third-party download not yet attempted**" | :104 | `upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt` | **present**, 3,438,307,109 B, mtime 2026-10-26 11:36 | **1 + 5** MISMATCH |
| S4 | "R1 blocked: needs ~100 generated training images" | :104 | `upstream_runs/imma/data/vangogh/*.png` | **100 PNGs**, mtime 11:39–11:43 | **1 + 5** MISMATCH |
| S5 | "R1 blocked: `accelerate launch` config. None is done" | :104 | `logs/r1b_imma_train.log` (full 5000-step trace with `lr=`/`step_loss=` telemetry) + `upstream_runs/imma/REF_imma_vangogh/imma_unet_xatten_layer.pt` (374,210,382 B, mtime 12:21) | trainer executed and saved its artifact | **1 + 5** MISMATCH |
| S6 | Meta "training done ❌ **R2 not run**" | :20 | `logs/r2_meta.log` | **partially true** — pools built, trainer never invoked | **1 + 5** overstated-as-nothing; see S7 |
| S7 | "R2 blocked: `./dataset/{hrm,irt,tgt}` **are not in the repo and must be constructed**" | :105 | `logs/r2_meta.log:45-...`, `upstream_runs/meta/dataset/{hrm,irt,tgt}/{train,test}` | pools **built** 12:39–12:47: hrm 30/30, tgt 30/30, irt 30/30; `[r2pools] done`. First attempt failed `rc=1` (`ModuleNotFoundError: pandas`) at 12:21 | **1 + 5** MISMATCH — blocker discharged |
| S8 | UCE "eval done ⏳ **R4 queued, running**" / "R4 none — queued, waiting only on a free GPU" | :22, :88, :107 | `outputs/r4_uce_*.json` (11:32), `r4b_uce_lamb{0.1,0.5,1000.0}.json` (11:46), `r4c_uce_lamb{0.1,0.5}.json` (12:23–12:25), `r4d_uce_lamb{0.1,0.5}.json` (12:42–12:45) | **four phases executed**; upstream code `executed: true` in r4b/r4c/r4d | **1 + 5** MISMATCH |
| S9 | "R3 blocked ... Not started" | :87, :106 | no `logs/r3*`, no `outputs/r3_*`; `logs/*r3*` hits are our own FRP arm label `_r3_l05` | **confirmed not started** | **3 VALID** |
| S10 | "In flight (not interrupted): E4m (GPU 0), E2g (GPU 1), R4 queued behind them" | :44-45 | `logs/e4mg_grid.log` ends 11:40, `logs/e2g_grid.log` ends 11:29, `logs/e2h_stronger.log` `E2H DONE 12:38:40`; `ps` shows **no project process running** | all finished; nothing in flight | **1 + 5** MISMATCH |
| S11 | "**No method is marked reproduced.**" | :10 | `UPSTREAM_FIDELITY_MANIFEST.md:252` defines `REF_*` = "**official reproduction** — authors' code, authors' protocol", and `REF_imma_vangogh` now exists | the prefix convention **does** mark a run reproduced | **1 + 5** internally contradicted — see F5 |
| S12 | "All five repositories cloned and pinned; three audited revisions resolved exactly as requested" | :29 | `outputs/upstream_manifest.json` | 5 repos; 3 non-null `requested_pin`, all `resolved_commit == requested_pin` | **3 VALID** |
| S13 | "FRAG: `README.md` + 2 PNGs and **zero Python files**" | :33 | `outputs/upstream_manifest.json` FRAG `python_files: 0` | 0 | **3 VALID** |
| S14 | IMMA scope table: 93.5M / 10.88% ref; 49.6M attn1; 44.0M attn2; 44.0M ours | :52-56 | `outputs/imma_scope_fidelity.json` | 859,520,964 full; 93,536,640 (10.88%); 49,574,080 (5.77%); 43,962,560 (5.11%); ratio 2.1276 | **3 VALID** |
| S15 | "the official `--lamb` default is **0.5** against our **0.1**" | :78 | `outputs/r4_uce_lamb0.1.json` `default_mismatch_note`; r4b/r4c/r4d configs run both | consistent | **3 VALID** |
| S16 | UCE "two candidate discrepancies, **being measured not asserted**" | :75 | r4b/r4c/r4d | **now measured, and large** — see Q7–Q9 | **1 + 4 + 5** stale framing |

### 1.2 `T2I_REPORT_r5.md` — quantitative claims

| # | claim | doc:line | source artifact | verified value | verdict |
|---|---|---|---|---|---|
| Q1 | "JSON files under `outputs/` **1,278**" | `T2I_REPORT_r5.md:40` | `outputs/registry_v1.json` `counts.json_files_under_outputs` = 1278; live `glob('outputs/*.json')` | registry **1278**; **actual 1,424** (1,428 recursive) | **1** STALE +146 |
| Q2 | "attack **records** **696**" | :41 | registry `attacks` length = 696; live `glob('outputs/atk_*.json')` | registry **696**; **actual 808** on disk | **1** STALE +112 |
| Q3 | "quarantined 0" | :42 | registry `counts` | 0 | **3 VALID** (vs registry) |
| Q4 | "vacuous **9**"; "**six** on `gift_church_b0.1` and **three** on `uce_sd15_church_lam50000`" | :43, :58-62 | registry `vacuous_no_erasure` flag; `TABLES_v1_generated.md:539` names all nine run_ids | 9 = 6 + 3, labels exact | **3 VALID** |
| Q5 | "VOID post-attack utility **136**"; "admissible for recovery **687**"; "paired utility **324**"; "defence builds evaluated **335**" | :44-47 | registry `counts` | 136 / 687 / 324 / 335 — all exact | **3 VALID vs registry**, but all four inherit Q1/Q2 staleness |
| Q6 | "*(Counts regenerated 2026-10-26 after FINDING65–70 added ~54 attack records ... no new defence builds)*" | :50-52 | registry mtime **05:11:25**; FINDING71 05:39 → FINDING78 12:00; 112 newer `atk_*` (E12 24, E2G 18, E4MG 12, E2H 12, E11J 9, E11M 9, E11K/L/N 6 each, E11H 5, E11I 3, E11O 2); new builds `meta2_church_lt{0.5,1.0}`, `utrap2*` (8), `REF_imma_vangogh` | the parenthetical is **itself stale**, and "no new defence builds" is now false (≥11 new builds) | **1** |
| Q7 | §3.1 `utrap_church_l0.15k8` "target suppression **0.0000**, retain8 **0.8667**, all-non-target **0.8364**" | :68-69 | registry `defence_builds['utrap_church_l0.15k8']` | 0.0, 0.8666667, 0.8363636 | **3 VALID** |
| Q8 | §3.1 "**Six attacks, all gate-valid (T 0.000 → 0.958–1.000), all six reaching H = 1.000.**" | :71-72 | registry: 7 records at 300 steps / 6 distinct (seed, lr) configs | **`H84M_utrap_church_l0.15k8_sd0_lr1e-4`: T 0.000→0.9583, H_after = 0.9333**, and its paired utility collapses **0.8333 → 0.1875**. The quoted "0.958" lower bound *is* that record, so the sentence counts it in the T range and excludes it from the H claim | **1 CONFIRMED DEFECT** — self-inconsistent |
| Q9 | §3.1 paired table: sd0 0.8333→0.9792, sd1 0.8542→0.9167, sd6 0.8750→0.8958 | :76-78 | registry `pau_before`/`pau_after`, `pau_status: MEASURED` | 0.8333333→0.9791667; 0.8541667→0.9166667; 0.875→0.8958333 | **3 VALID** |
| Q10 | §3.1 "Seed 0 is 40/48 → 47/48" | :80 | 0.8333×48 = 40.0; 0.9791667×48 = 47.0 | exact | **3 VALID** |
| Q11 | §3.2 FRP SD-1.5 0.1333/0.9333; FRP BK-SDM 0.4667/0.9250; GIFT SD-1.5 0.4000/0.8167; GIFT BK-SDM 0.3333/0.7417; UTRAP SD-1.5 0.0000/0.9583; UTRAP BK-SDM 0.0000/0.7917; UCE BK-SDM 0.0000/0.9333; UCE SD-1.5 λ1000 0.2000/0.9500; Meta A SD-1.5 0.0000/0.7667 | :93-101 | registry `defence_builds` (`gift_*_b1.0`, `uce_bksdm_church`) | **all nine exact** | **3 VALID** |
| Q12 | §3.2 UCE SD-1.5 λ=1000 "gate-valid **2/2, third running**" | :100 | registry, 300 steps lr 1e-5: tested **6**, valid **6**, max-H-valid 1.0 | the third completed; "running" is stale | **1** |
| Q13 | §3.2 UTRAP SD-1.5 "**3/3**" | :97 | registry: tested **4**, valid **4** | 4 records (duplicate label) | **2 SUSPECTED** — undercount, same verdict |
| Q14 | §3.2 Meta variant A gate-valid "**—**" | :101 | registry: tested 2, valid 2, max-H 1.0 | measured, not absent | **2 SUSPECTED** understatement |
| Q15 | §3.3 λ table 0.1/0.5/1.0 → 0.5333/0.4667/0.4167; 10→0.3500; 30→0.0667/0.4583; 100→0.0000/0.5667; 300→0.0667/0.8083; 1000→0.2000/0.9500; 10000 & 50000 → 1.0000/1.0000 | :117-123 | registry builds `uce_sd15_church*` | **all thirteen values exact** | **3 VALID** |
| Q16 | "`cᵀc` has max eigenvalue **85,850**" | :125 | `outputs/r4_uce_lamb0.1.json` `default_mismatch_note` cites `ERRATUM_uce_lambda_scale.md`; the parent artifact is that erratum, not a raw JSON | no primary artifact in `outputs/` recomputes it | **2 SUSPECTED** — single-sourced to a document |
| Q17 | §1 "The corrected implementations exist and are unit-tested; **none has been run.**" | :30-31 | `outputs/gen_meta2_church_lt{0.5,1.0}.json`, `gen_utrap2*` (8), `gen_wdr2_church_a{1.0,4000}.json`, 29 `E4M*/utrap2*` records, 18 `atk_E2G_*` | **all three have been run** | **1 CONFIRMED DEFECT** |
| Q18 | §6 / §8c "The mandate is **unmet** ... at **any budget or learning rate tested**" | :175-176, :243-245 | `outputs/gen_meta2_church_lt0.5.json` retain8 **0.9000**; `atk_E2G_…_s25` → max-over-valid **0.0667** (1/3 valid); `logs/e2h_stronger.log:291-299` → same 25-step budget gives **0.0667** (lr 1e-5), **0.2000** (lr 5e-5), **0.5333** (lr 1e-4) | **conclusion VALID, basis was not.** Two of three attacker lrs clear 0.40 at 25 steps; the budget fails only on the lr 1e-4 cell, which was unmeasured when §6/§8c were written. Max-over-valid at 25 steps = **0.5333 > 0.40** | **3 VALID (re-evidenced) + 2** — the sentence was right by luck, not by measurement |
| Q18b | `FINDING78` §3 "first arm to satisfy **both halves** at any budget" | `FINDING78` (not owned) | `logs/e2h_stronger.log` lr 1e-4/25 = 0.5333; `FINDING35` S4 ratchet | **RETRACTED by the lead, registered D-008.** Independently confirmed; §1's +0.20/+0.27 retention result is unaffected and survives | **1 CONFIRMED DEFECT**, repaired by the lead |
| Q19 | §7 "Unblocked and cheap: **E1** ... **E2** ... **E4** ... (not yet run)" | :195-197 | `logs/e1_wdr.log`, `e1b_wdr.log`, `gen_wdr2_*`; `logs/e2_meta_v2.log`, `e2g_grid.log`, `e2h_stronger.log`; `logs/e4_h90.log`, `e4m_matched.log`, `e4mg_grid.log` | E1, E2, E4 **all ran** (E1 at α ∈ {1.0, 4000}, not the prespecified {0.3,1.0,3.0}) | **1** |
| Q20 | §8 heading "findings since r5 was written (**FINDING65–70**)" + "Summarised here so the current report is not stale against its own repository" | :204-209 | `docs/FINDING71…FINDING78*.md` exist (05:39 → 12:13) | **eight findings omitted**, several of which amend §8's own rows | **1 CONFIRMED DEFECT** |
| Q21 | §8 row 70: budget axis "**non-monotone** at lr 1e-4 ... **because** the attack starts destroying the model" | :218 | `FINDING71` title + body: damage is corpus-wide (n=229) but the **non-monotone H did not reproduce** and is arm-specific | mechanism half refuted | **1 + 4** |
| Q22 | §8 row 69 "no budget from 5 to 300 leaves any arm meeting the mandate's robustness half" | :217 | `logs/e2h_stronger.log:291-299` (max-over-valid at 25 steps = 0.5333 across the attacker-lr sweep) | **holds**, but only on evidence that post-dates the sentence; at the time it covered one attacker lr | **3 VALID (re-evidenced)** |
| Q22b | `FINDING78` §1 retention result: retain8 **0.9000** at target **0.0000** vs variant A **0.6250** (λ0.6) / **0.7000** (λ0.7), i.e. **+0.20 to +0.27**; variant A λ1.0 = **0.1167** | `FINDING78` §1 | `outputs/gen_meta2_church_lt0.5.json`; `outputs/gen_h80_meta_church_lam{0.6,0.7,1.0}.json`; `TABLES_v1_generated.md:477` | 0.9000 / 0.6250 / 0.7000 / 0.1167 — all four exact; deltas 0.275 and 0.200 | **3 VALID** — citable |
| Q23 | §8c "**No corrected baseline implementation has been validated on a model** beyond WDR's single α point: Meta-v2 (E2) and UTRAP-v2 (E4) remain **unrun**, and the three **released** reference implementations remain **unexecuted** (E7)" | :246-247 | E2 → `FINDING78`; E4 → `FINDING76`; E7 → R1 (IMMA upstream, 5000 steps + eval) and R4b/c/d (UCE upstream `executed: true`) | **three false sub-claims in one sentence** | **1 CONFIRMED DEFECT** |
| Q24 | §5 "Pre-2026-10-24 checkpoints were deleted ... the registry records `ABSENT`" | :170-171 | registry: **234 ABSENT**, 94 PRESENT, 7 with no checkpoint field, of 335 | mechanism confirmed; 70% of builds are not weight-verifiable | **3 VALID + 4** |
| Q25 | §5 "**H is 15 images**, 3 prompts × 5" | :155-157 | `atk_E2G_*.json` `H_prompts` = 3 entries, `hparams.eval_n` = 5 | exact | **3 VALID** |
| Q26 | §4 "GIFT β∈{1.0…1.35} target range ... **0.2000–0.4667**" | :142 | registry `gift_*` builds; the SD-1.5/BK-SDM builds on disk are `b1.0` (0.4000 / 0.3333) | range endpoints not reproducible from the two cross-model builds alone; needs the full β sweep | **2 SUSPECTED** |

### 1.3 `UPSTREAM_FIDELITY_MANIFEST.md`

| # | claim | doc:line | source | verified | verdict |
|---|---|---|---|---|---|
| F1 | §0 commit/licence/py-file table (5 rows) | :19-23 | `outputs/upstream_manifest.json` | all commits, dates and `python_files` exact | **3 VALID** |
| F2 | §0 "vision_relearning — **README declares MIT** (no standalone LICENSE file)" | :21 | `ls -a upstream/vision_relearning` → **no LICENSE/COPYING**; `upstream/vision_relearning/README.md:54-56` → `## License` / `MIT` | exactly as stated | **3 VALID** (task item 4c confirmed) |
| F3 | §7(c) "**Meta-Unlearning** has neither a LICENSE file nor a README licence declaration" | :306-307 | `ls -a upstream/Meta-Unlearning` → none; `grep -i licen README.md` → **no match** | exact | **3 VALID** (item 4d confirmed) |
| F4 | §2 / §7(a) `hrm` = forget, `irt` = unrelated retain, `tgt` = related retain; §7(b) README says paper used `fix_timesteps = False` | :117, :121, :285-300 | `upstream/Meta-Unlearning/README.md:18` and `:43`, quoted verbatim | both quotations are accurate to the source | **3 VALID** (items 4a, 4b confirmed) |
| F5 | §6 prefix table: "`REF_*` = **official reproduction** — authors' code, **authors' protocol**" | :252 | `logs/r1c_eval.log:20-33`: `REF_imma_vangogh` scored by **our** `gen_eval` on **our** 12-label closed-set CLIP benchmark and **our** DEV prompts; its 100 training images were **self-generated** (`upstream_runs/imma/data/vangogh`, 11:39–11:43) | authors' **code** yes; authors' **protocol** no — the measurement is ours | **1 CONFIRMED DEFECT** (task item 5) |
| F6 | §4 UCE "**Not yet reconciled against the official code** ... queued as **R4**"; "our UCE numbers are a **project reimplementation**" | :182-184 | r4b/r4c/r4d | R4 **ran**; the reimplementation label survives (and strengthens) | **1** stale status, **3** correct label |
| F7 | §3 "Native-domain sanity check ... Queued in §6 as **R3**, not yet run" | :168 | no R3 artifact | accurate | **3 VALID** |
| F8 | §6 "**Status: no R-experiment has been run.** §7 is written only when cells complete." | :277 | R1 complete; R4 four phases; R2 pools | **false for 2 of 4 cells** | **1 CONFIRMED DEFECT** |
| F9 | §1.2 scope table (859.5M / 93.5M / 49.6M / 44.0M / 2.13×, 160 and 80 tensors) | :78-83 | `outputs/imma_scope_fidelity.json` | every figure exact incl. tensor counts | **3 VALID** |
| F10 | §7(d) upstream selects **one** embedding at `attention_mask.sum()-2`; ours selects `sum()-2 .. 76` | :312-316 | corroborated independently by `outputs/r4b_uce_lamb*.json` column **B**: with upstream's selection, port-vs-upstream max rel err falls to **1.5e-3** (λ=0.1) from **0.384** | the selection difference is the dominant divergence | **3 VALID**, now **measured** |
| F11 | §3 "at α=1 the term's gradient norm is exactly **1/√N = 3.411e-5** against a retain gradient of ~0.137 ... Parity is α ≈ 4.0e3" | :160-163 | sourced to `FINDING67`; `outputs/wdr2_wdr2_church_a{1.0,4000}.json` exist but I did not find a JSON recomputing 3.411e-5 | **2 SUSPECTED** — single-sourced to a finding document |
| F12 | §5.1 verbatim env failure text (torch 2.14.1, hub 1.33.0, two tracebacks) | :213-221 | `logs/upstream_envs.log` | consistent | **3 VALID** |
| F13 | §0.1 "no upstream source is committed ... `upstream/` is in `.gitignore`" | :33 | `.gitignore` (10 bytes) | content is `upstream/`-scoped; consistent | **3 VALID** |
| F14 | Manifest **markdown** says vision_relearning licence = "README declares MIT"; the **JSON** says `"license": "NONE FOUND"` | :21 vs `outputs/upstream_manifest.json` | the two companions disagree | **1 CONFIRMED DEFECT** in an artifact I do not own — reported, see §4 |
| F15 | `upstream_manifest.json` records `"requested_pin": null` with `"pinned_as_requested": true` for UCE and FRAG | `outputs/upstream_manifest.json` | nothing was requested, so the flag asserts a vacuous truth | **2 SUSPECTED** schema defect — reported, not owned |

### 1.4 `TABLES_v1_generated.md`

| # | claim | doc:line | source | verified | verdict |
|---|---|---|---|---|---|
| T1 | §0 ten-row count block | :7-18 | `outputs/registry_v1.json` `counts` | **all ten exact** | **3 VALID vs registry** |
| T2 | "Every number below is computed from the raw records" | :3 | registry mtime 05:11 vs 150 newer JSONs incl. 112 `atk_*` | true of the registry, **not of the repository** | **1** — needs a staleness banner |
| T3 | §1 `utrap_church_l0.15k8` rows; §2 as-is row 0.0000/0.8667/0.8364 | :~, :tail | registry spot-checks | consistent | **3 VALID** |
| T4 | §2 rows `advunlearn_{official,parachute,tench,vangogh,garbage_truck}`, `base_sd14` print **—/—/—** | :406-411 | `outputs/gen_advunlearn_*.json`, `gen_base_sd14.json`: `unet: null`, `hparams.target: null`, no `active_target_metrics` | the dashes are **honest**; the underlying records are unattributable | **4 HISTORICAL RESULT** — label, do not delete |
| T5 | §1 rows `h63_church_a0.3` / `h63_van_gogh_a0.3`: **9 attacks each, max H 1.0000** | :119-120 | `outputs/gen_h63_*_a0.3.json`: `hparams.target: null`; registry checkpoint `state: ABSENT` | numbers real; the **scored build's target and weights are both unrecorded** | **1 + 4** — the one table row that cites a provenance-gap build *with numbers* |
| T6 | §3 names all 9 vacuous run_ids and says the 136 VOID are not listed | :535-543 | registry | exact | **3 VALID** |

### 1.5 Provenance gap in `gen_*.json` (coordinator-raised, independently verified)

Verified by reading all **346** `gen_*.json` records:

| condition | count | records |
|---|---:|---|
| `unet` is `null` | **7** | `gen_base_sd14`, `gen_E6cal_BASE_sd14_on_DEV`, `gen_advunlearn_{official,parachute,tench,vangogh,garbage_truck}` |
| — of which **genuinely base** | 2 | `gen_base_sd14` (1.000 on all 12), `gen_E6cal_BASE_sd14_on_DEV` (1.000 on all 12) |
| — of which **demonstrably scored an edited model** | **5** | `advunlearn_official` (church 0.400, monet 0.933), `advunlearn_parachute` (parachute 0.133), `advunlearn_tench` (tench 0.400, monet 0.867), `advunlearn_vangogh` (van_gogh 0.467, monet 0.933), `advunlearn_garbage_truck` (garbage_truck 0.133) |
| `hparams.target` is `null` **and** no `active_target_metrics` | **7** | the 5 advunlearn records + `gen_base_sd14` + `gen_h63_church_a0.3`, `gen_h63_van_gogh_a0.3` (these two *do* record `unet`) |
| registry `defence_builds` with `unet: None` | **7** | same 7 as row 1 |

**Correction to the coordinator's note:** it lists **four** advunlearn records; there are **five**
(`gen_advunlearn_official.json` is the fifth, church 0.400). And `gen_E6cal_BASE_sd14_on_DEV.json`
has `unet: null` but **does** record `target: 'church'` and a full `active_target_metrics` block, so
it is a base control with recorded provenance, not an unattributable record.

**So `unet: null` is genuinely ambiguous**: in 2 records it means "the unmodified base" and in 5 it
means "an edited checkpoint whose identity was never written down". Newer records
(`gen_REF_imma_vangogh.json`, `gen_meta2_*`) record both `unet` and `active_target_metrics`, so this
is an **older-record defect, not a current one**.

**Published tables that depend on them:** `TABLES_v1_generated.md:406-411` (six rows, printed as
`—/—/—`, so no number is attributed) and `TABLES_v1_generated.md:119-120` + `:463-464`
(`h63_*_a0.3`, which **do** carry 9 attacks and max H 1.0000 in §1 against a build whose target is
unrecorded and whose checkpoint is `ABSENT`). None of the six documents I own cites an
`advunlearn_*` as-is number in prose. **Classification: (4) HISTORICAL RESULT needing
reinterpretation** — the numbers stand, the attribution does not.

**Lead-verified facts adopted:** both base records score **1.000 on all twelve DEV concepts**
(independently reconfirmed here), so the closed-set scorer is sound and base utility is the 1.000
control denominator. R4c and R4d completed.

### 1.6 R4 — what the four phases actually measured

Recorded because three documents I own describe R4 as unrun or unreconciled.

| phase | artifact | finding |
|---|---|---|
| R4 | `r4_uce_*.json` | **`VOID: true`**, self-flagged *"DO NOT CITE"*: inputs char-split (`list('church')`), embedding selection matching neither side, **no upstream code executed**. `FINDING75` is retracted on this basis. |
| R4b | `r4b_uce_lamb{0.1,0.5,1000.0}.json` | upstream **executed** (`executed: true`, shim sets only module-level `device`/`dtype`). A (full pipelines) max rel err **0.384 / 0.322 / 0.108**; **B (upstream's own embedding selection) 1.5e-3 / 3.0e-4 / 1.3e-7**; C (fp32 vs fp64) 1.0e-3 / 2.1e-4 / 3.7e-7. |
| R4c | `r4c_uce_lamb{0.1,0.5}.json` | checkpoints + generations. λ=0.1: BASE 15/15, **UPSTREAM 15/15 (target_acc 1.0)**, OUR_PORT 0/15; label agreement **0/15**. λ=0.5: UPSTREAM 14/15, PORT 2/15; agreement **2/15**. |
| R4d | `r4d_uce_lamb{0.1,0.5}.json` | same with `preserve_concepts: []`; agreement **0/15** and **2/15**, per-matrix max 0.384 / 0.322. |

**Interpretation deliberately withheld.** Upstream's λ=0.1 arm scoring **15/15 target** is consistent
with the upstream edit having been **inert** in that run, which would invalidate the port-fidelity
comparison. Agent 1 is testing this. Per the coordinator, R4 is recorded in the documents as
**"executed; interpretation under verification"** and **no fidelity conclusion is drawn from it**.

---

## 2. Incorrect claims that stood underneath a warning banner

The instruction is that a banner is insufficient: the incorrect sentence itself must be rewritten.
Every one of the six documents I own carried the same two-clause banner (`FINDING76` §3 overstated,
`FINDING77` §2 wrong) while leaving the following wrong text intact beneath it.

| # | document | wrong text under the banner | status |
|---|---|---|---|
| B1 | `UPSTREAM_STATUS_r1.md` §1, §4, §5 | eleven false "not run / blocked / queued" cells (S1–S8, S10) | **5 REPAIRED** — table and §5 rewritten |
| B2 | `UPSTREAM_STATUS_r1.md` §6 | "the UCE comparison **is running**" | **5 REPAIRED** |
| B3 | `T2I_REPORT_r5.md` §1 | "none has been run" (Q17) | **5 REPAIRED** |
| B4 | `T2I_REPORT_r5.md` §3.1 | "all six reaching H = 1.000" (Q8) | **5 REPAIRED** |
| B5 | `T2I_REPORT_r5.md` §3.2 | "2/2, third running" (Q12) | **5 REPAIRED** |
| B6 | `T2I_REPORT_r5.md` §6, §8c | "the mandate is unmet at any budget or learning rate tested" — **conclusion kept**, but it was asserted ahead of the evidence that now supports it (Q18) | **5 REPAIRED** — re-evidenced to `logs/e2h_stronger.log`, with the near-miss stated |
| B7 | `T2I_REPORT_r5.md` §8c | "E2/E4 remain unrun; the three released references remain unexecuted" (Q23) | **5 REPAIRED** |
| B8 | `T2I_REPORT_r5.md` §2 | stale counts presented as current (Q1, Q2, Q6) | **5 REPAIRED** |
| B9 | `UPSTREAM_FIDELITY_MANIFEST.md` §6 | "Status: no R-experiment has been run" (F8) | **5 REPAIRED** |
| B10 | `UPSTREAM_FIDELITY_MANIFEST.md` §6 | `REF_*` = "official reproduction — authors' protocol" (F5) | **5 REPAIRED** |
| B11 | `UPSTREAM_FIDELITY_MANIFEST.md` §4 | "Not yet reconciled against the official code" (F6) | **5 REPAIRED** |
| B12 | `HANDOFF_r5.md` §4 | "the third (seed 0) ... **is running now**" | **5 REPAIRED** |
| B13 | `HANDOFF_r5.md` §5 blocker 5, §6 | "E7 blocked"; "Not one corrected implementation has produced a model"; "the mandate remains unmet" | **5 REPAIRED** |
| B14 | `CORRECTION_REGISTER_r5.md` §B.10 | "**None of these has been run.**" — E1, E2, E4, E6 and E7's IMMA/UCE halves have | **5 REPAIRED** |
| B15 | `CORRECTION_REGISTER_r5.md` §C | "No corrected implementation has been run"; "the three published methods with released code have not been executed"; "The mandate's bar is unmet" | **5 REPAIRED** |
| B16 | `TABLES_v1_generated.md` header | "Every number below is computed from the raw records" (T2) | **5 REPAIRED** — staleness banner added |

**Reported, not repaired** (not mine to edit): `FINDING75` is retracted but
`UPSTREAM_STATUS_r1.md`'s banner never mentioned it; `outputs/upstream_manifest.json` disagrees with
its own manifest on the vision_relearning licence (F14) and carries
`pinned_as_requested: true` against `requested_pin: null` (F15); `outputs/registry_v1.json` is
150 JSON files stale (T2).

---

## 3. Fidelity-label audit (task item 5)

Every "verbatim / official / reproduced" phrasing found in the six documents, and whether it is
earned:

| phrase | where | earned? | disposition |
|---|---|---|---|
| `REF_*` = "official reproduction — authors' code, **authors' protocol**" | manifest §6 | **No.** `REF_imma_vangogh` used upstream code and the authors' released ESD checkpoint, but **our** 100 self-generated training images, **our** `gen_eval`, **our** 12-label closed-set CLIP scorer and **our** DEV prompts | relabelled **"upstream code, our benchmark"**; a third tier added |
| R1 "IMMA **author protocol** → our attack" | status §4, manifest §6 | **Partly.** Author-documented *training* setting; the evaluation is entirely ours | relabelled "author-documented training setting, our evaluation" |
| R2 "run `src/meta.py` with **its own three-pool data**" | manifest §6 | **No.** The pools were built by `scripts/r2_build_meta_pools.py` from **our** prompt lists (`a photo of a church`; cathedral/chapel/basilica; fridge/door/firefly/oven/potted plant/coffee table) — the paper's pools are not in the repo | relabelled "three-pool **structure**, our constructed pools" |
| "**No method is marked reproduced**" | status §10 | contradicted by the `REF_*` definition it relies on | both reconciled |
| UCE "project reimplementation" | manifest §4 | **Yes** — and R4b/c/d strengthen it | kept |
| IMMA arms "**project reimplementation**, not a reproduction" | manifest §1 | **Yes** — 2.13× scope gap measured | kept |
| Meta v2 "**our** corrected implementation, not an official reproduction" | manifest §2 | **Yes** | kept |
| WDR "**domain adaptation**, never a reproduction" | manifest §3 | **Yes** | kept |
| GIFT "**not** described as an official reproduction anywhere" | manifest §0.3 | **Yes** — grep confirms | kept |

**Rule recorded in all four affected documents:** an upstream-code run scored on *our* constructed
benchmark is **not** a reproduction of the paper's reported experiment. Three fidelity tiers now
exist, not two: *official reproduction* (nothing qualifies), *upstream code / our benchmark*
(R1, R4b–d), *our reimplementation or adaptation* (everything else).

---

## 4. Issues in documents I do not own — reported, not edited

1. **`docs/AUDIT_REPAIR_REGISTER_v1.md`** (lead): should record that R1 and R4 completed and that
   R2's dataset blocker is discharged; §10's review gate still refers to R1–R4 as pending.
2. **`FINDING75`** (Agent-owned): correctly retracted, but `UPSTREAM_STATUS_r1.md` §3's UCE
   paragraph was built on R4's VOID numbers and is now repaired on my side; the finding's own
   retraction banner is accurate and needs no change from me.
3. **`FINDING76` / `FINDING77`**: their self-corrections (§7 and §6) are accurate against
   `r4b_uce_lamb*.json` column B, which I verified independently. The banner text replicated into my
   six documents was correct; I have kept it and added the newly-verified items beside it.
4. **`outputs/upstream_manifest.json`**: `vision_relearning.license` says `"NONE FOUND"` where the
   manifest markdown (correctly) says *README declares MIT*. One of the two must change; the
   markdown is right.
5. **`outputs/upstream_manifest.json`**: `pinned_as_requested: true` for UCE and FRAG where
   `requested_pin` is `null`.
6. **`outputs/registry_v1.json`**: **stale by 150 JSON files** (112 `atk_*`, ≥11 new defence
   builds). Regeneration is required before any count is republished. I did **not** run
   `scripts/registry_v1.py` — it sha256s many GB and the project is frozen. Recommended as a lead
   decision, with `TABLES_v1_generated.md` regenerated in the same step.
7. **R2 is mid-flight-but-idle**: pools are built and `[r2pools] done`, but `src/meta.py` was never
   invoked and no process is running. Someone should decide whether R2 resumes under the freeze.
8. **`FINDING78` (lead-repaired, D-008)**: I independently confirmed both halves of the lead's
   correction — `logs/e2h_stronger.log` gives 0.5333 at lr 1e-4/25 steps, and the "first arm / both
   halves" wording appears in **no** document I own (grep over `docs/` and `*.md`; the only other
   hits are `FINDING63`, `PREREG_H64` and `CORRECTION_REGISTER_r5.md:183`, all pre-existing and
   about different arms). `FINDING78` §1's retention figures are artifact-exact and citable.

---

## 5. Classification summary

| class | count | items |
|---|---:|---|
| **(1) CONFIRMED DEFECT** | **23** | S1–S8, S10, S11, S16, Q1, Q2, Q6, Q8, Q12, Q17, Q18b, Q19, Q20, Q21, Q23, F5, F6, F8, F14, T2, T5 |
| **(2) SUSPECTED** | **7** | Q13, Q14, Q16, Q18 (basis), Q26, F11, F15 |
| **(3) VALID** | **31** | S9, S12–S15, Q3–Q5, Q7, Q9–Q11, Q15, Q18, Q22, Q22b, Q24, Q25, F1–F4, F7, F9, F10, F12, F13, T1, T3, T6 |
| **(4) HISTORICAL RESULT needing reinterpretation** | **6** | Q21, Q24, T4, T5, S16, F11 |
| **(5) REPAIR PERFORMED** | **16** | B1–B16 |

Categories overlap by design: a stale status is (1) and its repair is (5); a number that is right
but means something narrower than claimed is (3) and (4). **Q18 sits in both (3) and (2)** — the
mandate conclusion is valid, but it was asserted before the measurement that supports it existed,
which is a process defect rather than a factual one.

### 5.1 Defects registered by the lead during this audit, independently confirmed

| id | claim | my verification |
|---|---|---|
| **`D-008`** | `FINDING78` §3's "first arm to meet both mandate halves at any budget" | **CONFIRMED retracted.** `logs/e2h_stronger.log` gives **0.5333** at lr 1e-4 / 25 steps; max-over-valid is a ratchet. The wording appears in **no** document I own. §1's retention figures (0.9000 vs 0.6250 / 0.7000 / 0.1167) are artifact-exact and citable |
| **`D-016`** | `erase_wdr_v2.py` alternates where the reference **sums** | **CONFIRMED** against `vision_unlearner.py:411,637-682`; recorded in manifest §3 |
| **`D-017`** | `FINDING67`'s mechanism attribution, the ~4,000×, the "third instance" framing, and "at gradient parity" | **CONFIRMED.** Measured step fractions 0.104 / 0.537 / 0.998 give ≈**9.6×**. The derivation ‖∇L‖ = α/√N stands; retain8 **0.5917** and the K1 FAIL stand; the unifying three-instance story does not |

---

## 6. What I could not verify

- **Q16** (`cᵀc` max eigenvalue 85,850) and **F11** (WDR gradient-norm 3.411e-5, parity α ≈ 4.0e3)
  trace only to `ERRATUM_uce_lambda_scale.md` and `FINDING67`. No raw JSON under `outputs/`
  recomputes either. Both are plausible and dimensionally consistent; neither is artifact-backed.
  Recomputing them is CPU-cheap and should be a follow-up.
- **Q26** (GIFT β range 0.2000–0.4667) needs the full β sweep; only the two cross-model `b1.0`
  builds were locatable.
- Whether **upstream UCE's edit was inert** at λ=0.1 (R4c/R4d). Agent 1 owns this; I recorded the
  raw agreement figures and drew no conclusion.
- The **R1 result's meaning**. `REF_imma_vangogh` scores target(van_gogh) **0.8667** with
  all-non-target **0.0424** and legacy_retain8 **0.0333** — near-total collateral collapse. Whether
  that is IMMA behaving as designed (it immunises rather than erases, and was composed on top of an
  ESD-erased UNet), a composition artifact in `r1c_eval`'s `overlaid 160/160 IMMA tensors`, or a
  genuine failure, is **not** something this audit can settle, and I have not asserted any of the
  three in the documents.
