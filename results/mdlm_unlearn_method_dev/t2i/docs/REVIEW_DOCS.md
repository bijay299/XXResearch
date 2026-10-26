# Independent review of the documentation reconciliation

**Reviewer:** independent; did not write any document reviewed here.
**Date:** 2026-10-26 · **Scope:** `STATUS_CURRENT_v2.md`, `T2I_REPORT_r5.md`, `UPSTREAM_STATUS_r1.md`,
`UPSTREAM_FIDELITY_MANIFEST.md`, `HANDOFF_r5.md`, `TABLES_v1_generated.md`,
`CORRECTION_REGISTER_r5.md`, `AUDIT_REPAIR_REGISTER_v1.md`, `FINDING65*`, `FINDING66*`.
**Method:** every number recomputed from `outputs/*.json`, `attacked/`, `adapters/`, `upstream/` and
code with CPU python. No GPU work. Nothing committed. No reviewed document edited.

Verification scripts were run ad hoc; the two non-trivial recomputations are reproduced inline below
so they can be re-run.

---

## Item 1 — ONE AUTHORITATIVE STATUS PER ISSUE

### DEFECT FOUND — twelve register defects have NO verdict anywhere

`STATUS_CURRENT_v2.md:61` scopes itself "**REG/D-001 … REG/D-034**". The register contains **more**:

| register heading | line | in `STATUS_CURRENT_v2.md`? |
|---|---:|---|
| `D-035` — a **passing test** was enforcing a false claim. **Worst defect class found.** | `AUDIT_REPAIR_REGISTER_v1.md:859` | **no verdict** |
| `D-036` — Meta reload guard would certify a no-op as successful training | `:891` | **no verdict** |
| `D-037` — IMMA blank-image filter never executed; header claimed it did | `:914` | **no verdict** |
| `D-038` — Meta trainer would crash *after* taking a GPU | `:929` | **no verdict** |
| `D-039` — two reservation-API gaps | `:954` | **no verdict** |
| `D-040` — G2 declared passed on a guard nothing called | `:967` | **no verdict** |
| `D-041` — review premise for the CVD/UUID defect was false | `:1004` | **no verdict** |
| `D-042` — freeze masking a fail-open allowlist | `:1018` | **no verdict** |
| `D-043` — re-scoring write isolation defeatable | `:1038` | **no verdict** |
| `D-044` — all-DISCREPANT re-score would report `run_status: complete` | `:1053` | **no verdict** |
| `D-045` — builder's own exposure figure wrong; drift test blind to it | `:1067` | **no verdict** |
| `D-046` — **two verification claims rest on weaker evidence than stated** | `:1088` | **no verdict** |
| `D-022b` — six FINAL records can never be re-scored | `:545` | **no verdict** (see item 8) |

`STATUS_CURRENT_v2.md:16` claims "**For every issue** it gives exactly **one** verdict". Thirteen
issues, including the one the register calls the worst defect class found and one that explicitly
challenges other verification claims (`D-046`), have none. The document cannot serve as the single
authoritative status while silently truncating the register at D-034.

### DEFECT FOUND — stale "changes needed" rows (reverse-direction contradiction)

Three of `STATUS_CURRENT_v2.md` §7's fifteen rows report register defects that the register no longer
has (register mtime is **later** than this document's):

| STATUS §7 row | claims | register actually says |
|---|---|---|
| `:388` | §4 quarantine table "**still** gives the reason as composition **unproven**" and "upstream edit **may be inert**" | `AUDIT_REPAIR_REGISTER_v1.md:1189` already reads "composition is **VALID** … the earlier 'composition unproven' reason is **REFUTED and withdrawn**"; `:1190` already reads "**REFUTED and withdrawn**" |
| `:389` | `D-022a` "**still** argues the direction" | `:479-485` already reads "That was wrong and is withdrawn" |
| `:391` | G13 "lists a reject option as gate evidence **without saying it is a diagnostic** rather than a repair, and **without the calibration conditions**" | `:1279` already reads "A reject option is **not** accepted as the evidence — it needs independent calibration and versioning first"; `:719-727` already states the four conditions |

Not a live refuted explanation, but a reader is told the register is wrong where it is right. **Does
not block publication; must be re-verified before anyone acts on §7.**

### VERIFIED

`STATUS_CURRENT_v2.md:65-98` carries exactly one primary verdict for each of REG/D-001…D-034, each
drawn from the stated taxonomy. I found **no** item with two conflicting verdicts, and no verdict
contradicted by the register's *current* content within that range. `D-001/002/013/019` do still read
"Status: OPEN" in the register (`:53`, `:64`, `:372`, `:212`), exactly as §7 reports.

---

## Item 2 — REFUTED EXPLANATIONS MUST NOT STAND AS LIVE

### VERIFIED — both are dead everywhere in the reviewed set

I grepped all ten documents for `composition unproven`, `fuzzy`, `reshap`, `unedited model`,
`editing nothing`, `may be inert`, `writes nothing`.

(a) **IMMA composition.** Every surviving occurrence is an explicit refutation:
`STATUS_CURRENT_v2.md:107-132`, `UPSTREAM_STATUS_r1.md:164-166,269`,
`UPSTREAM_FIDELITY_MANIFEST.md:432-433,575`, `AUDIT_REPAIR_REGISTER_v1.md:154-155,1189`,
`T2I_REPORT_r5.md:3`. The quarantine reason is restated as the inapt closed-set metric plus an
unreported configuration; `REG/D-019`'s ESD checkpoint is named as the live candidate with its
control outstanding.

(b) **Upstream UCE inertness.** Same: `STATUS_CURRENT_v2.md:74,134-162`,
`UPSTREAM_STATUS_r1.md:117,228,257,268`, `UPSTREAM_FIDELITY_MANIFEST.md:252,539`,
`AUDIT_REPAIR_REGISTER_v1.md:1190`. `AUDIT_REPAIR_REGISTER_v1.md:258`'s heading retains the original
hypothesis wording, which is correct for a chronological record and is superseded in its own §4.

Five documents carry an identical status pointer at line 3 naming both refutations. **No instance of
either explanation stands as a current reason.** Clean.

I did not re-measure the 32 `attn2` matrices or the 160/526 tensor comparison (would require loading
checkpoints; borderline against the no-GPU rule and not necessary — the refutations' *direction* is
not what is at risk here).

---

## Item 3 — NO UNSUPPORTED DIRECTIONAL CLAIMS

### VERIFIED — the sanctioned sentence is propagated; all six forbidden claims are withdrawn

The sanctioned sentence appears verbatim at `STATUS_CURRENT_v2.md:189-190`,
`FINDING65…:5-6`, `FINDING66…:5-6`, `CORRECTION_REGISTER_r5.md:408,409`, `HANDOFF_r5.md:140,154`,
`T2I_REPORT_r5.md:336`, `UPSTREAM_STATUS_r1.md:270`, `UPSTREAM_FIDELITY_MANIFEST.md:577`.

All six listed claims are explicitly withdrawn, not merely absent — `STATUS_CURRENT_v2.md:199-205`
(one table row each), `FINDING65…:19-26`, `FINDING66…:20-24`, `CORRECTION_REGISTER_r5.md:396,408,409`,
`T2I_REPORT_r5.md:336,456`, `HANDOFF_r5.md:221`, `AUDIT_REPAIR_REGISTER_v1.md:479-485`. I found **no
surviving directional assertion** in any reviewed document. This item is the strongest part of the
reconciliation.

### DEFECT FOUND — `STATUS_CURRENT_v2.md:184-185` states a false count

> "Post-attack arms have **zero** such classifications, at margins an order of magnitude larger."

Recomputed from `outputs/final_*.json`:

```python
import json, glob, statistics
LABELS=['church','parachute','garbage truck','van gogh painting','cassette player','chain saw',
        'english springer spaniel','french horn','gas pump','golf ball','tench','monet painting']
LAB_KEY=['church','parachute','garbage_truck','van_gogh','cassette_player','chain_saw',
         'english_springer','french_horn','gas_pump','golf_ball','tench','monet']
diff=[k for l,k in zip(LABELS,LAB_KEY) if l!=k]            # 9
for f in sorted(glob.glob('outputs/final_*.json')):
    j=json.load(open(f)); p=j['per_image_predictions']
    h=[x for x in p if x['pred_label'] in diff]
    print(j['label'], f"{len(h)}/{len(p)}", round(statistics.median(x['margin'] for x in h),4) if h else '-')
```

| arm | differing-label wins | median margin |
|---|---:|---:|
| `frp_church_r3_l05` | 7/15 | 0.0051 |
| `uce_sd15_church_lam1000` | 5/15 | 0.0027 |
| `utrap_church_l0.15k8` | 10/15 | 0.0044 |
| `utrap_sd15_church_l0.15k8` | 12/15 | 0.0063 |
| `BASE_sd14_anchor` | 2/15 | 0.0110 |
| **`ATK_uce_sd15_church_lam1000`** | **1/15** | **0.0081** |
| other six arms | 0/15 | — |
| **total** | **37/180** | |

A post-attack arm has **one**, not zero, and its margin **0.0081** is the *same* order as 0.0063 —
not "an order of magnitude larger". `CORRECTION_REGISTER_r5.md:408`'s per-arm list sums to **36**
(7+5+10+12+2) while the same sentence reports 37, so the missing image is the same one. Both
documents also fail to note the **base anchor itself** carries 2 of the 37, which bears directly on
the §3 row about base accuracy.

### CLAIM UNSUPPORTED — `FINDING65…:31` margin range

States "median decision margins of **0.0027–0.0065**". Recomputed maximum across the four pre-attack
arms is **0.0063**. `STATUS_CURRENT_v2.md:392` flags the identical `0.0065` error in
`scripts/gen_eval_final.py:178` but not in `FINDING65` itself.

### Count discipline — VERIFIED

**Nine** differing / three identical (`church`, `parachute`, `tench`) reproduced mechanically from
`scripts/gen_eval.py:48,50`. Propagated as nine in all six documents §3 lists. `gen_eval_final.py`
now reads `list(ge.LAB_KEY)` for identity and `list(ge.LABELS)` for CLIP text (`:181-195`) with
runtime asserts — repair confirmed in code. `AGENT4_EVAL_FINDINGS.md:120-129,681` does still say
"four of twelve", "26 of 180", "0.0048–0.0131", exactly as §7 reports — verified accurate.
**Propagating nine nowhere asserts a direction.** Clean.

---

## Item 4 — NUMBERS RECOMPUTED INDEPENDENTLY

| claim | my recomputation | verdict |
|---|---|---|
| registry covers **696 of 808** attack records, **112 absent** | `registry_v1.json.counts.attack_records_total` = **696**; `glob('outputs/atk_*.json')` = **808**; 808−696 = **112** | **VERIFIED** |
| **1,278 of 1,424** JSONs | `counts.json_files_under_outputs` = **1278**; `glob('outputs/*.json')` = **1424** (1428 recursive) | **VERIFIED** |
| **five** `gen_advunlearn_*` with `unet: null` **and** `target: null` | exactly 5 (`official`, `parachute`, `tench`, `vangogh`, `garbage_truck`); 7 records total have `unet: null`; `gen_E6cal_BASE_sd14_on_DEV.json` does carry `hparams.target = 'church'` and is correctly excluded | **VERIFIED** (but see the DEFECT below) |
| **37 of 180** FINAL images on one of nine differing labels | **37 / 180** | **VERIFIED** |
| `cᵀc` max eigenvalue **3.2420e5**, not 85,850 | reproduced on CPU, `3.2420e+05` | **VERIFIED** |
| **30** duplicate `(arm, seed, lr, steps)` cells disagree **by up to 8/15 images** | 68 cells / 178 rows reproduced; **29** disagree on `H_after` (30 if `gate_valid` is included); **max spread 0.7333 = 11 of 15 images**, not 8/15 | **cells VERIFIED; "8/15" DEFECT FOUND** |

### The eigenvalue, reproduced

```
venv_t2i python, CUDA_VISIBLE_DEVICES="", pinned SD-1.4 @133a221b, erase_uce.py's own
last..76 selection over 'church' + the 10 PRESERVE prompts:

  C shape (826, 768)          max eig(cᵀc) = 3.2420e+05     <- matches to 5 digits
  upstream single-token (11, 768)  max eig = 2.2170e+03     <- documents say 2.2171e+03
```

**VERIFIED** against `STATUS_CURRENT_v2.md:97`, `UPSTREAM_FIDELITY_MANIFEST.md:292-293`,
`ERRATUM_uce_lambda_scale.md:41-46`. Three caveats:

1. **No artifact.** The value is sourced to `AGENT1_UCE_FINDINGS.md:634-635` and to
   `scripts/erase_uce.py`'s `--lam` help text. There is **no `outputs/*.json` recording it**, which
   `UPSTREAM_FIDELITY_MANIFEST.md:541` itself notes and then does not fix. It happens to be correct;
   it is still single-sourced to a document, which is the failure mode this reconciliation exists to
   remove.
2. **Four artifacts still carry 85,850**: `outputs/r4_uce_lamb0.1.json:18`,
   `r4_uce_lamb0.5.json:18`, `r4_uce_erasescale2.json:18`, `r4_uce_official_vs_port.json:18` all
   embed `"…c^T c's max eigenvalue of 85,850"` in `default_mismatch_note`.
   `STATUS_CURRENT_v2.md` §7 does not list them.
3. **"rank 136" is tolerance-dependent and the tolerance is nowhere stated.** `136` reproduces only
   at `σ > 1e-3·σ_max`; at `torch.linalg.matrix_rank` default tolerance **both** systems are rank
   **768**. The conclusion "both systems remain **massively rank-deficient**, so λ regularises null
   directions" (`STATUS_CURRENT_v2.md:97`) therefore rests on an unstated threshold. Separately the
   documented `cond` at λ=0.1 (**3.2420e+06**) is `λ_max/λ`, not the measured condition number
   (**3.3824e+06**) — a 4% idealisation presented as a measurement.

### DEFECT FOUND — the duplicate-cell max spread is understated

Reproduced with key `(defence_label, attack_seed, attack_lr, attack_steps, attack_family, target)`,
which is the only key giving the documented 68 cells / 178 rows:

| scope | my result | documented |
|---|---|---|
| duplicate cells (696 indexed) | **68** | 68 ✓ |
| rows | **178** | 178 ✓ |
| disagreeing on `H_after` | **29** (30 incl. `gate_valid`) | 30 ~✓ |
| **max spread, all disagreeing cells** | **0.7333 = 11/15** | **"up to 8/15"** ✗ |
| max spread, strictest-key subset | **0.5333 = 8/15** (7–8 cells) | 0.5333 = 8/15 ✓ (13 cells) |
| all 178 rows `admissible_for_recovery` | **True** | ✓ |

`0.5333 = 8/15` is correct **only** for the strict-key subset. `STATUS_CURRENT_v2.md:257` makes it
the headline for all thirty ("disagree with each other by up to 8/15 images"), and the same
understatement propagates to `TABLES_v1_generated.md:72-73`, `HANDOFF_r5.md:154,178-179` and
`CORRECTION_REGISTER_r5.md`. The true figure is **11 of 15**.

### DEFECT FOUND — a material fact about the duplicates is disclosed nowhere

**All 29 disagreeing cells belong to a single arm: `BASE(unmodified)`.**

```
Counter(k[0] for k in disagreeing) -> {'BASE(unmodified)': 29}
```

No reviewed document says this. It cuts both ways and both ways matter: the irreproducibility does
**not** reach any defence arm's own cells, *and* the one thing that is irreproducible is the
**attack-on-base control** on which every attack-validity and T-rise argument is anchored. A reader
of §5.1 would reasonably infer the disagreements are spread across defence arms. They are not.

### DEFECT FOUND — §5.2 overstates the provenance loss

`STATUS_CURRENT_v2.md:294-295`: "All five **demonstrably scored an edited model** whose identity was
**never written down**, and the active target is inferable **only from the filename**."

Both halves are false. Every one of the five records a complete, pinned path:

```
gen_advunlearn_official.json        text_encoder = …models--OPTML-Group--AdvUnlearn/snapshots/
                                     bca3b27922b1e70cfe5b858e35beccc2fbc3b5ee/church_unlearned
gen_advunlearn_parachute.json       …/bca3b279…/parachute_unlearned
gen_advunlearn_tench.json           …/bca3b279…/tench_unlearned
gen_advunlearn_vangogh.json         …/bca3b279…/vangogh_unlearned
gen_advunlearn_garbage_truck.json   …/bca3b279…/garbage_truck_unlearned
```

The edited component is identified by **snapshot SHA plus concept subdirectory**, in both the
top-level field and `hparams`. The target is recoverable from the *path*, not merely the filename.
And AdvUnlearn edits the **text encoder**, so `unet: null` is the method's correct behaviour, not a
missing field — classifying these five as records whose "identity was never written down" is a
misdescription of the only provenance gap in the set (the absent `target` key).

---

## Item 5 — FRAG RECORD

### VERIFIED

`STATUS_CURRENT_v2.md:331-371` and `T2I_REPORT_r5.md:273-287`:

- arXiv **2608.25429** recorded as **VERIFIED** and as already this project's FRP source, cited to
  `PREREG_H66_frp_port.md:11`, `PREREG_H64_frag_allocation.md:4,14`,
  `PREREG_H67_frag_predictor_transfer.md:4`, and to every `FRP/FRAG port (2608.25429)` row in
  `TABLES_v1_generated.md`. `github.com/Yi1-Chen/FRAG` named.
- **Artifact-confirmed:** `upstream/FRAG/` contains `README.md` + `assets/{Intro.png, rank.png}` and
  **zero** `.py` files. "Our FRP is a reimplementation from the paper" is supported on disk.
- Overlap confined to **Table 11** (attack validity) and **Table 15** (initial forgetting vs
  recovery), neither inflated to a general priority claim nor minimised. Related prior art listed
  separately (MUSE, TOFU, Łucki, "Erase or Hide?", Carlini §4.14) without conflating it with FRAG.
- **LLM-vs-T2I setting difference stated** in both documents ("FRAG's tables are LLM relearning;
  this project's are T2I diffusion").
- **"publication therefore requires a new defence method" is GONE** and explicitly withdrawn —
  `STATUS_CURRENT_v2.md:357-361`, `T2I_REPORT_r5.md:284-287`,
  `AUDIT_REPAIR_REGISTER_v1.md:1349`. The **claim-by-claim comparison requirement is stated** and
  stated as not done (`STATUS_CURRENT_v2.md:363-366`, `T2I_REPORT_r5.md:283-285`). Neither side of
  the contribution question is foreclosed.

**One limit on this verification:** the arXiv ID↔FRAG identification and the *contents* of Tables 11
and 15 are traceable only to documents. No PDF or capture exists under the project tree, and I have
no network access. The claim is internally consistent and the repository-contents half is
artifact-verified; the bibliographic half is not independently checkable from here.

---

## Item 6 — NO IMPLIED RESIDUAL SAFETY

### VERIFIED

Grepped all reviewed documents for `audit complete/concluded/closed`, `now defensible`,
`safe to`, `can now be cited`, `gate passed`, `repair cycle ran`. Zero hits other than the explicit
denial at `STATUS_CURRENT_v2.md:252`: "**Nothing in this project is more defensible because a repair
cycle ran over it.** The resume gate … is **NOT PASSED**." `:407` repeats that the document "confers
**no** standing on any result".

Both required items are named **OPEN**:

- duplicate-cell discrepancies — `STATUS_CURRENT_v2.md:255,279-280` ("Confirmed defect. Unresolved.
  Additive reporting repair only — no disagreement has been adjudicated and no number changed"),
  `TABLES_v1_generated.md:72-73,634`, `HANDOFF_r5.md:154,178-179`.
- missing-provenance records — `STATUS_CURRENT_v2.md:282,305`, `HANDOFF_r5.md:154,179-180,222`.

Subject to the accuracy defects at items 4 above (8/15, the BASE-only concentration, and the
"identity never written down" overstatement), the *disposition* is correct throughout.

---

## Item 7 — REJECT OPTION

### VERIFIED

`STATUS_CURRENT_v2.md:218-246` — "**A reject option is a diagnostic, not a repair**", verdict "a
potential additional diagnostic. **Not a repair, not implemented, not required reading of any
existing score**", with all five conditions present: (1) calibrated independently of the arms being
judged, circularity named; (2) versioned; (3) assumptions documented; (4) **existing closed-set
scores preserved**, K1/K2 decisions not reopened; (5) **no threshold tuned to favour a result**.
`:228` states it "would **not** correct, replace or rehabilitate any existing number".

Independently present at `CORRECTION_REGISTER_r5.md:214-232` (§B.8a) and
`AUDIT_REPAIR_REGISTER_v1.md:719-727,1279,1288` (G13 already: "**not** accepted as the evidence —
it needs independent calibration and versioning first"; "an unauthorised metric change until then").

`STATUS_CURRENT_v2.md:241-246` also correctly states the symmetric limit — that "broad destruction"
*and* "clean erasure" both require saved images and an independent assessment, neither of which
exists — and that R4c/R4d measured no retain axis at all. Clean, and better than asked for.

---

## Item 8 — FINDING66 TERMINAL STATE

### VERIFIED at source — artifact-confirmed

```
ls attacked/  ->  8 KEEP_* directories only
ls -d attacked/E6*  ->  No such file or directory
adapters/{frp_church_r3_l05,uce_sd15_church_lam1000,utrap_church_l0.15k8,
          utrap_sd15_church_l0.15k8}/unet.pt  ->  all present
```

Six of the twelve FINAL records (`ATK_frp_church_r3_l05`, `ATK_uce_sd15_church_lam1000`,
`…_sd1`, `…_sd6`, `ATK_utrap_church_l0.15k8`, `ATK_utrap_sd15_church_l0.15k8`) point at
`attacked/E6B_*`, `E6C_*`, `E6D_*` — **all absent**. The four defence arms and two base anchors are
regenerable. **6 of 12** confirmed.

`FINDING66…:36-54` and `AUDIT_REPAIR_REGISTER_v1.md:545-570` state this correctly and well: "a
**terminal** state, not a pending one: **no rerun** can close it"; "**fresh analogue, not a
reproduction**"; "**permanently unverifiable** under a corrected scorer"; "**withdrawn**, and not
recoverable by re-measurement of the same arms"; "Status: CONFIRMED, loss is permanent, no rerun can
close it". Root cause attributed to `D-005`/`D-028`. Exemplary.

### DEFECT FOUND — the terminal state is NOT propagated, and is contradicted

`D-022b` appears in **exactly two files**: `FINDING66` and `AUDIT_REPAIR_REGISTER_v1.md`. It is
**absent** from `STATUS_CURRENT_v2.md`, `T2I_REPORT_r5.md`, `HANDOFF_r5.md`,
`CORRECTION_REGISTER_r5.md`, `TABLES_v1_generated.md`, `UPSTREAM_STATUS_r1.md` — verified by grep for
`D-022b|terminal|fresh analogue|not re-verifiable|never be re-scored|E6B|E6C|E6D`: zero hits in all
six. The authoritative status document then states the opposite:

| document | text | contradicted by |
|---|---|---|
| `STATUS_CURRENT_v2.md:86` | "Re-scoring requires regenerating the 180 images (not saved) — **deterministic**, so it is a **pure re-score of the same samples**" | `FINDING65…:34-38` "**not a pure re-score**, and not byte-identical merely because the seeds match. **Determinism is an assumption to be tested**"; `FINDING66…:32-34` "**not a pure re-score**" |
| `STATUS_CURRENT_v2.md:314` | "FINAL re-score … **Required rerun** — **CPU-only** on regenerated images" | 90 of the 180 images have **no checkpoint to regenerate from**; and regenerating the other 90 is diffusion sampling, i.e. not CPU-only |
| `HANDOFF_r5.md:140,154` | "a **CPU-only re-score** of the 18 retained records with `LABELS`" | same |
| `CORRECTION_REGISTER_r5.md:408` | "**Re-scoring is CPU-only on retained artifacts** and is the cheapest outstanding repair" | same |

This is precisely the failure item 8 asks about: a **terminal, unrecoverable** state presented as a
**pending, cheap, CPU-only rerun** in the one document designated as the single authority on current
state, and in the handoff a successor will read first.

### DEFECT FOUND — "18 FINAL records" does not exist

`STATUS_CURRENT_v2.md:86` ("the **18** records predate the repair"),
`CORRECTION_REGISTER_r5.md:386` ("all **18** FINAL records"), `HANDOFF_r5.md:140,154` ("**18**
retained records") against **twelve** on disk:

```
ls outputs/final_*.json | wc -l            -> 12
records with study/split_role == FINAL     -> 12
```

`FINDING66…:40` and `AUDIT_REPAIR_REGISTER_v1.md:545` both say "**twelve**". And
`STATUS_CURRENT_v2.md`'s own §3 uses **180** images = 12 × 15; 18 records would be 270. The
authoritative document carries the wrong count.

### DEFECT FOUND — `FINDING66`'s own title and §6 still assert withdrawn claims

- **Line 1 (the title)**: "… and on SD-1.4 that is **more reliable than the unmodified base model**,
  which scores 13/15." Its own notice at `:28-30` **withdraws** exactly that sub-claim ("rests on a
  **2-image gap** … **withdrawn pending measurement**"). A title that asserts a claim the document
  withdraws twenty-six lines later will be the part that gets quoted.
- **`:171-172`**: "there is now **no sub-ceiling number anywhere** in the FINAL measurements: all
  four arms … reach 15/15." False as written. `final_ATK_uce_sd15_church_lam1000.json` records
  `target_hits = 14/15` and `final_BASE_sd14_anchor.json` records `13/15`. Only the
  **max-over-valid** is 15/15; the sub-ceiling records still exist.

---

## Item 9 — DEFECT-ID COLLISIONS

### DEFECT FOUND — the collision table is wrong in three ways

`STATUS_CURRENT_v2.md:40-53` asserts "**Four** independent numbering schemes reuse the same
identifiers for **unrelated** defects" with columns for the register, Agent 1, Agent 4 and Agent 5.
Measured against the files:

| problem | evidence |
|---|---|
| **The "Agent 1 means" column has no referent.** `AGENT1_UCE_FINDINGS.md` contains **zero** `D-0NN` identifiers — it uses `D-100 … D-160` throughout, a disjoint scheme. The table's sole Agent 1 entry (`D-010` = "upstream wrote an unedited model") **does not exist in that file**. | `grep -cE "D-0[0-9][0-9]" AGENT1_UCE_FINDINGS.md` → **0**; `grep -oE "D-1[0-9][0-9]"` → D-100…D-160 |
| **Two rows list the SAME defect in two columns**, contradicting "unrelated". `D-013`: register `:355` "the freeze marker I created is read by nothing" ≡ `AGENT5_INFRA_FINDINGS.md:109` "`FREEZE_ACTIVE` was read by nothing". `D-010`: register `:258` "upstream UCE may be editing nothing" ≡ the table's own Agent 1 gloss. | same headings, same defect |
| **Two further schemes are omitted.** `AGENT6_PROVENANCE_FINDINGS.md` uses `D-008`, `D-016`, `D-017`, `D-022`, `D-022a` (`:86,270,297,298,299`) — aligned with the lead's meanings, so *not* collisions, but the table has **no Agent 6 column** while `:56` invokes `A6/Q18` as though it did. `AGENT2_BASELINES_FINDINGS.md` uses `D-002`. | greps above |

The one **genuine** collision the table gets right is **Agent 4**, which states its intent openly at
`AGENT4_EVAL_FINDINGS.md:38` — "New defect IDs start at D-013 to avoid collision with D-008–D-012" —
i.e. it deliberately *continued* the lead's scheme, so `A4/D-013…D-017` really do name unrelated
defects (`gen_eval_final.py` label set, precommitment fields, registry `defence_identity`,
`vacuity_determinable`, `integrity` block — all confirmed at `:141,163,354,651-655`). Agent 5's
`D-001…D-026` is the second genuine one.

### VERIFIED — namespacing is applied, with one lapse

Every `D-NNN` in `STATUS_CURRENT_v2.md` §1 and §5 is namespaced `REG/`, and `A5/D-008` is used at
`:323`. **Lapse:** `:383` cites `AGENT6_PROVENANCE_FINDINGS.md §D-022 / §5.1` and `§D-022a` **un-namespaced**,
in the very section that tells readers a bare `D-NNN` is ambiguous. I found **no cross-reference
pointing at the wrong defect** — the namespacing is correct where applied; it is the *table
describing* the collisions that is inaccurate.

---

## What I would block publication on

1. **`STATUS_CURRENT_v2.md` truncates the register at REG/D-034.** Twelve register defects
   (`D-035`–`D-046`) and `D-022b` have **no verdict in the authoritative status document** — among
   them `D-035` ("a passing test was enforcing a false claim — **worst defect class found**") and
   `D-046` ("**two verification claims rest on weaker evidence than stated**"). A document whose
   stated purpose is one verdict per issue cannot ship while silently dropping the register's last
   thirteen entries, one of which impugns other verification claims.

2. **The terminal loss is presented as a cheap pending rerun.** Six of twelve FINAL records can
   **never** be re-scored (artifact-confirmed: `attacked/E6B_*`, `E6C_*`, `E6D_*` absent; only 8
   `KEEP_*` survive). `FINDING66` and the register say so plainly; `STATUS_CURRENT_v2.md:86,314`,
   `HANDOFF_r5.md:140,154` and `CORRECTION_REGISTER_r5.md:408` instead call it a "pure re-score",
   "deterministic", "**CPU-only**" — directly contradicting `FINDING65…:34-38`. Every document a
   successor reads first understates a permanent evidentiary loss.

3. **`FINDING66`'s title asserts a claim the document withdraws** (`:1` vs `:28-30`), and `:171-172`
   claims "no sub-ceiling number anywhere in the FINAL measurements" when `14/15` and `13/15`
   records exist on disk.

4. **Two load-bearing counts are wrong in the authoritative document.** "18 FINAL records" (there
   are **12**; 12×15=180 matches its own §3) and "post-attack arms have **zero** differing-label
   classifications, at margins an order of magnitude larger" (there is **one**, at margin
   **0.0081** vs 0.0063 — same order). `CORRECTION_REGISTER_r5.md:408`'s per-arm list sums to 36
   while reporting 37.

5. **The duplicate-cell disagreement is understated and its concentration undisclosed.** Maximum
   spread is **0.7333 = 11 of 15 images**, not 8/15 (8/15 is the strict-key subset only), and **all
   29 disagreeing cells are the single arm `BASE(unmodified)`** — the attack-on-base control that
   anchors every T-rise and attack-validity argument. Stated nowhere.

6. **The `cᵀc` eigenvalue has no artifact.** `3.2420e+05` is correct — I reproduced it exactly on
   CPU — but it is sourced only to `AGENT1_UCE_FINDINGS.md:634-635` and a help string, which is the
   single-sourcing this reconciliation exists to eliminate. Four live artifacts
   (`outputs/r4_uce_{lamb0.1,lamb0.5,erasescale2,official_vs_port}.json:18`) still embed the
   withdrawn **85,850**, and §7 does not list them. The companion "rank 136 / massively
   rank-deficient" claim holds only at an unstated `1e-3·σ_max` tolerance (default tolerance: 768
   for both systems), and the published `cond` at λ=0.1 is `λ_max/λ`, not the measured 3.3824e+06.

7. **The defect-ID collision table is itself inaccurate** (`:40-53`): Agent 1 has **zero** `D-0NN`
   identifiers so its column is spurious; `D-010` and `D-013` list the **same** defect in two
   columns while the header says "unrelated"; Agent 6 (`D-008/016/017/022/022a`) and Agent 2
   (`D-002`) are omitted. Since this table is offered as the key for resolving every `D-NNN`
   citation in the corpus, an inaccurate key is worse than none.

**Not blocking, but fix before anyone acts on it:** `STATUS_CURRENT_v2.md` §7 rows `:388`, `:389`,
`:391` report register defects the register has already corrected (`:1189-1190`, `:479-485`,
`:1279`); `FINDING65…:31`'s `0.0065` should be `0.0063`; `:383` cites Agent 6's `D-022`
un-namespaced; `STATUS_CURRENT_v2.md:294-295` misdescribes the five `advunlearn` records, whose
text-encoder identity **is** fully pinned by snapshot SHA `bca3b279…` plus concept subdirectory and
whose `unet: null` is correct for a text-encoder method.

**What is genuinely sound, and should not be weakened in fixing the above:** both refuted
explanations are dead everywhere (item 2); the label-text direction discipline is thorough and
nowhere violated (item 3); the FRAG record is factual, correctly hedged on setting, and the "new
defence method" conclusion is properly withdrawn with the claim-by-claim requirement stated (item 5);
there is no residual-safety language anywhere and the resume gate is stated NOT PASSED (item 6); the
reject-option treatment carries all five conditions in three independent documents (item 7); and
`FINDING66`'s and the register's own statements of the terminal loss (item 8) are models of the kind
— the defect is that nothing else repeats them.
