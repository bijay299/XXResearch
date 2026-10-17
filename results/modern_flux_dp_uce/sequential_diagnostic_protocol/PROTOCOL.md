# Sequential-erasure diagnostic protocol — chain saw -> parachute (FLUX.1-schnell)

**Status: PROTOCOL ONLY. Nothing in this document has been executed.** No
model was loaded, no encoder was run, no feature was computed, no checkpoint
was edited, no image was generated or classified, and no GPU was allocated
to produce this document. Every number below is read from, or arithmetic
performed on, the already-published tables at commit
[`01f8832`](https://github.com/bijay299/XXResearch/blob/01f88326e3e66a2baa60fa9b5cedbdaedb9c7437/results/modern_flux_dp_uce/FINAL_REPORT.md).
Execution requires a separate, future instruction.

This is a **separately declared diagnostic**, not a continuation of the
closed cassette-player/chain-saw reproduction and not a reopening of the
closed SD1.x/RECE line. It does not change the frozen 80/100 M0 floor or
75% relative-suppression gate used there, and it does not broaden that
experiment's semantic-equivalence mapping.

## 0. Recovered state

- Repo `bijay299/XXResearch`, branch `research/audit-and-matched-effectiveness-protocol`, HEAD `01f88326e3e66a2baa60fa9b5cedbdaedb9c7437` at the time this protocol was written — matches the coordinator-verified commit exactly; nothing was reset or overwritten.
- Source: `czhang024/CVPR_DouleProjection` @ `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (already cloned, unmodified, at `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo` — reused, not re-cloned).
- Model: `black-forest-labs/FLUX.1-schnell`, revision `741f7c3ce8b383c54771c7003378a50191e9efe9` (already downloaded and hash-verified; see `../RUN_MANIFEST.md` — reused, not re-downloaded).
- Existing per-image predictions, the frozen 1,000-case manifest, and the completed reproduction's gate computation were read, not regenerated. See `SOURCE_AND_ASSUMPTION_LEDGER.md` and `ELIGIBILITY_AND_ENDPOINTS.json` for exact paths and hashes.

## 1. Question and rejection criterion

**Primary claim under test:** loading an A-erased (chain saw) FLUX.1-schnell
checkpoint as the parent for a second, independent closed-form edit (erasing
B = parachute), with chain saw correctly included in that second edit's
native preservation list and chain saw's own preserved-parent response used
as the reference (not the pristine M0 response), still measurably degrades
chain-saw suppression relative to the untouched A-parent — i.e., native
per-edit preservation of A's *parent* response is not sufficient to prevent
some chain-saw reappearance when a second, unrelated concept is erased
afterward.

**Competing explanations for an observed chain-saw reappearance**, in order
checked:
1. **Technical defect** — the parent-loading adapter failed to actually load
   A's edited weights before the B-edit's reference/guide/preserve
   computation, or chain saw was omitted from the preserve list, or a
   weight-key mismatch occurred. (Ruled out by the technical-integrity gate
   below, not by the historical-outcome numbers.)
2. **Evaluator/classifier artifact** — ResNet50 flags a case "chain saw
   absent" for a reason unrelated to the model's actual depiction (compare
   `FINAL_REPORT.md`'s own UCE spot-check: a metal chain draped over a stick
   was correctly *not* classified as chain saw, which was the right call, not
   an artifact — the risk runs the other way too: a chain-containing image
   could also register a classifier false positive without being a real
   chain saw). Checked by the apparent-positive verification rule in
   `## 6 Retention and rendering` below, not assumed away.
3. **General multi-edit interference, not sequential-specific** — any two
   edits to the same two global weight matrices interfere, regardless of
   whether the second edit is applied to a freshly loaded already-edited
   checkpoint (our SEQ design) or to both concepts at once from pristine M0
   in a single call (the JOINT control, `## 2`). If JOINT shows the same
   degradation as SEQ, the deficit is not specific to the save/reload
   sequential boundary this protocol is named for.
4. **Deletion-strength mismatch, not reappearance** — B's sequential
   suppression is simply weaker than its direct suppression, with no actual
   A-reappearance (A's own numbers hold steady); this is a different,
   already-anticipated tradeoff (see `## 4 Prospective gates`), not the
   claimed deficit.
5. **Bounded deficit specific to the sequential save/reload boundary** — SEQ
   shows A-reappearance beyond what JOINT shows, with technical integrity
   intact and no classifier artifact. This is the only outcome that
   motivates a follow-up method question (`## 8 Outcome branches`).

**Result that would reject the claim:** SEQ's chain-saw semantic recognition,
on the paired A-parent -> after-B cases, returns to within the completed
reproduction's own M0-vs-parent noise band (i.e., chain-saw reappearance in
SEQ is no larger, case-for-case, than in the JOINT control) — explanation 3
or better.

## 2. Endpoint name mapping (old experiment -> this protocol)

| Old name (completed reproduction) | Checkpoint file (host-only) | New notation | Chain-saw semantic recognition |
|---|---|---|---|
| `DP_B` | `runs/DP_B/models/flux_double_proxy_ChainSaw.safetensors` | **DP A-parent** | M0 100/100 -> parent 0/100 |
| `UCE_B` | `runs/UCE_B/models/flux_UCE_proxy_ChainSaw.safetensors` | **UCE A-parent** | M0 100/100 -> parent 0/100 |
| `DP_A` | `runs/DP_A/models/flux_double_proxy_CassettePlayer.safetensors` | *(unrelated to this protocol — cassette player, closed, not reused)* | — |
| `UCE_A` | `runs/UCE_A/models/flux_UCE_proxy_CassettePlayer.safetensors` | *(unrelated to this protocol)* | — |

"A" in this protocol = chain saw (historical target), reusing the two
checkpoints above verbatim. "B" = parachute (selected below, `## 3`), not
yet edited by either method — no existing checkpoint to reuse for B.

## 3. Selected pair and order

**A = chain saw** (fixed by assignment). **B = parachute**, guide = `"cloth"`
(reused verbatim from the released `Flux/scripts/Double_Parachute_flux.sh`
single-edit example). **Order: A-then-B** (erase B from A's already-saved
parent) — the only order in scope; no B-then-A reverse order is proposed.

### Candidate table (verified against `evidence/recognition_summary.json`, commit `01f8832`)

| B candidate | M0 semantic | DP A-parent sees B | UCE A-parent sees B | Released guide (DP) | Released guide (UCE) | Status |
|---|---:|---:|---:|---|---|---|
| cassette player | 28 | — | — | `box` | `box` | **excluded** — below the 80/100 floor at M0 itself |
| tench | 42 | — | — | *(none released)* | *(none released)* | **excluded** — below the 80/100 floor at M0 itself |
| english springer | 89 | 87 | 80 | `cat` | *(none released)* | **excluded** — UCE A-parent recognition sits exactly on the 80/100 floor (zero margin), and english springer was an actual erasure target in the closed SD1.x RECE line (S2) |
| golf ball | 100 | 100 | 100 | `sphere` | *(none released)* | **excluded** — perfect recognition, but golf ball was an actual erasure/historical target throughout the closed SD1.x RECE line; excluded to keep this selection's basis entirely independent of that prior work, per instruction |
| church | 96 | 96 | 94 | `factory` | *(none released)* | eligible, not selected (lower margin than the perfect-100 candidates) |
| gas pump | 100 | 100 | 99 | `booth` | *(none released)* | eligible, not selected |
| garbage truck | 99 | 99 | 100 | `bus` | *(none released)* | eligible, not selected |
| **parachute** | **100** | **100** | **100** | **`cloth`** | *(none released)* | **selected** |
| french horn | 100 | 100 | 100 | `drum` | *(none released)* | eligible, not selected (french horn was an actual erasure target in the closed SD1.x line's S1 sequence — same exclusion logic as golf ball) |

All seven non-floor-failing counts above were independently recomputed from
`evidence/recognition_summary.json` as part of writing this protocol (CPU
arithmetic on an existing table, no new generation or classification) and
match the counts supplied in the assignment exactly.

### Selection rule (stated before reading outcome data, applied in this order)

1. **Implementation/guide support**: require a released single-edit example
   providing a genuine guide-concept value (not invented). All seven
   candidates satisfy this via their `Double_<Class>_flux.sh` script — **no
   candidate has a released UCE example** (`Flux/scripts/` contains only two
   UCE scripts: cassette player and chain saw). This is disclosed, not
   hidden: applying `UCE_original.py` to parachute with guide `"cloth"` is
   an **extension beyond the released UCE examples** — the guide *string*
   itself is not new (reused verbatim from DP's released script), but its
   use with `UCE_original.py` is. `UCE_original.py` accepts `--guide_concepts`
   as a generic argument identical in form to `Double_proxy.py`'s, so this
   is a direct, non-inventive reuse of an existing value in a new (but
   structurally identical) call, not a newly authored guide word.
2. **Robust M0 and parent recognition**: restrict to candidates at 100/100
   on all three existing readings (M0, DP A-parent, UCE A-parent) —
   maximum available margin above the 80/100 floor before any new edit is
   attempted. This leaves golf ball, parachute, french horn.
3. **Independence from the closed SD1.x line**: exclude any candidate that
   was itself an actual erasure or historical target anywhere in the closed
   `rece_seq_golfball_englishspringer/` line (golf ball: historical target
   throughout; french horn: S1's erased new-target;
   `results/README.md`, `history_preservation_confirmation/FINAL_REPORT.md`).
   This is an explicit precaution against the appearance of selecting B
   because of a prior SD1.x finding, not a claim that golf ball or french
   horn are invalid choices on their own merits. **Parachute** was never an
   erasure target in that line — it appears only in SD1.x retention panels
   ("French horn/parachute stayed clean") — and is the sole survivor of
   steps 1-3.

No step used expected success/failure of the *new* chain-saw/parachute pair,
and no newly generated output was consulted.

### Provisional eligibility (explicit, not glossed over)

M0 and parent recognition for both targets are established from existing
data (table above). **B's own direct single-edit suppression is untested**
— no parachute single edit exists yet for either method. This is a real,
open gate for the execution phase (`## 4`, "Direct B suppression"), not
assumed to pass. If it fails for a method, that method's SEQ/JOINT endpoints
for this pair do not run, and no automatic fallback pair is substituted.

## 4. One mechanism-based prediction

**Central question** (as given): does erasing B from an existing A-erased
parent cause A to reappear despite preserving A's parent response, while B
deletion and untouched-class retention remain acceptable?

### What the source code actually proves (not assumed)

Both `Double_proxy.py` and `UCE_original.py` start every edit by loading
`pipe.transformer` fresh from `model_id` and copying its two targeted
modules (`Double_proxy.py:34-54`, `UCE_original.py:15-30`) — `original_modules`
*is* whatever checkpoint was loaded at call time. The planned sequential
implementation's only required change is to load the saved A-parent
`.safetensors` into that freshly-constructed pipeline's `transformer`
**before** this module-copy step (the same two-key merge already used by
`generate_erased_flux.py:67-81` in the completed reproduction), so every
downstream reference, guide and preserve computation is built from A's own
edited response, not M0's pristine one. "Include A in the second edit's
native preservation list" means adding the literal string `"chain saw"` to
`preserve_concepts` for the B-edit.

**DP's preservation is an exact algebraic guarantee, not an approximation**,
*for the isolated preserve-concept string*: `Double_proxy.py:191-233`
constrains `ΔW_B` to the left null-space of the stacked preserve-concept
embedding vectors (SVD of `C_pres^T`, `ΔW = Z @ U_2.T` where `U_2` spans
that null space) — by construction, `ΔW_B @ c_i = 0` exactly for every
preserve-concept token vector `c_i`, including chain saw's, once chain saw
is added to the list. Since `W_0` is already the loaded A-parent weight and
nothing else in the model changes, `(W_0 + ΔW_B) @ c_{chain saw} = W_0 @
c_{chain saw}` **exactly** — the two edited layers' output for the literal
string `"chain saw"`'s own token embeddings is unchanged, to float32
precision, not merely approximately preserved. This is checked against
existing shape records, not re-run: `time_text_embed.text_embedder.linear_1`
has `n=768` and `context_embedder` has `n=4096` (both logged in
`evidence/edit_DP_A.log`, carried over unchanged for this protocol's edits).
Nine preserve concepts (the other 8 untouched classes + chain saw) contribute
at most 9 CLIP vectors (rank <=9 of 768) and at most ~27 T5 token vectors
(rank <=27 of 4096, each class contributing 1-3 tokens per the same log) —
both far short of making the null space collapse to zero dimensions, so the
constraint remains well-posed by the existing shape arithmetic alone.

**UCE's preservation is a soft, weighted least-squares trade-off, not an
exact guarantee** — a real, pre-existing structural asymmetry with DP, not
an artifact introduced by this protocol. `UCE_original.py:85-113` solves
`W_new = mat1 @ inverse(mat2)` where `mat1`/`mat2` sum weighted outer
products over **all** erase and preserve concepts plus an `λ·w_old`
regularization term anchoring toward the loaded parent. The preserve term's
target (`uce_guide_outputs[preserve_concept]`) is explicitly `w_old`'s own
output for that preserve concept — so UCE's solve *is* trying to match A's
parent response for chain saw — but only as one weighted term in a single
joint least-squares optimum shared with the new erase objective
(`erase_scale=10`) and the regularizer (`lamb=0.1`); nothing in the closed
form guarantees `W_new @ c_{chain saw} == w_old @ c_{chain saw}` even
approximately. (UCE also uses only the *last* content token per concept,
`UCE_original.py:55-65`, vs. DP's per-token loop under `replace_indices=all`
— native, pre-existing, not changed here.)

### What is *not* proved (an honest gap, not glossed over)

The exact guarantee above holds for the **isolated** string `"chain saw"`
as encoded by `pipe.encode_prompt(prompt="chain saw", ...)` — precisely how
`preserve_concepts` embeddings are built in both scripts. The actual
evaluation prompt is `"Image of chain saw"` (`SD/data/small_imagenet_prompts.csv`).
FLUX's T5 encoder is contextual (full self-attention over the whole prompt),
so the token embeddings for "chain"/"saw" inside `"Image of chain saw"` are
**not** identical to their embeddings when "chain saw" is encoded alone —
meaning the null-space guarantee does not automatically transfer to the
real generation-time prompt. Whether it transfers closely enough in practice
is an empirical question this protocol does not answer by inspection alone;
it is exactly what SEQ generation (once authorized) would measure. No toy
closed-form argument was constructed here to bound this gap quantitatively
— marked **UNESTABLISHED**, not claimed either way.

### Three-way distinction (as required)

1. **Proved fixed-feature identity**: DP's constrained update leaves the two
   edited matrices' output for the isolated preserve-string "chain saw"
   algebraically unchanged, given correct parent-loading and chain saw's
   inclusion in `preserve_concepts` (exact, from source, `## 4` above).
2. **Possible failure demonstrated by toy mathematics**: **not attempted in
   this protocol.** No closed-form bound was derived showing the
   contextual-embedding gap (isolated vs. in-sentence "chain saw" token
   vectors) is large enough to matter, or that UCE's soft preservation term
   is quantitatively likely to drift past some threshold. This is a real gap
   in the mechanistic argument, stated plainly rather than papered over.
3. **Untested prediction about generated images**: SEQ (both methods) will
   show some chain-saw reappearance beyond the completed reproduction's own
   M0-vs-parent noise floor, with DP's reappearance (if any) smaller than
   UCE's, because DP's preserve constraint is exact at the embedding-input
   level while UCE's is only a soft least-squares trade-off — this is the
   prediction `## 1`'s rejection criterion is built to test, not a settled
   claim.

### Strongest applicable existing control: native ordered/joint multi-concept call

Both scripts natively accept multiple `;`-separated `edit_concepts` in one
call — already-released capability, no new method. **Critically, these are
not mechanistically equivalent between methods**, and must not be
conflated:

- **DP's multi-concept loop is genuinely sequential, even within one call**
  (`Double_proxy.py:153-235`): each concept's update modifies `W_0_f32` in
  place before the next concept's own `b = W_0_f32 @ (...)` baseline is
  computed — so a DP joint call erasing `"chain saw;parachute"` together is
  an **ordered** two-step update from pristine M0, not a simultaneous solve.
  Calling it "simultaneous" would be exactly the error this assignment
  warns against.
- **UCE's multi-concept loop *is* a true simultaneous closed-form solve**
  (`UCE_original.py:96-113`): `mat1`/`mat2` are built by summing weighted
  outer products over every entry in `edit_concepts`/`preserve_concepts`
  (commutative sums) and solved once — order of the `;`-separated list does
  not change the result. A UCE joint call is a genuine simultaneous
  multi-target solve, not merely an ordered pair.

This control directly addresses competing explanation 3 (`## 1`): if DP's
and UCE's **joint** calls (both targets erased together, from pristine M0,
single checkpoint, single generation batch) show chain-saw suppression
degraded to the same degree as SEQ, the deficit is not specific to the
sequential save/reload boundary — it is general multi-target interference,
and UCE_JOINT in particular (a true simultaneous solve) would show whether
even simultaneity removes it. This is source-supported and requires no new
method; it is included **conditionally** (`## 5`), not by default, to keep
the design minimal.

### CEASE and BLOCK (acknowledged, not re-investigated)

**CEASE** ("Continual concept erasure in diffusion models by suppressing
cross-edit interference," `yongliang-wu/CEASE`) targets precisely this
mechanism by name, but as last inspected (`docs/research/feasibility_sources_2026-10-11.json`,
fetched 2026-10-11) its repository contains no method code — only a static
project page ("Code coming soon"). It is cited here as directly relevant
prior art for the predicted mechanism, not as an available control; its
absence of an implementation is unchanged and not re-checked in this
protocol-only task.

**BLOCK** (`proceedings.mlr.press/v306/kumar26c.html`,
`sprintml/block-concept-removal` @ `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5`,
cited in `FINAL_REPORT.md`) already studies sequential removal on modern
models. A bounded deficit found here, if any, would **not** establish that
BLOCK fails, and would not make sequential erasure a new research topic —
it would be a narrow, method-specific (DP/UCE) finding on one concept pair.
No additional model or baseline search is proposed in response to either
citation.

## 5. Endpoints and budget

### Existing (reused, not regenerated)

| Endpoint | Images | Role here |
|---|---:|---|
| M0 | 1,000 (existing) | B's M0 baseline; A's M0 baseline |
| DP A-parent (`DP_B`) | 1,000 (existing) | A-parent for DP SEQ; B's "parent recognition" baseline |
| UCE A-parent (`UCE_B`) | 1,000 (existing) | A-parent for UCE SEQ; B's "parent recognition" baseline |

No new M0 or chain-saw-parent generation is proposed.

### Proposed (not executed) — primary, required for the central question

| Endpoint | Depends on | Changed arguments vs. the released single-edit example | Purpose | Images |
|---|---|---|---|---:|
| `DP_B_direct` | pristine M0 (existing) | `--edit_concepts "parachute" --guide_concepts "cloth"` (preserve list = all 9 other classes, native single-edit settings, `Flux/scripts/Double_Parachute_flux.sh`'s own list, unchanged) | Direct-B suppression gate; direct-vs-sequential comparator | 1,000 |
| `UCE_B_direct` | pristine M0 (existing) | Same `edit_concepts`/`guide_concepts`; `UCE_original.py` native settings (`erase_scale=10, preserve_scale=1, lamb=0.1`); **extension beyond released examples** (no author UCE-parachute script) | Same, UCE | 1,000 |
| `DP_SEQ` | `DP_B` checkpoint (existing) loaded as parent | `--edit_concepts "parachute" --guide_concepts "cloth" --preserve_concepts "cassette player;chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;french horn"` (chain saw added; parent-loading adapter inserted before module copy) | Central-question endpoint, DP | 1,000 |
| `UCE_SEQ` | `UCE_B` checkpoint (existing) loaded as parent | Same preserve-list change; `UCE_original.py` native settings; parent-loading adapter | Central-question endpoint, UCE | 1,000 |

**4 primary endpoints, 4,000 images.**

### Proposed (not executed) — conditional control, only if `DP_SEQ`/`UCE_SEQ` show a chain-saw deficit against explanation 1-2 being ruled out

| Endpoint | Depends on | Changed arguments | Purpose | Images |
|---|---|---|---|---:|
| `DP_JOINT` | pristine M0 (existing) | `--edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth"` in one call, single checkpoint | Rules out general (non-sequential-specific) ordered-multi-edit interference for DP | 1,000 |
| `UCE_JOINT` | pristine M0 (existing) | Same, `UCE_original.py` | Rules out the same for UCE's true simultaneous solve | 1,000 |

**+2 conditional endpoints, +2,000 images.**

### Hard ceiling

**6 new endpoints, 6,000 new images maximum.** No reverse order (B-then-A),
no additional pairs, no hyperparameter sweep, no exploratory ablation is
proposed by default. The only cost beyond the 4 primary endpoints is the
2 JOINT controls, included conditionally and only because `## 4` identifies
them as the strongest applicable existing, source-supported control for
ruling out the leading alternative explanation — not as an open-ended
extension.

## 6. Cases and settings (reused verbatim, not regenerated)

- Case manifest: `evidence/full_case_manifest.csv`, sha256
  `559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26`
  (1,000 cases: 100 CSV rows x 10 variations; same file for every new
  endpoint).
- Native generation settings carried over unchanged: 512x512, 4 inference
  steps, `guidance_scale=0.0`, `max_sequence_length=256`, bf16 inference,
  batch size 1, `global_seed=42`, `base_seed_offset=0`, per-image seed =
  `evaluation_seed + variation_index`.
- Evaluator: the pinned `Flux/compare_accuracy.py`'s own
  `ErasureAccuracyEvaluator` (ResNet50, `ResNet50_Weights.DEFAULT` =
  `IMAGENET1K_V2`), imported unmodified, as in the completed reproduction's
  `evidence/code/eval_all.py`. Strict exact-label and semantic
  (tape-player/radio-equivalent) scores kept separate, as before.

## 7. Prospective gates

| Gate | Definition | Default reference |
|---|---|---|
| Technical integrity | Exactly 2 weight keys matched (`context_embedder.weight`, `time_text_embed.text_embedder.linear_1.weight`), finite values, correct native dtype (DP bf16, UCE fp32) before the bf16 evaluation cast, verified save/reload, parent-loading adapter confirmed to run *before* module copy | Same as the completed reproduction's check |
| Direct B suppression | `DP_B_direct`/`UCE_B_direct` reduce parachute's own semantic recognition by >=75% relative to M0's 100/100 | Original 75% relative-suppression rule, unchanged |
| Sequential B suppression | `DP_SEQ`/`UCE_SEQ` reduce parachute's semantic recognition by >=75% relative to the A-parent's own parachute recognition (100/100 for both methods, from the existing table) | Same rule, applied to the parent baseline instead of M0 |
| Deletion-strength comparability | Direct and sequential B-suppression counts reported side by side; treated as "comparable" only if both independently clear the 75% gate **and** their absolute counts differ by <=10/100 cases — a descriptive band, not a statistical test (see `## 9`) | New criterion for this study, justified here; does not alter the closed experiment's own gates |
| Retention | Each of the 8 untouched classes (all 10 minus chain saw minus parachute), per-class and aggregate, M0 vs. A-parent vs. SEQ vs. (conditionally) JOINT | Same structure as the completed reproduction |

Passing the suppression gates does **not** by itself establish matched
deletion strength — that is the separate, explicit comparability criterion
above.

## 8. Historical outcome (to be tabulated once SEQ exists)

For each of the 100 chain-saw cases, paired across M0 -> A-parent -> after-B
(SEQ), four categories, reported separately per method:

- Recognized at M0, absent after A's own edit, **present again after B**
  (candidate relapse).
- Recognized at M0, absent after A's own edit, **absent after B** (clean).
- Recognized at M0, **still present after A's own edit** (residual A-edit
  failure — carried forward, not counted as relapse if it stays present
  after B too).
- Not recognized at M0 at all (excluded from the relapse question; chain
  saw's own M0 baseline is 100/100 so this category is expected to be
  empty, but will be reported if not).

Residual first-edit failures are explicitly not relapse. The minimum
practically meaningful contrast is frozen now, before any execution: **any
non-zero count in the first category, beyond what JOINT shows on the same
cases, is the candidate finding** — no threshold tuning after seeing the
numbers.

## 9. Retention and rendering

- Untouched-class aggregate excludes chain saw and parachute (8 classes:
  cassette player, church, gas pump, tench, garbage truck, english springer,
  golf ball, french horn), per-class and aggregate, at M0 / A-parent / SEQ /
  (conditionally) JOINT.
- **Apparent-positive verification rule**: any case where the classifier
  labels "chain saw" present after B must be visually inspected against the
  actual image before being counted as a real reappearance — "a chain is
  not a chain saw" (per the completed reproduction's own UCE_B spot-check,
  which showed a chain draped on a stick that the classifier correctly did
  *not* call chain saw). Inspecting the positive witnesses this surfaces
  does **not** establish a full-panel false-negative rate or prove semantic
  absence elsewhere in the panel — it only checks the specific apparent
  positives.
- Descriptive rendering panel, frozen now (not chosen after viewing
  outputs): variation 0 of the first two CSV rows for chain saw (cases 10,
  11) and parachute (cases 80, 81) — 4 cases x each new endpoint. Any
  apparent-positive case outside this fixed panel gets its own full-
  resolution crop, identified by case id, rather than being folded into an
  aggregate "degeneration rate."
- No reuse of the paused human-review questionnaire. No new human review is
  proposed or required by this protocol.

## 10. Analysis limits

The 100 parachute (or chain-saw) cases are 10 prompt-template/seed
combinations across 10 CSV rows, not 100 independent prompt contexts — the
same caveat already stated in `FINAL_REPORT.md`. No IID significance test is
proposed; the comparability band in `## 7` is a fixed descriptive rule, not
an inferential one. No statistical equivalence is inferred from a null
result, and no small-sample FID or CLIP-loss proxy is introduced.

## 11. Outcome branches (every branch stops this diagnostic)

1. **Existing preservation works adequately**: SEQ's chain-saw recognition
   stays within the completed reproduction's own M0-vs-parent noise band for
   both methods -> report a clean result; no follow-up method question.
2. **Only a weak/absent control fails**: a gate other than historical
   outcome fails first (e.g., direct B suppression fails for one method) ->
   report which gate, stop; that method's endpoints for this pair do not
   run further, no fallback pair substituted.
3. **Apparent relapse is explained by deletion, retention, rendering, or
   evaluator tradeoffs**: a positive witness fails the verification rule in
   `## 9`, or the comparability band in `## 7` is violated in a way that
   explains the numbers -> report the explanation, stop.
4. **Technical validity fails**: wrong key count, non-finite values, or the
   parent-loading adapter verified not to have run before module copy ->
   report the defect, stop; no scientific claim is drawn from that
   endpoint's numbers.
5. **A bounded deficit remains against the strongest applicable control**
   (JOINT, if run): SEQ shows chain-saw reappearance beyond what JOINT shows
   on the same cases, with every other gate clean -> **this is the only
   outcome that motivates a subsequent method question** (e.g., whether a
   stronger historical-preservation mechanism than the native one is
   needed) — stated as a question for a future, separately authorized task,
   not an automatic next experiment.

No outcome triggers an automatic sweep, a reverse order, or an additional
pair.

## Appendix — proposed commands (reference only, not executed)

```
# DP_B_direct (new) — from pristine M0
python Double_proxy.py \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <DP_B_direct>/models --exp_name flux_double_proxy_Parachute

# UCE_B_direct (new) — from pristine M0
python UCE_original.py \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;french horn" \
  --concept_type object --erase_scale 10 --preserve_scale 1 --lamb 0.1 --expand_prompts false \
  --save_dir <UCE_B_direct>/models --exp_name flux_UCE_proxy_Parachute

# DP_SEQ (new) — parent-loading adapter loads runs/DP_B/models/flux_double_proxy_ChainSaw.safetensors
# into the freshly constructed pipeline's transformer BEFORE Double_proxy.py's module-copy step,
# then runs with chain saw added to the preserve list:
python Double_proxy.py \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <DP_SEQ>/models --exp_name flux_double_proxy_ChainSaw_then_Parachute
  # [+ parent-checkpoint load, not a native CLI flag -- the required adapter]

# UCE_SEQ (new) — same adapter pattern, UCE_original.py, parent = runs/UCE_B/models/flux_UCE_proxy_ChainSaw.safetensors

# DP_JOINT (conditional) — single call, pristine M0, both targets together (ordered, not simultaneous for DP)
python Double_proxy.py \
  --edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth" \
  --preserve_concepts "cassette player;church;gas pump;tench;garbage truck;english springer;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <DP_JOINT>/models --exp_name flux_double_proxy_ChainSaw_Parachute_joint

# UCE_JOINT (conditional) — single call, pristine M0, true simultaneous solve for UCE
python UCE_original.py \
  --edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth" \
  --preserve_concepts "cassette player;church;gas pump;tench;garbage truck;english springer;golf ball;french horn" \
  --concept_type object --erase_scale 10 --preserve_scale 1 --lamb 0.1 --expand_prompts false \
  --save_dir <UCE_JOINT>/models --exp_name flux_UCE_proxy_ChainSaw_Parachute_joint
```

No execution adapter was written or run to produce this appendix; the
commands document the intended arguments only, mirroring the released
single-edit scripts' own argument style.
