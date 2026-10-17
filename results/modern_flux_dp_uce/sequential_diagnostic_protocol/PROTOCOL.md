# Sequential-erasure diagnostic protocol — chain saw -> parachute (FLUX.1-schnell)

**Status: PROTOCOL ONLY, REVISED (revision 02).** Nothing in this document
has been executed. No model was loaded, no checkpoint tensor was loaded, no
encoder was run, no feature was computed, no checkpoint was edited, no image
was generated, classified or annotated, and no GPU was allocated to produce
this document. Existing checkpoint files were hashed at the file level only
(no tensor loading) to pin their identity — see `## 2`. Every other number
below is read from, or CPU arithmetic performed on, already-published
tables. Execution requires a separate future instruction.

Revision 01 corrected the coordinator's NEEDS_CHANGES review of the first
published version (commit `7e16653`) — see `REVISION_01.md`. **Revision 02
makes targeted decision-rule and numerical corrections to revision 01**
(commit `281a407`) without redesigning the accepted substantive design —
see `REVISION_02.md` for the issue-by-issue mapping. Both prior commits are
preserved unchanged in git history; each revision is a new commit, never an
amend or force-push.

This remains a **separately declared diagnostic**, not a continuation of the
closed cassette-player/chain-saw reproduction and not a reopening of the
closed SD1.x/RECE line. It does not change the frozen 80/100 M0 floor or
75% relative-suppression gate used there, and it does not broaden that
experiment's semantic-equivalence mapping.

## 0. Recovered state

- Repo `bijay299/XXResearch`, branch `research/audit-and-matched-effectiveness-protocol`; revision 01's parent was `7e166534e2fdced7fa0cf034d159d46aae38a925`, and revision 02's parent is `281a4076babdd3fd1d2fd9b0206d17606ad6c511` — nothing was reset, amended, or force-pushed at any point.
- Source: `czhang024/CVPR_DouleProjection` @ `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (already cloned, unmodified, at `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo` — reused, not re-cloned).
- Model: `black-forest-labs/FLUX.1-schnell`, revision `741f7c3ce8b383c54771c7003378a50191e9efe9` (already downloaded and hash-verified; see `../RUN_MANIFEST.md` — reused, not re-downloaded).

## 1. Primary question and rejection criterion

**Primary claim under test** (corrected wording, replacing the prior
"save/reload/serialization" framing):

> Does adding the B edit cause verified historical A relapse relative to its
> saved A-parent, despite native parent-response preservation, while the
> specified deletion and retention requirements hold?

This question is about the **sequential (two-call, parent-loaded) workflow
as a whole**, not about isolating "serialization" or "reload" specifically
as the cause. The native batch-edit comparators defined in `## 6` differ
from the sequential workflow in several ways at once (guide semantics,
preservation-set membership, and — for DP — an extra precision-rounding
step), not in save/reload alone. **They can compare available workflows.
They cannot, by themselves, establish that serialization or reload caused
any observed difference.** Exact parent loading and reload remain separate
**technical-integrity** gates (`## 9`), verified directly from the loaded
checkpoint's weights and the adapter's execution order — not inferred from
a workflow comparison. No dedicated serialization-isolation experiment is
proposed.

**Competing explanations for an observed chain-saw relapse**, checked in
this order:

1. **Technical defect** — the parent-loading adapter failed to load A's
   edited weights before the B-edit's reference/guide/preserve computation,
   the loaded parent's identity does not match the recorded checkpoint hash
   (`## 2`), chain saw was omitted from the preserve list, or a weight-key
   mismatch occurred. Ruled out by the technical-integrity gate, not by the
   historical-outcome numbers.
2. **Evaluator/classifier artifact** — a positive "chain saw present"
   reading, or negative "absent" reading, that does not match the image's
   actual content. Checked by the paired witness-verification rule (`## 11`)
   on every candidate case, not assumed away.
3. **General multi-target interference, not workflow-specific** — the
   native batch-edit comparators (`## 6`) also show chain-saw degradation on
   the same or different cases. If so, the deficit is not specific to the
   two-call sequential workflow — it is shared with the one-call batch
   workflow (which differs from SEQ in guide pooling/pairing, preservation
   membership and precision boundary, not merely in "no reload"). This
   changes the **comparative** conclusion; it does not erase a confirmed SEQ
   relapse observation (`## 10`).
4. **Deletion-strength mismatch, not relapse** — B's sequential suppression
   is weaker than its direct suppression with A's own numbers holding
   steady; a different, already-anticipated tradeoff (`## 9`), not the
   claimed deficit, and not itself proof that any mismatch *caused*
   historical relapse (`## 9`).
5. **Confirmed relapse materially exceeding a qualified comparator
   (`D_min >= 10`, `## 10`) and surviving every other check** — this flags
   a bounded candidate for a later, separately authorized method
   assessment, subject to `## 14`'s precedence rules (in particular, a
   stronger applicable existing baseline succeeding takes precedence over
   any inference that a new method is needed).

**Result that would reject the claim:** zero confirmed relapse witnesses
**and zero ambiguous candidates** (`C=0, U=0` in `## 10`'s authoritative
decision table) on the chain-saw panel after SEQ — this alone can end the
diagnostic without running any comparator (`## 14`, outcome 1). `C=0` with
`U>0` is **inconclusive, not a rejection** (`## 10`).

## 2. Endpoint name mapping and checkpoint provenance

| Old name (completed reproduction) | Checkpoint file (host-only) | sha256 (file-level, hashed for this revision) | New notation | Chain-saw semantic recognition |
|---|---|---|---|---|
| `DP_B` | `runs/DP_B/models/flux_double_proxy_ChainSaw.safetensors` | `d9e3d91c1cb2b544ca8a70353146ce68bdf2d330b795223d01caea62911cfd0a` | **DP A-parent** | M0 100/100 -> parent 0/100 |
| `UCE_B` | `runs/UCE_B/models/flux_UCE_proxy_ChainSaw.safetensors` | `bd21369aed67f69f199d037fce27041da300aab17ba2179f23a9f08f9155ee20` | **UCE A-parent** | M0 100/100 -> parent 0/100 |
| `DP_A` | `runs/DP_A/models/flux_double_proxy_CassettePlayer.safetensors` | *(unrelated to this protocol — cassette player, closed, not reused)* | — | — |
| `UCE_A` | `runs/UCE_A/models/flux_UCE_proxy_CassettePlayer.safetensors` | *(unrelated to this protocol)* | — | — |

These two sha256 values were computed directly from the files on `/data` for
this revision (file-level hashing only, no tensor/safetensors deserialization)
and are the values a future execution phase must match before trusting that
the loaded parent is the recorded one. **Matching the two expected weight
*keys* at load time is a separate, necessary-but-not-sufficient check — it
does not by itself establish that the correct *parent checkpoint* (as
opposed to some other file with the same two key names) was loaded; the
file-hash check above is what establishes parent identity.**

"A" in this protocol = chain saw (historical target), reusing the two
checkpoints above verbatim. "B" = parachute (`## 3`), not yet edited by
either method — no existing checkpoint to reuse for B.

## 3. Selected pair and order

**A = chain saw, guide `stick`** (fixed by assignment). **B = parachute,
guide `cloth`** (reused verbatim from the released
`Flux/scripts/Double_Parachute_flux.sh` single-edit example). **Order:
A-then-B** (erase B from A's already-saved parent) — the only order in
scope; no B-then-A reverse order is proposed.

### Candidate table (independently recomputed from `evidence/recognition_summary.json`, commit `01f8832`)

| B candidate | M0 semantic | DP A-parent sees B | UCE A-parent sees B | Released guide (DP) | Released guide (UCE) | Status |
|---|---:|---:|---:|---|---|---|
| cassette player | 28 | — | — | `box` | `box` | excluded — below the 80/100 floor at M0 itself |
| tench | 42 | — | — | *(none released)* | *(none released)* | excluded — below the 80/100 floor at M0 itself |
| english springer | 89 | 87 | 80 | `cat` | *(none released)* | excluded — UCE A-parent recognition sits exactly on the 80/100 floor (zero margin) |
| golf ball | 100 | 100 | 100 | `sphere` | *(none released)* | **provisionally recognition-qualified**, not selected (see preference note below) |
| church | 96 | 96 | 94 | `factory` | *(none released)* | **provisionally recognition-qualified**, not selected (lower margin than the perfect-100 candidates) |
| gas pump | 100 | 100 | 99 | `booth` | *(none released)* | **provisionally recognition-qualified**, not selected |
| garbage truck | 99 | 99 | 100 | `bus` | *(none released)* | **provisionally recognition-qualified**, not selected |
| **parachute** | **100** | **100** | **100** | **`cloth`** | *(none released)* | **provisionally recognition-qualified — SELECTED** |
| french horn | 100 | 100 | 100 | `drum` | *(none released)* | **provisionally recognition-qualified**, not selected (see preference note below) |

"Provisionally recognition-qualified" means exactly what it says: M0 and
both A-parents recognize the candidate well above the 80/100 floor. It does
**not** mean the candidate has cleared every gate — B's own direct
single-edit suppression is untested for every candidate, parachute
included (`## 9`).

### Selection rule (stated before reading outcome data, applied in this order)

1. **Implementation/guide support**: require a released single-edit example
   providing a genuine guide-concept value. All seven non-floor-failing
   candidates satisfy this via their `Double_<Class>_flux.sh` script — **no
   candidate has a released UCE example** (`Flux/scripts/` contains only two
   UCE scripts: cassette player and chain saw). Applying `UCE_original.py`
   to parachute with guide `"cloth"` is an **extension beyond the released
   UCE examples** — the guide *string* is reused verbatim from DP's
   released script, not invented, but its use with `UCE_original.py` is new.
2. **Robust M0 and parent recognition**: restrict to candidates at 100/100
   on all three existing readings (M0, DP A-parent, UCE A-parent). This
   leaves golf ball, parachute, french horn.
3. **A documented preference, not a proof, for independence from the closed
   SD1.x line**: among the three perfect-recognition candidates, parachute
   was never an erasure or historical target in
   `results/rece_seq_golfball_englishspringer/` (golf ball was the
   historical target throughout that line; french horn was S1's erased
   new-target), while parachute appears only in that line's retention
   panels. **This exclusion of golf ball and french horn is a selection
   preference adopted out of caution, chosen to keep this pair's basis
   visibly separate from a different, closed experiment — it is not a
   claim of statistical independence between concepts, and the original
   assignment did not require it.** No newly generated outcome informed
   this choice, and no new search of the old SD1.x line was performed to
   make it (only a grep of its existing published `README.md`/
   `FINAL_REPORT.md` text) — this is a preference stated plainly, not
   evidence manufactured to justify it after the fact.

Parachute is the candidate selected. No step used expected success/failure
of the *new* chain-saw/parachute pair.

### Provisional eligibility (explicit, not glossed over)

M0 and parent recognition for both targets are established from existing
data (table above). **B's own direct single-edit suppression is untested**
for both methods. This is a real, open gate for the execution phase
(`## 9`, "Direct B suppression"), not assumed to pass. If it fails for a
method, that method's SEQ/BATCH endpoints for this pair do not run, and no
automatic fallback pair is substituted.

## 4. What the source code actually establishes (and what it does not)

Both `Double_proxy.py` and `UCE_original.py` start every edit by loading
`pipe.transformer` fresh from `model_id` and copying its two targeted
modules (`Double_proxy.py:34-54`, `UCE_original.py:15-30`) —
`original_modules` *is* whatever checkpoint was loaded at call time. The
planned sequential implementation's required change is to load the saved
A-parent `.safetensors` (identity pinned by the file hash in `## 2`) into
that freshly-constructed pipeline's `transformer` **before** this
module-copy step, so every downstream reference, guide and preserve
computation is built from A's own edited response, not M0's pristine one.
"Include A in the second edit's native preservation list" means adding the
literal string `"chain saw"` to `preserve_concepts` for the B-edit — already
present by default, since the released `Double_Parachute_flux.sh` example's
own preserve list already names chain saw as one of parachute's 9 untouched
classes (`## 7`).

### DP's preservation: an exact identity, with three named qualifications

`Double_proxy.py:191-233` constrains `ΔW_B` to the left null-space of the
stacked preserve-concept embedding vectors (SVD of `C_pres^T`,
`ΔW = Z @ U_2.T`). **In ideal (infinite-precision) arithmetic**, this gives
`ΔW_B @ c_i = 0` exactly for every preserve-concept vector `c_i`, so with a
loaded parent `W_0` and chain saw in `preserve_concepts`,
`(W_0 + ΔW_B) @ c_{chain saw} = W_0 @ c_{chain saw}` for the isolated string
`"chain saw"`'s own token embeddings. **The native computation departs from
this ideal identity in three concrete, source-confirmed ways:**

1. **SVD rank cutoff** (`Double_proxy.py:216`): `rank_V =
   int((S_svd > 1e-6).sum().item())` — singular values at or below `1e-6`
   are treated as zero. Discarding a *genuinely nonzero* singular value
   this way **lowers the retained rank and therefore enlarges the effective
   null space** relative to exact arithmetic: directions that the exact
   constraint would have held fixed become free, so the update can move
   along weakly-represented preservation directions. It does not shrink the
   null space. (The first published version of this revision stated the
   direction backwards; corrected here — see `REVISION_02.md`.)
2. **Projected-direction skip threshold** (`Double_proxy.py:225-228`):
   `x = U_2.T @ v_i`; if `x_norm_sq < 1e-10`, that token's update is
   **skipped entirely** (with a printed warning), not applied. **A large
   null space does not by itself establish that parachute's own erase-token
   vectors have a usable nonzero projection into it** — whether they do is
   unknown without running the edit, and is explicitly left UNESTABLISHED
   here (see `SOURCE_AND_ASSUMPTION_LEDGER.md`). A skipped update affects
   whether the *erasure* is applied for that token; **it is not itself a
   preservation violation** and must not be logged or reported as one.
3. **Finite-precision fp32 arithmetic throughout, with a final bf16 cast**
   (`Double_proxy.py:215-238`): the SVD, the null-space projection and the
   update are all computed in fp32, then `W_0_f32.to(torch_dtype)` casts the
   result to bf16 at the end of each module's loop (`Double_proxy.py:238`).
   **This cast is not a rounding-error footnote — it can change an exactly
   preserved response into a measurably different one.** Illustrative
   arithmetic example (not a measured FLUX effect; no re-execution was
   performed to produce it — this is CPU bookkeeping on hypothetical
   numbers, verified by hand here):

   > Toy parent weight row `[1, 1]` (exactly representable in bf16, so the
   > parent is identical as-stored and as-evaluated), update
   > `Δw = [3/1024, -3/1024]`, preserved input `c = [1, 1]`. Before casting:
   > `Δw · c = 3/1024 - 3/1024 = 0` — preservation is exact. After bf16
   > rounding of the resulting **child weights** (bf16 spacing near 1.0 is
   > `1/128`; near `0.997` it is `1/256`): `1 + 3/1024 = 1.0029...` rounds
   > to `1.0`; `1 - 3/1024 = 0.9971...` rounds to `255/256`. The rounded
   > child row is `[1, 255/256]`; the **child-minus-parent difference as
   > actually stored** is `[0, -1/256]`, giving a response change of
   > `-1/256` — a small but nonzero, directly-attributable-to-casting
   > deviation from "exact" preservation.
   >
   > Note the residual that matters is this **rounded child minus the
   > parent, both as represented at evaluation and promoted to fp32 for the
   > arithmetic** — rounding `Δw` on its own and applying that is a
   > different (and not equivalent) quantity.

**Protection also only concerns the specific vectors selected as inputs to
the constraint — not the whole generator.** This applies to **both**
conditioning streams the two edited matrices consume, not only the T5
stream: `context_embedder` operates on T5 sequence embeddings (full
self-attention, contextual) and `time_text_embed.text_embedder.linear_1`
operates on the CLIP **pooled** embedding — itself computed over the whole
input sequence, so it too is contextual despite being a single vector per
prompt. Both streams' preserve-constraint inputs are built from encoding
the **isolated** string `"chain saw"` (`pipe.encode_prompt(prompt="chain saw", ...)`
in both scripts), not from the actual evaluation prompt `"Image of chain
saw"`. Because both T5 and CLIP are contextual encoders, the token/pooled
vectors for "chain saw" differ, in general, between the isolated string and
the full prompt — meaning the ideal (and even the finite-precision) identity
above is proved only for the isolated-string input, not for the actual
generation-time conditioning of either stream. Whether it transfers closely
enough to matter in practice is UNESTABLISHED here (no toy bound was
derived for this gap, and none is claimed).

### UCE's preservation: a soft least-squares trade-off, structurally different from DP

`UCE_original.py:85-113` solves `W_new = mat1 @ inverse(mat2)`, where
`mat1`/`mat2` sum weighted outer products over all erase and preserve
concepts plus an `λ·w_old` regularizer anchored to the loaded parent. The
preserve term's target is explicitly `w_old`'s own output for that preserve
concept, so UCE's solve *is* trying to match the parent's chain-saw
response — but only as one weighted term in a single shared least-squares
optimum with the new erase objective (`erase_scale=10`) and the regularizer
(`lamb=0.1`); nothing in the closed form guarantees exact, or even tightly
bounded, preservation. UCE also encodes only the **last** content token per
concept (`UCE_original.py:55-65`), unlike DP's per-token loop under
`replace_indices=all` — both native, pre-existing, unchanged here.

### DP-vs-UCE relapse ranking: explicitly PREDICTION, and explicitly confounded

**Prediction** (unproven): SEQ will show some chain-saw relapse for both
methods, with DP's relapse (if any) smaller than UCE's, motivated by DP's
exact-in-ideal-arithmetic constraint vs. UCE's soft trade-off. **This
ranking cannot be cleanly attributed to "hard vs. soft preservation" alone**
— DP and UCE also differ in native token selection (all tokens vs. last
token only) and in edit/encoder precision pathway (bf16-native-with-fp32-
internal-math for DP vs. fp32-throughout for UCE), both of which are
independently plausible contributors to any observed ranking. If a ranking
is observed, it will be reported as an observation consistent with the
hard/soft distinction, not as isolating that distinction as the cause.

### Preservation-residual logging — specified for a future run, collected now: nothing

If execution is authorized, the implementation should **record measured
residuals with their numerical context — not assert an expected value**.
Per edited module and per preserve-concept token: (a) the pre-cast residual
`‖ΔW_B @ c_i‖`, computed in fp32 from the as-computed update; and (b) the
post-cast residual `‖(W_child - W_parent) @ c_i‖`, where `W_child` and
`W_parent` are both taken **as represented at evaluation** (i.e. after the
native cast, exactly the tensors the generation pipeline would load) and
promoted to fp32 only for this residual arithmetic — casting `ΔW_B` alone
and applying that is a different quantity and must not be substituted.
Each logged residual should be accompanied by its numerical context (the
vector norms involved, the retained rank `rank_V`, and whether that token
hit the `1e-10` skip threshold). **No exact-zero runtime assertion should
be imposed on (a)**: finite-precision fp32 SVD and projection need not
return a bitwise zero residual, and a nonzero small value there is a
measurement to report, not a failure to flag. Skipped-update tokens
(point 2 above) are recorded as skipped erasures, not as preservation
violations. If contextual residuals (point under "Protection also only
concerns...") are examined, the exact inputs must be the frozen evaluation
prompts (`"Image of chain saw"`, `"Image of parachute"`) with their T5 and
CLIP encodings computed once and cached, not re-derived per case. These
logs add no new image endpoints and do not change the 6,000-image ceiling
(`## 7`). **No log, measurement, or computation of this kind was collected
for this protocol** — this section specifies the plan only.

## 5. Native multi-target calls: ordered-with-pooled-guides (DP) vs. simultaneous paired-target solve (UCE) — two different mechanisms, not interchangeable comparators

Both scripts natively accept multiple `;`-separated `edit_concepts` in one
call — already-released capability, no new method. **The two methods'
batch behavior is not mechanistically equivalent, and the first published
version of this protocol mischaracterized DP's.**

### DP: ordered erase loop with *pooled*, not per-target, guides

`Double_proxy.py:153-189`: the outer loop iterates `edit_concepts` in order,
updating `W_0_f32` in place before the next concept's baseline is computed
from the updated matrix (genuinely sequential, even within one call). But
the inner guide-resolution step (`Double_proxy.py:174-189`) rebuilds
`guide_vecs` from the **entire** `guide_concepts` list **for every erase
token of every erase concept** — passing `--edit_concepts "chain
saw;parachute" --guide_concepts "stick;cloth"` does **not** assign `stick`
to chain saw and `cloth` to parachute; it pools `{stick, cloth}` into a
joint projection subspace (`V_g`) used as the proxy target for **both**
erase concepts' tokens (`Double_proxy.py:182-189`, the `len(guide_vecs) > 1`
branch). This is a real behavioral difference from the sequential design
(where each edit call has exactly one single-item `guide_concepts` list),
not merely a different call boundary.

The batch call's `preserve_concepts` also necessarily excludes **both**
erase targets — chain saw is no longer preserved (it is itself being
erased in this call) and parachute is no longer preserved either. This
is a different preservation configuration from SEQ's second edit (which
does preserve chain saw) and from the completed reproduction's original
single edits (each of which preserved the other target) — not a smaller
version of the same preservation question, a different one.

**Precision**: a one-call DP batch edit computes both concepts' updates in
fp32 and casts to bf16 **once**, at the very end. The two-call SEQ sequence
casts to bf16 after edit 1 (A), reloads that bf16 checkpoint, casts back up
to fp32 for edit 2 (B)'s internal math, then casts to bf16 again at the end
— **one extra bf16 rounding step that the batch call does not have.** This
is specific to DP: DP's native edit dtype is bf16 (`RUN_MANIFEST.md`).
UCE's native edit dtype is fp32 throughout (`RUN_MANIFEST.md`), and fp32
safetensors save/reload is lossless, so UCE's two-call SEQ does **not**
have an analogous intermediate-rounding step — this asymmetry is
DP-specific, not shared by UCE.

### UCE: a true simultaneous paired-target solve, with one precision caveat

`UCE_original.py:96-113`: the erase loop uses `zip(edit_concepts,
guide_concepts)`, so `"chain saw;parachute"` paired with
`"stick;cloth"` correctly assigns `stick` to chain saw and `cloth` to
parachute (positional pairing, not pooling). `mat1`/`mat2` are built by
summing weighted outer products over every listed concept and solved
**once** — in exact arithmetic this is order-invariant. **Order invariance
here means: reordering corresponding erase/guide pairs together leaves the
sum unchanged in exact arithmetic — it is not a claim of bitwise
floating-point invariance.** Floating-point summation is not strictly
associative, so a different evaluation order can produce a numerically
different result; **the magnitude of that difference, and whether it is
large enough to matter downstream, is unestablished here** — no bound was
derived and none is claimed in either direction.

UCE's batch call's `preserve_concepts` likewise excludes both chain saw and
parachute (both are erase targets, not preserved) — the same preservation-
configuration difference from SEQ noted for DP above.

### Naming

**The endpoints formerly named `DP_JOINT`/`UCE_JOINT` are renamed
`DP_BATCH`/`UCE_BATCH`** ("native batch-edit comparator") throughout this
revision, to avoid implying either a simultaneous solve for DP (it is not)
or an exact equivalence to UCE's simultaneous solve (UCE's own batch call
differs from its sequential calls in reference weights, preservation-set
membership, and regularization structure, even though it is a single joint
linear solve). No new per-target guide semantics were implemented to "fix"
DP's pooling, and no replacement exact-joint algorithm was constructed —
DP's and UCE's native batch behavior is used exactly as released.

## 6. Changed objectives and precision boundaries (summary table)

| | SEQ (this protocol's central endpoint) | BATCH (comparator) |
|---|---|---|
| **DP guide assignment** | Each call has one single-item `guide_concepts` (`stick` for A, `cloth` for B) — no pooling | Pooled: both erase concepts' tokens projected onto the joint span of `{stick, cloth}` |
| **UCE guide assignment** | Each call has one single-item `guide_concepts` | Paired positionally via `zip` — `stick`<->chain saw, `cloth`<->parachute (correct per-target assignment) |
| **Preservation membership (edit 1, erasing A)** | Preserves parachute (parachute is in chain saw's original 9-class preserve list) | N/A — A and B erased together, neither preserved |
| **Preservation membership (edit 2 / the B step)** | Preserves chain saw (added per `## 4`) | N/A — see above |
| **DP precision** | fp32 math, bf16 cast after edit 1, reload+recast to fp32, fp32 math, bf16 cast after edit 2 (one extra rounding step) | fp32 math throughout both concepts, bf16 cast once at the end |
| **UCE precision** | fp32 math and fp32 save/reload throughout (lossless) | fp32 math throughout, solved once |
| **What a difference between SEQ and BATCH can be attributed to** | — | Any combination of the above, not isolable to "sequential vs. batch" alone without further, unauthorized experimentation |

## 7. Endpoints and budget

### Existing (reused, not regenerated)

| Endpoint | Images | Role here |
|---|---:|---|
| M0 | 1,000 (existing) | B's M0 baseline; A's M0 baseline |
| DP A-parent (`DP_B`, hash in `## 2`) | 1,000 (existing) | A-parent for DP SEQ; B's "parent recognition" baseline |
| UCE A-parent (`UCE_B`, hash in `## 2`) | 1,000 (existing) | A-parent for UCE SEQ; B's "parent recognition" baseline |

No new M0 or chain-saw-parent generation is proposed.

### Proposed (not executed) — primary, required for the central question

Preserve lists below use the **exact released order** from
`Flux/scripts/Double_Parachute_flux.sh` wherever applicable (`## 3`); no
reordering was introduced.

| Endpoint | Depends on | `--preserve_concepts` (exact order) | Other changed arguments | Purpose | Images |
|---|---|---|---|---|---:|
| `DP_B_direct` | pristine M0 (existing) | `"cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn"` (released order, unchanged — chain saw already included as an untouched class) | `--edit_concepts "parachute" --guide_concepts "cloth"` | Direct-B suppression gate; direct-vs-sequential comparator | 1,000 |
| `UCE_B_direct` | pristine M0 (existing) | Same list and order, reused for UCE (no independent released UCE order exists; reuse is the documented deviation) | Same `edit_concepts`/`guide_concepts`; `UCE_original.py` native settings (`erase_scale=10, preserve_scale=1, lamb=0.1`); extension beyond released examples | Same, UCE | 1,000 |
| `DP_SEQ` | `DP_B` checkpoint (existing, hash-pinned `## 2`) loaded as parent | **Identical list and order to `DP_B_direct`** — chain saw was already present in the released order, satisfying "include A in B's preserve list" without reordering | `--edit_concepts "parachute" --guide_concepts "cloth"`; parent-loading adapter inserted before module copy | Central-question endpoint, DP | 1,000 |
| `UCE_SEQ` | `UCE_B` checkpoint (existing, hash-pinned `## 2`) loaded as parent | Identical list/order to `UCE_B_direct` | Same; parent-loading adapter | Central-question endpoint, UCE | 1,000 |

**4 primary endpoints, 4,000 images.**

### Proposed (not executed) — conditional native batch-edit comparators

Triggered **only** per the objective, per-method rule in `## 14` — not run
by default.

| Endpoint | Depends on | `--preserve_concepts` (released 9-class list minus both erase targets, order otherwise unchanged) | Other changed arguments | Purpose | Images |
|---|---|---|---|---|---:|
| `DP_BATCH` | pristine M0 (existing) | `"cassette player;church;english springer;gas pump;tench;garbage truck;golf ball;french horn"` | `--edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth"` in one call — guides pooled per `## 5`, not per-target | Compares the sequential *workflow* against DP's native batch workflow | 1,000 |
| `UCE_BATCH` | pristine M0 (existing) | Same list | `--edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth"`, `UCE_original.py` — guides paired per-target correctly per `## 5` | Compares the sequential workflow against UCE's native simultaneous-solve workflow | 1,000 |

**+2 conditional endpoints, +2,000 images.**

### Hard ceiling

**6 new endpoints, 6,000 new images maximum.** No reverse order, no
additional pairs, no hyperparameter sweep, no exploratory ablation is
proposed by default.

## 8. Cases and settings (reused verbatim, not regenerated)

- Case manifest: `evidence/full_case_manifest.csv`, sha256
  `559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26`.
- **Exactly one literal prompt string per target**: `"Image of chain saw"`
  and `"Image of parachute"` — confirmed directly from
  `SD/data/small_imagenet_prompts.csv`, where all 10 rows for a given class
  share identical prompt text and differ only in `evaluation_seed`. The 100
  images per target come from **100 distinct generation seeds** (10 base
  `evaluation_seed` values x 10 variation offsets), **not 10 distinct
  prompt contexts and not 100 independent prompt contexts of any kind**
  (corrects the prior revision's imprecise "10 CSV-row contexts" framing).
- **Edit-time vs. generation-time sequence length, frozen separately
  though currently equal**: `max_sequence_length` for collecting edit-time
  text embeddings in `Double_proxy.py`/`UCE_original.py` resolves to `256`
  for any `model_id` containing `"schnell"` (source-confirmed); the same
  value, `256`, is also the native default passed at generation time in
  `generate_erased_flux.py`. Both are recorded here as `256` explicitly
  rather than assumed identical because they are set by separate code
  paths.
- Native generation settings carried over unchanged: 512x512, 4 inference
  steps, `guidance_scale=0.0`, bf16 inference, batch size 1,
  `global_seed=42`, `base_seed_offset=0`, per-image seed =
  `evaluation_seed + variation_index`.
- Evaluator: the pinned `Flux/compare_accuracy.py`'s own
  `ErasureAccuracyEvaluator` (ResNet50, `ResNet50_Weights.DEFAULT` =
  `IMAGENET1K_V2`), imported unmodified. Strict exact-label and semantic
  (tape-player/radio-equivalent) scores kept separate, as before.
- Native method precision and token behavior are preserved unchanged for
  both methods (DP: bf16 model/fp32 internal math/`replace_indices=all`;
  UCE: fp32 throughout/last-token-only) — no new per-target guide semantics
  and no replacement algorithm were implemented anywhere in this protocol.

## 9. Prospective gates

| Gate | Definition | Reference endpoint | Status |
|---|---|---|---|
| Technical integrity | Exactly 2 weight keys matched, finite values, correct native dtype (DP bf16, UCE fp32) before the bf16 evaluation cast, verified save/reload, parent-loading adapter confirmed to run *before* module copy, **loaded-parent file hash matches `## 2`** | — | prospective |
| Direct B suppression | `DP_B_direct`/`UCE_B_direct` reduce parachute's own semantic recognition by >=75% relative to M0's 100/100 | M0 | **UNTESTED** |
| Sequential B suppression | `DP_SEQ`/`UCE_SEQ` reduce parachute's semantic recognition by >=75% relative to the A-parent's own parachute recognition (100/100 for both methods) | DP/UCE A-parent | **UNTESTED**, depends on direct B suppression passing first |
| Deletion-strength comparability | Direct and sequential B-suppression counts reported side by side; treated as "comparable" only if both independently clear the 75% gate **and** their absolute counts differ by <=10/100 cases. **Practical rationale**: 10/100 is a round, prospectively chosen working tolerance for calling two deletion strengths close enough that a historical-outcome comparison between them is not obviously confounded by unequal deletion. It is **not** derived from any noise estimate — no repeatability or classifier-noise measurement exists for this panel, and the per-class differences recorded in the completed reproduction (e.g. UCE's -13 on one retained class) were **observed effects of an edit, not noise**, so they do not bound this band and 10 is not claimed to exceed them. Descriptive, not a power-derived or statistical bound. **An out-of-band result is reported as a tradeoff/confound for interpreting the historical-outcome finding — it is not itself evidence that a deletion-strength mismatch *caused* any observed historical relapse.** | `DP_B_direct`/`UCE_B_direct` vs. `DP_SEQ`/`UCE_SEQ` | **UNTESTED** |
| Retention (SEQ vs. M0, cumulative) | Aggregate mean semantic recognition across the 8 untouched classes (all 10 minus chain saw minus parachute) drops <=8 points, and no single class drops more than 15/100 cases, relative to M0 | M0 | **UNTESTED**; bound chosen as a judgment call extrapolating from the completed reproduction's worst single-edit retention cost (UCE: aggregate -3.0, max single-class -13) to allow headroom for two stacked edits — **descriptive, not a proof of acceptable quality; failing this bound alone does not establish "all quality gates clean" and passing it alone does not either** |
| Retention (SEQ vs. its own A-parent, incremental) | Same 8-class aggregate drops <=5 points, no single class drops more than 12/100 cases, specifically attributable to the B-edit on top of A's own already-measured retention cost | DP/UCE A-parent (`DP_B`/`UCE_B`'s own existing retention table) | **UNTESTED**; same descriptive status as above |
| Batch comparator qualification | A `DP_BATCH`/`UCE_BATCH` result is only used to inform the comparative conclusion (`## 1`, explanation 3) if it itself passes: both erased targets' own >=75% suppression, B-deletion comparability against the direct-B baseline (same <=10/100 band), and both retention bounds above. **A comparator that fails any of these is reported as a failed comparator, honestly, and is not silently used to draw a conclusion.** | — | **UNTESTED** |

Passing the suppression gates does **not** by itself establish matched
deletion strength, and passing the retention bounds does **not** by itself
establish "all quality gates clean" — both are explicitly descriptive
criteria, stated as such.

## 10. Explicit rules replacing noise-band language

The prior revision referred to an "M0-vs-parent noise band." **No
repeatability or classifier-noise measurement was ever made; M0 100/100 ->
parent 0/100 is the measured erasure effect of the completed reproduction,
not noise, and that language is withdrawn.** In its place, three distinct
quantities are defined and must be reported separately, never conflated:

1. **Raw classifier parent -> SEQ transitions**: for every chain-saw case,
   the 2x2 table of (present/absent at A-parent) x (present/absent at SEQ)
   — unfiltered, directly from classifier output, no visual verification
   applied yet.
2. **Confirmed detector-discovered relapse witnesses**: the subset of
   quantity 1's "absent at parent -> present at SEQ" cases that survive the
   paired visual-verification rule in `## 11`. This is the operative count
   for the rejection criterion (`## 1`) and for every gate above.
3. **Matched-case SEQ-vs-BATCH transitions, if a comparator runs**: reported
   as **both** of the following, named separately because they are
   different quantities and must not be substituted for one another —
   - **SEQ-only / BATCH-only discordant cases**: cases where one workflow
     shows a confirmed relapse witness and the matched case in the other
     does not, counted and reported in **both** directions separately.
   - **Net witness difference**: `(confirmed SEQ witnesses) -
     (confirmed BATCH witnesses)`, which can be zero even when the
     discordant counts are large, if the two workflows relapse on different
     cases.

   **Discordance alone does not establish that one workflow relapses less
   in total**: swapping *which* cases fail, at the same total count, is not
   a reduction in failure frequency. Material comparative improvement is
   defined only by the `D_min`/`D_max` rule below; discordance counts are
   preserved as complementary information and cannot independently support
   a "materially less relapse" claim.

**Existence vs. meaningful aggregate deficit**: one confirmed witness (from
quantity 2) is sufficient to support an **existence** finding — "relapse
occurs under this mechanism, at least once, on this panel" — and no more.
It is not automatically equated with a meaningful aggregate deficit. A
separate, prospective (not outcome-derived) threshold is fixed now for the
latter: **a confirmed-witness count of >=10 (out of 100 chain-saw cases) is
treated as a meaningful aggregate deficit** for the purposes of `## 14`'s
outcome branches. This is a round, interpretable number chosen for this
study, explicitly **not** derived from any significance or power
calculation (`## 13` prohibits that kind of inference) — a judgment call,
stated as one.

### Authoritative ambiguity-aware decision table

Define, **per endpoint**:

- **C** = confirmed detector-discovered relapse witnesses (`## 11`).
- **U** = ambiguous potential relapse cases under the same frozen
  candidate-selection and verification rules (`## 11`).

The range **`[C, C+U]`** describes the possible confirmed-relapse count on
**that discovered candidate set only**. It is **not** a confidence
interval, **not** a population bound, and **not** a count of total semantic
failures on the panel.

**This table is authoritative and takes precedence over any conflicting
prose elsewhere in this protocol:**

| Condition | Required interpretation |
|---|---|
| `C = 0, U = 0` | No verified detector-discovered relapse; **not** a semantic-absence certificate |
| `C = 0, U > 0` | Inconclusive candidate evidence — **do not call it clean and do not treat relapse as rejected** |
| `C >= 1` | Confirmed existence, with ambiguous cases reported separately |
| `C >= 10` | Practical confirmed-witness threshold met |
| `C + U < 10` | Practical threshold not reached by the current discovered candidate set |
| `C < 10 <= C + U` | Practical-threshold determination **unresolved** |

Existence and practical-threshold findings are applied **separately**: e.g.
`C = 0, U = 2` is inconclusive about existence even though the whole
current candidate set, if every ambiguous case were confirmed, could not
reach ten. A *minority* of ambiguous cases can still leave a threshold
determination unresolved — there is no "ambiguity dominates" or "no clear
majority" criterion in this protocol, and any such wording in an earlier
revision is withdrawn.

Ambiguity is **never** reclassified as negative (or positive) to force a
decision, and candidate-selection rules are not changed to reduce `U`. Raw
classifier transitions (quantity 1) remain reported separately throughout.

### Material comparative improvement (`D_min` / `D_max`)

For a **qualified same-method comparator** (`## 9`'s batch-comparator
qualification gate), with `s` = SEQ and `b` = BATCH:

```
D_min = C_s - (C_b + U_b)
D_max = (C_s + U_s) - C_b
```

`[D_min, D_max]` is a **conservative range for the SEQ-minus-BATCH
confirmed-relapse count difference on the verified candidate set** — again
not a statistical interval. Resolved against the already-chosen ten-case
practical scale:

| Condition | Required interpretation |
|---|---|
| `D_min >= 10` | BATCH shows **materially fewer** verified-candidate relapse cases than SEQ |
| `D_max < 10` | The ten-case comparative criterion is **not met** |
| `D_min < 10 <= D_max` | Comparative threshold **unresolved** |

**Applying the existing ten-case practical scale to this previously
undefined comparative condition is a prospective operational choice — not a
power calculation, and not derived from any observed outcome.**

### Reasoned interpretation examples (hypothetical counts; no simulation, no image inspection)

| Hypothetical counts | Required interpretation |
|---|---|
| SEQ `C=0, U=20` | Inconclusive, **not** clean; ambiguity alone does **not** trigger a batch comparator |
| SEQ `C=9, U=2` | Existence confirmed; practical threshold **unresolved** (`C < 10 <= C+U`) |
| SEQ and BATCH each `C=10, U=0`, on disjoint cases | Discordance exists in both directions; net difference zero; **no material comparative improvement** (`D_min = 0`, `D_max = 0`) |
| SEQ `C=12, U=0`; BATCH `C=1, U=2` | `D_min = 12-3 = 9`, `D_max = 12-1 = 11` -> `[9, 11]`: comparative threshold **unresolved** |
| SEQ `C=10, U=0`; BATCH `C=0, U=0` | `D_min = D_max = 10`: comparative criterion **met**, subject to comparator qualification (`## 9`) and the stronger-baseline precedence rule (`## 14`) |

## 11. Historical outcome and paired witness verification

For every chain-saw case, tabulate quantity 1 from `## 10` (raw parent ->
SEQ transitions), separately reporting:

- Recognized at M0, absent after A's own edit, **present again after B**
  (candidate relapse — subject to verification below).
- Recognized at M0, absent after A's own edit, absent after B (clean).
- Recognized at M0, **still present after A's own edit** (residual A-edit
  failure — carried forward, not counted as relapse if it stays present
  after B too).
- Not recognized at M0 at all (expected empty, chain saw's M0 baseline is
  100/100; reported if not empty).

### Paired witness verification (frozen rule, to be applied during a future execution; no image has been inspected to write this rule)

For **every** candidate-positive case — the **union** of candidate positives
across every workflow actually compared (SEQ, and BATCH if run, so
verification is not asymmetric in favor of or against either workflow) —
inspect the **M0, A-parent, SEQ, and (if applicable) BATCH** images for that
case together, and classify into exactly one of:

- **Confirmed relapse**: the A-parent image does not depict a chain saw
  (visually, independent of the classifier label) **and** the SEQ (or
  BATCH) image does.
- **Residual parent content**: the A-parent image already depicts a chain
  saw that the classifier missed — the apparent SEQ/BATCH positive is not a
  new relapse, since the content was already present and undetected.
- **Final classifier false positive**: the SEQ/BATCH image does not
  actually depict a chain saw despite a positive classifier label (e.g., a
  bare chain or a stick alone is not a chain saw — the same caution already
  established by the completed reproduction's own `UCE_B` spot-check).
- **Ambiguous**: neither of the above can be confidently determined from
  the image. Ambiguous cases are counted as **U** (`## 10`), reported
  separately from **C**, and are never reclassified as confirmed or as
  clean in order to force a decision. Their effect on every finding is
  governed by `## 10`'s authoritative decision table (`[C, C+U]`), which
  takes precedence over any conflicting prose: in particular `C=0, U>0` is
  inconclusive rather than clean, and a *minority* of ambiguous cases can
  still leave a practical-threshold or comparative determination
  unresolved.

A case counts as confirmed relapse only with evidence of **both** parent
absence and final presence — a final chain-saw-looking image alone is
insufficient if the parent already had one the classifier missed.

**These are assistant visual checks on a fixed, union-based candidate set —
not independent human validation, not a comprehensive true relapse-rate
estimate, and not an absence certificate for the rest of the panel.** Raw
classifier results (quantity 1) are preserved and reported separately from
verified results (quantity 2) at every stage. **No image has been inspected
and no witness has been labeled to produce this protocol — this section
documents the rule only.**

## 12. Retention and rendering

- Untouched-class aggregate excludes chain saw and parachute (8 classes:
  cassette player, church, gas pump, tench, garbage truck, english
  springer, golf ball, french horn), per-class and aggregate, at M0 /
  A-parent / SEQ / BATCH (if run) — bounds and reference endpoints per
  `## 9`.
- Descriptive rendering panel, frozen now, reusing the **exact same 20
  cases** as the completed reproduction's own fixed panel (variation 0 of
  the first two CSV rows per class: cases 0, 1, 10, 11, 20, 21, ..., 90,
  91), shown across M0, the relevant A-parent, the direct-B edit, SEQ, and
  the batch comparator if available — five columns per case where
  applicable. No image has been selected or viewed to write this rule.
- **Operational rule for quality-confounded or ambiguous results**: a
  rendered image that depicts the stated guide concept (box/stick/cloth,
  as applicable) in place of its original target is expected substitution,
  not damage. A rendered image whose defect (malformation, wrong object,
  low fidelity) is already present in the **matched M0 case** for that same
  case id is an ordinary, pre-existing M0 defect, not newly introduced
  editing damage. Only a defect that is both new (absent at the matched M0
  case) and not explainable as guide substitution is counted as rendering
  damage.
- **This fixed 20-case panel cannot establish a full-panel quality rate or
  substitute for independent human validation** — the same limitation
  already stated for the completed reproduction's own panel, repeated here
  because it applies again. No reuse of the paused human-review
  questionnaire. No new human review is proposed or required by this
  protocol.

## 13. Analysis limits

Each target (chain saw, parachute) has **exactly one literal prompt
string**, evaluated at 100 distinct generation seeds (`## 8`) — not 100,
nor even 10, independent prompt contexts. No IID significance test is
proposed; the comparability and retention bands in `## 9`, and the
meaningful-aggregate-deficit threshold in `## 10`, are fixed descriptive
rules, not inferential ones. No statistical equivalence is inferred from a
null result, and no small-sample FID or CLIP-loss proxy is introduced.

## 14. Outcome branches (every branch stops this bounded diagnostic)

### Outcome precedence (applied in this order; a higher item settles the report before a lower one is considered)

1. **Technical/eligibility failures** (`## 9`'s technical-integrity gate,
   direct/sequential B-suppression gates).
2. **Ambiguity-aware primary findings** — `## 10`'s authoritative decision
   table applied to SEQ's own `C`/`U`.
3. **Comparator conclusions**, only after that comparator passes `## 9`'s
   batch-comparator-qualification gate, resolved by the `D_min`/`D_max`
   rule (`## 10`).
4. **A stronger applicable existing baseline succeeding** — this takes
   precedence over any inference that a new preservation method is needed.

### Branches

1. **`C = 0, U = 0` for SEQ on the chain-saw panel (both methods)** -> no
   verified detector-discovered relapse; **this can end the diagnostic
   without running either batch comparator**, with no follow-up method
   question. Explicitly not a semantic-absence certificate (`## 10`).
2. **`C = 0, U > 0`** -> **inconclusive**, not clean. Report `C`, `U` and
   `[C, C+U]`; do not record relapse as rejected and do not record the
   endpoint as passing a "clean" conclusion. **Ambiguity alone does not
   trigger a batch comparator** (branch 4).
3. **A gate other than historical outcome fails first** (direct B
   suppression, technical integrity) -> report which gate and why, stop;
   that method's remaining endpoints for this pair do not run, no fallback
   pair substituted (precedence item 1).
4. **Objective per-method batch-comparator trigger**: for a given method,
   run `DP_BATCH` or `UCE_BATCH` **if and only if** that method's SEQ shows
   `C >= 1` **and** that method's required gates (`## 9`: technical
   integrity, direct/sequential B suppression, retention) are not already
   failed. Checked and reported per method independently — DP and UCE may
   trigger differently. Ambiguous-only evidence (`C = 0, U > 0`) never
   triggers another endpoint.
5. **`C >= 1` but the practical threshold is not met or is unresolved**
   (`C < 10`, per `## 10`'s table) -> report a bounded **existence**
   finding, stating explicitly whether the practical threshold is *not
   reached* (`C + U < 10`) or *unresolved* (`C < 10 <= C + U`). Not
   generalized to a meaningful aggregate deficit.
6. **Shared SEQ/BATCH relapse**: a triggered, qualified comparator also
   shows confirmed relapse -> this does **not** erase the primary SEQ
   observation, which stands on its own; it changes only the **comparative**
   conclusion, toward "present in both the sequential and the native batch
   workflow" (ties to explanation 3, `## 1`). Resolve the magnitude
   question with `D_min`/`D_max`, never with discordance counts alone.
7. **Comparative criterion met** (`D_min >= 10`, qualified comparator, SEQ
   at `C >= 10`, every other gate clean) -> report that SEQ shows a
   materially larger verified-candidate relapse count than the qualified
   batch comparator. Subject to precedence item 4, **this flags a bounded
   candidate for a later, separately authorized method assessment** — it is
   not a novelty claim, not an established method deficiency, and not an
   automatic follow-up experiment.
8. **Weaker baseline fails, stronger applicable baseline succeeds**: e.g.
   UCE reaches the practical threshold while DP's SEQ does not (`C = 0,
   U = 0`, or `C` below threshold with all other gates clean) -> report as
   a method-specific result for this pair. By precedence item 4, this
   **does not establish a need for a new preservation method**, and is a
   different conclusion from an eligibility or technical failure.
9. **Technical validity fails** (wrong key count, non-finite values, the
   parent-loading adapter verified not to have run before module copy, or
   the loaded-parent file hash not matching `## 2`) -> report the defect,
   stop; no scientific claim is drawn from that endpoint's numbers.

In every branch, `C`, `U`, `[C, C+U]`, the raw classifier transitions
(quantity 1), and both discordance directions are reported separately — no
quantity substitutes for another. No outcome triggers an automatic sweep, a
reverse order, an additional pair, or an unauthorized follow-up run.

## Appendix — proposed commands (reference only, not executed)

Preserve-concept lists below are the exact released order (`## 7`); no
execution adapter was written or run to produce this appendix.

```
# DP_B_direct (new) — from pristine M0, released preserve order unchanged
python Double_proxy.py \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <DP_B_direct>/models --exp_name flux_double_proxy_Parachute

# UCE_B_direct (new) — from pristine M0, same preserve order reused (no independent released UCE order exists)
python UCE_original.py \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --erase_scale 10 --preserve_scale 1 --lamb 0.1 --expand_prompts false \
  --save_dir <UCE_B_direct>/models --exp_name flux_UCE_proxy_Parachute

# DP_SEQ (new) — parent-loading adapter loads runs/DP_B/models/flux_double_proxy_ChainSaw.safetensors
# (file hash d9e3d91c... pinned in ## 2) into the freshly constructed pipeline's transformer BEFORE
# Double_proxy.py's module-copy step; preserve list identical to DP_B_direct (chain saw already present):
python Double_proxy.py \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <DP_SEQ>/models --exp_name flux_double_proxy_ChainSaw_then_Parachute
  # [+ parent-checkpoint load, not a native CLI flag -- the required adapter]

# UCE_SEQ (new) — same adapter pattern, parent = runs/UCE_B/models/flux_UCE_proxy_ChainSaw.safetensors
# (file hash bd21369a... pinned in ## 2)

# DP_BATCH (conditional, per ## 14's trigger) — single call, pristine M0, both targets together.
# Guides are POOLED (## 5), not per-target; preserve list excludes both erase targets:
python Double_proxy.py \
  --edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth" \
  --preserve_concepts "cassette player;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <DP_BATCH>/models --exp_name flux_double_proxy_ChainSaw_Parachute_batch

# UCE_BATCH (conditional, per ## 14's trigger) — single call, pristine M0, true simultaneous
# paired-target solve for UCE (guides correctly paired per-target via zip, ## 5):
python UCE_original.py \
  --edit_concepts "chain saw;parachute" --guide_concepts "stick;cloth" \
  --preserve_concepts "cassette player;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --erase_scale 10 --preserve_scale 1 --lamb 0.1 --expand_prompts false \
  --save_dir <UCE_BATCH>/models --exp_name flux_UCE_proxy_ChainSaw_Parachute_batch
```
