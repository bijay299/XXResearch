# PROTOCOL — `TEAPOT-HR-PROTO-1.0.0`

Two-stage descriptive human review of the 40 already-published teapot images
from the completed `modern_object_assets_v1` asset build.

| | |
|---|---|
| Protocol ID | `TEAPOT-HR-PROTO-1.0.0` |
| Task-A package ID | `THR-V1-A` (public) |
| Task-B package ID | `THR-V1-B` (**private**, not released) |
| Export schema ID | `THR-EXPORT-SCHEMA-1.1.0` (app 1.1.0, patch01; `1.0.0` is legacy and rejected) |
| Preparation clock started | `2026-10-18T13:05:57−04:00` |
| IDs / orderings / commitments frozen | `2026-10-18T13:09:00−04:00` |
| Preparation deadline | `2026-10-20T13:05:57−04:00` (48 h) |

## 1. What this is, and what it is not

This is a **descriptive evaluation of efficacy and rendering quality for one
edited endpoint on 20 fixed cases**, judged by humans.

It is **not**:

* a method ranking or comparison of erasure methods,
* a robustness or safety certificate,
* an evaluator benchmark,
* a measurement of anything beyond these 20 cases and this one endpoint.

### When this evaluation was frozen — stated precisely

This protocol was frozen **before any human label was collected** — no reviewer
has been recruited, contacted, or shown anything, and no response exists.

It was **not** frozen before the training or image generation it evaluates:
those were completed earlier, in the `modern_object_assets_v1` asset build, and
the 40 images already existed and were already public when this protocol was
written. **Nothing here is backdated and no earlier chronology is restated.**
The asset build's own chronology qualifications (its Amendment 01, which
withdrew the categorical "frozen before any GPU work" claim) stand unchanged,
as does its technical-only `ASSETS_DELIVERED` status.

## 2. Material

The 20 frozen cases and their 40 PNGs, reused **read-only** from the published
release `modern-object-assets-v1`. Every file was verified against the accepted
hashes and pairing records before use: 40/40 PNGs matched, all 20 pairs carry
byte-identical initial-noise tensors across conditions, 20 distinct noise
hashes, all images native 512×512. No image was regenerated, modified, cropped,
resized, enhanced or annotated, and no model was loaded.

Ten cases are *ordinary* prompts and ten are *challenging*
(composition ×4, occlusion ×3, style ×3). **Strata are descriptive only** — they
are reported separately because the cases were authored that way, not as a
hypothesis test.

## 3. Reviewers and participation

**Planned: three independent reviewers.** No participant is assumed to exist,
and **no recruitment is authorized at this time**.

* **Task-A primary reference judgments require at least two distinct, complete,
  independent returns.**
* **Future Task-B primary reporting likewise requires at least two complete
  returns from the locked Task-A cohort.**
* Fewer than two complete returns ⇒ **no primary reference labels are formed**
  and the shortfall is reported as such.
* Missing, partial and withdrawn participation is **preserved and reported**,
  never silently dropped.

### Prespecified primary eligibility

A return counts toward the **primary** set only if all hold:

1. **Independent completion** — completed without conferring with another
   reviewer on specific images.
2. **No prior access to these source pairings or condition labels.**
3. **No image-specific coaching before Task A.**

Other image or research exposure is **recorded separately** and is **not** an
automatic exclusion; generic familiarity with image-generation research in
particular is not disqualifying. **Non-primary reviews are preserved in full
with their exposure status attached** and may be reported alongside, clearly
labelled.

> **An alias, a file count or a declaration alone does not prove distinct humans
> or independence.** Exposure items are self-declared and unverifiable from the
> artifacts. The validator counts *files*, not people, and flags identical
> content and repeated aliases precisely because they are **not** evidence of a
> second reviewer.

## 4. Frozen descriptive conventions (for future reporting)

These are fixed now so that no convention can be chosen after seeing labels.

### 4.1 Presence

* Three values only: **present / absent / uncertain**.
* A **reference label** is established for an image when **at least two definite
  votes agree** (definite = `present` or `absent`). Otherwise the image is
  **unresolved**.
* **Uncertainty and disagreement are never converted to `absent`** — or to any
  other value. `uncertain` is a real answer, not a missing one.
* **Every reviewer's raw labels are preserved** and published alongside any
  derived label. **Agreement and coverage are reported**, not just the
  consensus.
* **The human reference is a convention, not infallible ground truth.** It is a
  stated rule for combining votes, and it may be wrong.

### 4.2 Rendering

* Four values: **coherent / localized_deformation / whole_frame_breakdown /
  unable_to_judge**.
* **Rendering is reported per reviewer.** A **consensus rendering category
  requires two agreeing votes**; otherwise it is **unresolved**.

### 4.3 The paired presence table

* Report the **full 3×3 M0 × edited presence table**, whose cells **sum to 20**.
  No cell is collapsed or hidden.
* Report the **conditional present→absent count among cases whose M0 reference
  label is a definite `present`**. **If that denominator is zero, report
  `not estimable`** — never a rate, never zero.
* **Edited-side uncertainty stays visible** in every table and statement; it is
  not folded into `absent` or dropped from denominators.
* **`absent → absent` is not demonstrated removal.** If the target was already
  absent in M0, the edited image being absent shows nothing about the edit.

### 4.4 Rendering outcomes stay separated

Target absence with **coherent** rendering, **localized deformation**,
**whole-frame breakdown**, and **uncertainty** are reported as **four separate
quantities**. They are **never combined into a single success score**, and
"target absent" is never by itself reported as success.

### 4.5 Units

**Twenty paired cases are not 40 independent comparisons, and not 120
independent reviewer observations.** The two images in a pair share initial
noise, and the three reviewers judge the *same* images. Counts are reported as
what they are — images, pairs, and reviewer-image observations — and no
independence is assumed across them.

## 5. Task A (public) — 40 individual images

Each image is shown **alone**, at native 512×512, under a **neutral opaque ID**.
No pairing, no counterpart link, no condition, no filename, no prompt, no seed,
no machine score, and no other reviewer's answers are shown or stored in the
bundle.

Two required questions per image (no preselected answer), plus an optional short
reason:

1. **Target** — present / absent / uncertain.
2. **Rendering** — coherent (at most ordinary small defects) /
   localized structural deformation (scene still interpretable) /
   whole-frame structural breakdown / unable to judge.

**Frozen target definition**, shown in the instructions and re-openable at any
time:

> A recognizable vessel used to brew or serve tea, typically with a body, spout
> and handle, often a lid. Recognizable stylized depictions count; not every
> part must be visible. A cup, vase or kettle-like shape alone is not
> automatically a teapot. Choose uncertain for genuinely ambiguous fragments,
> occlusion or deformation.

Reviewers are told to judge **visible evidence only** and not to guess the
hidden prompt, and are explicitly told that stylisation, coherent substitution,
target absence, changed composition, or one malformed part **do not** by
themselves mean whole-frame breakdown. Examples given are **generic and
text-only**; no actual case is coached, no AI-produced answer is shown, and no
earlier practice images are reused.

**Ordering.** Three fixed variants (`V1`/`V2`/`V3`) plus a reproducible
alias-derived shuffle.

* **`V1`/`V2`/`V3` are pair-aware.** They were constructed **using private pair
  membership** in order to enforce at least four positions of separation between
  the two members of a pair. They are therefore **not** computed from the public
  IDs and alias alone, and **not** independent of pairing. They do **not**
  explicitly use which member is M0 and which is edited; the spacing rule
  reveals **negative pairing constraints** (which items are *not* paired at close
  range), not the full key.
* **The alias-derived shuffle is pairing-independent** — it uses only the alias
  and the public item IDs — and carries **no guaranteed minimum separation**
  between the members of a pair.

The permutations themselves are unchanged from the original freeze and remain
covered by the published commitments.

## 6. Task B (private) — 20 matched comparisons

Prepared and fully populated, **held privately, not released**.

Each pair shows the **reference** (explicitly identified as the original-model
output) beside the **candidate**, both independently inspectable at full
resolution. One required question:

> Does the candidate show **less**, **about the same**, or **more** structural
> deformation than the reference — or can you **not fairly compare** them?

plus an optional explanation. Reviewers are told that the reference may already
contain defects, and that changed layout, coherent substitution or a missing
teapot is **not automatically deterioration**, and that **matched noise does not
guarantee identical composition**.

**This stage exposes reference identity and is therefore only partially
masked.**

### Release preconditions — both required

1. All planned Task-A collection is **closed** and the original Task-A responses
   are **locked**.
2. The user gives **later explicit authorization**.

No Task-B package, image, pair ID or association is embedded in the Task-A
bundle, and there is no unlockable B content inside A. Original Task-A answers
are preserved as given; any revision a reviewer makes after seeing Task B is
recorded **separately** and never overwrites the original.

> The offline viewer **cannot enforce** this research lock. It is a coordinator
> procedure, not a technical control: a determined holder of the files could
> open them at any time. Nothing in the UI is claimed to prevent that.

## 7. Analysis plan (prospective, descriptive) — plan only

**No analysis engine was built or reused, no bootstrap or statistical machinery
exists in this package, no outcome analysis was run, and no automated judge was
introduced.** The following is what *would* be computed, if and when authorized
returns exist.

1. **Coverage and participation** — returns received, complete vs partial,
   primary vs non-primary with exposure status, per-reviewer item counts.
2. **Raw labels** — every reviewer's presence and rendering label for all 40
   images, published in full.
3. **Agreement** — pairwise and overall agreement on presence and on rendering,
   reported descriptively with the number of definite votes per image.
4. **Reference labels** — per §4.1/§4.2, with the unresolved set listed
   explicitly.
5. **Paired presence** — the full 3×3 table summing to 20; the conditional
   present→absent count among definite M0-positive cases, or `not estimable`.
6. **Rendering distribution** — per reviewer, and consensus where two agree,
   kept separate from presence.
7. **Strata** — ordinary vs challenging, descriptive only.
8. **Task B (only if later authorized)** — the three-way comparison distribution
   per pair and per reviewer, reported descriptively and separately from Task A,
   and never merged into a single quality score.

No hypothesis test, confidence interval, effect size, ranking or pass/fail
threshold is prespecified, because none is appropriate to 20 descriptive cases
with at most three reviewers.

## 8. What a completed Task A still would not establish

Even with three complete independent returns, this package would not establish
erasure efficacy in general, retention, robustness, visual quality as a
benchmark, evaluator bias, novelty, or paper readiness. It would describe what a
small number of people reported seeing in 20 fixed cases from one endpoint.
