# CLAIM_CORRECTIONS_V2 — corrections to the AUDIT-01 round itself

AUDIT-01 (commit `f7f4ba8`) corrected nine claims in the pilot narrative. Central
review then found that **the audit's own write-up introduced or retained eight
overstatements of its own**. They are corrected here.

No measured value changed. Every number in `f7f4ba8` still re-derives exactly
from the raw predictions. These are corrections to *what the numbers were said
to mean*.

Originals preserved: [`preserved_originals/v1/`](preserved_originals/v1/) and
git `f7f4ba8`. The pre-audit originals remain at
[`preserved_originals/`](preserved_originals/) and git `a9de625`.

---

## V1 — "the two training runs are genuinely independent"

**What AUDIT-01 said** (`AUDIT_REPORT.md` §1, `CORRECTED_TABLES.md` §1):
> "**0** images byte-shared between the two seeds other than the known M0
> reuse — the two training runs are genuinely independent."

**Why it is wrong.** Distinct image bytes establish that no file was
accidentally shared and that the seed-29 table is not a copy of seed 17's.
That is a file-level fact. It is not statistical independence.

The two runs differ **only in the training RNG seed**. They share:

- the same pretrained M0 backbone (one base model, never resampled);
- the same anchor caches, byte-identical, `anchor_seed: 17` for both;
- the same concepts and anchor mappings (`horse+cat`, `horse+dog`, `flower+sandwich`);
- the same 70 frozen evaluation prompt texts and the same four generation seeds;
- the same detector and thresholds.

**Corrected.** The two runs are two draws from **one training pipeline on one
base model**. They bound **training-seed** variability and nothing else — not
variability over models, concepts, prompts, anchors or detectors. Distinct bytes
do not license calling them independent replications of anything broader, and
n=2 remains a direction check.

---

## V2 — the direct contrast described as isolating the L2-SP effect and as independent of MA

**What AUDIT-01 said** (`AUDIT_REPORT.md` §3, `CORRECTED_TABLES.md` §2–3,
`make_report.py`, `compare_seeds.py`):
> "these contrasts carry **no** `D_cat(MA)` uncertainty"; "independent of
> `D_cat(MA)`"; "the parent-free contrast `D_cat(L2) − D_cat(no-L2)` is the
> quantity that **isolates the L2-SP effect**".

**Why it is wrong.** Two separate errors.

1. **"Independent of MA" overstates the cancellation.** What cancels is the
   *measured* `D(MA)` term in the estimator, so the contrast does not inherit
   that term's sampling noise. But **both children were trained starting from
   MA**, so each child's weights and behaviour still depend on that parent. The
   contrast is not causally independent of MA in any sense.
2. **It is a different estimand, not a better version of the same one.**
   `D_cat(child) − D_cat(MA)` asks how far a child moved *from its own parent* —
   historical recovery. `D_cat(L2) − D_cat(no-L2)` asks how the two arms differ
   *from each other*. Neither substitutes for the other, and the second does not
   answer the recovery question at all.

**Corrected.** State the algebra narrowly: the measured parent term cancels from
the estimator, so its sampling noise is not inherited. Then state that the two
quantities answer different questions, and that the children's dependence on MA
is unaffected. "Isolates the L2-SP effect" is withdrawn.

---

## V3 — the non-replication attributed "largely" to the parent

**What AUDIT-01 said** (`AUDIT_REPORT.md` §3, `COMBINED_RESULTS.md` banner):
> "the non-replication is **largely** in the `D_cat(MA)` reference term";
> "largely a reference-term effect".

**Why it is wrong.** It is a decomposable quantity, and the decomposition is
exactly even. Sandwich-L2 cat recovery moves −20.0 pp between seeds at t=0.5:

| component | seed 17 | seed 29 | contribution |
|---|---|---|---|
| child `D_cat(MAC_L2)` | 32.5% | 22.5% | **−10.0 pp** |
| parent `D_cat(MA)` | 17.5% | 27.5% | **−10.0 pp** |

The child's cat presence **falls** 10 pp and the parent's **rises** 10 pp.
Each accounts for exactly half.

**Corrected.** **Both contribute, here in equal measure.** The separate
observation that the *parent's own* across-seed gap sits entirely on paraphrase
prompts (literal `D_cat(MA)` is identical in both runs at all three thresholds)
remains true, but it describes the parent alone and does not make the
non-replication mainly a parent effect.

---

## V4 — the +40 to +47.5 pp result stated without naming its target

**What AUDIT-01 said:**
> "the most stable quantity in the study" — presented in a document whose
> subject is cat-history interference, without consistently naming what the
> quantity is about.

**Why it is wrong.** `D_new(L2) − D_new(no-L2)` is the difference in how much of
the **newest requested target** (dog in the B branch, sandwich in the C branch)
each arm left standing. It says nothing whatever about cat. Reported beside
cat-history claims without a label, it invites exactly the wrong reading.

**Corrected.** It is labelled the **newest-target residual difference**
throughout, with its target named, and flagged as carrying no information about
the cat history. Its stability (+40.0 to +47.5 pp across both seeds, both
branches, all three thresholds) is a real and useful fact about deletion
strength — about the newest request only.

---

## V5 — +7.5 pp called a "resolution floor"

**What AUDIT-01 said** (`AUDIT_REPORT.md` §4, protocol §3, `COMBINED_RESULTS.md`):
> "the sandwich arm retains only `+7.5 pp [+0.0, +17.5]` of measured
> suppression — **at or below the resolution floor**".

**Why it is wrong.** 7.5 pp is a measured value well inside what the grid can
represent: with 10 prompt clusters of 4 images the representable step is 2.5 pp,
so 7.5 pp is three steps, not a floor. What is true is that the *interval*
reaches zero, so the value is imprecise. "Resolution floor" conflates a small
estimate with an unmeasurable one, and it was being used to justify exclusion.

**Corrected.** The sandwich branch is excluded because **L2-SP suppression there
is too weak for the proposed question** — a matched comparison needs a
meaningful deletion level to match *to*, and +7.5 pp does not provide one.
That is a statement about fitness for the question, not about measurement
limits. The interval reaching zero is reported separately, as imprecision.

---

## V6 — "280 prompts"

**What AUDIT-01 said:**
> "the 280-prompt pilot set"; "the original pilot set"; "10 prompt clusters".

**Why it is wrong.** Verified against the frozen manifest: 280 **records**, of
which **70 distinct `prompt_id` values and 70 distinct prompt texts**, each
paired with 4 generation seeds.

**Corrected.** The evaluation set is **70 distinct prompt texts** (7 categories
× 10 texts), giving **280 prompt × generation-seed pairs** per checkpoint.
"280 prompts" overstates prompt diversity four-fold. The bootstrap clustering
was already correct — 10 clusters per category, each a prompt text carrying its
four generation seeds — and is now described in those terms.

---

## V7 — upstream stopping described as a certified per-arm maximum, and as newest-target-only

**What AUDIT-01 said** (protocol §2):
> "It is a convergence and compute-saving rule… **Each arm stops at *its own*
> maximum achieved deletion.**" and "upstream's UA is computed on the current
> forget target only".

**Why it is wrong.** Three errors, all checkable in the pinned tree
`9932ac3271a122f6d38d19e0b8c8908fe5237ff7`.

1. **Not a maximum.** `Regularizers/Simultaneous/simultaneous.py:check_early_stopping`
   stops when `ua >= stop_threshold` (default 99.0) **or** when
   `no_improvement_count >= patience`. That is **threshold-or-patience**
   stopping. It tracks `best_ua` but the weights saved at stop time are the
   current ones, not the best ones, so nothing certifies a per-arm maximum.
2. **The helper says nothing about which concepts are scored.** The pool comes
   from the caller: `ConAbl/train_conabl.py:336-344` passes
   `args.target_concepts` (the arg is `--anchor_target_concepts`). So target-pool
   semantics must be read at the caller, and `check_early_stopping` is **not**
   evidence for newest-target-only scoring. In upstream's sequential object
   script the caller passes a single `"${anchor_name}+${object_name}"` mapping.
3. **It is not wired into the sequential setting at all.** `--eval_interval`
   gates the whole mechanism, and it is passed in exactly six
   `BashScripts/Simultaneous/**` scripts and in **no** `BashScripts/Sequential/**`
   script. Our pilot's own runs recorded `eval_interval: null`.

**Corrected.** Upstream has **threshold-or-patience early stopping** in ConAbl's
trainer, gated on `--eval_interval`, scored on whatever concept pool the caller
supplies via `--anchor_target_concepts`, using an UnlearnCanvas classifier
(`--eval_classifier_dir`) rather than a COCO detector. In the pinned tree it is
enabled only in the simultaneous scripts. The brief states that appendices D and
E report early-stopping controls; that is taken as given and is reported
separately from what the code shows, since the paper was not read here.

---

## V8 — three inference overreaches in the protocol's decision rules

**What AUDIT-01 said** (protocol §10 and §8):
> "A1 − A2 retention is **small and includes zero in both seeds** → … L2-SP's
> apparent preservation benefit is consistent with being an effect of reduced
> update magnitude"; "A1 − A2 retention **excludes zero favouring A1** → **Where
> the update is constrained matters beyond how much**"; and the reproducibility
> check comparing the re-run's `delta.bin` sha256 against the stored value.

**Why it is wrong.**

1. **Lack of significance is not equivalence.** An interval including zero does
   not show the two arms retain equally. Ruling out a benefit of a stated size
   requires the interval to lie wholly inside a pre-declared margin — a
   different, stronger claim.
2. **A matched benefit does not prove a geometric mechanism.** "Where the update
   is constrained matters" is a claim about *which parameters* L2-SP spares.
   A retention difference at matched deletion is consistent with that, but also
   with differences in update direction, effective step schedule, or
   optimisation path. The experiment does not distinguish them.
3. **A file-hash mismatch is not a training difference.** `delta.bin` hashes can
   differ from serialisation order, metadata, or any bit of nondeterminism,
   while the tensors remain numerically equal — and tensors can differ slightly
   while the model is behaviourally indistinguishable. Hash identity, tensor
   equality and training equivalence are three distinct claims.

**Corrected.** The protocol now states an explicit equivalence margin and
requires intervals to fall wholly inside it before any practical-equivalence
claim; describes a matched benefit as *not attributable to a mechanism* by this
design and names mechanism as a separate follow-up; and specifies the
reproducibility check as **tensor-level comparison** (max absolute difference
per tensor), with hash mismatch recorded as a serialisation observation only.

---

## What survived this round unchanged

- Every measured value: 0 mismatches over 1026 re-derived values, still exact.
- Completeness: 0 duplicate, 0 missing, 0 extra rows; no de-duplication rule
  needed anywhere.
- The nine original claim corrections in
  [`CLAIM_CORRECTIONS.md`](CLAIM_CORRECTIONS.md), all of which stand. V1–V8
  sharpen the audit's own language; none reverses a correction it made.
- M0 reuse: 3080 generated images across 11 distinct checkpoint evaluations.
- The paraphrase-localisation of the **parent's own** across-seed gap.
- The identification of unequal deletion strength as the confound that makes the
  pilot's retention comparison uninterpretable — the finding that motivates the
  proposed diagnostic.
