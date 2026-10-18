# Masking audit, privacy and limitations

## 1. The honest headline

**The 40 images and their condition and pair provenance are already public.**
They were published, with per-file hashes and full pairing records, in the
[`modern-object-assets-v1`](https://github.com/bijay299/XXResearch/releases/tag/modern-object-assets-v1)
release before this review package existed.

Neutral opaque IDs (`A-001`…`A-040`) **reduce accidental disclosure** while a
reviewer works. They **cannot prevent a determined reverse match, and they do
not guarantee blinding.** Anyone willing to download the public image archive
and compare images — or compare hashes — can recover which neutral ID is which
image, and therefore its condition and its pair.

**Nothing in this package makes the already-public source provenance private,
and no such claim is made.**

## 2. What the masking does achieve

* No condition, pairing, counterpart link, source filename, prompt, seed or
  machine score appears anywhere in the reviewer bundle.
* Item IDs are assigned by a cryptographically random permutation, so ID order
  carries no information about case, condition, pairing or seed.
* In all three fixed order variants, the two members of a pair are kept **at
  least 4 positions apart**, so a pair is never shown back to back.
* Task A shows **one image at a time**; there is no counterpart view.

### Ordering provenance — corrected

An earlier version of this document said all orderings were computed from the
public IDs and alias alone and were independent of the arm mapping. **That was
wrong for the fixed variants**, and is corrected here:

* **`V1`/`V2`/`V3` are pair-aware.** They were constructed **using private pair
  membership** to enforce the four-position separation. They are **not**
  derivable from public data alone and are **not** independent of pairing.
  They do **not** use which member is M0 versus edited. The spacing rule
  encodes **negative pairing constraints** — it tells you which items are *not*
  each other's partner at close range — not the pairing key itself.
* **The alias-derived shuffle is pairing-independent** (alias + public IDs
  only) and has **no guaranteed minimum separation**; a pair can land adjacent.

The permutations themselves were not regenerated and remain exactly as frozen.

## 3. Public integrity hashes — a deliberate, disclosed trade-off

**Public integrity hashes are not secret, and they can facilitate reverse
identification.** A published per-item hash can be matched against a hash of the
corresponding public release image, recovering the mapping with no image
comparison at all.

This package therefore makes a deliberate choice:

* The **ZIP as a whole** is published with its sha256, so anyone can verify they
  downloaded the authentic bundle, and the full member inventory is published.
* **Per-item image hashes are withheld** from the public tree, and so are the
  per-file source rows keyed by case and condition. Publishing *both* row sets
  would have let anyone join them on the hash and read off the entire mapping
  from this directory alone, with no effort — defeating even the
  accidental-disclosure reduction that neutral IDs are for.
  (`verify/source_verification_summary.json` and
  `verify/taskA_completeness_summary.json` carry the verdicts and counts with
  the row detail marked `REDACTED`; the full rows are held privately.)

This narrows *casual* re-identification. It does **not** make re-identification
impossible, and it is not claimed to.

## 4. Task B is only partially masked — by design

Task B **explicitly identifies the reference as the original-model output**
beside the edited candidate. That is required for the comparison to be
meaningful, and it means reviewers know the reference's identity while judging.
Task B is therefore **only partially masked**, and is labelled as such in its own
instructions.

Task B is **not published**. Its release requires **both** that all planned
Task-A collection is closed with original responses locked, **and** later
explicit authorization from the user.

> **The offline viewer cannot enforce that lock.** It is a coordinator
> procedure, not a technical control. Nothing in the UI prevents a holder of the
> files from opening them, and no such enforcement is claimed. There is no
> unlockable Task-B content embedded in the Task-A bundle.

Original Task-A answers are preserved as given; any revision a reviewer makes
after seeing Task B is recorded **separately** and never overwrites the original.

## 5. Automated audit result

[`scripts/masking_audit.py`](scripts/masking_audit.py) scans every public file
and every ZIP member — contents, filenames, code, comments, JSON keys and PNG
container chunks — for source filenames, case IDs, evaluation seeds, condition
keys, pair associations, mapping keys, answer keys, distinctive verbatim case
prompts, and the private salt.

**Result: clean.**

| check | result |
|---|---|
| Public files scanned | 34 |
| Content leaks | **0** |
| Filename leaks | **0** |
| Salt present in any public file | **no** |
| ZIP members | 47 (40 PNG) |
| Staged preview assets outside the allowlist | **0** |
| Disallowed preview assets live on either release | **0** |
| Synthetic test export or completed response inside the bundle | **none** |
| ZIP member-name leaks | **0** |
| ZIP content leaks | **0** |
| Task-B members in the ZIP | **none** |
| PNG container metadata chunks | **none** (containers clean) |
| Item IDs neutral (`A-nnn`) | **yes** |

Two scoping decisions in that audit, stated so they can be challenged:

1. **Field names in source code are not treated as leaks.** The published build
   scripts declare keys such as `source_filename` while carrying no mapping
   data; structural key patterns are checked in data files, value patterns
   everywhere. It was separately confirmed that no published script contains any
   actual case ID, seed, condition filename or association value.
2. **The bare target term ("a teapot") is not treated as a leaked prompt.** It
   is disclosed to reviewers by design and is not case-identifying; only
   distinctive prompts (≥15 characters) are flagged.

Generic shared renderer code without associations is acceptable, and the Task-A
and Task-B viewers deliberately share that code.

## 6. A leak that was caught and fixed

During preparation, two QA screenshots showed a **real research image beside its
neutral ID**, which would have disclosed that mapping to anyone who read the
preview. They were moved into the private area and replaced with previews
rendered against an obviously **synthetic placeholder** image. The published
item-view previews therefore show the real UI layout but **no research image**.

This is recorded in the preparation ledger as
`quarantine_leaking_screenshots`.

**It then recurred, briefly, and was fixed properly.** Restoring
`browser_qa.py` from Git during the patch01 repair re-created those two
screenshots, and the publisher was staging *everything* in the screenshots
directory, so they were uploaded to the patch01 release. They were removed as
soon as the asset list was reviewed. Three changes prevent recurrence: the QA
script now writes item-view screenshots **into the private area**; the publisher
stages previews from an **explicit four-file allowlist** and fails on anything
else; and **this audit now also checks the staged preview directory and the live
asset list of every release** — the gap that let it through, since it previously
looked only at the results tree and the ZIP.

## 7. Attribution and licensing kept out of reviewer instructions

Source attribution, provenance and licensing live in the technical
documentation (this file, the README and the protocol). **Reviewer instructions
are kept neutral**: they do not name the model, the method, the source release
or the conditions, and they do not suggest any expected outcome.

The underlying images are AI-generated outputs of FLUX.1-dev and of a GEM
fine-tuned derivative of it, published under the terms recorded in the
`modern_object_assets_v1` release (FLUX.1 [dev] Non-Commercial License v1.1.1;
GEM source MIT). Reviewers are told the images are AI-generated; they are not
told which model produced which image.

## 8. Other limitations

* **Self-declared exposure cannot be verified** from the artifacts. A reviewer's
  answers about prior access and coaching are taken at their word.
* **The validator counts files, not people.** Identical content and repeated
  aliases are flagged precisely because they are **not** evidence of a second
  independent reviewer.
* **Commitments prove existence and non-alteration, not privacy, and they do
  not cover everything.** The published HMACs cover the **populated Task-A and
  Task-B mappings, the order variants and the item-ID list** — they do **not**
  cover every byte of the private Task-B package (its images, viewer files or
  layout are not committed). A commitment shows the committed material has not
  changed since it was written; it does not make the underlying images or their
  provenance private.
* **A future opening should be anchored to public publication, not to a
  self-declared timestamp.** The commitments were published in this repository
  at a commit anyone can see; that public publication is what a later reveal can
  be checked against. The `frozen_iso` field inside the artifacts is
  self-declared and is **not** independently proven.
* **20 paired cases are not 40 independent comparisons**, and three reviewers
  judging the same images do not produce 120 independent observations.
* **No outcome of any kind is reported here**, because none exists.
