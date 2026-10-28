# D3b — an **author-released checkpoint** is fully reversed, over 51 admissible cells

**2026-10-28.** D3 established that nothing in this project is an official reproduction, which
forced the central claim down to *"across nine re-implementations, every arm is reversed."* The fix
named was to attack **author-released checkpoints**, removing our implementation from the loop.

One such attack set **already exists on disk** and had not been identified as answering D3.

## The artifact

**AdvUnlearn** (OPTML-Group), pinned at revision `bca3b27922b1e70cfe5b858e35beccc2fbc3b5ee`, the
authors' released `church_unlearned` **text encoder**. Not our code, not our training — the authors'
own published weights.

Clean panel on our benchmark:

| | |
|---|---:|
| target (church) | **0.4000** |
| mean non-target, 11 concepts | **0.9939** |

Every non-target concept at 1.0000 except monet at 0.9333. So the released checkpoint suppresses
church to 0.40 while leaving general capability essentially untouched — a real, working erasure, by
the authors' own hand.

## The result

**57 attack cells** across thirteen experiment families (H46, H49, H53, H54, H71, H72, H73, LOWB,
KEEP and others), spanning full fine-tuning, LoRA and text-inversion adversaries, multiple learning
rates and seeds.

| | |
|---|---:|
| cells | 57 |
| **admissible** (gate-valid, not destroyed) | **51** |
| destroyed | 1 |
| gate-invalid | 5 |
| **max-over-admissible `H_after`** | **1.0000** |

**51 admissible cells is the best-supported single result in the project** — better than any
baseline in the U33 ladder (22, 17, 15, 14).

## What this restores

The claim can now be stated at two strengths, both true:

- **Strong, for AdvUnlearn:** *the authors' own released checkpoint, which genuinely suppresses the
  concept while preserving everything else, is driven to full recovery by a relearning attack over
  51 admissible cells.* No re-implementation is involved.
- **General:** *across nine re-implementations of published methods plus one author-released
  checkpoint, every arm is reversed.*

D3 is therefore **narrowed, not closed**. One author artifact is covered; UCE, ESD, GIFT, IMMA, FRP
and Meta-Unlearning remain our implementations.

## What is still needed, and the blocker

**ResAlign** is the method that matters most here, because it is the only published method that
demonstrates resistance to anything. It releases two SD-1.4 checkpoints.

**They cannot be downloaded from this machine.** Their OneDrive links return HTTP 403 to a plain
request and the sign-in page to a browser user-agent; there is no Hugging Face mirror (searched: 0
model hits, only the `randyli/ResAlign_train` dataset). **Fetching them needs an interactive browser
session.** Flagged to the owner.

## Limits

- One author checkpoint, one concept (church), one model family.
- AdvUnlearn's released artifact is a **text encoder**, so this tests their erasure locus, not a
  UNet-side defence.
- Its un-attacked suppression is 0.4000, which is weak erasure to begin with — a point that must be
  stated whenever this result is quoted, and which is visible in the table above rather than hidden.
- The 57 cells were run across many earlier studies under their own protocols, not a single
  pre-registered sweep; the admissible-cell count is a pooled figure.
