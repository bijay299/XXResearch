# Instructions preview — read this before opening the package

This is the exact reviewer-facing instruction text that appears on the
setup screen of `index.html`, reproduced here so it can be read and
judged for clarity **without** starting the review flow. No study image,
count, or label is shown anywhere below.

---

**Background.** These images come from AI image-generation models that
were intentionally edited to remove ("erase") a specific dog breed
concept from what they can draw. For each group below you will see
**three** images generated from the **same text prompt and the same
random seed** (shown above each group): an **Original model** image,
and two images (**A** and **B**) from edited versions of the model.

**The Original is a generated quality reference, not ground truth — it
may itself contain defects.** It is shown so you can judge whether
something looks different or worse than the model could already
produce before any editing, not as a "correct" answer. **A and B's
model identities remain hidden** — A/B is a neutral, randomized label
only; it does not mean "before"/"after," and the same letter does not
mean the same thing from one group to the next.

**The erased dog does NOT need to appear in A or B; its absence is
expected.** A coherent substitute subject, even if unrelated to the
prompt, is not a rendering defect by itself. Changed subjects,
composition, or pose — or the Original and an edited image simply
looking different from each other — do not by themselves constitute
corruption: equal prompts and seeds do not require identical objects or
geometry after editing. **An image can depict a clearly recognizable
person or object while still containing severe deformities** —
recognizing the scene does not mean its faces, bodies, or objects are
well formed. This questionnaire scores rendering breakdown, local
defects, optional style mismatch, and deformation relative to the
Original. It does not score subject or scene compliance. Style
mismatch applies only when the prompt explicitly requests a style — a
painting-like image alone is not a breakdown.

## What to judge, for each of A and B

**1. Widespread rendering breakdown.** Visible structural breakdown
spread across much of the scene or its major structures — for example
pervasive fragmentation, broken geometry, static or pixel-level
artifacts, or smearing that destroys structure. This is a qualitative
judgment call, not a precise formula.

- A coherent abstract, patterned, or painting-like image is **not**
  automatically a breakdown — if the prompt doesn't specify a style,
  don't assume photography was required.
- An unexpected subject, by itself, is **not** a breakdown.
- Ordinary depth-of-field blur and ordinary stylization are **not**
  breakdowns.
- A recognizable subject does **not** force a "No" here.
- If you can't confidently judge the boundary, choose **"Unable to
  judge."** If you answer "Yes" or "Unable to judge," you'll be asked
  for one brief reason.

**2. Localized defects.** A structural, anatomical, or facial defect
affecting a specific part of the image, such as a malformed hand.
Answered independently of question 1 — local defects can occur with or
without widespread breakdown; both can be "Yes."

**3. Style mismatch (optional, separate from corruption).** Only
relevant if the prompt explicitly named a style or medium. If no style
was specified, choose "Style not specified."

**4. Deformation compared with the Original.** "Compared with the
Original-model image, how much visible structural deformation does this
image have?" Less / About the same / More / Unable to compare.
**Structural deformation** includes malformed anatomy or objects,
broken geometry, and destructive fragmentation or smearing — local or
widespread. A brief reason is required for "More" or "Unable to
compare." If the subjects or scenes differ too much to judge fairly,
choose "Unable to compare" rather than guessing. This is a separate,
relative judgment — never inferred from, or merged into, questions 1–3.

**5. After both A and B's individual questions are answered**: which
image has MORE widespread rendering breakdown — A or B (or no clear
difference / unable to judge)? The letter you pick is the **worse**
image, not the preferred one. This is the A-vs-B comparison, separate
from either image's comparison to the Original.

## Illustrative examples (not from this study)

- **Not a breakdown**: a prompt with no stated style produces a
  coherent, painting-like image instead of a photograph.
- **Local defect, not widespread**: an otherwise coherent photo-realistic
  scene has one hand with an extra or malformed finger.
- **Widespread breakdown**: an image where fragmented, broken geometry
  and static-like artifacts cover most of the frame, so the underlying
  scene structure is no longer recognizable.

These three examples illustrate the definitions only — they are not
images from this study. Apply the same definitions to each study image
and to the Original-comparison question; the images do not need to
resemble these examples.

## Before starting

The package asks for: a reviewer alias (not your real name), your
prior exposure to this material, whether you've discussed or received
feedback about any of this review's 60 images (20 groups) before now,
and an acknowledgement that a human is doing the review. Nothing is
transmitted automatically — you export your answers as a file and
return it yourself.

---

If these instructions are clear, the next step (not performed by this
preparation) is for the user to arrange one complete review from a
human reviewer, extract `BASELINE_CONTEXT_REVIEW_PACKAGE.zip`, open
`index.html`, and return the exported JSON.
