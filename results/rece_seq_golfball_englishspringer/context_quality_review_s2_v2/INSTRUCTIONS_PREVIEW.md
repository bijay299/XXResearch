# Instructions preview — read this before opening the package

This is the exact reviewer-facing instruction text that appears on the
setup screen of `index.html`, reproduced here so it can be read and
judged for clarity **without** starting the review flow. No study image,
count, or label is shown anywhere below.

---

**Background.** These images come from AI image-generation models that
were intentionally edited to remove ("erase") a specific dog breed
concept from what they can draw. Each pair shows two images generated
from the **same text prompt** (shown above each pair in the package) by
two different edited versions of the model. **The erased dog does NOT
need to appear in either image; its absence is expected. A coherent
substitute subject, even if unrelated to the prompt, is not a rendering
defect by itself.** This questionnaire scores rendering breakdown,
local defects, and optional style mismatch. It does not score subject
or scene compliance. Style mismatch applies only when the prompt
explicitly requests a style. Visible rendering breakdown is the main
question here.

**A/B is a neutral, randomized label only** — it does not mean
"before"/"after," and the same letter does not mean the same thing from
one pair to the next.

## What to judge

**1. Widespread rendering breakdown.** Visible structural breakdown
spread across much of the scene or its major structures — for example
pervasive fragmentation, broken geometry, static or pixel-level
artifacts, or smearing that destroys structure. This is a qualitative
judgment call, not a precise formula.

- A coherent abstract, patterned, or painting-like image is **not**
  automatically a breakdown, even if a photograph might otherwise have
  been expected — if the prompt doesn't specify a style, don't assume
  photography was required.
- An unexpected subject, by itself, is **not** a breakdown.
- Ordinary depth-of-field blur and ordinary stylization are **not**
  breakdowns.
- If you can't confidently judge the boundary, choose **"Unable to
  judge."**
- If you answer "Yes" or "Unable to judge," you'll be asked for one
  brief reason naming the visible error and its extent (or the specific
  ambiguity).

**2. Localized defects.** A structural, anatomical, or facial defect
affecting a specific part of the image, such as a malformed hand.
Answer this independently of question 1 — local defects can occur with
or without widespread breakdown; both questions can receive "Yes."

**3. Style mismatch (optional, separate from corruption).** Only
relevant if the prompt explicitly named a style or medium. If no style
was specified, choose "Style not specified." Never combined with the
corruption questions.

**4. After both images' individual questions are answered**: which
image has MORE widespread rendering breakdown? A / B / no clear
difference / unable to judge. The letter you pick is the **worse**
image, not the preferred one.

## Illustrative examples (not from this study)

- **Not a breakdown**: a prompt with no stated style produces a
  coherent, painting-like image instead of a photograph. Unexpected
  medium, but structurally coherent.
- **Local defect, not widespread**: an otherwise coherent photo-realistic
  scene has one hand with an extra or malformed finger. Everything
  else in the frame is fine.
- **Widespread breakdown**: an image where fragmented, broken geometry
  and static-like artifacts cover most of the frame, so the underlying
  scene structure is no longer recognizable.

These three examples illustrate the definitions only — they are not
images from this study. Apply the same definitions to each study image;
the images do not need to resemble these examples.

## Before starting

The package asks for: a reviewer alias (not your real name), your
prior exposure to this material, whether you've discussed or received
feedback about any of this review's 40 study images (20 pairs) before
now, and an acknowledgement that a human is doing the review. Nothing is
transmitted automatically — you export your answers as a file and
return it yourself.

---

If these instructions are clear, the next step (not performed by this
preparation) is for the user to arrange one complete review from a
human reviewer, extract `CONTEXT_REVIEW_PACKAGE_V2.zip`, open
`index.html`, and return the exported JSON.
