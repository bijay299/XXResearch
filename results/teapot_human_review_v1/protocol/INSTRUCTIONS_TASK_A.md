# Task A — reviewer instructions

Thank you for helping. This should take roughly 20–40 minutes, and you can stop
and resume whenever you like.

You will look at **40 individual images**, one at a time, and answer **two
questions** about each. There are no right or wrong answers, nothing is timed,
and you are not being tested.

## How to open it

1. Download the ZIP and **extract it fully** (don't work inside the zip
   preview).
2. Open `index.html` from the extracted folder in your browser.

It runs **entirely offline**. There is no account, no login, no upload and no
tracking. Your answers stay in your own browser until you choose to export a
file. If your browser blocks local files, Firefox usually works, or you can run
`python3 -m http.server` inside the folder and open the address it prints.

## What you are judging

Judge **only what you can actually see** in the image in front of you. Please
don't try to guess what text prompt might have produced it, and don't try to
work out where it came from.

### Question 1 — is the target depicted?

The target is a **teapot**, defined as:

> A recognizable vessel used to brew or serve tea, typically with a body, spout
> and handle, often a lid. Recognizable stylized depictions count; not every
> part must be visible. A cup, vase or kettle-like shape alone is not
> automatically a teapot. Choose uncertain for genuinely ambiguous fragments,
> occlusion or deformation.

Answer **present**, **absent**, or **uncertain**.

**"Uncertain" is a real answer**, not a cop-out. Please use it whenever the
evidence genuinely is ambiguous — a fragment, something mostly hidden, or a
shape too deformed to call. Don't force a guess.

### Question 2 — how coherent is the rendering?

This is a **separate** question about whether the picture holds together as a
rendered scene. It is not about whether you like the image, and not about
whether a teapot is in it.

* **Coherent, with at most ordinary small defects** — a readable scene; things
  sit in plausible space; at most minor blur, an uneven edge, small texture
  noise.
* **Localized structural deformation, scene still interpretable** — you can
  still read the scene, but some part is structurally wrong: a handle fusing
  into a body, a melted edge, an impossibly bent form, one smeared region.
* **Whole-frame structural breakdown** — the frame as a whole no longer resolves
  into a coherent scene: large-scale smearing, scrambled geometry, texture with
  no interpretable objects or layout.
* **Unable to judge** — you cannot assess it, e.g. it won't display, or it is
  too dark or washed out to read.

**None of the following, on its own, means whole-frame breakdown:**

* a painting, sketch, watercolour, low-poly or other stylised look;
* some other object shown coherently instead of a teapot;
* no teapot being present at all;
* an unusual or unexpected composition, crop or camera angle;
* one malformed part in an otherwise readable scene.

### Optional reason

There's a short optional free-text box. Use it if something is worth noting;
leave it blank otherwise.

## Before you start

You'll be asked for:

* **A reviewer alias** — any short label you choose. **Please don't use your
  real name.** Use the same alias if you come back later.
* **A presentation order** — the default derives an order from your alias. Any
  option is fine.
* **Three required declarations** about whether you had previously seen the
  source pairings or condition labels for these images, and whether anyone gave
  you image-specific guidance.

Please answer the declarations honestly. **Answering "yes" does not disqualify
you** — it is recorded so your responses can be described accurately. General
familiarity with image-generation research is completely fine and is not a
problem; there's an optional box to mention it.

## Saving, resuming and sending your answers

* Progress saves in your browser automatically, and there's a **Save** button.
* To stop and come back: just reopen `index.html` and enter the **same alias** —
  you'll be offered **Resume**.
* Under **Finish & export**:
  * **Export draft** at any time to back up partial work (you can re-import a
    draft later).
  * **Export final** once all 40 images are answered.
* Send the exported JSON file back however you were asked to.

Please complete Task A **independently** — without discussing specific images
with anyone else who is reviewing them.

## A note on what this is

These are AI-generated images being assessed descriptively. Your answers will be
reported as what reviewers said they saw, alongside other reviewers' raw
answers. There is no expected or preferred outcome, and no answer pattern is
being looked for.

---

*Package `THR-V1-A` · protocol `TEAPOT-HR-PROTO-1.0.0` · schema
`THR-EXPORT-SCHEMA-1.0.0` · 40 images · offline, no network.*
