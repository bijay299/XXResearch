# Visual check of the erasure method, 2026-11-02

Generated because every metric in this project is a **proxy**, and the owner asked to judge by eye.
Looking at the images overturned two conclusions I had drawn from the metrics.

Each row: **base SD-1.4** | **ESD init** (the cheap erasure we build on) | **ours** (the method),
on the evaluator's held-out prompts, same seed across arms.

Nudity panels are deliberately **not** included here; they are kept local. The point of that panel
is that the erased model produces no nudity, but the base column necessarily does, and publishing
it is a different act from generating it.

## What the images show, and where they contradict the metrics

**1. The 12-way closed-set CLIP classifier is unreliable in BOTH directions.**
It must pick one of twelve labels and has no reject option.

- `van_gogh_*`: our arm outputs **pure blue/white noise**. The classifier scored it
  **H = 1.000 ("van Gogh present")**, which I had reported as *"the method destroyed the erasure."*
  It did not. The concept did not return — the **model** was destroyed, and the classifier matched
  noise to the label whose colour statistics it resembles.
- `church_*`: our arm outputs a **coherent non-church** (a house on a hillside) while the ESD init
  still shows a church-like clock-tower building. By eye ours is the better erasure; the metric said
  the opposite (H 0.733 ours vs 0.611 init).

So the same underlying behaviour is scored as success or failure depending on what the output
happens to resemble. Every H-based conclusion in this project needs re-reading with that in mind.

**2. Our method tends to DEGRADE rather than REDIRECT on some concepts.**
ESD replaces the erased concept with a different coherent image (a building for a parachute, a
different painting for van Gogh). Ours often produces incoherent texture instead:

- `parachute_*`: base = a parachute; ESD init = a coherent building; ours = grey rubble.
  The classifier rewards this (H 0.167 vs the init's 0.389) because rubble is not a parachute --
  but "unrecognisable output" is not the same achievement as "clean erasure".

**3. Utility on unrelated prompts is genuinely preserved -- this claim survives.**
`utility_*`: base and ours are near-identical on ordinary COCO captions, consistent with the
measured CLIP 0.2611 vs base's 0.2639. The degradation above is specific to the erased concept,
not general damage.

**4. On nudity (not shown here) the method REDIRECTS coherently**, unlike parachute/van Gogh.
That is the concept it was developed against, and it is the only one measured with a continuous
detector rather than a closed-set argmax.

## Consequence

The "works on 2 of 4 concepts" summary is not safe as stated. Nudity's result rests on a sound
instrument and coherent outputs. Parachute's apparent success may be degradation rewarded by a
metric that cannot tell degradation from erasure. Van Gogh's apparent failure was a measurement
artefact. Church's apparent failure may be understated.
