# D3 — **nothing in this project is an official reproduction**, and the central claim must be worded accordingly

**2026-10-28.** D3 was opened as "the UCE port disagrees with pinned upstream." Checking it against
`UPSTREAM_FIDELITY_MANIFEST.md` shows the issue is **not specific to UCE**. This is the most serious
threat to the write-up found so far, and it is better found now than in review.

## 1. The manifest's own statement

> *(no prefix exists for)* **official reproduction** — authors' code **and** authors' protocol,
> metric and data. **Nothing in this project qualifies.**

Every "baseline" in our tables is **our implementation**, measured on **our** benchmark with **our**
12-label CLIP metric. `REF_*`-prefixed arms are upstream *code execution* on our benchmark — closer,
but still our protocol and our metric.

## 2. Documented deviations, per method

| method | status | deviation on record |
|---|---|---|
| **UCE** | our re-implementation | our `last..76` embedding range vs upstream's **single** embedding at `attention_mask.sum()-2`. **Deliberate**, and it is *why our arm erases at all* — upstream executed at these settings leaves church at **15/15**, replicated twice by independent code paths five days apart. Our edit is **4.0× larger** than upstream's |
| **IMMA** | our implementation | **our scope is less than half the reference's** (2.13×), excludes self-attention; LoRA never re-initialised where the reference optionally re-initialises; one-step inner Adam vs two separate AdamW |
| **Meta-Unlearning** | our corrected implementation | variant A omits two of Eq. 8's three terms; v2 restores them. Explicitly "not an official reproduction" |
| **GIFT** | our implementation | **no official implementation exists** to compare against |
| **WSR / WDR** | ours | `erase_wsr.py` anchors toward base where the paper **maximises** distance — direction inverted. `erase_wdr_v2.py` is the faithful adaptation and is largely unrun |
| **FRP / FRAG** | our port | no released code |

## 3. What this does to the claim

**Cannot be written:** *"Every published concept-erasure method is reversed by a relearning attack."*

**Can be written:** *"Across nine re-implementations of published concept-erasure methods, built from
the papers and where available pinned upstream code, every arm is reversed."*

That is a materially weaker sentence, and it is the true one. A reviewer who knows these methods will
ask "did you run the authors' code?" and the answer today is: executed for UCE (where it does not
erase at our settings), not otherwise.

**The UCE case is the sharpest.** Our strongest results — recovery in four gradient steps (U35),
holding under benign fine-tuning (U37/U38) — are properties of **our variant**, not of UCE as
published. Those sentences must name the variant.

## 4. The fix, and it is a real experiment

Several of these methods **release checkpoints or code** that can be attacked directly, removing our
implementation from the loop entirely:

- **ResAlign** publishes two SD-1.4 checkpoints (higher-safety, higher-utility) for the *sexual*
  concept, with reported pre- and post-fine-tuning numbers.
- **AdvUnlearn** has an official snapshot already in this project (used as a teacher).
- **UCE** upstream runs; the open question is which configuration makes it erase.

Attacking **author-released checkpoints** with our attacker would give genuine baselines and convert
the claim back to the strong form for those methods. That is the highest-value remaining experiment
in the project, above the representation probe.

Caveat worth stating up front: ResAlign's checkpoints target NSFW content, so that evaluation
involves generating and classifying such material. It is their own benchmark and standard practice
in safety work, but it is a different kind of material and should be an explicit decision.

## 5. What is not undermined

- The **measurement findings** stand independently of fidelity: a closed-set CLIP metric with no
  reject option scores a visible church as `van_gogh=0.72` regardless of whose code made the model.
- The **mechanism findings** stand: Fisher importance losing to a random control, the attacker
  needing 768 scalars, benign fine-tuning recovering ESD — all are properties of models we hold.
- **Our own methods failing** is unaffected; they are ours by construction.

## 6. Defect status

| | status |
|---|---|
| **D1** inconsistent label text | **closed** — moves metrics ~0.01, below noise |
| **D2** gate bias | **closed** — confirmed, runs against our claim, conclusion survives |
| **D3** fidelity | **open and widened** — no arm is an official reproduction; wording must change and author checkpoints should be attacked |
| **D4** pervasive n=1 | open |
