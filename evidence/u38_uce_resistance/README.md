# U38 — what UCE's resistance actually is, checked in the images

**2026-10-28.** U37 found UCE's suppression holds under benign fine-tuning (0.000–0.200, n=3) where
ESD's collapses. That is a **low number from `target_suppression_acc`**, the metric U29/U30 showed
can be fully satisfied by a model emitting texture mush. This audits the images at four eval seeds.

## 1. The numbers replicate

| arm (self-source benign fine-tuned) | target accuracy, eval seeds 0–3 |
|---|---|
| `uce_church` sd0 | 0.0000, 0.0000, 0.2222, 0.1111 |
| `uce_church` sd6 | 0.0000, 0.1111, 0.1111, 0.2222 |
| `esd_church` sd1 *(positive control)* | 0.7778, 0.7778, 1.0000, 0.7778 |

The split is stable. ESD recovers, UCE does not.

## 2. What UCE is actually generating

**Target prompts: fibrous, streaked, abstract texture.** No churches — and no coherent subject of any
kind. This is the same degenerate output UCE produced *before* any fine-tuning (U30). Nothing
changed; it stayed broken on that prompt.

**Retain prompts: intact and coherent.** Cassette player, english springer, french horn, golf ball —
all recognisable, all scored 1.00. General capability survives the fine-tune.

## 3. The honest reading, which depends on the threat model

Against the project owner's stated criterion — *the concept must not appear, other concepts must not
be damaged, and what the model draws instead does not matter* — **UCE passes all three bars for the
benign threat.** The church does not appear, the other concepts work, and it stays that way after
200 steps of unrelated fine-tuning. That is a genuine pass and should be reported as one.

What must **not** be claimed is the stronger thing it superficially resembles. UCE is not redirecting
"church" to some coherent alternative and holding that redirection. It is **degenerate on the target
prompt and remains degenerate**. The defence is "it stays broken", not "it stays erased and usable".

A reviewer who holds target-prompt output quality as a requirement would reject this as a defence. A
reviewer using the safety framing — the forbidden thing must not be producible, everything else must
work — would accept it. **The paper must state which framing it uses and show the images either
way.**

## 4. This does not rescue UCE on the axis that matters

UCE still falls to **targeted relearning in four gradient steps** (U35). The degenerate target output
does not protect it there. So:

| threat | UCE | ESD |
|---|---|---|
| benign downstream fine-tuning | **holds** (output degenerate throughout) | **fails** — 0.78–1.00, verified churches |
| targeted relearning | **fails in 4 gradient steps** | **fails in 1 gradient step** |

The three-tier taxonomy stands as an empirical finding. What U38 changes is the *description* of the
tier-2 pass, not its existence.

## 5. Correction to my own framing

In reporting U37 I called this "the project's first resistance result" without qualification. That
was premature by one experiment. The result survives, but its content is narrower than the phrase
implies, and the narrowing is visible only in the images.

## 6. Limits

- Two UCE seeds and one ESD seed audited, four eval seeds each. Church only, SD-1.4.
- "Texture mush" and "coherent" are assessments by inspection of the contact sheets, not a
  quantitative no-reference quality measure. The UCE case is unambiguous enough to read directly.
- Retain quality was checked on four concepts at one eval seed, not the full panel.
