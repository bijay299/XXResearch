# U29 — audit of the teacher proposed for the U28 method candidate

**Date 2026-10-27. Run on owner instruction before committing GPU time to U28.**

The arm `u4_k8_s400_b2` was recorded at **target 0.0000 / retain8 0.9833** on a single
`gen_eval` panel and was selected as the teacher for the U28 method candidate on that basis.
U29 re-measures it at four independent eval seeds and saves the images at identical
prompts and seeds for the unmodified SD-1.4 base, so a human can see what the numbers mean.

## Files

| file | what it shows |
|---|---|
| `sheet_teacher_target.png` | teacher, 3 held-out church prompts x 3 images x 4 eval seeds, each tile captioned `seed<N> <top-1 CLIP label>=<score>` |
| `sheet_base_target.png` | unmodified SD-1.4, **identical prompts and seeds** |
| `sheet_teacher_retain.png` | teacher on four non-target concepts |
| `sheet_base_retain.png` | base on the same four |
| `audit.json` | per-image top-3 CLIP labels and scores, all seeds |
| `u29_teacher_audit.log` | full run log |

## Finding 1 — the 0.0000 does not replicate

| eval seed | teacher target acc | base target acc |
|---:|---:|---:|
| 0 | 0.2222 (2/9) | 1.0000 (9/9) |
| 1 | 0.1111 (1/9) | 1.0000 (9/9) |
| 2 | 0.1111 (1/9) | 1.0000 (9/9) |
| 3 | 0.3333 (3/9) | 1.0000 (9/9) |

Mean **0.194**, spread 0.2222. The single-panel 0.0000 was a draw. The base reads 1.0000
at every seed, so the suppression itself is real and large — but the figure that justified
selecting this teacher straddles the pre-declared `target <= 0.20` success bar rather than
clearing it.

## Finding 2 — the erasure mode is degenerate output, not redirection

`sheet_teacher_target.png` shows the target-prompt images are **repetitive grid, stripe and
dot textures** — not coherent images of some other subject. `sheet_teacher_retain.png` shows
the non-target prompts are clean and sharp. So the model is not redirecting "church" to another
concept; it is breaking on that prompt while remaining intact elsewhere.

**Metric consequence.** H is a 12-way closed-set CLIP accuracy with **no reject option**, so
every degenerate image is forced onto some label. In `audit.json` a window-grid texture is
labelled `church=0.546` and a stripe texture `van_gogh=0.789`. The ~0.19 is therefore CLIP
scattering over broken output, not evidence that the model draws something else. **The primary
metric cannot distinguish clean erasure from a broken model**, and `legacy_retain8` cannot
either, because it is measured only on non-target prompts.

## Consequence for U28

U28 was **not launched**. Its pre-declared success criterion
(`retain8 >= 0.85 AND target <= 0.20 AND post-attack max-over-valid H <= 0.70`) is
insufficient as written: no term in it inspects the target-prompt image quality, which is
where this teacher fails. See the project's U28 prereg for the amendment.

**Not withdrawn by this audit:** the U23 alpha=0.3 results themselves. Those arms used a
different (AdvUnlearn) teacher and have not yet been audited the same way. Whether their
post-attack H of 0.6000/0.6667 is genuine robustness or the same degenerate-output artifact
is an open question and is the next thing to measure.
