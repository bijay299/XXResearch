# Does Concept Forgetting Survive Relearning?

Presentation and full supporting evidence. Bijay Pandey, Howard University, 27 October 2026.

**Question.** Can a text-to-image model forget a concept while preserving other capabilities, and
does that forgetting survive later fine-tuning?

**Answer, from the measurements in this folder: no.** Across ESD, UCE, GIFT, FRAG/FRP, an adapted
IMMA and both UTRAP configurations, a 25-step fine-tune restored the erased concept once the
attacker was properly tuned. Every arm reaches recovery H = 1.000 at some tested attacker setting.

## Files

| file | what it is |
|---|---|
| [`UTRAP_relearning_Pandey_HowardU.pptx`](UTRAP_relearning_Pandey_HowardU.pptx) | 12 slides, speaker notes embedded |
| [`UTRAP_relearning_Pandey_HowardU.pdf`](UTRAP_relearning_Pandey_HowardU.pdf) | same slides as PDF |
| [`speaker_script.md`](speaker_script.md) | verbatim read-aloud script, 1,712 words |
| [`SUPPORTING_MATERIAL.md`](SUPPORTING_MATERIAL.md) | scope limits, fidelity tiers, excluded methods, withdrawn claims |
| [`render/`](render/) | every slide as a PNG, for quick viewing |
| [`figures/`](figures/) | the five plots |
| [`data/`](data/) | snapshot, per-figure CSVs, all 747 attack cells, references, evidence manifest |
| [`images/`](images/) | the 29 images used on slides, plus the generation manifest |

## How to check any number on a slide

[`data/evidence_manifest.csv`](data/evidence_manifest.csv) maps all 282 claims and images to their
source artifact. Every per-cell value was re-derived from its source JSON before delivery.

## Three qualifications that bound everything here

1. **Nothing is an official reproduction.** That needs the authors' code *and* their protocol,
   metric and data. The IMMA cell is the closest: their code, our dataset and our evaluation.
2. **retain8 is not an image quality measure.** Slide 4 shows methods scoring 0.87 to 0.97 on a
   concept they visibly degrade, because a closed-set classifier still labels a textured blob a
   golf ball.
3. **"Worst observed recovery" is the maximum over the attack grid tested**, not a bound on what a
   better-tuned attacker could achieve. The attacker response is non-monotone in learning rate.

## Reproducing

Scripts live in the experiment tree (`scripts/deck_snapshot.py`, `deck_images.py`,
`deck_figures.py`, `deck_build.py`, `deck_manifest.py`). The PDF is not a conversion of the PPTX;
both render from one declarative slide spec so they cannot drift.
