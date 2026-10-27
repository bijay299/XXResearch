# Supporting material — "Does Concept Forgetting Survive Relearning?"

Backup detail for the 12 slide deck. Presenter: Bijay Pandey, Howard University, 27 October 2026.
Nothing here is required to follow the talk; it exists so every slide claim can be traced.

---

## 1. Evidence snapshot

| | |
|---|---|
| frozen at | see `deck/data/snapshot.json` → `frozen_at` |
| experiment tree | `/data/bijaypandey/t2i_unlearn` (not a git repository; code pinned by sha256) |
| published mirror commit | `deck/data/snapshot.json` → `published_repo_commit` (lags the newest cells) |
| base model | `CompVis/stable-diffusion-v1-4`, revision pinned in `MODEL_PIN.txt` |
| code pins | sha256 of `erase_utrap_v2.py`, `attack_core.py`, `attack_relearn.py`, `attack_relearn_v2.py`, `gen_eval.py` |
| upstream pins | IMMA, Meta-Unlearning, ResAlign, unified-concept-editing git revisions |
| attack cells indexed | 747 |

**Running at freeze.** The snapshot records any job still executing when it was taken. Completed
cells from an unfinished experiment are included and labelled; the experiment as a whole is not
presented as finished.

## 2. Files

| file | what it is |
|---|---|
| `UTRAP_relearning_Pandey_HowardU.pptx` | editable deck, speaker notes embedded |
| `UTRAP_relearning_Pandey_HowardU.pdf` | same slides, same spec, rendered to PDF |
| `speaker_script.md` | the identical notes, exported slide by slide |
| `render/slide_NN.png` | every slide as an image, used for the pre-delivery inspection |
| `figures/*.png` | the five plots |
| `data/*.csv` | one CSV per figure plus `all_attack_cells.csv` (747 rows) |
| `data/snapshot.json` | the frozen snapshot |
| `data/references.json` | references with the verification note for each |
| `data/evidence_manifest.csv` | 282 entries: claim → value → source artifact |
| `images/` | 125 generated images plus `manifest.json` |

There is no LibreOffice on this machine, so the PDF is not a conversion of the PPTX. Both are
rendered from the same declarative slide spec in `scripts/deck_build.py`, so they cannot drift.

## 3. Image protocol

Every qualitative image was produced by inference from a retained checkpoint. No defence training
was run to populate the deck.

| setting | value |
|---|---|
| sampler / scheduler | as shipped with SD-1.4 (recorded in `images/manifest.json`) |
| inference steps | 25 |
| guidance scale | 7.5 |
| resolution | 512 x 512 |
| generator seed | 0 |
| batch shape | `pipe([prompt] * 5, generator=seed 0)`, matching how the stored panels were written |

**Selection rule, fixed before any output was inspected:** image index 0 of prompt index 0 of the
held-out H pool.

Five images per prompt were generated. This repository carries the 29 that appear on slides; the
116 unselected variants are not committed, because the repository's `.gitignore` excludes bulk
image directories and they add 54 MB. They remain in the experiment tree at
`/data/bijaypandey/t2i_unlearn/deck/images/` and every one of them is listed in
`images/manifest.json`. Generation is deterministic at a fixed seed, proven by the
settings-match self-check below (maximum absolute pixel difference 0), so re-running
`scripts/deck_images.py` reproduces them bit for bit. Not committing them therefore costs nothing
in verifiability.

**Settings-match self-check.** The generator was pointed at an arm that already had a stored panel
image and asked to re-render it. Result: **identical, maximum absolute pixel difference 0**. That is
what licenses showing newly rendered methods beside previously stored ones in the same panel.

**Prompts used**

- church target: `a photo of a church` (held-out H pool, never seen by the attacker)
- unrelated: `a photo of a golf ball`, `a photo of a french horn`
- Van Gogh target: `a painting in the style of Van Gogh`

## 4. Method inventory and fidelity tier

Nothing in this project is an official reproduction, which would require the authors' code **and**
their protocol, metric and data. Three tiers are used.

| method | arm | tier | on slides |
|---|---|---|---|
| SD-1.4 | base | reference model | 4, 5, 7, 10 |
| ESD | `esd_church` | our from-paper reimplementation | 4, 6 |
| ESD Van Gogh | `ESD_vangogh_official` | authors' released checkpoint | 10 |
| UCE | `uce_church` | our from-paper port (upstream compared numerically) | 5, 6 |
| GIFT | `gift_church_b1.0` | our from-paper port; no official implementation exists | 5, 6 |
| FRAG/FRP | `frp_church_r3_l05` | our from-paper port; upstream ships no implementation | 4, 6, 7, 8 |
| IMMA (adapted) | `imma_church_frp_a1e-7_s1500` | domain adaptation onto our FRP teacher | 5, 6 |
| IMMA (official code) | `REF_imma_vangogh_s20` | **official-code execution**, our data and evaluation | 10 |
| UTRAP K=0 | `utrap2_church_l0.15k0` + 2 more builds | our method, control arm | 4, 6, 7, 8, 9 |
| UTRAP K=8 | `utrap2m_church_l0.15k8`, `u4_k8_s400_b{0,1,2}` | our method | 4, 5, 6, 7, 8 |

### 4.1 Excluded from any claim about a published method

| family | reason |
|---|---|
| AdvUnlearn (text-encoder arms) | post-attack evaluation loaded the **base** text encoder, so every text-encoder-arm post-attack number is void |
| Meta-Unlearning (our port) | our port omits two of the three terms of the paper's Eq. 8 meta objective, so it is not the published method |
| WSR / weight-space regulariser | implements the **opposite** regularisation direction from the paper it cites |
| `REF_imma_vangogh` (5x budget) | ran at 5.00x the README's documented ascent budget; utility collapsed (retain8 0.0333). Kept as a utility-failure datapoint, not a method comparison |
| internal h-series, random, undo variants | project-internal variants, not published methods |

These are excluded from *claims*, not hidden: each is named on slide 5 and in the manifest.

## 5. What the metrics do and do not mean

**H** = fraction of 15 images (3 held-out prompts x 5 images) that a fixed 12-way closed-set CLIP
classifier labels as the target. Denominator 15, so the quantum is 0.067. **There is no reject
option**: every image is forced into one of twelve labels. A high H means "the classifier picked the
target", not "the image is a good picture of the target".

**retain8** = closed-set accuracy over 8 never-targeted concepts, 120 images. **It is not an image
quality measure.** Slide 4 demonstrates this directly: three methods visibly degrade an unrelated
concept while scoring 0.87 to 0.97 on it, because a textured blob is still classified as a golf
ball. FID is not computed; `docs/FID_PROTOCOL.md` records why and what would be required.

**Validity gate** = an attack counts only if `T_rise >= 0.20` on the 24-image training-prompt panel,
i.e. the attacker demonstrably relearned what it was given. On a 24-image panel this correctly
accepts 5/24 = 0.2083 and rejects 4/24 = 0.1667.

**Gate-invalid cells are included and marked, never dropped.** A failed gate means *inconclusive*,
not *successful defence*. Counts are printed on every figure that uses them.

**"Worst observed recovery"** means the maximum over the attack grid actually tested. It is not a
bound on what a stronger or better-tuned attacker could achieve.

## 6. Attack grids actually run

- budgets 10 / 25 / 50 / 100 gradient steps
- attacker learning rates 1e-5, 5e-5, 1e-4
- 3 attack seeds per cell (0, 1, 6)
- 3 independent defence builds per UTRAP arm
- full per-cell table: `data/all_attack_cells.csv`

**Attacker response is non-monotone in learning rate.** On UTRAP K=0 build 0 at 25 steps, mean H is
0.267 at lr 1e-5, 0.933 at 5e-5 and 0.600 at 1e-4. A larger learning rate is not automatically a
stronger attack, so robustness can only be read as the maximum over a sweep.

## 7. Scope limits that bound every conclusion

1. One main concept (church) and one second concept (Van Gogh style). No claim beyond these.
2. One teacher model for all UTRAP arms. UTRAP-on-ESD would be a separate experiment.
3. Three defence builds per UTRAP arm. Baselines are n=1 build.
4. Development data throughout. No confirmatory protocol was frozen, so no confirmatory claim is
   available regardless of outcome.
5. The IMMA cell is an official-code execution on our dataset and our evaluation. It is **not** a
   reproduction of the paper's reported numbers, and it is **not** comparable to the church results,
   which were produced under different conditions.
6. H carries run-to-run noise of about 2/15 on some cells, measured by repeating one cell four
   times (0.133, 0.200, 0.067, 0.133) with the attack trajectory identical each time.

## 8. Claims deliberately absent from the deck

These were made earlier in the campaign and withdrawn when further measurement contradicted them.
They are listed so nobody reinstates them from an older summary.

1. A 0.57 separation between UTRAP K=0 and the teacher at 25 steps. **Withdrawn**: an artifact of
   measuring at a single attacker learning rate.
2. "UTRAP K=0 meets the forgetting/utility bar at 25 steps." **Withdrawn**: n=1 build artifact; a
   second build gives a gate-valid 0.4667 and the calibrated sweep gives 1.000.
3. "More outer updates make K=8 erase worse." **Withdrawn**: the build-to-build spread (0.533)
   exceeds the effect.
4. "K=0 costs utility." **Withdrawn**: the K=0 retain8 build range spans the teacher.
5. "Representation damage tracks resistance." **Withdrawn**: does not survive attacker calibration.

## 9. Reproducing the deck

```
venv_t2i/bin/python scripts/deck_snapshot.py     # freeze the evidence snapshot and CSVs
venv_t2i/bin/python scripts/deck_images.py       # generate images (needs one reserved GPU)
venv_t2i/bin/python scripts/deck_figures.py      # build the plots from the CSVs
venv_t2i/bin/python scripts/deck_build.py        # render PPTX, PDF and per-slide PNGs
venv_t2i/bin/python scripts/deck_manifest.py     # write the evidence manifest
```
