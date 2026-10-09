# DRAFT — NOT SENT

**Status:** awaiting review. Do not send without approval.
**Prepared:** 2026-10-09
**Suggested recipients:** CUIG maintainer (github.com/justinhylee135/CUIG — issue or email)
and the UnlearnCanvas maintainers (github.com/OPTML-Group/UnlearnCanvas — issue).
**Suggested channel:** one GitHub issue per repository, cross-linked.
**Before sending:** replace the signature block; decide whether to name the host
(`hyperplane`) or keep it generic; confirm we are willing to state that Drive is
reachable from our network but folder listing is not.

---

**Subject:** UnlearnCanvas generator and classifier checkpoints — download links return 404; requesting working links plus class-order and preprocessing details

Hello,

Thank you for releasing CUIG and UnlearnCanvas. We are setting up the
single-concept ConAbl baseline on UnlearnCanvas (CUIG @
`9932ac3271a122f6d38d19e0b8c8908fe5237ff7`, `BashScripts/Independent/Style/Base/ConAbl.sh`)
as the starting point for research on sequential concept erasure, and we have hit
a blocker on the checkpoints.

## 1. The documented download links return HTTP 404 for us

Every Google Drive folder referenced in the two repositories returns a 404 page
when fetched from our host:

| Folder ID | Referenced by | Contents | Observed |
|---|---|---|---|
| `1AoazlvDgWgc3bAyHDpqlafqltmn4vm61` | CUIG `Checkpoints/README.md` | style/object classifiers (`cls_model`) | HTTP 404 |
| `18x40pLBcfNFyxBWZBGncTjqJTs_75SLx` | CUIG `Checkpoints/README.md` | generator (`style50`) | HTTP 404 |
| `18dhkXyZQWjdMvlAlxZx3fZhdCZvlj2Hw` | UnlearnCanvas `README.md`, `machine_unlearning/README.md`, `style_transfer/README.md` | all paper checkpoints | HTTP 404 |
| `14iztBXs-GoBFVLePC2_psP00YUMK5-cy` | UnlearnCanvas `diffusion_model_finetuning/README.md` | finetuning checkpoints | HTTP 404 |
| `1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM` | UnlearnCanvas `README.md` | dataset | HTTP 404 |

`gdown --folder <any of the above>` reports `Retrieving folder contents / Failed
to retrieve folder contents` (gdown 5.2.1).

We want to be clear that **we have not established the cause**, and we are not
assuming the folders were withdrawn. From our host `https://www.google.com`
returns 200 and `https://drive.google.com/` returns 302, so general connectivity
is fine; but the last row above is the dataset folder your README describes as
publicly available, and it 404s for us too. That is consistent with
unauthenticated Drive *folder* access being refused from our network, and equally
consistent with a permissions change on your side. We cannot distinguish the two
from outside, so we would rather ask than guess.

Could you confirm whether these folders are still shared publicly, and if so
point us at links that work for unauthenticated download (or a mirror —
HuggingFace, Zenodo, or an institutional host would all be ideal)?

The assets we need are:

- the UnlearnCanvas generator in **diffusers** format (CUIG's
  `Checkpoints/Generators/UnlearnCanvas/`, upstream name `style50`);
- the style classifier (`style50.pth` → `style_classifier.pth`);
- the object classifier (`style50_cls.pth` → `object_classifier.pth`).

We have found no mirror of the two ViT-Large classifiers in the sources we
searched (HuggingFace models and datasets, the `OPTML-Group` namespace, and the
`UnlearnCanvas-Benchmark` Space). The `OPTML-Group/UnlearnCanvas` dataset on
HuggingFace is reachable and has been very useful, but it contains images only.

## 2. Questions we would like to resolve regardless of the download

These affect whether our numbers are comparable to yours, and we would rather
record your answers than infer them.

**a) Class index order.** `Evaluation/UnlearnCanvas/evaluate.py` maps a concept
name to an integer with `STYLES_AVAILABLE.index(name)` and
`OBJECTS_AVAILABLE.index(name)` from `constants.py`. Both lists are exactly
alphabetically sorted. Can you confirm the released classifiers were trained with
that same ordering (e.g. `torchvision.datasets.ImageFolder` over alphabetically
sorted class directories)? A permuted ordering would be shape-identical and would
silently corrupt UA/IRA/CRA, so we would like it confirmed rather than assumed.

**b) The 51st style class.** `STYLES_AVAILABLE` has 51 entries — 50 styles plus
`Seed_Images` — while the released dataset covers 60 styles. Could you confirm
that `style50.pth` is the 51-way head (and that `style60.pth`, which
`Checkpoints/README.md` instructs users to delete, is the 60-style variant)?

**c) Evaluation preprocessing.** `evaluate.py` uses
`Resize((224, 224)) → ToTensor() → Normalize(mean=[0.5], std=[0.5])`, and calls
`Image.open` without an explicit `.convert("RGB")`. Is that identical to the
preprocessing used when the classifiers were trained?

**d) Generator configuration.** Which base checkpoint was fine-tuned (SD v1.4 or
v1.5), at what resolution and for how many steps, and — for the diffusers
release — were the weights taken from the **EMA** or the **non-EMA** parameters?
We ask because the only copy of the generator we have located is a third-party
CompVis-format mirror that carries both `model.` and `model_ema.` tensors, so
converting it ourselves would require choosing, and the choice changes the model.

**e) Checksums.** If you can publish SHA-256 sums for the generator files and the
two classifiers, we can verify any copy we obtain actually matches your release.
Without a published checksum we can only hash what we hold, which identifies the
bytes but does not establish that they are yours.

## 3. What we are doing meanwhile

We have pinned CUIG, reproduced the environment from `env.yaml`, and validated
the ConAbl training/sampling/checkpoint path end-to-end on a substitute
generator. We are deliberately **not** reporting any UA/IRA/CRA, and our tooling
refuses to run the baseline until the real assets pass a validation step
(classifier label-order and accuracy checks against labelled held-out images from
your HuggingFace dataset, plus an untouched-generator reference run).

Happy to share our setup notes if useful, and happy to open a PR updating the
download instructions once we know what the working links are.

Thank you for your time and for releasing the benchmark.

Best regards,
*[name]*
*[affiliation]*

---

## Reviewer notes (not part of the message)

- Tone is deliberately non-accusatory on the 404s: our own evidence is
  ambiguous, and the dataset-folder result argues against "they deleted it".
- Item (d) matters most for comparability; (a) matters most for correctness.
- (e) is the only thing that would let us *prove* an obtained copy is theirs.
- If the maintainers go quiet, fallbacks are: ask for an institutional transfer,
  or ask whether they will accept results computed with a classifier we retrain
  on the public dataset (which would be a different, clearly-labelled metric).
