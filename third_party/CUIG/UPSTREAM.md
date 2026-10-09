# Upstream: CUIG (Continual Unlearning for Image Generation)

This pilot builds on the official CUIG implementation. **No upstream source is
copied into this repository.** Upstream is obtained by cloning at a pinned
commit; our own configs, launch scripts, validators and notes live here and
reference it by path. This keeps the boundary between upstream code and our
modifications unambiguous.

## Pinned commit

| Field | Value |
|---|---|
| Repository | https://github.com/justinhylee135/CUIG |
| Commit | `9932ac3271a122f6d38d19e0b8c8908fe5237ff7` |
| Commit date | 2026-07-13 18:03:12 -0400 |
| Commit subject | `Remove completed to-do` |
| Branch at clone | `main` (repository HEAD at clone time) |
| Local checkout | `/data/bijaypandey/cuig_pilot/CUIG` (git-excluded) |
| Checkout size | 68 MB |

## Reproducing the checkout

```bash
mkdir -p /data/bijaypandey/cuig_pilot && cd /data/bijaypandey/cuig_pilot
git clone https://github.com/justinhylee135/CUIG.git
cd CUIG && git checkout 9932ac3271a122f6d38d19e0b8c8908fe5237ff7
```

`scripts/preflight_check.py` asserts the checkout is at this exact commit with a
clean working tree, so drift is caught before a run.

## Paper

> Justin Lee, Zheda Mai, Jinsu Yoo, Chongyu Fan, Cheng Zhang, Wei-Lun Chao.
> *Continual Unlearning for Text-to-Image Diffusion Models: A Regularization
> Perspective.* ICLR 2026. arXiv:2511.07970

```bibtex
@inproceedings{lee2026continual,
  title={Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective},
  author={Lee, Justin and Mai, Zheda and Yoo, Jinsu and Fan, Chongyu and Zhang, Cheng and Chao, Wei-Lun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Licence

CUIG is released under the **MIT Licence**, Copyright (c) 2025 Justin Lee. The
full licence text as published at the pinned commit is preserved verbatim in
[`LICENSE.upstream`](LICENSE.upstream).

## Further attribution carried by upstream

CUIG itself builds on prior work, which must be credited alongside it:

- **Concept Ablation** — CUIG's `UnlearningMethods/ConAbl` is based on
  [nupurkmr9/concept-ablation](https://github.com/nupurkmr9/concept-ablation),
  with CUIG-specific changes for continual unlearning, regularizer integration
  and benchmark hooks. Portions of `src/model.py` are marked in-file as taken
  directly from the original Concept Ablation `main()`.
- **UnlearnCanvas** — the style/object benchmark, generator and classifiers come
  from [OPTML-Group/UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas)
  (Zhang et al., 2024; arXiv:2402.11846). CUIG's
  `Evaluation/UnlearnCanvas/README.md` asks that the UnlearnCanvas authors be
  credited when using this evaluation pipeline or checkpoints. We do.
- **Stable Diffusion** — the generator is a Stable Diffusion v1.x derivative and
  carries the CreativeML OpenRAIL-M licence and its use-based restrictions.

## Our modifications to upstream behaviour

We do not patch the upstream tree. Behavioural changes are expressed entirely in
our own files and are enumerated in [`../../docs/DEVIATIONS.md`](../../docs/DEVIATIONS.md).
If a future change does require editing upstream code, add it as a patch file
under `third_party/CUIG/patches/` and apply it in a script, so the delta against
`9932ac3` stays reviewable.
