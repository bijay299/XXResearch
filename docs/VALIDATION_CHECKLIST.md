# Remaining validation checklist

What must still happen before any UA / IRA / CRA may be reported. Items marked
**[blocked]** cannot start until the benchmark assets are in hand.

Legend: `[x]` done · `[ ]` outstanding · `[blocked]` waiting on assets

---

## 0. Already done (no assets required)

- [x] Hardware verified: 4 × A100-SXM4-40GB, 40960 MiB each, driver 550.144.03.
- [x] Environment pinned to upstream `env.yaml`; versions recorded in
      `env/requirements.lock.txt`.
- [x] Upstream CUIG pinned at `9932ac3271a122f6d38d19e0b8c8908fe5237ff7`;
      licence and attribution preserved.
- [x] Training/sampling/checkpoint path exercised end-to-end on a substitute
      generator; losses finite, `delta.bin` reloads with 0 unexpected keys.
- [x] `evaluate.py` code path exercised in isolation (synthetic heads).
- [x] Asset gate implemented and **negative-control tested**: shape-correct
      random heads are rejected.
- [x] Labelled held-out probe set built from the UnlearnCanvas HF dataset
      (137 images, 10 styles, 20 objects, 0 unparsed captions).
- [x] Machine-readable manifest with explicit `unknown` fields.
- [x] Maintainer request drafted (`docs/outbox/asset-request-DRAFT.md`) — **not sent**.

---

## 1. Obtain the assets

- [ ] Send the maintainer request (needs review/approval first).
- [ ] Obtain `style_classifier.pth`, `object_classifier.pth`, and the diffusers
      generator directory by any working route.
- [ ] Place them at the paths in `docs/ASSETS.md` §2.
- [ ] `python scripts/record_asset_hashes.py --source-url ... --revision ...`
      → state becomes `PRESENT_UNVERIFIED`; SHA-256, size, path recorded.

---

## 2. Classifier validation **[blocked]**

- [ ] **Shape** — heads are `Linear(1024, 51)` and `Linear(1024, 20)`; weights
      under `model_state_dict`.
      *Necessary only. Never sufficient — a permuted checkpoint is shape-identical.*
- [ ] **Label order** — top-1 on held-out labelled images under the
      `constants.py` ordering, plus modal-predicted-index per true class.
      Verdict must be `confirmed`.
      - `refuted_permuted` → the checkpoint is informative but ordered
        differently; **do not** compute metrics with `constants.py` ordering.
        Recover the true permutation and raise it with the maintainers.
      - `inconclusive` → neither ordering supported; treat the checkpoint as
        unusable until explained.
- [ ] **Accuracy** — top-1 clears the threshold (default 0.70; local policy,
      recorded in the receipt, *not* a published value). Revisit the threshold
      against whatever accuracy the maintainers state for the released
      classifiers.
- [ ] Confirm `Seed_Images` handling: `style50.pth` is the 51-way head, and
      dataset styles outside the list (`Dreamweave`, `Pointillism`, …) are
      excluded rather than scored.

## 3. Generator validation **[blocked]**

- [ ] Pipeline loads: `model_index.json` + `scheduler/ text_encoder/ tokenizer/
      unet/ vae/`.
- [ ] **Untouched-generator reference run** — sample with the released config
      (CFG 9.0, 100 steps, 512 px, seed 188) *before any unlearning* and score
      with the validated classifiers. This is the pre-unlearning baseline and
      the check that catches a generator that was never fine-tuned on
      UnlearnCanvas.
- [ ] Record per-style and per-object pre-unlearning accuracy as the reference
      point every later UA/IRA/CRA is read against.
- [ ] Resolve **EMA vs non-EMA** (`generator_ema_selection` is `unknown`). If the
      generator has to be converted from a CompVis checkpoint, this choice
      changes the model and must be stated, not guessed.

## 4. Gate **[blocked]**

- [ ] `python scripts/validate_assets.py --gpu <idle> --update-manifest`
      → `assets_verified: true`.
- [ ] `python scripts/preflight_check.py --gpu <idle>` → `BASELINE GATE: OPEN`.
- [ ] Confirm `run_single_concept_conabl.sh` no longer exits 5.

---

## 5. Baseline run **[blocked]**

- [ ] Re-check GPU idleness immediately before launch; pin one device.
- [ ] `bash scripts/run_single_concept_conabl.sh --style Abstractionism`
      at the preserved released settings (`docs/DEVIATIONS.md` §6.1).
- [ ] Verify the sampled grid is complete — `13 × 8 × 5 = 520` images. `evaluate.py`
      skips missing images with a warning, so an incomplete sweep still yields a
      `summary.json`; check the count before trusting it.
- [ ] Record from `run_manifest.json`: GPU identity, peak memory, per-stage
      timings, checkpoint path.
- [ ] Report UA / IRA / CRA **with** the pre-unlearning reference from §3.

## 6. Before any number is called a reproduction

- [ ] Provenance resolved — a checksum published by the authors, or a copy
      received directly from them. Until then `*_provenance` stays `unknown`,
      and results are "our run of the released configuration", not "the paper's
      numbers".
- [ ] `preprocessing_matches_classifier_training` confirmed by the maintainers.
- [ ] Repeat across the other 11 unlearn styles before drawing any conclusion
      from a single concept.

---

## Fields that remain `unknown` even after a full technical PASS

Behavioural agreement does not establish origin. These are carried explicitly in
`assets/asset_manifest.json` and `results/assets/validation_receipt.json`:

| Field | Resolved only by |
|---|---|
| `classifier_provenance` | authors' checksum or direct copy |
| `generator_provenance` | authors' checksum or direct copy |
| `generator_ema_selection` | maintainer answer |
| `preprocessing_matches_classifier_training` | maintainer answer |
| generator version / base model / finetune steps / resolution | maintainer answer |
