# Pre-freeze activity log (cpr_direct_agreement_controls_v1)

Everything below happened BEFORE the prospective freeze commit. No numerical loss test, detector inference, FLUX forward
pass or image output exists at freeze time. Task clock: first task action 2026-10-19T17:31:02Z.

1. Records: protected git paths at `30f9559…`, pre-task releases/tags, COCO annotation hashes (`records/`).
2. Metadata-only design (annotation JSON only): `scripts/select_design.py` -> `configs/{exclusion_inventory,design_enumeration,design}.json`;
   triple [6,9,10] accepted first (dev target refrigerator; held-out toilet, bed); 9 scored classes; 333 unique qualification photographs planned.
3. Supervisor smoke checks (`scripts/smoke_supervisor.py`, `artifacts/smoke_supervisor.json`, `logs/smoke_*.log`): mechanical, no model work, 0 GPU.
   An earlier version of this smoke check failed (cap/overrun assertion) and was fixed before the passing run; the failing run is not a launch of model work.
4. Native GEM checkout at the pin `c328fc6…` and attributed patches (`scripts/apply_patches.py`, `patches/`): P1a capture hook, P1b loss hook (None => original
   native block), P2 preview-decode skip (counted), P3 step hook, P4 optional `datasets`, P5a/b optional `wandb`, P6 optional `Flux2Pipeline`, P7 local
   snapshot path override, **P8 optional `QwenImagePipeline` and P9 optional `RepoFolder` (added during the import preflight, see 6)**.
5. Analysis code and 17 synthetic-data tests (`scripts/test_analysis.py`, `artifacts/analysis_synthetic_tests.json`): CPU/numpy only.
   One fixture was redesigned (noisy planted effect) and a dummy assertion replaced before the passing run.
6. Import/structure preflight (`scripts/prefreeze_structural.py`, CPU only, supervised through the ledger with `devices=[]`; zero GPU-h, CPU-core-h charged):
   * run 1 `prefreeze.structural` FAILED at import: `QwenImagePipeline` missing from diffusers 0.30.2 -> P8 (log: `logs/prefreeze.structural.run1_failed_QwenImage_import.log`)
   * run 2 `prefreeze.structural2` FAILED at import: `RepoFolder` missing from installed huggingface_hub -> P9
   * run 3 `prefreeze.structural3` FAILED in my own script (`chk()` keyword collision `n=`) after PEFT inventory checks
   * run 4 `prefreeze.structural4` FAILED, same keyword collision at a second call site
   * run 5 `prefreeze.structural5` PASSED (`artifacts/prefreeze_structural.json`, `all_pass=true`)
   These were import/harness defects found before the freeze; they are not post-freeze repair cycles (repair cycles used: 0 at freeze). The structural check builds the
   native PEFT model on CPU, round-trips an adapter, checks digests/hook signatures and re-derives every case's noise hash from its seed. It performs no forward
   pass and computes no loss. Peak RSS 34 GiB, 617 CPU-s.
7. GPU state: 4 x 40 GB devices idle at last check; 0 GPU-h charged; 0 images; 0 COCO photographs.
