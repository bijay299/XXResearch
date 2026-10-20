# Evidence Index

Generated from **tracked files** (`git ls-files`), not a filesystem walk. `EVIDENCE_INDEX.md`
is omitted — it cannot contain its own hash.

Study `llada_emdu_mdu_rwku_v1`. Outcome **PAPER_ALIGNED_RUN_BLOCKED**, scoped to the tested
custom CPU-offload implementation. Comparison not run. Corrections in
[`AMENDMENT_01.md`](AMENDMENT_01.md).

Reviewed publication: [`cc54d8389cb06223bb9296e7afa620afea2fa6e8`](https://github.com/bijay299/XXResearch/commit/cc54d8389cb06223bb9296e7afa620afea2fa6e8).

## Published files (33)

| Path | Bytes | SHA-256 |
|---|---|---|
| `AMENDMENT_01.md` | 15424 | `dae94e7decfa75da6cbddc2e1cdf64600bf98186c5ee6928bea95008e75bb1a1` |
| `FINAL_REPORT.md` | 21040 | `92e687542442216ec03d3927874705edbdd934cbe62bac73fa7aad45a6d7d629` |
| `README.md` | 4275 | `54a85648fafe43db621433d38bf78e3e597bb680225cbe397588dd88b9daef82` |
| `RESOURCE_LEDGER.md` | 4437 | `23841d6f9bab1c73aee004905cbccfdde68dbbcb850c39829206ac35b04fbb74` |
| `evidence/ATTEMPTS.tsv` | 977 | `fdf2f1599ef4986a5a13dac84cadda17966673c5903c27bcd40157df4a156e07` |
| `evidence/emdu_paper_extracted_text.txt` | 84918 | `741d45a0c478d32fce600972a06bf4144d1eb08650a55b8b028515610ea482eb` |
| `evidence/negative_control/README.txt` | 1082 | `1573295477cb64dbb5cec0a87d8c00f7591fa79438d21946bd0c4227c6fbc007` |
| `evidence/negative_control/broken_variant_diff.txt` | 152 | `ff50dc98c7cca001b6eda41216eef9d4393d487c820ca2b4e56454248079ebc9` |
| `evidence/negative_control/objectives_broken_tau.py` | 6518 | `c46ae3ec2f794263ef1dea4fa5137413203d63ae8223d0ebdd4ebfeae22bce78` |
| `frozen/stageb_mmlu.json` | 18938 | `ccc29352fc9f907e1ec7bd4f70c02db0902a83cb69cea6e1c8d474e98001918c` |
| `frozen/stageb_panels.json` | 121511 | `fb443578ae085650b575ac25e0697a466d8877f0f7995615f2685b8d4b19cce2` |
| `frozen/stageb_spec.json` | 6126 | `c38373075a92506382f6024135658f1cf7c9b5bc2273528cca06c78862d6d90c` |
| `frozen/stageb_trivia.json` | 24334 | `e50ae284afc78c0d5f4b15c6285afaaf0b53a19dcd1cfd667b5f9bee39b9675e` |
| `logs/b0001_stageA_audit_offline_recompute.log` | 597 | `f7a52bbbb593e5dcd71fa0204d530dd97618a0fae9de427713dca76a8d7529e4` |
| `logs/b0002_stageB_feasibility_fullparam_probe.log` | 1637 | `4f30f4386b84a3c77f3e2bc79d2a7eca9f88b9f7eadfa241899a580790666568` |
| `logs/b0003_stageB_feasibility_fullparam_repair.log` | 1614 | `f2a0fd867f117f6a344544214e517b004b0da6cb20abbd4e28284af213dbda49` |
| `logs/b0004_stageB_freeze_protocol_and_eval_ids.log` | 790 | `8ed535e04dab5c3945e6b28160f4b022bb85b49e7ef24dae502589267a1efba3` |
| `logs/b0005_stageB_verify_objectives_cpu.log` | 1066 | `53156342e3b1e986005d146c7c76112d584b64b19e874d65d0ffe894ecd84669` |
| `logs/b0006_stageB_verify_runtime_frozen_teacher.log` | 1744 | `19dcf3f88684a95f472fe0e3e3641360d89fd23a4aa5fc95f91c1afd3d9df408` |
| `outputs/feasibility_probe.json` | 1414 | `7a526fc84a56934c15ce54a51ef503118818b2435f00f718c12655c2aaeb3e71` |
| `outputs/feasibility_repair.json` | 1747 | `770e1fcfe4658641d26be56c4cd6389b636195ca35ae2ac1461c741e874e4d54` |
| `outputs/stageb_objective_checks.json` | 4141 | `99ae5610f401e0908555673d5437c7a7f1dcc8d96e5c36197d820bcfe0fba232` |
| `outputs/stageb_runtime_checks.json` | 3476 | `c6f31ea6cd23eccf96d5f6bc7cc4e5e86a288e7051531210ca3e9b5060c19bbb` |
| `scripts/attempt.sh` | 1856 | `22e8be62b3c9a984f7f9075e98eacaf9a1cef43e0f5de64027972daf8c0db75c` |
| `scripts/audit_stage_a.py` | 8494 | `ab5cbcb2b6b9099f25d8654f9005d08ddc6429d8e2d2fe64b3575f2191479604` |
| `scripts/env.sh` | 500 | `e81ef6278fb5e685f8a4f8e492b2ecb4c4a9571461ede837ce854adb05d3357c` |
| `scripts/feasibility_probe.py` | 9512 | `fa583fbe6af6c0c062891e6ab10008f88afc1944eacaa6feb576e8eeba209cbb` |
| `scripts/feasibility_repair.py` | 7647 | `2784458edc311b0546194fccb7221963f719b415f33ced97783c7156ba5ba6df` |
| `scripts/freeze_stage_b.py` | 13187 | `ff74fee1c23c61859de332966676f813a82d893ba111de1f0648655479904ed9` |
| `scripts/ledger.sh` | 1105 | `5a4e788d9ae6cf41d8a4fe2b239acfdfd5d37b5f8daeea94310870fd0f9676e3` |
| `scripts/objectives.py` | 6538 | `5c9b90796e36bbcf43261c4fecd8292a2e1ac114da0b93d2ad113ea0ad51d675` |
| `scripts/verify_runtime.py` | 9598 | `17cca77065f6757706bba536e802f1e288c96f9c835465ea185f6d879782df88` |
| `scripts/verify_stage_b.py` | 9949 | `218f0e0893f09a3e79cecea7f4070f4b4cb31157b8102005025c020586ab5440` |

## Amendment 01 (2026-10-20)

Additive corrections; **no new experiments** (no model load, no GPU, no generation, no
optimizer update, no re-run test, no repeated search, no frozen-code change).

- **CPU accounting corrected.** `evidence/ATTEMPTS.tsv` row `b0003` carries the raw cell
  `026063`; `scripts/ledger.sh` summed the column with shell arithmetic, which read it as
  **octal** (`11315`). Decimal total: `0+27233+1+152+26063+4891` = **58,340 core-s =
  16.2056 core-h**; previously reported 43,592 / 12.1089 — an undercount of **14,748 core-s
  = 4.0967 core-h**. The rows, counters and `ledger.sh` are **preserved unchanged** as
  historical evidence; `RESOURCE_LEDGER.md` is the corrected *derived* ledger.
  **Historical CPU remains unmetered, so 3.7944 core-h is at most the remaining allowance
  before unknown earlier usage — not a verified whole-envelope budget.** No overall
  CPU-compliance claim is made.
- **GPU unchanged:** 323 + 548 + 660 + 352 = **1,883 GPU-s ≈ 0.5231 GPU-h**, leaving
  **19,717 s** of the 21,600 s ceiling; max **2** concurrent GPUs preserved.
- **Withdrawn:** the single-80 GB-GPU recommendation and its "minutes" prediction (the
  stated layout is ≈134.37 GiB at 18 B/param; the bare optimizer state alone is ≈89.58 GiB);
  the bandwidth-bound attribution; and the native batch-reduction equivalence claim.
- **Narrowed:** extracted-native-helper execution vs complete-trainer verification;
  EMDU-only optimizer updates (2); 40/291 digest coverage; unconditional assertions;
  configured-vs-executed evaluator; chronology relative to the freeze.

## Artifacts that do NOT exist (and why)

| Artifact | Why absent |
|---|---|
| Trained MDU / EMDU checkpoints | **Comparison not run.** No trained state exists; **no checkpoint delivery or reload qualification is claimed.** |
| Per-item forgetting / neighbor / utility scores | The **evaluator was never executed**. Panels and scorers are configured only. |
| MDU optimizer trajectory | Only **2 EMDU** optimizer updates ran, as a correctness probe. |
| Full-model (291-tensor) digest invariance | Digests sampled **40 of 291** tensors. |
| Stored exit codes for the negative controls | Both logs record only check verdicts; the codes were observed in the shell and are **not** captured in any artifact. The controls were **not** re-run. |
| A normal-run Stage A audit JSON | Overwritten by its negative control (see Stage A Amendment 02); not recovered, not manufactured. |
| EMDU author code / checkpoints | Not located in the sources inspected 2026-10-20; that search was not repeated. |

## Local-only, deliberately not published

| Artifact | Local path | Why |
|---|---|---|
| EMDU paper PDF (825 KB) | `/data/bijaypandey/llada_emdu/artifacts/emdu_2610.25447v1.pdf` | third-party copyrighted PDF; sha256 pinned in the report, extracted text published |
| LLaDA-8B-Instruct weights (14.94 GiB) | `/data/bijaypandey/llada_smoke/model/` | public at the pinned revision; reused |
| RWKU download (~140 MB) | `/data/bijaypandey/llada_emdu/data/rwku/` | public at the pinned revision; frozen row IDs/counts published |
| venv / caches | reused under `/data` | environment, not evidence |

## Exclusions verified

- **Credentials:** none used or created; no credential pattern in any published file.
- **Model weights:** none staged — no `*.safetensors`, `*.bin`, `*.pt`; nothing above 2 MB.
- **Storage cap respected:** no full-model checkpoint retained (none exists).
- **Unrelated artifacts:** nothing from other studies or private memory.
