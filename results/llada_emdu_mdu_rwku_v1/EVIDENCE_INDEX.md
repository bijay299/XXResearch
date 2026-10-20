# Evidence Index

Generated from **tracked files** (`git ls-files`), not a filesystem walk. `EVIDENCE_INDEX.md`
is omitted from the table — it cannot contain its own hash.

Study `llada_emdu_mdu_rwku_v1`. Outcome **PAPER_ALIGNED_RUN_BLOCKED** (comparison not run).

## Published files (30)

| Path | Bytes | SHA-256 |
|---|---|---|
| `FINAL_REPORT.md` | 15684 | `6413659ab54934c07bd87841232c2773cb747d1d8712548af8ddb110cb2472f6` |
| `README.md` | 2883 | `836eda9c13add89e95d8ca40cf8ed81514b4bd57787b3bdca8d920eccf33d009` |
| `RESOURCE_LEDGER.md` | 2954 | `9b83485d242d88dd6a68ca89cb1f453cc04f2504a7655f512a105461956cb460` |
| `evidence/ATTEMPTS.tsv` | 977 | `fdf2f1599ef4986a5a13dac84cadda17966673c5903c27bcd40157df4a156e07` |
| `evidence/emdu_paper_extracted_text.txt` | 84918 | `741d45a0c478d32fce600972a06bf4144d1eb08650a55b8b028515610ea482eb` |
| `evidence/negative_control/README.txt` | 280 | `54d539495a2d90f323ecebad102fed85f2f69e08081b54e970c06ada6f269d5c` |
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

## Pins

| Asset | Pin |
|---|---|
| EMDU paper arXiv:2610.25447v1 | local PDF sha256 `d8059bebeceda16c2de2055f2127b84640fc1f7808d451e2e22cdb19f0bb915a` |
| MDU | `e5e8e8274ff8f56f16e978803d9e902359eb5272` |
| LLaDA-8B-Instruct | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| LLaDA code | `9182493720ed723ef8031210d85959364e51cbe0` |
| RWKU | `d72f493d481d1b0a9bdc6e869d32baeffad8904f` |

## Artifacts that do NOT exist (and why)

| Artifact | Why absent |
|---|---|
| Trained MDU / EMDU checkpoints | **The comparison was not run** — full-parameter training is budget-blocked (measured). No trained state exists, so **no checkpoint delivery or reload qualification is claimed**. |
| Per-item forgetting / neighbor / utility scores | Not reachable without training. No forgetting number is produced. |
| Pristine-arm generations | Deliberately not produced: a pristine-only arm cannot answer the comparison, and generating it would spend budget on a partial artifact. |
| Paper's EMDU reference code / checkpoints | Do not exist publicly (paper: "the checkpoints we will release"). |

## Local-only, deliberately not published

| Artifact | Local path | Why |
|---|---|---|
| EMDU paper PDF (825 KB) | `/data/bijaypandey/llada_emdu/artifacts/emdu_2610.25447v1.pdf` | third-party copyrighted PDF; its sha256 is pinned above and the extracted text is published |
| LLaDA-8B-Instruct weights (14.94 GiB) | `/data/bijaypandey/llada_smoke/model/` | public at the pinned revision; reused |
| RWKU download (~140 MB incl. 582-pair files) | `/data/bijaypandey/llada_emdu/data/rwku/` | public at the pinned revision; frozen row IDs/counts are published |
| venv / caches | reused under `/data` | environment, not evidence |

## Exclusions verified

- **Credentials:** none used or created; all fetches anonymous. No credential pattern in any published file.
- **Model weights:** none staged — no `*.safetensors`, `*.bin`, `*.pt`; nothing above 2 MB.
- **Storage cap:** respected; **no full-model checkpoint was retained** (see above).
- **Unrelated artifacts:** nothing from the closed studies or private memory is included.
