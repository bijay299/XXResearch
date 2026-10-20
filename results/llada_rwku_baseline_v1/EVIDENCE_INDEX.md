# Evidence Index

Generated from **tracked files** (`git ls-files`), not a filesystem walk, so every row
below is genuinely published. `EVIDENCE_INDEX.md` is omitted from the table — it cannot
contain its own hash.

Study: `llada_rwku_baseline_v1`. Outcome **INSUFFICIENT_BASE_RECALL** (stopped without training).

## Published files (25)

| Path | Bytes | SHA-256 |
|---|---|---|
| `FINAL_REPORT.md` | 19284 | `d980f1f8acb18e7268804bec56d5eb3dcf2e8d4f14f8a23092f4d56125a37d4b` |
| `FREEZE_NOTE.md` | 4648 | `cd14949a5a141423f38a1e248094f079857c5cf0e2288d1f1e7b5f274523c62c` |
| `README.md` | 2337 | `86a513aa6b10a2a49dc1e35c6f8b42cd7a229333481746fe6217b83444d31dfe` |
| `evidence/ATTEMPTS.tsv` | 1002 | `f435c6bf61c3aed5846cebd34c4016e298de9a69f6e1dc626a5a5b6c868ce6d9` |
| `frozen/frozen_mmlu_panel.json` | 19858 | `ea77d6a995dbcffe6ae8330544811fa4f32c0ddf52bb47a45ad7423cea8b1033` |
| `frozen/frozen_neighbor_availability.json` | 1733 | `d04533b45c032d9f6346626197a10d5c2c325bb945676bb1940038448f497e09` |
| `frozen/frozen_panels.json` | 445126 | `7c7ce60149b3945f9a834e251c3150f963fd8967ee16509574d24a2cd069edfd` |
| `frozen/frozen_spec.json` | 2363 | `84554f932e075dab62d9347f76c274ef97c462e9797de1fba8491b7d127acf6b` |
| `logs/a001_recall_screen10_primary.log` | 1742 | `ec0a84f74525eed178c357e3d619998509a2b4773f82280fcbb30b452410eaf1` |
| `logs/a002_objective_verify_native.log` | 1157 | `4b34adf73f22c4b179c2b4b5d92c139df0efe43132c5df4d1cb1e0ef76af8518` |
| `logs/a002_recall_screen10_secondary_and_mmlu.log` | 1806 | `9e69ea26a39d78ae882b1baafba1624d7d58056596d98c9751943f5e3bd52d89` |
| `logs/a003_objective_verify_native_retry.log` | 381 | `9b9fcff145fd455feab231ee08293e59cfc4e3d73a25711f89d9b75319a22602` |
| `logs/a004_objective_verify_native_fp32tol.log` | 380 | `d330e615decd42d90e5b7ea4e7b45e1c1f8d57496d23ce388db6c48ada71b04e` |
| `logs/a006_evaluator_verify_scoring.log` | 524 | `d3c4648fda8d7b736c954fb93911300dd18671728afef355688c686f87ec643a` |
| `logs/a007_evaluator_verify_scoring_v2.log` | 375 | `b32f4e047311dcd62e5a7ea226ab7d5cdd35c4d1ba75e780ea154f595f2dd8a3` |
| `logs/a008_wiring_teacher_equivalence.log` | 1221 | `bccc2b56e007f4b456d6cd6ca759ff3898dd8fc311fe20135509c1b7d8abee40` |
| `scripts/attempt.sh` | 1335 | `35438768e51a67a8d4a4b00fd316db300ce0ce7b7e0ec8c18342bb849ca033f0` |
| `scripts/env.sh` | 498 | `319358f8e0f2edd43229474584e1675023b003998447295457ebca178f1f2156` |
| `scripts/evaluate.py` | 8553 | `243ef4979b9194c57963684abd85ff96b7d9058e718188ab0915da466d4fe72b` |
| `scripts/freeze_eval.py` | 17440 | `6bb748af1c9792ed0db1a6869e6be17178338272142357f0a602e1f3877b6f4f` |
| `scripts/ledger.sh` | 638 | `e461a663d3b6ba2dd93d29e49ef221254c3b3edff5ca186244d0e40512ba593c` |
| `scripts/mdu_objective.py` | 6189 | `b323420a3d36bb912bcc76ab5a74775a4db312531aab4de04de9a34a7e0c70bd` |
| `scripts/verify_evaluator.py` | 7515 | `f133d2632dc614e0c299d1a58603cdceda2618f53fa3ed1b8f70c4d89212eca0` |
| `scripts/verify_objective.py` | 13491 | `a6a85c1e4e346667b03c5a98a2f98b0e9baa11f6113a42c9ad0e8bbc73673d8a` |
| `scripts/verify_teacher_wiring.py` | 2093 | `f0bade36282343de7a5cd24e621574cb4e63954ad953a62b592c88c75bf9de9e` |

## Pinned sources

| Asset | Revision |
|---|---|
| LLaDA-8B-Instruct (reused) | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| LLaDA official code | `9182493720ed723ef8031210d85959364e51cbe0` |
| MDU | `e5e8e8274ff8f56f16e978803d9e902359eb5272` |
| RWKU dataset | `d72f493d481d1b0a9bdc6e869d32baeffad8904f` |
| open-dlu (reference only, loader NOT adopted) | `fb984334a4007b031246d0b33973644d4845bafd` |
| Pre-inference freeze commit | `203525b6b656aa9220f4ff431e992e3c9ab1ff29` |

## Artifacts that do NOT exist (and why)

| Artifact | Why absent |
|---|---|
| Trained LoRA adapters | **No training occurred.** Pristine recall was INSUFFICIENT_BASE_RECALL, so the protocol required stopping before §5. Zero optimizer steps were taken. |
| Selected-configuration adapter hash + reload evidence | No adapter exists to select or reload. |
| Per-configuration behavioral results | Not reachable without training. |
| Teacher invariance-after-updates check | Requires an optimizer step; equivalence (adapter-disabled ≡ pristine, diff 0.0) **was** verified. |

## Local-only, deliberately not published

| Artifact | Local path | Why excluded |
|---|---|---|
| LLaDA-8B-Instruct weights (14.94 GiB) | `/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct/` | Public at the pinned revision; reused, not re-downloaded |
| RWKU download (131 MB, incl. 34,200-row MMLU source) | `/data/bijaypandey/llada_rwku/data/rwku/` | Public at the pinned revision; the 40 selected MMLU items are published in `frozen/frozen_mmlu_panel.json` |
| Cloned source repos | `/data/bijaypandey/llada_rwku/repos/` | Third-party code, pinned by SHA above |
| venv / caches | reused from the closed smoke study under `/data` | Environment, not evidence |

## Exclusions verified

- **Credentials:** no token was used or created; RWKU and all metadata were fetched
  anonymously. No credential pattern appears in any published file.
- **Model weights:** none staged — no `*.safetensors`, `*.bin`, `*.pt`, and no file above
  2 MB in the published tree.
- **Unrelated artifacts:** nothing from the closed LLaDA smoke study, other research
  branches, or private memory is included.
