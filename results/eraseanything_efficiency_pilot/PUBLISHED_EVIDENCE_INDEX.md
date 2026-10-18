# Published evidence index - EraseAnything efficiency pilot

**EXECUTED PILOT, status INCOMPLETE.** Effectiveness measurement is blocked by
scorer adequacy; a measured engineering result is delivered. See `README.md`.

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `cb9cf23dc7cb` - the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

**Image originals are NOT in git** (441 newly decoded PNGs, too large for sensible
git storage; no release upload was available). `evidence/IMAGE_MANIFEST.json`
carries byte counts and SHA256 for all 430 retained originals, which remain on the
work volume at `/data/bijaypandey/cuig_pilot/eraseanything_efficiency_pilot/`.
Fixed-order review grids for all 9 M0 concept x panel combinations ARE published.
No model weights and no credentials are committed.

Regenerate: `python scripts/seq/publish_eraseanything_efficiency_pilot_evidence_index.py`
- run it LAST.

## Reports and manifest (top level)

| path | bytes | sha256 |
|---|---|---|
| `results/eraseanything_efficiency_pilot/.gitignore` | 427 | `bcf6bb62ec9c56c5129943794ebc644cee8e021d987444925acc08bbfaeb53c7` |
| `results/eraseanything_efficiency_pilot/AMENDMENT_01.md` | 13,337 | `5db3bd8c648f102dab70996ff4fd298d4e1991ee74648c3d2966c700ae5a3b25` |
| `results/eraseanything_efficiency_pilot/AMENDMENT_02_CLOSEOUT.md` | 13,341 | `174815864cd670449aba78714d5770264cc826c549ae0bba273598d2d691fbc8` |
| `results/eraseanything_efficiency_pilot/FINAL_REPORT.md` | 11,903 | `fb5c92440ceeb4bf02981d4017835ee91f56f65c403831856722a6c06842dd5b` |
| `results/eraseanything_efficiency_pilot/README.md` | 7,352 | `1fe2f1029f39ceb353bd228600ad36f75a09e07f52c9a374b2a8d6527a744363` |
| `results/eraseanything_efficiency_pilot/RUN_MANIFEST.md` | 8,477 | `30fd6028477da43ddd00d26123ff34d36e0246370ffed4b0c7913eec4ddbcf92` |

## Runnable code and configuration

| path | bytes | sha256 |
|---|---|---|
| `results/eraseanything_efficiency_pilot/code/README.md` | 3,606 | `efa7d66d298dbc525aa064dd6c00c9e83a0dde850ad4589e834848ad6f6101da` |
| `results/eraseanything_efficiency_pilot/code/apply_ea_patch.py` | 10,725 | `7b8fb53fe438e45badb7b4669803cc3ee4b2b310585bf1a27e3856a029b73398` |
| `results/eraseanything_efficiency_pilot/code/config_frida.yaml` | 1,300 | `0e37ac238c2b7a191d731ade4d69f1653ffa5657b72e7085dfb41d16e601b133` |
| `results/eraseanything_efficiency_pilot/code/download_flux_dev.py` | 885 | `d1d594210cc2b9b3cf611d94ff0a0079980ce81527bae9a5d33910c14644f09d` |
| `results/eraseanything_efficiency_pilot/code/ea_arms.py` | 11,359 | `60c7577c7f24ebc9bb17359d9b838c6e716f08c50e473d98ae5e44d0bdda3953` |
| `results/eraseanything_efficiency_pilot/code/env.sh` | 1,031 | `9e6e0775e81dc87a2f7e4c032f2957ac99e19f417599a5f51653a5e7b846a837` |
| `results/eraseanything_efficiency_pilot/code/freeze_cases.py` | 9,528 | `ae5bb055c84c057d87e57eeef2dfb506ca8c2e83e3e7bef7f2963022c85f1a02` |
| `results/eraseanything_efficiency_pilot/code/gen_eval.py` | 10,208 | `48ab4c757ff78b236b2e87d1f314cfab3eb1a921df51658afcbaecd50883083e` |
| `results/eraseanything_efficiency_pilot/code/ledger.py` | 7,927 | `f0bfb81ee697df2bf849dfaeb876137dc3df2a4223026bfb6e0420aff7695e8c` |
| `results/eraseanything_efficiency_pilot/code/make_train_images.py` | 2,593 | `e9b88ff2e7480a33eb9c1a2fec362ec1e6aca54b80a2e3d6366d348fa71f2bf7` |
| `results/eraseanything_efficiency_pilot/code/train_ea.py` | 17,270 | `c417367c0c4d02c0d97c714e62d714602ef718de703ec5219b14b0c65c63e63e` |

## Frozen research contract artifacts

| path | bytes | sha256 |
|---|---|---|
| `results/eraseanything_efficiency_pilot/frozen/cases.csv` | 58,073 | `2b8ceb9aa3b43f95218a0c77bf9591adc1b1a24b002381072662db3476abebcc` |
| `results/eraseanything_efficiency_pilot/frozen/cases.json` | 112,707 | `89673233d25951e45c5e5c2fce3df573824a4ed5af468d8d4774f636e84ad90a` |
| `results/eraseanything_efficiency_pilot/frozen/irrelevant_Frida_Kahlo.json` | 279 | `5857b341f220ebd9b41cd26a2e071eccafe784769be6e67c2aa44cb58eb5f24e` |
| `results/eraseanything_efficiency_pilot/frozen/measured_summary.json` | 2,221 | `7a0d059f17e25b7d575b5b81da20dd369d1d2adbdbed89d3c83a08a2348ff19d` |
| `results/eraseanything_efficiency_pilot/frozen/train_image_manifests/frida.json` | 503 | `64baefebd00e40d1b8c77395721dbd13905d65c1a5755995732ddada9e509802` |
| `results/eraseanything_efficiency_pilot/frozen/train_image_manifests/hokusai.json` | 516 | `0719d08f87bbb54f63d71f7e036372ee289d1688f013866873537fb5e8f827c5` |
| `results/eraseanything_efficiency_pilot/frozen/train_image_manifests/vangogh.json` | 514 | `5771f81e08198b67082f61f5acdb5212189cc18784053ee7678c6ccc4d0a8305` |

## Measured data: per-case scores, profiles, ledger, diagnostics

| path | bytes | sha256 |
|---|---|---|
| `results/eraseanything_efficiency_pilot/measured/attn_equivalence.json` | 277 | `cc297801c60973bcf1f5d8cf8d79ffe5aeec19c91ed3346dd0019186441cff95` |
| `results/eraseanything_efficiency_pilot/measured/ledger.jsonl` | 5,894 | `bb172633aee5a0a4c0847596169fb612b4ed5f1de86c24c317cdd23d8b4ba579` |
| `results/eraseanything_efficiency_pilot/measured/profile/A_native_frida.run_meta.json` | 1,963 | `a2ff74fd51b0b756130db8e081be7875ab8f7cd3cb55e5c0762ff8697c5301bb` |
| `results/eraseanything_efficiency_pilot/measured/profile/B1_sharedattn.run_meta.json` | 1,970 | `7b40200afe14a0d1d16fdeb3eb2cfee6f5207e696a78642d8303d0047f946a1d` |
| `results/eraseanything_efficiency_pilot/measured/profile/B2_shared_plus_earlystop.run_meta.json` | 2,176 | `8241dad5032d41c462c53dc5456c107687c544b5bcdce4c3eb611646365d682c` |
| `results/eraseanything_efficiency_pilot/measured/scorer_adequacy_diagnostic.json` | 2,700 | `e189e6ff826c193f521f863fa13ef902f4113d74403e881265206d3d8b988a2c` |
| `results/eraseanything_efficiency_pilot/measured/scores_M0_Frida_Kahlo.json` | 114,374 | `53292d90f742348d9b12988e247af3ead085acde5646dc629a3aad1ec25471c1` |
| `results/eraseanything_efficiency_pilot/measured/scores_M0_Katsushika_Hokusai.json` | 117,366 | `924a900d1ddad971c72ba714c4454e1a68f5646dbc8a40adaffae15dc98b990b` |
| `results/eraseanything_efficiency_pilot/measured/scores_M0_Vincent_van_Gogh.json` | 116,806 | `91a9989710f780df219b6e534a3be1b5504cd0661be0f798140df83bb162a966` |

## Evidence: fixed-order review grids and image hash manifest

| path | bytes | sha256 |
|---|---|---|
| `results/eraseanything_efficiency_pilot/evidence/IMAGE_MANIFEST.json` | 81,222 | `89dcbd2b105b37cb5948d7cf85e34fb6bc9fd43d573873ca58a1a619595eee1b` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Frida_Kahlo_canonical.jpg` | 389,537 | `a6e6f29add724f22e0b04c1c5d8449a45c09e05a4777fc88ed07fb578c95cab3` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Frida_Kahlo_perturbed.jpg` | 354,686 | `2b31a17ee27764e158b2bbd2c6bfbade07553f0e8a252758cafbc28b40f4f25b` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Frida_Kahlo_retention.jpg` | 323,029 | `278b87872561ad329d44c64a5692fc1b9465c832a2141cbd63770c16823f81eb` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Katsushika_Hokusai_canonical.jpg` | 353,129 | `038553614a6b631156e7a717008f33ef460af19bbb60cec9cb2d9718ded0132d` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Katsushika_Hokusai_perturbed.jpg` | 338,234 | `c298bb3ab56d1cf774e7e88dff5fdda21503651f3f493c15580e9e0680d234f6` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Katsushika_Hokusai_retention.jpg` | 322,197 | `cdaae7d9357cf4f5661f53645609a4e8623a8b3bcbe4fa3e05200959930eaa39` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Vincent_van_Gogh_canonical.jpg` | 387,172 | `7f833cf8ff7f623aafe12676ac3bf38fc02de975d707e86f6a3c6bf717e55c06` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Vincent_van_Gogh_perturbed.jpg` | 393,287 | `d7670bf9c380c6dff85335fe9bd5e983532cccb9f166f2462d6a7bb95164e22b` |
| `results/eraseanything_efficiency_pilot/evidence/contact_sheets/M0_Vincent_van_Gogh_retention.jpg` | 323,793 | `918c55d7f22ae091cdd9d8580d21216b4a851419eef8e2eba3793f54d2dcf92b` |

## Index generator

| path | bytes | sha256 |
|---|---|---|
| `scripts/seq/publish_eraseanything_efficiency_pilot_evidence_index.py` | 4,323 | `74e73c7b2d5128ef2b40f7ad188fa012e152396a35db1c556e9ab537485b7953` |

---

**44 files listed**, none absent.

`PUBLISHED_EVIDENCE_INDEX.md` itself is not listed: it is generated from the
rows above and cannot contain its own hash.
