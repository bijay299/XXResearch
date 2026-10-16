# Published evidence index — RECE golf-ball/English-springer sequential editing

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `f6fe138edbe5` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

**Deliberately excluded, unchanged:** the ~3.44 GB author-released
`GolfBall_ep0.pt` checkpoint and the pristine SD1.4 base snapshot
(identified by sha256 in `evidence/upstream/rece_golfball_author_screen_RUN_MANIFEST.md`
and in the edit logs, not vendored here), the pinned RECE GitHub source
tree (public at commit `403a67fd86dce6c124b88798340e001e625b28c9`,
https://github.com/CharlesGong12/RECE — the two functions actually used
are vendored verbatim in `evidence/code/vendored_functions.py` instead
of cloning the whole tree), the full 400-image raw generation output for
both experiments (only the review grids and the specific corrected/
ambiguous case crops are published), and this host's venv/caches.

Regenerate: `python scripts/seq/publish_rece_evidence_index.py` — run it
LAST, after every other file in this result set is final.

## Reports

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/FINAL_REPORT.md` | 13,390 | `6103a5635cdcd366ca21035195804399c02f9774aeba43c5f9ea7566eff8ca99` |
| `results/rece_seq_golfball_englishspringer/refinement_ablation/FINAL_REPORT.md` | 11,953 | `7800f70f5a372935b8afc5842df523dbf2075128d7d5f63ae74c703bdc94bdb4` |
| `results/rece_seq_golfball_englishspringer/refinement_ablation/CLOSEOUT.md` | 5,444 | `c1f01785ffe01c2931b561312ba1c82105eaafb02f596c6bdd1fce1a6500ce25` |
| `results/rece_seq_golfball_englishspringer/reference_A_intervention/FINAL_REPORT.md` | 18,793 | `a7381a7b39c0948d2d091f0ecb7dc8dba28042b9c9a8cbf4092cddb0bb63bf9a` |
| `results/rece_seq_golfball_englishspringer/embedding_cross_swap/FINAL_REPORT.md` | 18,931 | `8ac9d8c1094f8868afde3d66779a5e0c065db487302ba99500802313ae674d10` |
| `results/rece_seq_golfball_englishspringer/numerical_optimality_audit/FINAL_REPORT.md` | 18,057 | `9dfbd58f5c2fdb0cf0b386e513dd8c99a791126cf13e00cef9bede555b62c9ca` |
| `results/rece_seq_golfball_englishspringer/solver_only_intervention/FINAL_REPORT.md` | 15,030 | `02080a601b7553b535afb378ab93bbe14ef1c04385a50e5289570d4bf54bb9ec` |
| `results/rece_seq_golfball_englishspringer/recovery_optimality_audit/FINAL_REPORT.md` | 16,194 | `a2494b800a067e9a6b89c48f919131bb5c0df1033318f609374b6fe3f06c725a` |
| `results/rece_seq_golfball_englishspringer/preservation_content_control/FINAL_REPORT.md` | 14,917 | `370214d3721a10067ffa15ff368e031b778df25ca233914c55e5da1618dba01b` |
| `results/rece_seq_golfball_englishspringer/CORRECTIONS.md` | 71,780 | `b433d5345745a0feab8eef226e401bdfd75ab4685db9a3b58b375f32e2098fba` |

## Recognition tables and visual-review labels — direct-vs-sequential experiment

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/ANALYSIS_RESULTS.json` | 2,029 | `6f300b524cdf262e2812515c35867b92f8208c96188722fc5ac6e5f56c7ff7e6` |
| `results/rece_seq_golfball_englishspringer/evidence/VISUAL_REVIEW.json` | 6,865 | `670777a3a67c7e1a72d47d85116e583ae25b270fac64e7128fddbde05bb8f356` |
| `results/rece_seq_golfball_englishspringer/evidence/RUN_MANIFEST.md` | 10,203 | `57d0675fe4a5f196b9ffeb5c193c4ad65eb6e7b840596ee9f407f41031def468` |
| `results/rece_seq_golfball_englishspringer/evidence/ELIGIBILITY.md` | 3,777 | `8356153ed872f153859bb224344ee5cadf2088b28a1df88b8e483b7e043274b5` |

## Recognition tables and visual-review labels — stage-ablation diagnostic

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/refinement_ablation/ANALYSIS_RESULTS.json` | 1,851 | `fd0772ac0a8aad6426950965fa5275b0e88e374dd8105fe7829f79491d3dd7e5` |
| `results/rece_seq_golfball_englishspringer/evidence/refinement_ablation/VISUAL_REVIEW.json` | 6,729 | `48beaa64bcf74f6e00a57bf1a96b6ffe9f97b49c78fed0d7d28b8b7e1a0706cd` |

## Edit logs — per-edit checkpoint hashes and runtime verification

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/edit_logs/B_english_springer.edit_log.json` | 524 | `63a8beddd94f2caecd4df7aefab674388413680e8a588c4b6a0becfa18c27f59` |
| `results/rece_seq_golfball_englishspringer/evidence/edit_logs/AtoB_english_springer.edit_log.json` | 749 | `053dbeab551020faa5c32f65c0b0a0cda7eeb6327938b5b1958637d91b2e7458` |
| `results/rece_seq_golfball_englishspringer/evidence/refinement_ablation/edit_logs/B_UCE.uce_only_log.json` | 943 | `8436cb841b5da1b3a80a2fb9b8e664feaa66d549cc0b1c209a5642526ba11b04` |
| `results/rece_seq_golfball_englishspringer/evidence/refinement_ablation/edit_logs/AB_UCE.uce_only_log.json` | 1,169 | `a583b19ca9f920fa52567c44c19482c897d5a01e437782cfc77506c08598a160` |

## Upstream provenance — shared frozen manifest, checkpoint/source identity, prior keyed labels

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/upstream/frozen_manifest.json` | 26,427 | `ca1401c0064441969fdf3ab329fb9ab8eaab980b63ade62b6f14bd9bd1ef43ad` |
| `results/rece_seq_golfball_englishspringer/evidence/upstream/rece_golfball_author_screen_RUN_MANIFEST.md` | 12,412 | `1520b207912bad8702ae5ef2f705b2d12d3bbbb2acb93b5188197db56c92f1a4` |
| `results/rece_seq_golfball_englishspringer/evidence/upstream/rece_golfball_confirmation_VISUAL_REVIEW.json` | 10,404 | `98500bad180c01653e816114b43278d950517f036c89f5e349f3fc22c4713f9b` |

## Code — vendored native-source edit functions and the intermediate-reconstruction script

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/code/vendored_functions.py` | 18,728 | `5b05f8cfb133f6ee9ff96d1131af1b021d659ed92825fbdac157d7e476e77d89` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_edit.py` | 8,872 | `37c21b15077ebb39cf4ad6e3459526d6401fdece7cd4ed860868ad184d91fb07` |
| `results/rece_seq_golfball_englishspringer/evidence/code/make_uce_only_checkpoint.py` | 5,353 | `cf07f4db9e50a3f564dc6f1c02a88c1d8dd1978c0e87c25469f137012f1e9bc7` |

## Reference-A intervention diagnostic — does using parent A instead of M0 as the close_form_emb_regzero reference rescue the sequential corruption?

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/FROZEN_MANIFEST.json` | 6,656 | `1807dcf881cc669f00af76dad6793845e584677bbf7685a83e7b58355fe37543` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/VERIFICATION_LOG.json` | 8,119 | `f6a84c621bc8c6312267f4712281dfc20c98f97f6997ba9b51a4c9d03f30709c` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/UPDATE_NORMS.json` | 5,953 | `aed476381f89513fca9c220785f1ba61ff6eb0442de099fef2c7f1b6d81f9fb1` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/ANALYSIS_RESULTS.json` | 5,298 | `5f592fd06eb003426b25a44377860cd34f10c50a6f3e9147ea1b58fe79805bf4` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/VISUAL_REVIEW.json` | 8,581 | `c3bcb41a208583db1b96cadcf1903bdc1f3e0b93f3dbd99ec49d9572764b44d3` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/edit_logs/AB_REF_A.ref_a_intervention_log.json` | 1,003 | `2386203296e38d5ec0a77836a7b8d3a1363c027899ea3a952c5bdbe4748b5a97` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/AB_REF_A.classifications.jsonl` | 50,306 | `6854c4c4d1e9fd6c0898562cd0ee34767bc5449a90004fe9b414719fcc981b82` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/AB_REF_A.classify_report.json` | 232 | `8b66555741352b3d93b257d7f5ebb860ccc0b2109ee1a3aabf994c42073ccd94` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_reference_A_intervention.py` | 22,132 | `e17d983637272ce4b96531177d2f47c035e86e1e9c889d884d1968df3c96dfa0` |

## Embedding cross-swap diagnostic — does corruption follow the recovered embedding, the working checkpoint, or their interaction? (reference held fixed at M0 throughout)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/FROZEN_MANIFEST.json` | 11,644 | `a845ceca4a89c3323e44d4c63d99f42bd382d09e5c19977ca16dc65817138adf` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/VERIFICATION_LOG.json` | 5,069 | `50221b135fd72367d7c99a53d59ce74c37c1def8cf1e1c5450f64c127c16cb1f` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/DIAGNOSTICS.json` | 1,988 | `a0ba1a3f5526dfd02925f5bd8bed49445686a15585131cdbb66988a51d8d68bd` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/PER_LAYER_DISPLACEMENT.json` | 12,085 | `0bf10c6f0a08be687c8d90ae975a8bba4505a6202f0b2656e5c0f039f4a5d15a` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/ANALYSIS_RESULTS.json` | 6,112 | `c75620fe16547eeabe1044cc4bbecec6203fc0862d057a8140e52277f74cee3a` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/VISUAL_REVIEW.json` | 8,921 | `75d21f2edf7fd5d22909fa53bf1e2e12dbafb9d7e8894da9e4a35e847e96584e` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/EMBEDDING_PROVENANCE.json` | 2,063 | `853165865a664752a9426846e401727a43b85cf8800a65d7249f7b86c07f4f88` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/edits/D_WITH_SEQ_EMB.cross_swap_log.json` | 482 | `e5d194297c5cd376e9d4570a79a5174bc4c4433ee306f63c1ed5d22d40227b0e` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/edits/S_WITH_DIRECT_EMB.cross_swap_log.json` | 485 | `68fcdf1138f194ffbebbdac175a1f3a21e7a98208efe7a44761b54bb3710b080` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/D_WITH_SEQ_EMB.classifications.jsonl` | 42,076 | `99180a4d4ff19298f9ba163349284e0987a69509951bfb384d67e3b6591e45ff` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/D_WITH_SEQ_EMB.classify_report.json` | 238 | `a2ae688225d777502bde991e3a1875ba49b050b6c1251028859cfc03f97f2011` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/S_WITH_DIRECT_EMB.classifications.jsonl` | 42,337 | `445cdbd68a861e914cf4b9aff506c80399f199afd6b3f6e969e5c9a949a0748c` |
| `results/rece_seq_golfball_englishspringer/evidence/embedding_cross_swap/S_WITH_DIRECT_EMB.classify_report.json` | 241 | `fb24a6b9bcd747e8a21d3423c2c6bd8b927f29c4145fe3f580870415f0d7e6b4` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_embedding_cross_swap.py` | 18,203 | `50538d32059e77b96ce8e035517a3be0cdc245d81d74b7a6831ac93e3a0a8378` |
| `results/rece_seq_golfball_englishspringer/evidence/code/compute_diagnostics.py` | 6,587 | `11114579991edd2ba646a3b894aa84439d01b6d74c47d573cdfd6f5cc39c4799` |
| `results/rece_seq_golfball_englishspringer/evidence/reference_A_intervention/CORRECTED_LABELS.json` | 7,988 | `ba41c8be129f943ea62094f914ac170664b29dcdd950eb5513c9721e325884c4` |

## Numerical-optimality audit — does the native closed-form UCE solve actually solve its own stated regularized objective, for all four embedding_cross_swap cells and all 32 edited K/V layers? (CPU-only, fp64 reference solve)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/numerical_optimality_audit/FROZEN_MANIFEST.json` | 14,140 | `deda4dd93af61b5ef192f40a6dd818c9907cf8b2c5bd23e3e28d554622af9468` |
| `results/rece_seq_golfball_englishspringer/evidence/numerical_optimality_audit/RESULT.json` | 284,896 | `cef3c5b33f5a1d864ccffceb15495dcd7e2be8c425ea51ea43eccca1e6340bae` |
| `results/rece_seq_golfball_englishspringer/evidence/numerical_optimality_audit/CASE_LEVEL_COMPLETION_RECORD.json` | 10,557 | `c885e63e4a33b310af88ccf79ce203bdd718ba9507314b68ea6106b04997b1f7` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_numerical_optimality_audit.py` | 15,822 | `c14336ae30d2e57b23cb4019ea097c0549d24abe6f670a5d308bf18310d2d086` |

## Solver-only intervention — AB_STABLE_SOLVE: does accurately solving the existing refinement equations for the native sequential (AtoB) checkpoint remove the original corruption while preserving deletion and retention?

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/FROZEN_MANIFEST.json` | 12,005 | `5e0dfdb81f0e08f4e5b191f8947ec97162fea0d8c73d5b5f435499bfcc4e1c42` |
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/OPERAND_PROVENANCE_CHECK.json` | 2,025 | `ef0ae0155a5d7d98f2d0e71e1c0ccb86c566bf5423775852fbff939b92fa3ac9` |
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/RESULT.json` | 65,957 | `07a52de1ac03db9cd0e599f43bc99d33fed19bc5f7c4f18b8bb4d304cb8efbb9` |
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/VISUAL_REVIEW.json` | 9,046 | `e61f2bccf7ce526e574a7a5de05e43408b2cd4f5dc46cbe80ecdb69728ceeab5` |
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/CASE_LEVEL_COMPLETION_RECORD_120.json` | 68,790 | `fbf421940ac904fa434a2efe0c0737910fec65810158764d51319e3f29c68122` |
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/classifications.jsonl` | 50,964 | `39a4fea1e46eae3fa7afb9fad2088d602f7c8f0135dcf28d68a018102ec48f8c` |
| `results/rece_seq_golfball_englishspringer/evidence/solver_only_intervention/classify_report.json` | 239 | `fee80f9e6519bf3ca400f8bae04f258c7b9fe35eba1c33c5716872ed0020d785` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_operand_provenance_check.py` | 7,485 | `bddfd5e0536079eb4697c924441ad2d28db5bc93ff3371623c5cd579a68195f4` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_solver_only_build.py` | 11,566 | `2d77817dae275c43348d59ea408c1b46a652c867253911a73917bdf5b63f4c13` |
| `results/rece_seq_golfball_englishspringer/evidence/code/make_review_grids_solver_only.py` | 1,682 | `09a6aaa19eb81d1d37ac101b7cd93a60dfce75697e78554614e9ae678feac70f` |

## Recovery-optimality audit — does the native embedding-recovery output (E_D/E_S, close_form_emb_regzero) accurately solve its own stated closed-form objective? (CPU-only, fp64 reference solve; different computation from the numerical-optimality audit above) plus the corrected AB_STABLE_SOLVE golf-ball case table (C17)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/recovery_optimality_audit/FROZEN_MANIFEST.json` | 8,044 | `285824dbc28c94ce5d1790fc761f535a9b243b092651966f223a402feda81cf7` |
| `results/rece_seq_golfball_englishspringer/evidence/recovery_optimality_audit/RESULT.json` | 61,704 | `1b385c98c8cf579fd6e531e7566c974e3221915295a98430afa5e9a5ce591415` |
| `results/rece_seq_golfball_englishspringer/evidence/recovery_optimality_audit/CASE_LEVEL_AB_STABLE_SOLVE_GOLF_CORRECTED.json` | 2,937 | `43cc1ea6a98c91f6cce288f9d1ff11f7f5a37577603c3d0bd497b028837ed55b` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_recovery_optimality_audit.py` | 18,826 | `f727179db33991f1f20cb3e13a37833335bcb270e2ba508eb90c5b137aeedb11` |

## Preservation-content control — does RECE's existing preservation term, pointed at the already-erased golf-ball response, prevent the golf-ball corruption characterized throughout this results line, and does it beat an unrelated-content control ('French horn')? Two new endpoints (AB_RETAIN_GOLF, AB_RETAIN_HORN), fp64 Cholesky build + validation, frozen pre-classification visual review of all 200 generated images, joint comparison against the existing AB_STABLE_SOLVE comparator

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/FROZEN_MANIFEST.json` | 8,631 | `772bf5dcaa7ad4816bc3f3c0347a45d44ae1bff506b552524b3b954c7de22930` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/RETAIN_ENCODING_PROVENANCE.json` | 6,510 | `f32bc58635fd60afdeb48b9b4deb6a6450c972677947700584f296213f344324` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/RESULT_GOLF.json` | 66,186 | `1ebeb40590403fbcdc8894c8397cd309537ab8b6cb13abb9220882f917505bc1` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/RESULT_HORN.json` | 66,204 | `6bc95b228626399a3dadd6a21141cdf10350b03bf839e8e4c57f35b97a056841` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/VISUAL_REVIEW_INITIAL.json` | 1,996 | `614b0190e0def411ef8a12a51466ccd830484a805a76b98ef4c91ebe2f59bc26` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/CASE_LEVEL_COMPLETION_RECORD.json` | 96,138 | `5d63ca2e74616a3f9c14187ad80723834d7fbcd50d14113531ffd2fd3d890f2a` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/CLASSIFIER_PRECONDITION_GATE.json` | 645 | `a7c29de7ae55fc5247690213ee5162bda47df9ae5bfa3f883d4a9593e7e9c02a` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/JOINT_ANALYSIS.json` | 5,108 | `3cc62e734b37484cfc7aa93cb17d0882aaed6e0ce79b45064bb59cf769b0f2ec` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/AB_RETAIN_GOLF.image_report.json` | 37,884 | `4d212e8662b9021a331fc9872787eae1d8defb123a91a56c22be53c6999d392b` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/AB_RETAIN_GOLF.classifications.jsonl` | 42,037 | `403e89274509c1d8cb896151d57fa13377e85d76b486e6928870190f8829b8fb` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/AB_RETAIN_GOLF.classify_report.json` | 237 | `e359c1486d31e9410116ea2a99819ca53ae008b3253181f74bcbbc5e7efc3767` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/AB_RETAIN_HORN.image_report.json` | 37,885 | `d16460955d5c739dc9ebd9db5b9b0d5477d9e92c17600f5a2b5dda0b380ed95f` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/AB_RETAIN_HORN.classifications.jsonl` | 42,123 | `030882ea80027c4bc204b7fc30e4a95aa2528a3f7353e4086bf6f74b6d962843` |
| `results/rece_seq_golfball_englishspringer/evidence/preservation_content_control/AB_RETAIN_HORN.classify_report.json` | 237 | `5570fa9efaea3948c9171661d3240d2fc5fad685a2995b357c750ae852625557` |
| `results/rece_seq_golfball_englishspringer/evidence/code/encode_retain_texts.py` | 5,404 | `94cd3742140fb87bc089e956b6916559d3a8192d70bf0627e7c4846b04fb16c8` |
| `results/rece_seq_golfball_englishspringer/evidence/code/run_build_arm.py` | 11,896 | `fc38b7a7f77ba82a17c16349367062f684410208cfd2fbcb347e0f697ed8788a` |
| `results/rece_seq_golfball_englishspringer/evidence/code/build_visual_review.py` | 11,162 | `a6db0cd19c4da095e4b34cd1a959758ed1e40132f73b8a5b47e9e6e4bcacbe08` |

## Visual evidence — preservation-content control review grids and disputed/representative full-resolution cases (case 30's classifier false positive, case 39's lone persisting degeneration under historical preservation, case 16's lone recovery under the unrelated control, one representative unrelated-control degenerate case)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_GOLF/golf_ball.png` | 2,020,315 | `4a6ce5e318470cd6bcd285d4a818b7a2ae6ffd561dfa6ca5a8d7e51239c20e60` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_GOLF/English_springer.png` | 986,439 | `95f6baf66272a21109e776109c03668b719b56b52def79245c0095b832e207c9` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_GOLF/parachute.png` | 914,247 | `183f78213814356e372f0a2baf3ccdce70c55d4e465933461728134047d9db96` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_GOLF/French_horn.png` | 976,200 | `8c509f15e6eeaef0cc318d24318616f621fe1c9f20aca4a77fe826fa48b50b6f` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_HORN/golf_ball.png` | 2,324,416 | `abb930627ceba80535aca6834d610e8ceb1db30d125d380f1401412ccbf11c4d` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_HORN/English_springer.png` | 999,745 | `8fe18594237fc946ff027b64a5e568435201b4d9e72ded06fc245b79c914bd01` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_HORN/parachute.png` | 914,847 | `ebb08514215a64e81f8c373ca0587bff2f819e1d10a848c9116924fb98634ce4` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_RETAIN_HORN/French_horn.png` | 999,202 | `21a21f419321293a92bd90c1896c38cf4ad10d67a8e758157aa5e4f783b956ff` |
| `results/rece_seq_golfball_englishspringer/images/disputed_cases/AB_RETAIN_GOLF_case30_classifier_false_positive.png` | 466,761 | `0e057cd767ba72eec834c6f8490d833a24b623b09aa6da3c9027e933f9cc32eb` |
| `results/rece_seq_golfball_englishspringer/images/disputed_cases/AB_RETAIN_GOLF_case39_only_degenerate.png` | 426,357 | `572b62dee430752f74f8933c10f01246ead4d5d1853d27b2a3afced99699f767` |
| `results/rece_seq_golfball_englishspringer/images/disputed_cases/AB_RETAIN_HORN_case16_only_nondegenerate.png` | 469,416 | `16e9c43e2ef56a46e4f9676986f7edf196ac38d364ed213f3b66892a5c062664` |
| `results/rece_seq_golfball_englishspringer/images/disputed_cases/AB_RETAIN_HORN_case0_representative_degenerate.png` | 440,880 | `fa0e2aaee789639b52e19fb214f1c976a86b84c5912ccb3e2fdc5bea119d936c` |

## Visual evidence — fixed-order review grids (golf ball + English springer only; French horn/parachute stayed clean at every endpoint and are not pictured)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/grids/B/golf_ball.png` | 2,282,082 | `f99fdb7ffa81b108745a4b9025ee513edb06df3d654f3eb740f43382efface27` |
| `results/rece_seq_golfball_englishspringer/images/grids/B/English_springer.png` | 1,199,131 | `0b0c943bbc600024b70ed377aab6af5a654687243e858d2cfa356a293bab5523` |
| `results/rece_seq_golfball_englishspringer/images/grids/AtoB/golf_ball.png` | 2,954,850 | `cc42797d363b9b45add472a8a829e55e45bb03fea36488c986c26659c1fae3c7` |
| `results/rece_seq_golfball_englishspringer/images/grids/AtoB/English_springer.png` | 1,254,791 | `e6a3712a5aefdbd45de255e5c1fd014fe37446c6078e1df7daf161e18d240d8f` |
| `results/rece_seq_golfball_englishspringer/images/grids/B_UCE/golf_ball.png` | 2,269,594 | `487b18e1998dc599a8a8642fa606c3fea99b88ac9ce464e1644da649fb973a0e` |
| `results/rece_seq_golfball_englishspringer/images/grids/B_UCE/English_springer.png` | 1,246,928 | `020d1e252ebf9175e22af80fca16b84d74674eaf5e8a6f784bbf5ddc5cc62e90` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_UCE/golf_ball.png` | 2,527,837 | `695b464a5b31432d48b8b1019d18771421d1e08642e4b21b8c4af507412fa017` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_UCE/English_springer.png` | 1,249,871 | `72107b6eea7b1af613a11108be97dba092f6ef09e1457d19aff80c6e62647092` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_REF_A/golf_ball.png` | 2,968,295 | `22402d192f280dfe0f81d51d2dfee0eef4d9a9dc71d2a397afb77e14e9abed2e` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_REF_A/English_springer.png` | 1,262,326 | `329a94ef376833356cbdb2ad4937d89e4aae4bbe644ca52081e7b940684a62af` |
| `results/rece_seq_golfball_englishspringer/images/grids/D_WITH_SEQ_EMB/golf_ball.png` | 2,610,563 | `efb46fde71d60fdd6343cbd849345770153ed0f646d2ed295b3330de6fcd655e` |
| `results/rece_seq_golfball_englishspringer/images/grids/D_WITH_SEQ_EMB/English_springer.png` | 1,614,283 | `7c0b473da7bf7ab062ab290e178b565987aaaf4fd3f52bde6f15cbd5e09c07b5` |
| `results/rece_seq_golfball_englishspringer/images/grids/S_WITH_DIRECT_EMB/golf_ball.png` | 2,678,848 | `1a93887a6eee5f9b35b93e15431a30cb4ed5d342aaebc4e72c4d26c456cd48d7` |
| `results/rece_seq_golfball_englishspringer/images/grids/S_WITH_DIRECT_EMB/English_springer.png` | 1,199,106 | `b8eff7c8e7b900cb60b362afedefae88077fe6ca0d5bbd7d32b2f302ab619f49` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_STABLE_SOLVE/golf_ball.png` | 2,923,342 | `463f6c8301985ba11320233f6a1423e911dddb1854720df3d6b20f1d98c4aeb8` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_STABLE_SOLVE/parachute.png` | 1,139,401 | `400764cff55b3b7924d3407cc5f90e39f128f00585c4bab0faf6e473d2997f03` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_STABLE_SOLVE/French_horn.png` | 1,245,541 | `19b19f46bec32649b2f13060dada41d8d45780a6f38193ca2f2a8ec601d6c74f` |
| `results/rece_seq_golfball_englishspringer/images/grids/AB_STABLE_SOLVE/English_springer.png` | 1,248,878 | `31e6029f24d6ceb26f19c31548590e4f6ea145de85dc8e7acb1e6736cb6943ba` |

## Visual evidence — AB_STABLE_SOLVE, all 60 new target images at full 512x512 resolution (40 golf ball + 20 English springer; parachute/French horn stayed clean, pictured only in the grids above)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case0.png` | 424,995 | `006eb14961a168f94b2f89a5251fb51e42314ca88dba527a276a501cd80df79a` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case1.png` | 462,408 | `490b0cea66baaa26d0707d2861108204585161930da9532b7fa19d8d144c3e54` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case2.png` | 373,654 | `b96a44c28d52f1870580b12ed2687e172d55f3ed22b8a32676dfe0a237c8228e` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case3.png` | 492,744 | `c9192aad666639f644824599e70805d265288b671f8a9ee0055b99975246e2d0` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case4.png` | 417,565 | `fa92fd80e50a05d072ffb119c528a3569efa1e9084971f8325b872461166be17` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case5.png` | 484,902 | `fb603d9b83fac3f6b038dcfbfaaa68cc71c6b1442ec227e68ce1ff74d52ac356` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case6.png` | 436,646 | `fa0f4b6a2cb5543a5bedeccb1e31b20bc5c3d0d2a4a451b57dbd59bb8b3e9fb4` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case7.png` | 521,855 | `2743b4b1178fb5af508e65c325b8a7fafa8476837c5c23c08b90cdc085c5653a` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case8.png` | 479,196 | `a5939a7a0ad79be2592b76ef0e0cc7adbaddc7fc7433ebfe3fbf3609ae31f7f3` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case9.png` | 477,110 | `c6b796a1e0987281d04181330bfea7b49f147447b39bda5ce518a9c9ab1f5b0b` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case10.png` | 459,484 | `02d8fdc90b169ba86e56015f4918291d0c29264dc45e83abbca06085bf9aaf98` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case11.png` | 470,600 | `0390195d7e5d5a9b16072f70bfb3cd7beb09f2dcdb2a56730c7f57adcdf553ec` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case12.png` | 460,318 | `ac866971e7767725c79b7a0146ec2f5008e55efe77a2fa505f4c86e8eca39e05` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case13.png` | 475,711 | `602fc73c2a62a1a725f5feb30f47c543f1b0c17ba55a87c6b60061593a14560a` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case14.png` | 461,308 | `ffc9c67691af919740c3ed695fe46a65c7375d47a3663c91cb1292220025bbd5` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case15.png` | 459,523 | `21d1ffb066342c3d94f74dfaf1cfb4e9646244b2ebbbb2e18f76b5819e0c99fa` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case16.png` | 465,245 | `850843b0aaf59b61c2f857f3cae7d710d170a257d3fd7a7bb1b70c824cce339d` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case17.png` | 557,214 | `b9e6eaf75323f1b7f2665dfec59f7da7b8feb0d302c6152e6556e37dbfa81c52` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case18.png` | 336,525 | `ff78731dc5b493974984c14ac7f4f52d954c16ff3887becbf48f49777da4c3a7` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case19.png` | 526,905 | `ca1315753e977290e33bef3c995c34ae7ed5e11638c25e6ca2d61f16336560eb` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case20.png` | 444,760 | `8e32e95e638d0a58f6788feb1e5a89a1f3b8e97df053f4491feb82f4d1608d4d` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case21.png` | 508,015 | `14c4cfe6281a906a2d0c3777d37b0ebace2e8c0e6db8c5b355736e3693eee594` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case22.png` | 443,218 | `63f5c8863fa3ed2cba34c12b37426e2e78d299f384c7b034797ac6e1f649db8e` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case23.png` | 493,433 | `b36eafb7eaf8cd7ef3f389194e8cd63900d1872ac0df4588de0728e6d3971373` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case24.png` | 488,366 | `9f0d9c5a8f9f84e467f143c0360443444cd8db184033c0737f2141385f6c00dc` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case25.png` | 411,607 | `be7b58fbb47567a751a03c2a11b1a8ca55d8c412f6df11207720c2d2902bd8ed` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case26.png` | 368,247 | `86f36226410f7185bdafb30976ac2ceddea3d9a33cbc744af1c209603a263409` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case27.png` | 448,117 | `e84731b0af14a621df2b8a444af2aeb8a0de12672518993129aa67d37b2bb818` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case28.png` | 451,694 | `4bf3163d3f9e51ec098c0c994bd355919b1ef6af1d116a7141c29c674b1d8af5` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case29.png` | 380,748 | `7744b304f4143b784aa228c0720c65b35913179bc36cf3e15b6440735cb43b74` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case30.png` | 384,158 | `757022a3f68d57d608f11ab0977bf79cd12c1f0559a70009ffaa8ec83afd7ede` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case31.png` | 402,614 | `87f34855ffd181dd766ee4bf680ae30c2764f6b91678e0767ba90e6c287a5be1` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case32.png` | 452,363 | `ab849253a38acc1499f7b778a7f6a63d2ca0acd713839302a0ff1cd76dc56423` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case33.png` | 468,279 | `c9d05880f4ccb14cd419c593db07e626626dafdc3e9a778dc069a43e32436d88` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case34.png` | 411,639 | `91784537fd4a82f2318a6f59ce1276360ddcbb671ce98f80cb4ce3cd8cb764f1` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case35.png` | 490,230 | `55135b62ee9e7f6aee9f863cb6ef0e334dc88cb56f31456c0cb55c41fb8cc55f` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case36.png` | 519,028 | `46846a027c1ff8ce8ccd63f6a5c3d9b01944ec8354ef44a3df04c2c0d44a8d22` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case37.png` | 476,015 | `16cf709ccd7e514f33c7a6594a00bc640476fe7ac17f0780db98e4cb5463a15e` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case38.png` | 353,854 | `fde40d28e3c6976c521572094d2ea45b26223a3453ce28babacef7cbafb51d81` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case39.png` | 487,995 | `297cbadbff8e2354d85a7ea62fdcf79578c16ff72e39fea39be431cb48284f48` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case80.png` | 550,037 | `8d2baec6269b6ea402a996fed3f8b29a39f89a7e38784efc153f8dd0cab9dc9c` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case81.png` | 497,580 | `f40ce135d6de31471d80d8a0e039047c57d1e6984bba9e9cb171d364dee082c7` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case82.png` | 557,625 | `6efbb6b82af74718d910cadd5b3a8f6bb0ec3206048b9ce3e4ad62d0994da63f` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case83.png` | 544,917 | `0476cf75acdb30fd26f1329f8178eae80a390ec5f3c9e21f886647a55fb67403` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case84.png` | 587,971 | `a9afa85e1514e552d36b0e86f8e46800dcc84d3fc649574db825e0136d1dea6a` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case85.png` | 492,516 | `b28a98f1ca9fb0a757d8f3eeb3425c5fbaaec126b06b0e220046dd52ea72c811` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case86.png` | 567,135 | `d41b2edda3f47e7638b9867768bdf6864c7b1989d49c8b85771ef3e99349b53d` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case87.png` | 472,314 | `9c0dc59861b4d33a129fcfc52ec24440674e90fd668ce918ab44458601e8f0e8` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case88.png` | 463,012 | `583e79df4a32a68226d768c7a165780ebdd9c2507e1d978a5ab70f9ded85e988` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case89.png` | 436,177 | `ced55dbfb1a47d019c46f0adc3028088d4a481de191fa142fd081425f9729b9e` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case90.png` | 516,313 | `814adfd6465d4a80acf4ef831abd5be46a4fa24480388d01fa04b776b806776e` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case91.png` | 479,314 | `16da0e329c79776eed23e33c96c4bcb324093f00ab2b6fd122b8ae9b2d43c38b` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case92.png` | 422,636 | `9dd61099e0fea5647ec2fd26a44a0036dc684fcc33a2f847d10e6136453ab7b5` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case93.png` | 492,531 | `8451d59dcf7c123d56a005bbd5470ef8d1a3cc9d4f03496156448d247a7c7e33` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case94.png` | 432,119 | `a36ef1bcb318038e04ae8f347b3473e94a1bc7ea692d1ecddc4730f107698f84` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case95.png` | 460,079 | `42d6990995f69200d5fa924519967127ce610d24751dd69b404dbce5e4dac922` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case96.png` | 529,126 | `da9c646f36546d3e1f17f1d7729a3f7fcd03717cb3da7f23b2e3f90fb65ffeb5` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case97.png` | 440,961 | `f3ead903dbce4a6ae234ceb3afd6d44a8e4c8b833d6723fd081e2862350ae076` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case98.png` | 509,668 | `b2423b82676a142128cc65e83191b0732f21e3e0569621bb0c9ce4c42a3551a8` |
| `results/rece_seq_golfball_englishspringer/images/full_resolution/AB_STABLE_SOLVE/english_springer/case99.png` | 436,329 | `a0987cfb759e915374e2d9deeb43aa443b4368477da10d1b68c3ce62828756cc` |

## Visual evidence — full-resolution crops for the corrected and ambiguous cases (CORRECTIONS.md C1/C2)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case22.png` | 458,335 | `626b552cc5ca82a92987646b1c2a1bf3e6988e997f103f617150a39325ebbb5a` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case22.png` | 421,817 | `8cc4ac3529e62bc13539c868922a196b4c3bc9864095f45397cf6df3fd7c8e00` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case34.png` | 426,082 | `11dd9f92565c160f9fa8265c73d59e1a8f1a8ef60cb4c43418e57b62e1c0a179` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case34.png` | 339,602 | `410b45fc81ea83a71129e46745c87cde0d814eab3085a30c9882dd9cc306811e` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case39.png` | 464,646 | `98e7cd7ff057bd6d4cc9401c8333dece41e28f872939bde2e9722cefd05a8bcd` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case39.png` | 405,733 | `a565bb1e2f14e0befcd6b5cae6b342556e515af75d8cf5aaf058d100fd943fd3` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case84.png` | 600,086 | `312dd0a2d9b3323e9d43d67eb37b358bf3bb66c19c050205c8068d238dae6387` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case84.png` | 567,671 | `0033453ecb5dd536ec1f00b292f5289a4b4ef61e0dfb1b3dc9f93d0cb35b53dd` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case88.png` | 458,192 | `ef8490d1b48be066cb345e172248429a70f2f194c6aa9589e97d0af68cb9c1c5` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case88.png` | 571,242 | `9f13bfd8034069247ccf46d81a8677a1039c93a05250d6c1f0a9d8179636cfd1` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case92.png` | 407,616 | `ea880f1f893a2b3e3c76161eaa78b3248ae7d490e29d604f56ce9a54d7873636` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case92.png` | 423,275 | `8d5aa980f784da1ef7e73c8cd85200713a2c840f6505a89957d2f40c3971d026` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AtoB_case94.png` | 417,135 | `254e626a6de2ace1d7dc9793b7143e10bd0f2bf87abb4b825575328867013b1e` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_UCE_case94.png` | 422,382 | `d44c8ae5d3b442cab6f90524c439b4ed018950cf4e444f62df3f7da629722fff` |

## Visual evidence — AB_REF_A full-resolution case crops (reference-A intervention: recovered, newly-damaged, persisted-degenerate, and the English-springer exact-hit case)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case1.png` | 416,059 | `667730106d497c10ef363438ae01053cd005338d5748977465f9831e26e037d6` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case15.png` | 461,653 | `acecdb6f1611894ec1ef63968429b045ed4e25535b522de0741d21820cb14e9f` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case28.png` | 407,497 | `8693747e77c53ec5a2cc05536aa55dc5d7f72b3e11db2a07c1ef8f02f1a58547` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case29.png` | 400,790 | `ee6c37571c8e277c98d32f5511f56d61603ffce121c276b9d26599175aea7f24` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case30.png` | 346,950 | `16b2bc9d86f4289b6e15cd541dfe784b47b573e45822ed90f3869d60387ca39a` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case31.png` | 347,094 | `bc12a766eadd87404c68cd1bc6a9f7dbdbc429528153accf9e44756d81ed7a61` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case33.png` | 429,635 | `193d48fe129b53985f31a67f19aa79aa9daf26870de96243298fa24819d9ee0d` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/AB_REF_A_case84.png` | 569,231 | `da43729c6bc6cf474ca566cf7bd2872cc3fd424e2a48ee191f75083a90fef60e` |

## Visual evidence — embedding cross-swap full-resolution case crops (D_WITH_SEQ_EMB's 'flying ball' corruption spillover; S_WITH_DIRECT_EMB's recovered golf-ball cases and its one dog-present English-springer case)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/D_WITH_SEQ_EMB_case33.png` | 361,226 | `29aeae8ad4e5b16cfcd8923af315e1dc2dce02750f0019a38c63bf519e924c1d` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/D_WITH_SEQ_EMB_case34.png` | 385,293 | `f9acb8d5d43369410619aa8bc0f9d1cb2c9a43830d09a1a1bc5d487bb71d72df` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/D_WITH_SEQ_EMB_case35.png` | 390,676 | `49c52e10c77ac629123f10aa71d9109718c78efde74735fbc15d14c6be99f172` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/D_WITH_SEQ_EMB_case37.png` | 480,319 | `26426b61a341dfb69bb776321016628802ad4acb72daecc68c4a251b556fabe1` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/D_WITH_SEQ_EMB_case38.png` | 485,330 | `e107036e823a1be783b5c088b128a9049b6e084468aec0f20cb0daba4f4dcca2` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/D_WITH_SEQ_EMB_case39.png` | 427,042 | `a8f46b595f3e90e5824e64de4c85c609b1d0cc0f134308b5ca4a4af5c9d4bdf4` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/S_WITH_DIRECT_EMB_case1.png` | 440,138 | `83ebfd1d98308e9a2b50d8160c3649b3145280795edc66b804bcf27b46ce20ab` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/S_WITH_DIRECT_EMB_case17.png` | 479,972 | `2cf3f60a1085969f77240d966a30d4f6a0634f1ff5dd18fd638b093158c5b4b6` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/S_WITH_DIRECT_EMB_case35.png` | 464,661 | `3ce18f17c005b5d01236e8f72df2cd11eafa0b046174bd9379979bf9cf87549f` |
| `results/rece_seq_golfball_englishspringer/images/corrected_cases/S_WITH_DIRECT_EMB_case90.png` | 455,592 | `1c40d214291788d64ef7562fecec0414567e7f1312e80c4e9e8ada7ab3d3a8b3` |

## Reports — history_preservation_confirmation and stage1_history_intervention

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/FINAL_REPORT.md` | 13,378 | `f8c063a0b058b41c4854a992a55ea4274a24e933176d0a1d16b56be05eb55115` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/closeout/CLOSEOUT_CORRECTIONS.md` | 12,066 | `9273d99849e8d927bfdd4b50f969008bb2ff63656f069e9c180c9552f6f43e94` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/closeout/VERIFICATION_AND_CORRECTIONS_2.md` | 9,787 | `01f453ca91e1c8fa41d63bccf6a781d5811834c048ad7bc29d84f273f9a8ae21` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/FINAL_REPORT.md` | 10,093 | `c3580514f96a4ccbb415bd1346f1da0c79728119cdc3832f14978e961bea29ed` |

## Evidence — history_preservation_confirmation (manifests, provenance, verification logs, joint tables)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/FROZEN_MANIFEST.json` | 7,150 | `30ccead3074ad5e58cbc50b6ce637ecc4cea6cb6b5149d81d8b4d81f88a6bb1f` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/FRESH_PANEL_MANIFEST.json` | 21,099 | `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/FRESH_PANEL_MANIFEST.sha256` | 65 | `350aad0e4e9c71eb38a7b9d33a7834e8a34b49d9200e510e4bca372c02df04f4` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/ELIGIBILITY.json` | 1,936 | `3d967ecffa3350494112849d71b271d458fe504a0a12da6959cc6968accfd2d8` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/expected_32_keys.json` | 2,198 | `adc9fa07f51b9d21add33670ddca83213276744bdaa46c5bb97d7bb65f6892ff` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/VISUAL_REVIEW_INITIAL_SHARED_INPUTS.json` | 190 | `0782dab7337209e1b15e3c5bac986299301d56539ab285966ec23f7989c2b4cc` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/CASE_LEVEL_COMPLETION_RECORD_SHARED_INPUTS.json` | 158,128 | `4156e907b032acd1ea6da3d91d24ae720091cff37dd89aad22e37a25af6f63d9` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/CLASSIFIER_PRECONDITION_GATE_SHARED_INPUTS.json` | 628 | `e490022ced333a674ce4d17e6109676109bfca142bd1dfa00a210f263061f355` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/VISUAL_REVIEW_INITIAL_NEW_ENDPOINTS.json` | 268 | `53b31cc8e5809d1d3c40dd3ee394db23a641e4cc009fd408fc935ea1a5209f28` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/CASE_LEVEL_COMPLETION_RECORD_NEW_ENDPOINTS.json` | 324,669 | `573ab25307428ca1844997cacff80326423e07457259ee9257ee5c1384dbee18` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/CLASSIFIER_PRECONDITION_GATE_NEW_ENDPOINTS.json` | 628 | `0cc8536425fbfdce20c0334eaef36814487491a38b182b946c236b7136504718` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/VISUAL_JOINT_RECOMPUTED.json` | 1,995 | `98b6a0996dcbb8d012de86978bfd81b68523362e2644a1b83a4490a2acdecb8a` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/history_preservation_confirmation_JOINT_ANALYSIS.json` | 6,387 | `e4d8be46b7a13eb1545de730b9095ba78970b463a03ef40a1964af6e50e9a34a` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/stage1_logs/S1_stage1.uce_only_log.json` | 1,164 | `d991254cea4faeaa51445f4c3170546068de05891ee4f5275a919f9996b3cc21` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/stage1_logs/S2_stage1.uce_only_log.json` | 1,168 | `2953f41ec1f4f3bee19b4a7ba9ec18c001871eb8e48d6b3e58837150e4b69712` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/recover/RECOVER_ENCODE_S1_RESULT.json` | 1,677 | `806ea8067bffc14d0c44a12b6c78f3a3474abdb51275cd892e4735e041a6e102` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/recover/RECOVER_ENCODE_S2_RESULT.json` | 1,693 | `3a64d18857c222afdf900284f44798d13d3b35f474d92e0d60a1753c8a7ec37e` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/arm_builds/RESULT_S1_BLANK.json` | 53,353 | `dbbf90a7497426c694c049e3c86e290b8e8c17586be98bc2703197860aca75b4` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/arm_builds/RESULT_S1_HISTORY.json` | 53,351 | `96e4c8a187201003c11993afa5b32aa57df22b52835ef920bb589b69e1d2248f` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/arm_builds/RESULT_S2_BLANK.json` | 53,368 | `61a3e5297f15e2704d308b934206f121a4fc07e494ff16dfbc82cd5b0f4c9138` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/arm_builds/RESULT_S2_HISTORY.json` | 53,312 | `46b2a6bb731cc544a0ccf4af0c5e372dc969186888693a8594890ed657b08c98` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/closeout/ORIGINAL_IMAGES_HASH_MAPPING.json` | 56,191 | `156230be484899f94e34868ba7f5a34e6cf96fd7cdcb6c12cc11a4842b7aaede` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/closeout/JOINT_ANALYSIS_CORRECTIONS.json` | 4,150 | `1cf3e2c3a0993d7f742037d0b5507f452af45eaed75f3cb0fddd835a40dc5ae7` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/M0/image_report.json` | 38,320 | `1fc8645824b14d2d64efa2d1269e67920c082d3ca80cbed4ed948255c0b53bd8` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/M0/classifications.jsonl` | 41,849 | `24712cee2a1cca428ebc366c9ddf3c0c2188957fbcf40fb81b7c3ddcab95c049` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/M0/classify_report.json` | 224 | `e5e5f8e930bca2ce93c4788c0fa75dc7e63076b6df6f44a41eeaec8abdf1089a` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/GolfBall_parent/image_report.json` | 39,646 | `d7e5575125c84cc011a8365eb018d264f3af74bb6739fab922aa74409bacbce5` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/GolfBall_parent/classifications.jsonl` | 42,894 | `06330a591444c1046cdd4e2c00e7542075b8ac532e5057e33b742e347fff56e5` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/GolfBall_parent/classify_report.json` | 236 | `ead4607338a06067785b9fd804cacb8114ba66c32c5e32d42b46f491948a756c` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/EnglishSpringer_parent/image_report.json` | 40,220 | `a34e609e1d91fdda84974715cc1f6b9bf7e81786c85d8b0228cb0d9b50682ae0` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/EnglishSpringer_parent/classifications.jsonl` | 43,279 | `712b2d5dc1d87853842a4b97b36009aa69d72741eb3035a0d23e7f6ce7fab570` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/EnglishSpringer_parent/classify_report.json` | 243 | `1cc77c669776aa0daa45d324076e16da13deebd59b7a507156b1ebb7a4375c1a` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_Stage1/image_report.json` | 39,193 | `d85a429229f0428df83fd949802a573d351bc09a5ce18f695b817eb07b04e340` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_Stage1/classifications.jsonl` | 42,401 | `7d93a7468de06eb1ead377900f7d950784e74742fac3c3ca2280e3907c26b048` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_Stage1/classify_report.json` | 231 | `925ade75385b0ef80693700e8bafedba1908211ebdba81fd176df80fd5652bcc` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_BLANK/image_report.json` | 39,110 | `c7e0ea0ca657482dfcbeb7c560d48e5d4d512c39f4a2fbbf1cae36937eca6590` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_BLANK/classifications.jsonl` | 42,337 | `18b13ece84ad1f08f204cd51373889dc2a395fb382d17aa1ac35412d67a1572b` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_BLANK/classify_report.json` | 230 | `1b0b67637c979d0418488a0df3702b9ce1f8630109c3d2f99395f59bac17a598` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_HISTORY/image_report.json` | 39,273 | `c3da9c456257f062c8cc99636730f347869ba3a410bd0f2563760e7f76de4f4c` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_HISTORY/classifications.jsonl` | 42,509 | `8753a646eefa52ad5f09930521158ff69f1c667180ab02e04df306572dd12cf9` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S1_HISTORY/classify_report.json` | 232 | `4a98b90338d7cd4893f5542afaff81d02fa611cccb5d4b3401f5a7d3c94bec8c` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_Stage1/image_report.json` | 39,193 | `d2e3a4db9fec31df90069952028e1a95fc63fe64b343346876881047f2ea9b20` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_Stage1/classifications.jsonl` | 42,266 | `1d43684b80983d8230af5f0fb6d798fa35d4b3d8a894e34ab2ec09eb5c8001dd` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_Stage1/classify_report.json` | 231 | `42006caa677ff177951ea7e2a723ce70efa0ad3545bd1785017513e19816f5d6` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_BLANK/image_report.json` | 39,111 | `4b606a032b1ef6a5e1a26518ed5bfd5e7b3ea03eff9af0c0df96b0eba3eacbd1` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_BLANK/classifications.jsonl` | 42,196 | `37b42f5ec61d245c8d0ff9823ae786b01901d9bff425ef881af686a43d339ac1` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_BLANK/classify_report.json` | 230 | `970398fad3ad9b670c15148314bbe6671ece680381220dcb11ad1f9d7758f3f0` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_HISTORY/image_report.json` | 39,275 | `474437807d338c8321abe24d4c327f99ed4b9607f1ece7f13dbaa63b5fba70f1` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_HISTORY/classifications.jsonl` | 42,358 | `321b283802c6fe12b4dd2946314fb6d8c8c7b1e05ff6cb2c94de470c4db26148` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/image_reports/S2_HISTORY/classify_report.json` | 232 | `a80d58a58b99b322daa8e45dbdfd9c9017fa5fedaf6e4d3f9cec6d5acf3d8271` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/encode_retain_texts.py` | 5,404 | `94cd3742140fb87bc089e956b6916559d3a8192d70bf0627e7c4846b04fb16c8` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/run_build_arm.py` | 11,896 | `fc38b7a7f77ba82a17c16349367062f684410208cfd2fbcb347e0f697ed8788a` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/build_visual_review.py` | 11,162 | `a6db0cd19c4da095e4b34cd1a959758ed1e40132f73b8a5b47e9e6e4bcacbe08` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/recover_and_encode.py` | 6,110 | `c04a8150478eb1687916183984f9fda97ecd0837eac3caca6a7ad72e7c042800` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/run_build_arm_seq.py` | 11,084 | `8ca2f8f81c5d98764728930d7a8bcbd9f831cdd4171bfad94c4cc3aebbd23ef0` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/review_lib.py` | 3,123 | `9d5e929ccf647f7f1176965519151490bc86f062dfd34b258ad58ea84e93e31e` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/build_fresh_panel.py` | 6,242 | `e24a88bd8615288a558d3c17d91a6ecf4eea0dc1c1cf1f99c1a94effed6ac787` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/build_shared_input_review.py` | 5,522 | `8fa1f9cccf7e66c449ae7234c87fb2b9a22ddfc531c712b25d15ac6e8d0dabe8` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/build_new_endpoints_review.py` | 4,014 | `e119090542e696d8670e74149bd244554a9747f5818d99bb98909fd9df3f126d` |
| `results/rece_seq_golfball_englishspringer/history_preservation_confirmation/evidence/code/make_uce_only_checkpoint.py` | 5,353 | `cf07f4db9e50a3f564dc6f1c02a88c1d8dd1978c0e87c25469f137012f1e9bc7` |

## Visual evidence — preservation_content_control, all 120 original target images (both arms), retroactively published

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case00_golf_ball.png` | 503,935 | `e41230db5bf2acdf2396595ec7fccd17311fc47193325a7910f27d79090f7bd0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case01_golf_ball.png` | 431,465 | `63da36eaf20a4a0009ce359450d80f2759867fd1469100278a98a522a51e4f90` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case02_golf_ball.png` | 342,199 | `132b4e94606ac8f8593a0783c09dd966a52893d8452ab9eefd0ef151d6c631f8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case03_golf_ball.png` | 460,943 | `337e60a1c0447dc1ac7a187d0a1947b8c7c5cc0637ce33f9e40b8c60552651a4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case04_golf_ball.png` | 403,862 | `7ac620dece2f0ef25fe7ef4fac7b21984de24d1f38ea781e921b86729b2d764f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case05_golf_ball.png` | 428,922 | `d54dff492da0f1cabb31f34fe50a2422813c9de37664550001b83a11acaf7d23` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case06_golf_ball.png` | 378,685 | `9b4faf5658f38a2252c1e1784db3b9644945ff55c3ba4991efd096a1c73d195c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case07_golf_ball.png` | 494,327 | `01891a62abd3604b3f3db87f543cebb90f530b52cc2a228d951b563b321b34d2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case08_golf_ball.png` | 402,761 | `aad45d365bde288a247798bf21014ec834c42fbe3068d6d5c8de236031e66c6e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case09_golf_ball.png` | 404,321 | `84597349214b451989906e6642a719967d09a56f9e3f3cb20f6a8b296d97d6d9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case10_golf_ball.png` | 369,587 | `bfa7962115d9dc2b5e33c78d63429505afb2226dccc175bd1e1d1f85dace722f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case11_golf_ball.png` | 399,795 | `cbb2264314c61867382364962a307db15e881ad899d05b7d4cfee330aeb91cfe` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case12_golf_ball.png` | 368,178 | `9f9e7748a9fd6e89d3df2ec2087310419a49da9ae6da929f2844e38858da05d0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case13_golf_ball.png` | 413,110 | `03c44d86c73a941da25f938344197d385d452455dbd83264f813e2a90a86498c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case14_golf_ball.png` | 376,114 | `0f4aacd1b916db086a231bc989556820f81ea7387b84de12bb0c005c820550c6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case15_golf_ball.png` | 433,549 | `fabf657c1a813e806f2ba9046ae631bea40b83649a5ff64daae751153d17c44d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case16_golf_ball.png` | 455,733 | `1b2f80688d17b425ded3bdd0786fe2b1f256ec6048223af9af1509ce8ce9794a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case17_golf_ball.png` | 445,781 | `55a13b92e7124c404b68fcaa0f7c93e43f68a5361d63f53a798f6c84739f0455` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case18_golf_ball.png` | 412,333 | `e08d7ccfcb18ceaf7db991b2b44c64320c2748de87b08204ade1bfa13acc45fc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case19_golf_ball.png` | 503,310 | `e3b3b43cd1965c0190cc0080e77146c76a29bb54a2efdd8e47a39dce379a0414` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case20_golf_ball.png` | 431,538 | `9bce0e09a1c13048b1c9fffbb29de6862e9ce7c231e47f81610dd02bec1cbef6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case21_golf_ball.png` | 420,892 | `71cdbcd1fefdc6176abeefb664021a6f301aa855deb3da7b6b0db6f0e32f995c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case22_golf_ball.png` | 408,533 | `8f8a9fbeabdcd5abeb52bf14bca922962b593eda04af1046f8e442c5d4823dd3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case23_golf_ball.png` | 452,802 | `f380e01bc086839f31a2377869e50c1a7dc7a9a8146f2ace7bfbdbb9886716a4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case24_golf_ball.png` | 493,427 | `b91a604a166b9a7602f50d789ed22d3ede9891db93d8ec4c5c53fe35dffe27a9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case25_golf_ball.png` | 422,522 | `7775ef2b7510312edb355bc9a57ab29c29b33997793df025a91cade6cc7f854b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case26_golf_ball.png` | 420,918 | `3aeedcac81b9d6c54563fda00a0938a72a69d967acb2951c24b6aefd4187b125` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case27_golf_ball.png` | 488,422 | `26c0860afcf84dd98c1feeb1406d74b9fd0fa6836bf969990799e8ed9cddeab8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case28_golf_ball.png` | 468,210 | `27200677684e4fa7dd9e657de458d05f34e0cd4a5b85c9a45852d568c00f0431` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case29_golf_ball.png` | 352,408 | `c627584b62b9ef325315d02de5a906c8c8dec105921c281f7c592734ef07b018` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case30_golf_ball.png` | 466,761 | `0e057cd767ba72eec834c6f8490d833a24b623b09aa6da3c9027e933f9cc32eb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case31_golf_ball.png` | 486,913 | `cc0920b8999041c19f7c2d89535831b7aea2c88710921e3d34437203369e486d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case32_golf_ball.png` | 360,136 | `a5a701c1ba8ea82fbf5714a6d82535ac501e044792a9794d2bd571b54ff4c75a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case33_golf_ball.png` | 371,175 | `efccaffbdb1de0f9ba796fb740c15c4598ad772914cd64263b6e862cc415081c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case34_golf_ball.png` | 366,599 | `5fbb49e68a0fdc90ee9478d4862b5a3336e5d5778bf0887a21837f317f5b1687` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case35_golf_ball.png` | 437,221 | `a91a454de1a1d5c70f2b9be8e76ebbd8cec89ec482d9da63b7f35168527644b5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case36_golf_ball.png` | 492,746 | `6b84a66103f6c7106c5d1ea6f47b47b0490b5ca962776c3c07be959025cfb8ef` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case37_golf_ball.png` | 452,593 | `57bfbc636eee9e00162868500b0523d89e7f26dac1a07a924614d69aa8a68ed9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case38_golf_ball.png` | 424,120 | `4414b372fa6bb6c58b2395a1696496decadb78ddf7c09b9a84c8e481cb389708` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case39_golf_ball.png` | 426,357 | `572b62dee430752f74f8933c10f01246ead4d5d1853d27b2a3afced99699f767` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case80_English_springer.png` | 521,393 | `e3b99060ecf555139d1889dc37e587cd9f5201dff0c0891f4d890bad35e74a78` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case81_English_springer.png` | 499,041 | `286dd6c0f0995f969cc4bb75cc4abb744f38bce6eacd8627f6d510c76ff53a9c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case82_English_springer.png` | 547,935 | `faed599cff0ca25b34440fa52f6e9aefcb30b91a57b7eb3dfa145b381abe7a09` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case83_English_springer.png` | 531,227 | `126e629397049e8aeff492680c416c26ef946394ac5b9c85417630368442adf9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case84_English_springer.png` | 569,700 | `c6549932ca7c254010688121c9d1fb13e3776b1d3426185d1cf28ed639874663` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case85_English_springer.png` | 498,459 | `a742be9848b9afd2b0a2bc30e96423bb9a2ef25991031b4735187ec5e489a647` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case86_English_springer.png` | 572,179 | `8205ee376221d5f29d875268d6da2b67bdb622d9d6e757b30cc9981964c0a96d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case87_English_springer.png` | 478,622 | `f90c2b000e14a346c76e86119fd0f2a7e480b55bb7fcce248c42dffb2e08b0f4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case88_English_springer.png` | 465,777 | `910973fb724ad778c9054bd186beee87eb9aa58b8b6aecdf605612d2276a1f04` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case89_English_springer.png` | 432,352 | `a7f174374d784dcb1eb199b45467f9abf0731367f026bf0d2acf6796870b031f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case90_English_springer.png` | 500,038 | `20a7a2152d2e5bb45e9e313ef6f1f12ed03626c9d8d0eef880e4fdaabe1e141d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case91_English_springer.png` | 492,256 | `f520977d1f033b051c98cb6f09229e0632577327e23900e8f157bb1e08d76b15` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case92_English_springer.png` | 411,390 | `21545eff983d2934d295170c7e1371749ccdbf85b9c9cec1dc61fee45d57bc28` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case93_English_springer.png` | 486,628 | `40be035ef40583641fea168b959d1a3bc8b2cf17d2db8054135f797fde369af1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case94_English_springer.png` | 412,586 | `a0716c374acba3c5bfaa35eb492f80c4bd8bb3384d7fa8c4c619af1bc54693f9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case95_English_springer.png` | 450,621 | `8058640ea2421cb597fba47e0fc86afd0527f50a1bb6785bfc1e255d536b6834` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case96_English_springer.png` | 516,412 | `7ed39ec8c56bf22b2a889f92511eb712b788e8c571cc5f9e2a18f3a7551535b1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case97_English_springer.png` | 465,164 | `5f9170190a8bb80d89366acdd4f240683937e270a030b4dc984b611a8be0993e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case98_English_springer.png` | 509,693 | `ac4adc51d472cdad040fadc3cb33085af4f5cbdd4bcfa8bd54dadac2b5c527a3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case99_English_springer.png` | 458,379 | `46364d6ef496e6ee3c3613215320c05f145f338244d5c881dfd6c6d49b753135` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case00_golf_ball.png` | 440,880 | `fa0e2aaee789639b52e19fb214f1c976a86b84c5912ccb3e2fdc5bea119d936c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case01_golf_ball.png` | 475,273 | `4811bad2861db6159b076b45d4c827feac9ba21777ab295b06cdc1524e1c6357` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case02_golf_ball.png` | 334,802 | `a3065e8c6f693d452b5d003f0ea34c0d89cea1eef4228ff7752a4992e7e511fa` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case03_golf_ball.png` | 478,787 | `81316422842117a7420d9d176d1931c68631bead7a2b5aef14198e262f6c9467` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case04_golf_ball.png` | 443,622 | `330237c542fc7a09ac5e5e12b4d8fd890d4600ef245755a9841efd6af8dd8851` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case05_golf_ball.png` | 501,368 | `b8d400f0eabce79a546209c03ec1b69122ef033f770412076372b23bebfb3ca0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case06_golf_ball.png` | 473,013 | `29be8faca352c460c83be74343cf7bc92d73f1e18cc6acff5a7bb87fa932d9a0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case07_golf_ball.png` | 488,972 | `8c369cd8e2c9d759d26c73740ea9d9e42f332c123ef0850615df3d4e699dcb10` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case08_golf_ball.png` | 479,795 | `15939947ec0de3f2a2d93ecc297eec6b9064f5fab0fca5de3e043283056363a5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case09_golf_ball.png` | 486,620 | `ba7f2caa55c4e86ed7fd2ae86b7a7b1f9a677c64e2224f84309bf494f639f8ba` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case10_golf_ball.png` | 459,931 | `c6f084bd5a9821118a37e78d960d7e71da8c0cfc6cbaf57be6da2fe7d6de05fc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case11_golf_ball.png` | 471,941 | `5ec6574ac79424b739aeb97857290330720645887c022c15f0dcdcb0455044e7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case12_golf_ball.png` | 463,185 | `121ffc036597f12d9da713a6dd71eb3a8a3353479ddd5a50be4c576ae7e5bcb0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case13_golf_ball.png` | 476,495 | `eb9aec4097a3b2d1d23365ddcb35e9ee91fd26ae9537d8411cf15692dfc8e5f5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case14_golf_ball.png` | 464,268 | `a899b15c4dfa259bed42940f549a955acbe0fb90191efda209bb4abf1b9408cc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case15_golf_ball.png` | 439,578 | `2b9b6dfe9086bbccd8591987d680d626c057143672f571c10b1e01cb192d5939` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case16_golf_ball.png` | 469,416 | `16e9c43e2ef56a46e4f9676986f7edf196ac38d364ed213f3b66892a5c062664` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case17_golf_ball.png` | 549,743 | `145d6ce7309e15054bc7412af50fa0e0fb5c77420bc7cd85f50c528c76468a91` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case18_golf_ball.png` | 339,072 | `47f5cecdd6f93ce18d6c2eb07de76d1c70072385e5d393ef7314a1b9cec6ae69` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case19_golf_ball.png` | 528,724 | `9211ce8849a556ef5390d5ebfdb932918171ff2674f713c92f1c244d507f776b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case20_golf_ball.png` | 474,049 | `a286133341074469a1a42684682b654ca4415e2d8cd8f16704a7d74fc862bf90` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case21_golf_ball.png` | 508,433 | `9d18aaf79aa502f8b0057e08f1fd88aeceaf3cb5d9d02e295f511849998dc9e4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case22_golf_ball.png` | 440,877 | `18836e9e2cf61e2f84345a28301fa5ed374e81e140e1ce7fbc71484f9ba8e732` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case23_golf_ball.png` | 500,119 | `d5a89717d118b70980ebf981a5e34a93adb4681b0b0aaf0bc944193de1091c99` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case24_golf_ball.png` | 483,813 | `4b0e359737064d7e754c35e5067cfa17703d9c4382616616dc4d79f0dad13f72` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case25_golf_ball.png` | 420,884 | `4b17a2c778b825d96791544e3a39518db20450d858fe724449bc542d60fefe3e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case26_golf_ball.png` | 354,136 | `2f91f6bb8ab050665570fbf8cd330c94f49a92b16467efe22c3c7d7b71b6809e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case27_golf_ball.png` | 446,601 | `6512962dd60ce75595639551667c41b067a698d1466eceb74a2c11163c1c4695` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case28_golf_ball.png` | 459,682 | `e699cd543bb2e80c2fa4aaa71a8c38de07badaebc193fb6f5b693b1b3c44dbdc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case29_golf_ball.png` | 373,756 | `ac84894883984b055b5886779c617745d72b2065ec82e2f0e63f26e38b978cee` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case30_golf_ball.png` | 414,542 | `5f4e13527d94f595b2eb46e95dff51860a428009dd28f306ec06b18435108a07` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case31_golf_ball.png` | 400,485 | `38602ff8c178bbecd07491ffd3ba26a81ffb10d3fb73f0b7dbda0093cb6cde4b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case32_golf_ball.png` | 448,728 | `0f37992f9c572e292c947d5307b1f73de0107d77f66a5e90abd23aa97ee24e57` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case33_golf_ball.png` | 448,182 | `61df166440d39db24c0b67e96d9a6f1efdd4b761907fbd9639eac6e5e723d096` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case34_golf_ball.png` | 357,875 | `70224ed39346be05b48921e8eb63a73fb5757f565d480c4afd8a318453b6815c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case35_golf_ball.png` | 486,997 | `5a7f850a3592da197a686320758e1e224f333b4087f8cd8ff395ca7396afc603` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case36_golf_ball.png` | 516,898 | `1b7396d88c4cdafe29c585abd24dad929a825cb7915696ab0656f7fa1bc297f1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case37_golf_ball.png` | 476,192 | `34020d30c3e69ef4c14308139b417b0ea8f42a4ca058ebbd6f426a36a276b24c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case38_golf_ball.png` | 355,521 | `92b16471d863a245db1b97dd6a207a9a789a42d9b214996a954be9adccac8c3e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case39_golf_ball.png` | 479,100 | `7a06bfb47b21d61cc15dc6124db65f7207d6bbf40ca8efac45e32aca9b3180f2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case80_English_springer.png` | 547,286 | `daf81dc0d56cfbae6886cc4203173c0229a12aaf1bfac2cd58d8564af08edab6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case81_English_springer.png` | 505,801 | `5d1bc3f5136240d0e7e44a358d77661d8349f815ac20934df62fbfb8e04f06c7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case82_English_springer.png` | 554,950 | `025833c92dd651db85404ef6a444c25fc86497dfadec9c42fc0c1bd6604d5f13` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case83_English_springer.png` | 541,192 | `8a4db72176bc68de618bc8bf1936804bf7b0521bef057e69e4e15d25775172b9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case84_English_springer.png` | 572,049 | `f113ab3d0de73f6aee61dc3c24b072e7718ab048ce9fb94ae265b0da3cbcf462` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case85_English_springer.png` | 491,752 | `f6f4121df9ab83fd8f0d3529bd04cbe9c4f6e0b6eee9da946900cb6b42a3a9fb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case86_English_springer.png` | 569,597 | `2f431eba759c65ad50031bbfbf76905a25ce2aa5bf19867a6d568ba51aecb6e3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case87_English_springer.png` | 489,766 | `f4fb0bcfdfd4a93df1fa5c151b6224cdb999b31b3c6471b7a814f2cc40434025` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case88_English_springer.png` | 467,705 | `ed25d2cb219ebf4a16a5949d58778b7e522adda4180bea2a2e6977bfa4f22b11` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case89_English_springer.png` | 433,750 | `587cdd53cf532529443fdde010e6d01336bd6cc01fc9998c74bbc9a0a7706346` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case90_English_springer.png` | 511,349 | `e0def889e8cab4f31d942503ab3ba8fb248c38d42f7dbb7add882a14acef3f41` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case91_English_springer.png` | 504,275 | `c3f5af876db514ccb7b6347fccf81ebc1d0ebfeaba5cc029b9991f1bd164e22f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case92_English_springer.png` | 419,343 | `17632d8bd0e2e5fc8703349c6e588a1b5a36ac1c61441a14268ee312574ab0fb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case93_English_springer.png` | 496,655 | `ed86fdd91b53601998994a2329b2728954ca2cb7c22cd2e2b197e62153447e20` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case94_English_springer.png` | 428,169 | `d781d586dfd960de124b5df1f84e8d4affa595e25e408c3f3043413063faeeba` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case95_English_springer.png` | 461,135 | `47bd9050238612a18974611fb812fd3b473089bad28ed29047bb9c59f63fc5fd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case96_English_springer.png` | 537,212 | `41a721711b3b9148419de7c74883bb9ad889671dacbf220adbc4f6a609ff547f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case97_English_springer.png` | 455,396 | `f939067bc373b94aad8372f3f46e16114e595d58bdd6e4fd86cc4dac3a816d5a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case98_English_springer.png` | 521,653 | `86bc2683132c0a8f17a587a93a0b25f479375a720130a40ba774369deb5e718c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case99_English_springer.png` | 448,855 | `bdae7b90349ea25b97673e8225fb9c3e640d7febcf649e4539f7d782622f0976` |

## Visual evidence — history_preservation_confirmation, all 380 individually-reviewed target-role full-resolution originals (M0/parents/Stage-1 references + 6 new endpoints) and 17 fixed-order retained-class grids

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case00_golf_ball.png` | 497,075 | `c9bf2091088016dd573fdd623f5c114a0b3b9f024b88f7595cf19cd8628b1760` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case01_golf_ball.png` | 528,643 | `c6db0df4614090ea7bd612d49e6b03fd280eefb6625fc00a49616beb5effcbad` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case02_golf_ball.png` | 504,035 | `a2d1c85804355e6b4495afeb6f759a8c1dd066c0142439d45b57fc521bd339d2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case03_golf_ball.png` | 543,962 | `fe8afe4c74771d304f1fb86327dda670bd2e8395a51426d0bcee9cc5706a37ce` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case04_golf_ball.png` | 410,605 | `f296688c6dc512ca6d4aed5e69bfa61136ae1dd4f0d6e34e76a29988f4fdf5cb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case05_golf_ball.png` | 392,191 | `0f289596ead32d15dda446df92ccc54c43f882b2fd95da978831e6a611b49e0b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case06_golf_ball.png` | 401,476 | `14daaf691bac0735665db77f403e2bbc4d99f358f8ac82502c012f514eb16291` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case07_golf_ball.png` | 395,917 | `6e38d97b5b71e2a770f025d571f2a7fb798419493bdfc26e4fff0bd84baeff98` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case08_golf_ball.png` | 457,449 | `18ef7da8d864a752a3d2f05a3dd3cc8bb4e04665a0e0b6967d8af4de69c672c4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case09_golf_ball.png` | 525,894 | `9dd54f60fc1f84f831a8ffa84a7b2364063b024e8e4f62a90b90f715208a1f2d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case10_golf_ball.png` | 428,656 | `642f3f8b835d82716524b641a250bf165edc1e72aa988acf114a1922c3d10784` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case11_golf_ball.png` | 548,204 | `00cf5f22b7f2e2fbd33c574232bc54f5616367cc3be36c441ef61a366259c290` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case12_golf_ball.png` | 366,860 | `ec402820105a6645770bd8dab8ec821922906140759ffebd96b7694cb6bc4f84` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case13_golf_ball.png` | 294,588 | `faba4d0614799b68d85f3171af9edff3a625ae101aaf5d4ede26911455022f7a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case14_golf_ball.png` | 354,201 | `9f2799f43860d3765946f631f1339fed59ef73ce63a984bb2bf013f008dbc865` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case15_golf_ball.png` | 344,442 | `9bea885434a0e39e145a91e9e7c615c8ab7f4e82db66e679488c2018043e4b87` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case16_golf_ball.png` | 427,520 | `eff1e05081b21ab7a29414b2b1155d1dd8acbad342123d4a25c711d22ab3b0d2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case17_golf_ball.png` | 323,855 | `113efce7e08e9488bed43b2b004230457cf803b9b8e8b4ff76c6fd8928496e57` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case18_golf_ball.png` | 404,911 | `c74fb281b540b630bcf48a468c6092c566e33875ef1b390359a5a1a705a01b8c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case19_golf_ball.png` | 324,024 | `eb40922009c4f9d14350abb3483d38df799e369078d13fb858c6b556ee7df48f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case60_English_springer.png` | 412,684 | `8f7036dbc95c57eeb88614c4421782bf5a6147b8012cf46bf35727d2507e6eff` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case61_English_springer.png` | 396,611 | `a4cce15041ac84b02fcf1309aa511035168ef7f1f655fec357e121aa3c306a4a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case62_English_springer.png` | 428,920 | `897a8ab43bdb627a65b8c9e37c75500b0373042e31fdc13f789237af5dcb461d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case63_English_springer.png` | 393,132 | `bbcc2b0884257e0a853c8b313378f8bda022c97dc9fbf8106c78f7f43e440c5a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case64_English_springer.png` | 447,388 | `2bf1fe5e4bb7d27a71b486c58c5f1949e1fc9c98cf57f0ee0afef9d50aa17b81` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case65_English_springer.png` | 457,303 | `826aab76903fb63b8df2f6198b1f5620dc6efbac6db7b5cd9429c11434741a9b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case66_English_springer.png` | 440,842 | `465e5fb6b89305af53eebfb15935276b73c54e24667b03000818381cfd7353c5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case67_English_springer.png` | 442,477 | `0280d2750a75b25b736dd928cee2796b08385db53f1dddce98c56a3d2d151b74` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case68_English_springer.png` | 378,721 | `59da96484bd66a6d9bf21173691d166652e2edb37872088d6f582f0c88371536` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case69_English_springer.png` | 373,506 | `6a516e0645052543fa7bb095e1d451e157ab0f49c6338e2b88d0edf7f5b098ce` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case70_English_springer.png` | 361,108 | `23026d294fd7a9240a4b22785d6567c4c1262ba356a7477a314538aa3b68efe4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case71_English_springer.png` | 359,574 | `4d280c543414496f7b6b1db88a5d64e65983187d211147b0ed9bef4e6f24f966` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case72_English_springer.png` | 434,766 | `c100352ab88e22431685c51c22f56af6cefab03e031449b3f7d59fbc1586c5a6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case73_English_springer.png` | 460,987 | `e33f8f8f9354b02bd874c071413200c92a2046889e276f1a79ec69c9736fec84` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case74_English_springer.png` | 449,037 | `ccb2549dcc09cf558a0b1366d78a3356b40319b278e003c86d25dfde06dfdbfe` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case75_English_springer.png` | 451,140 | `cd34d6cdd127c1a448a13c6cdf5d2b3a341ce25aa4bc7154a5397aec2f6506c5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case76_English_springer.png` | 408,855 | `cd6019e8558fc071c5785c5ec3f5f486cf25741cf8b3a75ae46a27c4d90a92f6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case77_English_springer.png` | 350,961 | `d6a4b5368916af214a7ead122288542500e4f1068652ec8484448720694e7f5d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case78_English_springer.png` | 406,754 | `9a5cc2f6fdef06a7a4db6fa7a17123346c5f4526aa2fdcf39972f47f23c2c3cd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/EnglishSpringer_parent_case79_English_springer.png` | 338,347 | `adfabf469ffe360d4831fa240613d9e99697b3f9e3412fa20b50d66c79e2e0f7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case00_golf_ball.png` | 430,514 | `bd4b9f7111ed9e2997fe0f319573503f10930ad0203e348498d450b87a443dd1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case01_golf_ball.png` | 498,067 | `716896cfe7c13a1f1dc9f12074ddb30e0a5c764d8be2e37c29a3b952f48cb385` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case02_golf_ball.png` | 432,413 | `29468c4a3a83ae66507c3377724e9c20b039af005d2295f8606c5cd512f7c11e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case03_golf_ball.png` | 511,875 | `bd522bf43792c9e118d8dda47af60b3651f44573d95464a43098d0a1def52d00` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case04_golf_ball.png` | 453,836 | `a2802e7c431c3bcbcd8f9404f99d94daeb46628eb12577f76fa185200db19ec0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case05_golf_ball.png` | 500,068 | `29a693a8ef31db8e50a256ed7accb7c4b58a57a7739c8d70505d9b6083774a9b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case06_golf_ball.png` | 466,618 | `243e59205b9dd2a83e99fd8ea7f9569c40e6cff358dd913af5a4a450014dedd2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case07_golf_ball.png` | 541,481 | `c6a1f9f9b4a381f106b0c217537b0fa91a851c3e6a9031efaa28212d4e52c88b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case08_golf_ball.png` | 508,970 | `6911f32f4aaa9f5f68b761cec9f0bea05c6cac61b10ad2bc0c074b172f333873` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case09_golf_ball.png` | 453,354 | `3ec20c5a43a9e1d82a9e0c2ab7a11f674b1b04094ac389d017554cdc94a4fad7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case10_golf_ball.png` | 483,747 | `b667bda57247f95106665d39c943753b4c492f6fb3ef4f2a73a5687dda03bdb2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case11_golf_ball.png` | 521,240 | `082c01c93c1d12242342db17c37c99735627fc74110ed98dbead7c00e3098694` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case12_golf_ball.png` | 580,631 | `da2ce912a1d95209a8650e7a2e79e714e07fbfe5f68c6595cb7229b520652073` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case13_golf_ball.png` | 352,209 | `2c375c705f459eb81d571c40d9861579aadd4578b97bcd122bdba9afdf7d499e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case14_golf_ball.png` | 683,008 | `41c2fac7ef9c63d7566035341fcafc180f3fa0e24c16932a9b712b54fd58c7d0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case15_golf_ball.png` | 392,713 | `3a4d1dad7d8789856330a282d85643f385bcb89e7df3313f5b753012f9d4f5b1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case16_golf_ball.png` | 421,732 | `797cf1826cd36cdfa4ffe224e135751f3c781e46d9147a487a33a97ba9b1ee46` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case17_golf_ball.png` | 428,888 | `b15a973c5e928ae168384d022f5f3fa48015e94e624d083046b498e76c8a39e0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case18_golf_ball.png` | 419,529 | `2ee1aba17264b3734a3527cf2380306e374d4b947de7ca66ed5fec1bc6ed4009` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case19_golf_ball.png` | 388,500 | `ddd791695d8abd1cf44c99d125e318048816c88393db54d343093799e0175189` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case40_French_horn.png` | 467,948 | `8db8514af812d19ff87495ae9d1e8161745ff30b99e289eb2f4a4564fc02316a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case41_French_horn.png` | 466,517 | `50e40048e80823b91bc8f8cab3c42b0cccfdeec7011f11a9405aa99f32e3cc2c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case42_French_horn.png` | 455,565 | `3a32e08c4aacbbcbdb2b27d1ab4f5c7ead36e8598d42192bde12a44815e77ab8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case43_French_horn.png` | 510,435 | `37575f7488c77cada79f145bc38a827ed081f3084e2bc202bfdb2dcf35455362` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case44_French_horn.png` | 465,826 | `3a2f6362ff3d18330c5bdb2f0356b417043a1772b434111e1e2f641bb0ef3254` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case45_French_horn.png` | 454,954 | `892fcc76feddde2b3d1b683b47a1e85ca69d21851d2b9861a1f0196ebb708850` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case46_French_horn.png` | 413,626 | `8ab683e9ca89763d19189f1fcef5115cd8bc3b7f5bf6c4a224c56de6e0830a02` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case47_French_horn.png` | 464,803 | `02e28cf44ba18bd0eb086404b3a39911b5b69feffbf9ef0a77f01aa1263c1078` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case48_French_horn.png` | 580,500 | `26078105f46cce685e84ac1851b274bb626164806cbb91c439cb9f095dcb03e9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case49_French_horn.png` | 489,727 | `30edd445c411d8b4f46fc0d83cdde9aa70bf5fb8189d2e1a7d75d6c906e4c2a9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case50_French_horn.png` | 584,648 | `f36af1acdf715fa0df8cf39bbcb5ffc044df3f157131ffc425173c45180d7cfe` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case51_French_horn.png` | 469,565 | `67f7d069c5be6eae8c1ae346cfce4a5f2f3128937567c820605c034d6a94940a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case52_French_horn.png` | 440,734 | `6b8b4375e94c9bfeeeceafaeca767d3d607d1571e9e39b3aa886c3702d327722` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case53_French_horn.png` | 449,219 | `693a1237043b784e4e7550c3f25fb6426eb441b27449a64b18344873bb2222e5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case54_French_horn.png` | 429,764 | `4abc773c924323a0f9f24e98137d4f829aea7c3f50015b248ccbeb7877d7f12b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case55_French_horn.png` | 415,574 | `fc721412ab68343fe163b7be1fa6d9d16ab8fa29a3bad49abbed70e8331d82c3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case56_French_horn.png` | 420,652 | `3f987bdca07390f5008bcb646a5553d6a259b59c295de7bc53570d13575f2938` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case57_French_horn.png` | 423,110 | `785b09087b8587495d79a8410e0adc4f8e17d065f0da92fe19e84e4fda99fe46` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case58_French_horn.png` | 406,204 | `da149a153f4874cd65eb11b323445c3a07a6cc92001f00e68d44c1bd091af59e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/GolfBall_parent_case59_French_horn.png` | 371,087 | `8e34599888d010cedbe3f579beb0b38d8ef04893770fbcac3cdd7a117a4e8a34` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case00_golf_ball.png` | 548,238 | `9c75a3b1349bbdaf1edbafdb050133cd90025a6b894146f67348c6adef6f15a2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case01_golf_ball.png` | 554,973 | `64c7b0c19a99f5597055655beb4039d771264a0f4728151a33f8282206b7bb94` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case02_golf_ball.png` | 514,822 | `02f6c5dfea5da3c1c4e5ec7b6a01e0f579fddc27c59a8233196ec5e63b909e7e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case03_golf_ball.png` | 510,162 | `fd0e8bec2350cff561dec98bd5590e0a05b688f4d4aa0e145835be3f8699b5a1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case04_golf_ball.png` | 357,439 | `5f61610b6e5ab7765999a9015306f0a936cfe85d331ff6d87486aaa9c63b152e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case05_golf_ball.png` | 344,306 | `2ab4e7bbafcfe883740e7e5888b414a11277a87db42c5414d7d78975a27fd094` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case06_golf_ball.png` | 366,046 | `f129a3659ec028854bcd5588df338cee3807750367c9a9ed28aa0d8c3a039ef0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case07_golf_ball.png` | 359,120 | `b28c2c4064f0c51bb3132706af979ae0a5942098cdfc719c8ca3661ac4711470` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case08_golf_ball.png` | 580,265 | `0bb87f661d3cf4740ecea63a49021125f384684b7dc5f0ba04aeee42a8b6b271` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case09_golf_ball.png` | 512,348 | `d1feb88606c6690f7c6838ff73abd84391a26ba1be4a249c4f765c11ec2a58e2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case10_golf_ball.png` | 593,013 | `0ab4a691b93a284c9b46162758d04d917ecd28fba22a2c2075e1c33f455b19db` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case11_golf_ball.png` | 547,634 | `bf179adaedfd3d2d536371b5b7302bbfef7c50b95876e4fa528c606ddd11baa5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case12_golf_ball.png` | 363,491 | `0829279eb8a8d6a79f394b174475cbef08c086fe871314711feedaa001556b80` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case13_golf_ball.png` | 302,318 | `3388df62d11e06427711e1773b620f1d1df8456100c481711a560c5ba3ac8d69` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case14_golf_ball.png` | 376,085 | `8120bf55a72e7714111848f2ebe44ef463d4f0d3485af31ac278646339aecc58` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case15_golf_ball.png` | 336,975 | `19196425c0e47aaba2f3c6e5cb3a001784142565c299040b40e658e6a8f76706` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case16_golf_ball.png` | 423,058 | `87f6d056949328a210d88c9c3b85fe81ad7f5739bdb9407f0dec5ecd221c5b5f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case17_golf_ball.png` | 333,046 | `3e8b189a162ba4a0ed8c0e99c3081fdbc69b2f365862303c80591620149d6339` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case18_golf_ball.png` | 403,856 | `b9512ab7f0a49519cafac26433eff5f3715eb512bd5629ac99f40254be32e2ef` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case19_golf_ball.png` | 310,288 | `46c39901d8a4e7d9b05f3551b85c4c42d34462b1b42bf253c99c8785f14b8995` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case40_French_horn.png` | 506,377 | `5a01b0064bc5319b5a542873e3bea3f4b273a1ed99471cbc5b1bb353f2e6e741` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case41_French_horn.png` | 467,767 | `824cb4031390b0fa8b3198285220eebe644650236a7ee12d835d19743586a88c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case42_French_horn.png` | 448,391 | `89a82112b7fffead374b1db24fc5b2c0462d9b91cfa810f27bf936bac0226bcc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case43_French_horn.png` | 463,477 | `37dce2df00e30cfe4bb50d034a68972d4239d5be6121d335865d59a663347c2c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case44_French_horn.png` | 444,777 | `8d243140cebdcb2231d9137d75eae20d7d52ea339f1fd2cde062c92159fa0f51` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case45_French_horn.png` | 444,721 | `4dddb093c6691693a765a52eb6dce173383575277bab273584f0d7d42ad659af` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case46_French_horn.png` | 411,564 | `7edc3872d8f72348317b6385db932187d57904a6d3bb29804275980ee089d12b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case47_French_horn.png` | 415,711 | `4774866e60285a47d94d3f2735c1664211b6cbda96ec0a9b0e973a3160abba55` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case48_French_horn.png` | 568,955 | `f1fc02b1a93fdb05c6333141af2c53156ee0e67cc0e4ac90621f95c8e72c14e6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case49_French_horn.png` | 467,687 | `8661409d389c6fca762707732020d964ab1ecaa07b5f1b42bf619ae81ebfc2bc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case50_French_horn.png` | 574,332 | `6ec1c9ff567c35c09d4a9e7ae62f4dfe89ce55ee4fd73989badc7e7a4a0a996d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case51_French_horn.png` | 449,203 | `d69febca25ec4a2a3c662556f642b332e3ae826f564e161e623151fcb319a8bd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case52_French_horn.png` | 420,257 | `da4ecfa8046d69e469c22bae2fa8ca7b9dddf707f3bba4a82efd5fdcdd6bc434` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case53_French_horn.png` | 431,656 | `e9f91d864c1e5a8c9865f1ca6d5e32c39253fa17ba8a4569c86917f148909a12` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case54_French_horn.png` | 443,921 | `df4e010cf3317b735800ca61a5593b9fb7ee8e746d3e34e3ef2560e5276c37f3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case55_French_horn.png` | 414,910 | `397eaa68ebfd50880e2da6c46396acd5391be75c852043d3c8f02ea81362a1f3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case56_French_horn.png` | 295,380 | `1e37bf5d43d7628304c968c80cd962b95ca7e92c2008c85c49d6c9335668b1ec` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case57_French_horn.png` | 301,510 | `8154039390969c5b2278df9a52a756c8a7cc8df670b87ce361efcdf95a1f3bf8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case58_French_horn.png` | 287,925 | `fcb8c817826d241d1c072e6fde554ef5fea714c495ffc4ef1c69c941ea10ea7f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case59_French_horn.png` | 297,588 | `49773b117f43a338cb6f31c7e2ea4e5159cda6725c2783cee205b8c5f7411258` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case60_English_springer.png` | 341,713 | `a7575f413f182541bdb9c75019ba80959cc223c58d0c7edc9d1cc321f2c257ba` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case61_English_springer.png` | 392,408 | `cf7e97a317262d444ac95490fe82e1e854e27b9af6db0570449d4850fe88a451` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case62_English_springer.png` | 355,373 | `f7447c58b20c22374ab7da1a60bc9052390bb7e2b83cbd3dec7c1f9c7fbc4270` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case63_English_springer.png` | 413,747 | `3002671fbc7cb5caf329048f83cef87c87e1ee460ca8ffbb7d4e26bc2de079ff` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case64_English_springer.png` | 457,308 | `bcb04ee35e69c769ba8c24cbd0073a6eaf2abdb950421be7f9b36a481071239f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case65_English_springer.png` | 488,659 | `127f858fb0f299ec4c0d81e561aa462a31e2c1001b0f4c961a47879aa8b9f491` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case66_English_springer.png` | 425,820 | `e6ac6b33d1a865510e5e936f69d4a9aa0837c493b20a42ed46d4f8a4324d2198` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case67_English_springer.png` | 423,602 | `fdda08e9a0dbd48a802461bc635bd15dcdde91cabe4ff301bde759e188102f2a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case68_English_springer.png` | 362,759 | `f0bee91a7438b506611bc26d85528ab6a64ec185f0c7be2bd06d1ece9448fc87` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case69_English_springer.png` | 379,148 | `31077bfe851e02071ac8df9a022c896cf93e90b6f2c14f0e1c3d43473a125f73` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case70_English_springer.png` | 367,224 | `f73bac0a8cda17c773849fe412f4d0b00bf776957daf62ac3e3e1d64f316adb2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case71_English_springer.png` | 389,377 | `6eb24cf92f35a0ed1d5b48e6d6b2a20b317ab7e6559d53a39aa59d30b1a82e12` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case72_English_springer.png` | 457,935 | `8cd22253680dd5afae87c1bdda521d9634be9e2857814a389e714cb2cf5cf8ec` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case73_English_springer.png` | 419,177 | `332acdc65f5a2f420c171844299dc346f1a6c93442c0af0c5bbbc02d594f2c72` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case74_English_springer.png` | 458,658 | `e5079ad049d39516bbb92db4265ab1a2d2bc622ed85435817e821b402d4760f0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case75_English_springer.png` | 423,020 | `40f04eb721848e8a394f733f21174c4620d2955b6a9f27089cb12524533b5389` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case76_English_springer.png` | 319,426 | `af094fb0d8121b493f4461c8925475d63a0e8bedd092d02566be1f00f9b06a7e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case77_English_springer.png` | 361,222 | `ffcd77d6d455d0f470408c190db79c95eb4434b2bc0add7cba7756b9e7412b6f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case78_English_springer.png` | 331,741 | `9616254773b4dda2056296744d511fdd32b4e8e975c367d36968015538bc865a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/M0_case79_English_springer.png` | 369,365 | `476c038c0036e59fc1c633d62a52481a76d9d7351c1d4a8b40aed3031c34f80d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case00_golf_ball.png` | 487,595 | `d94b66458775eddfb97525a330021cf45970d2ec95d3e6aa6788b82d9f7bd827` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case01_golf_ball.png` | 466,194 | `d05037d2647dfdc38c69cea74c1c1fb70862a4327e35080f19ccbc87b2f0c866` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case02_golf_ball.png` | 433,273 | `e7fdaa5ef051a6a2ea3728af88cbb66b3582007f1d0b2fa548ad0fc03278d2ca` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case03_golf_ball.png` | 463,013 | `27235837cc6edfbad663966a405c76aa06afabeb80bb99860ad922ad63ecf7a4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case04_golf_ball.png` | 423,002 | `5a30a2d942adf0d6fa2a958b89ab2918ce7efb27252a68d6d4cb11341d4551fa` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case05_golf_ball.png` | 501,345 | `ee8d41afe3e3476a9f383f4ac45d4a07f996abc4fdc3c25cb2b2aa8aab0c4220` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case06_golf_ball.png` | 372,186 | `0e56f53334beb48807e6ce05f1c309e26664dbd55e3d97e97659ba140d5362ba` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case07_golf_ball.png` | 546,474 | `c6585f43f312ade1a9d431749f6f736a6ef2fdcd3cc78e1c25d78dc0ce8dd9f2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case08_golf_ball.png` | 441,993 | `ca3b7fd9beb33761c2d9daa16542b34d1624cf3adbb592bb0b5c63e72e618b52` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case09_golf_ball.png` | 515,799 | `27e9fa7c587e4486cdf23c9073cd3445a2e7fd1a16f0f4ad939e2f2e1b4ec56d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case10_golf_ball.png` | 461,523 | `04b93fff6f8f2f896daaf7ead2ccc7d9dd025e8bbf72cb504dc19c5cc06163f7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case11_golf_ball.png` | 494,176 | `6cc3be6fd2823809ee8f71c7f2ade4beddb1ebf919145f686425a163e14e3a54` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case12_golf_ball.png` | 491,309 | `1a4b2a4c8ee2be8f2b49a63af811c3e973cc0e6d36ebc0d436231d8849d7dcae` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case13_golf_ball.png` | 485,514 | `5264e2babb68c3b88d94e269c142800da675ee23e03e9d88ed0da1a4e4780a9e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case14_golf_ball.png` | 639,451 | `7f095788a8207c257d1e4b48e8faef16916642bf6d86af410e5d49d945a2f38a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case15_golf_ball.png` | 647,125 | `d28bfcca0234f1a54be752e2071c68c9720d00f81c4e72f656e5355a918c7d79` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case16_golf_ball.png` | 440,282 | `e6aa74d782aaaef1e564af8b60cff444d7b1d36e0fae2bfb6d125febc3b33304` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case17_golf_ball.png` | 441,635 | `fafe79beac04c75d6ba6985b1c12704b0a84dfe108a0e7ad8eabd32f896a5287` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case18_golf_ball.png` | 404,131 | `1f416a1d29d57f2fbb86d35b4a7eaaba0be63b289ae0e16401a4a318ea7d590e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case19_golf_ball.png` | 439,064 | `22010c79c37fc9a5aeaaea4bfa081d6adb4f41ff56445579b5618e5bb18db991` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case40_French_horn.png` | 446,687 | `fc1ed74b34157b1d83a8304ea187bcc7a1ebb8990ca9a8b3d99cec80731942cf` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case41_French_horn.png` | 479,675 | `7200f06719b651fcaa262c563f92adb293c96c6a208d564d2d0f1121e36ce6e7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case42_French_horn.png` | 454,947 | `c61fc275d0926a5ce5cc5067c6b280be6571801744df284a7658079f43c0eae2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case43_French_horn.png` | 427,338 | `4091744e5451691df87aaa7d72899442eb7ef9ea1cb4a00d719796b3ba59797f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case44_French_horn.png` | 361,059 | `37fd962f50fd9c7dd9229f94231ea0a996fd6e31f1781ecccd40b83ddcbd8c45` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case45_French_horn.png` | 394,587 | `cddf68c2b6627a12162841d47452030fd28505599cfdf83053439607c9acccc8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case46_French_horn.png` | 371,699 | `ac0dc417f48d25eee8ae716fa4776ef58be4f8edd55a16f589c47fb20bb0531b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case47_French_horn.png` | 350,415 | `e9336dab01628acf730aaa2bf6dbf2bd58ae8e7d7866de27b261091d31830616` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case48_French_horn.png` | 507,309 | `76a0e70d537d2dec0f9487ca1e637d7a99dcf779e5a16df4ff638376a9a54812` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case49_French_horn.png` | 575,603 | `60d6fe8a0b97f6bff64d6f3ebb65d129c0c46ef8d32f918c9bff1a1213a31138` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case50_French_horn.png` | 483,852 | `e19dbefd31a2208ad38776cfa77f2d914c6848104b845ab5e5e2af0a3fde7aec` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case51_French_horn.png` | 554,128 | `51a801a7b0ff9c249a614fca53eedd49d31828bbf0173184ed63b4edaf430e45` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case52_French_horn.png` | 377,061 | `ba242c9d4e1f6e5606c65269ce01d1bb61fe81cec6e907a289b3c94cf2bd027a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case53_French_horn.png` | 392,998 | `4d620a0f62ade7fc43d2ebee522df68d71563c09ceac6e30c4156d12b5dd08b5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case54_French_horn.png` | 376,205 | `aacc25f9b75e3f02036f1b5191050b40ed420b4e2f1751cbf84959f7644d0d40` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case55_French_horn.png` | 555,914 | `912aac6fda25db5a5285502e6aece579f50b5679a976b4166e14413bbf29ce16` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case56_French_horn.png` | 527,559 | `bf134b27cce61be7268e0c28b3f4942ce94861b03d80b4bad54b486eb46fc793` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case57_French_horn.png` | 339,436 | `a2c8227a8dcf99c15933ee24a118e9a206b82212c1ff1b50e2daf54926df47fd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case58_French_horn.png` | 327,966 | `3651ad46691dc1af8a5aa602f71010d7d0d089511321d69d46ebc10e0b37883b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_BLANK_case59_French_horn.png` | 535,382 | `84260a442344bbff8baa2400d719640a8e57a93bfa5bd9e843f05a0660a65628` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case00_golf_ball.png` | 484,374 | `8d777f29cc34c182cde52cc9eb6fd9fc010eb2638416bcf73830f4211f74cb16` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case01_golf_ball.png` | 530,538 | `cfdc60deb1bc8f6d659e3fc0c26cb49024a45fa9c22d1223ec87ae3c742d7ca9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case02_golf_ball.png` | 485,215 | `279e52b8c78d6f5ea9f8961db24b007322c1d76e549a0419d1829d44c015ec84` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case03_golf_ball.png` | 503,547 | `7fa9574401d2636b7f90c046e738e512d73fe730be62efcfa2fcdc86e447d03d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case04_golf_ball.png` | 453,619 | `1d62bf2aaa332a0eedc460905f95a79ea1d7c55980b72a78c4fd66785d269d72` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case05_golf_ball.png` | 414,845 | `b484eae590a64701f7a227dc1775ef0eb7a16bafcee9c4fb87e7e4eadcb09f29` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case06_golf_ball.png` | 489,391 | `9fe6e6762c654e0dc41fb85d9bc2bccf7b9dd10bc3fe6ec7054950318f51faed` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case07_golf_ball.png` | 622,071 | `b1aa64bcf3e107d4cfaf9ba1701da87f812c3d5a9bae3d0b331fd213c06553d2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case08_golf_ball.png` | 502,187 | `cebbb536ae70a4cfffd2c0587605f5bbca65b5481fe09658125457a4149fbfd3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case09_golf_ball.png` | 363,956 | `6f1369ac3dbd8d2c1c6cf3592c2c74b04dea5db131edd80a59e3a0c9915fa87a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case10_golf_ball.png` | 468,454 | `7e61f135399a957795caddbdd322787b6d384fb070a4edb46cd8fa5ccf929438` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case11_golf_ball.png` | 441,689 | `6c65b23b10152e16e5deaaa8d6bea7cb70c3f9fd5256d84ac34b52a9605c03dd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case12_golf_ball.png` | 510,599 | `7f7a65ceddf11e250a10921eccf6dc595f9266f395f18c167d10f55e997908fc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case13_golf_ball.png` | 480,961 | `5b9e93e3279f12dbb298dfc3f71544a8e53f117b1951a18fedc4436551e53a2b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case14_golf_ball.png` | 583,951 | `e8648648b951d93b9235f053af547501b8260b1ad940550978529f05f21cf914` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case15_golf_ball.png` | 552,887 | `c2cb0c6d04283f99ac123b8b14cc6ba6bf6e12bcdd4884e8dbe9c6cd4ac79c4c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case16_golf_ball.png` | 359,063 | `af58eba56fb38ecba0a2a08f7b328a3487d2b18aa08f1a2106436a70378fdff0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case17_golf_ball.png` | 443,841 | `f40ee7cb861b9ff07725b6921362875349de6dcf89ae059d391cf6477c92ea7b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case18_golf_ball.png` | 419,046 | `ca12c49c92043f2aaad86d922c6b05d98f1ecfe2aa65bfb8c12f5b7e81eecadd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case19_golf_ball.png` | 455,510 | `7250c6706dcee982982b76ad63c52f801cfce4ff37488c1793d40662c6321656` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case40_French_horn.png` | 422,088 | `f4b6efddee9ee53d8603f1f3e20be93e3d526571576d6097c3fbbc7e976ec4eb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case41_French_horn.png` | 466,759 | `9fc204880e7c960223767c1a9a4cb83cdd87a43ac887014819b15fda48c99466` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case42_French_horn.png` | 375,861 | `265e85186b82ef0df34214eeaad0ce00583a53fd82a4aa272b14b7966d21044c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case43_French_horn.png` | 486,860 | `d942884f5cfd1ffcbc7d081dee2006a1b5359e1a98baed421bd886765a91578d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case44_French_horn.png` | 390,088 | `b4a88b7ab65ce9c7fbdf4fc17be2dc08591abccba3fd61f7301ec8ca39f5eb96` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case45_French_horn.png` | 395,557 | `e20c7319233c60256d1d841631f0e24108b41505f017c4ece23ff624760a30ff` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case46_French_horn.png` | 394,341 | `b8e030aa443a4906e61a072b04ed0be98df4c89ddc9d5388bb72e29e2caadbb6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case47_French_horn.png` | 382,635 | `62141be7e901e6bedc1019006d52505acfcff349a48de7a6667a30e24180cfb4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case48_French_horn.png` | 520,130 | `7400ec40b854fa358aebedecccee5bb8384c470727b971d1ebb373abe285885f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case49_French_horn.png` | 528,393 | `c2622fd17e8827b3227b84d0325edf28e23cfaa6d5649ef460ea31e63ea4a12b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case50_French_horn.png` | 576,923 | `b71f898f8cfff2d758bc0ee8c4e0d07a2c55ec47529bfbe1037e1b5ca6724098` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case51_French_horn.png` | 572,695 | `1aca463c34952aab23e46c37d717803a662f61844b56f3d2b5efbd6402cef6d3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case52_French_horn.png` | 392,336 | `797fb396045229bebc0caa5326cca2036554c195a24a23f05df3d16560226110` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case53_French_horn.png` | 389,188 | `f02251304104318a0c95b8e05ff8ba05b4b5abd3acb5f6a608a64a494b962764` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case54_French_horn.png` | 412,233 | `543aaed9810614dbb31234b853a07f861e4a6b3d8e0d18ff37059886cf5e6157` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case55_French_horn.png` | 411,518 | `02d257cf651c98af22e4411f64a9c47f831cb37b3a5930575f36efe59851f152` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case56_French_horn.png` | 439,858 | `8d098130e5ee780ba5b0704d959ff075596ff9a1aef962c3d9bffb84846ce282` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case57_French_horn.png` | 342,172 | `3c411e6113abea3d5083f7177fc1d111a245c4d4807ec7f178a68b033e1d25ba` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case58_French_horn.png` | 319,966 | `5612a52a9fab7ee55a952540a939645b22bbc7ea286095a082e3dff7e45bb72a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_HISTORY_case59_French_horn.png` | 386,701 | `30e005f87f8b3e85c2f30d98411ca16503b9d27305f70133fe5d195463f97b71` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case00_golf_ball.png` | 545,131 | `5f7dc485b97225232b4fbe417ffc9944944542a0e703feba90d68f7789df41d5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case01_golf_ball.png` | 538,857 | `cca5a7c8a449ca7969083b0ce63fd232f5d0365c24509965fb6403a779e47a8d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case02_golf_ball.png` | 520,547 | `e33537f416eb74a6a2d4d37072daef6c5b498fd16f2fa7090d1849c614d5e95d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case03_golf_ball.png` | 509,066 | `877392e7d09c41d8a24ece3fa9c9919ab58e42b6592c97cc6710a081d4c948af` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case04_golf_ball.png` | 465,530 | `2e63c4defe3944b7c53c36acbf6c91e3f12f36b6e7add60eabf6095c6e50752c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case05_golf_ball.png` | 402,142 | `74b9bbd0570c8a89d0b83f1bf16db06beb6ec781f3bb0c32397fcdc2ca942d21` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case06_golf_ball.png` | 481,310 | `4b6351a1f6baf8eda878592c6a7b0caf6c1f05a12e0aed5dde454b533dabd772` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case07_golf_ball.png` | 503,557 | `31dcb572cb00ebc7ce3df45dbd6a0f93fdf54e6a1836bef9176b34bf8fa5a8e9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case08_golf_ball.png` | 429,372 | `19b7c8c97f3416dd280fe826c2d4f5d98f3661d15e984f23a850572bff67ed09` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case09_golf_ball.png` | 566,770 | `bbf5beabc0af9883fa43d3a54f4591b13b1ba548e79cd6707e15c65729c76e67` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case10_golf_ball.png` | 519,119 | `f41b8464e5c6fab8bce11f95badc9861656420ea7fae4f40d31dc1e5c14a60ca` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case11_golf_ball.png` | 480,652 | `b7575467cbe2963b3f82149d7a9597f4acb894967ab589b73764366ebe4c2697` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case12_golf_ball.png` | 490,569 | `995ad2d2ec22a93b0e55c4a571b99de68845f3cc010b3e4fa7eadbf94ad74463` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case13_golf_ball.png` | 552,073 | `3f614e803d239a86e587e25b8e34fc26ef7441adf610bb80672099c57e66011a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case14_golf_ball.png` | 485,261 | `26ef48d4c793229d3079f3822f09a71a72ffddb57181cc5eac21030479f811b7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case15_golf_ball.png` | 628,022 | `fe8b66403f93d23e5fb173ae99e23068b9bc605ed6e5b01a1778ebf034a37e23` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case16_golf_ball.png` | 368,233 | `4b350ce28f0ebbea46bbe2b904f6bd2746ad12c73876e6ec45130edf3d51fc6f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case17_golf_ball.png` | 450,443 | `da9f69e15d4603bbbb431cebb352b0c774b16f56fb8a317dda669cc9b13ad96b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case18_golf_ball.png` | 411,453 | `c09cb4bc8a0c6618740cb9e839f0ba01c4665e5cb838333f57bc04418ccaaca2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case19_golf_ball.png` | 393,831 | `75960f75fc960a95c078463f67616ce6566e7db762cfbcbbde13021e7f785dd7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case40_French_horn.png` | 440,166 | `c2f2d4d7104621c9be79ea6187e70dacabce7dfe985cce3b7b5682a07cb24a14` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case41_French_horn.png` | 416,421 | `2ddfc371e3dc3438ed9f3dd4cf91f55a7b0b3edbfcd2514532ae66dac1b60bc8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case42_French_horn.png` | 500,374 | `9e5d838c59ce17bf2de33abb375bfb0e0d371e58714c3edf9c0f4cf6544c6a48` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case43_French_horn.png` | 466,902 | `4f00c6a0b281ecb6e20d074156511f83f13fabf8c216a38cee659c55debe7e15` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case44_French_horn.png` | 402,341 | `112e41ea5757d171d0c748e13cc5de6dfd439739f675f03c7e7b30d20474fe71` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case45_French_horn.png` | 382,444 | `7290e4be7e97ad3f2c00b48b43c9bdc8865d7cef105cc1f3a235f175d9d41c8d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case46_French_horn.png` | 362,760 | `d03143abf7f34fe3d46d4bf658eed91f4e58cab2e2fc403ffa6d6a057b1b6115` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case47_French_horn.png` | 352,450 | `65d1b0d4854da9dad01695ed38f3e3a94d99e8f58697df14940587ecb1442024` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case48_French_horn.png` | 544,977 | `1da94e75f80904a7fbf93e2a0de2f048e2c9bca93e2bca1a0e48d07fd8952b8b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case49_French_horn.png` | 545,410 | `426b84e30539fb3f1af46a92a001e3efca8a734f3a3440f1763b9af5741a8c00` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case50_French_horn.png` | 577,164 | `e9f9c0c335729c5223d326097af297336b77371951838aadf7630f31314384f9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case51_French_horn.png` | 614,465 | `ddcecbeee9cbc02744a17706cd75648cb2fefc013a37582123863af889c4aa5e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case52_French_horn.png` | 394,004 | `636ce6f3eb7b45f3b53edddcbcc5ba85099ee1c466328eb947289f623a9d1008` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case53_French_horn.png` | 438,196 | `6e4d08d7ba2fc58b64f0302070fc72e5e668588d4e54b3611c54e8d409e93392` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case54_French_horn.png` | 411,246 | `5d0b571c15a4f149dc8a0cbcf0c87854ce12b9880701ca705fd45c088175c71f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case55_French_horn.png` | 385,778 | `055136298ab4d056c112035f7c8b833e776ddf1653648a21d9d2be3a13d82b25` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case56_French_horn.png` | 487,489 | `039aeeb750453b087b31c3b878efdfb37e043f972b8bc78256e945918bdfc6e5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case57_French_horn.png` | 441,318 | `c1c013737d3071c6541b8f6ff05e00cbd1fc82854863f83ee79ce6c860a898b8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case58_French_horn.png` | 362,383 | `eef54069cae05fdfd5c997db9c02a9cd1f57cf988d89c36fbb5a1a55edc14522` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S1_Stage1_case59_French_horn.png` | 497,705 | `d0212356a90dc8f614a09cfc6b79c9c4335b4d21cd171c0e915a1b53e9457b07` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case00_golf_ball.png` | 441,242 | `b738adf82a4091de2bed9c306407d5e2c594589d6e950b6a10f0d95b7880f689` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case01_golf_ball.png` | 420,569 | `14672f2429f27dbfa361e01e1a37f4131005e9bc6120ebf6b7803ed11f9df16d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case02_golf_ball.png` | 521,682 | `2fb88ee7d2baef58b7a00716fd5a4696abdc42e3d46f112fdce1cb0c662a71ce` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case03_golf_ball.png` | 516,891 | `ba93368cce7d174a2081586d21809828fa16c2a9e5307e4de72c2372084186ee` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case04_golf_ball.png` | 487,279 | `604761ea3aab04f1f8312a84bb85d0357637d75a0ac2e59a173d23777d237364` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case05_golf_ball.png` | 432,223 | `db870c82632fead230f590dd564560eed7164702603b5f0482f0bf753c588d60` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case06_golf_ball.png` | 476,652 | `ccf35e5dbd0a055fd2704b9170522577e406b9c94c39675cc0ba21c962ac516b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case07_golf_ball.png` | 524,457 | `75aaa6ed14770198439e645603b432915b66574e4411e72415f8a61b81df92a8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case08_golf_ball.png` | 515,549 | `9eb2e6e7d1a06d5889fd8a0409ee4ba57f452461c1abb18746431c81add4c6bc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case09_golf_ball.png` | 528,303 | `d3d53b9da5846ce0e1a0be441130e177e02e7b5253283b5a63c255a1c292f8de` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case10_golf_ball.png` | 455,487 | `b85dad8543445ebf1853ed52d4675b097d6aa116178dfcbb4257b7be83396f2c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case11_golf_ball.png` | 447,841 | `1603c77c96a5a31c78c9842cbaa91f647c6bb20359bfb6d541d6f8768bf84d89` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case12_golf_ball.png` | 443,367 | `d75a4501a1e05063c5cade92e9858e8a9814c6438477ae1dcf4d6a8cb214bf71` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case13_golf_ball.png` | 666,540 | `1fd149481552947c635a00cdb8b012e04777a6df4b097829acc3d5ceaf1073c2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case14_golf_ball.png` | 420,742 | `fda8ce3d9f38afca690658ae36ff688067983f3aa40edb3b5f9c7e1dae40ddbb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case15_golf_ball.png` | 506,960 | `82f8c0d2348a67f2ae26b63af09a1b390911292674ea95ef9ed782a6aca6ca74` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case16_golf_ball.png` | 506,312 | `80963d688e2450a6de744cc05298c697be929213dd5a26da450547719afc9331` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case17_golf_ball.png` | 455,197 | `097b9e797f7f0ea21689347314d97da68468a8862f7fdf1c7d1cd867b7187239` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case18_golf_ball.png` | 462,191 | `ce684042fe6d698c17b75ea68f2d09e00a15eb9a2f8745cd568a6b25c302a89d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case19_golf_ball.png` | 402,296 | `367cc8c4709470096bfd63c4ec845e72cece3833b5312784d9faa2377d758ebf` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case60_English_springer.png` | 377,435 | `a1206c61a4303b9aa64f5498d9fd9cf1b0a0420b8f2c46c318cf0cda4aae4a78` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case61_English_springer.png` | 407,634 | `2913e567323c1325562127ee91e9a9569fc834726df6f4ba7475240a12c80b38` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case62_English_springer.png` | 444,297 | `02b9e900ed5ce90451dd91b326453dadee54266f7676919c564e28fa5e816b8e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case63_English_springer.png` | 418,267 | `259c2c51ab2e39a32cc93084160c419bd1f4a8b274ac0b3a6ddb541fe865572e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case64_English_springer.png` | 435,938 | `2315a7ad3d03cb4d47df33de2c3a7fa9fcfb7f269eb2e9d874fb7b41a3c3d049` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case65_English_springer.png` | 571,002 | `d1314119274b9ccb36b7914a22f9e28476333e63a0f7ecd3aef702cb2c237e2f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case66_English_springer.png` | 463,351 | `547b4803d3fe8c937ddb2dd9964492aaffab24936a8531f979b7cf3bcc578c4e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case67_English_springer.png` | 499,212 | `e473e7dc6ea047f333ebf911510437cc01bef10bbd12122c2d18e7195dfcc945` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case68_English_springer.png` | 403,012 | `bdfb0b838544e12d4277040319aa7753358c7d2574515390ff2999e363beddf5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case69_English_springer.png` | 440,392 | `13a1a1e3610afe3dd1eed8f09f9c387d0a1d4879cb07d9c51443858166d8f419` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case70_English_springer.png` | 444,091 | `cd83dcb0f1dabf36dd8a6710877e970e89381e5285961ee8e637b1ff84926cae` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case71_English_springer.png` | 467,614 | `85331ff57aad9e164f5c0f250ff14d9e411dce83b6fe1e10459507986e3c3bb0` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case72_English_springer.png` | 537,595 | `3e16694adcd8e962a4c5c88786dbb828704de250eddb1c85544a4d51e67e8852` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case73_English_springer.png` | 468,206 | `0be7538b1881f5544f51dbe4b6cbe872555f85180e40208eaf4610f523a6ea51` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case74_English_springer.png` | 459,082 | `daa417a2ef0dbf189069221216069823a382f67b0716dd4eaaad13ef1da1b34d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case75_English_springer.png` | 471,312 | `47256b26c2e3f22098d228e9aa1438d8359315ed3f4e4db1256a89ff0b547b12` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case76_English_springer.png` | 415,184 | `77450516e19dd90448d4bd4b2c7e378c196180460ed94d37379d8123e08001ce` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case77_English_springer.png` | 362,474 | `c6847c01edbad51c1d849bad143d610009b88c0bbc82c6b1b84de499636728ec` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case78_English_springer.png` | 414,553 | `a600962bfdf40412e90295c0dfceee59801a802646486bfca31be049cc89d3cc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_BLANK_case79_English_springer.png` | 433,366 | `03dd75ca88735a18d4332a396d36f8edba53bfb73bb2e3d09ea65660da5cae6e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case00_golf_ball.png` | 442,322 | `d3052ed9e1682679bff030e62b7a7183036010d714a7b1860848865c75a1775b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case01_golf_ball.png` | 424,198 | `afbf3b05614a6af0cc2cfa7767bf35c360e6294c090165cc76c93f6d4ada9dac` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case02_golf_ball.png` | 371,984 | `61c5b5c1acfa3b334ffa3b36d5cf14e5d2c2be7eff65b99111acfca302006aa5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case03_golf_ball.png` | 497,870 | `30ba9cbf4388165b47b0ef2c14f1bb64d3736a57cec35a6f26711c8a63a0146d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case04_golf_ball.png` | 433,467 | `9733c417d4656ded2ec4929e5db22dbb1f0ade791d8ece912a4fd81934769002` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case05_golf_ball.png` | 583,517 | `a2a649b5d69b5624f841254bfcf35178759cf554f2a0ad88b8fe83b60e034a8e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case06_golf_ball.png` | 480,813 | `bdb40ff791803800c7a230db103dad75fc4d530a090a1c0b32eb8deba460fd78` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case07_golf_ball.png` | 565,394 | `eea93fd0dbfd180bad1c7ddb06252a9fde62dc36ab10d646cdf6d31620e48c4b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case08_golf_ball.png` | 505,371 | `da2b827528d071f2c1d2185a2d1f76d3d93d51dcf73173900ab04c7b792916cc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case09_golf_ball.png` | 433,965 | `a6e9db6fb256f56b9d7a6569584abbbd4495b35f9927f23a8a4538dc5f28841d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case10_golf_ball.png` | 526,493 | `34f79cc8ad095be6e17e0f91db532f6b3e7de6e8a0eb345e0d11a330626ad780` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case11_golf_ball.png` | 492,976 | `77d55a5c6f53a98494b050586e0855fcb06ef4a91b6039b045c9768845d9f39d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case12_golf_ball.png` | 475,975 | `71822826390537b32a93934472cfc51b4e2fd042f1727ed6a012aa465018bcfe` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case13_golf_ball.png` | 405,559 | `9c651e17549c1e6bbd250e9b852965368ae46e59c54faba52bfdee01807e7e35` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case14_golf_ball.png` | 438,999 | `0dc67150e52670701bd2ccb0a656d5c661f85b62182172ea1aeee33bed745f53` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case15_golf_ball.png` | 626,123 | `f384940dde4469feda731e0923040b79584946c8302050ea6306e514ed520c85` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case16_golf_ball.png` | 421,400 | `c5e250f1ea143bae36605e1fade94f752b27df4aa722779e8961e3952b0ff49d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case17_golf_ball.png` | 383,028 | `8d865955eae8a12438fe579833e2afe784ce3b705f18f615c9f017d1d15136db` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case18_golf_ball.png` | 401,571 | `e70dd6d6b289897afcaad69a1c0dfeceefd9b9f796848b8a2b7b4690551686b8` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case19_golf_ball.png` | 426,893 | `69c8e59eb682dad04ad29e986330125f886e022b17b4e9db2b91c301075604e7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case60_English_springer.png` | 357,132 | `f94be7079981ef718d5cf49e24502fd173104b541b25c2690134f64dac701c3d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case61_English_springer.png` | 368,257 | `c8b83f05b4e893e23630babd25ed3f6f87bd183be314d3278e89ac9e8d9c9148` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case62_English_springer.png` | 428,639 | `c6b836838bca87d7e1017a0d862d9ab4798dea115a8e489c199d7ccc67c29368` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case63_English_springer.png` | 484,716 | `58aa87ff75727291e2e9c0b43e434e64706af2cdaee8a19cb3ad211203351afe` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case64_English_springer.png` | 402,381 | `798d50f0fc902d7ab99553065493eb5d75ea6a9b677096a00ff53038eb452039` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case65_English_springer.png` | 398,274 | `d0220e25e1d7acb5ad904969f8ce01ec34cfb78c497e718f990f4ba0d165f95d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case66_English_springer.png` | 474,115 | `385e66bc5590785b22cefb2a6e4689f61be21ac1c1e647c6ccd82f3b1d6fb4b1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case67_English_springer.png` | 397,974 | `ef1951c9cd245aeb7e60e67c0a2d8d2bc5cbac746f9cd2c37945b2a901182a21` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case68_English_springer.png` | 358,040 | `43aceaa11ee5c13a7164be2bfa7f6a3d41833e1e39260e5bcedb520e49b06cff` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case69_English_springer.png` | 347,859 | `1d676fefb27cde068f2cab16284d6797ff1054ec3a28dee1a63c95d314183265` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case70_English_springer.png` | 369,876 | `6db5a5b172531b32a4f83f99ff71763d0dd3f32c2bf94775daac9ba6283091d5` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case71_English_springer.png` | 440,023 | `9c68454170d76870f2016e5d152c557b41aac51ad997a13a282838a90f7db961` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case72_English_springer.png` | 443,798 | `f5315827db0507370355199cb7b0569d854a8a3e950ed7a6a714a33ac114ef2c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case73_English_springer.png` | 409,658 | `962e452aa4e21927f26622149bc61de24ef9885c5f7b38970c24685cbd987ee9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case74_English_springer.png` | 380,372 | `1b86caf394e926058eb71181052967a7856dccddb0492de8bce44b0bf86d21b1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case75_English_springer.png` | 541,948 | `559e284e7df738806860cd660506415cd7e921ee1901469306dc178ba61d8194` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case76_English_springer.png` | 397,540 | `268c0cd77c33398ae9c49e3b8b13c70429c5cedaf905b7c96d591c86259d1979` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case77_English_springer.png` | 357,031 | `28f26d6fb5e9eb7cc4663db24840eb1a9c0b6e19c57a490b897e76d01bb43971` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case78_English_springer.png` | 296,319 | `f9368e11c4886de85115052eed559f7bf1768243f10852aa2e593ea70530a15a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_HISTORY_case79_English_springer.png` | 328,584 | `028b692072d0199ce1fc999bbf1547e15673e3b421df7103a76d8642a6eb84c6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case00_golf_ball.png` | 529,218 | `329506430f754818a4ceba8d488e99db6fcf0958a098e17faf5f2d3af8d3d62c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case01_golf_ball.png` | 385,516 | `4f8714cf27da84a47349fa2b554e7c6579b1801160905ce7772558b5cac4f61f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case02_golf_ball.png` | 438,611 | `2868a9c253202574fa37f5fc78e7d44487b0b63c0e0cf3cdff4c676f8825019b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case03_golf_ball.png` | 474,711 | `4a4bd37cda241e7fc0c89ba559a256f9397592527cb689a89ead7b8e71c04c0c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case04_golf_ball.png` | 457,975 | `d81c212aeab04cec94b4452fcb7f7577f0a707085b00b79a4bf11c0d6fae2d6c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case05_golf_ball.png` | 503,098 | `05fedcf44198c617a84012ae67cb8ed48c7910377b9371e69d806c80ef157f8f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case06_golf_ball.png` | 519,544 | `4ad07868b9995a635665b5fb92837ad0ee026444803b22a296e0ac95eaec3123` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case07_golf_ball.png` | 434,320 | `13632ebc49087135c37fe1fe0c6dcefb34ec7fe14a37e244fe38cb1267603b0e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case08_golf_ball.png` | 442,768 | `a20dbe89ed7db614002ee2090b186549da7a783bdc29bc2f22d60292eeedfb7c` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case09_golf_ball.png` | 490,183 | `44bd09b8036066564ace7296714a523a833e990096f2d2b3a68e606b69965d39` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case10_golf_ball.png` | 528,799 | `20e05eb09dd6812169fe2c45b3baa4c14a0e6e55b3c57d0c3c9a0c52de8395b1` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case11_golf_ball.png` | 491,339 | `701b14bdfdd36a7fba607dcaf1d620412fa56cacc1802849f2d239fb87569289` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case12_golf_ball.png` | 629,741 | `1c821d654d4152b6e87de9f3c9522ad0eba1b77dfc93c9c248cd7edc68a2ac06` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case13_golf_ball.png` | 464,491 | `30136381325c68a229b0cf79a5ecb73913aca7fc244cca84f03a04966802d2b3` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case14_golf_ball.png` | 582,397 | `2b1da197f7f38cb7ab37fa545ea657110e2cfc646a0ee932f21be90938719294` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case15_golf_ball.png` | 482,547 | `8414b409e72a6877688bfb4efedda012c9a61757808055e5d3711891af4857eb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case16_golf_ball.png` | 437,126 | `f59c69972f72eb17a16502b746bd739718ee03cfd54332db2f0aa1f1c64e74cc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case17_golf_ball.png` | 484,342 | `f5aa7a16d34f814a4d33de9207e79465ca765fb16db775ce509e0d8a41522d08` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case18_golf_ball.png` | 416,845 | `edd7fe7b9bb31254189a641201ae4be263b48f1a7f0fdc1c00f58a5f29e722ff` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case19_golf_ball.png` | 468,970 | `455731044a045161c605e862eb46ca93402330f5e290d6da531a1ba72a6f3622` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case60_English_springer.png` | 353,359 | `2a8e80ab9bdd94cca6e1e9def8e366683a4a401c9f7930578f2eedf95a03a2cb` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case61_English_springer.png` | 395,174 | `82ecb5f0f33e4c1308017014252d2d535b213ca85793d066465433edca14b713` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case62_English_springer.png` | 350,983 | `cf2b2412de86977a247073c9fa6632de8b8446354cf0c17d3f663ab5d38e8b02` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case63_English_springer.png` | 396,935 | `75e44ffd74ebcd72f79979be7ad395d6025b2cf528731380f89cbd67d2ad6080` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case64_English_springer.png` | 529,733 | `97dc011afd9b89c3dd06c38993037627eda4fb099fe6dfae1e1b5d5cfe48463b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case65_English_springer.png` | 519,965 | `35b1afc881acf604cba050fdaf6e40db8d5e866f72edd152f1432384ac051d36` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case66_English_springer.png` | 491,118 | `aad5fabaf3adf3ceb8ae222be450742ac1606fe229466e66e231108737d5c74e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case67_English_springer.png` | 445,983 | `f67a5cd79b7a5a2afac3100eb8dbb0a204b4551ab478e3e111edbe4b07bd1e5e` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case68_English_springer.png` | 444,669 | `ce0f4638c40ffdc163c7bc77270483c24a62d9516f132edce9ec4af36ba94eb6` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case69_English_springer.png` | 347,714 | `2956c83bce220976adf62d2874fdfeff1d6b7f262d56ca9f16fcfe55e6635a74` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case70_English_springer.png` | 357,839 | `6e5b21f221d0df04f325a6557e2ebd295e8852a06e3f61413dbc2e958746f808` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case71_English_springer.png` | 358,657 | `8f4508a6ac4c06eda805343f8118680f925030ff6b0ac0b7fa2037b0cbec561f` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case72_English_springer.png` | 442,980 | `a51456502437a4f0090741226685e883fea826d845f4f43a639e4184596b1852` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case73_English_springer.png` | 467,761 | `e74d5e11235c3e2b203ed4f7bd080457196ff52aac6d2639bf50b4fb184d5f8a` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case74_English_springer.png` | 441,320 | `314b9be9d8f0b4be5e718a93567c64b9db063432bedb135c4f168a2e58411864` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case75_English_springer.png` | 416,534 | `ad3600e467d5694c63895e98f986017ccec82eec6002fe8ec2a94b9ecdd1e7f7` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case76_English_springer.png` | 365,716 | `51579b6c29ccc50cf3d6730d3bf1cb4168160df1891c316e8244ec82c7b06b41` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case77_English_springer.png` | 361,251 | `217c46a99ce2d4aa2cf31e75f4535755c77ee69db1096ce453212fd7f4d16286` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case78_English_springer.png` | 361,725 | `61569e5af978c2ad4313a05fbe5af4df6a7529c049540434e8dc34c914090427` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res/S2_Stage1_case79_English_springer.png` | 377,073 | `1627bb10de8f6effec497929a264731a5fab30fb9bd5ac43f5cc530287ec8f64` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/EnglishSpringer_parent_horn.png` | 988,010 | `e4534951b4315f9b3a38b472aa948fa2cd2e4fcf3e43b8a4ea4f8845d318f0c4` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/EnglishSpringer_parent_parachute.png` | 842,704 | `4c4f4cd33d3cbf3b5b9705d487f8cf78e4c958eacd1f3c5a592596800624ab31` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/GolfBall_parent_parachute.png` | 845,815 | `421fdec1657b6e4becec8ce9451e38920aa1c8f1f1525431a30bf5d15ac3bbfd` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/GolfBall_parent_springer.png` | 992,738 | `300762c1c0ce4f949c37bb365cc479bd38be4fcd5dccf1930c75e68470086354` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/M0_parachute.png` | 863,339 | `ec24a5b518f138a7d8f5fcbb612cba7a1a73b960eb748f35fcae1e130511f33b` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S1_BLANK_parachute.png` | 909,666 | `31f481967b761602c0a97ee65e46c7d37b560852e7fd9f1284e91f0b909b8bcc` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S1_BLANK_springer.png` | 1,006,065 | `e4a28d346cae54feb628a132469ed89af1b48fdd16caed256efa2a0d111e7c54` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S1_HISTORY_parachute.png` | 917,940 | `362edbff1ed69c7a6d373d009d4b7906c6cefeba913a8d31b1dd2aa4d83dc7b9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S1_HISTORY_springer.png` | 1,033,289 | `0d02353024217be381bd2d8b43edeaaafe59c88d9f08b98d10a51a4a597d9646` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S1_Stage1_parachute.png` | 919,543 | `4a326c7e444ddc2e2e8857d9f65d908bff030ed8f2cbdfdc6833108b8c9f9fd2` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S1_Stage1_springer.png` | 1,034,865 | `1c43c24d85574c935781721ffaa0685fd80eaf881955bf44044d5fb643eacb73` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S2_BLANK_horn.png` | 1,007,321 | `2af90163124575987d776070af8f446a3123b166937d7b7f8308405aa005d930` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S2_BLANK_parachute.png` | 919,871 | `5e55bb6d69d8ca9588cf6001746461aaea7d6f5f3da21653533cda33aac70f98` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S2_HISTORY_horn.png` | 950,543 | `4cedb86b6c9ed887f39fdf67293588ea9a2b372ba73a8f97fb8c99eef00e6cb9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S2_HISTORY_parachute.png` | 921,525 | `4383ee92cee75405bcf4e31881eb9a793bea4a1ca0e105d4fd50156362f43786` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S2_Stage1_horn.png` | 1,001,875 | `49391f052e804d90a0545b86c1e8cf4636a3ec013ede0c2da31fbc69e5a0bbf9` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids/S2_Stage1_parachute.png` | 913,502 | `27d284f087813917176fb93f5aee2da4b79bbdee5ffe89dee5ef52e4e13ce71d` |
| `results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/disputed_cases/S2_Stage1_case8_classifier_false_positive.png` | 442,768 | `a20dbe89ed7db614002ee2090b186549da7a783bdc29bc2f22d60292eeedfb7c` |

## Evidence — stage1_history_intervention (manifest, provenance, verification, joint outcomes)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/FROZEN_MANIFEST.json` | 3,702 | `b883e7ee3056be983850715bde63c54146a711ba20bf2b5cf822d1b88f1adaa2` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/S1_STAGE1_HISTORY.stage1_history_log.json` | 1,636 | `c7f5ebd86a581bbc2fde6873b97014505ed341069e8101977015523e5693eec1` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/CASE_LEVEL_COMPLETION_RECORD.json` | 50,086 | `56d44ac8f729c10d63caf141c1e0c5a9ad17bd2c3845a15c4a16d8f321bb1973` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/VISUAL_REVIEW_INITIAL.json` | 154 | `38bded8fc0a51b2fe6fd6b36afbfae6588abcaecf3e68df83d821c06a15e22b0` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/CLASSIFIER_PRECONDITION_GATE.json` | 609 | `f57ac0d7cd7c3767bdee9dbd745219295c4f9fed0cd0ffc14fd400b93de58ecd` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/image_report.json` | 38,877 | `7a2152d1c28e5b582f715721dffe3a79a156f706154127f322de415eb86e9241` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/classifications.jsonl` | 42,078 | `385af64b70977ecc8a29db35ce6b193fe4ff1d77cb7e3cb7b9c0236ef5f939ee` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/classify_report.json` | 238 | `59d959dd17fbf147a586c53b0e1d8501acc014b80b396b33f4012daee42ac1b8` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/JOINT_OUTCOMES_COMPUTED.txt` | 847 | `1f4a90b8f9a91c5ccd4d859efa1e31b599437aba12c031dca5d22b18cbaf9049` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/code/make_stage1_history_checkpoint.py` | 6,822 | `99fb79ec3b1760b672aa22af00050074ccf42b50a825f5070abc2692d7090710` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/evidence/code/build_review.py` | 4,494 | `144d497919819d63fb19607a4a9d3a18851750d315aeabc4b4aee89e9f7eea31` |

## Visual evidence — stage1_history_intervention, all 80 new original images and 2 retained-class grids

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/0_0.png` | 442,591 | `f1a3de0e267650176a89d051a3f06e0e34018d69ce82140a32321cceea137449` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/10_0.png` | 484,652 | `ca1e81130ce81756085a45a924a296687b7a4c336390ef880a8d1073c0aaea2a` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/11_0.png` | 513,772 | `384ebbf09f0807956147b002bc04cba763600521f962c72692d9a4c2ad5efb57` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/12_0.png` | 601,194 | `d8558c40c1bc3019d17bef13f163305709ab59f03b96c2f8c5d1458d3b43cd83` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/13_0.png` | 359,496 | `da336c2ea19ac465d9a57bce49d90d5e811139bf6e4eb0fe876d8addc3ea0ea1` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/14_0.png` | 691,483 | `fb0f27544c0d9604b85842fc8df99b41869df51164bfbc0d13523c74f93df98b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/15_0.png` | 413,902 | `7e3d3991789882114a2f13e84168895306861874009d7ab7be17863a1216018e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/16_0.png` | 414,632 | `42597adcf5d376ef9b4bb4887ee4622d513bef8465832fa48168e15e9b9333f5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/17_0.png` | 411,209 | `d067f61ffec00cc4e64ab7736c48949245f1a85e20b9956570b3beeaf024c71a` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/18_0.png` | 411,478 | `82f5841a6c2df530b29be1e7ef1d3804b80cd027cafdb340db9df7b6696eb9d1` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/19_0.png` | 381,112 | `d176144b1385861b6df14d8cf54809701b182d72ab1c36025c9cbb5c835c2ed5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/1_0.png` | 493,705 | `d3d8a833b50da5b20109745e448729303c0df166d3a76a8fa7bccca63c2eea3e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/20_0.png` | 354,738 | `8f9f01cc2b954c117d6421ba22a65f4b50c1e825bbf8dabd87b60450e98eac92` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/21_0.png` | 407,216 | `d092256743318e610a793d2c66f703f498a38454062fd0f01841d80f9e420792` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/22_0.png` | 435,232 | `76b3acd9ef19f8fe111ef98a7119c8b40130193cbc324b632f478c547f05c6f7` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/23_0.png` | 441,165 | `6ab6fd46442b0d82230926b608b7fa66ddd6f65f1e56924a6ec7961cc420c6aa` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/24_0.png` | 263,108 | `e57a0abff4426b7254b1deabd017c52fd4f12cd1d5c56ec335fa336ca19acd0b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/25_0.png` | 302,277 | `80366bdf498a9c52bbcdd78c906d9ba98376f186661ce7043b6ee5ce2517b5f8` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/26_0.png` | 261,746 | `bf9cf0b8e9e2ec6e1c00d5bfe7c569ce59465990c2804faa40909adbdeaf9b34` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/27_0.png` | 310,590 | `ede91d57cda767b51f6710b24218819754601a746fa3108957b3ec19f5b7e42d` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/28_0.png` | 483,247 | `2b90c8dad876eff931770e738d1079bc92eb10763bcfef5d8ce0ca8fc09a681a` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/29_0.png` | 469,483 | `744934cae02de22681594caec005dda3061181b5db5f1867485cbfd55783d328` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/2_0.png` | 427,981 | `486f995fac27e7658e1f379d380ddd031785543967991ba9441c6de47107b282` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/30_0.png` | 448,067 | `7e1a295f9e2ee399335387e6db5aaed5d823be3920dea63480840e6647677ce2` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/31_0.png` | 475,300 | `e0620af95f989b94fa6f0c8b538709b4367b925734f76f5a86cecf84a8fda482` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/32_0.png` | 353,495 | `5bf63051cbb5cfcb1da2ae1c6ebb138e914bc8d3c83582f4dca21374556a03b4` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/33_0.png` | 364,756 | `14f9630ad6be72d19fde63b0935cc646f924d4c2a26b7b2b9e2c00a72f7b45a6` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/34_0.png` | 313,183 | `16edc9c966eedeaca77b967a198a9d922bcca49593c864da25f7aa1cba5acfdd` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/35_0.png` | 339,067 | `f4beec36453064684ed7f8428d8121c65b1b76f24689a353077fcd0a2ef27c37` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/36_0.png` | 397,414 | `eb07ee59d34a46cc4598b74471ddb3564b262b7640c4540e524bdff4ac15479b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/37_0.png` | 459,298 | `f5e4e99ceaa98c44b0d9fe4b22dca5444d5d275011d39a49c777b757e0f6c718` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/38_0.png` | 414,518 | `9c5a40ba82e518d66b3a18635edff297637d2c4c5bf797e1dd95cd66547881c8` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/39_0.png` | 418,645 | `58fbd9ddd71aa1f5467f2f3bf5ae048f63c08c0f194425c436131f71b080b76f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/3_0.png` | 514,489 | `5c9057a0f021de6b356a05e165d44064702fba7211312603db18e51348760c3d` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/40_0.png` | 420,643 | `bcf1742b36ec5759ac6bcb04a370380ab7b9797dd9bee379aa73a36c4a3c0811` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/41_0.png` | 427,134 | `b9048a5e47cbfad169065586a3a6656b90267cc1d3a4d2cf74d2ca5df159bfec` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/42_0.png` | 405,860 | `6043625650753fb783c6f6d4f62b5449cbff9691a436f71268475eb0c3da36a4` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/43_0.png` | 426,843 | `e1e5a473059fb41dc6904dd8ad9f9181f4843fc797d57cdaa1a37499c2c32c93` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/44_0.png` | 366,734 | `003143a2b4b0f6c1ecb95feb4ca4019bfe641eb50915f31fd60b13d1846dc712` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/45_0.png` | 425,859 | `0ea78129e99282b2f375a062754d8ca6c18709e4d4a50495ef2e7f5da49fcf9b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/46_0.png` | 366,313 | `d25b7137e93e80c7f77de559bbaa0c8b65e97c2448cafba20b784f7d536474d7` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/47_0.png` | 414,769 | `a75d8b66062043f6f14e1424bef5c996137414e44692cf6bc3d751e1da37f7b9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/48_0.png` | 540,468 | `795f004f1bc573a9f5afafad89806b11edc0de58c4c9840fd0feadb18801cbc2` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/49_0.png` | 557,650 | `89e322b1df6804392cc656f6b34e53861a5720fa4ba6c23598c0f7cdc987429f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/4_0.png` | 443,880 | `d927bcff3b3ce029a1f4a4411eee2bbcda1d81dc70d2cff516c6db863574d1e2` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/50_0.png` | 544,826 | `6c08b006932c450c79bc0742f8910d221fc859188a04dcbd5a6d1d0130c10077` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/51_0.png` | 515,317 | `2001285580c553fb2d02f1613c78aa3470bf2e11d473b43bd279f980de74737b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/52_0.png` | 429,377 | `1cc2dbeee5fc4bf61e90dd145c1ceeeefae4ba0a7d5a9e413f9f4f72a95cbc3e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/53_0.png` | 393,425 | `56b11158987a193084319ebee06579cb44875a329ab24014f732c4b0dd09fdcc` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/54_0.png` | 447,101 | `a9ad4a6f4cfc26f9dc4c8158d30e6780f6f294fb9f4c19e8a66ec408a93677b9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/55_0.png` | 423,550 | `6413e54583bf9e6ea1c07736f27d3b5c098c46e40dc1b638988924db65e7be80` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/56_0.png` | 344,288 | `3f762f168cc87296020da6c3234873828f62d55d30d43daef00f319c570ddd8a` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/57_0.png` | 401,265 | `a98b1de645cd1b5d0fee6e5addf3811bd477454464967d5fdf8fc9f93976c7c9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/58_0.png` | 303,946 | `7563b50cbf7bb24c3b6f2f7fb419bb909260e6809f29f7d9d02980cd07807357` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/59_0.png` | 368,570 | `a07aac50e854b06970232d393603c823bdb616a22eb80ef0120a087eeb4757d9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/5_0.png` | 499,976 | `10fced63deaee9ea7873f62084f0f49192cec2b974fb87566913c9e21df56bed` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/60_0.png` | 372,373 | `0ab6ee908cf99d7a2613d69a874b2e15293ed2caf7e80322227a0d580b13a0b2` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/61_0.png` | 402,415 | `8b667c263207a63cfb14adad077007d5412126a659d63529e4d7892d4c64f5d5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/62_0.png` | 374,490 | `a3be13716f63a2a12fc56822689cc793019aadf9364569e2eaffa83acb4de3d0` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/63_0.png` | 394,475 | `9e3260d368b3adca158f4e5b8b1be18a944557d128add8d4f446e8762dfb5ccf` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/64_0.png` | 502,271 | `7a3aa7e447fcb618900878b417eecd420dca24b31c34347ec6702c1261752370` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/65_0.png` | 463,179 | `cd81cb9787888c34e06b22d1f5effe57260bbc9e59b7cfa6c1ae0ec298efa855` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/66_0.png` | 482,208 | `645aea8115fd90577764fb590b8840b46ea91d109dfb18095ec8c90e29761c6d` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/67_0.png` | 441,554 | `54aa0654c155f7bf0ffe3c81a20b06d42c7909f0405c654ef09c4b3178cf8f39` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/68_0.png` | 414,695 | `1773a438a9afaff3490bb20c4de5f9cdc5125bb144e7efdef59de6bae003ee12` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/69_0.png` | 395,331 | `86ec6fbc135dcfd1b4f9da352e629da245505bf42755b874ff4625063a315831` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/6_0.png` | 487,099 | `f87c9bc8bb1adf859d87a2ea952d049188ec365f1959a751bc155ea29e7d0e49` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/70_0.png` | 406,119 | `bdd268f7160927d9a2a8565676798b78eaee1be74ea0cece1baa4b2a4d7b2861` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/71_0.png` | 420,143 | `9f7a01bdf9a87c66914f34e385a7151a46bad59ecd0a4f48a9a1037ca4b51ffb` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/72_0.png` | 455,930 | `fe4e6e5fc6cc7885650eeadaca3ded27b839468a76f4bc8410b03e3a3154a7d4` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/73_0.png` | 420,636 | `41a56a0536051e83237cf3c846709224de00f8aed3263a67049a7ab5273c4b70` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/74_0.png` | 468,500 | `dd966a312f81393d796ff99dd3f916c9471ac0a2200fe476347e78b3af57433f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/75_0.png` | 414,973 | `805809ed254fe02e94cb18909954167e4f9320d22ef1809c9e901925340a65df` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/76_0.png` | 350,619 | `75d4b110fcd218c3f5a02984a7952a8041e3b425762e3d1e8629e6026994c17c` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/77_0.png` | 382,257 | `f1be5f188beed58a54af221768b26092ff829db3f5aef77e04c62e4156cd289f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/78_0.png` | 368,153 | `82d778222e5ebb7e0cd901fea0c272f9642b16edc437c3ee62015ade24ccb347` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/79_0.png` | 388,905 | `b4b2c4fe6447e987c3d2ec038b06554743caf3c6c70bf67ec3ddf32ffd9bec05` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/7_0.png` | 534,953 | `a96e000ee84441af7d97edca4a61c0db80775a821827db832a0c7c4f6a3739f8` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/8_0.png` | 497,139 | `ceaaf5ce631a4a9855b26bfbd4090548781e5c0c87831ed57b9052f49353d278` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images/9_0.png` | 437,833 | `39db36ee97a5f2f85c7ebd259bcee16c1fca48c1e8a117addec71ad44874aaf0` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/review_grids/parachute.png` | 828,189 | `50df466f59b30ce77c33eb8386ad1753499c8815c0983ed8e8541bbd79384c81` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/review_grids/springer.png` | 1,012,446 | `011829fd26ec13341a76bceb45dde4e5630a1caab1ef31a3a5bfd10447b77959` |

## Corrections bundled ahead of the S2 Stage-1 history intervention (C29): S1 cases 12-15 relabeling, false claim removal, M0 parachute fix

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/closeout/CORRECTIONS_BEFORE_S2.md` | 9,567 | `389c34d83a27b73326eebf1107248568f7093e23e62f6dfc7002b6be85e5a903` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention/closeout/CORRECTED_LABELS.json` | 4,690 | `4b6351fb15b76d1a37c876099c1d5ec6e26611e5774c1296c3be8935bcb790c5` |

## Reports — stage1_history_intervention_s2 (C30): does the S1 Stage-1 fix generalize to the reverse sequence?

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/FINAL_REPORT.md` | 16,471 | `adacc5261965ed133e25c32352cf76e150786ade21532d3c2dd8242c85b9ac17` |

## Evidence — stage1_history_intervention_s2 (manifest, provenance, verification, classifier invocation guard, joint outcomes)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/FROZEN_MANIFEST.json` | 4,444 | `26c73d8d2209c0adca067652b36a8d2e91b6ed8ab4182ab09553a58041ccd182` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/S2_STAGE1_HISTORY.stage1_history_log.json` | 1,911 | `3ca0add8090e2cf8f142f5284b5d12212c0acc21022f2b370c79f3a3189220fb` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/CASE_LEVEL_COMPLETION_RECORD.json` | 46,826 | `276bec761e6c0daed431f0ffca786f714cb05c74c20cae7534bfed3ff9424795` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/VISUAL_REVIEW_INITIAL.json` | 268 | `a0e92d84e245b4c767910bbba81d1bdc6571231f451ede01c25f092297974692` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/CLASSIFIER_PRECONDITION_GATE.json` | 615 | `19f823bbd5d9d53cc069665b9fa4b2736310a1b1780942ea7680dbdef175cfe7` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/CLASSIFIER_INVOCATION_GUARD_RESULT.json` | 287 | `2da904f4b7f1d5320fa936f3ca6550ef18a0a982fe702674265d27756c0a086c` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/image_report.json` | 34,016 | `8c557df19fa00e803968c7ce5060b5df0da360739357bd36c1d279cff85e3df8` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/classifications.jsonl` | 37,204 | `6f96c44f3ea93b031a8f3ebc685aba9d574fdc37f89f2c061a0b57d1fdff3c09` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/classify_report.json` | 239 | `bf9ce2bf532d0bf1509f2fee9769ac70effcdac9aef024ba276cc77b7d8d2215` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/code/make_stage1_history_checkpoint_s2.py` | 7,344 | `44f6e3df0a6e3d028922e1d902f7896f52495929950e8ef3ca74f4c25cd032f2` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/code/build_review_s2.py` | 9,641 | `4630e8d6d23124345d80754230b0dad1730d2f488c0ca31da1fe4363dd0c4270` |
| `results/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/evidence/code/make_review_grids_s2.py` | 1,794 | `7de9920ca291b49f4b3d2c9e9aa6626bacd3b72ea54aea0ccfe6ab23a3967410` |

## Visual evidence — stage1_history_intervention_s2, all 80 new original images and 2 retained-class grids

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/0_0.png` | 427,788 | `d24ceadf104e34657656a7a96aee79fc2384e3325c06902f90c41516137524ca` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/10_0.png` | 466,017 | `581e2e443b72fb0b284d48c6900ad7237b275fcc097d8730f76554e80a995e45` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/11_0.png` | 515,996 | `65eeacbf90b7a38dcc0fe751a6878a80023f600c0a073273300fedda8599c791` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/12_0.png` | 527,929 | `aaa83628b3ef15567326f25b4aca020f0407ad73b6049f8570e55ad03d62aef1` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/13_0.png` | 347,068 | `e138f7113e25c77359d5466caa14faf6ecf2504f3cf6c13d07264364f16d3c81` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/14_0.png` | 625,632 | `e932347588c18ba23516aa5230d6c03b163c2f46a08f5712a94071df803ebae9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/15_0.png` | 364,790 | `36e4778cd320d1322b020f0b7231f38ddbe283325fd909421533bc74a1b23054` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/16_0.png` | 381,700 | `e2c2178a87011d8d162a9483c43054012e6af03587b7e5d569ef16ea02e9ccaa` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/17_0.png` | 380,269 | `78fb382f1a0ecf6889293c12124ba77ad22a3b3f82cfa4d9e8ba42e12dd87a3e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/18_0.png` | 410,223 | `b6c2431bf3722cbbe68dca1166539aa34292dde44bef25438d45180523257cd0` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/19_0.png` | 373,073 | `b5a970f74c7532fb2cd2d82fa7ba71b344ecf9503e05d5118e099a26ace339b6` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/1_0.png` | 537,479 | `1d44078183ecff895e7bdd8ed9d60e0fd47a52ee5ce828a7b66455c1ef16763f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/20_0.png` | 370,522 | `482fe51c6d3247fe605be2d419e9c91d32cd1972444168c9c751b83241ad6f99` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/21_0.png` | 425,142 | `befd161ed8fc8526a184493163a89783fbf3141a196cde9ef39baa4e7d0ab3f0` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/22_0.png` | 481,197 | `57aabf78a5dc6971ca69b6f615299378d01c2cc7d4ab3ed7737fa2ffdc5bae32` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/23_0.png` | 460,073 | `32b8699e31464d9280c8d381627fa3aacb1ade6443b5a4e0d1d21fb7e733897d` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/24_0.png` | 259,533 | `5fd4cd6985f134fd25f1e506c8302d367a20232989535c0a95c01435b3b35127` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/25_0.png` | 286,552 | `3fd729a68170fb716f3529035e4b86155d6c68efa5573a5ae64188d8048aebfc` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/26_0.png` | 264,490 | `9590bd4ec508e5951b6d79d365c17b379f8bac542b0ac3f0a146d1e9b51b9da5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/27_0.png` | 301,750 | `281d41722dafefffc94579d4436210e5d8dc941494ba12864deb2277b113d92b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/28_0.png` | 472,226 | `a513f3b0499ce38f9f3fdec9b8124c4a406c56ece0609916c79c93a6007a8357` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/29_0.png` | 463,849 | `f15ee484d04c96170dbbfbf1ddfba553bb1d0a02210d6eb805e8bb87fe47e311` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/2_0.png` | 413,521 | `0e2800a6238c41cb0d02664e946bd96b10869db85a8ef57a93cbe31b623409a8` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/30_0.png` | 429,750 | `e20c8b4bb18fa82acaff0b013bb189777807d46dcd7979845a7db00f16d21fd6` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/31_0.png` | 431,154 | `29ab74acd42175f53555bbfadd0cc76f8ac6e30c897f9fa46d09bbe0727e51b6` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/32_0.png` | 360,742 | `e974b5dba88dfba060559e7ccb519eaeb9a79a70fd05f3a0a942b4e26315df8c` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/33_0.png` | 342,723 | `4a746132777f540b241e0b7dd82663ff68cc1f3942da18370ff7b900c78b70d7` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/34_0.png` | 298,269 | `803807daa46f9d757b6bea1543ab40ea9cbd4d3d2d2e0f65168fb147f6c1501b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/35_0.png` | 352,783 | `ba3d9e917cdc1240cc8cbb8b9a7e12e29e512b537ba3cd1efa0f22b44076f7ad` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/36_0.png` | 393,415 | `5bfd9558da87793ec954865aea3ac199591ed3286bde9f63a57c2ea71bde080e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/37_0.png` | 410,340 | `c09de52abd12e7f952ec9a2b7ab6e556626bfefd21bf445d4bff776f81dfb8dc` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/38_0.png` | 425,764 | `645a5c45ac7c991b15b82498bb89339536e2e067984e5178dadb7a695ca04899` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/39_0.png` | 431,551 | `5074f3ff0e0c3d25f0bbdb78448de81a55c16f2e6d6e97c6c06fa561e04e58aa` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/3_0.png` | 507,129 | `60c524d28e66dec8cf2558abd4a724a584d45076438e9f0ec3ab0dd5b974198a` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/40_0.png` | 447,742 | `91f79d3c4284cd7200cb04941d22575a8eaa92a744314cf741e4dfa8b6614913` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/41_0.png` | 471,620 | `34b2b95027d7745748c7305a6698aae82c5b936e0f747565da1aff78ba74cc8f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/42_0.png` | 436,214 | `58732b01b132fe21450ec29467de610fc65a31bb8362c383e2729f89a27a2026` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/43_0.png` | 452,278 | `9f62f9cac2810d5f8b26b042ccf7cb2de2dbc57a36e399cbc8579a11e1f53851` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/44_0.png` | 466,626 | `129b9cd31e29d44f6055e5914b676dc6a712e17a8381ef839a113483a695c25d` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/45_0.png` | 455,558 | `84a4008988f35cf693a7ca1243ef37564230c8dd9f35b1717ec2013033d33418` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/46_0.png` | 412,459 | `42dd837ccca9db60a7ea6e5241284c065dfd49885b7ebc9b8e99f117a0dd798e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/47_0.png` | 465,089 | `e5f2413c5a509fead5e5bf85ac7866295626a2eacd639447470cfefcf4eb37f1` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/48_0.png` | 578,115 | `25248d5dd5b73867b982c1f5aaa175260ba8b31dd0b5d1e65d393f3af41d3f43` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/49_0.png` | 482,758 | `c3816b33bd1b7fc881452fad055562af83e2d51d19580e11fe02bd8b057020b5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/4_0.png` | 414,795 | `5b7ac29e9904c133f72ca2fa5b0c535c7c91c102bba3c06b61526aab2a109d6b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/50_0.png` | 583,151 | `58584b9f3edb071bed40dfcea20a86f98b1c74e6fd4dbc6a4d2b55d17899e921` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/51_0.png` | 475,538 | `fb0043edc3d6b63eb61d3827c2e56682091179be7d5868a4896177645e226460` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/52_0.png` | 448,050 | `de394d9801c27192c60d6d3cf37a588b4f323cadf3bd6667c16a3a804e031f76` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/53_0.png` | 419,386 | `9c88d57bff0d560aead09302e99305253b2a5a42d7526e62592d604f98cd8668` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/54_0.png` | 434,747 | `bfac878b46678229da25608c27a642a57c402ae21857308b181f2ef68ce7afb9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/55_0.png` | 446,415 | `b01721f3125240bd92ab8a8274c76e07de65bfa6030997a71c921cbce5bc85f9` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/56_0.png` | 289,051 | `a08918a45b1ecea3f8cce52d775d74c075b61b696b7249005b88dd68401b2388` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/57_0.png` | 290,385 | `8a9a89fca26e945a9c53c5546f1c636ec6bd2687fcd6199e6bba541fca4736cf` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/58_0.png` | 269,101 | `434cb64b791508d1fa2cae6e11c056af8e628bfeeb66341e953afc860a51b00e` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/59_0.png` | 311,062 | `af1fc5936f43c331f2e3f6dfe5302c418471b7955cfb902557ba02f8b99fcf22` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/5_0.png` | 468,724 | `f0ea547c6a6abe8327f645b9a41cb84b3a86de5332035b18bda594613918bb5f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/60_0.png` | 421,213 | `a105eb8eefa31b0ec041960aa460e72e6c9e4c0adc610adfd7a70d79d49bb9c3` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/61_0.png` | 405,849 | `9ab6dc04805a668391a6ea9ad2a89fc0d6c8af6bac2e0e9fa3b501fc0bd56784` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/62_0.png` | 436,077 | `53fd75416e76aa30351e8ab13e170340d18e548b2c824598e4f86e4f4c64ca73` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/63_0.png` | 389,355 | `e32cf8e722968ed58dba70d9b2fc293d6151f1e229933d7fba79b303fa95531b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/64_0.png` | 435,084 | `6a88b82ac1ba3b78c4f935c99d523ef55ead1330ca5ae8d38dff27a80aec8071` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/65_0.png` | 448,260 | `4871e99974ed661be4e4b2021f65a0d002b3f45021cd14eb0d5aa66e05274183` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/66_0.png` | 428,687 | `637553fdf1ccaad39f1d72768f0a155821a2ddb652337fa3071140e2055dc35c` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/67_0.png` | 424,678 | `4962c279e2f644ddf1abced920fe3c6b31b115a8583e8bbe711c828d2d5dc3ca` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/68_0.png` | 392,294 | `4d6a74b22032659e8f84486b9005b05be564a20a533c53c8227a373d423e1b32` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/69_0.png` | 354,847 | `e8797954c2bf027dc3e9c5bdaf6de91fb569c85c56e1b99003b326a03e070fcd` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/6_0.png` | 468,770 | `09078c9613f4e57b739490341ba29a6470b0238645f7ce25d6681f948f476c8d` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/70_0.png` | 350,206 | `784a20d2beed8a9d927baa12b2a8a9804842a8f4d629b983bb4e579a3a075843` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/71_0.png` | 359,614 | `042ea3c329ef4ec88eec1ee0159664560b7db2f8322d7721af48db809cb0f7df` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/72_0.png` | 427,139 | `c01752a4108cced8db328957ce94d3da9e088bb7ada4765c699a3e4a7eca6e6b` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/73_0.png` | 460,171 | `d56ced3b6cc2cd0fe9b727b701a952bd09f86dabbd1a46458f01873f3324cc51` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/74_0.png` | 455,149 | `9f6f8f3656f85fb859aad0ad6b9b48a959befd6d21226c26d66a77c79272b251` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/75_0.png` | 469,944 | `cc3acd549a25a9331d0abeb4d3f2aebba22d3556c87f70b4dcaf1f2e2d65e04f` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/76_0.png` | 425,869 | `c64c94446da62698bf8b7bd67d313b31e356d9e098aa40b3a4fa95f9e1a56ea6` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/77_0.png` | 351,041 | `58045da363074a233fc70d2dbf6470726930579d2b0c8f9ef2b35cc23a701e1c` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/78_0.png` | 414,187 | `0670fb1b70d420816a35a52d5d847f221c4f6b8444932e8e046ac30d8636e7bb` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/79_0.png` | 338,135 | `133821bf040664db44886f5b785ec07d2d3fa5c086126e1b85504d8b7f823cab` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/7_0.png` | 517,465 | `85dfefe191020bc40996b9b6f1c78ef19db6a184eb970220ee2d92bb2fbc03b2` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/8_0.png` | 471,967 | `ada5776e136991d04b00fd1132866b0a98401d153bacbcb028ef0924afe19f65` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images/9_0.png` | 516,611 | `be34b9cbb5d902479877143c224017e6980d3552d59ca9c67e29a011abf2cfb5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/review_grids/french_horn.png` | 1,004,939 | `193cc4ecb1b78691acdbbe6242d1b2426eedd68205f609153849cf119c737ff5` |
| `results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/review_grids/parachute.png` | 842,914 | `09a928aadee0951c8bc76cfe88334c1bb667fd967a42df640d85b8a34cf1c9ae` |

## Corrections bundled ahead of the S2 both-stages build (C31): objective reframing, springer case 63, retained cases 32/54, S1 arithmetic propagation, verification-claim honesty

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` | 9,762 | `702d8492690a573754b91b1c429eaa68b9261f364646f682e4b2a3968cb41618` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/closeout/CORRECTED_LABELS.json` | 3,756 | `d4c63874ae10d3b3032a111328dcb5ea18bd69013bdd4731a9289c9ca31b2266` |

## Reports — history_both_stages_s2 (C32): does the S2 Stage-1 quality gain survive full-pipeline refinement?

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/FINAL_REPORT.md` | 16,840 | `0421818808b420eab87a51d2b58c804e6250ae1cb4002deeebc890a8daaf36fb` |

## Evidence — history_both_stages_s2 (manifest, recovery result, build log, visual review, classifier guard invocation log, joint outcomes)

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/FROZEN_MANIFEST.json` | 6,180 | `b4ab79ee1b863f1dc7749270963d1fc9760352bbbe410d7b81d84896fd7d761b` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/RECOVER_S2_BOTH_STAGES_RESULT.json` | 699 | `d21eaccc1a441a7730d0616bf0649f134fc863b7f779d7aaeeb90dafff6bc359` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/S2_HISTORY_BOTH_STAGES.build_log.json` | 53,936 | `95a0863d49cd6fa3840ec45d6e444f50e28454961414091e7b9ce1c7c7b37ac7` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/CASE_LEVEL_COMPLETION_RECORD.json` | 46,280 | `b830d0b49efbf0a044b16c915bfaf527f6a4f109b7c51c94b2c28a7bf5255afa` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/VISUAL_REVIEW_INITIAL.json` | 404 | `5c6bbf8155cb6f3809e7f324646002bd1aec5b5b07e7c36cc797752359a22ec4` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/CLASSIFIER_PRECONDITION_GATE.json` | 1,058 | `6c6fb64152f7a15a773372d9912d256ca32b74293ec9b551da3e225a31548af7` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/CLASSIFIER_GUARD_INVOCATION_LOG.json` | 1,036 | `f35f5969100c044e7e4826fb30800e30eb6801a974c459770757fa545366c544` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/image_report.json` | 33,779 | `581309e0983812773e484541a40403a0f1a182d9ec7819ad2e5e04670b5ffba3` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/classifications.jsonl` | 36,901 | `28b679a96150853715f5a1e32d47163a3548901ec11581c19ea7e2673acd3a5a` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/classify_report.json` | 243 | `595f8721a0a3df2304a590b8f4b93cec4b1dadd388559bb1567fd0e9855c99c5` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/code/recover_newtarget_s2_both_stages.py` | 5,151 | `786b94f68c19578d71e1c4223b8096b8d9bb384d45600ed08194695a40aba75c` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/code/build_both_stages_s2.py` | 12,137 | `837cc774158e3f0f5c6e4979507b25e92568997c58360c3412c647aec32c8554` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/code/build_review_both_stages.py` | 13,871 | `911bb47121b29e99584ae00fa71a22964c151ac7865715f23bf3ca0afb8c8896` |
| `results/rece_seq_golfball_englishspringer/history_both_stages_s2/evidence/code/guarded_classify.py` | 7,888 | `ff0b4a563f759c28b83b9008fdc5dc86effacb5e68a3227d386860e5036c5452` |

## Visual evidence — history_both_stages_s2, all 80 new original images

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/0_0.png` | 433,550 | `e08971947fc4fff916b3d02e4a4878c96caa81d2d61343257b502c2d88432609` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/10_0.png` | 498,825 | `ec5197339e608a0bc4cb368eceb5d70c2f382ff29f8a90ba7a80317251733f48` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/11_0.png` | 477,507 | `24328e4bee8c6ef3fac7dcc12c0141109885cf6cd2a602c3f21ee64ead8a66ce` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/12_0.png` | 617,155 | `b80978425244efcb40d9c587db73324e352ae61f41e15199d1263108e6e03115` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/13_0.png` | 409,528 | `cd71cec86c7973dcecd43f7e6f58d431fb54e961b4bbb3da3f408b6794f0753c` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/14_0.png` | 655,997 | `a598425f204ef0b25f37ce7354c03938aeb53596df32b5dfafeda73d1baa465c` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/15_0.png` | 390,328 | `50e35b08388f36b864746734500d74d999a5183ab8a0fe57d265d9bebf70b49b` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/16_0.png` | 397,675 | `76f32617bdbfea15c801f2f8d4ed339bde7c314fee4f8aefadb83499524d5056` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/17_0.png` | 413,772 | `94b35051102c38a7e5d269daa43cbaa6140fad0d38c82ef007177e4aca3bfde5` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/18_0.png` | 408,678 | `20f017ad3bce78d9fd1384834a983ef17b97704cfaeb635dd98c8a2ae95f06df` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/19_0.png` | 395,350 | `4ce2c48e15199bdb44985514603aa07f6e47083c1eecd998dabfd9947e0667a4` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/1_0.png` | 511,204 | `0c2498884fb8811dd93c8c16fa5e49fca4fd477f844c682dcf846b391efc3dca` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/20_0.png` | 370,947 | `51396329af6837488ea52374f4130c828ce61c9b041b12000ea891c1543ed281` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/21_0.png` | 414,410 | `29dab5ce84c18029b5c4192f7ac71971098bedb3e687a848c4463cec1793470b` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/22_0.png` | 440,367 | `036de02400b0a84e87d2801ec5a27f1564956b6b21dd846b9c6077cde1a6dd36` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/23_0.png` | 464,178 | `934caa11a044c7246332f8c4b532565a048ec2c7d1683ee2d9d8c3df5c280402` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/24_0.png` | 258,264 | `2d5243de34a0369ce6bf9b89df29cb333e015dd4941ac222a8c960e45dbf914a` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/25_0.png` | 305,059 | `0791cf468bb380f8968b0ce6a56eec75adf2b239b5096102701f8a71387e8f20` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/26_0.png` | 264,568 | `bce1afed971337b3eaa519ebc3f2aefccde74e14d71a0a4465adede39ef759f9` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/27_0.png` | 311,848 | `44fde09b773384945b843db772dba9585ba94da8b11d40df1868f1755524e185` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/28_0.png` | 475,100 | `1116079316956756110fecc1be183c3af9be15db2e1a145ff1c706f258d42130` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/29_0.png` | 464,204 | `f8bce6fb0b5c70e24c97bc35b4617689ac66d12ed6f9242ef59b762e74007569` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/2_0.png` | 494,285 | `67e426a5fd25458f078a23af9a7d0f4b5159e9056d9bc49a433bbce2cbfa2787` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/30_0.png` | 397,059 | `ccb7a47616fd3ff3f518c9750a1ab47dc2b0657c418b83ba5259cc5b3f7440ec` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/31_0.png` | 427,962 | `7c77e6268984fabc4544a074e94bd1f2dd22ffe8f02f2ed1ea2c041c2ae77bac` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/32_0.png` | 353,075 | `ada8a8770c9996449085e8d48bb20304a8f40b0344aea3f3a6293ba3bec81964` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/33_0.png` | 368,956 | `f83003c061ea0a2d2c70dab6fb8520740e584242a1938ef9e03843a03e6ebd69` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/34_0.png` | 301,767 | `1cf50f6e70a2dcc2e63d351075a474b72cb3e2ac0f24d0c77c017b417f7aceb6` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/35_0.png` | 340,085 | `df2bbc05672dd6c01aa40ec8adfb8e1ef00c796d0cc29a9cbdd8ab0136d27b7f` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/36_0.png` | 388,637 | `fc8a02d898b03d25de67cc5747feedb43a41686258413e0004329df84edf0e3c` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/37_0.png` | 405,083 | `29fbbf83e9507740c8d5359ee485eeb8ae7e2de5599780f8c98b416df20aa2d4` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/38_0.png` | 413,183 | `f575a076c2f5ce438eb1027adbe0b35fae97dc7fb57ace50af405cd27af5bdb4` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/39_0.png` | 431,523 | `c71381028130daaff120b105204d289ff266b7c66cd78b381482bdbf4c33fff9` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/3_0.png` | 499,922 | `21a58a5e1af8c0a58044b96b6792ff03a6337c8570d709c989a2033a090e29c9` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/40_0.png` | 447,611 | `9c55e0fff655798897762a76e1bda7eab645ffa5cb9b8a9e2f419cf0f9bb1faa` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/41_0.png` | 501,085 | `b4c15052b42007ff610fb34d73e636a52e2c5bea6b790ff7b1e65d67708b4d46` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/42_0.png` | 427,826 | `5636d99906a0b27a01ffc5e64bac5425324dfda187a4d52ce0e56b477cf0e758` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/43_0.png` | 482,811 | `618373f3bf5486ff8967046c62b68cd70a2a0dd453e455cc2d3ee6264de99cec` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/44_0.png` | 472,714 | `16a0e67ea4f88ce58e61cdb955140d58590236ac30ad8c2487ed5e60cf46dc56` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/45_0.png` | 459,897 | `6adfe82f10ecac924fa3e87be437e30f1be2e105dc23c206789680c0563f0a5f` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/46_0.png` | 443,469 | `4c1179d09f84ba536d01dd2437f53c5503c67ad77885dc5d99ab0c1f14067346` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/47_0.png` | 470,482 | `e158f7237203ce1c7b1bb0e91ad895e7458bb44bff4338299ce01a4efdbeecab` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/48_0.png` | 584,500 | `ca7e3625725f8bea185b99dc02945be144e559b4d2abe635a710d2b6d7f782d6` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/49_0.png` | 498,160 | `0c4cb0d4c1d838387c5304f488977af8646d65c8a148c7b7e988f5f3af506f73` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/4_0.png` | 497,003 | `52070b45af5d82e8d206d0a89bb48096c93957089050a07bdf16ca14ea0d2f08` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/50_0.png` | 577,767 | `274394f1555861d399a8d51b035b7e2ee08cdbbdbf9491052242de2041cd57b3` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/51_0.png` | 483,991 | `d5009f5bbd8755d990574e9f50d46fa56bf49f0757317e8ee7c9a288dbd40fce` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/52_0.png` | 448,381 | `cdf5c14200c86508ba76b364068af591383c1dd6b48704e81c554317e8a64d35` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/53_0.png` | 427,174 | `466404d7598fc27a0765e8683e16e7df897ed15b85239852210d324bc1c83054` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/54_0.png` | 430,290 | `271f5913c5ef4199d91a905bd1edeec3d3b3201c249a7d947e6932b22e62ea07` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/55_0.png` | 432,046 | `535c9023e03b5fcaa403f18a556bd8fb020d96d379ecdf58e40be4bd5aa29cc9` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/56_0.png` | 308,146 | `aa7b36c5e0b0ad5c7c762bb4be9213cbde42e3a741867626c01de9230bd4b0e7` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/57_0.png` | 291,882 | `feb5564e6539fdcbabd027d937426e9f8d24d220f1316eca217978b4c2d82da0` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/58_0.png` | 347,847 | `c7308fd18be85b495af2665f9f815f3c243eb4742cbf1d1166429487979a1aa4` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/59_0.png` | 296,893 | `ec21e292ca06f3513689e1fc041be3faabd20d1734b400bbc1d1845e7cbd5aaf` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/5_0.png` | 531,654 | `4885aa04b4f78d69dc1425b95bbd3b7e56bbad9f717b1f9604ddf8430c89bb98` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/60_0.png` | 417,917 | `5965ffa1da4e21bb24ab15a5654f1b27800a474fffc4e5a2e245b6529d61463a` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/61_0.png` | 403,618 | `e2d7d9591727130395d2241c25c225b67917d7f0a31cf1138687c7a283a9aac8` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/62_0.png` | 435,788 | `a1425015d719848675bfff4ef29bdc8743d9fa31a400bf281f3f697f552bf445` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/63_0.png` | 393,278 | `b4dd1fbac91661b0a9dfda55b046c9b35f965bdcb64936058f013fcfb3cd1fae` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/64_0.png` | 440,322 | `42ea5ac3db2d938ed5354bd8d26b287c3c9e46f1515f31855d2a8822ca047535` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/65_0.png` | 465,421 | `a076df6e0a92537274fc5bfea60acdc0103b1e16b60b85cdc772f03f251c07c5` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/66_0.png` | 421,208 | `28ce12d437e1b756001d0f384c15b891451ea5b1c70719c4173eddde910309d4` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/67_0.png` | 439,322 | `472f76fe2dc7da9265a9030949fd5f1f2e84733507d868e501264ca6390af34e` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/68_0.png` | 375,053 | `8fd937c4f19bbc2805c056af5fe31b7423af1a4b98a0fa57f418515e92755e30` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/69_0.png` | 367,050 | `b548186811a2e5100200ffdf680c908b681693f7811cbd77fd24aa5d38f1cc57` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/6_0.png` | 435,175 | `1f6872ed188c815cec9f7d9e965d6c634c1b4986a55d6692d46e35e1830a1d4f` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/70_0.png` | 371,696 | `7ec1398b491a25e809cc4ced07e8c6cfae9cdf0d484d86fe7f409338e5266b72` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/71_0.png` | 361,360 | `43d9014d61d6b9727440851016032d84bb7b2d59134deb0e36aaabce906323fd` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/72_0.png` | 450,249 | `0b28badbdbcc40c58f47610bf77900c5ed2d6a0cc6c0388123bd0474df8ea6db` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/73_0.png` | 488,316 | `d2b254a018b683e66bbd650679fa86940670f140e6390bcd6e0912d389ad2c5a` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/74_0.png` | 455,145 | `cc2f9ca1f06880e035339ce753d1dcdf61e1e6c69ca2398d751de44739fd2bf1` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/75_0.png` | 474,261 | `6055127befb27b0abd1388ba2e3d69439fd1c8b3269d81944c99838bcf081371` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/76_0.png` | 429,216 | `eb5d74f2cda1e21ec54aa84148058ddfd28fbeb50e7dd863b17a34162b563233` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/77_0.png` | 352,329 | `ad62beab09e23a0bc983166e4a80c0f0ff9c352dd5e691dba7293956111d99cd` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/78_0.png` | 430,831 | `2433c1a6bc40ac219214fd6d10f68ab27ad25413b25f8fd5910ad59b39f626e4` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/79_0.png` | 344,358 | `7ca7e863688122afdceffb29fad1bcf4bc0da1b30932073e86ee98ae049bca1d` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/7_0.png` | 504,449 | `559e7f046e28aefd02dc390515ff6665ae7bfb44608ad120f535f3e07a0c78f5` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/8_0.png` | 427,433 | `1c81525e490cc87af96da3d80d2f6a0bd5a07f5f1d97361fbac9d671af6fe6bf` |
| `results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images/9_0.png` | 430,110 | `0cd2762cc3f804d07087ad477ca59726985c486506e274cac45770a5ee865d35` |

## Blinded human review package (C33) — protocol, manifest, mapping commitment, bounded corrections, prep code, and the reviewer zip itself. The private neutral-ID mapping and its random seed are intentionally NOT listed here (kept outside the git checkout; only a SHA256 commitment is published).

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/FROZEN_PROTOCOL.md` | 8,788 | `8f37086ccc7d9876a3c732315ecc78eac533e2811cfa8901c7f4fb86cdf0b5ed` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/VALIDATION_REPORT.md` | 6,382 | `08cc19c5aa69e24667039cdc37b9eded358593915e78fe18a0521bd6f69f355a` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/PACKAGE_MANIFEST.json` | 4,177 | `9f7a55ffe3a4cc2000242e27bbf72b97e30b81819418e0e3a055c8e8cb818fe1` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/MAPPING_COMMITMENT.json` | 470 | `a074f89d9b88d629888b4fce164351566ab82fedce5c56b229bcad8e3205c7df` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/READY_FOR_HUMAN_REVIEW.md` | 2,831 | `d2aa610cb4463a2c08bbbe38b73c61a6aa4afc08a6bfb7d3436031de8815f685` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/CORRECTIONS_BEFORE_HUMAN_REVIEW.md` | 6,795 | `7ad65c3235dd021712df5868a8fa959d71956ada20743369d8d5219f75589060` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/prep_code/build_reviewer_package.py` | 6,411 | `5d1d7603bcba8abb9b1d8cc9083fb3fc25327a5e1effd8e6eef1ee419ab79d82` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/prep_code/verify_source_images.py` | 6,497 | `6b37ed628b07c948d361d0816203bf9fa1318468af2344d49fd1b7fcfb260470` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/BLINDED_REVIEW_PACKAGE.zip` | 16,507,991 | `7af1cac52bbeb36d7f8e580081ec882992dd6d41fbd65559f01a5565fb06ac39` |

## Unblinded human review response (C34) — preserved selected response, receipt, mapping verification, unblinded mapping/joined table, case-level agreement with assistant labels, and the final human review report. An earlier survey submission was explicitly withdrawn by the user and is intentionally NOT listed or published here.

| path | bytes | sha256 |
|---|---|---|
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/responses/review_export_final_2026-10-16T14-06-23-096Z.json` | 4,705 | `7c4c509f470abbdbc79b8bef3aacc2f065bca35cc5f792cdf59fe957fc3a57b8` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/responses/RESPONSE_RECEIPT.json` | 1,664 | `2fb7e7a7187f6b70e73b12efc7262c6cf87f05e286fc04ae02c50669e0ab140c` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/UNBLINDING_VERIFICATION.json` | 2,501 | `9a3665d1de1ef524fcfd4dfc8f02511892da5bd77c574142db873174b84c8d19` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/UNBLINDED_MAPPING.json` | 14,206 | `bbf8a9eb8b6459bdb2e4ff33ed8638c5708fe4ecb10e58c09a02ecbdcb47689a` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/UNBLINDED_JOINED_TABLE.json` | 10,096 | `97c4e04c05de61614e63defdbbec2437f5bcd0d5a2963f419ff81cf6830ce9b1` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/CASE_LEVEL_AGREEMENT_TABLE.json` | 4,812 | `28bd14dd3100fbd6f2811f17468b275c51b6cf73f62862a0fa3161e02bca5c26` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/unblind_and_analyze.py` | 9,643 | `6d2eae294dd9c41502a6143a5ff4b4693c8c68a0acc342c4d91a8f7b00d5cf57` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/compare_with_assistant_labels.py` | 3,004 | `93d727d36a83d518b672fe41557db2b953160d879410335e80c0f35625135053` |
| `results/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/HUMAN_REVIEW_REPORT.md` | 11,271 | `f0ed592acb9e4e944878d0ff47afd0ef3b05d98868108de1fbd03189170bb976` |

---

**1095 files listed**, none absent.
