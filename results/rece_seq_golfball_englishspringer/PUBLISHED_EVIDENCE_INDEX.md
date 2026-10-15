# Published evidence index — RECE golf-ball/English-springer sequential editing

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `ff4866675078` — the commit that *records*
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
| `results/rece_seq_golfball_englishspringer/CORRECTIONS.md` | 33,916 | `a750146097532640d415780153769e474f248bc7415bca1aacb9ede3a814c617` |

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

---

**173 files listed**, none absent.
