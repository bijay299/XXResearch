# Published evidence index — RECE golf-ball/English-springer sequential editing

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `5ce305171ae7` — the commit that *records*
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
| `results/rece_seq_golfball_englishspringer/CORRECTIONS.md` | 7,174 | `6ad442074fbcdd3a61c8b2b7d954bf4c546516b3417fe1b5e27d596cc49ece25` |

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

---

**42 files listed**, none absent.
