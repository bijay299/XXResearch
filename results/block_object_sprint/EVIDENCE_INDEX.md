# Evidence index

Generated after all other files were finalised; **this index excludes itself**.
Generated (UTC): 2026-10-18T04:01:12Z

**128 tracked files, 16.33 MiB.** SHA256 of every tracked file below.

## Full-resolution originals (GitHub release asset)

Hashes alone are not accessible image evidence, so the originals themselves are published:

| | |
|---|---|
| asset | `block_object_sprint_originals.tar` |
| release tag | `block-object-sprint-v1` |
| URL | https://github.com/bijay299/XXResearch/releases/download/block-object-sprint-v1/block_object_sprint_originals.tar |
| bytes | 835123200 (796.4 MiB) |
| sha256 | `51da8bd519031bea16e72fe309c365e0e4482c8bb08068645f660527d9ef6ccb` |
| contents | 560 PNG originals (M0 160, T0 160, N 240) + `ORIGINALS_MANIFEST.json` |

`ORIGINALS_MANIFEST.json` maps every archive path to its case id, object, stratum, class,
theme, init seed, prompt, generation order, byte size and SHA256; each hash was verified
against the file on disk when the manifest was written. Downscaled fixed-order contact
grids of the same images are tracked directly in [`grids/`](grids/).

Not published: trained transcoder weights (2.0 GB, local scratch only) and all credentials.

## Tracked files

| file | bytes | sha256 |
|---|---:|---|
| `BASELINE_VALIDATION.md` | 8021 | `e97fadef0c9ab21f9cb4a040e685009a87dc691b8444073ea4668878f7b2e18a` |
| `INTEGRATION_AUDIT.md` | 10106 | `6a9e3ca547edb7a27bd961cc28acd0b1394f4922a36d678de94883733897baa0` |
| `PROTOTYPE_DECISION.md` | 5448 | `3399db1e8869f6c5a8d82179165369097393cc823e07fcf53101460884f6505f` |
| `REPORT.md` | 16927 | `894b1eaf802b0267c3af686461874ca05dd50b841dea0cd26eaf2cbb02c14563` |
| `RUN_MANIFEST.md` | 9693 | `fac099abeaecef9c726c019a4df9548f26ee87f73eb50d57ce281047588b04d2` |
| `cases.json` | 100060 | `2c6b4d0805b79c02206b636064b374c264d14b5bd66be50bfe70e48a9c10e532` |
| `code/analyse.py` | 7699 | `e88c716f66964127964eb4a6f16e89abe4b6d3eafca8120c307ffb1b20e996f7` |
| `code/apply_patches.py` | 12700 | `971a168eb90f2ae6857fb987af6d23d9b47373415bc5426bb2735f2f652e487f` |
| `code/build_coco_assets.py` | 8609 | `8fa3058c9c98837d18b2a2c1638a150d370f85f4b377f5c572b7dd32999eeaa6` |
| `code/build_eval_views.py` | 3188 | `0dd6b47a7c5b969069802aae7365f1dd29049539e92d0f0045321ea8d7d16da3` |
| `code/drive.sh` | 1652 | `004ecf82a1416898552570cdbd2d5209f9e74a63c1651c0ff694a43c9f47b4ef` |
| `code/freeze_cases.py` | 6804 | `ce0218c471586d1be2d79762d7760e9ae2d29ada9a1f5759436163aa33b57bce` |
| `code/ledger.py` | 4640 | `ae2c9b5fad87e939070da2e717c56b372582a6904929f4ce4e3512f94ff5e57e` |
| `code/make_audit.py` | 1774 | `3698f26535a3c1d220df21e8dba2ae4ba5a832a84cab5edd1aa491d45762e8c6` |
| `code/make_grids.py` | 1584 | `4b9e4fb54513f9dc5316d4badf54cf50b99bec0c69c51553d24fe44b13dc3133` |
| `code/run_arm.py` | 11767 | `d01af2bfae6cb722b779715d4a12c83781f6b849e7895dca57c3bc2554a0f424` |
| `code/score_presence.py` | 7513 | `51b664a8637df9e9d1f337d516a9b9bc6326afd622d138a6d32fb1869f573bf6` |
| `code/smoke_flux.py` | 5195 | `d3a21c2718c5fb781f6b074fca158fae669b79b62ce2c7719ef5383ec5b7cd94` |
| `code/validate_scorer.py` | 4809 | `557f1d2bccc3b62e4df9024b1c9734b3abc2191b71ed22a62ba5f70fb7c008b6` |
| `code/verify_activations.py` | 3637 | `dac582c2c32e7232e3db58efa814c8d84ed7474dce534d0baff0bf342ca1d807` |
| `code/verify_transcoders.py` | 8346 | `62ba7b529e8b8532a6b56747d9789ca045179bd74f3964c507fd9ffd6353c02b` |
| `env/pip_freeze_block.txt` | 1078 | `c34aa0a12caeed130101eb7f985f0d4c119e1ab31ed634ab2323b6cc2add5e5c` |
| `env/requirements-block.txt` | 250 | `a01c899865f6848f7c5cf39e1ce11845deee1dbf4535bf3ddb014f8a59a916af` |
| `env/runenv.sh` | 1406 | `8f6c4f679bf16ec9b2a831b04a244417ef3a639eaf5be245e0c87b301c86939c` |
| `env/setup_llava.sh` | 1120 | `d236b676a78583abbdd6deba9aba290105d20541fd81de48ff45a429123eaaa4` |
| `evidence/ORIGINALS_MANIFEST.json` | 256022 | `6701696107440724e1e3728c753a8689a0a9dd82701405b8c8ec606cbba79b42` |
| `evidence/ORIGINALS_TAR_SHA256.txt` | 100 | `5380d7a00091260f5a379ca24aac2ddd1a769fbdffae31f15e71886553169eee` |
| `evidence/activations/verify_activations.json` | 1353 | `a9ea74348a94146a9039f1b774a52132307c5f57f1bd0f5ca14515c8355575cc` |
| `evidence/arm_reports/M0_Bears_target/arm_report.json` | 1237 | `722caecb50a8900b15bf799c1a0800811b5127e8291e4115d0813171272bcbf4` |
| `evidence/arm_reports/M0_Bears_target/cases_generated.json` | 18666 | `7fcaf0cd0187b54fc62f2cabb270b337c7d15f1c2865fb04f3d9d4dc481e7aa0` |
| `evidence/arm_reports/M0_Cats_target/arm_report.json` | 1236 | `1e9501507b190877a211b561cd1d0df651a38f9c4c33da792abc7cdcf0dc2b42` |
| `evidence/arm_reports/M0_Cats_target/cases_generated.json` | 18389 | `59be9835611810332d93f1549a03e8473ccfcfe05fd1c619566003ef75e6c935` |
| `evidence/arm_reports/M0_Horses_target/arm_report.json` | 1239 | `847d18d4d885e3699b89e5655e389fdf43310662a55864037b438ec4aa605b81` |
| `evidence/arm_reports/M0_Horses_target/cases_generated.json` | 18948 | `049c0803dc1a8039fd7bb1d19341da938082d6a0b48e2509f28757aba48ce2b8` |
| `evidence/arm_reports/M0_SHARED_retained/arm_report.json` | 1255 | `9e3ba495494bbd1a62de77ef44f0da93b670b8983da6eac4005cd2808db29e9e` |
| `evidence/arm_reports/M0_SHARED_retained/cases_generated.json` | 19526 | `1f4d1b2b1871a3de7b0fd8116ecdefc56550ad7eed70e1fdca62d46ae1971b4a` |
| `evidence/arm_reports/N_Bears_all80/arm_report.json` | 3295 | `b71fa98477e7ca533e1f13fbabd830ed37500559e2edb290ae2a77c5478a5825` |
| `evidence/arm_reports/N_Bears_all80/cases_generated.json` | 38205 | `d9c37c9c7b9ef0cca87ef5bb94b49f2522beb061efb04e51793ecc0d76d18d9d` |
| `evidence/arm_reports/N_Cats_all80/arm_report.json` | 3291 | `eb1ed7762b64c1ca621ae4f977105468f5807ac5fbf587f013280e1d9b561077` |
| `evidence/arm_reports/N_Cats_all80/cases_generated.json` | 37926 | `354f9c658854946282aa3ce4a9dbbf6038f5d24659bd845f58d88c2dcb78f50d` |
| `evidence/arm_reports/N_Horses_all80/arm_report.json` | 3291 | `1c6c3bdac00fb1a8db582e27727765ea969e6fb05950518e4ea8e8e11120385c` |
| `evidence/arm_reports/N_Horses_all80/cases_generated.json` | 38485 | `273530ac4a3ffe5835d12a247871a8f348f88f3f41cf94efc71a9ba573233814` |
| `evidence/arm_reports/T0_Bears_target/arm_report.json` | 2080 | `9408cf1dffbfddd7bf1ddb8d52050eae08f68302fdf6c7a4c34b3055b00d186b` |
| `evidence/arm_reports/T0_Bears_target/cases_generated.json` | 18667 | `02c2239bb5729cbe3b04ef92211e11e7e10abdf41988f927fd0ad660e08f38e6` |
| `evidence/arm_reports/T0_Cats_target/arm_report.json` | 2080 | `d11389ead330be54db06df7cadc073bdcf1ebb42911f971ae2f16e128375c967` |
| `evidence/arm_reports/T0_Cats_target/cases_generated.json` | 18389 | `518a83d097cc838fa3e8ada2f51ecbb9c34388b04e116f784bd8f87aea10dd58` |
| `evidence/arm_reports/T0_Horses_target/arm_report.json` | 2084 | `4a737ba6de6a7e7cc79030fb64bcb0cb2ffc0c60ff9f51e0ee5f78ba56fc091f` |
| `evidence/arm_reports/T0_Horses_target/cases_generated.json` | 18948 | `a6aeb0a9aab386b72a7b755d820b69bc1a655c129539811205000b8653968a82` |
| `evidence/arm_reports/T0_SHARED_retained/arm_report.json` | 2099 | `412825e403530fc8e920486212685e3dc460fe9b248f0c06b5528a02628379e2` |
| `evidence/arm_reports/T0_SHARED_retained/cases_generated.json` | 19526 | `e14e4dd42f4a6063ad48fb23398286b9fc425967d2192f8e19b7fff88fc283bf` |
| `evidence/audit/N_bears_all_1.jpg` | 591239 | `da93bf81d8576c4cb2cbfec1a4212fc5bae2b7649ffbbbe88a89efdb06cccb5f` |
| `evidence/audit/N_bears_all_1_key.json` | 5957 | `4639c637c6da450628dcf0da4d8682d47be0387a5b24970bc4c2f7cbacf7dbad` |
| `evidence/audit/N_bears_all_2.jpg` | 649607 | `c35a203eb949bad7e04c58b6eb44571c84243dc3ec28cd1ed8c759090816d284` |
| `evidence/audit/N_bears_all_2_key.json` | 5908 | `4b4b0fa21455b8699d64b0c3dde9223d2cf66e55967c02f092ae4f11b765bd22` |
| `evidence/audit/agent_audit_N_bears.json` | 2314 | `66684ca24a96dc5cc95d61c3e49d5470cb901c3db513524102f758caf53635e7` |
| `evidence/audit/audit_arch_blinded.jpg` | 258063 | `999d059d39a81c3cf194b3558236ab7512bcad036d7ce0e0543eb4fb88ab779b` |
| `evidence/audit/audit_arch_key.json` | 2786 | `76cfbfd82c346ef5246507c8ed7fb2cad76f12e730fdd95fcd225cca13d26a06` |
| `evidence/audit/audit_bears_blinded.jpg` | 644904 | `f7eec79894e9ca36af594dfd3411d3a596fe934132d949472ef332bf1b622e40` |
| `evidence/audit/audit_bears_key.json` | 6238 | `c90f1401967f9f2b0e1ebfa7283dc014d2641812dea1380b6488d3c7a1b8356e` |
| `evidence/audit/audit_spec.json` | 5898 | `64553eec229dbbc098f8077b81b97f5d1d565312d401d76940b931a05a7264f2` |
| `evidence/audit/audit_spec_N1.json` | 5617 | `914d3a6dbe7984917e1ad3272f106ee8fd1ce011c6e84066d611440fd41a6178` |
| `evidence/audit/audit_spec_N2.json` | 5568 | `282f71ab7dc42b728fbbcc6b5d0afc2031c8011f5bb291a356dc57045fb0e215` |
| `evidence/audit/audit_spec_arch.json` | 2650 | `0d4ac2b76110a519fc3e5157c134d0dc1007d7785903fc8fd65142522fe5445e` |
| `evidence/coco/calibration_panels.json` | 69203 | `8d91987d1e02370531b9b480942d4a641025ad7c0563ec144ded8c41af604538` |
| `evidence/coco/coco_30k_provenance.json` | 3472320 | `c6f083eac5d5f006752d4eddec72236a8be118cab64292aa614d8cac703ceccd` |
| `evidence/coco/coco_train2014_30k_prompts.txt` | 1592395 | `25c0dc8e7908446729371320e026636c2ae9abb1ca9fc7d9c718e18918b70573` |
| `evidence/efficiency.json` | 1127 | `8874318078e3ab076a12a164f0ad43a94731691a5007572fe0fccecab3e2d69c` |
| `evidence/ledgers/SPRINT_CLOCK.md` | 541 | `345987de32b18319cae7525cfdcddc6ad1b1f535ca5e6408cb0d950683818de1` |
| `evidence/ledgers/ledger.json` | 15529 | `45ee7d6f08defd818831c0a1a97720eec4aaf113cb05882d1715b90c199f21fd` |
| `evidence/native_eval/M0_Bears/per_image.csv` | 7585 | `f02dc1924b69f5c17708be0ac58266ee6df79d2b5e450c9d2ad32df9994f590b` |
| `evidence/native_eval/M0_Bears/per_image.jsonl` | 21943 | `cedcae60021b290c9d7a3b000049f5d2447fedb4d1fd66c4ed195e8dfd8b2a73` |
| `evidence/native_eval/M0_Bears/retain_and_unlearn.csv` | 68 | `e235d1daec48824fc0c56f6c3af31050faef548ff8bab62887b01b0e8818b06e` |
| `evidence/native_eval/M0_Cats/per_image.csv` | 7431 | `aac560df183a35a82d1ce3b018c9d3348e072c8a9002c4690e9ef110cb5868ad` |
| `evidence/native_eval/M0_Cats/per_image.jsonl` | 21789 | `bc16ded557c858ad398ceb39f66df9172ae694de950ff002ca530d8dc26d0ffa` |
| `evidence/native_eval/M0_Cats/retain_and_unlearn.csv` | 65 | `2f5f58884df33e85d34008c1b7c83e2dc3baf38960231b38933c1f77ac5345f1` |
| `evidence/native_eval/M0_Horses/per_image.csv` | 7725 | `c6ef9adc322aecbbf6ac49e50e2982360992fee99dcc28e81482a7c7c677950e` |
| `evidence/native_eval/M0_Horses/per_image.jsonl` | 22083 | `2b9c55542f70616eadbbf3685d1628990f26a2afc657ae525561cb285d632a44` |
| `evidence/native_eval/M0_Horses/retain_and_unlearn.csv` | 67 | `bd67fda64aa2332593d0bb21d4363407d9b5ade8cef82111bea154f48db08cca` |
| `evidence/native_eval/N_Bears/per_image.csv` | 7806 | `0d8e67fab429b52368d75acc0beaca73d551fc7b8e4502aa23d2b3dd7b1bc50c` |
| `evidence/native_eval/N_Bears/per_image.jsonl` | 22164 | `3a96633b0b8c648a6edb36ae5b888b0a5a7e38b11d66b84c45be24cb72dde742` |
| `evidence/native_eval/N_Bears/retain_and_unlearn.csv` | 69 | `f1f9d7e77592e02d6c23cb72702129e79aea9a12db385776a7768caa4c9262b1` |
| `evidence/native_eval/N_Cats/per_image.csv` | 7774 | `9fed38dc329edc3f77a2fe522d62ad1a00a25c09292f95b8fe1621d54af909c1` |
| `evidence/native_eval/N_Cats/per_image.jsonl` | 22132 | `97d1f29cf8e20efefc5d81a412c04f0a9246321f2f3840cbbbc15ab08c65dbe9` |
| `evidence/native_eval/N_Cats/retain_and_unlearn.csv` | 71 | `0c7a729746c2cb9824794ef4b6af64c61316990a24961330ff0d1415a93c397a` |
| `evidence/native_eval/N_Horses/per_image.csv` | 7945 | `5514a654b53084d47fd03ce7b938d110f5fe685f268642c565f422bf6747bdab` |
| `evidence/native_eval/N_Horses/per_image.jsonl` | 22303 | `cebddfefcc861b356364a39798d0f82d80d5e603b835f3667b1182b67237503c` |
| `evidence/native_eval/N_Horses/retain_and_unlearn.csv` | 69 | `81a9c76f4a38d780496b93f75391feee71b193a0d28229528dc2827dd5863c90` |
| `evidence/native_eval/T0_Bears/per_image.csv` | 7549 | `cb0cfa31228be457905d04ec53b190595d5377f0cd3d3bcadb7ae8c435746b29` |
| `evidence/native_eval/T0_Bears/per_image.jsonl` | 21907 | `77c8237af7f0b24d2cc34a9ad7ae861521e826863bf2fcbeced97cc95df09160` |
| `evidence/native_eval/T0_Bears/retain_and_unlearn.csv` | 67 | `823a5fe866d2a4e2936e70e1d2887be089c10f15637deb5bd4c72e1411807ca6` |
| `evidence/native_eval/T0_Cats/per_image.csv` | 7390 | `23e3c1782518bdf694eae86e76dd6e7761324f42056ff186772350b4bedafc7c` |
| `evidence/native_eval/T0_Cats/per_image.jsonl` | 21748 | `92e050d4573ec39910e9eb1278bb51796c0d14229b41b4073280d682b6c5f91d` |
| `evidence/native_eval/T0_Cats/retain_and_unlearn.csv` | 67 | `feb52bf8a45efd687479c0a4e5fd3cf10274f96badadcf0d0ddddc5a6dafa7cb` |
| `evidence/native_eval/T0_Horses/per_image.csv` | 7714 | `58107b784fc2681127731d2e274b0c1d763da29a67427069df86539a29ac6df6` |
| `evidence/native_eval/T0_Horses/per_image.jsonl` | 22072 | `8c27c5c7960178603f7fdb090b195900cb6ec5b63af14b1cdf357a0640cd3cd3` |
| `evidence/native_eval/T0_Horses/retain_and_unlearn.csv` | 70 | `06b04f52d9eefc8955e7cb4cf015060cada10892c6f26b31d925c40199db3dc5` |
| `evidence/scorer_validation/answers.jsonl` | 57453 | `fecb5c76b3dbf600ff5e0605e0bf95477e8150ec0d9e90b24fb9e71c1234a24c` |
| `evidence/scorer_validation/manifest.json` | 43106 | `a3652461d7d999bfd1205f436d2411883330d425820f0c98f90965dcd50b63e2` |
| `evidence/scorer_validation/validation_summary.json` | 1746 | `92f7e66d609680176b26bba1e5bb820f5dc5d5f3ff9a943c3502b8c39d5fd0c4` |
| `evidence/scores/M0_Bears/answers.jsonl` | 51060 | `a99a16fe1832e609dfdfa9c0cc77ae5fc9cb368099d24f663dcdc2da5428f103` |
| `evidence/scores/M0_Cats/answers.jsonl` | 50620 | `2ca715014efd26dc8255df5345c10fb5b7150856c66223b707c8fe832852353c` |
| `evidence/scores/M0_Horses/answers.jsonl` | 51500 | `aa2199f72dcea2e23ebe7008d46c57212f7466da58a9abc301f08478ac6c03f1` |
| `evidence/scores/N_Bears/answers.jsonl` | 50544 | `1a3e934d28160757f9f4f43dbd6ebe1abab31db431eacd5449eadc5820063578` |
| `evidence/scores/N_Cats/answers.jsonl` | 50058 | `fba90b6c0358aebe1c8cc3cdfd6f605a90963ded74c5cd3d36c0a0045d762d9b` |
| `evidence/scores/N_Horses/answers.jsonl` | 51012 | `bbc17ac3cf79b7589a528b0ea550ed529a8fb2c768dd404513d8f33a4e7f0505` |
| `evidence/scores/T0_Bears/answers.jsonl` | 51058 | `97e33429744818a6119b5b39d3c4d348df02511bb7720e4ae27719954293c570` |
| `evidence/scores/T0_Cats/answers.jsonl` | 50618 | `59ea1f9983a9f9cf1410458a6902c4a75a5bc231d2915c396a5e284c86421d92` |
| `evidence/scores/T0_Horses/answers.jsonl` | 51498 | `b2fe3f67df229158ca6f7cfa80ece18e0d461d9471401edcda7304d19b3733a6` |
| `evidence/scores/comparison.json` | 6114 | `d0517b89e76c75d77deb698de57ea251a4835297c6ac763ee024977fcc7fc37c` |
| `evidence/scores/gates_4_6.json` | 12556 | `0a8829c0ae616cd38d480494e099851d896a66b02cae9a38ac0bd69df759afc9` |
| `evidence/scores/native_three_objects.json` | 16406 | `c8db6be5ff972c59fee5c49f2029144da336ec8232e28ff4a69409b138384d30` |
| `evidence/scores/residuals.json` | 321 | `c7a7f6fcc3c5942665ad93e83232e1e9aa7e1ae6ec3596298797934cbc223ec1` |
| `evidence/smoke_report.json` | 1469 | `a9ecfec8c6a57a9f8a0496bb864c265978961e2608790e6ef1d335e8362429f3` |
| `evidence/transcoders/transcoder_training_config.json` | 454 | `36d0c5e2d5041e796847b97ec0a7a830d90eb99efaa53d1a78cae711f0a52cc1` |
| `evidence/transcoders/verify_transcoders.json` | 4666 | `26ea048402d44b329114226bf31f593a282b194d124579fcb06d44ed1cc43dd4` |
| `grids/M0_Bears_fixed_order.jpg` | 902468 | `5afaf31e4ae59cf650b18aea7e9a13eeff8f945d2a39754bece62391b0f8454c` |
| `grids/M0_Cats_fixed_order.jpg` | 890294 | `57aafb6370894dbfe2e77e2ea1c381a09dcd27a57ad9a9ba9289d9588d9e3590` |
| `grids/M0_Horses_fixed_order.jpg` | 876230 | `2e8233d8e804f373d017c639f5636ebc27a775f4914f5b2b5ed5e2e32dc21e87` |
| `grids/N_Bears_fixed_order.jpg` | 960542 | `80d01effcd48f762d40480b5ea2e2702e6a534716f3bb39ab0b3949f12de35e7` |
| `grids/N_Cats_fixed_order.jpg` | 939939 | `ed46cde614d2ebd5c930aaa5a058719ab34f0d9fd18448ec788145dcddd879e8` |
| `grids/N_Horses_fixed_order.jpg` | 932036 | `518f77b315b33bb766761ffc4e99b6a888c622bc765ac2fddc1564f8fc133af0` |
| `grids/T0_Bears_fixed_order.jpg` | 870098 | `050244e012773e1c8296b442937b149bdac3eb41acbbaa81fe28fc443ac13570` |
| `grids/T0_Cats_fixed_order.jpg` | 857811 | `94a08dec1edfe604f89d90845e0d05166b4001e0ba06a7541f7bb384b0def50d` |
| `grids/T0_Horses_fixed_order.jpg` | 872063 | `2bd8898e485bf75e0cd5b6817c5f9bf9064490e5e5451553e70f12722e5d05e4` |
| `patches/SOURCE_PIN.txt` | 137 | `8868dc174516b759fbd5daaa9fef3dcb105dd8779045196cd64dc18fc967a710` |
| `patches/block_release.patch` | 22684 | `ed2b1399872b01a140b6fc2a0f03dc0fc3bcf5875565dd779dbf5500a91c857d` |
| `patches/new_files/utils.py` | 1080 | `41a99461421baa45a0a955f89384644c63eff43af2dc43f0c5ea74d8271e9c88` |
| `patches/untracked_and_modified.txt` | 664 | `5fef82463c14580044c9bee758437834ad5a2007b12f0c49002010cd45bdc440` |
