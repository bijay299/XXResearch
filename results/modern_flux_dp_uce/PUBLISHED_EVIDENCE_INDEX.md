# Published evidence index — modern FLUX.1-schnell DP/UCE reproduction

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `a3edf358f549` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

**Deliberately excluded, unchanged:** the full 5,000-image raw generation
output, the four edited `.safetensors` checkpoints (2 tensors each), the
FLUX.1-schnell model snapshot, the isolated venv, and the pinned
(unmodified) upstream source tree `czhang024/CVPR_DouleProjection` at
commit `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (MIT-licensed; not
vendored since it was not modified — cited by commit and file hash in
`RUN_MANIFEST.md` instead). All of the above remain on
`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/` on host "hyperplane".

Regenerate: `python scripts/seq/publish_modern_flux_dp_uce_evidence_index.py`
— run it LAST, after every other file in this result set is final.

## Reports

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/README.md` | 1,913 | `45acc2923a93c02e551dc09f52005569c31d529ab9961d32673cca69530d484e` |
| `results/modern_flux_dp_uce/FINAL_REPORT.md` | 10,779 | `e4e16b13ec21515ab3bc0e4f6a32514ba38f1bc5837cb103419940c945224f94` |
| `results/modern_flux_dp_uce/RUN_MANIFEST.md` | 8,583 | `38fce22cb58eac0f1993bca6b44e673638e02adcc8776a489b189eb6199219dc` |

## Evidence — per-image classifier predictions (all 5,000 generated images)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/evidence/M0_per_image.csv` | 238,370 | `4217595b8ffacd7518387ad39dddb97abd4954ba7d02f4c3accaacd6a9b4dcfb` |
| `results/modern_flux_dp_uce/evidence/DP_A_per_image.csv` | 236,816 | `0698e2d6f2bc24d6636bc101d62e440d51693b9c7fe17348beff075307bf4dad` |
| `results/modern_flux_dp_uce/evidence/DP_B_per_image.csv` | 237,442 | `27aaa739e899fd674d760eeddd1949512a7d71ce3faedcdd5f1924537e57c388` |
| `results/modern_flux_dp_uce/evidence/UCE_A_per_image.csv` | 239,055 | `20e839e55584a7c92adf45d1bda129a0c4050125fe7f2a0d55572ae73cef4d63` |
| `results/modern_flux_dp_uce/evidence/UCE_B_per_image.csv` | 236,652 | `1030d6c8e83b8f72f768112facfbfd58d5a46364ba013f9aacc8f12715555ad5` |
| `results/modern_flux_dp_uce/evidence/recognition_summary.json` | 2,935 | `0dd43f81ed98fe53eecb26b72b0e423a643831df687fc9466ded3e2cedaa96d4` |
| `results/modern_flux_dp_uce/evidence/gates_and_retention.txt` | 4,503 | `3356735da7c5a2173cd6e8d127b9ae8293a80db86f4ea3905332e0042193a2c5` |

## Evidence — frozen case manifest, environment, provenance

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/evidence/full_case_manifest.csv` | 82,783 | `559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26` |
| `results/modern_flux_dp_uce/evidence/pip_freeze.txt` | 2,035 | `2526eba369fae27cc1c11d6017bcc157403ababf10569f426422ca5ed89b371f` |
| `results/modern_flux_dp_uce/evidence/downloaded_weight_hashes.txt` | 818 | `bb57f3157c294d462113e216d8f777d12c8c83a561a9fdae577390df3e63bea2` |
| `results/modern_flux_dp_uce/evidence/panel_hashes.csv` | 11,041 | `27939b39dd673903a7434a06b2400757af551bf3ae5a793c5c72887a9dd7269c` |

## Evidence — native single-edit logs (Double_proxy.py / UCE_original.py stdout)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/evidence/edit_DP_A.log` | 3,675 | `aae32b9c0f82617e92830c83ff49ba52fe2922b1e7c33272b745222bfd6eeaa4` |
| `results/modern_flux_dp_uce/evidence/edit_DP_B.log` | 3,641 | `869ed440c8f1dac8ec888d2cf402e56290302516a644b8feac81d6ca8968139e` |
| `results/modern_flux_dp_uce/evidence/edit_UCE_A.log` | 2,616 | `9beaae54e27738f1b045d90b7f857472e340b0a71b341e4ef985e674584cc130` |
| `results/modern_flux_dp_uce/evidence/edit_UCE_B.log` | 2,620 | `98dafb8fa68a65d70f6c4f5af5ac3ce94c7e7a581012472da1ede01fc5fbb61f` |

## Code — exact scripts used (case manifest, evaluation, gate computation, rendering panel)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/evidence/code/build_case_manifest.py` | 2,342 | `1c3d9dfbe868b7ac30b230f86c5cdc65cde2d403afc1a56e0018f169d8dc9c49` |
| `results/modern_flux_dp_uce/evidence/code/eval_all.py` | 4,622 | `3ff25a5536e50e51770eb8d6e4909b36193b98f2edd72c7ffa0a19c28e051100` |
| `results/modern_flux_dp_uce/evidence/code/compute_gates.py` | 3,299 | `a7fc52d76cca118305cb571b1fbcd61e5685a75d918c5516ec62444aaf385b43` |
| `results/modern_flux_dp_uce/evidence/code/build_rendering_panel.py` | 1,411 | `25bd690ffe871d0c9f4eb6d11e729be69903a24e047c883b216746f26ca229bd` |

## Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, M0

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/images/M0/000_cassette_player_case000_var00.png` | 387,770 | `deb13e3d8943da4153e79160399f2c3464d1eacecb5a02d5850e1b16b0adcf9e` |
| `results/modern_flux_dp_uce/images/M0/001_cassette_player_case001_var00.png` | 193,207 | `994e3cb6d447f1530cb15dcb5dc29e642bf4eceea04352cc68fce770b92dc1c9` |
| `results/modern_flux_dp_uce/images/M0/010_chain_saw_case010_var00.png` | 453,929 | `ad720b86a09ca22cebba7971a05348bb4b263ff7b9d017ba50d498c8b88c6469` |
| `results/modern_flux_dp_uce/images/M0/011_chain_saw_case011_var00.png` | 365,225 | `6a58916a4c1e22e562087e232329917a33ae8c81ad2f9316a79481a2f4502406` |
| `results/modern_flux_dp_uce/images/M0/020_church_case020_var00.png` | 481,200 | `f7318881913c3740a977620f09b267b3d2d33ccb7a9eb9252ad2f77d8684a570` |
| `results/modern_flux_dp_uce/images/M0/021_church_case021_var00.png` | 381,019 | `43099c38bc2c9c01d8ebc7964f80aafa75ff33bf61ad34ab4c5bf15875efe02f` |
| `results/modern_flux_dp_uce/images/M0/030_gas_pump_case030_var00.png` | 297,521 | `66d9053862415a9a03f1fe94502318b5ef6ed62f807bd1374fef79a5b635655a` |
| `results/modern_flux_dp_uce/images/M0/031_gas_pump_case031_var00.png` | 322,412 | `ea2e13d7898b9c52860b28dee4721c49ab028a8e68a1609360b3aeb488101bd8` |
| `results/modern_flux_dp_uce/images/M0/040_tench_case040_var00.png` | 306,408 | `ca72563b632be4eb49fb3d391bf38b9d67ae0788115990a8a9beb8b60105fe41` |
| `results/modern_flux_dp_uce/images/M0/041_tench_case041_var00.png` | 343,279 | `504d9cf7eb50129d8c3975cd06b04f00d49099b5d441bbbd4577efa9938e379f` |
| `results/modern_flux_dp_uce/images/M0/050_garbage_truck_case050_var00.png` | 462,527 | `808dccdaf374a03c3e7c5ccf9da4818f17016b1706441bbe163ba40c2439867e` |
| `results/modern_flux_dp_uce/images/M0/051_garbage_truck_case051_var00.png` | 264,740 | `c90f06a5390c92d475a44e88fffef6cd59b1b071dae6f2096a01d2cd0a77eab5` |
| `results/modern_flux_dp_uce/images/M0/060_english_springer_case060_var00.png` | 359,383 | `7b206f177d1c5cbf5ffe2c59791ab2194df05c007d6a71f3ea03edf74e9211df` |
| `results/modern_flux_dp_uce/images/M0/061_english_springer_case061_var00.png` | 337,063 | `9a47d61810c361b292a5d8e7191e0dc8845d50a158176dbbed2421a5701b1132` |
| `results/modern_flux_dp_uce/images/M0/070_golf_ball_case070_var00.png` | 306,639 | `3650d2d5ce5ed378b5203ad3fbf26c5718e16e56d0050f89702218239ec85d3f` |
| `results/modern_flux_dp_uce/images/M0/071_golf_ball_case071_var00.png` | 340,574 | `dc03a03da90adbc8611be6ba0e7f1ce962fe87f8e823f2e1bb570f5c4817509c` |
| `results/modern_flux_dp_uce/images/M0/080_parachute_case080_var00.png` | 218,206 | `f5f119378e57899584fa429597f8699368deda5d0aacfc301f4ab5e8a20a91b6` |
| `results/modern_flux_dp_uce/images/M0/081_parachute_case081_var00.png` | 240,697 | `92725801ddf847e1dbb8b86fbf3bf9981d5c91bfaf2258eeeec83795c4c950ff` |
| `results/modern_flux_dp_uce/images/M0/090_french_horn_case090_var00.png` | 318,286 | `b4a0d4d2e1679c28feaf59e254d4692e7f4326eaf3303e92081a36ed9e0df222` |
| `results/modern_flux_dp_uce/images/M0/091_french_horn_case091_var00.png` | 197,917 | `4fb682ffe5b68dc9ed28a281017e41686d126e776685fb7c87b96185db6d7615` |

## Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, DP_A (cassette player -> box)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/images/DP_A/000_cassette_player_case000_var00.png` | 264,646 | `8201b00e50d9c8b1931431c7471dd87077f59ad7a19faa26f2c30ab38371c059` |
| `results/modern_flux_dp_uce/images/DP_A/001_cassette_player_case001_var00.png` | 148,230 | `02fa541c34198aed47f12a8190b5366c6b00bc5f1283c1890cd04ff9809b3550` |
| `results/modern_flux_dp_uce/images/DP_A/010_chain_saw_case010_var00.png` | 458,860 | `02e1282a1465786e7f8b2449648fe32b5e6fc5836d26d1f562a1c3f48e5dcf2b` |
| `results/modern_flux_dp_uce/images/DP_A/011_chain_saw_case011_var00.png` | 397,591 | `a03279c97545a7964fb3327f0900432dcfe3a90b73151b88674af3cd3ab6dd61` |
| `results/modern_flux_dp_uce/images/DP_A/020_church_case020_var00.png` | 480,786 | `ef3bf19ae8c782e28f9d132c3583c759659dbb971f1fe3c28a2565a6bc89fdaf` |
| `results/modern_flux_dp_uce/images/DP_A/021_church_case021_var00.png` | 379,813 | `4fb1d14d5780a400e88ec7875788fa6cc4c0537e6ed72c0014b3159c433459ea` |
| `results/modern_flux_dp_uce/images/DP_A/030_gas_pump_case030_var00.png` | 256,591 | `787cf40e201c02548d10cb901a1ea0b0b7cbb07bd9dfeeac3c19f25ec4b6849e` |
| `results/modern_flux_dp_uce/images/DP_A/031_gas_pump_case031_var00.png` | 321,916 | `fc08ad93992ae22eab8f46b627f45447f42e5873a81a2f821c85aa4d46a1ad3f` |
| `results/modern_flux_dp_uce/images/DP_A/040_tench_case040_var00.png` | 312,428 | `c415860f32bd9e61a32b84e7062daaa472ab7875c9ccc8f40d22026c54f41151` |
| `results/modern_flux_dp_uce/images/DP_A/041_tench_case041_var00.png` | 344,137 | `16b3fab49dfa82b08c63676038fe94e6072119e53f4ab6c67fcee083bee5768f` |
| `results/modern_flux_dp_uce/images/DP_A/050_garbage_truck_case050_var00.png` | 460,964 | `0a7d03d7c0e9d15c7764233d20f745d338011407b65fb403562413914fee5c87` |
| `results/modern_flux_dp_uce/images/DP_A/051_garbage_truck_case051_var00.png` | 254,426 | `1fd3a078ae4abc1175d4c1d1e7c82ab2ab57b888b26e750c9224259bb6b4c6b0` |
| `results/modern_flux_dp_uce/images/DP_A/060_english_springer_case060_var00.png` | 358,449 | `389cf212be2066b0f2be2d6a7addd3d32943a402dd9eaf77767a3da2d67c9565` |
| `results/modern_flux_dp_uce/images/DP_A/061_english_springer_case061_var00.png` | 334,554 | `08a5e99eb08a762ffec39cf8dbf826339207c02a039a311cc0bacbb93ec4028b` |
| `results/modern_flux_dp_uce/images/DP_A/070_golf_ball_case070_var00.png` | 305,912 | `2a958faeac8c1e59773dcfe2b540320e38fba52226268c30688bc3da75ceb5a7` |
| `results/modern_flux_dp_uce/images/DP_A/071_golf_ball_case071_var00.png` | 345,773 | `f34089ab4020d4c4431588300599e5f7d62ab0aef03b36e3ed356eb2d4d24aab` |
| `results/modern_flux_dp_uce/images/DP_A/080_parachute_case080_var00.png` | 218,223 | `e834bccf7a79e5231afbb5e5e43f4d35150175a412f29ec5c505ca573ba129ca` |
| `results/modern_flux_dp_uce/images/DP_A/081_parachute_case081_var00.png` | 243,910 | `f1c152d29dbf5f69f81c9494c5ba5a394488f8e1db967e3bdeeca0438c4a4c06` |
| `results/modern_flux_dp_uce/images/DP_A/090_french_horn_case090_var00.png` | 311,988 | `f43a63948529638dd2610ff840b2b7bd2782a423e0a1f5d60ec44651da6bfd80` |
| `results/modern_flux_dp_uce/images/DP_A/091_french_horn_case091_var00.png` | 193,997 | `ce50e4babdfe87ca9f4de841e671c6202982d662c6bdff7e1a9b5413242a4b10` |

## Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, DP_B (chain saw -> stick)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/images/DP_B/000_cassette_player_case000_var00.png` | 380,441 | `f7fc964e80bca49d58ff1ca8cc7de05461a17cc2471a60c06b275868e627c3a4` |
| `results/modern_flux_dp_uce/images/DP_B/001_cassette_player_case001_var00.png` | 191,594 | `42267a436227bbdc8e698347fda413e5f940a75831b8d770d50fb147f16f30d5` |
| `results/modern_flux_dp_uce/images/DP_B/010_chain_saw_case010_var00.png` | 240,841 | `7b196204f7945ade964e62e80a66c7d73cbc3f0fbdd1ae542a9893a9de2f8906` |
| `results/modern_flux_dp_uce/images/DP_B/011_chain_saw_case011_var00.png` | 205,842 | `9675d47fd794b5011daaf1e577515dedac8accce7312f83f1bbf368af4e2f591` |
| `results/modern_flux_dp_uce/images/DP_B/020_church_case020_var00.png` | 487,007 | `2ec585f6e0edaacc0d2657c5e14e33d06774b7dcfd626bcc6f074c2b1f90bf97` |
| `results/modern_flux_dp_uce/images/DP_B/021_church_case021_var00.png` | 378,017 | `6e25f63e10fe21ce6cc1df288e2dffe8107fa414deae49438a26ac3ece0affe9` |
| `results/modern_flux_dp_uce/images/DP_B/030_gas_pump_case030_var00.png` | 265,150 | `ac6dbeb95fbcc6bc51b2f71bc5fea54fb7df7aeae70a3bd5a6f6c6a11e0af58a` |
| `results/modern_flux_dp_uce/images/DP_B/031_gas_pump_case031_var00.png` | 321,898 | `0c2cc92eeea35edfd1bac49f6f75ac74b33b1d9aef17d931e363f43d5987d645` |
| `results/modern_flux_dp_uce/images/DP_B/040_tench_case040_var00.png` | 318,029 | `e184aef864fa3e94ca3eeae0b09e151245687839b633526d5b2f500867525ef1` |
| `results/modern_flux_dp_uce/images/DP_B/041_tench_case041_var00.png` | 350,583 | `27bc6649fd26acbddf1d36b300aa2494db30350b0d48847ed301d2631a5f039a` |
| `results/modern_flux_dp_uce/images/DP_B/050_garbage_truck_case050_var00.png` | 461,070 | `944a4a3d941b62fdf3eb79df1e92029f827c4ba97658ac57be2f78ce3add4cec` |
| `results/modern_flux_dp_uce/images/DP_B/051_garbage_truck_case051_var00.png` | 254,147 | `f275336458e4eeb2f98d920bf61310b2ba53fdaff90282495ff2af6251fcd06f` |
| `results/modern_flux_dp_uce/images/DP_B/060_english_springer_case060_var00.png` | 357,385 | `c518020eebeb73b0dee8327e6bdbf7e2fb3d9cf90135cf9d6c2795b37a81153f` |
| `results/modern_flux_dp_uce/images/DP_B/061_english_springer_case061_var00.png` | 332,833 | `919d900e1df9f336362def6d7a030e8d9cd8d8194c285f52e1a404dd7e46846b` |
| `results/modern_flux_dp_uce/images/DP_B/070_golf_ball_case070_var00.png` | 304,480 | `43f7e08513cc4ffa15ed732717f27712d60cc2ba3850289f2a84e1d22015b0cc` |
| `results/modern_flux_dp_uce/images/DP_B/071_golf_ball_case071_var00.png` | 348,249 | `372fe228bae35089d3cb65b2ad456c0d5eb1a7d4d72efb7defc1850c8c5294fa` |
| `results/modern_flux_dp_uce/images/DP_B/080_parachute_case080_var00.png` | 219,320 | `07f8c5046e9db93a13b85f857d1a6551fe2c4cc46f9de15e7ab15b86ed929d22` |
| `results/modern_flux_dp_uce/images/DP_B/081_parachute_case081_var00.png` | 243,301 | `d684bff85fb5b91284226e0f39774c8c3f50b0d88bc67f4ba61380fe41a1a67c` |
| `results/modern_flux_dp_uce/images/DP_B/090_french_horn_case090_var00.png` | 318,365 | `8d1a63a0ab9b0cfbf466c58bfc7ab6069c76ab61683fbed1ab52135c7eb1c713` |
| `results/modern_flux_dp_uce/images/DP_B/091_french_horn_case091_var00.png` | 198,885 | `f4bcd1a38cd62b80de9c1e0b49e73fd7b8d2b238febefb2a84c353eaf044b3e1` |

## Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, UCE_A (cassette player -> box)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/images/UCE_A/000_cassette_player_case000_var00.png` | 283,290 | `cd751ab8e62f931a1aa44927988f3921ddea3adea4dc6301f31adee4d1fb6a6c` |
| `results/modern_flux_dp_uce/images/UCE_A/001_cassette_player_case001_var00.png` | 211,437 | `08ae3b91e5b107aea06c9d0a8644cfa0173287d4b5c5a14c82d65560588d5893` |
| `results/modern_flux_dp_uce/images/UCE_A/010_chain_saw_case010_var00.png` | 456,380 | `ed29accee6ac4312d6f206b3d23852e70d5f99bb1451d7a3ce60bc0a067b3ad5` |
| `results/modern_flux_dp_uce/images/UCE_A/011_chain_saw_case011_var00.png` | 374,163 | `fceb5eae8758ef41db396fef187b3da6b10a5e6ea570587e0ff5063a3fcbb12f` |
| `results/modern_flux_dp_uce/images/UCE_A/020_church_case020_var00.png` | 490,219 | `b142b35338c98b1fe703ee5bbc3b5d9205aa5e293f6bb94887ea5fee37862a3b` |
| `results/modern_flux_dp_uce/images/UCE_A/021_church_case021_var00.png` | 373,641 | `b1bbfc60de35adf04be8307c4f0ad10a7265096a6979150747bfac75e65db5ee` |
| `results/modern_flux_dp_uce/images/UCE_A/030_gas_pump_case030_var00.png` | 249,143 | `0c64b68493ac3373fccb3f677817133b0999d13503fed81dd5858898c7400c38` |
| `results/modern_flux_dp_uce/images/UCE_A/031_gas_pump_case031_var00.png` | 330,154 | `7beba1d0b56821945e75521771cb23de5f4028e2cd6b3a79d2edbf6d7f7579fe` |
| `results/modern_flux_dp_uce/images/UCE_A/040_tench_case040_var00.png` | 314,807 | `a1888d19e8991549e0550a1f6719038cb5020d4e73b70dc93097ba4dce655b72` |
| `results/modern_flux_dp_uce/images/UCE_A/041_tench_case041_var00.png` | 348,443 | `856c1acb23b917d5976633e183f9743c01a47cd1f611ee4212fffbc639a2e313` |
| `results/modern_flux_dp_uce/images/UCE_A/050_garbage_truck_case050_var00.png` | 460,752 | `6d8b3cb4742b72460232a678d8eaaf0b6cb4c94724a7a7938e2eb95552b44fd7` |
| `results/modern_flux_dp_uce/images/UCE_A/051_garbage_truck_case051_var00.png` | 246,109 | `72e06c7dd0b0cc030e9475ecfe63a9502f0ef65673cc7c37df76e8239c6ff019` |
| `results/modern_flux_dp_uce/images/UCE_A/060_english_springer_case060_var00.png` | 355,332 | `ec208b98c67d2fd03832b5b4e15ae2d69f84fe782f64854bc279caee9a8eac84` |
| `results/modern_flux_dp_uce/images/UCE_A/061_english_springer_case061_var00.png` | 334,526 | `35e8f56692e03446ffc6d11b75545179f61ffc12090a2c12cca2f74208b03385` |
| `results/modern_flux_dp_uce/images/UCE_A/070_golf_ball_case070_var00.png` | 309,392 | `a824def27d30d12d0159d99c0a5151ae0a3c2c704db29c931fd930ae68bef575` |
| `results/modern_flux_dp_uce/images/UCE_A/071_golf_ball_case071_var00.png` | 342,668 | `e111210538c03d937e6ef7afe00bd6f0aeb9d8f45309cbed14f18936e9f5e54d` |
| `results/modern_flux_dp_uce/images/UCE_A/080_parachute_case080_var00.png` | 218,552 | `48be7d31760ab3471a473ca26cf713257d0eeff097dc66e4d837b83310bc6b94` |
| `results/modern_flux_dp_uce/images/UCE_A/081_parachute_case081_var00.png` | 249,477 | `f65f722c43ce02f91b4e2711c75eaa834f5720865e1f1af26099db30eb7e9dd8` |
| `results/modern_flux_dp_uce/images/UCE_A/090_french_horn_case090_var00.png` | 315,919 | `1b2680321fd96bdae4af53d16dcbf922c8e36db5cfc468bcd3b9a4370524a669` |
| `results/modern_flux_dp_uce/images/UCE_A/091_french_horn_case091_var00.png` | 195,654 | `2dcd1f4b413f5635cdb32cd202b78f335e1596f50d82bfa69747d37d1c314e79` |

## Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, UCE_B (chain saw -> stick)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/images/UCE_B/000_cassette_player_case000_var00.png` | 389,878 | `ec5eff50f03b9ffe47b5efb196c42dd69276193c3eaa4211a87090697a3f6b2e` |
| `results/modern_flux_dp_uce/images/UCE_B/001_cassette_player_case001_var00.png` | 207,772 | `1e1e4b5435aa0b5ffa5d3cc5efe7742ef39a433e52e0cee3faab7108de78d893` |
| `results/modern_flux_dp_uce/images/UCE_B/010_chain_saw_case010_var00.png` | 319,165 | `db49c7a5bc2e50431fa7e6ebbcd936c97975729794f9f611f0933caa9df5a0f0` |
| `results/modern_flux_dp_uce/images/UCE_B/011_chain_saw_case011_var00.png` | 237,073 | `f27979c7aa94244c4b47c49da73f879044c49f12d0e900ad16bb92534b31d701` |
| `results/modern_flux_dp_uce/images/UCE_B/020_church_case020_var00.png` | 487,206 | `7d2fc4d957751c1162a890e1df82666bd9848f87fff921775b82877a4709a11f` |
| `results/modern_flux_dp_uce/images/UCE_B/021_church_case021_var00.png` | 369,600 | `5117923d5259fd057afcffb67f0543d87845e68439054c901f5ee3d10ce55975` |
| `results/modern_flux_dp_uce/images/UCE_B/030_gas_pump_case030_var00.png` | 257,834 | `2829bbc0945f83b8c13f0f5cdb9f78f1cc7f54c18f5c6019ad91e1bebf434a7d` |
| `results/modern_flux_dp_uce/images/UCE_B/031_gas_pump_case031_var00.png` | 312,666 | `1fcdabbdc5c46f2d601001a8ffd4da9de41914c5abd0040eb4f028b3f3a9b4ec` |
| `results/modern_flux_dp_uce/images/UCE_B/040_tench_case040_var00.png` | 314,764 | `3484bfd84a5264d379ff00e3377b3da4b8dab7aa63e8074d824d598c1181a322` |
| `results/modern_flux_dp_uce/images/UCE_B/041_tench_case041_var00.png` | 345,665 | `609cd3e47470668e471e4812b2c8a775a6c32a0ae278ad517b56dd116f3ddcb7` |
| `results/modern_flux_dp_uce/images/UCE_B/050_garbage_truck_case050_var00.png` | 461,520 | `7c49e47b0982de4afeea101017c1860d7d2c4851c66986a4650774be3451baa8` |
| `results/modern_flux_dp_uce/images/UCE_B/051_garbage_truck_case051_var00.png` | 253,749 | `0552676bdc12e73eef7f54cdf4da3476409eceb8fe97362025f840fa5d652066` |
| `results/modern_flux_dp_uce/images/UCE_B/060_english_springer_case060_var00.png` | 356,192 | `3c29f98c9e10574c846e203c5691f052cc9b638d1c299e5139bc0e6eada63408` |
| `results/modern_flux_dp_uce/images/UCE_B/061_english_springer_case061_var00.png` | 336,169 | `685ef4d50fc2cad7bd4630ec0a9967dc4218413ce96b99507e0b7df0b07a4d82` |
| `results/modern_flux_dp_uce/images/UCE_B/070_golf_ball_case070_var00.png` | 306,396 | `4c5ed668e6ecf0cc2b7e36138aa1f368299b022e9d8f789959c4616820dea679` |
| `results/modern_flux_dp_uce/images/UCE_B/071_golf_ball_case071_var00.png` | 340,236 | `c9e77e3d2377a9d6983f783955a424bd2eb231711958f75b2dfa06104b882dfc` |
| `results/modern_flux_dp_uce/images/UCE_B/080_parachute_case080_var00.png` | 216,004 | `ad34f610d975bf069722deef820cae4a1ada426b08974f5e5b0e990d6405e700` |
| `results/modern_flux_dp_uce/images/UCE_B/081_parachute_case081_var00.png` | 247,325 | `b1da49533f9557483f8970f40e4a5ade3fd3136b64062ed2088aa95c6431d5a7` |
| `results/modern_flux_dp_uce/images/UCE_B/090_french_horn_case090_var00.png` | 320,236 | `69fe04e0842071b7bec55d3bd9b85ade45ff291af4b2317916997fb990dea823` |
| `results/modern_flux_dp_uce/images/UCE_B/091_french_horn_case091_var00.png` | 201,771 | `82517bf785f8ebff522673bf9a0f8af489faddd51e37f2f19fc2e7782905b4ac` |

---

**122 files listed**, none absent.
