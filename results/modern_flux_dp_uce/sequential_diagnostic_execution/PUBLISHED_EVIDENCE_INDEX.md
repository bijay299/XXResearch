# Published evidence index — executed sequential-erasure diagnostic (chain saw -> parachute)

Execution of protocol commit `c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80`.

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `c2c0388da4b8` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

**Deliberately excluded** (repo policy, hashed instead in
`evidence/HOST_ONLY_ARTIFACTS.json`): the four edited `.safetensors`
checkpoints and the full 4,000-image raw generation output, both on
`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/`; plus the pinned
upstream source tree (`czhang024/CVPR_DouleProjection` @
`3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`, MIT, unmodified — cited, not vendored)
and this host's venv/model caches.

Regenerate: `python scripts/seq/publish_modern_flux_dp_uce_execution_evidence_index.py`
— run it LAST.

## Reports and manifest

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/README.md` | 3,290 | `b2c3859ebe5c19f44abbc4979d541921d27f0003e30bad93e490f3ba49be3d9e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/FINAL_REPORT.md` | 12,195 | `4144a0a21e7adf8dca1f110b82a714c3bb0849f48038bcbd868e002d7366aa45` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/EXECUTION_MANIFEST.md` | 8,558 | `5440a1304efde6eaf8c3d56d0d5a40440a3c748bef367adae87169e4a8f93550` |

## Gate evaluation, paired transitions and the C/U case-judgment record

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/ANALYSIS.json` | 8,133 | `5bdc00521dccfeb126ef9d81dc8230da13ddcb091732f6a976596f98e600ee9d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/CASE_JUDGMENTS.json` | 2,417 | `820cc7bc789a43966851abda8d60a864a14ff3b03ff24db7ebab7b98b21c12a4` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/DP_SEQ_chain_saw_transitions.csv` | 10,237 | `3557c978e6a5bff1153e7e125a98f41fe8547db044e628fc9ee8d8c48a9fd50f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/UCE_SEQ_chain_saw_transitions.csv` | 9,204 | `c9a37b544ca525acf16ae7aa2d8b8e40438d9837b6ae7d954cd5e4e572df08e2` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/panel_hashes.csv` | 16,105 | `6951c96dda4cedbe4c9b4030c79fb9e23171594414d95118e41c2488d9618e97` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/HOST_ONLY_ARTIFACTS.json` | 2,011 | `7e297377a68cf83247a622a3e9730f2789a082f2e0b044357838845f891e89c6` |

## Per-image classifier predictions — all 4,000 newly generated images

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/DP_B_direct_per_image.csv` | 237,095 | `de057271f6b9859940701d30d869a3befc7dc1d7e2410d2299225ae5d0d83bee` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/DP_B_direct_summary.json` | 605 | `fc31aa07f02e0a5bc48ec942dbdc32ee0ef40096f1d190e7cb860d213ed0b3aa` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/DP_SEQ_per_image.csv` | 236,491 | `c6328987eff1094d683fbf3d2cf855935a01d5706122f63bd1eb803d7c499dbd` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/DP_SEQ_summary.json` | 596 | `5d08f2bbba6020257716334f05c087dffcf44ab57f5cc764d6377b1d2002ff4d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/UCE_B_direct_per_image.csv` | 237,552 | `530da90a191e3d30acd8c116fb3371982266b76b608ded29ce08b253e11ea1b9` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/UCE_B_direct_summary.json` | 608 | `8db20cd2b3060df9970adc2b20d2a0d910db77763fb776fed3ab233588ff2d31` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/UCE_SEQ_per_image.csv` | 236,455 | `1d94031da020dd211c67381b0ecc3025d183c1b51aa4ff7844adf9eb4a1f8f8c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/per_image/UCE_SEQ_summary.json` | 598 | `99b2da00c7273d515c9121329f414411e73e3c3140bac27eec3d5ce5048ebcf8` |

## Technical verification — parent identity, bitwise load before module copy, keys, dtype, finiteness, save/reload

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/verification/DP_SEQ.verification.json` | 4,583 | `1d2dc5d57adfb5a08d4b2e07e6af813fd5816f257005e41b89f6f3f7ebca09a6` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/verification/UCE_SEQ.verification.json` | 4,266 | `951c643e9b86bc3b486032eed3a629bd49c9ff28e3e8bab469a2cc8157001575` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/verification/integrity_direct.json` | 1,786 | `865aeabdee117a92e3050a01df5e990555d9675194b23cb6952f70f349b039fc` |

## Measured preservation residuals (pre- and post-cast, with numerical context)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/residuals/DP_SEQ.residuals.json` | 19,317 | `c72af819559c4ad427d44548b1108885079b76c55219a285154168aa5b7eecb5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/residuals/UCE_SEQ.residuals.json` | 12,712 | `76cad6192d2de7395dafe26d11c8eba924a64b9e03f3137aaca7c5f2212da899` |

## Code actually used — parent-loading adapter, residual logger, classifier wrapper, analysis, panel builder

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/code/analyze.py` | 8,397 | `c995470f15bc5c190a7628a37852ff211b8a1567b06535f950eac8a6f154c64d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/code/build_panel.py` | 3,571 | `1a098213d4cca2c0152ad1928dacf5b4186eb395c3404785ec0d2aa86e19e43e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/code/classify_endpoint.py` | 3,137 | `ad44eef7f25b3ce9b3c524b4e5bd645b931bc1a9be7f956925fb3c810a46fd53` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/code/residuals.py` | 8,721 | `8af57966b21dc6a0658f6fed6836bac9a118eeb5deea9c03edc9de184dc10106` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/evidence/code/run_edit_with_parent.py` | 14,071 | `c46f2a364fee9b79a678b68167f3c5bf56fdeb59595bf1ab46d0413db2bd501a` |

## Visual evidence — readable per-class comparison grids (M0 | A-parent | B-direct | SEQ)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_cassette_player.png` | 758,234 | `af31278a11f78ae00d5a5ad8379baa42b49d8ef2095674c7242404f5bbccfd36` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_chain_saw.png` | 949,944 | `d63a89df9204f4ee1b27ab1b4521e69811d94b140431334c6e3ad25a77314b91` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_church.png` | 1,080,494 | `54793681854abc75356fb3bde1bf868e04f8b13e414ee0a3d4280458d509abce` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_english_springer.png` | 964,612 | `fa70f3a6ad486291f442cffd5c30410350ac026a49220e66df86303eb164204a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_french_horn.png` | 730,320 | `29aa4b9ba4c4ffbabb419f29fbe282c1275a9794f52b3072e437d33a57f689b0` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_garbage_truck.png` | 960,730 | `f3f0118b2a546c72fcbbe548f6d3eb9211be229c63a1aa9c73a5d9179e060d48` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_gas_pump.png` | 893,835 | `e69ed00efe20b03f56855bfacb544d05f9a40f7de1aea72c81ecd2ff533ba2b1` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_golf_ball.png` | 959,289 | `019dd39355a58b6e6c0628db34d986a2dc21f6c36673daa8ec13a7fa5360df08` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_parachute.png` | 774,467 | `ee9a255b23a24759d35a8ba75dcf5c1977a7ef0f610273cc833e7b74a17a6152` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/DP_tench.png` | 911,462 | `25d0cab1d92dc98a5053193a8d3b87ff463f9b970457dd9ef8aa740a26e64858` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_cassette_player.png` | 767,435 | `e59a6fa0564620b233b1692e9d84911d4e19999daa6f868fcd55e23f5e7e0352` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_chain_saw.png` | 1,025,152 | `c86c65ac68c6c793818ee7b3711e987d0802a1d4ed38f32967c419937efb6902` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_church.png` | 1,081,930 | `1153859d88af12d6be4c7726f261991996241e3231947e2d006112fe86af425c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_english_springer.png` | 964,278 | `f26e2d16e1a0ffd1720eda815411aa6da6f3cc4af5df557f0934a0a9b8271bd9` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_french_horn.png` | 730,702 | `1832f641d89c6cba23a2f9f97b2583e73d971bcff7698a4c11b5328d0e2f62bc` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_garbage_truck.png` | 958,161 | `138ea64d2575d69b48708bf13283687f8f502755dd08eecba667a3252fde1d51` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_gas_pump.png` | 841,859 | `bdd7a4f20bb72c31c01c7c2ad26e1b63fe67d2bec79b50b1006c15dcf17b89d8` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_golf_ball.png` | 950,361 | `2c6033892e1e68cbff60684d72a76bdf84c7bf47a901db358360b2de6ade1094` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_parachute.png` | 757,372 | `6a38b835f6bbc7fff91b5c0d9a5c6fc419a8312f579608e374d5d0e6374d420c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/grids/UCE_tench.png` | 906,981 | `b7642aefc960176416d1bc06fef11a47edd02ca759d480c0e60822eec121e14b` |

## Visual evidence — frozen 20-case rendering panel at full 512x512 resolution, all seven compared endpoints

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/000_cassette_player_case000_var00.png` | 380,441 | `f7fc964e80bca49d58ff1ca8cc7de05461a17cc2471a60c06b275868e627c3a4` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/001_cassette_player_case001_var00.png` | 191,594 | `42267a436227bbdc8e698347fda413e5f940a75831b8d770d50fb147f16f30d5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/010_chain_saw_case010_var00.png` | 240,841 | `7b196204f7945ade964e62e80a66c7d73cbc3f0fbdd1ae542a9893a9de2f8906` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/011_chain_saw_case011_var00.png` | 205,842 | `9675d47fd794b5011daaf1e577515dedac8accce7312f83f1bbf368af4e2f591` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/020_church_case020_var00.png` | 487,007 | `2ec585f6e0edaacc0d2657c5e14e33d06774b7dcfd626bcc6f074c2b1f90bf97` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/021_church_case021_var00.png` | 378,017 | `6e25f63e10fe21ce6cc1df288e2dffe8107fa414deae49438a26ac3ece0affe9` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/030_gas_pump_case030_var00.png` | 265,150 | `ac6dbeb95fbcc6bc51b2f71bc5fea54fb7df7aeae70a3bd5a6f6c6a11e0af58a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/031_gas_pump_case031_var00.png` | 321,898 | `0c2cc92eeea35edfd1bac49f6f75ac74b33b1d9aef17d931e363f43d5987d645` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/040_tench_case040_var00.png` | 318,029 | `e184aef864fa3e94ca3eeae0b09e151245687839b633526d5b2f500867525ef1` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/041_tench_case041_var00.png` | 350,583 | `27bc6649fd26acbddf1d36b300aa2494db30350b0d48847ed301d2631a5f039a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/050_garbage_truck_case050_var00.png` | 461,070 | `944a4a3d941b62fdf3eb79df1e92029f827c4ba97658ac57be2f78ce3add4cec` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/051_garbage_truck_case051_var00.png` | 254,147 | `f275336458e4eeb2f98d920bf61310b2ba53fdaff90282495ff2af6251fcd06f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/060_english_springer_case060_var00.png` | 357,385 | `c518020eebeb73b0dee8327e6bdbf7e2fb3d9cf90135cf9d6c2795b37a81153f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/061_english_springer_case061_var00.png` | 332,833 | `919d900e1df9f336362def6d7a030e8d9cd8d8194c285f52e1a404dd7e46846b` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/070_golf_ball_case070_var00.png` | 304,480 | `43f7e08513cc4ffa15ed732717f27712d60cc2ba3850289f2a84e1d22015b0cc` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/071_golf_ball_case071_var00.png` | 348,249 | `372fe228bae35089d3cb65b2ad456c0d5eb1a7d4d72efb7defc1850c8c5294fa` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/080_parachute_case080_var00.png` | 219,320 | `07f8c5046e9db93a13b85f857d1a6551fe2c4cc46f9de15e7ab15b86ed929d22` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/081_parachute_case081_var00.png` | 243,301 | `d684bff85fb5b91284226e0f39774c8c3f50b0d88bc67f4ba61380fe41a1a67c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/090_french_horn_case090_var00.png` | 318,365 | `8d1a63a0ab9b0cfbf466c58bfc7ab6069c76ab61683fbed1ab52135c7eb1c713` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_A_parent/091_french_horn_case091_var00.png` | 198,885 | `f4bcd1a38cd62b80de9c1e0b49e73fd7b8d2b238febefb2a84c353eaf044b3e1` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/000_cassette_player_case000_var00.png` | 381,625 | `bdad7753a8800499df0079008e870a68310f74c55263eee7bcefc20d4940dbfa` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/001_cassette_player_case001_var00.png` | 194,817 | `4a4ec5986d1a255b73c88293acf26e1ceb26afffe7264624068cd2261fa584fa` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/010_chain_saw_case010_var00.png` | 457,870 | `83f86918f67840786f8b90996b7ed84ad92680fdb801fa05d33a1dc24b6d693b` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/011_chain_saw_case011_var00.png` | 382,169 | `1fe64a41a00597f1fb65876c7d4549ab3fc2915006a17e626cd352aaf7c74f6a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/020_church_case020_var00.png` | 475,023 | `7b9bc9f8ee242a9c07befd986f5a4c0233c453411d4fad4babac1250eca75e40` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/021_church_case021_var00.png` | 387,277 | `bfbb865df3705856e869298c0402495281ad00eea00503704ddc794fac64de6f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/030_gas_pump_case030_var00.png` | 309,764 | `20561a10e6b235a50574f945bb2ae1a114abdf3c332a838e0525b2f6a8174234` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/031_gas_pump_case031_var00.png` | 339,128 | `cb86385640990ad4b9edb7c3395da4307255f7680469c97fdc6518f80e16773f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/040_tench_case040_var00.png` | 309,036 | `606df0151b10fcc15a925c7dd96d09be5dce444c115a181b47ea2956fddcea2a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/041_tench_case041_var00.png` | 343,766 | `50f0e7b642ea043b2225226d2a93a5bb1802ba4b537715648773c403f3613019` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/050_garbage_truck_case050_var00.png` | 462,268 | `f5d5da14aa9a9b389ea626c635b45fa32f6a37b625186e6dac8879810d4b7ad8` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/051_garbage_truck_case051_var00.png` | 254,535 | `4fd56809d6acd049602479c6d4ea9dddfce9fe8c6894fe7ba2be24e7b62b9cfc` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/060_english_springer_case060_var00.png` | 360,180 | `976887457badd8f08345b003f650a4478d5b4828a57a1061ec2cd074f450ed44` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/061_english_springer_case061_var00.png` | 333,633 | `34cc9ac1bff8778c9010eacf0c0a69bf186334903e0e2b9fb37a5db18c980881` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/070_golf_ball_case070_var00.png` | 307,933 | `bd8c826154279b5c0ebeb2f3f124240d61d3e5bfee68bb91957541c6f54a1182` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/071_golf_ball_case071_var00.png` | 347,814 | `04580dc25cb5c8892e64d3736235b1652fa2a4696c890c4bed7d4baa39b65700` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/080_parachute_case080_var00.png` | 402,541 | `9553b1872db5b38a0246afa57a5c36e5dfcfcf0ae835e19274a17df10648c122` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/081_parachute_case081_var00.png` | 416,977 | `95b059c6fb6229aefed14111eb178200e9d6f7ec0cd09cee5b75b57caeab0799` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/090_french_horn_case090_var00.png` | 312,972 | `3a08163d50069cd92fdca1bfddb77e58c9a47ec9cd8d3206eb56a564e8a3b5cd` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_B_direct/091_french_horn_case091_var00.png` | 204,246 | `9cac32ffb7243215513f8d4a58fae7118199eedb023fe611e17525a954fabad2` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/000_cassette_player_case000_var00.png` | 378,300 | `ae4983cfd4d07c6969625958c5c515b3e2910ddc5c62b86979949c3f987e99cf` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/001_cassette_player_case001_var00.png` | 193,372 | `818a9da0b3099b2591e7549975757a1d017bd34f055b4c14950700c8803a7d16` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/010_chain_saw_case010_var00.png` | 241,010 | `656ad12b83f5a14ea7e4895589e5e3e3986b935e9f931b5346c8947f33e3fee7` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/011_chain_saw_case011_var00.png` | 205,665 | `8cdf0413d376415ab19ec4a483a786c3e8cc39cabd6f890729be8ab06eaf6acd` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/020_church_case020_var00.png` | 481,010 | `90662c11097622d0bf22cd0639d13600e98be203c6a3d393f3e9275b580841e7` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/021_church_case021_var00.png` | 389,647 | `5838b15e99699de210a3b5502f8a96a645f2aebc92652d9448d639aefe26928f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/030_gas_pump_case030_var00.png` | 296,591 | `87ff196d53bc76816b070429d4a844642fd3edab459a500d4ceffa55a9ba57d5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/031_gas_pump_case031_var00.png` | 326,128 | `dd7cbddbfff5f4a2ebdea23492e18f2d0431e6d226f352f17a13b0936df5565e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/040_tench_case040_var00.png` | 313,005 | `4e83898548f15a2b0c31aec392151a76f4a6fc68723186827b812e8045a321f7` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/041_tench_case041_var00.png` | 347,588 | `faa5b6ef350daa5c022467cacea20a7d6dc4b619657d00aff36939ba4dacba1f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/050_garbage_truck_case050_var00.png` | 462,784 | `9659d2f4c7f10a7c223fdba436ccd1389fdc13ae833c96ab3865e1ba58dbb15c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/051_garbage_truck_case051_var00.png` | 242,803 | `169bbe066c3d3b36a16f9cebd3241956a60849a0cbab38957479c4fd743576c3` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/060_english_springer_case060_var00.png` | 358,268 | `ac2c57a2a2c4ed203dc63a802d279f0a1baa41b778c023ad3b54d553292e8ccb` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/061_english_springer_case061_var00.png` | 333,041 | `18b360e15b7a5bef5e665e33acb7a3837ea4450b27dd073a113845adfc5a3286` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/070_golf_ball_case070_var00.png` | 302,971 | `7e4a23eb69dfcea9e12e11a2994247eb2abf1835dee88c1edc49f2b7891208e9` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/071_golf_ball_case071_var00.png` | 348,541 | `7b35dd8f206501a5e0a1d32e3640eb26d5be4114b953f2afb184f03b4375b7e5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/080_parachute_case080_var00.png` | 413,266 | `33f6a13d96bd50b3d2d55ef59b3f834265ce5f6113da3bdf74f90eefa1551f87` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/081_parachute_case081_var00.png` | 428,541 | `3e5d97873d44da9164a84c451c90c0e07fc4218f60f1e08b0dd6d575da2fb23a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/090_french_horn_case090_var00.png` | 312,112 | `433b0de2d9a6cd8a08320133fe16eb2195f9e2aae4df837e90edfac3ab12923d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/DP_SEQ/091_french_horn_case091_var00.png` | 201,001 | `17990dff45cf093aa4593c50ef0dd8c6ab2118d2b477c845252c2e30675a1173` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/000_cassette_player_case000_var00.png` | 387,770 | `deb13e3d8943da4153e79160399f2c3464d1eacecb5a02d5850e1b16b0adcf9e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/001_cassette_player_case001_var00.png` | 193,207 | `994e3cb6d447f1530cb15dcb5dc29e642bf4eceea04352cc68fce770b92dc1c9` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/010_chain_saw_case010_var00.png` | 453,929 | `ad720b86a09ca22cebba7971a05348bb4b263ff7b9d017ba50d498c8b88c6469` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/011_chain_saw_case011_var00.png` | 365,225 | `6a58916a4c1e22e562087e232329917a33ae8c81ad2f9316a79481a2f4502406` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/020_church_case020_var00.png` | 481,200 | `f7318881913c3740a977620f09b267b3d2d33ccb7a9eb9252ad2f77d8684a570` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/021_church_case021_var00.png` | 381,019 | `43099c38bc2c9c01d8ebc7964f80aafa75ff33bf61ad34ab4c5bf15875efe02f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/030_gas_pump_case030_var00.png` | 297,521 | `66d9053862415a9a03f1fe94502318b5ef6ed62f807bd1374fef79a5b635655a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/031_gas_pump_case031_var00.png` | 322,412 | `ea2e13d7898b9c52860b28dee4721c49ab028a8e68a1609360b3aeb488101bd8` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/040_tench_case040_var00.png` | 306,408 | `ca72563b632be4eb49fb3d391bf38b9d67ae0788115990a8a9beb8b60105fe41` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/041_tench_case041_var00.png` | 343,279 | `504d9cf7eb50129d8c3975cd06b04f00d49099b5d441bbbd4577efa9938e379f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/050_garbage_truck_case050_var00.png` | 462,527 | `808dccdaf374a03c3e7c5ccf9da4818f17016b1706441bbe163ba40c2439867e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/051_garbage_truck_case051_var00.png` | 264,740 | `c90f06a5390c92d475a44e88fffef6cd59b1b071dae6f2096a01d2cd0a77eab5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/060_english_springer_case060_var00.png` | 359,383 | `7b206f177d1c5cbf5ffe2c59791ab2194df05c007d6a71f3ea03edf74e9211df` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/061_english_springer_case061_var00.png` | 337,063 | `9a47d61810c361b292a5d8e7191e0dc8845d50a158176dbbed2421a5701b1132` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/070_golf_ball_case070_var00.png` | 306,639 | `3650d2d5ce5ed378b5203ad3fbf26c5718e16e56d0050f89702218239ec85d3f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/071_golf_ball_case071_var00.png` | 340,574 | `dc03a03da90adbc8611be6ba0e7f1ce962fe87f8e823f2e1bb570f5c4817509c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/080_parachute_case080_var00.png` | 218,206 | `f5f119378e57899584fa429597f8699368deda5d0aacfc301f4ab5e8a20a91b6` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/081_parachute_case081_var00.png` | 240,697 | `92725801ddf847e1dbb8b86fbf3bf9981d5c91bfaf2258eeeec83795c4c950ff` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/090_french_horn_case090_var00.png` | 318,286 | `b4a0d4d2e1679c28feaf59e254d4692e7f4326eaf3303e92081a36ed9e0df222` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/M0/091_french_horn_case091_var00.png` | 197,917 | `4fb682ffe5b68dc9ed28a281017e41686d126e776685fb7c87b96185db6d7615` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/000_cassette_player_case000_var00.png` | 389,878 | `ec5eff50f03b9ffe47b5efb196c42dd69276193c3eaa4211a87090697a3f6b2e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/001_cassette_player_case001_var00.png` | 207,772 | `1e1e4b5435aa0b5ffa5d3cc5efe7742ef39a433e52e0cee3faab7108de78d893` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/010_chain_saw_case010_var00.png` | 319,165 | `db49c7a5bc2e50431fa7e6ebbcd936c97975729794f9f611f0933caa9df5a0f0` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/011_chain_saw_case011_var00.png` | 237,073 | `f27979c7aa94244c4b47c49da73f879044c49f12d0e900ad16bb92534b31d701` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/020_church_case020_var00.png` | 487,206 | `7d2fc4d957751c1162a890e1df82666bd9848f87fff921775b82877a4709a11f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/021_church_case021_var00.png` | 369,600 | `5117923d5259fd057afcffb67f0543d87845e68439054c901f5ee3d10ce55975` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/030_gas_pump_case030_var00.png` | 257,834 | `2829bbc0945f83b8c13f0f5cdb9f78f1cc7f54c18f5c6019ad91e1bebf434a7d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/031_gas_pump_case031_var00.png` | 312,666 | `1fcdabbdc5c46f2d601001a8ffd4da9de41914c5abd0040eb4f028b3f3a9b4ec` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/040_tench_case040_var00.png` | 314,764 | `3484bfd84a5264d379ff00e3377b3da4b8dab7aa63e8074d824d598c1181a322` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/041_tench_case041_var00.png` | 345,665 | `609cd3e47470668e471e4812b2c8a775a6c32a0ae278ad517b56dd116f3ddcb7` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/050_garbage_truck_case050_var00.png` | 461,520 | `7c49e47b0982de4afeea101017c1860d7d2c4851c66986a4650774be3451baa8` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/051_garbage_truck_case051_var00.png` | 253,749 | `0552676bdc12e73eef7f54cdf4da3476409eceb8fe97362025f840fa5d652066` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/060_english_springer_case060_var00.png` | 356,192 | `3c29f98c9e10574c846e203c5691f052cc9b638d1c299e5139bc0e6eada63408` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/061_english_springer_case061_var00.png` | 336,169 | `685ef4d50fc2cad7bd4630ec0a9967dc4218413ce96b99507e0b7df0b07a4d82` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/070_golf_ball_case070_var00.png` | 306,396 | `4c5ed668e6ecf0cc2b7e36138aa1f368299b022e9d8f789959c4616820dea679` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/071_golf_ball_case071_var00.png` | 340,236 | `c9e77e3d2377a9d6983f783955a424bd2eb231711958f75b2dfa06104b882dfc` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/080_parachute_case080_var00.png` | 216,004 | `ad34f610d975bf069722deef820cae4a1ada426b08974f5e5b0e990d6405e700` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/081_parachute_case081_var00.png` | 247,325 | `b1da49533f9557483f8970f40e4a5ade3fd3136b64062ed2088aa95c6431d5a7` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/090_french_horn_case090_var00.png` | 320,236 | `69fe04e0842071b7bec55d3bd9b85ade45ff291af4b2317916997fb990dea823` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_A_parent/091_french_horn_case091_var00.png` | 201,771 | `82517bf785f8ebff522673bf9a0f8af489faddd51e37f2f19fc2e7782905b4ac` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/000_cassette_player_case000_var00.png` | 389,155 | `3190b6ad2e2dbafe4a84942631d9b0ee85ce72fc2fde1fe5e7401655ed322a04` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/001_cassette_player_case001_var00.png` | 205,888 | `434c848b4f8b536290d9308575013ed02f74ddc931dad8648e88cd506e925899` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/010_chain_saw_case010_var00.png` | 452,127 | `4cc6ab4344390030266c259a70300eebeceb176329e82f72136351805f6d22ab` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/011_chain_saw_case011_var00.png` | 382,695 | `79e905443205eb888c2e2d62ef9b9a239a7994189d33ccbb17da31392fcb74a5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/020_church_case020_var00.png` | 494,636 | `7638b41e2c74b38240eb6158ca47206f149a4bc665b529e345d0021569771172` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/021_church_case021_var00.png` | 385,214 | `209221499dd085f79d53deea1c04b7b384d28a3e3fc570922c5c5e97a4a14cd2` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/030_gas_pump_case030_var00.png` | 260,752 | `8fb7008a01c9c20eba929afdc4f2bf87f452055b5ed220011a83b95847304b3c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/031_gas_pump_case031_var00.png` | 314,988 | `02ed9ad80dfb97457a4b7aabf6afec8fd329c8c2fa1b0c609a63bc48d6564c6f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/040_tench_case040_var00.png` | 308,942 | `ea359964da4f9d9d27f0e0a11ed75cb710b7d4a404b37aa9dd1875a9ef411c4d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/041_tench_case041_var00.png` | 356,322 | `79ee3b734d75c6ffbb0ccd4d5e740092c57b54a6c9c40db319fa7537fdd68dbd` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/050_garbage_truck_case050_var00.png` | 468,276 | `f9150128601ec3f1e5eff3365bbbae6a0eb91021daf4e9b856e3695f0d2935c5` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/051_garbage_truck_case051_var00.png` | 248,440 | `757fd336f726ac0e72ec71519746bb736c815ad116ec826eaa49ec0c1ff5474c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/060_english_springer_case060_var00.png` | 357,278 | `b603784df636220b39db36a6bccad715d0ea832bd24fcddcaae8e56fdff5b432` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/061_english_springer_case061_var00.png` | 335,418 | `b40598bbb7760413ff0ebf757b090a7d5b25609a008e4a45aac994fe7b21d8a3` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/070_golf_ball_case070_var00.png` | 304,991 | `3ed802fee6dca4a59468399a540f84298a4ff8a2cf26270069d2796716b665cf` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/071_golf_ball_case071_var00.png` | 341,122 | `feaaed153aa1f918cb1a2628b5fb2deec9609945b0cadcbeb8cb702fffa961a6` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/080_parachute_case080_var00.png` | 366,954 | `ff3469f69cf4971b2eb3f87b0c93af633e0fec73a1981dd62f30c4dc0d204448` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/081_parachute_case081_var00.png` | 394,575 | `295ddf066875730ebae3b1311bc4bdfd4490a4b96a800d746feb60fb2e8bb63e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/090_french_horn_case090_var00.png` | 318,311 | `4e44fc3845cc59203a3f20672f4aca9a59fb44975c7e3c3a4b1b08af48c8e52f` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_B_direct/091_french_horn_case091_var00.png` | 199,216 | `511bf1d8fe6aa7f892f7d6210d53fdf86dd49d7cdbe566cf2d875e2f2805bd05` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/000_cassette_player_case000_var00.png` | 366,832 | `a254ea15c5a5a3e38ac51dce81eccd470d77430a34028f143922663996be78c0` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/001_cassette_player_case001_var00.png` | 206,188 | `63f82ff806f6e9d3e429a95bbe7e3258c5655c42b2214e4001546d6787c84df9` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/010_chain_saw_case010_var00.png` | 314,354 | `3cf1b6fc159329b7ab1d614bb348ec619517a2b1c332f331319b29595968ef7e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/011_chain_saw_case011_var00.png` | 224,664 | `ff4f83192e4f90ecda46327bc59bd1b4d8e0d9d44d0542dd9452f1cda0e0c739` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/020_church_case020_var00.png` | 490,390 | `370eddf582051956c926e97b85cd0eff2eb2a7913f57ec1a298a3c4a48cdc095` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/021_church_case021_var00.png` | 390,028 | `3f5e1f7296e8f3ce2f6942473449c3b560667c9dcbaaf86f6bf54a16fc8a5741` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/030_gas_pump_case030_var00.png` | 265,404 | `4640d83e12ef77f727f208d3cfb7def44a40c7448d9adb0c8f38f83a5c76ba80` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/031_gas_pump_case031_var00.png` | 323,974 | `043ea401877b0388b82ca1d87f434d8b0f84cde0c756e8de78aac1521d4c2610` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/040_tench_case040_var00.png` | 317,785 | `2112c9053ab7030edcc60038edd731d2feecacabb3e759fd667cbcb04295325c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/041_tench_case041_var00.png` | 340,998 | `83a27ca62f0fbde4d3075c54db165b98b38969a2f9f97e1fcc3462dc0470c112` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/050_garbage_truck_case050_var00.png` | 466,680 | `85379eb31d12798cc723b0020ecd02cc81f19407f039bc9d491aee7fd8f7085e` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/051_garbage_truck_case051_var00.png` | 237,607 | `996e455090084fef4eaa7e87170b2a14c7e8353105a80a669ce979d2b43ae4d4` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/060_english_springer_case060_var00.png` | 351,903 | `8922c51d17a7556fcea0ecde6c32a92d3d87f08de2b9383010a9f080eebb7257` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/061_english_springer_case061_var00.png` | 343,759 | `2ca37c98f6425551b8304ac2f2cc47729ffb3e101a361eb64073691f3227925b` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/070_golf_ball_case070_var00.png` | 303,418 | `d5a9573f755b187f59fce1e5acc47a47dcddc2e65d754ec49f66cd4afa44290c` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/071_golf_ball_case071_var00.png` | 341,874 | `7be6ff2c18e79aa636fd0ebcef9ba044d2beb651c5056a8fcf82048307ef826a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/080_parachute_case080_var00.png` | 336,689 | `1ad34d27ca63f8d6780d26168a6898d8f0c4f61793a85b26299e80d73e921c48` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/081_parachute_case081_var00.png` | 409,722 | `c363c3ba5c2ebefb99ccf3979194a31a613a0bffbc073f7fbc25061e71288fce` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/090_french_horn_case090_var00.png` | 319,084 | `805f41fb69df17f19b84e8149a04d828b90710b8a4c842b9716542dc2e9a5078` |
| `results/modern_flux_dp_uce/sequential_diagnostic_execution/images/full_resolution/UCE_SEQ/091_french_horn_case091_var00.png` | 196,512 | `24549c2a4a9f0c53b1d243d12ca8cdc8d6393e9865f97274e5def93fd4a19b57` |

---

**187 files listed**, none absent.
