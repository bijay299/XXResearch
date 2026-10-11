# Published evidence index — exact paths and identities

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `43a94e409d0d` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

**No annotation key, key-derived field, blinding salt or label value
appears in any file listed here.** Keys and salts live only under the
private data root; this was checked by grepping the repository for the
secret values and for any `item_id` line co-occurring with an arm or
training-seed identifier (0 hits each).

Deliberately excluded, unchanged: model weights (digests only, in
`selection.json` and `bridge_check.json`) and generated images (~592 MB;
per-image digests are inside the two archives).

Regenerate: `python scripts/seq/publish_evidence_index.py` — run it LAST,
after every listed file is final.

## Corrected analysis and its preserved predecessors

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/amendment01_evidence/analysis_test.json` | 107,052 | `a79e8fc501bb5222b39fe3d51d59612583c3316ea0e6721c3438d514ad90461b` |
| `results/audit_v1/amendment01_evidence/analysis_test.20261010T205657.superseded.json` | 96,566 | `fc551d50c9fb05c0a964fcddf9f78c7f222dd2263d1cf914a25c2ed07b0f05cd` |
| `results/audit_v1/amendment01_evidence/analysis_test.20261010T205055.superseded.json` | 94,103 | `4ff576991f9214e237e97b5673de92c847bab4acde4c251df50afde490ba9600` |

## Raw detector rows — selection and every interval re-derivable without images or weights

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/amendment01_evidence/development_slots.tar.gz` | 488,420 | `b471593ca50154fee221bfa1a76629f892af063845a4ddef0cb66daa9a095397` |
| `results/audit_v1/amendment01_evidence/development_slots_index.json` | 19,267 | `5c2606ec00ebd926e80f6555baf3f7d8dad3ef37f92c3e4a85f123383d722ed8` |
| `results/audit_v1/amendment01_evidence/frozen_test_slots.tar.gz` | 1,159,093 | `9251276a85975cdc697ff030810df293663d44765ae4d62e68569ca08b9a4821` |
| `results/audit_v1/amendment01_evidence/frozen_test_slots_index.json` | 6,090 | `e495d293bde3d3c219203e3197439754ecc93db22a8343b350a96360364245eb` |

## Decision, bridge, binding, accounting, and the supervisor log

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/amendment01_evidence/selection.json` | 34,028 | `f8d7ea0a3e130afbc7329f91ed6cf3b195522a6b5b70d430a165256855d17c98` |
| `results/audit_v1/amendment01_evidence/bridge_check.json` | 32,423 | `a0d326284b72e1800e7b14cc7dc05db3040c27926c44f44a5a273fb78cabc25e` |
| `results/audit_v1/amendment01_evidence/reference_bindings.json` | 5,598 | `5b6e774cf2981fdc52f537fa7fe31c1bab08e053bfd5520d77deef52c5b82d24` |
| `results/audit_v1/amendment01_evidence/STATUS.json` | 2,100 | `60627a7f10f99c46b696555e8d473eb9f324e75e63aad1a09d7277828278bf2c` |
| `results/audit_v1/amendment01_evidence/unattended.log` | 73,561 | `8ab803ee43796e68f541d1e44a8a1fd59f7246796a1dcdcebb0b585176b396ca` |
| `results/audit_v1/amendment01_evidence/gpu_usage_record.json` | 27,765 | `e07753d1cc18b1a320801e64e62053f6dcfbbdf19e87156fd733da864c20f69c` |
| `results/audit_v1/amendment01_evidence/budget_reconciliation.json` | 881 | `07e6d42c643609ba618ee1ea92706f71896773e28153d36ed80bac0eb079ac64` |

## Manifests

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/amendment01_evidence/MANIFEST.json` | 10,804 | `9421d990ede1a060b85b3c694013fb1ebe94e3def95efb52cb92e87a53ced993` |
| `results/audit_v1/amendment01_evidence/MANIFEST.published_at_7e01fbd.json` | 1,580 | `b723ddb6ac90039242b8f6de69c9bc41c396672500e47e8a24cca7d28c4b2af8` |

## Source-bound test logs and source identity

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/amendment01_checks/CLOSEOUT_SOURCE_HASHES.json` | 5,684 | `2ab0c7a5a7ee8578eb3486d3e297f14e635ea0a984832b8570d6e10762e677b8` |
| `results/audit_v1/amendment01_checks/closeout_test_diag_analysis.log` | 2,218 | `edf123950ba21831af3c877e03651898c5e60b22086fb4306431631308632b31` |
| `results/audit_v1/amendment01_checks/closeout_test_diag_test_conditions.log` | 3,584 | `714f452065566ed9171ae290797ec5ea8d3047ba0e28b6ad2747357a8c4ffdf7` |

## Prescribed paired audit — key-free records only

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/annotation_packet_paired180/README.md` | 8,699 | `1216055d2cbf0627c9ac5cba2c24ba966a131054fbc88e9218f5e588f9067d87` |
| `results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md` | 8,889 | `266228ce93fdd8d0c93d08bb197816ff620486127137cdca21e57fc1d52a588b` |
| `results/audit_v1/annotation_packet_paired180/blinding_audit.log` | 6,970 | `081b9e360d80550efa1dc3930379443c20e0f90d2d6f381a889a293ab55c29e5` |
| `results/audit_v1/annotation_packet_paired180/label_sheet_EMPTY.csv` | 19,449 | `4efb1d175db7ad548e0f556e57fae543dd00e3f139840c8b1951089f2f06a25d` |
| `results/audit_v1/annotation_packet_paired180/packet_manifest.json` | 15,683 | `0338191979f70e48371b8db060258ea3a4aee3efd87a3bfa5def7fd2bd315bc3` |
| `results/audit_v1/annotation_packet_paired180/INSTRUCTIONS.md` | 840 | `482c8888ffe58a53f5f903a0a36bf10eea3c7b5c5a3e2096f0a000157853dbbf` |
| `results/audit_v1/annotation_packet_paired180/overlap_summary.json` | 700 | `efac72605916d542aa01a87adb0ed2ca1347a99c34a8750ad57bdaf37ff06c5b` |

## Write-ups

| path | bytes | sha256 |
|---|---|---|
| `results/audit_v1/AMENDMENT_01_CLOSEOUT.md` | 5,959 | `3bb200885a23ed7f17244310c8da64faf6f9fc594dc0925f1bfe0507e6476a7e` |
| `results/audit_v1/AMENDMENT_01_RESULTS.md` | 24,252 | `56e26489178630e44bca7ef6f9bfc90e7ec7e4c5e212358171b41d4c5835e08c` |
| `results/audit_v1/CLAIM_CORRECTIONS_V5.md` | 13,731 | `9f7b21e3079131ce11f0ced6e770c1d06e946b22c539d0336c838ea9fcc94db8` |
| `results/audit_v1/ARTIFACT_INVENTORY.md` | 17,824 | `033999e2519ee503d6ef9473644e45d8dd693e5124f830be6dd59f1fd52ba261` |
| `docs/research/NEXT_SCREEN_PROPOSAL.md` | 29,904 | `c5e917f302e0dcae984d910ab0bf38cbca5dd282d50391c2c6fa99d6cdb3913a` |

## Code

| path | bytes | sha256 |
|---|---|---|
| `scripts/seq/diag_analysis.py` | 38,196 | `90059e03c132f33a8cc29856bff83671f8e87d2ffd573258651132bb6f0296e0` |
| `scripts/seq/test_diag_analysis.py` | 12,591 | `069c36782d621a5aeea2a57e2728376fc474f21bc14a94d7a126fc4f766b6851` |
| `scripts/seq/test_diag_test_conditions.py` | 16,353 | `6a532df3b861509bea32c1c025583b2344b84d2f07b042426c7a3515fd3c6aea` |
| `scripts/seq/build_paired_annotation_packet.py` | 36,094 | `52184bf18e4df96fd0e85aff7ffc40ce205d8b7f9146398f784d6de0e1b47b7b` |
| `scripts/seq/build_amendment01_evidence.py` | 18,374 | `62fbe6078b1cef76592d4f572f6e0191c2740279b0a6283486ba71c620f7e1fc` |
| `scripts/seq/audit_packet_blinding.py` | 15,869 | `62169d1540e83cc857f53bf9d3ae37880da2630cffd116818db8325a9df06de5` |
| `scripts/seq/publish_evidence_index.py` | 5,596 | `a1c0c8390afea828bf388dcf65bc5a1c4e8e084cdc6f44d82562734e42ffd548` |

---

**38 files listed**, none absent.
