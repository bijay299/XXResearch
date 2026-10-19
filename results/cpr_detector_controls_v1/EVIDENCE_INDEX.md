# EVIDENCE_INDEX — CPR detector-defined controls screen v1 (after Amendment 01)

> **Status: `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED` — attempt CLOSED.**
> Production never started: no training beyond bring-up probes, **zero images
> generated**, no six-arm comparison. CPR improvement remains untested.

**Payload commit (indexed bytes):** `f4c7fa35a96bf3896770e123128aa6811d931398` — the amendment's post-publication verification
commit, whose parent is the amendment payload `58a2534ecb20c171e5603b48b4f850412e857e8c`.

**Lineage:** prospective freeze `3fba0ff3e4eaba65f1f6466249e4dfb939a29a72` → additive
repair-01 freeze `67d8b74d4d571d1641bd840a9adc5f7b4139ca35` → entry-gate payload
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f` (release/tag target) → verification payload
`42614565cd79f911c194d9f7c8d1dea017fbe17c` → historical index child
`9940e9403452e2b382c108c2d4d50bd3d054d707` → **Amendment 01 payload `58a2534ecb20c171e5603b48b4f850412e857e8c`** → verification
record `f4c7fa35a96bf3896770e123128aa6811d931398` → this index-only child. No published commit was amended, reset or force-pushed;
no tag moved.

**Generated:** 2026-10-19T16:58:10Z (UTC), **after** the payload commit, from the bytes Git stored at that
commit (`git show <commit>:<path>`) — not the working tree. Written by an *index-only child*
and **excludes its own hash**. It replaces the 38-row index published at `9940e94`
(6,119 bytes, sha256 `1a30fe3656123631ab3304978bab18407f2d8a54c6d0d6076bae6e4c76da8b5d`),
which remains in Git history and was re-verified row by row against its own payload commit
`42614565…` (38/38 match; `amendment_01_records/historical_index_reverification.json`).
`FINAL_REPORT.md` and `ENFORCEMENT_GAPS.md` differ from that historical index by design;
every protected file (`code/**`, `configs/**`, `artifacts/**`, `RUN_MANIFEST.md`,
`RECONSTRUCTION.md`) is byte-identical to it.

## Start here

[`FINAL_REPORT.md`](FINAL_REPORT.md) (corrected current report) ·
[`AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md`](AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md) ·
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md) ·
[`ENFORCEMENT_GAPS.md`](ENFORCEMENT_GAPS.md)

## Verification

```bash
git show f4c7fa35a96bf3896770e123128aa6811d931398:results/cpr_detector_controls_v1/<path> | sha256sum
```

| # | File | SHA-256 (committed bytes) | Bytes |
|---|---|---|---|
| 1 | `AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md` | `5502974c039c8c3670c2f979bee0bbb61b8afb1e0a74fde869c626a345c11ad9` | 21,949 |
| 2 | `ENFORCEMENT_GAPS.md` | `8f00e715284e5b448c84cebf9e9892d261ec73cf5b96a93b23e3a00116fe481b` | 4,051 |
| 3 | `FINAL_REPORT.md` | `34a8ae21a6b4fb4760ef7a3bbd10f3e3946b831e32980a977bfd40e6bc01c54d` | 15,683 |
| 4 | `PRESERVATION_AND_ACCOUNTING.md` | `a3731ac49367b6ec55c1a99ac0cfbdd96f1b90de464899559553c7833d5f966c` | 12,912 |
| 5 | `RECONSTRUCTION.md` | `118122b8a78eaed09f861b1f6bb42538e47fcf6a650ee2947f686738a176c6b3` | 2,738 |
| 6 | `RUN_MANIFEST.md` | `34299e1ef9984612decc380a496f9bb2c83e5d0863f0dc41dbcbfc5542614e56` | 19,241 |
| 7 | `amendment_01_records/all_releases_pre_amendment.json` | `6174f30cabecb669c2a11e765ba803434f24b1b13ec82a759bae95ca99878df3` | 13,386 |
| 8 | `amendment_01_records/historical_index_reverification.json` | `ac50d8c355650ead62c2aee506c3ecb88475754ccc197b77f1b769c477e475ab` | 7,065 |
| 9 | `amendment_01_records/mechanical_checks.json` | `af49e311aaa523d3b823cebdbe957401d1d58f95bec812afce41d2f81556a0d0` | 14,584 |
| 10 | `amendment_01_records/mechanical_checks.py` | `1e7ac9588f8bf66a2d02e4d19a70da9182e65eaaf9aa3bc75d4d1f5895f60725` | 8,460 |
| 11 | `amendment_01_records/post_publication_verification.json` | `ef46a0051bc384f6ecf95df15d5679df13f8535bf43a93eabe9ab29070ae5002` | 4,167 |
| 12 | `amendment_01_records/post_publication_verification.py` | `43043f1d771f8018254ec96e7103a994a29d1023e1a3b3a32c4138ad803c7cbc` | 5,320 |
| 13 | `amendment_01_records/protected_files_sha256_at_9940e94.txt` | `2262349477b437432124c438c417c9bb0200686fc19586099dcf3f0ab76c1589` | 3,873 |
| 14 | `amendment_01_records/release_417012770_pre_amendment.json` | `1fdc74834fe8dde9c56c4ebcd34ccb96588f586c2ebd4801b0ab5f8534a1cad5` | 2,830 |
| 15 | `amendment_01_records/release_body_amended.md` | `914cc652b6472841b480f00fcafe670b31f94c1cb0472e7eb9a77a9d165e886d` | 4,433 |
| 16 | `amendment_01_records/release_body_original.md` | `9e1556675b8a1c4a3c0ef735de62f3896c5cdb1345f4ccd32e1efcef67c17b84` | 2,400 |
| 17 | `amendment_01_records/reverify_historical_index.py` | `439f48aeac2f4d4b8de7ce50b6f6f5dae0783681a69f81b5ee5f375a4a10abde` | 1,212 |
| 18 | `amendment_01_records/tags_pre_amendment.txt` | `e43ea91a8cbe7bbe6d7000b2fa924108dd51751ade82d335ac83067c3fc812f3` | 571 |
| 19 | `artifacts/allocation_journal.jsonl` | `d81c8ce40c28083c2027c20324fc2c8ebd5b4b4cf094a12b2824db73e952c197` | 2,884 |
| 20 | `artifacts/allocation_ledger.json` | `fc817811471550a3d2286bb1626fd9360141d4c768babcfe64c25d95ea39ad91` | 1,974 |
| 21 | `artifacts/bringup_cpu.json` | `3dade973d40a1193ceaab38d7604a79fbc692df355a36b549ff20dea35706609` | 2,378 |
| 22 | `artifacts/bringup_gpu.json` | `9e667954db374cb4e3f38ad1973b971ad4a4b8f2c9b21fdf9bef8f0cda319eca` | 2,190 |
| 23 | `artifacts/cpu_gate.json` | `1999cb52c29a192fdab4ac79b865aac4d5450abcb77a710b34c0f0579a9c3c32` | 11,821 |
| 24 | `artifacts/qualification.json` | `e0ac5066b9b59139df17c4acc7467c2bb50275a3c27a88a37c562067084a3338` | 31,607 |
| 25 | `artifacts/qualification_raw_detr_resnet50_no_timm.json` | `8a309c48a679e7baa7f72aa770c2d1829b3044fd77e036f4c177b312e90bc708` | 81,201 |
| 26 | `artifacts/qualification_raw_rtdetr_r50vd.json` | `ee264eb73018f0f4db2ece7f2456efb2a2e627b58c9a4bbc8ecd18e7f8df7eef` | 81,276 |
| 27 | `artifacts/release_SHA256SUMS.txt` | `46570247ec28459a2277f83111a1798903a4feff92352cabf66bec8648848694` | 463 |
| 28 | `artifacts/release_verification.json` | `bf3ecac6685c75ce07189363c2269b07f55e345b9692ac375a7b7d79c1613884` | 542 |
| 29 | `artifacts/repair_01/README.md` | `e58cf4ddc3757234611fdebe916f886d4740409ba8ee99eb270dc35139ca043e` | 1,416 |
| 30 | `artifacts/repair_01/cpu_gate.py.before` | `cc351abc77a6f340dade89a9e13593c8c3f8a64d7d94922f8989feb155a10518` | 13,202 |
| 31 | `artifacts/repair_01/cpu_gate_FAILED.json` | `ea176b62f30c7da04363dd5e55d9a6076080987a70c75438302062c0b05b72b5` | 11,669 |
| 32 | `artifacts/repair_01/cpu_gate_PASSED.json` | `1999cb52c29a192fdab4ac79b865aac4d5450abcb77a710b34c0f0579a9c3c32` | 11,821 |
| 33 | `artifacts/repair_01/repair_01.diff` | `61ae6024dab87045e98758ff0a5c266097d8c2b8f02d14a613b504d1fe160b3b` | 1,943 |
| 34 | `code/cprd/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| 35 | `code/cprd/detect.py` | `2102f4884604ce6c632f8a176cf5a64071fd8bf397fed6cbd05d08476646cc9b` | 5,100 |
| 36 | `code/cprd/labels.py` | `4f9f4a2a92cb42be48fc9d8380b70184dd49fac47728f9a290f9ac740a7a928b` | 3,847 |
| 37 | `code/cprd/ledger.py` | `178020179b7337ecaf086570b2bb6a19fce8566e112cd7ba97812fca6c4b6bde` | 13,421 |
| 38 | `code/cprd/losses.py` | `25df738cc8a87e8a82c250313ae0d1c89319f0e850ac0e585c5b1acfcd14ec5d` | 4,729 |
| 39 | `code/gate/bringup.py` | `47a1f27a3d510dc26db1355fcdcb9bb5763e0c406b1c47e68e8c784fc37d2036` | 10,017 |
| 40 | `code/gate/bringup_cpu.py` | `9438229d0d3020e2a05505d152e4815a7c99143157d21ae6869f6c865bdba840` | 8,114 |
| 41 | `code/gate/bringup_gpu.py` | `f8b8b09f3fbb17021e7c34c94fb121d53f294c0dafad895891b297707d734e04` | 8,367 |
| 42 | `code/gate/cpu_gate.py` | `d7627dbf8a85fa013e8ae55e5b4090022c50e3e5110f381d0cb805ab9a006306` | 14,021 |
| 43 | `code/gate/qualify_detectors.py` | `549e1514904a23781b980b4db51f19c56118ac23289786fc9be42c66d129d79f` | 7,204 |
| 44 | `code/patches/P1_P3_erasure_gem.diff` | `6840345c3f3234f85a8116973909daaa5859d42fae94f068c1a87f0c48ec88da` | 2,811 |
| 45 | `code/patches/P2_base_pipeline_wrapper.diff` | `0a31346af174ab43286ae0f654264450bc2c003a596490bf2f26b0ad06c1466d` | 1,014 |
| 46 | `code/patches/P4_P6_optional_imports.diff` | `993c5affb0ddc19c36ddd16747aef1fc72ccdfdc722f26ff8a4b7f7341ff024c` | 4,038 |
| 47 | `code/patches/P7_factory_path.diff` | `6d58be9ea1414abaff6d19535d28070c367f191726956ede523f3146dae6ba6e` | 1,001 |
| 48 | `code/runenv.sh` | `21554093ac3e3e1b9e1447e9284295dff59daedd88d247ab6ba67f2d8ce22df4` | 1,153 |
| 49 | `code/scripts/build_cases.py` | `e648e2b36d3883150f8dd6e098ee8bd290a5b89c41fea10bce084e0215650401` | 8,690 |
| 50 | `code/scripts/select_categories.py` | `4742980f6f776e99210607791050af9da0ecf242486e9601659433ee4843882f` | 9,443 |
| 51 | `configs/cases.json` | `815604dab52bdedeac580f4f433befacf613000f9c16cff0f0900136b34e06e1` | 31,207 |
| 52 | `configs/categories_and_qualification.json` | `acf5bad481205d07e66a79ea1abaa4c27da38d659a017aebf7e1a379719f0bca` | 83,712 |

**52 files indexed**, 619,151 bytes (index excluded).

## Release

Tag **`cpr-detector-controls-v1-entrygate`** (id 417012770), target
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f`, six assets unchanged (IDs, sizes, digests and
timestamps re-read after the amendment; `amendment_01_records/post_publication_verification.json`).
Only the release **body** was edited: a dated notice pinned to `58a2534ecb20c171e5603b48b4f850412e857e8c`, then the original
body verbatim under a historical-body delimiter (original sha256
`9e1556675b8a1c4a3c0ef735de62f3896c5cdb1345f4ccd32e1efcef67c17b84`).

## Scope

No FLUX or detector base weights, no COCO photographs, no credentials and no vendored GEM
source are published. Closed studies, their releases and their tags are untouched.
