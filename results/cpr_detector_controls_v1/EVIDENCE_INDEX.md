# EVIDENCE_INDEX — CPR detector-defined controls screen v1

> **Terminal status: `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`.**
> Production never started: no training beyond bring-up probes, **zero images
> generated**, no six-arm comparison. CPR remains untested.

**Payload commit:** `42614565cd79f911c194d9f7c8d1dea017fbe17c`

**Freeze lineage:** prospective freeze `3fba0ff3e4eaba65f1f6466249e4dfb939a29a72`
→ additive repair-01 freeze `67d8b74d4d571d1641bd840a9adc5f7b4139ca35`
→ payload `42614565cd79f911c194d9f7c8d1dea017fbe17c`. No freeze was backdated or overwritten.

**Generated:** 2026-10-19T07:28:24Z (UTC), **after** the payload commit, from the bytes Git
stored at that commit (`git show <commit>:<path>`) — not the working tree.
Written by an *index-only child* and **excludes its own hash**.

## Verification

```bash
git show 42614565cd79f911c194d9f7c8d1dea017fbe17c:results/cpr_detector_controls_v1/<path> | sha256sum
```

| # | File | SHA-256 (committed bytes) | Bytes |
|---|---|---|---|
| 1 | `ENFORCEMENT_GAPS.md` | `ac25f3e064417f899260e9532f936a88c71150f0b5d97dc905fe4031009edc65` | 1,126 |
| 2 | `FINAL_REPORT.md` | `bfd55931b18a6f9d190f06d14540579964c86038852c249e900fdde28b6a239c` | 15,959 |
| 3 | `RECONSTRUCTION.md` | `118122b8a78eaed09f861b1f6bb42538e47fcf6a650ee2947f686738a176c6b3` | 2,738 |
| 4 | `RUN_MANIFEST.md` | `34299e1ef9984612decc380a496f9bb2c83e5d0863f0dc41dbcbfc5542614e56` | 19,241 |
| 5 | `artifacts/allocation_journal.jsonl` | `d81c8ce40c28083c2027c20324fc2c8ebd5b4b4cf094a12b2824db73e952c197` | 2,884 |
| 6 | `artifacts/allocation_ledger.json` | `fc817811471550a3d2286bb1626fd9360141d4c768babcfe64c25d95ea39ad91` | 1,974 |
| 7 | `artifacts/bringup_cpu.json` | `3dade973d40a1193ceaab38d7604a79fbc692df355a36b549ff20dea35706609` | 2,378 |
| 8 | `artifacts/bringup_gpu.json` | `9e667954db374cb4e3f38ad1973b971ad4a4b8f2c9b21fdf9bef8f0cda319eca` | 2,190 |
| 9 | `artifacts/cpu_gate.json` | `1999cb52c29a192fdab4ac79b865aac4d5450abcb77a710b34c0f0579a9c3c32` | 11,821 |
| 10 | `artifacts/qualification.json` | `e0ac5066b9b59139df17c4acc7467c2bb50275a3c27a88a37c562067084a3338` | 31,607 |
| 11 | `artifacts/qualification_raw_detr_resnet50_no_timm.json` | `8a309c48a679e7baa7f72aa770c2d1829b3044fd77e036f4c177b312e90bc708` | 81,201 |
| 12 | `artifacts/qualification_raw_rtdetr_r50vd.json` | `ee264eb73018f0f4db2ece7f2456efb2a2e627b58c9a4bbc8ecd18e7f8df7eef` | 81,276 |
| 13 | `artifacts/release_SHA256SUMS.txt` | `46570247ec28459a2277f83111a1798903a4feff92352cabf66bec8648848694` | 463 |
| 14 | `artifacts/release_verification.json` | `bf3ecac6685c75ce07189363c2269b07f55e345b9692ac375a7b7d79c1613884` | 542 |
| 15 | `artifacts/repair_01/README.md` | `e58cf4ddc3757234611fdebe916f886d4740409ba8ee99eb270dc35139ca043e` | 1,416 |
| 16 | `artifacts/repair_01/cpu_gate.py.before` | `cc351abc77a6f340dade89a9e13593c8c3f8a64d7d94922f8989feb155a10518` | 13,202 |
| 17 | `artifacts/repair_01/cpu_gate_FAILED.json` | `ea176b62f30c7da04363dd5e55d9a6076080987a70c75438302062c0b05b72b5` | 11,669 |
| 18 | `artifacts/repair_01/cpu_gate_PASSED.json` | `1999cb52c29a192fdab4ac79b865aac4d5450abcb77a710b34c0f0579a9c3c32` | 11,821 |
| 19 | `artifacts/repair_01/repair_01.diff` | `61ae6024dab87045e98758ff0a5c266097d8c2b8f02d14a613b504d1fe160b3b` | 1,943 |
| 20 | `code/cprd/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| 21 | `code/cprd/detect.py` | `2102f4884604ce6c632f8a176cf5a64071fd8bf397fed6cbd05d08476646cc9b` | 5,100 |
| 22 | `code/cprd/labels.py` | `4f9f4a2a92cb42be48fc9d8380b70184dd49fac47728f9a290f9ac740a7a928b` | 3,847 |
| 23 | `code/cprd/ledger.py` | `178020179b7337ecaf086570b2bb6a19fce8566e112cd7ba97812fca6c4b6bde` | 13,421 |
| 24 | `code/cprd/losses.py` | `25df738cc8a87e8a82c250313ae0d1c89319f0e850ac0e585c5b1acfcd14ec5d` | 4,729 |
| 25 | `code/gate/bringup.py` | `47a1f27a3d510dc26db1355fcdcb9bb5763e0c406b1c47e68e8c784fc37d2036` | 10,017 |
| 26 | `code/gate/bringup_cpu.py` | `9438229d0d3020e2a05505d152e4815a7c99143157d21ae6869f6c865bdba840` | 8,114 |
| 27 | `code/gate/bringup_gpu.py` | `f8b8b09f3fbb17021e7c34c94fb121d53f294c0dafad895891b297707d734e04` | 8,367 |
| 28 | `code/gate/cpu_gate.py` | `d7627dbf8a85fa013e8ae55e5b4090022c50e3e5110f381d0cb805ab9a006306` | 14,021 |
| 29 | `code/gate/qualify_detectors.py` | `549e1514904a23781b980b4db51f19c56118ac23289786fc9be42c66d129d79f` | 7,204 |
| 30 | `code/patches/P1_P3_erasure_gem.diff` | `6840345c3f3234f85a8116973909daaa5859d42fae94f068c1a87f0c48ec88da` | 2,811 |
| 31 | `code/patches/P2_base_pipeline_wrapper.diff` | `0a31346af174ab43286ae0f654264450bc2c003a596490bf2f26b0ad06c1466d` | 1,014 |
| 32 | `code/patches/P4_P6_optional_imports.diff` | `993c5affb0ddc19c36ddd16747aef1fc72ccdfdc722f26ff8a4b7f7341ff024c` | 4,038 |
| 33 | `code/patches/P7_factory_path.diff` | `6d58be9ea1414abaff6d19535d28070c367f191726956ede523f3146dae6ba6e` | 1,001 |
| 34 | `code/runenv.sh` | `21554093ac3e3e1b9e1447e9284295dff59daedd88d247ab6ba67f2d8ce22df4` | 1,153 |
| 35 | `code/scripts/build_cases.py` | `e648e2b36d3883150f8dd6e098ee8bd290a5b89c41fea10bce084e0215650401` | 8,690 |
| 36 | `code/scripts/select_categories.py` | `4742980f6f776e99210607791050af9da0ecf242486e9601659433ee4843882f` | 9,443 |
| 37 | `configs/cases.json` | `815604dab52bdedeac580f4f433befacf613000f9c16cff0f0900136b34e06e1` | 31,207 |
| 38 | `configs/categories_and_qualification.json` | `acf5bad481205d07e66a79ea1abaa4c27da38d659a017aebf7e1a379719f0bca` | 83,712 |

**38 files indexed**, 513,340 bytes (index excluded).

## Release

Tag **`cpr-detector-controls-v1-entrygate`** (id 417012770), target
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f`, six assets. Asset hashes are in
`artifacts/release_SHA256SUMS.txt` (indexed above) and were re-verified against
bytes **downloaded back from GitHub**; see `artifacts/release_verification.json`.

## Scope

No FLUX or detector base weights, no COCO photographs, no credentials and no
vendored GEM source are published. GEM appears only as attributed diffs in
`code/patches/`. Closed studies, their releases and their tags are untouched.
