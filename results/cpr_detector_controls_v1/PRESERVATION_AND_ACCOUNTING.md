# Preservation and accounting record — Amendment 01

**Mechanical verification, not new validation.** Everything here comes from reading Git
objects and local files, parsing existing JSON/JSONL, arithmetic, hashing and GitHub
release/tag **metadata**. **No candidate, test, scorer or pipeline module was imported; no
`torch` was imported; no checkpoint or model was loaded; no GPU was used; no detector was
run; no image was generated, scored or labelled; no reproduction command was invoked;
nothing was installed; no test was run.** The only write to GitHub other than Git
objects is a prepended notice in the release body of
`cpr-detector-controls-v1-entrygate` (§3).

Reviewed HEAD: `9940e9403452e2b382c108c2d4d50bd3d054d707`
Prospective freeze `3fba0ff3e4eaba65f1f6466249e4dfb939a29a72` · additive freeze
`67d8b74d4d571d1641bd840a9adc5f7b4139ca35` · payload / release target
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f` · verification payload
`42614565cd79f911c194d9f7c8d1dea017fbe17c`.
The amendment payload commit, the post-publication verification commit and the
regenerated-index child are named in `EVIDENCE_INDEX.md` and in
`amendment_01_records/post_publication_verification.json`.

Companion records (all in `amendment_01_records/`): `mechanical_checks.py` / `.json`
(qualification recount, accounting arithmetic, chronology), `protected_files_sha256_at_9940e94.txt`,
`release_417012770_pre_amendment.json`, `release_body_original.md`,
`tags_pre_amendment.txt`, `all_releases_pre_amendment.json`,
`reverify_historical_index.py` / `historical_index_reverification.json`, and (written after
publication) `post_publication_verification.json`.

---

## 1. Protected files — byte-for-byte unchanged

The protected set is `code/**`, `configs/**`, `artifacts/**`, `RUN_MANIFEST.md` and
`RECONSTRUCTION.md` of this study (36 files). The table is the pre-edit listing, recorded
from the committed blobs at `9940e94` and cross-checked against the working tree **before
any edit**; the working tree was re-checked against it again immediately before the
amendment commit (36/36 identical). Equality of Git trees between `9940e94`, the amendment
payload commit and the final commit for these paths (`git diff --stat` empty) is recorded
in `post_publication_verification.json`.

| Path (under `results/cpr_detector_controls_v1/`) | SHA-256 | Bytes | Identical now |
|---|---|---|---|
| `RECONSTRUCTION.md` | `118122b8a78eaed09f861b1f6bb42538e47fcf6a650ee2947f686738a176c6b3` | 2,738 | yes |
| `RUN_MANIFEST.md` | `34299e1ef9984612decc380a496f9bb2c83e5d0863f0dc41dbcbfc5542614e56` | 19,241 | yes |
| `artifacts/allocation_journal.jsonl` | `d81c8ce40c28083c2027c20324fc2c8ebd5b4b4cf094a12b2824db73e952c197` | 2,884 | yes |
| `artifacts/allocation_ledger.json` | `fc817811471550a3d2286bb1626fd9360141d4c768babcfe64c25d95ea39ad91` | 1,974 | yes |
| `artifacts/bringup_cpu.json` | `3dade973d40a1193ceaab38d7604a79fbc692df355a36b549ff20dea35706609` | 2,378 | yes |
| `artifacts/bringup_gpu.json` | `9e667954db374cb4e3f38ad1973b971ad4a4b8f2c9b21fdf9bef8f0cda319eca` | 2,190 | yes |
| `artifacts/cpu_gate.json` | `1999cb52c29a192fdab4ac79b865aac4d5450abcb77a710b34c0f0579a9c3c32` | 11,821 | yes |
| `artifacts/qualification.json` | `e0ac5066b9b59139df17c4acc7467c2bb50275a3c27a88a37c562067084a3338` | 31,607 | yes |
| `artifacts/qualification_raw_detr_resnet50_no_timm.json` | `8a309c48a679e7baa7f72aa770c2d1829b3044fd77e036f4c177b312e90bc708` | 81,201 | yes |
| `artifacts/qualification_raw_rtdetr_r50vd.json` | `ee264eb73018f0f4db2ece7f2456efb2a2e627b58c9a4bbc8ecd18e7f8df7eef` | 81,276 | yes |
| `artifacts/release_SHA256SUMS.txt` | `46570247ec28459a2277f83111a1798903a4feff92352cabf66bec8648848694` | 463 | yes |
| `artifacts/release_verification.json` | `bf3ecac6685c75ce07189363c2269b07f55e345b9692ac375a7b7d79c1613884` | 542 | yes |
| `artifacts/repair_01/README.md` | `e58cf4ddc3757234611fdebe916f886d4740409ba8ee99eb270dc35139ca043e` | 1,416 | yes |
| `artifacts/repair_01/cpu_gate.py.before` | `cc351abc77a6f340dade89a9e13593c8c3f8a64d7d94922f8989feb155a10518` | 13,202 | yes |
| `artifacts/repair_01/cpu_gate_FAILED.json` | `ea176b62f30c7da04363dd5e55d9a6076080987a70c75438302062c0b05b72b5` | 11,669 | yes |
| `artifacts/repair_01/cpu_gate_PASSED.json` | `1999cb52c29a192fdab4ac79b865aac4d5450abcb77a710b34c0f0579a9c3c32` | 11,821 | yes |
| `artifacts/repair_01/repair_01.diff` | `61ae6024dab87045e98758ff0a5c266097d8c2b8f02d14a613b504d1fe160b3b` | 1,943 | yes |
| `code/cprd/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 | yes |
| `code/cprd/detect.py` | `2102f4884604ce6c632f8a176cf5a64071fd8bf397fed6cbd05d08476646cc9b` | 5,100 | yes |
| `code/cprd/labels.py` | `4f9f4a2a92cb42be48fc9d8380b70184dd49fac47728f9a290f9ac740a7a928b` | 3,847 | yes |
| `code/cprd/ledger.py` | `178020179b7337ecaf086570b2bb6a19fce8566e112cd7ba97812fca6c4b6bde` | 13,421 | yes |
| `code/cprd/losses.py` | `25df738cc8a87e8a82c250313ae0d1c89319f0e850ac0e585c5b1acfcd14ec5d` | 4,729 | yes |
| `code/gate/bringup.py` | `47a1f27a3d510dc26db1355fcdcb9bb5763e0c406b1c47e68e8c784fc37d2036` | 10,017 | yes |
| `code/gate/bringup_cpu.py` | `9438229d0d3020e2a05505d152e4815a7c99143157d21ae6869f6c865bdba840` | 8,114 | yes |
| `code/gate/bringup_gpu.py` | `f8b8b09f3fbb17021e7c34c94fb121d53f294c0dafad895891b297707d734e04` | 8,367 | yes |
| `code/gate/cpu_gate.py` | `d7627dbf8a85fa013e8ae55e5b4090022c50e3e5110f381d0cb805ab9a006306` | 14,021 | yes |
| `code/gate/qualify_detectors.py` | `549e1514904a23781b980b4db51f19c56118ac23289786fc9be42c66d129d79f` | 7,204 | yes |
| `code/patches/P1_P3_erasure_gem.diff` | `6840345c3f3234f85a8116973909daaa5859d42fae94f068c1a87f0c48ec88da` | 2,811 | yes |
| `code/patches/P2_base_pipeline_wrapper.diff` | `0a31346af174ab43286ae0f654264450bc2c003a596490bf2f26b0ad06c1466d` | 1,014 | yes |
| `code/patches/P4_P6_optional_imports.diff` | `993c5affb0ddc19c36ddd16747aef1fc72ccdfdc722f26ff8a4b7f7341ff024c` | 4,038 | yes |
| `code/patches/P7_factory_path.diff` | `6d58be9ea1414abaff6d19535d28070c367f191726956ede523f3146dae6ba6e` | 1,001 | yes |
| `code/runenv.sh` | `21554093ac3e3e1b9e1447e9284295dff59daedd88d247ab6ba67f2d8ce22df4` | 1,153 | yes |
| `code/scripts/build_cases.py` | `e648e2b36d3883150f8dd6e098ee8bd290a5b89c41fea10bce084e0215650401` | 8,690 | yes |
| `code/scripts/select_categories.py` | `4742980f6f776e99210607791050af9da0ecf242486e9601659433ee4843882f` | 9,443 | yes |
| `configs/cases.json` | `815604dab52bdedeac580f4f433befacf613000f9c16cff0f0900136b34e06e1` | 31,207 | yes |
| `configs/categories_and_qualification.json` | `acf5bad481205d07e66a79ea1abaa4c27da38d659a017aebf7e1a379719f0bca` | 83,712 | yes |

Incorrect historical flags and prose inside these files — the `passed` and `QUALIFIED`
fields of `artifacts/qualification.json`, the `pos_hit` predicate at
`code/gate/qualify_detectors.py:112`, the "stricter" text in `artifacts/repair_01/README.md`,
the panel and enforcement statements in `RUN_MANIFEST.md`, the "raw boxes, scores" wording
of `RECONSTRUCTION.md` — are **preserved, not edited**; they are superseded by the
amendment. This includes the 64-row CPU record, the failed outputs, the repair source and
diff, the patches and the ledger and journal.

## 2. Allowed tracked changes, and nothing else

`FINAL_REPORT.md`, `ENFORCEMENT_GAPS.md` (corrected interpretation on top, original text
verbatim under a historical-text delimiter), the new
`AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md`, this file and
`amendment_01_records/`, this study's pointer row in `results/README.md`, and the
regenerated `EVIDENCE_INDEX.md`. The allowed-diff check against `9940e94` is recorded in
`post_publication_verification.json`. Everything outside this study directory other than the single `results/README.md` row is
unchanged: the post-publication record compares `git ls-tree -r` of `9940e94` and of the
final commit with this study's directory and `results/README.md` excluded, and requires
identical output.

## 3. Release and tags

Release `cpr-detector-controls-v1-entrygate` (id 417012770), tag target
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f`, created 2026-10-19T07:25:04Z, published 2026-10-19T07:26:22Z,
not a draft, not a prerelease. Observed 2026-10-19T16:46:53Z before any edit. **Six assets —
IDs, sizes, digests and timestamps — as listed below, to be re-read after the edit:**

| Asset ID | Name | Bytes | SHA-256 (GitHub digest) | Created / updated (UTC) |
|---|---|---|---|---|
| 655554191 | `allocation_ledger.tar.gz` | 1,613 | `4a3c9ee91ac3eb6c0816a371a930724a2660f9f92868730353819d87a0efd742` | 2026-10-19T07:26:24Z / 2026-10-19T07:26:25Z |
| 655554101 | `entry_gate_peft_adapter_init.pt` | 29,941,586 | `90dd39cb8adf4de4dca7ecdf1103b361ab8bfcea322c4df3111e2f7ca55cb394` | 2026-10-19T07:26:22Z / 2026-10-19T07:26:23Z |
| 655554203 | `frozen_configs.tar.gz` | 10,270 | `17279c0e7c38e1975debc1a0b65614c03dd7e85ad310f10d88376d46963fec70` | 2026-10-19T07:26:25Z / 2026-10-19T07:26:25Z |
| 655554165 | `qualification_raw_detections.tar.gz` | 14,875 | `505b4caa61e6f244ad00a73a0704cd406ec2b048f6536334e30ac5e3e3de5f2f` | 2026-10-19T07:26:24Z / 2026-10-19T07:26:24Z |
| 655554215 | `RECONSTRUCTION.md` | 2,738 | `118122b8a78eaed09f861b1f6bb42538e47fcf6a650ee2947f686738a176c6b3` | 2026-10-19T07:26:26Z / 2026-10-19T07:26:26Z |
| 655554232 | `SHA256SUMS` | 463 | `46570247ec28459a2277f83111a1798903a4feff92352cabf66bec8648848694` | 2026-10-19T07:26:26Z / 2026-10-19T07:26:26Z |

The release body is the **only** release metadata edited. The original body (2,400 bytes
UTF-8, sha256 `9e1556675b8a1c4a3c0ef735de62f3896c5cdb1345f4ccd32e1efcef67c17b84`) is saved verbatim in
`amendment_01_records/release_body_original.md`. The edited body begins with a dated,
commit-pinned amendment notice, then a historical-body delimiter, then that original body
verbatim. Tag object ids before the amendment (7 tags; none moved):

| Tag | Target |
|---|---|
| `block-object-sprint-v1` | `4177b4c1fbb623a1db2521ee8747b2c6a8ad685a` |
| `cpr-detector-controls-v1-entrygate` | `00bc8ae0a3d98640cb49cbbe616d2101e827e64f` |
| `cpr-real-model-controls-v1-entrygate` | `a462b94800d44441c6c056c4cd220350862741ad` |
| `modern-object-assets-v1` | `2a98cb3538ba71666f68fb8b95563a9cc966e0f9` |
| `teapot-human-review-v1-taskA` | `28db2f48a209a0d072c3a294e8ca27a57d868831` |
| `teapot-human-review-v1-taskA-patch01` | `ce3b95f1cf72bf6260f4210bcbc0649ebeb0ab3a` |
| `teapot-human-review-v1-taskB` | `898c5525d9bc3db71324914281243ba0d9ad4336` |

The other six releases (ids 416222173, 416312159, 416601851, 416619081, 416717874,
416896477) were recorded with their targets, assets and body hashes in
`all_releases_pre_amendment.json` and are re-read in the post-publication record.

## 4. Earlier index verification — preserved

The historical `EVIDENCE_INDEX.md` at `9940e94` (6,119 bytes, sha256
`1a30fe3656123631ab3304978bab18407f2d8a54c6d0d6076bae6e4c76da8b5d`) listed 38 files with
hashes of the bytes Git stored at payload `42614565…`. Re-checked mechanically now
(`reverify_historical_index.py`): **38 rows, 0 mismatches**. Two of those rows —
`FINAL_REPORT.md` and `ENFORCEMENT_GAPS.md` — describe the files **as they were then**; the
current versions are amended, and the regenerated index lists the current bytes. The earlier
verification stands as a verification of those historical bytes.

## 5. What tracked history can and cannot show

Tracked in Git and release assets: the five study commits, the ledger and journal, the
recorded GPU and CPU results, and the six assets. **Not tracked and not observed by this
amendment** (host history): the local run directory beyond what was published
(`GEM_src/`, local checkpoints, caches, the dependency overlay), the unpublished
`artifacts/preflight.log` (sha256
`f628c048380069bb06dc02e177e48fb0b40cf847f21854a4e410c4820a55ec8a`; hand-entered times that
are internally inconsistent with the commit chronology and are not relied on), shell and
process history, whole-task CPU and peak storage, and the exact first-action time. Where a
number depends on any of these, the amendment says so. Nothing was reconstructed or
invented.

## 6. Amendment-only resource usage (recorded separately; the scientific budget is not renewed)

| Bound for this amendment | Limit | Usage |
|---|---|---|
| Elapsed | 2 h | work began 2026-10-19T16:41:31Z; the final elapsed figure is in `post_publication_verification.json` |
| CPU threads | 4 | every command was a single-threaded read-only Git/Python/hash/HTTP call; none exceeded one thread. CPU time was **not metered**, so no CPU-hour figure is claimed |
| Extra scratch | 1 GiB | new files are small text/JSON records under `amendment_01_records/` plus saved Git/release listings in the session scratchpad on `/data`; the measured `du` is in the post-publication record |
| GPU / model loading / scientific execution | 0 | none |
