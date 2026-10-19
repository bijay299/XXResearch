# EVIDENCE_INDEX — CPR real-model controls pilot v1 (Amendment 01)

> **Status retained: `ENTRY_GATE_FAILED — SCORER_NOT_QUALIFIED`.**
> No production CPR training, no operating-point selection, no held-out
> comparison, no CPR-versus-control result. **CPR improvement remains untested.**
> See [`AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md`](AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md).

**Amendment payload commit:** `5c2d99516164eae426358d7a390191cc184079bb`

**Prior commits, unchanged and still valid for their own payloads:**
`a462b94800d44441c6c056c4cd220350862741ad` (initial payload),
`5ea308a356938b44a5aee031b80193fa8953e296` (verification payload),
`5253df00f184dfed772c9c04d3270a86f8a872af` (reviewed HEAD).

**Generated:** 2026-10-19T04:39:00Z (UTC), **after** the amendment payload commit, from the
bytes Git stored at that commit (`git show <commit>:<path>`) — not the working
tree. Written by an *index-only child* and **excludes its own hash**.

## Verification

```bash
git show 5c2d99516164eae426358d7a390191cc184079bb:results/cpr_real_model_controls_v1/<path> | sha256sum
```

`P` marks a **protected** file, byte-for-byte unchanged from reviewed HEAD
`5253df00f184dfed772c9c04d3270a86f8a872af` (26/26 verified; see
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md)).

| # | P | File | SHA-256 (committed bytes) | Bytes |
|---|---|---|---|---|
| 1 |  | `AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md` | `7aef9730e49924a0001597f0aded720f657c66bc0f9b51fe8fb19c6c9c1917c3` | 17,484 |
| 2 |  | `FINAL_REPORT.md` | `c5140d69a2723096fc9335950a26e80dcb9d5e571b635b242878f7c65e5431f0` | 22,559 |
| 3 |  | `PRESERVATION_AND_ACCOUNTING.md` | `e52894c41d3cb13d40e790e1881788980dd0d364120009ecab31fe1313c0510f` | 9,975 |
| 4 | **P** | `RECONSTRUCTION.md` | `0a5f942cbcde02922439156afafafc394da335fea13c30d7b825db558bb9f0b8` | 2,484 |
| 5 | **P** | `RUN_MANIFEST.md` | `0923bd09f67f66c0c3599312e72c574f6979626c3b65ed9898f797c81ab4b106` | 20,057 |
| 6 | **P** | `artifacts/bringup_patches.md` | `3dd50e166b2f850269966f818093a29feab6002f266965f2108a37c7cdedb4ce` | 2,154 |
| 7 | **P** | `artifacts/bringup_results.json` | `33bb4482dc23799038422dcf41b314b4dc152225aecdc67da3af5b82a00f61b7` | 3,104 |
| 8 | **P** | `artifacts/calibration_images.json` | `06d7621eb3b79bcd82d3653b74bb6e30b1e332c0bbb5027a93f6aed72a8495c5` | 30,144 |
| 9 | **P** | `artifacts/calibration_visual_judgments.json` | `9e1e8ea70201efbb5d365a657c2069a025560995cd09183d96c221d973249909` | 12,909 |
| 10 | **P** | `artifacts/counters.json` | `eff73346ff8d85f154c9d846125476fbdb0828aa27ef5150629c59c5cb3d873e` | 215 |
| 11 | **P** | `artifacts/cpu_gate_results.json` | `c341331bb270d8d5070b89645b4caecad376bd838dda756f69bb39a3dd610fec` | 17,283 |
| 12 | **P** | `artifacts/freeze_seal.json` | `8c2b7e56b5776def263c2a2439433ba4f2bad2a677445a78d0d0841a486ddfcc` | 2,512 |
| 13 | **P** | `artifacts/release_SHA256SUMS.txt` | `cadd7ed140646d7a1b7eefa332a5f6992fd7051680363d2295acfe2c2268a965` | 456 |
| 14 | **P** | `artifacts/release_verification.json` | `009b448ca33e6d59eb56e201c058781b47d0356a2e6bf62bc2c05017535212d6` | 656 |
| 15 | **P** | `artifacts/resource_ledger.jsonl` | `33ad39e15d600d85b0c098b745319acdc2dc70fa7ff6b926542c1a48cf31877c` | 62,565 |
| 16 | **P** | `artifacts/scorer_qualification.json` | `4d007202d4eb5e0e335f284b7f4de605aee52d612008b18d62b5b422d93591d2` | 14,860 |
| 17 | **P** | `code/configs/build_cases.py` | `3f9e345fd13afb9d185adc3b15e0892c3d03968ea8f7c26b91b9eb55ac3075fd` | 9,986 |
| 18 | **P** | `code/configs/cases.json` | `fca4dd02105458686eda80cef508aeaa7c8d456c4385c5b0fff53d986d58bf3c` | 47,340 |
| 19 | **P** | `code/cprlib/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| 20 | **P** | `code/cprlib/flux_harness.py` | `2e1128fd0544ff012133aae2a91031de09b4471f4974c1bda1470beaa5e8b4b3` | 16,491 |
| 21 | **P** | `code/cprlib/ledger.py` | `58d567a2e9242b77d73965070d9ec39a9e5909104feb55286083d889c178ab30` | 8,021 |
| 22 | **P** | `code/cprlib/lora.py` | `746b29abfc728af0d68ba5517a3be8d4838d2eb285464dc3fe4d27d52f965b21` | 7,809 |
| 23 | **P** | `code/cprlib/losses.py` | `37ce7577652e97f66c8139c85832d316ee65d060a0d7362ec94f92ba6891d857` | 6,619 |
| 24 | **P** | `code/cprlib/oracle.py` | `d2180185885b134600d766e9cc1f67431aed61868fd847731c40d7c9d4b25836` | 4,014 |
| 25 | **P** | `code/cprlib/scorer.py` | `1610de721851a4bc8971d96dc34252ba5af2c0a3d654a903d31d94522e1797fd` | 6,781 |
| 26 | **P** | `code/gate/bringup.py` | `3543577d6b1dc9f7ee1d3e8ad0839e68efe295af875795783e120ae8423d4377` | 10,837 |
| 27 | **P** | `code/gate/gen_calibration.py` | `1566304b8ffcb0f7cec318a6085e63fd70ff424986aaecbbab652f375b4564b2` | 2,000 |
| 28 | **P** | `code/gate/run_cpu_gate.py` | `9992308d269845b968d92ccfe62411388f9257cf8e77307fef15c84c570f7a33` | 16,578 |
| 29 | **P** | `code/runenv.sh` | `13842757bdd36f299f23f9c29503e8d208b4b337581e1661f5b8a0babe8acc8b` | 1,062 |

**29 files indexed** (26 protected), 356,955 bytes (index excluded).

## Historical fact preserved

The **27 files** indexed by the original `EVIDENCE_INDEX.md` at reviewed HEAD
**verified correctly** against the committed bytes at the verification payload.
That verification stands; this index is regenerated only because documentation
files were amended and added.

## Release — unchanged

Release `cpr-real-model-controls-v1-entrygate`, id 416896477, tag target
`a462b94800d44441c6c056c4cd220350862741ad`. **All six assets retain their
original IDs, bytes, sizes, digests and timestamps**; none was replaced or
re-uploaded and the tag was not moved. The only release metadata change is a
dated amendment notice prepended to the body, with the original body retained
verbatim below a marked historical-body delimiter; its pre-edit bytes and hash
are recorded in `PRESERVATION_AND_ACCOUNTING.md` §6.1.
