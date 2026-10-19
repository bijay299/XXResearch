# EVIDENCE_INDEX — CPR real-model controls pilot v1

> **Terminal status: `ENTRY_GATE_FAILED — SCORER_NOT_QUALIFIED`.**
> No production training ran; no arm comparison exists.

**Payload commit:** `5ea308a356938b44a5aee031b80193fa8953e296`

**Generated:** 2026-10-19T04:19:37Z (UTC), **after** the payload commit, from the bytes Git
actually stored at that commit (`git show <commit>:<path>`) — not from the
working tree. Written by an *index-only child* commit and **excludes its own
hash**, so every row is verifiable without reference to this file.

## Verification

```bash
git show 5ea308a356938b44a5aee031b80193fa8953e296:results/cpr_real_model_controls_v1/<path> | sha256sum
```

| # | File | SHA-256 (committed bytes) | Bytes |
|---|---|---|---|
| 1 | `FINAL_REPORT.md` | `152db521002ad19bf1a58b85780d5e10c69e998bffe7606d5ab89dccc9daa885` | 13,594 |
| 2 | `RECONSTRUCTION.md` | `0a5f942cbcde02922439156afafafc394da335fea13c30d7b825db558bb9f0b8` | 2,484 |
| 3 | `RUN_MANIFEST.md` | `0923bd09f67f66c0c3599312e72c574f6979626c3b65ed9898f797c81ab4b106` | 20,057 |
| 4 | `artifacts/bringup_patches.md` | `3dd50e166b2f850269966f818093a29feab6002f266965f2108a37c7cdedb4ce` | 2,154 |
| 5 | `artifacts/bringup_results.json` | `33bb4482dc23799038422dcf41b314b4dc152225aecdc67da3af5b82a00f61b7` | 3,104 |
| 6 | `artifacts/calibration_images.json` | `06d7621eb3b79bcd82d3653b74bb6e30b1e332c0bbb5027a93f6aed72a8495c5` | 30,144 |
| 7 | `artifacts/calibration_visual_judgments.json` | `9e1e8ea70201efbb5d365a657c2069a025560995cd09183d96c221d973249909` | 12,909 |
| 8 | `artifacts/counters.json` | `eff73346ff8d85f154c9d846125476fbdb0828aa27ef5150629c59c5cb3d873e` | 215 |
| 9 | `artifacts/cpu_gate_results.json` | `c341331bb270d8d5070b89645b4caecad376bd838dda756f69bb39a3dd610fec` | 17,283 |
| 10 | `artifacts/freeze_seal.json` | `8c2b7e56b5776def263c2a2439433ba4f2bad2a677445a78d0d0841a486ddfcc` | 2,512 |
| 11 | `artifacts/release_SHA256SUMS.txt` | `cadd7ed140646d7a1b7eefa332a5f6992fd7051680363d2295acfe2c2268a965` | 456 |
| 12 | `artifacts/release_verification.json` | `009b448ca33e6d59eb56e201c058781b47d0356a2e6bf62bc2c05017535212d6` | 656 |
| 13 | `artifacts/resource_ledger.jsonl` | `33ad39e15d600d85b0c098b745319acdc2dc70fa7ff6b926542c1a48cf31877c` | 62,565 |
| 14 | `artifacts/scorer_qualification.json` | `4d007202d4eb5e0e335f284b7f4de605aee52d612008b18d62b5b422d93591d2` | 14,860 |
| 15 | `code/configs/build_cases.py` | `3f9e345fd13afb9d185adc3b15e0892c3d03968ea8f7c26b91b9eb55ac3075fd` | 9,986 |
| 16 | `code/configs/cases.json` | `fca4dd02105458686eda80cef508aeaa7c8d456c4385c5b0fff53d986d58bf3c` | 47,340 |
| 17 | `code/cprlib/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| 18 | `code/cprlib/flux_harness.py` | `2e1128fd0544ff012133aae2a91031de09b4471f4974c1bda1470beaa5e8b4b3` | 16,491 |
| 19 | `code/cprlib/ledger.py` | `58d567a2e9242b77d73965070d9ec39a9e5909104feb55286083d889c178ab30` | 8,021 |
| 20 | `code/cprlib/lora.py` | `746b29abfc728af0d68ba5517a3be8d4838d2eb285464dc3fe4d27d52f965b21` | 7,809 |
| 21 | `code/cprlib/losses.py` | `37ce7577652e97f66c8139c85832d316ee65d060a0d7362ec94f92ba6891d857` | 6,619 |
| 22 | `code/cprlib/oracle.py` | `d2180185885b134600d766e9cc1f67431aed61868fd847731c40d7c9d4b25836` | 4,014 |
| 23 | `code/cprlib/scorer.py` | `1610de721851a4bc8971d96dc34252ba5af2c0a3d654a903d31d94522e1797fd` | 6,781 |
| 24 | `code/gate/bringup.py` | `3543577d6b1dc9f7ee1d3e8ad0839e68efe295af875795783e120ae8423d4377` | 10,837 |
| 25 | `code/gate/gen_calibration.py` | `1566304b8ffcb0f7cec318a6085e63fd70ff424986aaecbbab652f375b4564b2` | 2,000 |
| 26 | `code/gate/run_cpu_gate.py` | `9992308d269845b968d92ccfe62411388f9257cf8e77307fef15c84c570f7a33` | 16,578 |
| 27 | `code/runenv.sh` | `13842757bdd36f299f23f9c29503e8d208b4b337581e1661f5b8a0babe8acc8b` | 1,062 |

**27 files indexed**, 320,531 bytes (index excluded).

## Release assets (not in Git)

Release **`cpr-real-model-controls-v1-entrygate`**, target
`a462b94800d44441c6c056c4cd220350862741ad`. Asset hashes are published in
`artifacts/release_SHA256SUMS.txt` (indexed above) and were re-verified
against bytes **downloaded back from GitHub**; see
`artifacts/release_verification.json`.

## Scope

No model weights are redistributed. No secrets, questionnaire responses,
reviewer judgments, private maps or historical release assets are present in
any indexed file. Closed work is untouched.
