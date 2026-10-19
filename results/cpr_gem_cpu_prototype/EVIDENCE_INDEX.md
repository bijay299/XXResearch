# EVIDENCE_INDEX — CPR-GEM CPU-only correctness prototype (Amendment 01)

> **Status: `CPU_PROTOTYPE_INCOMPLETE — PARTIAL CORRECTNESS EVIDENCE`**
> See [`AMENDMENT_01_STATUS_AND_LIMITATIONS.md`](AMENDMENT_01_STATUS_AND_LIMITATIONS.md).

**Payload commit (amendment):** `fc5bf0349ffae0ea375719ccbb0fb4747acd42fc`

**Prior commits, unchanged and still valid for their own payloads:**
`aa4f5c926ee2e231692924162848b116a15e3edd` (original payload), 
`8387fb80c589b1149370c1084924d061c25ac7ee` (reviewed HEAD).

**Generated:** 2026-10-19T03:14:29Z (UTC), **after** the amendment payload commit, from the bytes
Git actually stored at that commit (`git show <commit>:<path>`) — not from the
working tree.

**Noncircular by construction:** written by an *index-only child* commit and
**excludes its own hash**; no later verification output is indexed. Every row is
verifiable against the payload commit without reference to this file.

## Verification

```bash
git show fc5bf0349ffae0ea375719ccbb0fb4747acd42fc:results/cpr_gem_cpu_prototype/<path> | sha256sum
```

`P` marks a **protected** file — byte-for-byte unchanged from reviewed HEAD
`8387fb80c589b1149370c1084924d061c25ac7ee`; see
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md).

| # | P | File | SHA-256 (committed bytes) | Bytes |
|---|---|---|---|---|
| 1 |  | `AMENDMENT_01_STATUS_AND_LIMITATIONS.md` | `4571c45bfaf4b2e2363786e93a57a8e1a01aa0d92a0ca34a8418630b71c5f377` | 18,639 |
| 2 |  | `FINAL_REPORT.md` | `7dad02ed7c0769f1ffb4c7b47c11769c6f17785e0609f039455c86548216cc56` | 28,551 |
| 3 |  | `PRESERVATION_AND_ACCOUNTING.md` | `71fde03b6a036066b8a41d3985b847efebf294b5b798c06dcaf576a356605cc3` | 7,580 |
| 4 | **P** | `REPAIR_HISTORY.md` | `0aa30ee067a972529596fe96a20b0bf83c335e9570974e4d6ea2cf4bf1807f4c` | 3,443 |
| 5 | **P** | `RUN_MANIFEST.md` | `9cb492dfe83e3d9675f52036bbfb2cad314bcc70376fc9cb7dc81ebd20e3923f` | 16,347 |
| 6 | **P** | `artifacts/FREEZE_SEAL.txt` | `99ad09573d0d2136f6d823a2989f0d7f8a200840a165ddc0e1c0bf7e6318bfe7` | 783 |
| 7 | **P** | `artifacts/attempt_01/resource_ledger.jsonl` | `192be9f34441ced07c7c2e7bd23fb0f1e1fb1d5aeff122139e440926b8c41f06` | 25,488 |
| 8 | **P** | `artifacts/attempt_01/results.json` | `facfbf77c1da04a1b4523ca84fe012dcf457da85b2d2b0c61b63b042de582d78` | 44,645 |
| 9 | **P** | `artifacts/attempt_01/run_suite.py` | `8038fdbeca5ce67074a0ba8863a07751399e7438e96e48c177fe30543116dc0c` | 29,796 |
| 10 | **P** | `artifacts/attempt_02/resource_ledger.jsonl` | `015993a08395f46dc95346f64060f09ab01fd6350f39f2b92fdfe5b2f40c30f3` | 50,934 |
| 11 | **P** | `artifacts/attempt_02/results.json` | `e6cd28f178f2f0d403c54227b9e61dc6a46a586147eb9ee0acf9d8b9f3f15089` | 44,980 |
| 12 | **P** | `artifacts/repair_01.diff` | `316396dd7380d60e026fb9ec47a243e15bd0afed4a79f9e07a078ecba8301fc8` | 4,288 |
| 13 | **P** | `artifacts/resource_ledger.jsonl` | `015993a08395f46dc95346f64060f09ab01fd6350f39f2b92fdfe5b2f40c30f3` | 50,934 |
| 14 | **P** | `artifacts/results.json` | `e6cd28f178f2f0d403c54227b9e61dc6a46a586147eb9ee0acf9d8b9f3f15089` | 44,980 |
| 15 | **P** | `code/cpr_gem/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| 16 | **P** | `code/cpr_gem/fixtures.py` | `ed46106fc4fa3ce07da39572f2cf36adf9f1e7f0600e34e628706d42f9017ed0` | 6,799 |
| 17 | **P** | `code/cpr_gem/geometry.py` | `e64cdcbf497151b99a162ed23f0240daae7106652ceefbd550377422b8e3b976` | 7,388 |
| 18 | **P** | `code/cpr_gem/ledger.py` | `5875d6279780c0ccb2d7d6cb7e4bd7cee7dc61367e8efb342f1977175fad8fec` | 3,772 |
| 19 | **P** | `code/cpr_gem/losses.py` | `622da536485b1bc7ff9538bfe252eee4a2db59e4c90ec3c679c78016c2314b86` | 12,717 |
| 20 | **P** | `code/cpr_gem/oracle.py` | `d0eadda3055a2ffec6a5a6a8f70e2f2f0b91118e4dee692ebfaa083307b55c12` | 4,698 |
| 21 | **P** | `code/run_suite.py` | `c01153d677c59ae87a8be2274f3de24f2d079449c26ebe47c990e611efd9435a` | 31,637 |
| 22 | **P** | `code/run_suite.sh` | `ba38d3626727bcc5917c7b1a9601af7b2a95092ae4dcac2f8870060dafa9691a` | 1,468 |

**22 files indexed** (19 protected), 439,867 bytes total
(the index itself excluded).

## Historical fact preserved

The **20 rows of the original `EVIDENCE_INDEX.md`** published at reviewed HEAD
were re-checked against the committed bytes at the original payload
`aa4f5c926ee2e231692924162848b116a15e3edd`: **20 rows re-verified, 0
mismatches.** That verification stands; this index is regenerated only because
the amendment changed two documentation files and added two.

## Scope

Model-free, CPU-only. No model weights, no generated images, no secrets, no
questionnaire responses, no private maps and no historical release assets are
present in any indexed file; every indexed file is plain text or JSON.

**No code was changed, no test was run and no experiment was executed** by the
amendment that produced this index.
