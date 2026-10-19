# EVIDENCE_INDEX — CPR-GEM CPU-only correctness prototype

**Payload commit:** `aa4f5c926ee2e231692924162848b116a15e3edd`

**Generated:** 2026-10-19T02:49:57Z (UTC), **after** the payload commit, from the bytes Git
actually stored at that commit (`git show <commit>:<path>`) — not from the
working tree.

**Noncircular by construction:** this file is written by an *index-only child*
commit and **excludes its own hash**. Every row below is therefore verifiable
against the payload commit without reference to this file.

## Verification

```bash
git show aa4f5c926ee2e231692924162848b116a15e3edd:results/cpr_gem_cpu_prototype/<path> | sha256sum
```

| # | File | SHA-256 (committed bytes) | Bytes |
|---|---|---|---|
| 1 | `FINAL_REPORT.md` | `93e448b601cdc36c99b9c99d5fa7680a764231e48b41146e9d19cb8d89aa8a05` | 18,777 |
| 2 | `REPAIR_HISTORY.md` | `0aa30ee067a972529596fe96a20b0bf83c335e9570974e4d6ea2cf4bf1807f4c` | 3,443 |
| 3 | `RUN_MANIFEST.md` | `9cb492dfe83e3d9675f52036bbfb2cad314bcc70376fc9cb7dc81ebd20e3923f` | 16,347 |
| 4 | `artifacts/FREEZE_SEAL.txt` | `99ad09573d0d2136f6d823a2989f0d7f8a200840a165ddc0e1c0bf7e6318bfe7` | 783 |
| 5 | `artifacts/attempt_01/resource_ledger.jsonl` | `192be9f34441ced07c7c2e7bd23fb0f1e1fb1d5aeff122139e440926b8c41f06` | 25,488 |
| 6 | `artifacts/attempt_01/results.json` | `facfbf77c1da04a1b4523ca84fe012dcf457da85b2d2b0c61b63b042de582d78` | 44,645 |
| 7 | `artifacts/attempt_01/run_suite.py` | `8038fdbeca5ce67074a0ba8863a07751399e7438e96e48c177fe30543116dc0c` | 29,796 |
| 8 | `artifacts/attempt_02/resource_ledger.jsonl` | `015993a08395f46dc95346f64060f09ab01fd6350f39f2b92fdfe5b2f40c30f3` | 50,934 |
| 9 | `artifacts/attempt_02/results.json` | `e6cd28f178f2f0d403c54227b9e61dc6a46a586147eb9ee0acf9d8b9f3f15089` | 44,980 |
| 10 | `artifacts/repair_01.diff` | `316396dd7380d60e026fb9ec47a243e15bd0afed4a79f9e07a078ecba8301fc8` | 4,288 |
| 11 | `artifacts/resource_ledger.jsonl` | `015993a08395f46dc95346f64060f09ab01fd6350f39f2b92fdfe5b2f40c30f3` | 50,934 |
| 12 | `artifacts/results.json` | `e6cd28f178f2f0d403c54227b9e61dc6a46a586147eb9ee0acf9d8b9f3f15089` | 44,980 |
| 13 | `code/cpr_gem/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| 14 | `code/cpr_gem/fixtures.py` | `ed46106fc4fa3ce07da39572f2cf36adf9f1e7f0600e34e628706d42f9017ed0` | 6,799 |
| 15 | `code/cpr_gem/geometry.py` | `e64cdcbf497151b99a162ed23f0240daae7106652ceefbd550377422b8e3b976` | 7,388 |
| 16 | `code/cpr_gem/ledger.py` | `5875d6279780c0ccb2d7d6cb7e4bd7cee7dc61367e8efb342f1977175fad8fec` | 3,772 |
| 17 | `code/cpr_gem/losses.py` | `622da536485b1bc7ff9538bfe252eee4a2db59e4c90ec3c679c78016c2314b86` | 12,717 |
| 18 | `code/cpr_gem/oracle.py` | `d0eadda3055a2ffec6a5a6a8f70e2f2f0b91118e4dee692ebfaa083307b55c12` | 4,698 |
| 19 | `code/run_suite.py` | `c01153d677c59ae87a8be2274f3de24f2d079449c26ebe47c990e611efd9435a` | 31,637 |
| 20 | `code/run_suite.sh` | `ba38d3626727bcc5917c7b1a9601af7b2a95092ae4dcac2f8870060dafa9691a` | 1,468 |

**20 files indexed**, 403,874 bytes total
(the index itself excluded).

## Scope

Model-free, CPU-only. No model weights, no generated images, no secrets, no
questionnaire responses, no private maps and no historical release assets are
present in any indexed file; every indexed file is plain text or JSON.

Verdict at the payload commit: **`CPU_PROTOTYPE_VERIFIED`** — 92/92 declared
configurations, 0 missing, 0 undeclared, 0 failures.
