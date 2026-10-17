# Published evidence index — chain saw -> parachute sequential diagnostic protocol

**This is a protocol-only publication. Nothing listed here was
produced by running a model, encoder, GPU job, or experiment.**

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `7e166534e2fd` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

The second group below lists files **already published** at commit
`01f88326e3e66a2baa60fa9b5cedbdaedb9c7437` that this protocol reads
from and cites — included here only so their hashes are verifiable
from this same table; they are not re-published or modified.

Regenerate: `python scripts/seq/publish_modern_flux_dp_uce_protocol_evidence_index.py`
— run it LAST, after every other file in this result set is final.

## This protocol's own files (revision 01)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/sequential_diagnostic_protocol/PROTOCOL.md` | 48,106 | `12c94b390613c35a7ab1bbf8fa6f813017f3d9cc8f6ee54be37389f73d9adc1d` |
| `results/modern_flux_dp_uce/sequential_diagnostic_protocol/ELIGIBILITY_AND_ENDPOINTS.json` | 11,029 | `4e14b6340e29eea0761e5fd2eecae36a04ce7f941d61ca1c29f99191f24c4a1a` |
| `results/modern_flux_dp_uce/sequential_diagnostic_protocol/SOURCE_AND_ASSUMPTION_LEDGER.md` | 13,572 | `cdee8c2034d3be4272c92ead2d19a3d3d86a9c073123fae10124dda26a5f4e09` |
| `results/modern_flux_dp_uce/sequential_diagnostic_protocol/REVISION_01.md` | 15,768 | `b65ef18780a65a262ae677b418bb3ca142a41f9638b3b88cb013ab64eb40daf5` |

## Existing evidence this protocol depends on (referenced, not copied -- already published at commit 01f8832, listed here again only for a single-table traceability check)

| path | bytes | sha256 |
|---|---|---|
| `results/modern_flux_dp_uce/RUN_MANIFEST.md` | 8,583 | `38fce22cb58eac0f1993bca6b44e673638e02adcc8776a489b189eb6199219dc` |
| `results/modern_flux_dp_uce/FINAL_REPORT.md` | 10,779 | `e4e16b13ec21515ab3bc0e4f6a32514ba38f1bc5837cb103419940c945224f94` |
| `results/modern_flux_dp_uce/evidence/recognition_summary.json` | 2,935 | `0dd43f81ed98fe53eecb26b72b0e423a643831df687fc9466ded3e2cedaa96d4` |
| `results/modern_flux_dp_uce/evidence/M0_per_image.csv` | 238,370 | `4217595b8ffacd7518387ad39dddb97abd4954ba7d02f4c3accaacd6a9b4dcfb` |
| `results/modern_flux_dp_uce/evidence/DP_B_per_image.csv` | 237,442 | `27aaa739e899fd674d760eeddd1949512a7d71ce3faedcdd5f1924537e57c388` |
| `results/modern_flux_dp_uce/evidence/UCE_B_per_image.csv` | 236,652 | `1030d6c8e83b8f72f768112facfbfd58d5a46364ba013f9aacc8f12715555ad5` |
| `results/modern_flux_dp_uce/evidence/full_case_manifest.csv` | 82,783 | `559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26` |
| `results/modern_flux_dp_uce/evidence/gates_and_retention.txt` | 4,503 | `3356735da7c5a2173cd6e8d127b9ae8293a80db86f4ea3905332e0042193a2c5` |

---

**12 files listed**, none absent.
