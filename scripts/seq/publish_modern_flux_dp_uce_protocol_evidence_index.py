#!/usr/bin/env python3
"""Write results/modern_flux_dp_uce/sequential_diagnostic_protocol/PUBLISHED_EVIDENCE_INDEX.md.
Same convention as scripts/seq/publish_rece_evidence_index.py: exact path
and sha256 of every published file in this protocol-only result set, so a
reviewer can verify any single row without trusting the table:

    git show <commit>:<path> | sha256sum

Run this LAST, after every other file in the result set is final.

    python scripts/seq/publish_modern_flux_dp_uce_protocol_evidence_index.py
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

OUT = "results/modern_flux_dp_uce/sequential_diagnostic_protocol/PUBLISHED_EVIDENCE_INDEX.md"
R = "results/modern_flux_dp_uce/sequential_diagnostic_protocol"

GROUPS: list[tuple[str, list[str]]] = [
    ("This protocol's own files (revision 01)", [
        f"{R}/PROTOCOL.md",
        f"{R}/ELIGIBILITY_AND_ENDPOINTS.json",
        f"{R}/SOURCE_AND_ASSUMPTION_LEDGER.md",
        f"{R}/REVISION_01.md",
    ]),
    ("Existing evidence this protocol depends on (referenced, not copied -- "
     "already published at commit 01f8832, listed here again only for a "
     "single-table traceability check)", [
        "results/modern_flux_dp_uce/RUN_MANIFEST.md",
        "results/modern_flux_dp_uce/FINAL_REPORT.md",
        "results/modern_flux_dp_uce/evidence/recognition_summary.json",
        "results/modern_flux_dp_uce/evidence/M0_per_image.csv",
        "results/modern_flux_dp_uce/evidence/DP_B_per_image.csv",
        "results/modern_flux_dp_uce/evidence/UCE_B_per_image.csv",
        "results/modern_flux_dp_uce/evidence/full_case_manifest.csv",
        "results/modern_flux_dp_uce/evidence/gates_and_retention.txt",
    ]),
]


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    L = [
        "# Published evidence index — chain saw -> parachute sequential diagnostic protocol",
        "",
        "**This is a protocol-only publication. Nothing listed here was",
        "produced by running a model, encoder, GPU job, or experiment.**",
        "",
        f"Every path below is **in the repository** on branch `{branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "The second group below lists files **already published** at commit",
        "`01f88326e3e66a2baa60fa9b5cedbdaedb9c7437` that this protocol reads",
        "from and cites — included here only so their hashes are verifiable",
        "from this same table; they are not re-published or modified.", "",
        "Regenerate: `python scripts/seq/publish_modern_flux_dp_uce_protocol_evidence_index.py`",
        "— run it LAST, after every other file in this result set is final.", "",
    ]
    n, absent = 0, 0
    for title, paths in GROUPS:
        L += [f"## {title}", "", "| path | bytes | sha256 |", "|---|---|---|"]
        for p in paths:
            f = pathlib.Path(p)
            if not f.is_file():
                L.append(f"| `{p}` | — | **ABSENT** |")
                absent += 1
                continue
            b = f.read_bytes()
            n += 1
            L.append(f"| `{p}` | {len(b):,} | `{hashlib.sha256(b).hexdigest()}` |")
        L.append("")
    L += ["---", "",
          f"**{n} files listed**" + (f", **{absent} ABSENT**." if absent
                                      else ", none absent."), ""]
    pathlib.Path(OUT).write_text("\n".join(L))
    print(f"{OUT}: {n} files listed, {absent} absent")
    return 1 if absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
