#!/usr/bin/env python3
"""Write results/minimalist_erasure_design_sprint/PUBLISHED_EVIDENCE_INDEX.md.

Same convention as the other publish_*_evidence_index.py scripts: exact path, byte count
and sha256 of every published file in this result set, so a reviewer can verify any row
without trusting the table:

    git show <commit>:<path> | sha256sum

Run this LAST, after every other file in the result set is final.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

R = "results/minimalist_erasure_design_sprint"
OUT = f"{R}/PUBLISHED_EVIDENCE_INDEX.md"

GROUPS: list[tuple[str, list[str]]] = [
    ("Design sprint — training efficiency of Minimalist Concept Erasure (no experiments)", [
        f"{R}/README.md",
        f"{R}/DERIVATION.md",
        f"{R}/ALGORITHM.md",
        f"{R}/COST_MODEL.md",
        f"{R}/FALSIFICATION_PLAN.md",
        f"{R}/SOURCE_LEDGER.md",
    ]),
    ("Index generator", [
        "scripts/seq/publish_minimalist_erasure_design_sprint_evidence_index.py",
    ]),
]


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    L = [
        "# Published evidence index — Minimalist erasure design sprint",
        "",
        "**DESIGN SPRINT — NO EXPERIMENTS.** Nothing listed here was produced by",
        "running a model, encoder, detector, GPU job, profiler, or any experiment.",
        "",
        f"Every path below is **in the repository** on branch `{branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "No paper PDF/HTML and no third-party source is vendored. The author source is",
        "cited by pinned commit and line in `SOURCE_LEDGER.md`",
        "(`YaNgZhAnG-V5/minimalist_concept_erasure@d178324ec4a48d0536a595b25fdad587b441add8`),",
        "read as text and never imported or executed. Decision: **STOP — the mask gradient",
        "is a free byproduct of the state-chain reverse sweep; remaining headroom is exact",
        "and ordinary, and the reference objective is unresolved**.", "",
        "Regenerate: `python scripts/seq/publish_minimalist_erasure_design_sprint_evidence_index.py`",
        "— run it LAST.", "",
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
                                     else ", none absent."),
          "",
          "`PUBLISHED_EVIDENCE_INDEX.md` itself is not listed: it is generated from the",
          "rows above and cannot contain its own hash.", ""]
    pathlib.Path(OUT).write_text("\n".join(L))
    print(f"{OUT}: {n} files listed, {absent} absent")
    return 1 if absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
