#!/usr/bin/env python3
"""Write results/changing_replacement_assessment/PUBLISHED_EVIDENCE_INDEX.md.

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

R = "results/changing_replacement_assessment"
OUT = f"{R}/PUBLISHED_EVIDENCE_INDEX.md"

GROUPS: list[tuple[str, list[str]]] = [
    ("Analytical assessment (no experiments)", [
        f"{R}/README.md",
        f"{R}/ASSESSMENT.md",
        f"{R}/SOURCE_LEDGER.md",
        f"{R}/CORRECTIONS.md",
    ]),
]


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    L = [
        "# Published evidence index — changing-replacement candidate assessment",
        "",
        "**ANALYTICAL ASSESSMENT — NO EXPERIMENTS.** Nothing listed here was produced by",
        "running a model, encoder, detector, GPU job, or any experiment.",
        "",
        f"Every path below is **in the repository** on branch `{branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "No external paper PDFs/HTML are vendored: sources are cited by arXiv ID, version",
        "and inspected section in `SOURCE_LEDGER.md`. The pinned DP source",
        "(`czhang024/CVPR_DouleProjection@3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`) is",
        "cited by commit and line, not copied.", "",
        "Regenerate: `python scripts/seq/publish_changing_replacement_assessment_evidence_index.py`",
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
                                     else ", none absent."), ""]
    pathlib.Path(OUT).write_text("\n".join(L))
    print(f"{OUT}: {n} files listed, {absent} absent")
    return 1 if absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
