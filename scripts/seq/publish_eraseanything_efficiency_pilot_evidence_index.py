#!/usr/bin/env python3
"""Write results/eraseanything_efficiency_pilot/PUBLISHED_EVIDENCE_INDEX.md.

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

R = "results/eraseanything_efficiency_pilot"
OUT = f"{R}/PUBLISHED_EVIDENCE_INDEX.md"

GROUPS: list[tuple[str, list[str]]] = []


def _walk(title, sub):
    import os
    base = pathlib.Path(R) / sub if sub else pathlib.Path(R)
    files = []
    for dirpath, _d, fnames in os.walk(base):
        for fn in sorted(fnames):
            rel = str(pathlib.Path(dirpath) / fn)
            if sub == "" and pathlib.Path(dirpath) != base:
                continue
            if fn == "PUBLISHED_EVIDENCE_INDEX.md":
                continue
            files.append(rel)
    return (title, sorted(files))


def main() -> int:
    head_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    groups = [
        _walk("Reports and manifest (top level)", ""),
        _walk("Runnable code and configuration", "code"),
        _walk("Frozen research contract artifacts", "frozen"),
        _walk("Measured data: per-case scores, profiles, ledger, diagnostics", "measured"),
        _walk("Evidence: fixed-order review grids and image hash manifest", "evidence"),
        ("Index generator", ["scripts/seq/publish_eraseanything_efficiency_pilot_evidence_index.py"]),
    ]
    L = [
        "# Published evidence index - EraseAnything efficiency pilot",
        "",
        "**EXECUTED PILOT, status INCOMPLETE.** Effectiveness measurement is blocked by",
        "scorer adequacy; a measured engineering result is delivered. See `README.md`.",
        "",
        f"Every path below is **in the repository** on branch `{branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head_sha[:12]}` - the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "**Image originals are NOT in git** (441 newly decoded PNGs, too large for sensible",
        "git storage; no release upload was available). `evidence/IMAGE_MANIFEST.json`",
        "carries byte counts and SHA256 for all 430 retained originals, which remain on the",
        "work volume at `/data/bijaypandey/cuig_pilot/eraseanything_efficiency_pilot/`.",
        "Fixed-order review grids for all 9 M0 concept x panel combinations ARE published.",
        "No model weights and no credentials are committed.", "",
        "Regenerate: `python scripts/seq/publish_eraseanything_efficiency_pilot_evidence_index.py`",
        "- run it LAST.", "",
    ]
    n, absent = 0, 0
    for title, paths in groups:
        if not paths:
            continue
        L += [f"## {title}", "", "| path | bytes | sha256 |", "|---|---|---|"]
        for pth in paths:
            f = pathlib.Path(pth)
            if not f.is_file():
                L.append(f"| `{pth}` | - | **ABSENT** |")
                absent += 1
                continue
            b = f.read_bytes()
            n += 1
            L.append(f"| `{pth}` | {len(b):,} | `{hashlib.sha256(b).hexdigest()}` |")
        L.append("")
    L += ["---", "",
          f"**{n} files listed**" + (f", **{absent} ABSENT**." if absent else ", none absent."),
          "",
          "`PUBLISHED_EVIDENCE_INDEX.md` itself is not listed: it is generated from the",
          "rows above and cannot contain its own hash.", ""]
    pathlib.Path(OUT).write_text("\n".join(L))
    print(f"{OUT}: {n} files listed, {absent} absent")
    return 1 if absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
