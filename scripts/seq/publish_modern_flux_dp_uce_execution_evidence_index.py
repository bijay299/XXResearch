#!/usr/bin/env python3
"""Write results/modern_flux_dp_uce/sequential_diagnostic_execution/PUBLISHED_EVIDENCE_INDEX.md.

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

R = "results/modern_flux_dp_uce/sequential_diagnostic_execution"
OUT = f"{R}/PUBLISHED_EVIDENCE_INDEX.md"


def listing(sub: str, pattern: str = "*") -> list[str]:
    base = pathlib.Path(R) / sub
    return [str(p) for p in sorted(base.rglob(pattern)) if p.is_file()]


GROUPS: list[tuple[str, list[str]]] = [
    ("Reports and manifest", [
        f"{R}/README.md",
        f"{R}/FINAL_REPORT.md",
        f"{R}/EXECUTION_MANIFEST.md",
    ]),
    ("Gate evaluation, paired transitions and the C/U case-judgment record", [
        f"{R}/evidence/ANALYSIS.json",
        f"{R}/evidence/CASE_JUDGMENTS.json",
        f"{R}/evidence/DP_SEQ_chain_saw_transitions.csv",
        f"{R}/evidence/UCE_SEQ_chain_saw_transitions.csv",
        f"{R}/evidence/panel_hashes.csv",
        f"{R}/evidence/HOST_ONLY_ARTIFACTS.json",
    ]),
    ("Per-image classifier predictions — all 4,000 newly generated images",
     listing("evidence/per_image")),
    ("Technical verification — parent identity, bitwise load before module copy, keys, "
     "dtype, finiteness, save/reload", listing("evidence/verification")),
    ("Measured preservation residuals (pre- and post-cast, with numerical context)",
     listing("evidence/residuals")),
    ("Code actually used — parent-loading adapter, residual logger, classifier wrapper, "
     "analysis, panel builder", listing("evidence/code")),
    ("Visual evidence — readable per-class comparison grids (M0 | A-parent | B-direct | SEQ)",
     listing("images/grids")),
    ("Visual evidence — frozen 20-case rendering panel at full 512x512 resolution, "
     "all seven compared endpoints", listing("images/full_resolution")),
]


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    L = [
        "# Published evidence index — executed sequential-erasure diagnostic "
        "(chain saw -> parachute)",
        "",
        "Execution of protocol commit `c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80`.",
        "",
        f"Every path below is **in the repository** on branch `{branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "**Deliberately excluded** (repo policy, hashed instead in",
        "`evidence/HOST_ONLY_ARTIFACTS.json`): the four edited `.safetensors`",
        "checkpoints and the full 4,000-image raw generation output, both on",
        "`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/`; plus the pinned",
        "upstream source tree (`czhang024/CVPR_DouleProjection` @",
        "`3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`, MIT, unmodified — cited, not vendored)",
        "and this host's venv/model caches.", "",
        "Regenerate: `python scripts/seq/publish_modern_flux_dp_uce_execution_evidence_index.py`",
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
