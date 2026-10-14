#!/usr/bin/env python3
"""Write results/rece_seq_golfball_englishspringer/PUBLISHED_EVIDENCE_INDEX.md.
CPU only. Same convention as scripts/seq/publish_evidence_index.py: exact
path and sha256 of every published file in this result set, so a reviewer
can verify any single row without trusting the table:

    git show <commit>:<path> | sha256sum

Run this LAST, after every other file in the result set is final.

    python scripts/seq/publish_rece_evidence_index.py
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess

OUT = "results/rece_seq_golfball_englishspringer/PUBLISHED_EVIDENCE_INDEX.md"
R = "results/rece_seq_golfball_englishspringer"

GROUPS: list[tuple[str, list[str]]] = [
    ("Reports", [
        f"{R}/FINAL_REPORT.md",
        f"{R}/refinement_ablation/FINAL_REPORT.md",
        f"{R}/refinement_ablation/CLOSEOUT.md",
        f"{R}/CORRECTIONS.md",
    ]),
    ("Recognition tables and visual-review labels — direct-vs-sequential experiment", [
        f"{R}/evidence/ANALYSIS_RESULTS.json",
        f"{R}/evidence/VISUAL_REVIEW.json",
        f"{R}/evidence/RUN_MANIFEST.md",
        f"{R}/evidence/ELIGIBILITY.md",
    ]),
    ("Recognition tables and visual-review labels — stage-ablation diagnostic", [
        f"{R}/evidence/refinement_ablation/ANALYSIS_RESULTS.json",
        f"{R}/evidence/refinement_ablation/VISUAL_REVIEW.json",
    ]),
    ("Edit logs — per-edit checkpoint hashes and runtime verification", [
        f"{R}/evidence/edit_logs/B_english_springer.edit_log.json",
        f"{R}/evidence/edit_logs/AtoB_english_springer.edit_log.json",
        f"{R}/evidence/refinement_ablation/edit_logs/B_UCE.uce_only_log.json",
        f"{R}/evidence/refinement_ablation/edit_logs/AB_UCE.uce_only_log.json",
    ]),
    ("Upstream provenance — shared frozen manifest, checkpoint/source identity, prior keyed labels", [
        f"{R}/evidence/upstream/frozen_manifest.json",
        f"{R}/evidence/upstream/rece_golfball_author_screen_RUN_MANIFEST.md",
        f"{R}/evidence/upstream/rece_golfball_confirmation_VISUAL_REVIEW.json",
    ]),
    ("Code — vendored native-source edit functions and the intermediate-reconstruction script", [
        f"{R}/evidence/code/vendored_functions.py",
        f"{R}/evidence/code/run_edit.py",
        f"{R}/evidence/code/make_uce_only_checkpoint.py",
    ]),
    ("Visual evidence — fixed-order review grids (golf ball + English springer only; French horn/parachute stayed clean at every endpoint and are not pictured)", [
        f"{R}/images/grids/B/golf_ball.png",
        f"{R}/images/grids/B/English_springer.png",
        f"{R}/images/grids/AtoB/golf_ball.png",
        f"{R}/images/grids/AtoB/English_springer.png",
        f"{R}/images/grids/B_UCE/golf_ball.png",
        f"{R}/images/grids/B_UCE/English_springer.png",
        f"{R}/images/grids/AB_UCE/golf_ball.png",
        f"{R}/images/grids/AB_UCE/English_springer.png",
    ]),
    ("Visual evidence — full-resolution crops for the corrected and ambiguous cases (CORRECTIONS.md C1/C2)", [
        f"{R}/images/corrected_cases/AtoB_case22.png",
        f"{R}/images/corrected_cases/AB_UCE_case22.png",
        f"{R}/images/corrected_cases/AtoB_case34.png",
        f"{R}/images/corrected_cases/AB_UCE_case34.png",
        f"{R}/images/corrected_cases/AtoB_case39.png",
        f"{R}/images/corrected_cases/AB_UCE_case39.png",
        f"{R}/images/corrected_cases/AtoB_case84.png",
        f"{R}/images/corrected_cases/AB_UCE_case84.png",
        f"{R}/images/corrected_cases/AtoB_case88.png",
        f"{R}/images/corrected_cases/AB_UCE_case88.png",
        f"{R}/images/corrected_cases/AtoB_case92.png",
        f"{R}/images/corrected_cases/AB_UCE_case92.png",
        f"{R}/images/corrected_cases/AtoB_case94.png",
        f"{R}/images/corrected_cases/AB_UCE_case94.png",
    ]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", default="research/audit-and-matched-effectiveness-protocol")
    a = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    L = [
        "# Published evidence index — RECE golf-ball/English-springer sequential editing",
        "",
        f"Every path below is **in the repository** on branch `{a.branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {a.branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "**Deliberately excluded, unchanged:** the ~3.44 GB author-released",
        "`GolfBall_ep0.pt` checkpoint and the pristine SD1.4 base snapshot",
        "(identified by sha256 in `evidence/upstream/rece_golfball_author_screen_RUN_MANIFEST.md`",
        "and in the edit logs, not vendored here), the pinned RECE GitHub source",
        "tree (public at commit `403a67fd86dce6c124b88798340e001e625b28c9`,",
        "https://github.com/CharlesGong12/RECE — the two functions actually used",
        "are vendored verbatim in `evidence/code/vendored_functions.py` instead",
        "of cloning the whole tree), the full 400-image raw generation output for",
        "both experiments (only the review grids and the specific corrected/",
        "ambiguous case crops are published), and this host's venv/caches.", "",
        "Regenerate: `python scripts/seq/publish_rece_evidence_index.py` — run it",
        "LAST, after every other file in this result set is final.", "",
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
