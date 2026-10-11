#!/usr/bin/env python3
"""Write results/audit_v1/PUBLISHED_EVIDENCE_INDEX.md. CPU only.

Exact path and sha256 of every published, key-free deliverable, so a reviewer
can verify any single row without trusting the table:

    git show <commit>:<path> | sha256sum

Run this LAST, after every other file is final: the index is a snapshot of
bytes, and editing a listed file afterwards makes its row stale. The index
excludes itself for the same reason.

A listed file that is absent is written as ABSENT rather than skipped.

    python scripts/seq/publish_evidence_index.py
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess

OUT = "results/audit_v1/PUBLISHED_EVIDENCE_INDEX.md"
E = "results/audit_v1/amendment01_evidence"
C = "results/audit_v1/amendment01_checks"
P = "results/audit_v1/annotation_packet_paired180"

GROUPS: list[tuple[str, list[str]]] = [
    ("Corrected analysis and its preserved predecessors", [
        f"{E}/analysis_test.json",
        f"{E}/analysis_test.20261010T205657.superseded.json",
        f"{E}/analysis_test.20261010T205055.superseded.json",
    ]),
    ("Raw detector rows — selection and every interval re-derivable "
     "without images or weights", [
        f"{E}/development_slots.tar.gz",
        f"{E}/development_slots_index.json",
        f"{E}/frozen_test_slots.tar.gz",
        f"{E}/frozen_test_slots_index.json",
    ]),
    ("Decision, bridge, binding, accounting, and the supervisor log", [
        f"{E}/selection.json",
        f"{E}/bridge_check.json",
        f"{E}/reference_bindings.json",
        f"{E}/STATUS.json",
        f"{E}/unattended.log",
        f"{E}/gpu_usage_record.json",
        f"{E}/budget_reconciliation.json",
    ]),
    ("Manifests", [
        f"{E}/MANIFEST.json",
        f"{E}/MANIFEST.published_at_7e01fbd.json",
    ]),
    ("Source-bound test logs and source identity", [
        f"{C}/CLOSEOUT_SOURCE_HASHES.json",
        f"{C}/closeout_test_diag_analysis.log",
        f"{C}/closeout_test_diag_test_conditions.log",
    ]),
    ("Prescribed paired audit — key-free records only", [
        f"{P}/README.md",
        f"{P}/BLINDING_AUDIT.md",
        f"{P}/blinding_audit.log",
        f"{P}/label_sheet_EMPTY.csv",
        f"{P}/packet_manifest.json",
        f"{P}/INSTRUCTIONS.md",
        f"{P}/overlap_summary.json",
    ]),
    ("Write-ups", [
        "results/audit_v1/AMENDMENT_01_CLOSEOUT.md",
        "results/audit_v1/AMENDMENT_01_RESULTS.md",
        "results/audit_v1/CLAIM_CORRECTIONS_V5.md",
        "results/audit_v1/ARTIFACT_INVENTORY.md",
        "docs/research/NEXT_SCREEN_PROPOSAL.md",
    ]),
    ("Code", [
        "scripts/seq/diag_analysis.py",
        "scripts/seq/test_diag_analysis.py",
        "scripts/seq/test_diag_test_conditions.py",
        "scripts/seq/build_paired_annotation_packet.py",
        "scripts/seq/build_amendment01_evidence.py",
        "scripts/seq/audit_packet_blinding.py",
        "scripts/seq/publish_evidence_index.py",
    ]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch",
                    default="research/audit-and-matched-effectiveness-protocol")
    a = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    L = [
        "# Published evidence index — exact paths and identities", "",
        f"Every path below is **in the repository** on branch `{a.branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {a.branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "**No annotation key, key-derived field, blinding salt or label value",
        "appears in any file listed here.** Keys and salts live only under the",
        "private data root; this was checked by grepping the repository for the",
        "secret values and for any `item_id` line co-occurring with an arm or",
        "training-seed identifier (0 hits each).", "",
        "Deliberately excluded, unchanged: model weights (digests only, in",
        "`selection.json` and `bridge_check.json`) and generated images (~592 MB;",
        "per-image digests are inside the two archives).", "",
        "Regenerate: `python scripts/seq/publish_evidence_index.py` — run it LAST,",
        "after every listed file is final.", "",
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
