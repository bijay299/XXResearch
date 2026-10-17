#!/usr/bin/env python3
"""Write results/modern_flux_dp_uce/PUBLISHED_EVIDENCE_INDEX.md.
Same convention as scripts/seq/publish_rece_evidence_index.py: exact path
and sha256 of every published file in this result set, so a reviewer can
verify any single row without trusting the table:

    git show <commit>:<path> | sha256sum

Run this LAST, after every other file in the result set is final.

    python scripts/seq/publish_modern_flux_dp_uce_evidence_index.py
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

OUT = "results/modern_flux_dp_uce/PUBLISHED_EVIDENCE_INDEX.md"
R = "results/modern_flux_dp_uce"

GROUPS: list[tuple[str, list[str]]] = [
    ("Reports", [
        f"{R}/README.md",
        f"{R}/FINAL_REPORT.md",
        f"{R}/RUN_MANIFEST.md",
    ]),
    ("Evidence — per-image classifier predictions (all 5,000 generated images)", [
        f"{R}/evidence/M0_per_image.csv",
        f"{R}/evidence/DP_A_per_image.csv",
        f"{R}/evidence/DP_B_per_image.csv",
        f"{R}/evidence/UCE_A_per_image.csv",
        f"{R}/evidence/UCE_B_per_image.csv",
        f"{R}/evidence/recognition_summary.json",
        f"{R}/evidence/gates_and_retention.txt",
    ]),
    ("Evidence — frozen case manifest, environment, provenance", [
        f"{R}/evidence/full_case_manifest.csv",
        f"{R}/evidence/pip_freeze.txt",
        f"{R}/evidence/downloaded_weight_hashes.txt",
        f"{R}/evidence/panel_hashes.csv",
    ]),
    ("Evidence — native single-edit logs (Double_proxy.py / UCE_original.py stdout)", [
        f"{R}/evidence/edit_DP_A.log",
        f"{R}/evidence/edit_DP_B.log",
        f"{R}/evidence/edit_UCE_A.log",
        f"{R}/evidence/edit_UCE_B.log",
    ]),
    ("Code — exact scripts used (case manifest, evaluation, gate computation, rendering panel)", [
        f"{R}/evidence/code/build_case_manifest.py",
        f"{R}/evidence/code/eval_all.py",
        f"{R}/evidence/code/compute_gates.py",
        f"{R}/evidence/code/build_rendering_panel.py",
    ]),
    ("Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, M0", [
        f"{R}/images/M0/{f.name}"
        for f in sorted(pathlib.Path(f"{R}/images/M0").glob("*.png"))
    ]),
    ("Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, DP_A (cassette player -> box)", [
        f"{R}/images/DP_A/{f.name}"
        for f in sorted(pathlib.Path(f"{R}/images/DP_A").glob("*.png"))
    ]),
    ("Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, DP_B (chain saw -> stick)", [
        f"{R}/images/DP_B/{f.name}"
        for f in sorted(pathlib.Path(f"{R}/images/DP_B").glob("*.png"))
    ]),
    ("Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, UCE_A (cassette player -> box)", [
        f"{R}/images/UCE_A/{f.name}"
        for f in sorted(pathlib.Path(f"{R}/images/UCE_A").glob("*.png"))
    ]),
    ("Visual evidence — fixed 20-case rendering panel, full 512x512 resolution, UCE_B (chain saw -> stick)", [
        f"{R}/images/UCE_B/{f.name}"
        for f in sorted(pathlib.Path(f"{R}/images/UCE_B").glob("*.png"))
    ]),
]


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    L = [
        "# Published evidence index — modern FLUX.1-schnell DP/UCE reproduction",
        "",
        f"Every path below is **in the repository** on branch `{branch}`. Each",
        "`sha256` is of the file's bytes, so a reviewer can verify any single row",
        "without trusting this table:", "",
        f"    git fetch origin {branch}", "",
        "    git show <commit>:<path> | sha256sum", "",
        f"Generated at parent commit `{head[:12]}` — the commit that *records*",
        "these bytes is the next one, so quote that SHA when citing a row.", "",
        "**Deliberately excluded, unchanged:** the full 5,000-image raw generation",
        "output, the four edited `.safetensors` checkpoints (2 tensors each), the",
        "FLUX.1-schnell model snapshot, the isolated venv, and the pinned",
        "(unmodified) upstream source tree `czhang024/CVPR_DouleProjection` at",
        "commit `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (MIT-licensed; not",
        "vendored since it was not modified — cited by commit and file hash in",
        "`RUN_MANIFEST.md` instead). All of the above remain on",
        "`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/` on host \"hyperplane\".",
        "",
        "Regenerate: `python scripts/seq/publish_modern_flux_dp_uce_evidence_index.py`",
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
