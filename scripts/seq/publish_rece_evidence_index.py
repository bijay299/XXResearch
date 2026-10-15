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
        f"{R}/reference_A_intervention/FINAL_REPORT.md",
        f"{R}/embedding_cross_swap/FINAL_REPORT.md",
        f"{R}/numerical_optimality_audit/FINAL_REPORT.md",
        f"{R}/solver_only_intervention/FINAL_REPORT.md",
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
    ("Reference-A intervention diagnostic — does using parent A instead of M0 as the close_form_emb_regzero reference rescue the sequential corruption?", [
        f"{R}/evidence/reference_A_intervention/FROZEN_MANIFEST.json",
        f"{R}/evidence/reference_A_intervention/VERIFICATION_LOG.json",
        f"{R}/evidence/reference_A_intervention/UPDATE_NORMS.json",
        f"{R}/evidence/reference_A_intervention/ANALYSIS_RESULTS.json",
        f"{R}/evidence/reference_A_intervention/VISUAL_REVIEW.json",
        f"{R}/evidence/reference_A_intervention/edit_logs/AB_REF_A.ref_a_intervention_log.json",
        f"{R}/evidence/reference_A_intervention/AB_REF_A.classifications.jsonl",
        f"{R}/evidence/reference_A_intervention/AB_REF_A.classify_report.json",
        f"{R}/evidence/code/run_reference_A_intervention.py",
    ]),
    ("Embedding cross-swap diagnostic — does corruption follow the recovered embedding, the working checkpoint, or their interaction? (reference held fixed at M0 throughout)", [
        f"{R}/evidence/embedding_cross_swap/FROZEN_MANIFEST.json",
        f"{R}/evidence/embedding_cross_swap/VERIFICATION_LOG.json",
        f"{R}/evidence/embedding_cross_swap/DIAGNOSTICS.json",
        f"{R}/evidence/embedding_cross_swap/PER_LAYER_DISPLACEMENT.json",
        f"{R}/evidence/embedding_cross_swap/ANALYSIS_RESULTS.json",
        f"{R}/evidence/embedding_cross_swap/VISUAL_REVIEW.json",
        f"{R}/evidence/embedding_cross_swap/EMBEDDING_PROVENANCE.json",
        f"{R}/evidence/embedding_cross_swap/edits/D_WITH_SEQ_EMB.cross_swap_log.json",
        f"{R}/evidence/embedding_cross_swap/edits/S_WITH_DIRECT_EMB.cross_swap_log.json",
        f"{R}/evidence/embedding_cross_swap/D_WITH_SEQ_EMB.classifications.jsonl",
        f"{R}/evidence/embedding_cross_swap/D_WITH_SEQ_EMB.classify_report.json",
        f"{R}/evidence/embedding_cross_swap/S_WITH_DIRECT_EMB.classifications.jsonl",
        f"{R}/evidence/embedding_cross_swap/S_WITH_DIRECT_EMB.classify_report.json",
        f"{R}/evidence/code/run_embedding_cross_swap.py",
        f"{R}/evidence/code/compute_diagnostics.py",
        f"{R}/evidence/reference_A_intervention/CORRECTED_LABELS.json",
    ]),
    ("Numerical-optimality audit — does the native closed-form UCE solve actually solve its own stated regularized objective, for all four embedding_cross_swap cells and all 32 edited K/V layers? (CPU-only, fp64 reference solve)", [
        f"{R}/evidence/numerical_optimality_audit/FROZEN_MANIFEST.json",
        f"{R}/evidence/numerical_optimality_audit/RESULT.json",
        f"{R}/evidence/numerical_optimality_audit/CASE_LEVEL_COMPLETION_RECORD.json",
        f"{R}/evidence/code/run_numerical_optimality_audit.py",
    ]),
    ("Solver-only intervention — AB_STABLE_SOLVE: does accurately solving the existing refinement equations for the native sequential (AtoB) checkpoint remove the original corruption while preserving deletion and retention?", [
        f"{R}/evidence/solver_only_intervention/FROZEN_MANIFEST.json",
        f"{R}/evidence/solver_only_intervention/OPERAND_PROVENANCE_CHECK.json",
        f"{R}/evidence/solver_only_intervention/RESULT.json",
        f"{R}/evidence/solver_only_intervention/VISUAL_REVIEW.json",
        f"{R}/evidence/solver_only_intervention/CASE_LEVEL_COMPLETION_RECORD_120.json",
        f"{R}/evidence/solver_only_intervention/classifications.jsonl",
        f"{R}/evidence/solver_only_intervention/classify_report.json",
        f"{R}/evidence/code/run_operand_provenance_check.py",
        f"{R}/evidence/code/run_solver_only_build.py",
        f"{R}/evidence/code/make_review_grids_solver_only.py",
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
        f"{R}/images/grids/AB_REF_A/golf_ball.png",
        f"{R}/images/grids/AB_REF_A/English_springer.png",
        f"{R}/images/grids/D_WITH_SEQ_EMB/golf_ball.png",
        f"{R}/images/grids/D_WITH_SEQ_EMB/English_springer.png",
        f"{R}/images/grids/S_WITH_DIRECT_EMB/golf_ball.png",
        f"{R}/images/grids/S_WITH_DIRECT_EMB/English_springer.png",
        f"{R}/images/grids/AB_STABLE_SOLVE/golf_ball.png",
        f"{R}/images/grids/AB_STABLE_SOLVE/parachute.png",
        f"{R}/images/grids/AB_STABLE_SOLVE/French_horn.png",
        f"{R}/images/grids/AB_STABLE_SOLVE/English_springer.png",
    ]),
    ("Visual evidence — AB_STABLE_SOLVE, all 60 new target images at full 512x512 resolution (40 golf ball + 20 English springer; parachute/French horn stayed clean, pictured only in the grids above)", [
        f"{R}/images/full_resolution/AB_STABLE_SOLVE/golf_ball/case{i}.png" for i in range(40)
    ] + [
        f"{R}/images/full_resolution/AB_STABLE_SOLVE/english_springer/case{i}.png" for i in range(80, 100)
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
    ("Visual evidence — AB_REF_A full-resolution case crops (reference-A intervention: recovered, newly-damaged, persisted-degenerate, and the English-springer exact-hit case)", [
        f"{R}/images/corrected_cases/AB_REF_A_case1.png",
        f"{R}/images/corrected_cases/AB_REF_A_case15.png",
        f"{R}/images/corrected_cases/AB_REF_A_case28.png",
        f"{R}/images/corrected_cases/AB_REF_A_case29.png",
        f"{R}/images/corrected_cases/AB_REF_A_case30.png",
        f"{R}/images/corrected_cases/AB_REF_A_case31.png",
        f"{R}/images/corrected_cases/AB_REF_A_case33.png",
        f"{R}/images/corrected_cases/AB_REF_A_case84.png",
    ]),
    ("Visual evidence — embedding cross-swap full-resolution case crops (D_WITH_SEQ_EMB's 'flying ball' corruption spillover; S_WITH_DIRECT_EMB's recovered golf-ball cases and its one dog-present English-springer case)", [
        f"{R}/images/corrected_cases/D_WITH_SEQ_EMB_case33.png",
        f"{R}/images/corrected_cases/D_WITH_SEQ_EMB_case34.png",
        f"{R}/images/corrected_cases/D_WITH_SEQ_EMB_case35.png",
        f"{R}/images/corrected_cases/D_WITH_SEQ_EMB_case37.png",
        f"{R}/images/corrected_cases/D_WITH_SEQ_EMB_case38.png",
        f"{R}/images/corrected_cases/D_WITH_SEQ_EMB_case39.png",
        f"{R}/images/corrected_cases/S_WITH_DIRECT_EMB_case1.png",
        f"{R}/images/corrected_cases/S_WITH_DIRECT_EMB_case17.png",
        f"{R}/images/corrected_cases/S_WITH_DIRECT_EMB_case35.png",
        f"{R}/images/corrected_cases/S_WITH_DIRECT_EMB_case90.png",
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
