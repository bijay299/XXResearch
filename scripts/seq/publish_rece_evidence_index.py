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
        f"{R}/recovery_optimality_audit/FINAL_REPORT.md",
        f"{R}/preservation_content_control/FINAL_REPORT.md",
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
    ("Recovery-optimality audit — does the native embedding-recovery output (E_D/E_S, close_form_emb_regzero) accurately solve its own stated closed-form objective? (CPU-only, fp64 reference solve; different computation from the numerical-optimality audit above) plus the corrected AB_STABLE_SOLVE golf-ball case table (C17)", [
        f"{R}/evidence/recovery_optimality_audit/FROZEN_MANIFEST.json",
        f"{R}/evidence/recovery_optimality_audit/RESULT.json",
        f"{R}/evidence/recovery_optimality_audit/CASE_LEVEL_AB_STABLE_SOLVE_GOLF_CORRECTED.json",
        f"{R}/evidence/code/run_recovery_optimality_audit.py",
    ]),
    ("Preservation-content control — does RECE's existing preservation term, pointed at the already-erased golf-ball response, prevent the golf-ball corruption characterized throughout this results line, and does it beat an unrelated-content control ('French horn')? Two new endpoints (AB_RETAIN_GOLF, AB_RETAIN_HORN), fp64 Cholesky build + validation, frozen pre-classification visual review of all 200 generated images, joint comparison against the existing AB_STABLE_SOLVE comparator", [
        f"{R}/evidence/preservation_content_control/FROZEN_MANIFEST.json",
        f"{R}/evidence/preservation_content_control/RETAIN_ENCODING_PROVENANCE.json",
        f"{R}/evidence/preservation_content_control/RESULT_GOLF.json",
        f"{R}/evidence/preservation_content_control/RESULT_HORN.json",
        f"{R}/evidence/preservation_content_control/VISUAL_REVIEW_INITIAL.json",
        f"{R}/evidence/preservation_content_control/CASE_LEVEL_COMPLETION_RECORD.json",
        f"{R}/evidence/preservation_content_control/CLASSIFIER_PRECONDITION_GATE.json",
        f"{R}/evidence/preservation_content_control/JOINT_ANALYSIS.json",
        f"{R}/evidence/preservation_content_control/AB_RETAIN_GOLF.image_report.json",
        f"{R}/evidence/preservation_content_control/AB_RETAIN_GOLF.classifications.jsonl",
        f"{R}/evidence/preservation_content_control/AB_RETAIN_GOLF.classify_report.json",
        f"{R}/evidence/preservation_content_control/AB_RETAIN_HORN.image_report.json",
        f"{R}/evidence/preservation_content_control/AB_RETAIN_HORN.classifications.jsonl",
        f"{R}/evidence/preservation_content_control/AB_RETAIN_HORN.classify_report.json",
        f"{R}/evidence/code/encode_retain_texts.py",
        f"{R}/evidence/code/run_build_arm.py",
        f"{R}/evidence/code/build_visual_review.py",
    ]),
    ("Visual evidence — preservation-content control review grids and disputed/representative full-resolution cases (case 30's classifier false positive, case 39's lone persisting degeneration under historical preservation, case 16's lone recovery under the unrelated control, one representative unrelated-control degenerate case)", [
        f"{R}/images/grids/AB_RETAIN_GOLF/golf_ball.png",
        f"{R}/images/grids/AB_RETAIN_GOLF/English_springer.png",
        f"{R}/images/grids/AB_RETAIN_GOLF/parachute.png",
        f"{R}/images/grids/AB_RETAIN_GOLF/French_horn.png",
        f"{R}/images/grids/AB_RETAIN_HORN/golf_ball.png",
        f"{R}/images/grids/AB_RETAIN_HORN/English_springer.png",
        f"{R}/images/grids/AB_RETAIN_HORN/parachute.png",
        f"{R}/images/grids/AB_RETAIN_HORN/French_horn.png",
        f"{R}/images/disputed_cases/AB_RETAIN_GOLF_case30_classifier_false_positive.png",
        f"{R}/images/disputed_cases/AB_RETAIN_GOLF_case39_only_degenerate.png",
        f"{R}/images/disputed_cases/AB_RETAIN_HORN_case16_only_nondegenerate.png",
        f"{R}/images/disputed_cases/AB_RETAIN_HORN_case0_representative_degenerate.png",
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
    ("Reports — history_preservation_confirmation and stage1_history_intervention", [
        f"{R}/history_preservation_confirmation/FINAL_REPORT.md",
        f"{R}/history_preservation_confirmation/closeout/CLOSEOUT_CORRECTIONS.md",
        f"{R}/history_preservation_confirmation/closeout/VERIFICATION_AND_CORRECTIONS_2.md",
        f"{R}/stage1_history_intervention/FINAL_REPORT.md",
    ]),
    ("Evidence — history_preservation_confirmation (manifests, provenance, verification logs, joint tables)", [
        f"{R}/history_preservation_confirmation/evidence/FROZEN_MANIFEST.json",
        f"{R}/history_preservation_confirmation/evidence/FRESH_PANEL_MANIFEST.json",
        f"{R}/history_preservation_confirmation/evidence/FRESH_PANEL_MANIFEST.sha256",
        f"{R}/history_preservation_confirmation/evidence/ELIGIBILITY.json",
        f"{R}/history_preservation_confirmation/evidence/expected_32_keys.json",
        f"{R}/history_preservation_confirmation/evidence/VISUAL_REVIEW_INITIAL_SHARED_INPUTS.json",
        f"{R}/history_preservation_confirmation/evidence/CASE_LEVEL_COMPLETION_RECORD_SHARED_INPUTS.json",
        f"{R}/history_preservation_confirmation/evidence/CLASSIFIER_PRECONDITION_GATE_SHARED_INPUTS.json",
        f"{R}/history_preservation_confirmation/evidence/VISUAL_REVIEW_INITIAL_NEW_ENDPOINTS.json",
        f"{R}/history_preservation_confirmation/evidence/CASE_LEVEL_COMPLETION_RECORD_NEW_ENDPOINTS.json",
        f"{R}/history_preservation_confirmation/evidence/CLASSIFIER_PRECONDITION_GATE_NEW_ENDPOINTS.json",
        f"{R}/history_preservation_confirmation/evidence/VISUAL_JOINT_RECOMPUTED.json",
        f"{R}/history_preservation_confirmation/evidence/history_preservation_confirmation_JOINT_ANALYSIS.json",
        f"{R}/history_preservation_confirmation/evidence/stage1_logs/S1_stage1.uce_only_log.json",
        f"{R}/history_preservation_confirmation/evidence/stage1_logs/S2_stage1.uce_only_log.json",
        f"{R}/history_preservation_confirmation/evidence/recover/RECOVER_ENCODE_S1_RESULT.json",
        f"{R}/history_preservation_confirmation/evidence/recover/RECOVER_ENCODE_S2_RESULT.json",
        f"{R}/history_preservation_confirmation/evidence/arm_builds/RESULT_S1_BLANK.json",
        f"{R}/history_preservation_confirmation/evidence/arm_builds/RESULT_S1_HISTORY.json",
        f"{R}/history_preservation_confirmation/evidence/arm_builds/RESULT_S2_BLANK.json",
        f"{R}/history_preservation_confirmation/evidence/arm_builds/RESULT_S2_HISTORY.json",
        f"{R}/history_preservation_confirmation/closeout/ORIGINAL_IMAGES_HASH_MAPPING.json",
        f"{R}/history_preservation_confirmation/closeout/JOINT_ANALYSIS_CORRECTIONS.json",
    ] + [
        f"{R}/history_preservation_confirmation/evidence/image_reports/{ck}/{f}"
        for ck in ["M0", "GolfBall_parent", "EnglishSpringer_parent", "S1_Stage1", "S1_BLANK", "S1_HISTORY",
                   "S2_Stage1", "S2_BLANK", "S2_HISTORY"]
        for f in ["image_report.json", "classifications.jsonl", "classify_report.json"]
    ] + [
        f"{R}/history_preservation_confirmation/evidence/code/{f}"
        for f in ["encode_retain_texts.py", "run_build_arm.py", "build_visual_review.py",
                   "recover_and_encode.py", "run_build_arm_seq.py", "review_lib.py",
                   "build_fresh_panel.py", "build_shared_input_review.py", "build_new_endpoints_review.py",
                   "make_uce_only_checkpoint.py"]
    ]),
    ("Visual evidence — preservation_content_control, all 120 original target images (both arms), retroactively published", [
        f"{R}/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_GOLF/case{c:02d}_{cls}.png"
        for c, cls in [(i, "golf_ball") for i in range(40)] + [(i, "English_springer") for i in range(80, 100)]
    ] + [
        f"{R}/images/history_preservation_confirmation/preservation_content_control_originals/AB_RETAIN_HORN/case{c:02d}_{cls}.png"
        for c, cls in [(i, "golf_ball") for i in range(40)] + [(i, "English_springer") for i in range(80, 100)]
    ]),
    ("Visual evidence — history_preservation_confirmation, all 380 individually-reviewed target-role full-resolution originals (M0/parents/Stage-1 references + 6 new endpoints) and 17 fixed-order retained-class grids", [
        f"{R}/images/history_preservation_confirmation/target_role_full_res/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/target_role_full_res").glob("*.png"))
    ] + [
        f"{R}/images/history_preservation_confirmation/review_grids/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/review_grids").glob("*.png"))
    ] + [
        f"{R}/images/history_preservation_confirmation/disputed_cases/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/history_preservation_confirmation/disputed_cases").glob("*.png"))
    ]),
    ("Evidence — stage1_history_intervention (manifest, provenance, verification, joint outcomes)", [
        f"{R}/stage1_history_intervention/evidence/FROZEN_MANIFEST.json",
        f"{R}/stage1_history_intervention/evidence/S1_STAGE1_HISTORY.stage1_history_log.json",
        f"{R}/stage1_history_intervention/evidence/CASE_LEVEL_COMPLETION_RECORD.json",
        f"{R}/stage1_history_intervention/evidence/VISUAL_REVIEW_INITIAL.json",
        f"{R}/stage1_history_intervention/evidence/CLASSIFIER_PRECONDITION_GATE.json",
        f"{R}/stage1_history_intervention/evidence/image_report.json",
        f"{R}/stage1_history_intervention/evidence/classifications.jsonl",
        f"{R}/stage1_history_intervention/evidence/classify_report.json",
        f"{R}/stage1_history_intervention/evidence/JOINT_OUTCOMES_COMPUTED.txt",
        f"{R}/stage1_history_intervention/evidence/code/make_stage1_history_checkpoint.py",
        f"{R}/stage1_history_intervention/evidence/code/build_review.py",
    ]),
    ("Visual evidence — stage1_history_intervention, all 80 new original images and 2 retained-class grids", [
        f"{R}/images/stage1_history_intervention/original_images/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/original_images").glob("*.png"))
    ] + [
        f"{R}/images/stage1_history_intervention/review_grids/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/stage1_history_intervention/review_grids").glob("*.png"))
    ]),
    ("Corrections bundled ahead of the S2 Stage-1 history intervention (C29): S1 cases 12-15 relabeling, false claim removal, M0 parachute fix", [
        f"{R}/stage1_history_intervention/closeout/CORRECTIONS_BEFORE_S2.md",
        f"{R}/stage1_history_intervention/closeout/CORRECTED_LABELS.json",
    ]),
    ("Reports — stage1_history_intervention_s2 (C30): does the S1 Stage-1 fix generalize to the reverse sequence?", [
        f"{R}/stage1_history_intervention_s2/FINAL_REPORT.md",
    ]),
    ("Evidence — stage1_history_intervention_s2 (manifest, provenance, verification, classifier invocation guard, joint outcomes)", [
        f"{R}/stage1_history_intervention_s2/evidence/FROZEN_MANIFEST.json",
        f"{R}/stage1_history_intervention_s2/evidence/S2_STAGE1_HISTORY.stage1_history_log.json",
        f"{R}/stage1_history_intervention_s2/evidence/CASE_LEVEL_COMPLETION_RECORD.json",
        f"{R}/stage1_history_intervention_s2/evidence/VISUAL_REVIEW_INITIAL.json",
        f"{R}/stage1_history_intervention_s2/evidence/CLASSIFIER_PRECONDITION_GATE.json",
        f"{R}/stage1_history_intervention_s2/evidence/CLASSIFIER_INVOCATION_GUARD_RESULT.json",
        f"{R}/stage1_history_intervention_s2/evidence/image_report.json",
        f"{R}/stage1_history_intervention_s2/evidence/classifications.jsonl",
        f"{R}/stage1_history_intervention_s2/evidence/classify_report.json",
        f"{R}/stage1_history_intervention_s2/evidence/code/make_stage1_history_checkpoint_s2.py",
        f"{R}/stage1_history_intervention_s2/evidence/code/build_review_s2.py",
        f"{R}/stage1_history_intervention_s2/evidence/code/make_review_grids_s2.py",
    ]),
    ("Visual evidence — stage1_history_intervention_s2, all 80 new original images and 2 retained-class grids", [
        f"{R}/images/stage1_history_intervention_s2/original_images/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/original_images").glob("*.png"))
    ] + [
        f"{R}/images/stage1_history_intervention_s2/review_grids/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/stage1_history_intervention_s2/review_grids").glob("*.png"))
    ]),
    ("Corrections bundled ahead of the S2 both-stages build (C31): objective reframing, springer case 63, retained cases 32/54, S1 arithmetic propagation, verification-claim honesty", [
        f"{R}/history_both_stages_s2/closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md",
        f"{R}/history_both_stages_s2/closeout/CORRECTED_LABELS.json",
    ]),
    ("Reports — history_both_stages_s2 (C32): does the S2 Stage-1 quality gain survive full-pipeline refinement?", [
        f"{R}/history_both_stages_s2/FINAL_REPORT.md",
    ]),
    ("Evidence — history_both_stages_s2 (manifest, recovery result, build log, visual review, classifier guard invocation log, joint outcomes)", [
        f"{R}/history_both_stages_s2/evidence/FROZEN_MANIFEST.json",
        f"{R}/history_both_stages_s2/evidence/RECOVER_S2_BOTH_STAGES_RESULT.json",
        f"{R}/history_both_stages_s2/evidence/S2_HISTORY_BOTH_STAGES.build_log.json",
        f"{R}/history_both_stages_s2/evidence/CASE_LEVEL_COMPLETION_RECORD.json",
        f"{R}/history_both_stages_s2/evidence/VISUAL_REVIEW_INITIAL.json",
        f"{R}/history_both_stages_s2/evidence/CLASSIFIER_PRECONDITION_GATE.json",
        f"{R}/history_both_stages_s2/evidence/CLASSIFIER_GUARD_INVOCATION_LOG.json",
        f"{R}/history_both_stages_s2/evidence/image_report.json",
        f"{R}/history_both_stages_s2/evidence/classifications.jsonl",
        f"{R}/history_both_stages_s2/evidence/classify_report.json",
        f"{R}/history_both_stages_s2/evidence/code/recover_newtarget_s2_both_stages.py",
        f"{R}/history_both_stages_s2/evidence/code/build_both_stages_s2.py",
        f"{R}/history_both_stages_s2/evidence/code/build_review_both_stages.py",
        f"{R}/history_both_stages_s2/evidence/code/guarded_classify.py",
    ]),
    ("Visual evidence — history_both_stages_s2, all 80 new original images", [
        f"{R}/images/history_both_stages_s2/original_images/{f.name}"
        for f in sorted(pathlib.Path("results/rece_seq_golfball_englishspringer/images/history_both_stages_s2/original_images").glob("*.png"))
    ]),
    ("Blinded human review package (C33) — protocol, manifest, mapping commitment, bounded corrections, prep code, and the reviewer zip itself. "
     "The private neutral-ID mapping and its random seed are intentionally NOT listed here (kept outside the git checkout; only a SHA256 commitment is published).", [
        f"{R}/blinded_quality_review_s2/FROZEN_PROTOCOL.md",
        f"{R}/blinded_quality_review_s2/VALIDATION_REPORT.md",
        f"{R}/blinded_quality_review_s2/PACKAGE_MANIFEST.json",
        f"{R}/blinded_quality_review_s2/MAPPING_COMMITMENT.json",
        f"{R}/blinded_quality_review_s2/READY_FOR_HUMAN_REVIEW.md",
        f"{R}/blinded_quality_review_s2/closeout/CORRECTIONS_BEFORE_HUMAN_REVIEW.md",
        f"{R}/blinded_quality_review_s2/prep_code/build_reviewer_package.py",
        f"{R}/blinded_quality_review_s2/prep_code/verify_source_images.py",
        f"{R}/blinded_quality_review_s2/BLINDED_REVIEW_PACKAGE.zip",
    ]),
    ("Unblinded human review response (C34) — preserved selected response, receipt, mapping verification, unblinded mapping/joined table, case-level agreement with assistant labels, and the final human review report. "
     "An earlier survey submission was explicitly withdrawn by the user and is intentionally NOT listed or published here.", [
        f"{R}/blinded_quality_review_s2/responses/review_export_final_2026-10-16T14-06-23-096Z.json",
        f"{R}/blinded_quality_review_s2/responses/RESPONSE_RECEIPT.json",
        f"{R}/blinded_quality_review_s2/closeout/UNBLINDING_VERIFICATION.json",
        f"{R}/blinded_quality_review_s2/closeout/UNBLINDED_MAPPING.json",
        f"{R}/blinded_quality_review_s2/closeout/UNBLINDED_JOINED_TABLE.json",
        f"{R}/blinded_quality_review_s2/closeout/CASE_LEVEL_AGREEMENT_TABLE.json",
        f"{R}/blinded_quality_review_s2/closeout/unblind_and_analyze.py",
        f"{R}/blinded_quality_review_s2/closeout/compare_with_assistant_labels.py",
        f"{R}/blinded_quality_review_s2/closeout/HUMAN_REVIEW_REPORT.md",
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
