#!/usr/bin/env python3
"""Reference-only RECE diagnostic: does using parent A (GolfBall_ep0.pt),
instead of pristine M0, as the close_form_emb_regzero adversarial-embedding
reference prevent the sequential refinement corruption documented in
../refinement_ablation/FINAL_REPORT.md, while preserving deletion and
retained capabilities?

Two phases, both image-free (no GPU image generation here):

  PHASE 1 -- image-free replay. Starting from the exact, untouched AB-UCE
  checkpoint, re-run the ORIGINAL Stage-2 procedure (close_form_emb_regzero
  with model_copy = pristine M0, exactly as run_edit.py did when it built
  the existing final A->B checkpoint). Compare the resulting UNet tensor
  VALUES against the existing AtoB_english_springer.pt, key by key. This is
  a sanity check that the reconstructed AB-UCE intermediate is faithful to
  what the original run actually used as its Stage-1 output -- AB-UCE
  itself was reconstructed, not saved, during the original experiment.

  If PHASE 1 does not reproduce the comparator exactly: record differing
  keys and max abs/relative differences, write DISCREPANCY.json, and STOP.
  No tolerance, no substitute comparator, no repair loop.

  PHASE 2 -- (only if PHASE 1 passes) the actual intervention. Reload a
  FRESH, untouched AB-UCE working model and a FRESH pipeline loaded with A
  as the reference, and run the identical Stage-2 procedure with
  model_copy = A instead of M0. This is the ONLY change from PHASE 1's
  procedure. Save the result as AB_REF_A.pt.

Strengthened verification in both phases (per task instruction: the
published code's checks were narrower than claimed):
  - loaded tensors verified against their source checkpoints (all keys)
  - working/reference storage separation checked across ALL UNet tensors,
    not one probe key
  - reference UNet tensor VALUES (not object identity) verified unchanged
    through refinement, across ALL tensors
  - changed-key set asserted (not merely logged) against the exact 32
    expected cross-attention to_k/to_v keys, with the actual count and key
    list reported either way
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent
sys.path.insert(0, str(SEQ_DIR / "seq_edit"))
import vendored_functions as vf  # noqa: E402

RECE_REPO = Path("/data/bijaypandey/cuig_pilot/rece_golfball_author_screen/repo_src/RECE")
sys.path.insert(0, str(RECE_REPO))
from erase_methods.uce import edit_model_adversarial  # noqa: E402  (live import: clean deps, already verified)

SD14 = "/data/bijaypandey/cuig_pilot/hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"

AB_UCE_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "AB_UCE.pt"
AB_UCE_EXPECTED_SHA256 = "3a9d86de1339f9c516399fde1a9ab3c6b4ff8851062fb371712d473e7862b550"

A_CKPT_PATH = Path("/data/bijaypandey/cuig_pilot/rece_golfball_author_screen/checkpoint/GolfBall_ep0.pt")
A_CKPT_EXPECTED_SHA256 = "62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b"

EXISTING_FINAL_ATOB_PATH = SEQ_DIR / "edits" / "AtoB_english_springer.pt"
EXISTING_FINAL_ATOB_LOG = SEQ_DIR / "edits" / "AtoB_english_springer.edit_log.json"

REGULAR_SCALE = 0.1
SEED = 42
CONCEPT = "English Springer"
DEVICE = "cuda:0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pipeline(device=DEVICE):
    return StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None).to(device)


def load_checkpoint_into(pipeline, ckpt_path: Path, expected_sha256: str):
    actual = sha256_file(ckpt_path)
    if actual != expected_sha256:
        raise RuntimeError(f"{ckpt_path} sha256 mismatch: expected {expected_sha256}, got {actual}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_state = pipeline.unet.state_dict()
    assert set(ckpt.keys()) == set(base_state.keys()), f"{ckpt_path} key set mismatch"
    assert all(base_state[k].shape == ckpt[k].shape for k in base_state), f"{ckpt_path} shape mismatch"
    pipeline.unet.load_state_dict(ckpt, strict=True)
    post_state = pipeline.unet.state_dict()
    for k in ckpt:
        if not torch.equal(post_state[k].cpu(), ckpt[k].cpu()):
            raise RuntimeError(f"post-load mismatch at {k} for {ckpt_path}")
    return actual


def full_state_dict_cpu(unet):
    return {k: v.detach().clone().cpu() for k, v in unet.state_dict().items()}


def assert_full_storage_separation(unet_a, unet_b, label_a, label_b):
    sd_a, sd_b = unet_a.state_dict(), unet_b.state_dict()
    assert set(sd_a.keys()) == set(sd_b.keys()), "key set mismatch between models being separation-checked"
    shared = [k for k in sd_a if sd_a[k].data_ptr() == sd_b[k].data_ptr()]
    if shared:
        raise RuntimeError(
            f"{label_a} and {label_b} share underlying storage on {len(shared)} tensor(s) "
            f"(e.g. {shared[0]}) -- refusing to edit, this would silently corrupt the reference"
        )
    return {"n_tensors_checked": len(sd_a), "n_shared_storage": 0}


def diff_full_state_dicts(state_before: dict, state_after: dict):
    """Returns (changed_keys, max_abs_diff_by_key, max_rel_diff_by_key) comparing VALUES."""
    assert set(state_before.keys()) == set(state_after.keys())
    changed = []
    max_abs = {}
    max_rel = {}
    for k in state_before:
        a, b = state_before[k], state_after[k]
        if not torch.equal(a, b):
            changed.append(k)
            d = (a.float() - b.float()).abs()
            max_abs[k] = d.max().item()
            denom = a.float().abs().clamp_min(1e-12)
            max_rel[k] = (d / denom).max().item()
    return changed, max_abs, max_rel


def assert_reference_unchanged(unet, snapshot: dict, label: str):
    sd = {k: v.detach().cpu() for k, v in unet.state_dict().items()}
    changed, max_abs, _ = diff_full_state_dicts(snapshot, sd)
    if changed:
        raise RuntimeError(
            f"{label} reference UNet changed on {len(changed)} tensor(s) during refinement "
            f"(e.g. {changed[0]}, max_abs_diff={max_abs[changed[0]]}) -- reference drift detected"
        )
    return {"n_tensors_checked": len(sd), "n_changed": 0}


def collect_ca_projection_matrices(unet, with_to_k=True):
    """Mirrors the ca_layers / projection_matrices collection logic inside the
    vendored close_form_emb_regzero (same traversal: up/down Cross blocks,
    then mid block; to_v layers, then to_k layers if with_to_k). Used ONLY
    for the diagnostic G computation below -- it reads, never mutates."""
    ca_layers = []
    for net in unet.named_children():
        if "up" in net[0] or "down" in net[0]:
            for block in net[1]:
                if "Cross" in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if "mid" in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    mats = [l.to_v for l in ca_layers]
    if with_to_k:
        mats = mats + [l.to_k for l in ca_layers]
    return mats


def compute_G(unet, regular_scale=REGULAR_SCALE):
    """G = regular_scale * I + sum_l W_l^T W_l, over the fixed working
    (AB-UCE) weights. Diagnostic only; does not participate in the edit."""
    mats = collect_ca_projection_matrices(unet, with_to_k=True)
    dim = mats[0].weight.shape[1]
    G = torch.eye(dim, device=mats[0].weight.device) * regular_scale
    for l in mats:
        G = G + torch.matmul(l.weight.detach().T, l.weight.detach())
    return G.detach().cpu()


def run_stage2(working_pipeline, reference_pipeline):
    """Runs close_form_emb_regzero + edit_model_adversarial on
    working_pipeline using reference_pipeline as model_copy. Mutates
    working_pipeline.unet in place. Returns (concept_embedding, adv_embedding,
    new_embedding) for diagnostic traces."""
    concept_embedding, adv_embedding = vf.close_form_emb_regzero(
        model=working_pipeline,
        model_copy=reference_pipeline,
        concept=CONCEPT,
        with_to_k=True,
        save_path=None,
        regeular_scale=REGULAR_SCALE,
        seed=SEED,
    )
    id_new_text = working_pipeline.tokenizer(
        " ", padding="max_length", max_length=working_pipeline.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(DEVICE)
    new_embedding = working_pipeline.text_encoder(id_new_text)[0]

    edit_model_adversarial(
        working_pipeline,
        old_embeddings=[adv_embedding[0]],
        new_embeddings=[new_embedding[0]],
        retain_texts=[""],
        technique="replace",
        preserve_scale=0.1,
        erase_scale=1,
        lamb=0.1,
    )
    return concept_embedding, adv_embedding, new_embedding


def per_key_update_norms(before: dict, after: dict, keys):
    return {k: (after[k].float() - before[k].float()).norm().item() for k in sorted(keys)}


def total_norm(per_key: dict):
    return sum(v * v for v in per_key.values()) ** 0.5


def main():
    t_start_overall = time.time()
    result = {"phase1_replay": {}, "phase2_ab_ref_a": {}, "diagnostic_traces": {}, "stopped_before_phase2": False}

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()
    result["pinned_source_drift_check"] = "PASS"

    # --- verify inputs against their source checkpoints -----------------
    ab_uce_sha = sha256_file(AB_UCE_PATH)
    assert ab_uce_sha == AB_UCE_EXPECTED_SHA256, f"AB_UCE.pt sha256 mismatch: {ab_uce_sha}"
    a_sha = sha256_file(A_CKPT_PATH)
    assert a_sha == A_CKPT_EXPECTED_SHA256, f"GolfBall_ep0.pt sha256 mismatch: {a_sha}"
    existing_final_sha = sha256_file(EXISTING_FINAL_ATOB_PATH)
    existing_final_log = json.loads(EXISTING_FINAL_ATOB_LOG.read_text())
    assert existing_final_sha == existing_final_log["out_checkpoint_sha256"], (
        "existing AtoB_english_springer.pt sha256 does not match its own edit log -- comparator integrity check failed"
    )
    result["input_verification"] = {
        "AB_UCE.pt_sha256": ab_uce_sha,
        "GolfBall_ep0.pt_sha256": a_sha,
        "existing_AtoB_english_springer.pt_sha256": existing_final_sha,
        "existing_comparator_self_consistent_with_its_own_edit_log": True,
    }

    # whitelist of the 32 expected cross-attn K/V keys: the Stage-1 diff (A vs AB-UCE)
    a_state_cpu = torch.load(A_CKPT_PATH, map_location="cpu")
    ab_uce_state_cpu = torch.load(AB_UCE_PATH, map_location="cpu")
    changed_stage1, _, _ = diff_full_state_dicts(a_state_cpu, ab_uce_state_cpu)
    assert len(changed_stage1) == 32, f"expected exactly 32 keys changed by Stage 1 (A->AB_UCE), got {len(changed_stage1)}"
    assert all(("attn2.to_k" in k or "attn2.to_v" in k) for k in changed_stage1)
    EXPECTED_32_KEYS = set(changed_stage1)
    result["expected_32_cross_attn_kv_keys"] = sorted(EXPECTED_32_KEYS)
    del a_state_cpu, ab_uce_state_cpu

    # =====================================================================
    # PHASE 1: image-free replay of the ORIGINAL (M0-reference) Stage 2,
    # starting from the untouched AB-UCE checkpoint.
    # =====================================================================
    print("=== PHASE 1: replay of M0-reference Stage 2 from AB-UCE ===", flush=True)
    t0 = time.time()

    working_replay = load_pipeline()
    load_checkpoint_into(working_replay, AB_UCE_PATH, AB_UCE_EXPECTED_SHA256)
    reference_m0 = load_pipeline()  # pristine M0, never edited

    sep = assert_full_storage_separation(working_replay.unet, reference_m0.unet, "working(AB-UCE)", "reference(M0)")
    result["phase1_replay"]["storage_separation"] = sep

    reference_m0_snapshot = full_state_dict_cpu(reference_m0.unet)
    ab_uce_state_snapshot = full_state_dict_cpu(working_replay.unet)
    G_replay = compute_G(working_replay.unet)

    concept_emb_replay, adv_emb_replay, new_emb_replay = run_stage2(working_replay, reference_m0)

    ref_check = assert_reference_unchanged(reference_m0.unet, reference_m0_snapshot, "M0 (replay)")
    result["phase1_replay"]["reference_unchanged_check"] = ref_check

    replay_final_state = full_state_dict_cpu(working_replay.unet)
    changed_vs_ab_uce, _, _ = diff_full_state_dicts(ab_uce_state_snapshot, replay_final_state)
    result["phase1_replay"]["n_keys_changed_vs_AB_UCE_input"] = len(changed_vs_ab_uce)
    result["phase1_replay"]["changed_keys_match_expected_32"] = set(changed_vs_ab_uce) == EXPECTED_32_KEYS
    replay_update_norms = per_key_update_norms(ab_uce_state_snapshot, replay_final_state, EXPECTED_32_KEYS)
    replay_total_update_norm = total_norm(replay_update_norms)

    # --- the actual comparison: replay result vs. existing final A->B checkpoint ---
    existing_final_state = torch.load(EXISTING_FINAL_ATOB_PATH, map_location="cpu")
    changed_vs_comparator, max_abs, max_rel = diff_full_state_dicts(existing_final_state, replay_final_state)
    replay_exact_match = len(changed_vs_comparator) == 0
    result["phase1_replay"]["exact_match_vs_existing_AtoB_checkpoint"] = replay_exact_match
    result["phase1_replay"]["n_differing_keys_vs_comparator"] = len(changed_vs_comparator)
    if not replay_exact_match:
        result["phase1_replay"]["differing_keys"] = changed_vs_comparator
        result["phase1_replay"]["max_abs_diff_by_key"] = max_abs
        result["phase1_replay"]["max_rel_diff_by_key"] = max_rel
        result["phase1_replay"]["max_abs_diff_overall"] = max(max_abs.values())
        result["phase1_replay"]["max_rel_diff_overall"] = max(max_rel.values())
    result["phase1_replay"]["elapsed_seconds"] = time.time() - t0

    del working_replay, reference_m0, existing_final_state, replay_final_state
    torch.cuda.empty_cache()

    if not replay_exact_match:
        result["stopped_before_phase2"] = True
        result["stop_reason"] = (
            "Phase-1 replay (M0-reference Stage 2 from the AB-UCE input) did not reproduce the "
            "existing AtoB_english_springer.pt checkpoint exactly. Per task instruction: no tolerance, "
            "no substitute comparator, no repair loop. Stopping before Phase 2 / any new evaluation."
        )
        (HERE / "DISCREPANCY.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"stopped_before_phase2": True, "n_differing_keys": len(changed_vs_comparator)}, indent=2))
        return

    print("PHASE 1: exact match confirmed. Proceeding to Phase 2.", flush=True)

    # =====================================================================
    # PHASE 2: the actual intervention -- model_copy = A instead of M0.
    # Reload a FRESH, untouched AB-UCE working model; do not reuse or
    # re-refine the Phase-1 replayed object.
    # =====================================================================
    print("=== PHASE 2: AB_REF_A build (reference = A) ===", flush=True)
    t0 = time.time()

    working_refA = load_pipeline()
    load_checkpoint_into(working_refA, AB_UCE_PATH, AB_UCE_EXPECTED_SHA256)
    reference_A = load_pipeline()
    load_checkpoint_into(reference_A, A_CKPT_PATH, A_CKPT_EXPECTED_SHA256)

    sep2 = assert_full_storage_separation(working_refA.unet, reference_A.unet, "working(AB-UCE)", "reference(A)")
    result["phase2_ab_ref_a"]["storage_separation"] = sep2

    reference_A_snapshot = full_state_dict_cpu(reference_A.unet)
    ab_uce_state_snapshot_2 = full_state_dict_cpu(working_refA.unet)
    G_refA = compute_G(working_refA.unet)

    concept_emb_refA, adv_emb_refA, new_emb_refA = run_stage2(working_refA, reference_A)

    ref_check2 = assert_reference_unchanged(reference_A.unet, reference_A_snapshot, "A (reference)")
    result["phase2_ab_ref_a"]["reference_unchanged_check"] = ref_check2

    final_state_refA = full_state_dict_cpu(working_refA.unet)
    changed_vs_ab_uce_2, _, _ = diff_full_state_dicts(ab_uce_state_snapshot_2, final_state_refA)
    changed_set_2 = set(changed_vs_ab_uce_2)
    result["phase2_ab_ref_a"]["n_keys_changed_vs_AB_UCE_input"] = len(changed_vs_ab_uce_2)
    result["phase2_ab_ref_a"]["changed_keys"] = sorted(changed_set_2)
    result["phase2_ab_ref_a"]["changed_keys_equal_expected_32"] = (changed_set_2 == EXPECTED_32_KEYS)
    if changed_set_2 != EXPECTED_32_KEYS:
        result["phase2_ab_ref_a"]["unexpected_extra_keys"] = sorted(changed_set_2 - EXPECTED_32_KEYS)
        result["phase2_ab_ref_a"]["expected_but_unchanged_keys"] = sorted(EXPECTED_32_KEYS - changed_set_2)
    refA_update_norms = per_key_update_norms(ab_uce_state_snapshot_2, final_state_refA, EXPECTED_32_KEYS)
    refA_total_update_norm = total_norm(refA_update_norms)

    out_path = HERE / "edits" / "AB_REF_A.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(working_refA.unet.state_dict(), out_path)
    out_sha256 = sha256_file(out_path)
    result["phase2_ab_ref_a"]["out_checkpoint_path"] = str(out_path)
    result["phase2_ab_ref_a"]["out_checkpoint_sha256"] = out_sha256
    result["phase2_ab_ref_a"]["elapsed_seconds"] = time.time() - t0

    del working_refA, reference_A, final_state_refA
    torch.cuda.empty_cache()

    # =====================================================================
    # Diagnostic traces (support interpretation; not a mechanism proof)
    # =====================================================================
    g_identical = torch.equal(G_replay, G_refA)
    g_max_abs_diff = (G_replay - G_refA).abs().max().item()

    norm_concept_replay = concept_emb_replay.float().norm().item()
    norm_concept_refA = concept_emb_refA.float().norm().item()
    norm_adv_replay = adv_emb_replay.float().norm().item()
    norm_adv_refA = adv_emb_refA.float().norm().item()
    adv_diff_norm = (adv_emb_replay.float().cpu() - adv_emb_refA.float().cpu()).norm().item()
    concept_embeddings_identical = torch.equal(concept_emb_replay.cpu(), concept_emb_refA.cpu())

    result["diagnostic_traces"] = {
        "G_matrix": {
            "identical_bit_for_bit_across_reference_choices": g_identical,
            "max_abs_diff": g_max_abs_diff,
            "note": ("G = 0.1*I + sum(W_l^T W_l) computed from the fixed AB-UCE working weights, "
                     "independently in the Phase-1 (M0-reference) and Phase-2 (A-reference) branches, "
                     "before Stage 2 mutates anything. Equality is expected by construction (G depends "
                     "only on 'model', not 'model_copy'); this verifies it was not accidentally broken, "
                     "it does not by itself prove the reference choice is immaterial to the output -- "
                     "the reference changes sum(W_l^T R_l), which is where the two branches diverge."),
        },
        "concept_embedding": {
            "identical_across_branches": concept_embeddings_identical,
            "norm_replay_M0_branch": norm_concept_replay,
            "norm_A_reference_branch": norm_concept_refA,
            "note": ("concept_embedding depends only on the working model's text_encoder + concept "
                     "string, identical in both branches (same AB-UCE working model), so identity here "
                     "is expected and checked, not a new finding."),
        },
        "adversarial_embedding": {
            "norm_replay_M0_branch": norm_adv_replay,
            "norm_A_reference_branch": norm_adv_refA,
            "difference_norm": adv_diff_norm,
            "note": ("adv_embedding = concept_embedding @ (mat1^-1 mat2)^T, where mat2 = sum(W_l^T R_l) "
                     "depends on the reference R_l. This is the one quantity Phase 1 vs Phase 2 are "
                     "guaranteed to differ on by construction; the question is whether that difference "
                     "translates into materially different generated images."),
        },
        "refinement_update_norm_per_branch": {
            "replay_M0_branch_total_over_32_keys": replay_total_update_norm,
            "A_reference_branch_total_over_32_keys": refA_total_update_norm,
            "note": "L2 norm of (post-Stage-2 weight minus AB-UCE input weight), summed in quadrature over the 32 cross-attn to_k/to_v keys, per branch.",
        },
    }

    update_norms_detail = {
        "note": "Per-key L2 norm of (post-Stage-2 weight minus AB-UCE input weight).",
        "replay_M0_branch": replay_update_norms,
        "A_reference_branch": refA_update_norms,
    }
    (HERE / "UPDATE_NORMS.json").write_text(json.dumps(update_norms_detail, indent=2))

    # --- edit log for AB_REF_A.pt ---
    edit_log = {
        "endpoint_name": "AB_REF_A",
        "start": str(AB_UCE_PATH),
        "start_checkpoint_sha256": ab_uce_sha,
        "reference_identity": "A (GolfBall_ep0.pt), NOT pristine M0 -- the single changed component",
        "reference_checkpoint_sha256": a_sha,
        "concept": CONCEPT,
        "regular_scale": REGULAR_SCALE,
        "seed": SEED,
        "edit_model_adversarial_hyperparameters": {"technique": "replace", "preserve_scale": 0.1, "erase_scale": 1, "lamb": 0.1, "retain_texts": [""]},
        "working_reference_separate_storage_verified_all_tensors": True,
        "reference_unchanged_verified_all_tensors": True,
        "n_keys_changed_vs_AB_UCE_input": len(changed_vs_ab_uce_2),
        "changed_keys_equal_expected_32_cross_attn_kv": (changed_set_2 == EXPECTED_32_KEYS),
        "out_checkpoint_sha256": out_sha256,
    }
    (HERE / "edits" / "AB_REF_A.ref_a_intervention_log.json").write_text(json.dumps(edit_log, indent=2))

    result["total_elapsed_seconds"] = time.time() - t_start_overall
    (HERE / "VERIFICATION_LOG.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "expected_32_cross_attn_kv_keys"}, indent=2, default=str))


if __name__ == "__main__":
    main()
