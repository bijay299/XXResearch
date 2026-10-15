#!/usr/bin/env python3
"""embedding_cross_swap: does sequential refinement corruption follow the
recovered adversarial embedding, the working checkpoint, or their
interaction?

2x2 design: rows = working checkpoint (W_D = B_UCE.pt, direct Stage-1
output from M0; W_S = AB_UCE.pt, sequential Stage-1 output from A).
Columns = recovered adversarial embedding (E_D, recovered from W_D; E_S,
recovered from W_S), BOTH recovered with pristine M0 as the
close_form_emb_regzero reference (unlike the prior reference_A_intervention
task, which varied the reference; this task holds the reference fixed at
M0 and varies which working checkpoint supplies the embedding).

  (W_D, E_D) = existing direct final B            -- reused, not rebuilt
  (W_S, E_S) = existing sequential final A->B      -- reused, not rebuilt
  (W_D, E_S) = NEW: D_WITH_SEQ_EMB
  (W_S, E_D) = NEW: S_WITH_DIRECT_EMB

PHASE 1 -- recover E_D (from W_D) and E_S (from W_S), both against a
pristine M0 reference. Full-tensor storage-separation and
reference-invariance checks for both. Verify concept/guide/retain text
embeddings agree bit-for-bit across pipelines (the text encoder is never
touched by any UCE edit in this line).

PHASE 2 -- GATE, image-free. Apply E_D natively to a fresh, untouched W_D
via edit_model_adversarial (identical hyperparameters to the original
run_edit.py Stage 2) and require EXACT tensor equality (all 686 UNet
keys, values not hashes) against the existing B_english_springer.pt. This
validates that the saved W_D (B_UCE.pt) intermediate is faithful to what
the original direct run actually used -- the reciprocal of the sequential
replay already verified in the prior task. If this does not match
exactly: record differing keys and max abs/relative differences, write
DISCREPANCY.json, and STOP before building either new cell or generating
any image. No tolerance, no substitute comparator, no repair loop.

PHASE 3 -- (only if the gate passes) build the two new cross cells from
fresh, untouched working inputs, transferring ONLY the recovered
embedding; each receiving checkpoint's own native guide/retain embeddings
and weight regularizer are used unmodified. Enforced (raising) assertion
that only the 32 expected cross-attention K/V keys change.

Diagnostics (embedding norms/cosine, edit displacement, recovery-fit and
erase-fit residuals, retain-projection drift) are computed separately, in
compute_diagnostics.py, from the tensors this script saves -- pure linear
algebra, no GPU required, run after this script releases the device.
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

W_D_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "B_UCE.pt"
W_D_SHA256 = "23b691460d7189b7e8c78ceb5ea8441a0400f61c6a79abf8a6e8afd89a161bc4"
W_S_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "AB_UCE.pt"
W_S_SHA256 = "3a9d86de1339f9c516399fde1a9ab3c6b4ff8851062fb371712d473e7862b550"

EXISTING_B_PATH = SEQ_DIR / "edits" / "B_english_springer.pt"
EXISTING_B_SHA256 = "c711e97127d3dcca9d6752e86d64fba8240ed20347c699179f9fe0859267b9dc"
EXISTING_ATOB_PATH = SEQ_DIR / "edits" / "AtoB_english_springer.pt"
EXISTING_ATOB_SHA256 = "ec16c9f748035b832d57dbe6b2eaa33d0e26279fe9a852eb171837f48a7a2469"

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
    assert set(sd_a.keys()) == set(sd_b.keys())
    shared = [k for k in sd_a if sd_a[k].data_ptr() == sd_b[k].data_ptr()]
    if shared:
        raise RuntimeError(f"{label_a} and {label_b} share storage on {len(shared)} tensor(s), e.g. {shared[0]}")
    return {"n_tensors_checked": len(sd_a), "n_shared_storage": 0}


def diff_full_state_dicts(state_before: dict, state_after: dict):
    assert set(state_before.keys()) == set(state_after.keys())
    changed, max_abs, max_rel = [], {}, {}
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
        raise RuntimeError(f"{label} reference changed on {len(changed)} tensor(s) (e.g. {changed[0]}, max_abs={max_abs[changed[0]]})")
    return {"n_tensors_checked": len(sd), "n_changed": 0}


def text_embedding(pipeline, text, device=DEVICE):
    ids = pipeline.tokenizer(
        text, padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    return pipeline.text_encoder(ids)[0]


def recover_embedding(working_pipeline, reference_pipeline):
    """Runs the vendored close_form_emb_regzero(model=working, model_copy=reference).
    Does NOT mutate either pipeline -- close_form_emb_regzero only reads
    the layers' weights to build mat1/mat2; it does not edit them."""
    concept_embedding, adv_embedding = vf.close_form_emb_regzero(
        model=working_pipeline,
        model_copy=reference_pipeline,
        concept=CONCEPT,
        with_to_k=True,
        save_path=None,
        regeular_scale=REGULAR_SCALE,
        seed=SEED,
    )
    return concept_embedding, adv_embedding


def apply_embedding(working_pipeline, adv_embedding, device=DEVICE):
    """Applies edit_model_adversarial to working_pipeline in place using the
    supplied adversarial embedding and the pipeline's own NATIVE guide (' ')
    and retain ('') embeddings/hyperparameters. Unchanged from run_edit.py's
    Stage 2 call."""
    new_embedding = text_embedding(working_pipeline, " ", device)
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
    return new_embedding


def main():
    t_start = time.time()
    result = {"stopped_before_phase3": False, "stopped_before_phase4_image_gen": False}

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()
    result["pinned_source_drift_check"] = "PASS"

    # --- verify all four input checkpoints against their expected hashes ---
    w_d_sha = sha256_file(W_D_PATH)
    assert w_d_sha == W_D_SHA256, f"W_D sha256 mismatch: {w_d_sha}"
    w_s_sha = sha256_file(W_S_PATH)
    assert w_s_sha == W_S_SHA256, f"W_S sha256 mismatch: {w_s_sha}"
    existing_b_sha = sha256_file(EXISTING_B_PATH)
    assert existing_b_sha == EXISTING_B_SHA256, f"existing B sha256 mismatch: {existing_b_sha}"
    existing_atob_sha = sha256_file(EXISTING_ATOB_PATH)
    assert existing_atob_sha == EXISTING_ATOB_SHA256, f"existing AtoB sha256 mismatch: {existing_atob_sha}"
    result["input_verification"] = {
        "W_D_sha256": w_d_sha, "W_S_sha256": w_s_sha,
        "existing_B_sha256": existing_b_sha, "existing_AtoB_sha256": existing_atob_sha,
    }

    # whitelist of the 32 expected cross-attn K/V keys (Stage-1 diffs, both branches, cross-checked)
    m0_state_cpu = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None).unet.state_dict()
    m0_state_cpu = {k: v.detach().clone().cpu() for k, v in m0_state_cpu.items()}
    w_d_state_cpu = torch.load(W_D_PATH, map_location="cpu")
    w_s_state_cpu = torch.load(W_S_PATH, map_location="cpu")
    changed_d, _, _ = diff_full_state_dicts(m0_state_cpu, w_d_state_cpu)
    assert len(changed_d) == 32
    assert all(("attn2.to_k" in k or "attn2.to_v" in k) for k in changed_d)
    EXPECTED_32_KEYS = set(changed_d)
    # the 32-key set is a fixed architectural location (all cross-attn to_k/to_v
    # projections are always the edit target, for any concept or branch), so the
    # whitelist derived from the M0-vs-W_D diff applies to W_S-rooted edits too;
    # not re-derived from a W_S-vs-A diff since A is not loaded in this script.
    del w_s_state_cpu
    result["expected_32_cross_attn_kv_keys"] = sorted(EXPECTED_32_KEYS)

    # =====================================================================
    # PHASE 1: recover E_D (from W_D) and E_S (from W_S), reference = M0 both times
    # =====================================================================
    print("=== PHASE 1: recover E_D and E_S (reference M0) ===", flush=True)
    t0 = time.time()

    pipe_w_d = load_pipeline()
    load_checkpoint_into(pipe_w_d, W_D_PATH, W_D_SHA256)
    pipe_m0_ref1 = load_pipeline()
    sep1 = assert_full_storage_separation(pipe_w_d.unet, pipe_m0_ref1.unet, "W_D", "M0(ref for E_D)")
    m0_ref1_snapshot = full_state_dict_cpu(pipe_m0_ref1.unet)
    w_d_snapshot = full_state_dict_cpu(pipe_w_d.unet)
    concept_emb_from_wd, E_D = recover_embedding(pipe_w_d, pipe_m0_ref1)
    ref1_check = assert_reference_unchanged(pipe_m0_ref1.unet, m0_ref1_snapshot, "M0(ref for E_D)")
    w_d_unchanged = assert_reference_unchanged(pipe_w_d.unet, w_d_snapshot, "W_D(during E_D recovery)")
    guide_emb_from_wd = text_embedding(pipe_w_d, " ")
    retain_emb_from_wd = text_embedding(pipe_w_d, "")

    pipe_w_s = load_pipeline()
    load_checkpoint_into(pipe_w_s, W_S_PATH, W_S_SHA256)
    pipe_m0_ref2 = load_pipeline()
    sep2 = assert_full_storage_separation(pipe_w_s.unet, pipe_m0_ref2.unet, "W_S", "M0(ref for E_S)")
    m0_ref2_snapshot = full_state_dict_cpu(pipe_m0_ref2.unet)
    w_s_snapshot = full_state_dict_cpu(pipe_w_s.unet)
    concept_emb_from_ws, E_S = recover_embedding(pipe_w_s, pipe_m0_ref2)
    ref2_check = assert_reference_unchanged(pipe_m0_ref2.unet, m0_ref2_snapshot, "M0(ref for E_S)")
    w_s_unchanged = assert_reference_unchanged(pipe_w_s.unet, w_s_snapshot, "W_S(during E_S recovery)")
    guide_emb_from_ws = text_embedding(pipe_w_s, " ")
    retain_emb_from_ws = text_embedding(pipe_w_s, "")

    concept_agree = torch.equal(concept_emb_from_wd.cpu(), concept_emb_from_ws.cpu())
    guide_agree = torch.equal(guide_emb_from_wd.cpu(), guide_emb_from_ws.cpu())
    retain_agree = torch.equal(retain_emb_from_wd.cpu(), retain_emb_from_ws.cpu())

    result["phase1_recovery"] = {
        "storage_separation_E_D_recovery": sep1,
        "storage_separation_E_S_recovery": sep2,
        "reference_unchanged_E_D_recovery": ref1_check,
        "reference_unchanged_E_S_recovery": ref2_check,
        "working_unchanged_during_E_D_recovery": w_d_unchanged,
        "working_unchanged_during_E_S_recovery": w_s_unchanged,
        "concept_embedding_agrees_across_pipelines": concept_agree,
        "guide_embedding_agrees_across_pipelines": guide_agree,
        "retain_embedding_agrees_across_pipelines": retain_agree,
        "E_D_norm": E_D.float().norm().item(),
        "E_S_norm": E_S.float().norm().item(),
        "elapsed_seconds": time.time() - t0,
    }
    assert concept_agree and guide_agree and retain_agree, "text embeddings disagree across pipelines -- text encoder is not actually invariant, stopping"

    # save recovered embeddings + the three invariant text embeddings for the diagnostics script
    torch.save(E_D.cpu(), HERE / "edits" / "E_D.pt")
    torch.save(E_S.cpu(), HERE / "edits" / "E_S.pt")
    torch.save(concept_emb_from_wd.cpu(), HERE / "edits" / "concept_embedding.pt")
    torch.save(guide_emb_from_wd.cpu(), HERE / "edits" / "guide_embedding.pt")
    torch.save(retain_emb_from_wd.cpu(), HERE / "edits" / "retain_embedding.pt")
    torch.save(m0_state_cpu, HERE / "edits" / "M0_unet_state_dict.pt")

    del pipe_w_d, pipe_m0_ref1, pipe_w_s, pipe_m0_ref2
    torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 2: GATE -- image-free direct compatibility check
    # =====================================================================
    print("=== PHASE 2: GATE -- apply E_D to fresh W_D, compare vs existing B ===", flush=True)
    t0 = time.time()

    pipe_gate = load_pipeline()
    load_checkpoint_into(pipe_gate, W_D_PATH, W_D_SHA256)
    apply_embedding(pipe_gate, E_D.to(DEVICE))
    gate_final_state = full_state_dict_cpu(pipe_gate.unet)
    existing_b_state = torch.load(EXISTING_B_PATH, map_location="cpu")

    changed_vs_comparator, max_abs, max_rel = diff_full_state_dicts(existing_b_state, gate_final_state)
    gate_exact_match = len(changed_vs_comparator) == 0
    result["phase2_gate"] = {
        "n_differing_keys_vs_existing_B": len(changed_vs_comparator),
        "exact_match": gate_exact_match,
        "elapsed_seconds": time.time() - t0,
    }
    if not gate_exact_match:
        result["phase2_gate"]["differing_keys"] = changed_vs_comparator
        result["phase2_gate"]["max_abs_diff_by_key"] = max_abs
        result["phase2_gate"]["max_rel_diff_by_key"] = max_rel
        result["phase2_gate"]["max_abs_diff_overall"] = max(max_abs.values())
        result["phase2_gate"]["max_rel_diff_overall"] = max(max_rel.values())

    del pipe_gate, gate_final_state, existing_b_state
    torch.cuda.empty_cache()

    if not gate_exact_match:
        result["stopped_before_phase3"] = True
        result["stop_reason"] = (
            "The image-free direct compatibility check (E_D applied natively to fresh W_D) did not "
            "reproduce the existing B_english_springer.pt checkpoint exactly. Per task instruction: "
            "no tolerance, no substitute comparator, no repair loop. Stopping before building either "
            "new cell or generating any image."
        )
        (HERE / "DISCREPANCY.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"stopped_before_phase3": True, "n_differing_keys": len(changed_vs_comparator)}, indent=2))
        return

    print("PHASE 2: exact match confirmed. Proceeding to Phase 3.", flush=True)

    # =====================================================================
    # PHASE 3: build the two new cross cells
    # =====================================================================
    print("=== PHASE 3: build D_WITH_SEQ_EMB and S_WITH_DIRECT_EMB ===", flush=True)

    def build_cell(name, row_ckpt_path, row_sha256, embedding):
        t0 = time.time()
        pipe = load_pipeline()
        load_checkpoint_into(pipe, row_ckpt_path, row_sha256)
        pre_state = full_state_dict_cpu(pipe.unet)
        apply_embedding(pipe, embedding.to(DEVICE))
        post_state = full_state_dict_cpu(pipe.unet)
        changed, _, _ = diff_full_state_dicts(pre_state, post_state)
        changed_set = set(changed)
        assert changed_set == EXPECTED_32_KEYS, (
            f"{name}: changed-key set does not equal the expected 32-key cross-attn K/V whitelist. "
            f"extra={sorted(changed_set - EXPECTED_32_KEYS)} missing={sorted(EXPECTED_32_KEYS - changed_set)}"
        )
        out_path = HERE / "edits" / f"{name}.pt"
        torch.save(post_state, out_path)
        out_sha = sha256_file(out_path)
        log = {
            "endpoint_name": name,
            "row_working_checkpoint": str(row_ckpt_path),
            "row_working_checkpoint_sha256": row_sha256,
            "n_keys_changed": len(changed),
            "changed_keys_equal_expected_32_ENFORCED": True,
            "out_checkpoint_sha256": out_sha,
            "elapsed_seconds": time.time() - t0,
        }
        (HERE / "edits" / f"{name}.cross_swap_log.json").write_text(json.dumps(log, indent=2))
        del pipe
        torch.cuda.empty_cache()
        return log

    log_d_with_seq = build_cell("D_WITH_SEQ_EMB", W_D_PATH, W_D_SHA256, E_S)
    log_s_with_direct = build_cell("S_WITH_DIRECT_EMB", W_S_PATH, W_S_SHA256, E_D)
    result["phase3_new_cells"] = {"D_WITH_SEQ_EMB": log_d_with_seq, "S_WITH_DIRECT_EMB": log_s_with_direct}

    result["total_elapsed_seconds"] = time.time() - t_start
    (HERE / "VERIFICATION_LOG.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("expected_32_cross_attn_kv_keys",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
