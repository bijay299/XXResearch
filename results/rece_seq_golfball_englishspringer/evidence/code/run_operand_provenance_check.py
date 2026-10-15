#!/usr/bin/env python3
"""Narrow operand-provenance check for solver_only_intervention.

Verifies, on the SAME device/backend/pipeline configuration as the
original native edits (cuda:0, fp32, SD1.4 pipeline), that:
  (a) the native singleton guide embedding (text " ") matches the saved
      embedding_cross_swap/edits/guide_embedding.pt exactly,
  (b) the two rows of edit_model_adversarial's own native retain batch
      (tokenizer(["",""]) -> text_encoder, a batch of 2 since
      retain_texts=[""] zipped with itself) agree with each other, and
  (c) whether either row matches the saved singleton
      embedding_cross_swap/edits/retain_embedding.pt (a batch-of-1
      computation) exactly.

This does NOT recover any adversarial embedding and does NOT touch the
UNet weights. Text-encoder forward passes only.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

SD14 = "/data/bijaypandey/cuig_pilot/hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
OUT_DIR = ROOT / "solver_only_intervention"
DEVICE = "cuda:0"

SAVED_GUIDE = ROOT / "embedding_cross_swap" / "edits" / "guide_embedding.pt"
SAVED_RETAIN = ROOT / "embedding_cross_swap" / "edits" / "retain_embedding.pt"


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    t0 = time.time()
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info(0)
    gpu_idle_check = {"free_bytes": free, "total_bytes": total, "free_frac": free / total}

    pipe = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(DEVICE)
    text_encoder.eval()

    # --- (a) native singleton guide embedding, exactly as run_edit.py computes it ---
    with torch.no_grad():
        id_new_text = tokenizer(" ", padding="max_length", max_length=tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        new_embedding = text_encoder(id_new_text)[0]  # [1,77,768]
        native_guide = new_embedding[0].detach().cpu()  # [77,768], matches old_embeddings[0]-style indexing used downstream

    saved_guide_full = torch.load(SAVED_GUIDE, map_location="cpu")  # [1,77,768]
    saved_guide = saved_guide_full[0]

    guide_equal = torch.equal(native_guide, saved_guide)
    guide_max_abs_diff = (native_guide.float() - saved_guide.float()).abs().max().item()

    # --- (b)/(c) native retain batch, exactly as edit_model_adversarial computes it ---
    retain_texts = [""]
    with torch.no_grad():
        for old_text, new_text in zip(retain_texts, retain_texts):
            text_input = tokenizer([old_text, new_text], padding="max_length",
                                    max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]  # [2,77,768]
            native_retain_row0, native_retain_row1 = text_embeddings  # old_emb, new_emb
    native_retain_row0 = native_retain_row0.detach().cpu()
    native_retain_row1 = native_retain_row1.detach().cpu()

    saved_retain_full = torch.load(SAVED_RETAIN, map_location="cpu")  # [1,77,768]
    saved_retain_singleton = saved_retain_full[0]

    rows_equal = torch.equal(native_retain_row0, native_retain_row1)
    rows_max_abs_diff = (native_retain_row0.float() - native_retain_row1.float()).abs().max().item()

    row0_vs_singleton_equal = torch.equal(native_retain_row0, saved_retain_singleton)
    row0_vs_singleton_max_abs_diff = (native_retain_row0.float() - saved_retain_singleton.float()).abs().max().item()
    row1_vs_singleton_equal = torch.equal(native_retain_row1, saved_retain_singleton)
    row1_vs_singleton_max_abs_diff = (native_retain_row1.float() - saved_retain_singleton.float()).abs().max().item()

    result = {
        "device": DEVICE,
        "gpu_idle_check_before_allocation": gpu_idle_check,
        "guide": {
            "native_shape": list(native_guide.shape), "native_dtype": str(native_guide.dtype),
            "saved_shape": list(saved_guide.shape), "saved_dtype": str(saved_guide.dtype),
            "native_sha256": sha256_tensor(native_guide), "saved_sha256": sha256_tensor(saved_guide),
            "equal": guide_equal, "max_abs_diff": guide_max_abs_diff,
        },
        "retain": {
            "native_row0_shape": list(native_retain_row0.shape), "native_row0_dtype": str(native_retain_row0.dtype),
            "native_row1_shape": list(native_retain_row1.shape), "native_row1_dtype": str(native_retain_row1.dtype),
            "saved_singleton_shape": list(saved_retain_singleton.shape), "saved_singleton_dtype": str(saved_retain_singleton.dtype),
            "native_row0_sha256": sha256_tensor(native_retain_row0),
            "native_row1_sha256": sha256_tensor(native_retain_row1),
            "saved_singleton_sha256": sha256_tensor(saved_retain_singleton),
            "rows_equal_to_each_other": rows_equal, "rows_max_abs_diff": rows_max_abs_diff,
            "row0_equal_to_saved_singleton": row0_vs_singleton_equal, "row0_max_abs_diff_vs_saved_singleton": row0_vs_singleton_max_abs_diff,
            "row1_equal_to_saved_singleton": row1_vs_singleton_equal, "row1_max_abs_diff_vs_saved_singleton": row1_vs_singleton_max_abs_diff,
        },
        "elapsed_seconds": time.time() - t0,
    }

    # --- decision per task instructions ---
    if not guide_equal:
        result["stop_reason"] = "Native singleton guide embedding does not match saved guide_embedding.pt exactly. Stopping before any solve or image generation, per task instruction (no tolerance, no repair loop)."
    elif not rows_equal:
        result["stop_reason"] = "The two native retain batch rows unexpectedly disagree with each other. Stopping and reporting, per task instruction."
    else:
        result["resolution"] = (
            "Guide matches exactly. Native retain batch rows agree with each other "
            f"(max_abs_diff={rows_max_abs_diff:.3e}). "
            + ("Row also matches the saved singleton exactly -- using the saved singleton retain_embedding.pt as R is validated as bit-identical to the native operand."
               if row0_vs_singleton_equal else
               "Row does NOT match the saved singleton retain_embedding.pt exactly -- per task instruction, the solver will use the NATIVE two-row operands (row0 as context/R_ctx, row1 as value-side/R_val), preserving edit_model_adversarial's own original indexing and multiplicity, not the saved singleton, and will disclose the audit's prior singleton approximation as superseded for this build.")
        )
    result["gate_pass"] = guide_equal and rows_equal

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "OPERAND_PROVENANCE_CHECK.json").write_text(json.dumps(result, indent=2))

    # also save the native tensors as evidence (small, float32, 77x768 each)
    torch.save(native_guide, OUT_DIR / "native_guide_embedding.pt")
    torch.save(native_retain_row0, OUT_DIR / "native_retain_row0.pt")
    torch.save(native_retain_row1, OUT_DIR / "native_retain_row1.pt")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
