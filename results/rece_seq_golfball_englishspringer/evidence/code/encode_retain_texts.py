#!/usr/bin/env python3
"""Encode the two new retain texts ('golf ball', 'French horn') using the
native fp32 tokenizer/text_encoder/backend, exactly as edit_model_
adversarial's own retain_texts=[t] loop does it
(zip(retain_texts, retain_texts) -> tokenizer([t,t]) -> text_encoder).
Also re-verifies guide[0]'s raw tensor-byte hash as a provenance
cross-check. Text-encoder forward passes only -- no UNet weights touched,
no checkpoint loaded/edited, no image generated.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

SD14 = "/data/bijaypandey/cuig_pilot/hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
OUT_DIR = ROOT / "preservation_content_control"
DEVICE = "cuda:0"

SAVED_GUIDE = ROOT / "embedding_cross_swap" / "edits" / "guide_embedding.pt"
EXPECTED_GUIDE0_RAW_SHA256 = "24a6d0eea897a96a53bfe5c349bad87dbb81ad0b424fe02ff9413895fbdbb93f"

RETAIN_TEXTS = {"GOLF": "golf ball", "HORN": "French horn"}


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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

    result = {"device": DEVICE, "gpu_idle_check_before_allocation": gpu_idle_check}

    # --- provenance cross-check: guide[0] raw tensor-byte hash ---
    saved_guide_full = torch.load(SAVED_GUIDE, map_location="cpu")
    saved_guide0 = saved_guide_full[0] if saved_guide_full.dim() == 3 else saved_guide_full
    guide0_raw_sha = sha256_tensor(saved_guide0)
    guide0_match = guide0_raw_sha == EXPECTED_GUIDE0_RAW_SHA256
    result["guide0_raw_tensor_byte_check"] = {
        "shape": list(saved_guide0.shape), "actual_sha256": guide0_raw_sha,
        "expected_sha256": EXPECTED_GUIDE0_RAW_SHA256, "match": guide0_match,
    }
    print(f"guide0 raw tensor-byte check: {'MATCH' if guide0_match else 'MISMATCH'}", flush=True)
    if not guide0_match:
        result["stop_reason"] = "SCOPED STOP: guide[0] raw tensor-byte hash does not match the required value. No substitution."
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "RETAIN_ENCODING_PROVENANCE.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # --- encode each new retain text, native batch [t,t] ---
    arms = {}
    all_ok = True
    for arm, text in RETAIN_TEXTS.items():
        with torch.no_grad():
            text_input = tokenizer([text, text], padding="max_length",
                                    max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            ids = text_input.input_ids
            text_embeddings = text_encoder(ids.to(DEVICE))[0]  # [2,77,768]
            row0, row1 = text_embeddings
        row0 = row0.detach().cpu()
        row1 = row1.detach().cpu()
        ids_row0, ids_row1 = ids[0].cpu(), ids[1].cpu()

        rows_equal = torch.equal(row0, row1)
        rows_max_abs_diff = (row0.float() - row1.float()).abs().max().item()
        ids_equal = torch.equal(ids_row0, ids_row1)
        n_real_tokens = int((ids_row0 != tokenizer.pad_token_id).sum().item())

        arms[arm] = {
            "text": text,
            "token_ids_row0": ids_row0.tolist(), "token_ids_row1": ids_row1.tolist(),
            "token_ids_equal": ids_equal,
            "n_nonpad_tokens": n_real_tokens,
            "row0_shape": list(row0.shape), "row0_dtype": str(row0.dtype),
            "row1_shape": list(row1.shape), "row1_dtype": str(row1.dtype),
            "row0_sha256": sha256_tensor(row0), "row1_sha256": sha256_tensor(row1),
            "rows_equal_bit_identical": rows_equal, "rows_max_abs_diff": rows_max_abs_diff,
            "finite_row0": bool(torch.isfinite(row0).all()), "finite_row1": bool(torch.isfinite(row1).all()),
        }
        print(f"  {arm} ('{text}'): ids_equal={ids_equal} n_nonpad_tokens={n_real_tokens} "
              f"rows_equal={rows_equal} max_abs_diff={rows_max_abs_diff:.3e}", flush=True)
        if not (rows_equal and ids_equal and arms[arm]["finite_row0"] and arms[arm]["finite_row1"]):
            all_ok = False

        torch.save(row0, OUT_DIR / f"native_retain_{arm}_row0.pt")
        torch.save(row1, OUT_DIR / f"native_retain_{arm}_row1.pt")

    result["arms"] = arms
    result["all_rows_bit_identical_and_finite"] = all_ok
    result["gate_pass"] = guide0_match and all_ok

    if not all_ok:
        result["stop_reason"] = "SCOPED STOP: at least one new retain text's batch rows are not bit-identical (or non-finite). No averaging, no repair."

    result["elapsed_seconds"] = time.time() - t0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "RETAIN_ENCODING_PROVENANCE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
