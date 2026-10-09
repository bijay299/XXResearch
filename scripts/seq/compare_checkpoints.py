#!/usr/bin/env python3
"""Compare two checkpoints tensor by tensor, on CPU.

Prepared for the reproducibility check in the matched-effectiveness protocol:
the re-run unregularised trajectory's final dump against the existing MAB.

Three claims are kept strictly separate, because conflating them is how a
serialisation detail becomes a false claim about training:

  1. FILE HASHES EQUAL      byte-identical serialisation. A mismatch can come
                            from serialisation order, metadata or compression
                            alone and is NOT evidence of a training difference.
  2. TENSORS NUMERICALLY     the weights agree to a stated tolerance. This is
     EQUAL                   what a reproducibility question is actually about.
  3. TRAINING EQUIVALENT     a stronger claim that neither of the above
                            establishes, and that would need behavioural
                            comparison (same prompts, same seeds, same outputs).

This tool answers 1 and 2 and refuses to answer 3.

CPU only; never imports CUDA.

    python scripts/seq/compare_checkpoints.py A/delta.bin B/delta.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--contract", default="configs/checkpoint_contract.json")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs difference still called numerically equal "
                         "(default 0.0 = bit-for-bit on the tensor values)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=5,
                    help="how many most-divergent tensors to list")
    args = ap.parse_args()

    pa, pb = Path(args.a), Path(args.b)
    for p in (pa, pb):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    rep: dict = {
        "a": str(pa), "b": str(pb), "tolerance": args.tol,
        "claim_separation": {
            "file_hashes_equal": None,
            "tensors_numerically_equal": None,
            "training_equivalent": "NOT ASSESSED — this tool cannot establish it; "
                                   "behavioural comparison would be required",
        },
    }

    # ---- claim 1: file hashes ---------------------------------------------
    ha, hb = sha256_file(pa), sha256_file(pb)
    rep["file_sha256"] = {"a": ha, "b": hb}
    rep["file_bytes"] = {"a": pa.stat().st_size, "b": pb.stat().st_size}
    rep["claim_separation"]["file_hashes_equal"] = (ha == hb)

    # ---- claim 2: tensor values -------------------------------------------
    try:
        import torch
    except Exception as e:
        rep["claim_separation"]["tensors_numerically_equal"] = \
            f"UNCHECKED — torch unavailable ({e})"
        print(json.dumps(rep, indent=2))
        return 2

    top = "unet"
    if Path(args.contract).is_file():
        top = json.loads(Path(args.contract).read_text()).get("top_level_key", "unet")

    def load(p: Path):
        o = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(o, dict) and top in o and isinstance(o[top], dict):
            return o[top]
        if isinstance(o, dict):
            return o
        raise TypeError(f"{p} is {type(o).__name__}, not a state dict")

    try:
        sa, sb = load(pa), load(pb)
    except Exception as e:
        rep["claim_separation"]["tensors_numerically_equal"] = \
            f"UNCHECKED — could not load as state dicts ({type(e).__name__}: {e})"
        print(json.dumps(rep, indent=2))
        return 2

    only_a, only_b = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    shared = sorted(set(sa) & set(sb))
    rep["tensor_sets"] = {
        "n_a": len(sa), "n_b": len(sb), "n_shared": len(shared),
        "only_in_a": only_a, "only_in_b": only_b,
    }

    per: list[dict] = []
    shape_mismatch, nonfinite = [], []
    for k in shared:
        ta, tb = sa[k], sb[k]
        if tuple(ta.shape) != tuple(tb.shape):
            shape_mismatch.append(
                {"tensor": k, "a": list(ta.shape), "b": list(tb.shape)})
            continue
        fa, fb = ta.detach().float(), tb.detach().float()
        if not (bool(torch.isfinite(fa).all()) and bool(torch.isfinite(fb).all())):
            nonfinite.append(k)
        d = (fa - fb).abs()
        denom = fa.norm().item()
        per.append({
            "tensor": k,
            "max_abs_diff": d.max().item(),
            "mean_abs_diff": d.mean().item(),
            "relative_frobenius_diff": ((fa - fb).norm().item() / denom)
                                       if denom > 0 else None,
            "dtype_a": str(ta.dtype), "dtype_b": str(tb.dtype),
            "n_elements": int(fa.numel()),
        })

    rep["shape_mismatches"] = shape_mismatch
    rep["tensors_with_nonfinite_values"] = nonfinite
    if per:
        rep["summary"] = {
            "n_tensors_compared": len(per),
            "max_abs_diff_over_all_tensors": max(x["max_abs_diff"] for x in per),
            "mean_of_mean_abs_diff": sum(x["mean_abs_diff"] for x in per) / len(per),
            "max_relative_frobenius_diff": max(
                (x["relative_frobenius_diff"] or 0.0) for x in per),
            "n_tensors_bitwise_identical": sum(
                1 for x in per if x["max_abs_diff"] == 0.0),
        }
        rep["most_divergent_tensors"] = sorted(
            per, key=lambda x: -x["max_abs_diff"])[: args.top]
    rep["per_tensor"] = per

    equal = (not only_a and not only_b and not shape_mismatch and per
             and rep["summary"]["max_abs_diff_over_all_tensors"] <= args.tol)
    rep["claim_separation"]["tensors_numerically_equal"] = bool(equal)

    # ---- readable verdict -------------------------------------------------
    print(f"file sha256 A : {ha[:16]}…")
    print(f"file sha256 B : {hb[:16]}…")
    print(f"hashes equal  : {rep['claim_separation']['file_hashes_equal']}")
    if per:
        s = rep["summary"]
        print(f"\ntensors compared            : {s['n_tensors_compared']}")
        print(f"bitwise-identical tensors   : {s['n_tensors_bitwise_identical']}")
        print(f"max abs diff (all tensors)  : {s['max_abs_diff_over_all_tensors']:.3e}")
        print(f"max relative Frobenius diff : {s['max_relative_frobenius_diff']:.3e}")
        print(f"numerically equal at tol={args.tol:g}: "
              f"{rep['claim_separation']['tensors_numerically_equal']}")
        if rep.get("most_divergent_tensors"):
            print("\nmost divergent tensors:")
            for x in rep["most_divergent_tensors"]:
                print(f"  {x['max_abs_diff']:.3e}  {x['tensor']}")
    if only_a or only_b or shape_mismatch:
        print(f"\nSTRUCTURE DIFFERS: only_in_a={len(only_a)} "
              f"only_in_b={len(only_b)} shape_mismatches={len(shape_mismatch)}")

    print("\nInterpretation guard:")
    print("  A hash mismatch alone is NOT a training difference — serialisation")
    print("  order or metadata can differ while the tensors are identical.")
    print("  Tensor equality is in turn NOT training equivalence: that is a")
    print("  stronger claim this tool does not assess.")

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
