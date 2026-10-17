"""Minimal parent-loading adapter for the chain saw -> parachute sequential diagnostic.

Protocol: results/modern_flux_dp_uce/sequential_diagnostic_protocol/PROTOCOL.md
at commit c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80.

WHAT THIS DOES (and deliberately nothing else):

1. Verifies the A-parent checkpoint file's sha256 against the protocol-pinned value
   BEFORE loading anything (file identity, not just key names).
2. Wraps `diffusers.DiffusionPipeline.from_pretrained` so that the FIRST pipeline it
   returns that actually carries a transformer gets the parent's two edited weight
   tensors merged into `pipe.transformer` immediately on construction -- i.e. strictly
   BEFORE the pinned edit script scans `pipe.transformer.named_modules()` and
   deepcopies `original_modules`. Every reference / guide / preservation response the
   native code then computes is therefore derived from the edited parent, not pristine M0.
3. Optionally captures the final fp32 weight matrices immediately before the native
   bf16 cast (DP only), by a shape+dtype-filtered observation wrapper on Tensor.to that
   clones and delegates. It changes no value and no update order.
4. Calls the PINNED edit function unchanged, with the frozen CLI arguments.

The pinned source files are NOT modified. Native arithmetic, token handling, update
order, precision and guide semantics are untouched. There is no tuning knob here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import contextlib
from datetime import datetime, timezone

import torch
import diffusers
from safetensors.torch import load_file

SOURCE_FLUX_DIR = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo/Flux"
sys.path.insert(0, SOURCE_FLUX_DIR)

EXPECTED_KEYS = {
    "context_embedder.weight",
    "time_text_embed.text_embedder.linear_1.weight",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ParentLoader:
    """Merges the verified parent's two tensors into the first transformer-carrying pipe."""

    def __init__(self, parent_path: str, expected_sha256: str):
        self.parent_path = parent_path
        self.expected_sha256 = expected_sha256
        self.record: dict = {}
        self.applied = False

        actual = sha256_file(parent_path)
        self.record["parent_path"] = parent_path
        self.record["parent_sha256_expected"] = expected_sha256
        self.record["parent_sha256_actual"] = actual
        self.record["parent_sha256_match"] = (actual == expected_sha256)
        if actual != expected_sha256:
            raise SystemExit(
                f"PARENT IDENTITY FAILURE: {parent_path}\n"
                f"  expected {expected_sha256}\n  actual   {actual}"
            )

        self.parent_tensors = load_file(parent_path)
        keys = set(self.parent_tensors.keys())
        self.record["parent_keys"] = sorted(keys)
        if keys != EXPECTED_KEYS:
            raise SystemExit(f"PARENT KEY FAILURE: got {sorted(keys)}")
        self.record["parent_tensor_meta"] = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "finite": bool(torch.isfinite(v).all().item()),
                "fro_norm_fp32": float(v.float().norm().item()),
            }
            for k, v in self.parent_tensors.items()
        }
        if not all(m["finite"] for m in self.record["parent_tensor_meta"].values()):
            raise SystemExit("PARENT FINITENESS FAILURE: non-finite values in parent")

    def apply(self, pipe):
        """Merge parent tensors into pipe.transformer; verify byte-exact placement."""
        tr = pipe.transformer
        sd = tr.state_dict()
        pre = {}
        post = {}
        for k, v in self.parent_tensors.items():
            if k not in sd:
                raise SystemExit(f"PARENT MERGE FAILURE: key {k} absent from transformer")
            cur = sd[k]
            if tuple(cur.shape) != tuple(v.shape):
                raise SystemExit(f"PARENT MERGE FAILURE: shape mismatch on {k}")
            if cur.dtype != v.dtype:
                raise SystemExit(
                    f"PARENT DTYPE MISMATCH on {k}: pipeline {cur.dtype} vs parent {v.dtype} "
                    "(no silent cast is performed)"
                )
            pre[k] = float(cur.float().norm().item())
            sd[k] = v.to(cur.device)
        tr.load_state_dict(sd, strict=True)

        # verify the ACTUAL loaded tensors now equal the parent bit-for-bit
        live = tr.state_dict()
        exact = {}
        for k, v in self.parent_tensors.items():
            lv = live[k].detach().cpu()
            exact[k] = bool(torch.equal(lv, v.cpu()))
            post[k] = float(lv.float().norm().item())
        self.record["pristine_fro_norm_before_merge"] = pre
        self.record["loaded_fro_norm_after_merge"] = post
        self.record["loaded_equals_parent_bitwise"] = exact
        if not all(exact.values()):
            raise SystemExit("PARENT MERGE FAILURE: loaded tensors differ from parent")
        self.applied = True
        print(f"[adapter] parent merged into transformer BEFORE module copy: {self.parent_path}")
        print(f"[adapter] bitwise-equal to parent: {exact}")


@contextlib.contextmanager
def patched_from_pretrained(loader: ParentLoader | None):
    """Patch from_pretrained so the transformer-carrying pipe is parent-loaded on construction."""
    if loader is None:
        yield
        return
    orig = diffusers.DiffusionPipeline.from_pretrained

    def wrapper(*args, **kwargs):
        pipe = orig(*args, **kwargs)
        if not loader.applied and getattr(pipe, "transformer", None) is not None:
            loader.apply(pipe)
        return pipe

    diffusers.DiffusionPipeline.from_pretrained = wrapper
    try:
        yield
    finally:
        diffusers.DiffusionPipeline.from_pretrained = orig


@contextlib.contextmanager
def capture_pre_cast_fp32(enabled: bool, store: dict):
    """Observe (clone, never modify) fp32 [3072,*] matrices at the moment of the bf16 cast.

    DP's native code ends each module's edit with `W_0_f32.to(torch_dtype)`. This wrapper
    clones any fp32 2-D tensor of an edited-module shape being cast to bfloat16, then
    delegates to the real .to(). Values and update order are unchanged; this is purely
    observational instrumentation for the protocol's pre-cast residual log.
    """
    if not enabled:
        yield
        return
    orig_to = torch.Tensor.to
    shapes = {(3072, 4096), (3072, 768)}
    store["captures"] = []

    def wrapper(self, *args, **kwargs):
        try:
            if (self.dtype == torch.float32 and self.dim() == 2
                    and tuple(self.shape) in shapes):
                tgt = None
                for a in args:
                    if isinstance(a, torch.dtype):
                        tgt = a
                if isinstance(kwargs.get("dtype"), torch.dtype):
                    tgt = kwargs["dtype"]
                if tgt == torch.bfloat16:
                    store["captures"].append(self.detach().clone().cpu())
        except Exception:
            pass
        return orig_to(self, *args, **kwargs)

    torch.Tensor.to = wrapper
    try:
        yield
    finally:
        torch.Tensor.to = orig_to


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["DP", "UCE"])
    p.add_argument("--parent_checkpoint", default=None,
                   help="A-parent .safetensors; omit for a no-parent (native) run")
    p.add_argument("--parent_sha256", default=None)
    p.add_argument("--edit_concepts", required=True)
    p.add_argument("--guide_concepts", required=True)
    p.add_argument("--preserve_concepts", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--verification_log", required=True)
    p.add_argument("--precast_dump", default=None,
                   help="where to write captured pre-cast fp32 matrices (DP only)")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    rec: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "exp_name": args.exp_name,
        "argv": sys.argv,
        "native_source_dir": SOURCE_FLUX_DIR,
    }

    loader = None
    if args.parent_checkpoint:
        if not args.parent_sha256:
            raise SystemExit("--parent_sha256 is required with --parent_checkpoint")
        loader = ParentLoader(args.parent_checkpoint, args.parent_sha256)
        rec["parent_verification"] = loader.record
    else:
        rec["parent_verification"] = {"parent_path": None,
                                      "note": "no-parent path: native reproduction, adapter inert"}

    edit_concepts = [c.strip() for c in args.edit_concepts.split(";")]
    guide_concepts = [c.strip() for c in args.guide_concepts.split(";") if c.strip()]
    preserve_concepts = [c.strip() for c in args.preserve_concepts.split(";") if c.strip()]
    rec["edit_concepts"] = edit_concepts
    rec["guide_concepts"] = guide_concepts
    rec["preserve_concepts_in_order"] = preserve_concepts

    os.makedirs(args.save_dir, exist_ok=True)
    store: dict = {}

    if args.method == "DP":
        import Double_proxy  # pinned, unmodified
        torch_dtype = torch.bfloat16
        max_seq = 256 if "schnell" in args.model_id else 512
        rec["edit_time_max_sequence_length"] = max_seq
        rec["edit_torch_dtype"] = str(torch_dtype)
        rec["native_entrypoint"] = "Double_proxy.UCE_double_proxy"
        rec["replace_indices"] = "all (native default: None -> every token)"
        with patched_from_pretrained(loader), capture_pre_cast_fp32(bool(args.precast_dump), store):
            Double_proxy.UCE_double_proxy(
                model_id=args.model_id,
                edit_concepts=edit_concepts,
                guide_concepts=guide_concepts,
                preserve_concepts=preserve_concepts,
                save_dir=args.save_dir,
                exp_name=args.exp_name,
                torch_dtype=torch_dtype,
                device=args.device,
                max_sequence_length=max_seq,
                replace_indices=None,
            )
        if args.precast_dump:
            caps = store.get("captures", [])
            rec["precast_capture_count"] = len(caps)
            if caps:
                # keep the last capture per shape = the value actually cast and saved
                last = {}
                for t in caps:
                    last[str(tuple(t.shape))] = t
                torch.save(last, args.precast_dump)
                rec["precast_dump"] = args.precast_dump
                rec["precast_shapes"] = sorted(last.keys())
    else:
        import UCE_original  # pinned, unmodified
        torch_dtype = torch.float32
        max_seq = 256 if "schnell" in args.model_id else 512
        rec["edit_time_max_sequence_length"] = max_seq
        rec["edit_torch_dtype"] = str(torch_dtype)
        rec["native_entrypoint"] = "UCE_original.UCE"
        rec["uce_hyperparams"] = {"erase_scale": 10, "preserve_scale": 1, "lamb": 0.1}
        if len(guide_concepts) == 1 and len(edit_concepts) > 1:
            guide_concepts = guide_concepts * len(edit_concepts)
        with patched_from_pretrained(loader):
            UCE_original.UCE(
                model_id=args.model_id,
                edit_concepts=edit_concepts,
                guide_concepts=guide_concepts,
                preserve_concepts=preserve_concepts,
                erase_scale=10,
                preserve_scale=1,
                lamb=0.1,
                save_dir=args.save_dir,
                exp_name=args.exp_name,
                torch_dtype=torch_dtype,
                device=args.device,
                max_sequence_length=max_seq,
            )

    # ---- post-edit integrity of the produced child checkpoint ----
    child_path = os.path.join(args.save_dir, args.exp_name + ".safetensors")
    child = load_file(child_path)
    ck = set(child.keys())
    child_rec = {
        "path": child_path,
        "sha256": sha256_file(child_path),
        "keys": sorted(ck),
        "exactly_two_expected_keys": (ck == EXPECTED_KEYS),
        "tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype),
                "finite": bool(torch.isfinite(v).all().item()),
                "fro_norm_fp32": float(v.float().norm().item())}
            for k, v in child.items()
        },
    }
    # save/reload integrity: reload a second time and compare bitwise
    child2 = load_file(child_path)
    child_rec["save_reload_bitwise_identical"] = all(
        bool(torch.equal(child[k], child2[k])) for k in child
    )
    child_rec["all_finite"] = all(t["finite"] for t in child_rec["tensors"].values())
    rec["child_checkpoint"] = child_rec

    if loader is not None:
        # child must differ from parent (an edit happened) on at least one key
        diffs = {}
        for k in EXPECTED_KEYS:
            d = (child[k].float() - loader.parent_tensors[k].float())
            diffs[k] = {"delta_fro_norm": float(d.norm().item()),
                        "changed": bool(d.abs().max().item() > 0)}
        rec["child_minus_parent"] = diffs

    ok = (child_rec["exactly_two_expected_keys"] and child_rec["all_finite"]
          and child_rec["save_reload_bitwise_identical"]
          and (loader is None or loader.applied))
    rec["technical_integrity_pass"] = bool(ok)

    with open(args.verification_log, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[adapter] verification log -> {args.verification_log}")
    print(f"[adapter] technical_integrity_pass = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
