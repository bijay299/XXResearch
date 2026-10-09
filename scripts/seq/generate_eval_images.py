#!/usr/bin/env python3
"""Generate the frozen 280-image evaluation set for one checkpoint.

Identical generation settings at every checkpoint (manifest-enforced): same
prompt/seed pairs, scheduler, guidance, precision, 512x512, 30 steps. The
per-image latent seed is derived by a fixed documented rule from (prompt_id,
gen_seed), so images are reproducible and matched across checkpoints while
different prompts do not share identical initial noise.

Writes an image manifest with a SHA-256 per image. Missing or failed
generations are recorded explicitly and cause a non-zero exit -- they are never
silently dropped.

Reuse is bound to provenance
----------------------------
This script used to reuse any existing file at the expected PATHNAME, then write
the newly loaded checkpoint's hash and the current manifest/settings into a fresh
report. An existing evaluation whose detections had been quarantined therefore
kept its old images, and the recovery run relabelled those old bytes as evidence
from the current model. The validator could not see it: the metadata it checks
had all been rewritten truthfully about the CURRENT checkpoint while the pixels
came from an older one.

An existing image is now reused only when a prior report proves it was produced
under the SAME contract:

  * the same generating checkpoint identity (recomputed sha256 of delta.bin, or
    "no delta applied" plus the same base_model_dir for M0);
  * the same canonical manifest identity, recomputed from the manifest's own
    content, so changing a prompt text breaks reuse even when the file names,
    prompt ids and stored digest field are preserved;
  * the same generation settings and the same image-seed rule;
  * the same per-image recorded digest, re-verified against the bytes on disk,
    plus the same prompt text, prompt id and generation seed for that file.

If provenance is absent or mismatched the script FAILS BEFORE any GPU or model
setup (exit 3) and writes nothing: the old report is not overwritten and no old
image byte is reattributed to a new checkpoint. Recovering from that state is the
launcher's job -- it preserves the whole old evaluation (images and reports
together) and generates into a fresh, empty directory.

Exit codes
----------
0  complete
1  incomplete (generation failures, recorded in the report)
2  usage / environment / contract error
3  image reuse refused for want of valid provenance (nothing was written)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

# ONE canonical manifest-digest rule, imported rather than re-implemented, so
# this script and the validator can never disagree about a manifest's identity.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_stage import canonical_manifest_digest  # noqa: E402

RC_OK, RC_INCOMPLETE, RC_FATAL, RC_PROVENANCE = 0, 1, 2, 3
SEED_RULE = "(gen_seed * 1000003 + crc32(prompt_id)) % (2**31 - 1)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode()) & 0xFFFFFFFF


def image_seed_for(prompt_id: str, gen_seed: int) -> int:
    """Fixed, documented derivation. Same for every checkpoint."""
    return (gen_seed * 1_000_003 + stable_hash(prompt_id)) % (2**31 - 1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- provenance
def reuse_provenance(*, out_dir: Path, records: list[dict], prior_path: Path,
                     checkpoint_name: str, ckpt_sha: str | None,
                     base_model_dir: str, gs: dict, manifest_digest: str,
                     limited: bool) -> tuple[str, list[str], set[str]]:
    """May the files already in out_dir be reused as evidence for THIS contract?

    Returns (verdict, reasons, reusable_image_names) with verdict one of
    'nothing_to_reuse', 'authorized', 'refused'. Pure CPU: stats and hashes
    files, parses one JSON report. Never imports a model library.
    """
    wanted = {r["image_name"]: r for r in records}
    if not out_dir.is_dir():
        return "nothing_to_reuse", [], set()

    present = {p.name for p in out_dir.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    existing = sorted(present & set(wanted))
    strays = sorted(present - set(wanted))
    if not existing and not strays:
        return "nothing_to_reuse", [], set()

    reasons: list[str] = []
    # Files this manifest does not name are evidence the directory belongs to a
    # different contract. Skipped under --limit, where a short record list is
    # expected to leave the rest of a complete set behind.
    if strays and not limited:
        reasons.append(
            f"{len(strays)} image file(s) in {out_dir} are not named by this "
            f"manifest, e.g. {strays[:3]}: this directory holds a different "
            f"evaluation set")

    if not prior_path.is_file():
        reasons.append(
            f"{len(existing)} existing image file(s) but no prior image report "
            f"at {prior_path}: their generating checkpoint, manifest and "
            f"settings are unknown, so they cannot be attributed to any model")
        return "refused", reasons, set()
    try:
        prior = json.loads(prior_path.read_text())
    except Exception as e:
        reasons.append(f"prior image report {prior_path.name} does not parse "
                       f"({type(e).__name__}: {e}); provenance is unreadable")
        return "refused", reasons, set()

    # ---- contract-level identity of the prior run
    if prior.get("checkpoint") != checkpoint_name:
        reasons.append(f"prior report is for checkpoint "
                       f"{prior.get('checkpoint')!r}, this request is "
                       f"{checkpoint_name!r}")

    pcl = prior.get("checkpoint_load") or {}
    if ckpt_sha is None:
        # M0: identity IS the base model; no delta may have been applied.
        if pcl.get("applied") is not False:
            reasons.append(f"this request applies no delta, but the prior report "
                           f"has checkpoint_load.applied={pcl.get('applied')!r}")
        if pcl.get("unet_ckpt") is not None:
            reasons.append(f"this request applies no delta, but the prior report "
                           f"loaded {pcl.get('unet_ckpt')!r}")
        if pcl.get("sha256"):
            reasons.append(f"prior report records a delta hash "
                           f"({str(pcl['sha256'])[:12]}…) but this request "
                           f"applies none")
    else:
        if pcl.get("applied") is not True:
            reasons.append(f"prior report has checkpoint_load.applied="
                           f"{pcl.get('applied')!r}: those images may not have "
                           f"come from any delta at all")
        if not pcl.get("sha256"):
            reasons.append("prior report records no checkpoint sha256, so the "
                           "model that produced those images is unidentifiable")
        elif pcl["sha256"] != ckpt_sha:
            reasons.append(f"those images were generated from checkpoint "
                           f"{str(pcl['sha256'])[:12]}…, the checkpoint now on "
                           f"disk is {ckpt_sha[:12]}…: the model changed under "
                           f"the same name")

    if prior.get("base_model_dir") != base_model_dir:
        reasons.append(f"prior report used base model "
                       f"{prior.get('base_model_dir')!r}, this request uses "
                       f"{base_model_dir!r}")

    prior_man = prior.get("manifest_sha256")
    if prior_man != manifest_digest:
        reasons.append(f"those images were generated against manifest "
                       f"{str(prior_man)[:12]}…, this request's manifest content "
                       f"identity is {manifest_digest[:12]}…: the prompt set or "
                       f"the generation settings changed")

    pgs = prior.get("generation_settings") or {}
    if pgs != gs:
        differing = sorted(set(pgs) | set(gs))
        diff = {k: (pgs.get(k, "<absent>"), gs.get(k, "<absent>"))
                for k in differing if pgs.get(k, "<absent>") != gs.get(k, "<absent>")}
        reasons.append(f"generation settings differ from the prior run "
                       f"(prior, now): {diff}")

    prior_rule = prior.get("image_seed_rule")
    if prior_rule is not None and prior_rule != SEED_RULE:
        reasons.append(f"prior image-seed rule {prior_rule!r} != {SEED_RULE!r}")

    # ---- per-file identity, so a changed byte or a renamed prompt is caught
    prior_rows = {}
    for row in prior.get("images") or []:
        if isinstance(row, dict) and row.get("image_name"):
            prior_rows[row["image_name"]] = row
    unprovenanced, digest_mismatch, record_mismatch = [], [], []
    for name in existing:
        row = prior_rows.get(name)
        rec = wanted[name]
        if row is None or row.get("status") not in ("generated", "reused") \
                or not row.get("sha256"):
            unprovenanced.append(name)
            continue
        if row["sha256"] != sha256_file(out_dir / name):
            digest_mismatch.append(name)
            continue
        if (row.get("prompt_id") != rec["prompt_id"]
                or int(row.get("gen_seed", -1)) != int(rec["gen_seed"])
                or row.get("prompt") != rec["prompt"]
                or int(row.get("image_seed", -1))
                != image_seed_for(rec["prompt_id"], rec["gen_seed"])):
            record_mismatch.append(name)
    if unprovenanced:
        reasons.append(f"{len(unprovenanced)} existing image(s) carry no usable "
                       f"record in the prior report, e.g. {unprovenanced[:3]}")
    if digest_mismatch:
        reasons.append(f"{len(digest_mismatch)} existing image(s) no longer match "
                       f"the digest recorded for them, e.g. {digest_mismatch[:3]}: "
                       f"those bytes changed after they were recorded")
    if record_mismatch:
        reasons.append(f"{len(record_mismatch)} existing image(s) are recorded "
                       f"against a different prompt, generation seed or image "
                       f"seed than this manifest gives them, e.g. "
                       f"{record_mismatch[:3]}")

    if reasons:
        return "refused", reasons, set()
    return "authorized", [], set(existing)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--unet_ckpt", default=None, help="delta.bin; omit for M0")
    ap.add_argument("--checkpoint_name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--expect_manifest_sha", default=None,
                    help="frozen manifest identity, supplied from CONFIGURATION; "
                         "the manifest's content digest is recomputed and "
                         "compared against it")
    ap.add_argument("--expect_gen_settings", default=None,
                    help="path to configs/generation_settings.json. REQUIRED "
                         "when the manifest declares its settings by contract "
                         "reference (`generation_settings_contract`) rather "
                         "than inlining them, as the frozen dev and test sets "
                         "do. Never used to override settings a manifest "
                         "carries inline.")
    ap.add_argument("--prior_report", default=None,
                    help="report establishing provenance for images already in "
                         "--out_dir (default: --report)")
    ap.add_argument("--provenance_check_only", action="store_true",
                    help="decide whether existing images may be reused, print the "
                         "verdict and exit. Loads no model and writes nothing.")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap image count")
    args = ap.parse_args()

    # ---- contracts first. Everything below is CPU-only until the gate passes.
    try:
        man = json.loads(Path(args.manifest).read_text())
    except Exception as e:
        print(f"FATAL: manifest unreadable ({type(e).__name__}: {e})", file=sys.stderr)
        return RC_FATAL
    try:
        records = man["records"]
    except KeyError as e:
        print(f"FATAL: manifest lacks {e}", file=sys.stderr)
        return RC_FATAL

    # Generation settings: inlined in the manifest (the pilot set), or declared
    # by CONTRACT REFERENCE (the frozen dev/test sets, which carry
    # `generation_settings_contract` instead). The contract route exists so the
    # frozen manifests do not have to be rewritten -- and must not be, since
    # their published identities are what selection and the test freeze rest on.
    # Exactly one source is used, and the report records which.
    gs_source = "manifest.generation_settings"
    if "generation_settings" in man:
        gs = man["generation_settings"]
    elif man.get("generation_settings_contract"):
        if not args.expect_gen_settings:
            print(f"FATAL: this manifest declares its settings by contract "
                  f"({man['generation_settings_contract']!r}) and carries none "
                  f"inline, so --expect_gen_settings must supply the contract "
                  f"path. Refusing to invent generation settings.",
                  file=sys.stderr)
            return RC_FATAL
        try:
            contract = json.loads(Path(args.expect_gen_settings).read_text())
        except Exception as e:
            print(f"FATAL: generation-settings contract unreadable "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            return RC_FATAL
        gs = {k: v for k, v in contract.items() if not k.startswith("_")}
        gs_source = f"contract {args.expect_gen_settings}"
        print(f"[settings] from {gs_source} (manifest declares "
              f"{man['generation_settings_contract']!r})")
    else:
        print("FATAL: manifest carries neither 'generation_settings' nor "
              "'generation_settings_contract'", file=sys.stderr)
        return RC_FATAL
    for k in ("resolution", "num_inference_steps", "guidance_scale"):
        if k not in gs:
            print(f"FATAL: generation settings lack {k!r}", file=sys.stderr)
            return RC_FATAL

    digest = canonical_manifest_digest(man)
    stored = man.get("manifest_sha256")
    if stored != digest:
        print(f"FATAL: manifest's stored manifest_sha256 ({str(stored)[:12]}…) "
              f"does not match its recomputed content digest ({digest[:12]}…): "
              f"its contents were altered after the digest was written, or it "
              f"predates the canonical rule", file=sys.stderr)
        return RC_FATAL
    if args.expect_manifest_sha and digest != args.expect_manifest_sha:
        print(f"FATAL: manifest content identity {digest[:12]}… != frozen "
              f"expected identity {args.expect_manifest_sha[:12]}…: this is not "
              f"the frozen manifest", file=sys.stderr)
        return RC_FATAL

    limited = bool(args.limit)
    if args.limit:
        records = records[: args.limit]

    ckpt_sha = None
    if args.unet_ckpt:
        ck = Path(args.unet_ckpt)
        if not ck.is_file() or ck.stat().st_size == 0:
            print(f"FATAL: checkpoint missing or empty: {ck}", file=sys.stderr)
            return RC_FATAL
        ckpt_sha = sha256_file(ck)

    out_dir = Path(args.out_dir)
    prior_path = Path(args.prior_report) if args.prior_report else Path(args.report)

    verdict, reasons, reusable = reuse_provenance(
        out_dir=out_dir, records=records, prior_path=prior_path,
        checkpoint_name=args.checkpoint_name, ckpt_sha=ckpt_sha,
        base_model_dir=args.base_model_dir, gs=gs, manifest_digest=digest,
        limited=limited)

    print(f"[provenance] {args.checkpoint_name}: {verdict} "
          f"({len(reusable)} reusable image(s) in {out_dir})")
    if verdict == "refused":
        print(f"REFUSED: existing images in {out_dir} cannot be attributed to "
              f"this request, so reusing them would relabel older bytes as "
              f"evidence from the current model.", file=sys.stderr)
        for m in reasons:
            print(f"  - {m}", file=sys.stderr)
        print("Nothing was written: the existing report and images are "
              "untouched. Preserve that evaluation whole (images AND reports "
              "together) and generate into a fresh, empty output directory.",
              file=sys.stderr)
        return RC_PROVENANCE
    if args.provenance_check_only:
        if verdict == "authorized":
            print(f"[provenance] reuse authorized: same checkpoint "
                  f"{(ckpt_sha or 'none (M0)')[:12]}…, same manifest "
                  f"{digest[:12]}…, same settings, digests re-verified")
        return RC_OK

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in records if not (out_dir / r["image_name"]).exists()]

    # ---- a model is loaded only when there is something left to generate.
    torch = pipe = device = None
    load_info = {"unet_ckpt": args.unet_ckpt, "applied": bool(args.unet_ckpt)}
    if ckpt_sha:
        load_info["sha256"] = ckpt_sha
    if verdict == "authorized" and not todo:
        # Every expected image already exists AND passed the provenance gate, so
        # this run generates nothing. No model is loaded. checkpoint_load is
        # carried forward from the report that was just verified against the
        # checkpoint now on disk -- it is a verified fact about these image
        # bytes, not an assertion about work this process did.
        prior = json.loads(prior_path.read_text())
        load_info = dict(prior.get("checkpoint_load") or load_info)
        load_info["carried_forward_from_verified_prior_report"] = str(prior_path)
        load_info["this_run_generated"] = 0
        print(f"[reuse] all {len(records)} images already present with verified "
              f"provenance; no model loaded, nothing generated")
    else:
        import torch as _torch
        from diffusers import StableDiffusionPipeline
        torch = _torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(args.base_model_dir,
                                                       torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.safety_checker = lambda images, **kw: (images, [False] * len(images))

        load_info = {"unet_ckpt": args.unet_ckpt, "applied": False}
        if args.unet_ckpt:
            ck = Path(args.unet_ckpt)
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            sd = blob["unet"] if isinstance(blob, dict) and "unet" in blob else blob
            sd = {k: v.to(dtype) for k, v in sd.items()}
            res = pipe.unet.load_state_dict(sd, strict=False)
            if len(res.unexpected_keys) != 0:
                print(f"FATAL: unexpected keys loading {ck}: {res.unexpected_keys[:5]}",
                      file=sys.stderr)
                return RC_FATAL
            load_info.update({"applied": True, "tensors": len(sd),
                              "sha256": ckpt_sha,
                              "unexpected_keys": len(res.unexpected_keys),
                              "missing_keys": len(res.missing_keys)})
            print(f"[load] {ck.name}: {len(sd)} tensors, 0 unexpected")

    rows, failures = [], []
    n_reused = 0
    t0 = time.time()
    for i, rec in enumerate(records):
        path = out_dir / rec["image_name"]
        iseed = image_seed_for(rec["prompt_id"], rec["gen_seed"])
        if path.exists():
            # Unreachable: the gate above refuses any un-provenanced file. Kept
            # as a fail-closed assertion so a future caller that bypasses the
            # gate cannot silently fall back to pathname-only reuse.
            if rec["image_name"] not in reusable:
                print(f"FATAL: {rec['image_name']} exists but was not authorized "
                      f"for reuse; refusing to attribute it to "
                      f"{args.checkpoint_name}", file=sys.stderr)
                return RC_PROVENANCE
            rows.append({**rec, "image_path": str(path), "image_seed": iseed,
                         "sha256": sha256_file(path), "status": "reused"})
            n_reused += 1
            continue
        try:
            g = torch.Generator(device=device).manual_seed(iseed)
            img = pipe(prompt=rec["prompt"], width=gs["resolution"], height=gs["resolution"],
                       num_inference_steps=gs["num_inference_steps"],
                       guidance_scale=gs["guidance_scale"], generator=g).images[0]
            img.save(path, quality=95)
            rows.append({**rec, "image_path": str(path), "image_seed": iseed,
                         "sha256": sha256_file(path), "status": "generated"})
        except Exception as exc:  # recorded, never silently skipped
            failures.append({**rec, "error": repr(exc)})
            print(f"[FAIL] {rec['image_name']}: {exc}", file=sys.stderr)
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(records)}  {el:.0f}s  "
                  f"({el/(i+1):.2f}s/img, eta {(len(records)-i-1)*el/(i+1):.0f}s)", flush=True)

    elapsed = time.time() - t0
    report = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "checkpoint": args.checkpoint_name,
        "base_model_dir": args.base_model_dir,
        "checkpoint_load": load_info,
        # The RECOMPUTED canonical content identity (equal to the stored field,
        # which was checked against it above before anything else happened).
        "manifest_sha256": digest,
        "manifest_stored_sha256": stored,
        "generation_settings": gs,
        "generation_settings_source": gs_source,
        "image_seed_rule": SEED_RULE,
        "expected_images": len(records),
        "generated_or_reused": len(rows),
        "n_generated": len(rows) - n_reused,
        "n_reused": n_reused,
        "failures": failures,
        "complete": len(rows) == len(records) and not failures,
        "out_dir": str(out_dir),
        "provenance": {
            "reuse_verdict": verdict,
            "reuse_rule": (
                "An existing image file is reused ONLY when a prior report in "
                "this directory proves it was produced under the same contract: "
                "same checkpoint identity (recomputed delta.bin sha256, or no "
                "delta plus the same base_model_dir for M0), same canonical "
                "manifest content identity, same generation settings, same "
                "image-seed rule, and the same recorded per-image digest "
                "re-verified against the bytes on disk. Otherwise this script "
                "exits 3 before any model is loaded and writes nothing."),
            "checkpoint_sha256": ckpt_sha,
            "manifest_content_sha256": digest,
            "prior_report_consulted": str(prior_path) if n_reused else None,
        },
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(1, len(records)), 3),
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20)
                            if (torch is not None and device.type == "cuda") else None,
        "gpu": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "name": torch.cuda.get_device_name(0)
                        if (torch is not None and device.type == "cuda") else None,
                "model_loaded_this_run": pipe is not None},
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "images": rows,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n[{args.checkpoint_name}] {len(rows)}/{len(records)} images "
          f"({n_reused} reused with verified provenance) in "
          f"{elapsed:.0f}s ({elapsed/max(1,len(records)):.2f}s/img) -> {args.report}")
    if not report["complete"]:
        print(f"INCOMPLETE: {len(failures)} failure(s)", file=sys.stderr)
        return RC_INCOMPLETE
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
