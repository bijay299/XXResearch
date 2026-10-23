"""ISSUE 9. Emit an immutable provenance manifest, and audit RNG coverage by static inspection.

The audit's requirement: every result should be traceable to code commit, component hashes, model
and classifier revisions, dependency versions, precision, prompt/split identifiers, optimiser
settings, seeds, generation settings and metric definitions -- and the RNG streams used in
construction, Fisher estimation, latent sampling, perturbation and distillation should be audited
for coverage, with remaining nondeterminism disclosed rather than assumed absent.

This script does not re-run anything. It records what exists, hashes it, and says plainly which
checkpoints can and cannot be identified.

Run: python scripts/provenance.py [--hash-adapters]
     -> outputs/provenance_manifest.json
"""
import argparse, ast, hashlib, json, os, platform, re, subprocess, sys, time

ROOT = "/data/bijaypandey/t2i_unlearn"
OUT = f"{ROOT}/outputs"
SCRIPTS = f"{ROOT}/scripts"


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<error {e}>"


def file_sha(p, cap=None):
    """sha256 of a file. cap limits bytes read, for multi-GB checkpoints -- a capped hash is
    labelled as such, because a prefix hash is weaker evidence of identity than a full one."""
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            if cap is not None and n + len(b) > cap:
                h.update(b[:cap - n]); n = cap; break
            h.update(b); n += len(b)
    return h.hexdigest(), n


def rng_audit():
    """Static audit of RNG coverage: every stochastic call site, and whether it is seeded.

    A call with an explicit `generator=` argument is seeded deterministically; one without inherits
    the global stream, which is reproducible only if nothing else consumed it first. The second
    kind is reported as GLOBAL_STREAM, not as 'fine'.
    """
    pat_rand = re.compile(r"\b(torch\.(randn|rand|randint|randperm|normal)|\.normal_|\.uniform_|"
                          r"randn_like|rand_like|random\.|np\.random)")
    report = {}
    for fn in sorted(os.listdir(SCRIPTS)):
        if not fn.endswith(".py"):
            continue
        src = open(os.path.join(SCRIPTS, fn)).read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        sites = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if not pat_rand.search(seg.split("(")[0] + "("):
                # also catch method-form calls like x.normal_(...)
                if not pat_rand.search(seg[:60]):
                    continue
            seeded = any(k.arg == "generator" for k in node.keywords if k.arg)
            sites.append({"line": node.lineno,
                          "call": seg.split("\n")[0][:110],
                          "rng": "EXPLICIT_GENERATOR" if seeded else "GLOBAL_STREAM"})
        if sites:
            n_g = sum(1 for s in sites if s["rng"] == "EXPLICIT_GENERATOR")
            report[fn] = {"n_sites": len(sites), "n_explicit_generator": n_g,
                          "n_global_stream": len(sites) - n_g, "sites": sites}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-adapters", action="store_true",
                    help="hash the adapter checkpoints too (GBs of I/O)")
    ap.add_argument("--hash-cap-mb", type=int, default=64,
                    help="bytes per checkpoint to hash; 0 = full file")
    a = ap.parse_args()
    t0 = time.time()
    cap = None if a.hash_cap_mb == 0 else a.hash_cap_mb << 20

    man = {"study": "provenance_manifest", "issue": 9,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    # ---- code identity
    wt = "/data/bijaypandey/claude_tmp/mdlm_wt"
    man["code"] = {
        "publish_worktree": wt,
        "commit": sh("git rev-parse HEAD", cwd=wt),
        "branch": sh("git rev-parse --abbrev-ref HEAD", cwd=wt),
        "dirty": bool(sh("git status --porcelain", cwd=wt)),
        "script_sha256": {fn: file_sha(os.path.join(SCRIPTS, fn))[0][:16]
                          for fn in sorted(os.listdir(SCRIPTS)) if fn.endswith(".py")},
        "note": "the T2I working tree at /data/bijaypandey/t2i_unlearn is NOT a git repository; "
                "code identity is the publish worktree commit plus the per-file hashes above. "
                "Historical runs predating a given hash cannot be attributed to it.",
    }

    # ---- model + classifier revisions
    pins = {}
    for f, k in ((f"{ROOT}/MODEL_PIN.txt", "stable_diffusion"),
                 (f"{ROOT}/ADVUNLEARN_PIN.txt", "advunlearn")):
        if os.path.exists(f):
            pins[k] = dict(l.split("=", 1) for l in open(f).read().split("\n") if "=" in l)
    man["models"] = {
        "pins": pins,
        "base": "CompVis/stable-diffusion-v1-4 (image latent diffusion; NOT a diffusion LM)",
        "classifier": {"id": "openai/clip-vit-large-patch14", "revision": "UNPINNED",
                       "warning": "the CLIP classifier that produces EVERY forget/retain/recovery "
                                  "number in this phase was loaded without a pinned revision. It "
                                  "is cached locally, so results within this project are "
                                  "self-consistent, but the pin is not recorded and a future "
                                  "re-download could silently change every metric. This is a real "
                                  "provenance gap, disclosed not patched -- pinning it now would "
                                  "not retroactively identify the weights historical runs used."},
        "precision": {"generation_and_eval": "float16", "attack_training": "float32",
                      "tf32_matmul": "left at library default except in test_repairs.py, which "
                                     "disables it explicitly"},
    }

    # ---- dependency versions
    deps = {}
    for mod in ("torch", "diffusers", "transformers", "numpy", "accelerate", "safetensors"):
        try:
            m = __import__(mod)
            deps[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            deps[mod] = f"<not importable: {type(e).__name__}>"
    try:
        import torch
        deps["cuda"] = torch.version.cuda
        deps["cudnn"] = torch.backends.cudnn.version()
        deps["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    man["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                          "deps": deps}

    # ---- splits + metric definitions, by reference not by restatement
    man["splits"] = {"registry": "outputs/split_registry.json",
                     "defined_by": "scripts/verify_splits.py",
                     "verdict_summary": "ATK_TRAIN x EVAL_TARGET clean; DEF_TRAIN overlaps both "
                                        "evaluation sets (see FINDING38)"}
    man["protocol"] = {"frozen": "docs/PROTOCOL_FROZEN_v1.md",
                       "grid": "seeds {0,1,6}@1e-5, {0,1}@5e-5, {0}@1e-4; 300 steps; 16 images",
                       "metric_definitions": "docs/PROTOCOL_FROZEN_v1.md; H precision floor "
                                             "1/15 = 0.0667",
                       "errata": ["docs/ERRATUM_text_encoder_eval_bug.md",
                                  "docs/ERRATUM_r2_claim_corrections.md"]}

    # ---- RNG coverage audit
    man["rng_audit"] = rng_audit()
    tot = sum(v["n_sites"] for v in man["rng_audit"].values())
    glob_ = sum(v["n_global_stream"] for v in man["rng_audit"].values())
    man["rng_summary"] = {
        "n_stochastic_sites": tot, "n_explicit_generator": tot - glob_,
        "n_global_stream": glob_,
        "disclosure": "GLOBAL_STREAM sites are reproducible only given identical preceding "
                      "consumption of the global torch RNG. Attack training (latent/timestep/noise "
                      "draws per step) is on the global stream seeded once by --seed, so an attack "
                      "is repeatable end-to-end but NOT robust to inserting or removing any earlier "
                      "stochastic call. Arm CONSTRUCTION in undo_t2i.py was nondeterministic until "
                      "08742c1 and the same config+seed produced retain 0.867 and 0.783; runs "
                      "before that commit are not reproducible and any rebuild is a NEW arm.",
    }

    # ---- checkpoint identity
    ad = f"{ROOT}/adapters"
    ck = {}
    if os.path.isdir(ad):
        for d in sorted(os.listdir(ad)):
            p = os.path.join(ad, d)
            if not os.path.isdir(p):
                continue
            files = {}
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if not os.path.isfile(fp):
                    continue
                e = {"bytes": os.path.getsize(fp),
                     "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                            time.localtime(os.path.getmtime(fp)))}
                if a.hash_adapters:
                    h, n = file_sha(fp, cap)
                    e["sha256"] = h
                    e["hash_scope"] = "FULL" if cap is None or n < cap else f"FIRST_{n}_BYTES"
                files[f] = e
            ck[d] = files
    man["checkpoints"] = {
        "adapters": ck, "n_arms": len(ck), "hashed": bool(a.hash_adapters),
        "unresolved_identity": "adapters/uce_church/unet.pt was OVERWRITTEN between a retain-0.350 "
                               "and a retain-0.850 build (FINDING35 S5), so attack records naming "
                               "that path cannot be attributed to one checkpoint. Marked "
                               "UNRESOLVED; not reconstructed, because a rebuild would be a new "
                               "arm and must not be relabelled as the original.",
        "deleted": "attack_relearn.py deletes its attacked checkpoint (adapters/_atk_<label>) "
                   "after measuring it. The attacked weights behind every H_after in this phase "
                   "therefore no longer exist. The arm's INPUT checkpoint is retained, and the "
                   "attack is repeatable from it given the recorded seed and the GLOBAL_STREAM "
                   "caveat above.",
    }

    os.makedirs(OUT, exist_ok=True)
    json.dump(man, open(f"{OUT}/provenance_manifest.json", "w"), indent=1)
    print(f"[prov] commit={man['code']['commit'][:12]} dirty={man['code']['dirty']}")
    print(f"[prov] deps: " + ", ".join(f"{k}={v}" for k, v in deps.items()))
    print(f"[prov] RNG sites: {tot} total, {tot-glob_} explicit-generator, {glob_} global-stream")
    for fn, v in man["rng_audit"].items():
        print(f"         {fn:24s} {v['n_explicit_generator']:2d} explicit / "
              f"{v['n_global_stream']:2d} global")
    print(f"[prov] adapter dirs: {len(ck)} (hashed={a.hash_adapters})")
    print(f"[prov] CLIP classifier revision: UNPINNED  <- disclosed provenance gap")
    print(f"[prov] wrote {OUT}/provenance_manifest.json in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
