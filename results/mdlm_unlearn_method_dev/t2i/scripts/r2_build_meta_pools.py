"""Build Meta-Unlearning's three data pools for upstream src/esd_meta.py.

Pool meanings, verbatim from upstream/Meta-Unlearning/README.md L18:
  hrm = the UNLEARNED concept
  irt = retained concepts UNRELATED to it
  tgt = retained concepts RELATED to it
meta.py / esd_meta.py load them with `load_dataset("imagefolder", data_dir="./dataset/<pool>")`
(L671-673), so each pool needs train/ and test/ subdirectories each carrying a metadata.jsonl with
{"file_name", "text"}.

AUDIT REPAIRS 2026-10-26 (docs/AGENT2_BASELINES_FINDINGS.md B-6, B-7):

1. SIZING. upstream src/gen_images.py L90-92 generates `num_images=30` train AND 30 test PER
   PROMPT, so a 3-prompt pool is 90 images per split. The previous version of this file allocated
   30 per POOL *split across* prompts (10/prompt for tgt, 5/prompt for irt) while its docstring
   claimed to follow gen_images.py. Both sizings are now selectable and the choice is RECORDED:
     --per-prompt N     upstream semantics (DEFAULT, N=30)
     --total-per-pool N the historical compute adaptation (N images per pool per split)
   `--total-per-pool` is a legitimate adaptation; it is simply not "the author protocol" and
   pools.json now says which was used.

2. SEMANTICS. The legacy scheme erases "church" while RETAINING "cathedral/chapel/basilica" as the
   RELATED-RETAIN pool. Those are hyponyms of church, not neighbours of it -- a cathedral IS a
   church -- so that objective can only be satisfied by suppressing the TOKEN "church", not the
   CONCEPT. Upstream's own related-retain is a superordinate/sibling category (gen_images.py
   L62-66: hrm="Thomas Kinkade" with paint="a painting"), never a hyponym. Two schemes:
     --pool-scheme legacy    the existing pools, reproduced bit-for-bit (same prompts, same seeds)
     --pool-scheme coherent  hrm=church paraphrases; tgt=co-hyponyms+superordinate (mosque,
                             synagogue, temple, castle, town hall, stone building); and a FOURTH
                             pool `fgen` holding cathedral/chapel/basilica, to be scored as FORGET
                             (forget-generalisation), never as retain.
   Whichever is used, report the legacy scheme as PROMPT-SPECIFIC TOKEN SUPPRESSION and the
   coherent scheme as CONCEPT ERASURE. They are different claims.

3. FAIL-FAST. Blank (NSFW-filtered) frames are rejected and regenerated; counts are validated after
   writing; a pre-existing directory is no longer silently skipped (--resume is now explicit); any
   shortfall exits non-zero.

REPAIR v2 2026-10-26 (docs/REPAIR_BASELINES_v2.md A-3, A-2):

4. CONTENT validation, not just existence. Every split is now checked for
     * count on disk == count in metadata.jsonl == the planned count;
     * SPLIT ROLE: every caption must be one of THIS pool's declared prompts, and each prompt must
       carry exactly `per_prompt` images. Previously a pool could hold the right NUMBER of images
       of the WRONG CONCEPT and validate. That is the failure that matters here, because the whole
       forget/retain claim rests on which prompt produced which pool.
     * a stable CONTENT FINGERPRINT: sha256 over sorted (file_name, sha256(bytes), caption). It is
       recorded per split in pools.json, so a later run can prove it trained on the same bytes.

5. RESUME COMPATIBILITY. `--resume` used to validate only against the NEW plan's counts, so
   resuming the historical pools (scheme=legacy, sizing=total_per_pool) under the new default
   (sizing=per_prompt) died with a bare count error that read like corruption. The existing
   pools.json is now compared field by field (scheme, sizing, per-pool prompts, per_prompt) and a
   mismatch is reported as RESUME INCOMPATIBLE naming the exact field, with the fix.

6. NO SILENT DEFAULT TO THE LEGACY POOLS. `--pool-scheme` is REQUIRED (it had default="legacy"),
   and `legacy` additionally demands `--historical-reconstruction`, because that scheme cannot
   support a concept-erasure claim (see SCHEMES["legacy"]["claim"]).

7. `--verify-only` validates pools already on disk and exits. It imports no torch and touches no
   GPU, so the complete pool contract is CPU-checkable before any GPU time is requested.
"""
import argparse, csv, hashlib, json, os, sys

NEG = ("bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,"
       "low quality,lowres,error,blurry,text,cropped,username")   # gen_images.py L46, verbatim
P = "/data/bijaypandey/t2i_unlearn"
CFG = 7.5                                                          # gen_images.py L18
UPSTREAM_PER_PROMPT = 30                                           # gen_images.py L91-92

SCHEMES = {
    # name -> {pool: [prompts]}, plus "forget_side" listing pools scored as FORGET
    "legacy": {
        "pools": {
            "hrm": ["a photo of a church"],
            "tgt": ["a photo of a cathedral", "a photo of a chapel", "a photo of a basilica"],
            "irt": None,                       # sampled from coco_object_retain.csv
        },
        "forget_side": ["hrm"],
        "claim": ("PROMPT-SPECIFIC TOKEN SUPPRESSION: cathedral/chapel/basilica are hyponyms of "
                  "church and are held in the RETAIN pool, so a model that merely stops responding "
                  "to the string 'church' satisfies this objective. Do NOT report results from "
                  "this scheme as 'church concept erasure'."),
    },
    "coherent": {
        "pools": {
            # LEAK FIXED 2026-10-26 (AGENT B, REPAIR v3). This pool read
            #   ["a photo of a church", "a photo of a church building",
            #    "a photograph of a church exterior"]
            # and TWO of those three are EXACTLY evaluation target prompts:
            #   gen_eval.CONCEPTS["church"]        == ["a photo of a church", "a church building",
            #                                         "a photograph of a church exterior"]
            #   attack_relearn.SPLIT["church"]["H"] == the same three
            # So the scheme recommended for "any run intended to support a concept-erasure claim"
            # trained its FORGET pool on the prompts the claim is scored on. Found by
            # scripts/split_registry.py v3 once r2_build_meta_pools was registered and its pools
            # inspected (it was unregistered in v2, which is why this went unseen). The
            # replacements below are church paraphrases verified DISJOINT from gen_eval.CONCEPTS,
            # attack_relearn.SPLIT[*]["T"]/["H"] and split_registry.FINAL_PROMPTS by
            # `check_split_disjointness()`, which now runs on every build and verify.
            # NO POOL ON DISK IS RELABELLED: an existing coherent pool built to the old definition
            # fails check_resume_compat with the exact field, as a definition mismatch should.
            "hrm": ["a parish church on a quiet lane",
                    "a church with a slate roof and tall windows",
                    "a church courtyard under open sky"],
            "fgen": ["a photo of a cathedral", "a photo of a chapel", "a photo of a basilica"],
            "tgt": ["a photo of a mosque", "a photo of a synagogue", "a photo of a temple",
                    "a photo of a castle", "a photo of a town hall", "a photo of a stone building"],
            "irt": None,
        },
        "forget_side": ["hrm", "fgen"],
        "claim": ("CONCEPT ERASURE: hyponyms (cathedral/chapel/basilica) sit in `fgen` and are "
                  "scored as FORGET; the related-retain pool holds co-hyponyms and the "
                  "superordinate, which share structure with the target without being instances "
                  "of it. NOTE: upstream esd_meta.py reads only ./dataset/{hrm,irt,tgt}; `fgen` is "
                  "an EVALUATION pool and is not consumed by the trainer."),
    },
}
# train/test seed bases, per pool. Disjoint by construction so the splits never share a sample.
SEED_BASE = {"train": 1000, "test": 5000}


def die(msg):
    print(f"[r2pools] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def gen_one(pipe, text, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt=text, negative_prompt=NEG, guidance_scale=CFG, generator=g).images[0]


def is_blank(im):
    import numpy as np
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    return bool(a.max() == 0 or a.std() < 1.0)


def gen(pipe, text, n, seed0, max_retries=20):
    """n usable images for `text`. Blank (safety-checker-blanked) frames are regenerated on a
    fresh seed rather than silently written, which is the defect found in the IMMA data path."""
    out, seed, rejects = [], seed0, 0
    while len(out) < n:
        if seed - seed0 >= n + max_retries:
            die(f"could not get {n} usable images for {text!r}: {rejects} blank frames rejected")
        im = gen_one(pipe, text, seed)
        seed += 1
        if is_blank(im):
            rejects += 1
            continue
        out.append((im, text))
    if rejects:
        print(f"    rejected {rejects} blank frame(s) for {text!r}", flush=True)
    return out, rejects


def save(data, d):
    os.makedirs(d, exist_ok=True)
    meta = []
    for i, (im, txt) in enumerate(data):
        fn = f"{i}.png"
        im.save(os.path.join(d, fn))
        meta.append({"file_name": fn, "text": txt})
    with open(os.path.join(d, "metadata.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    return meta


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(d, rows):
    """Stable content digest over (file_name, sha256(bytes), caption), filename-sorted.

    This is what lets a later run PROVE it trained on the same bytes. Counts and directory
    existence cannot: a split can hold the right NUMBER of images of the WRONG CONCEPT.
    """
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["file_name"]):
        h.update(r["file_name"].encode())
        h.update(sha256_file(os.path.join(d, r["file_name"])).encode())
        h.update(r["text"].encode())
    return h.hexdigest()


def validate(d, expect, prompts=None, per_prompt=None):
    """Hard validation: counts, metadata agreement, no blanks, SPLIT ROLE, content fingerprint.

    Returns (count, fingerprint). `prompts`/`per_prompt` enable the split-role check, which is the
    one that actually protects the forget/retain claim: it asserts that this split's captions are
    exactly this pool's declared prompts, in the declared proportion.
    """
    import glob
    from PIL import Image
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    mp = os.path.join(d, "metadata.jsonl")
    if not os.path.isfile(mp):
        die(f"{d}: no metadata.jsonl")
    rows = [json.loads(l) for l in open(mp) if l.strip()]
    missing_keys = [i for i, r in enumerate(rows) if "file_name" not in r or "text" not in r]
    if missing_keys:
        die(f"{d}: {len(missing_keys)} metadata rows lack file_name/text (rows {missing_keys[:5]})")
    names = set(r["file_name"] for r in rows)
    disk = set(os.path.basename(f) for f in files)
    if len(files) != expect:
        die(f"{d}: {len(files)} png on disk, expected {expect}")
    if len(rows) != expect:
        die(f"{d}: {len(rows)} metadata rows, expected {expect}")
    if names != disk:
        die(f"{d}: metadata/disk mismatch: {sorted(names ^ disk)[:5]}")
    blank = [os.path.basename(f) for f in files if is_blank(Image.open(f))]
    if blank:
        die(f"{d}: {len(blank)} blank image(s) survived: {blank[:5]}")

    # ---- SPLIT ROLE: the captions must be this pool's prompts, in the declared proportion.
    if prompts is not None:
        texts = [r["text"] for r in rows]
        unknown = sorted(set(texts) - set(prompts))
        if unknown:
            die(f"{d}: {len(unknown)} caption(s) do not belong to this pool's declared prompts. "
                f"This split is holding the WRONG CONCEPT, which invalidates the forget/retain "
                f"assignment. unexpected={unknown[:3]} declared={list(prompts)[:3]}")
        absent = [t for t in prompts if t not in set(texts)]
        if absent:
            die(f"{d}: declared prompt(s) contribute no images: {absent[:3]}")
        if per_prompt is not None:
            skew = {t: texts.count(t) for t in prompts if texts.count(t) != per_prompt}
            if skew:
                die(f"{d}: per-prompt counts != {per_prompt}: {dict(list(skew.items())[:5])}")
    return len(files), fingerprint(d, rows)


def check_split_disjointness(scheme_name, plan, historical):
    """ENFORCE the split registry's verdict on THIS scheme's resolved pools (REPAIR v3, AGENT B).

    Why this exists. r2_build_meta_pools was NOT in split_registry v2's roster, so its pools were
    never inspected, and the `coherent` scheme's FORGET pool silently contained two of the three
    evaluation target prompts (see SCHEMES["coherent"]). Registering the script is not enough on
    its own: the registry reads source, while THIS function checks the pools the run has actually
    RESOLVED -- including `irt`, which is sampled from coco_object_retain.csv at run time and
    therefore appears in no source listing at all.

    Returns a record. For a non-historical scheme an overlap with the evaluation target pool is a
    HARD FAILURE; `legacy` is exempt because it is an acknowledged historical reconstruction whose
    hrm pool IS an evaluation prompt by construction -- that is the defect it reproduces, and it is
    reported, never silently passed.
    """
    import importlib.util as iu
    def _m(n):
        s = iu.spec_from_file_location(f"_r2p_{n}", f"{P}/scripts/{n}.py")
        m = iu.module_from_spec(s); s.loader.exec_module(m); return m
    try:
        sr, ge, ar = _m("split_registry"), _m("gen_eval"), _m("attack_relearn")
    except BaseException as e:
        die(f"cannot load the split registry to check pool disjointness: {type(e).__name__}: {e}. "
            f"A pool build that cannot be checked against the registry must NOT proceed.")
    nz = sr.norm
    eval_target, atk_train, final = {}, {}, {}
    for c in sorted(ar.SPLIT):
        for p in ar.SPLIT[c]["H"]:
            eval_target[nz(p)] = f"attack_relearn.SPLIT[{c}]['H']"
        for p in ar.SPLIT[c]["T"]:
            atk_train[nz(p)] = f"attack_relearn.SPLIT[{c}]['T']"
    for c, v in ge.CONCEPTS.items():
        for p in v:
            eval_target.setdefault(nz(p), f"gen_eval.CONCEPTS[{c}]")
    for c, v in sr.FINAL_PROMPTS.items():
        for p in v:
            final[nz(p)] = f"split_registry.FINAL_PROMPTS[{c}]"

    hits = {"vs_EVAL_TARGET": [], "vs_ATK_TRAIN": [], "vs_FINAL": []}
    for pool, p in plan.items():
        for s in p["prompts"]:
            n = nz(s)
            for key, ref in (("vs_EVAL_TARGET", eval_target), ("vs_ATK_TRAIN", atk_train),
                             ("vs_FINAL", final)):
                if n in ref:
                    hits[key].append({"pool": pool, "prompt": s, "collides_with": ref[n]})
    rec = {"split_version": sr.SPLIT_VERSION, "scheme": scheme_name, "overlaps": hits,
           "n_prompts_checked": sum(len(p["prompts"]) for p in plan.values()),
           "pools_checked": sorted(plan),
           "note": "checked on the RESOLVED pools of this run, irt included (irt is sampled from "
                   "coco_object_retain.csv at run time and appears in no source listing)."}
    for key, v in hits.items():
        if v:
            print(f"[r2pools] SPLIT OVERLAP {key}: {len(v)} prompt(s)", flush=True)
            for h in v[:6]:
                print(f"    {h['pool']}: {h['prompt']!r} <- {h['collides_with']}", flush=True)
    if hits["vs_EVAL_TARGET"] and not historical:
        die("SPLIT LEAK: %d pool prompt(s) in scheme %r are ALSO evaluation target prompts, so a "
            "model trained on these pools would be scored on prompts it was trained on. This is "
            "an EXPLICIT FAILURE, not a note: a run built on these pools cannot support a clean "
            "evaluation claim. Offenders: %s. Either change the pool definition or declare the "
            "scheme a historical reconstruction (--historical-reconstruction), which records the "
            "leak instead of removing it."
            % (len(hits["vs_EVAL_TARGET"]), scheme_name,
               [(h["pool"], h["prompt"]) for h in hits["vs_EVAL_TARGET"]]))
    if hits["vs_FINAL"]:
        die("SPLIT LEAK: %d pool prompt(s) collide with split_registry.FINAL_PROMPTS: %s. FINAL "
            "must be reissued under a new version rather than quietly amended."
            % (len(hits["vs_FINAL"]), [h["prompt"] for h in hits["vs_FINAL"]]))
    if hits["vs_EVAL_TARGET"]:
        print("[r2pools] the %d EVAL_TARGET overlap(s) above are PERMITTED ONLY because this is a "
              "declared HISTORICAL RECONSTRUCTION. They are recorded in pools.json and any result "
              "from these pools is a development measurement." % len(hits["vs_EVAL_TARGET"]),
              flush=True)
    return rec


def check_resume_compat(root, scheme_name, sizing, plan):
    """Compare the on-disk manifest against the requested plan, field by field.

    `--resume` used to validate only the new plan's COUNTS, so resuming the historical pools
    (legacy / total_per_pool) under a different sizing died with a bare count error that read like
    data corruption. Name the actual mismatch instead, and say how to proceed.
    """
    mp = os.path.join(root, "pools.json")
    if not os.path.isfile(mp):
        return None
    try:
        old = json.load(open(mp))
    except Exception as e:
        die(f"{mp} exists but is not readable JSON ({e}); move it aside to rebuild")
    probs = []
    if old.get("pool_scheme") != scheme_name:
        probs.append(f"pool_scheme on disk={old.get('pool_scheme')!r} requested={scheme_name!r}")
    if old.get("sizing") != sizing:
        probs.append(f"sizing on disk={old.get('sizing')!r} requested={sizing!r}")
    for name, p in plan.items():
        o = (old.get("pools") or {}).get(name)
        if o is None:
            probs.append(f"pool {name!r} is absent from the existing manifest")
            continue
        if list(o.get("prompts") or []) != list(p["prompts"]):
            probs.append(f"pool {name!r} prompts differ: on disk={list(o.get('prompts') or [])} "
                         f"requested={p['prompts']}")
        if o.get("per_prompt") != p["per_prompt"]:
            probs.append(f"pool {name!r} per_prompt on disk={o.get('per_prompt')} "
                         f"requested={p['per_prompt']}")
    if probs:
        die("RESUME INCOMPATIBLE -- the pools on disk were built to a DIFFERENT definition:\n    "
            + "\n    ".join(probs)
            + f"\n  The existing pools are NOT deleted. Either match the on-disk definition "
              f"(--pool-scheme {old.get('pool_scheme')} "
              f"--{'per-prompt' if old.get('sizing') == 'per_prompt' else 'total-per-pool'} ...), "
              f"or build the new definition under a DIFFERENT --root so both survive.")
    return old


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pool root; esd_meta.py expects it as ./dataset")
    ap.add_argument("--pool-scheme", choices=sorted(SCHEMES), required=True,
                    help="REQUIRED, no default. legacy = the historical church/cathedral pools "
                         "(token suppression; additionally requires --historical-reconstruction); "
                         "coherent = the recommended forget/forget-generalisation/related split")
    ap.add_argument("--historical-reconstruction", action="store_true",
                    help="acknowledge that --pool-scheme legacy reproduces a HISTORICAL setup "
                         "whose forget/retain split cannot support a concept-erasure claim")
    ap.add_argument("--verify-only", action="store_true",
                    help="validate the pools already on disk against this plan and exit. Imports "
                         "no torch and touches no GPU; use it to check the pool contract first.")
    ap.add_argument("--backfill-fingerprints", action="store_true",
                    help="validate the pools on disk and WRITE their content fingerprints into the "
                         "existing pools.json, leaving every other field untouched. Without this, "
                         "a manifest written before REPAIR v2 can never acquire fingerprints, so "
                         "the drift check can never bind. Regenerates no images; needs no GPU.")
    ap.add_argument("--allow-missing-fingerprints", action="store_true",
                    help="downgrade a missing fingerprint from an error to a warning during "
                         "--verify-only. Use once, to inspect a pre-REPAIR-v2 manifest before "
                         "running --backfill-fingerprints.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--per-prompt", type=int, default=None,
                   help=f"images per PROMPT per split, upstream semantics "
                        f"(gen_images.py uses {UPSTREAM_PER_PROMPT})")
    g.add_argument("--total-per-pool", type=int, default=None,
                   help="images per POOL per split, split across that pool's prompts "
                        "(the historical compute adaptation, NOT the author protocol)")
    ap.add_argument("--n-irt-prompts", type=int, default=6,
                    help="how many unrelated-retain prompts to draw from coco_object_retain.csv")
    ap.add_argument("--resume", action="store_true",
                    help="keep any split that already validates instead of failing")
    a = ap.parse_args()

    if a.per_prompt is None and a.total_per_pool is None:
        a.per_prompt = UPSTREAM_PER_PROMPT
    sizing = "per_prompt" if a.per_prompt is not None else "total_per_pool"

    if a.pool_scheme == "legacy" and not a.historical_reconstruction:
        die("--pool-scheme legacy is a HISTORICAL RECONSTRUCTION, not a scientific setup: "
            "cathedral/chapel/basilica are hyponyms of church and sit in the RELATED-RETAIN pool, "
            "so the objective is satisfied by suppressing the TOKEN 'church' rather than erasing "
            "the concept. Pass --historical-reconstruction to acknowledge that, or use "
            "--pool-scheme coherent for a run intended to support a concept-erasure claim.")

    scheme = SCHEMES[a.pool_scheme]
    coco_path = f"{P}/upstream/Meta-Unlearning/data/coco_object_retain.csv"
    if not os.path.isfile(coco_path):
        die(f"missing {coco_path}")
    import random
    coco = [r["prompt"] for r in csv.DictReader(open(coco_path))]
    if len(coco) < a.n_irt_prompts:
        die(f"coco_object_retain.csv has only {len(coco)} prompts")
    random.Random(0).shuffle(coco)
    pools = {k: (list(v) if v is not None else coco[:a.n_irt_prompts])
             for k, v in scheme["pools"].items()}

    plan = {}
    for name, texts in pools.items():
        if sizing == "per_prompt":
            per = a.per_prompt
        else:
            per = max(1, a.total_per_pool // len(texts))
        plan[name] = {"prompts": texts, "per_prompt": per, "per_split": per * len(texts)}
    print(f"[r2pools] scheme={a.pool_scheme} sizing={sizing}", flush=True)
    for name, p in plan.items():
        up = UPSTREAM_PER_PROMPT * len(p["prompts"])
        flag = "" if p["per_split"] == up else f"  <-- upstream would be {up}/split"
        print(f"  {name}: {len(p['prompts'])} prompt(s) x {p['per_prompt']} "
              f"= {p['per_split']}/split{flag}", flush=True)

    # ---- SPLIT DISJOINTNESS, BEFORE any image is generated or validated (REPAIR v3). This is the
    #      check that was missing: r2_build_meta_pools was absent from split_registry v2's roster,
    #      so nothing ever compared these pools against the evaluation prompts.
    split_rec = check_split_disjointness(a.pool_scheme, plan, a.historical_reconstruction)

    # ---- resume compatibility BEFORE any per-split work, so a definition mismatch is named as
    #      such instead of surfacing as a count error on the first split.
    prior = check_resume_compat(a.root, a.pool_scheme, sizing, plan)
    if prior is not None:
        print(f"  existing manifest at {a.root}/pools.json is COMPATIBLE "
              f"(scheme={prior.get('pool_scheme')} sizing={prior.get('sizing')})", flush=True)

    if a.verify_only or a.backfill_fingerprints:
        what = ("--backfill-fingerprints: validating pools and RECORDING fingerprints"
                if a.backfill_fingerprints
                else "--verify-only: validating pools on disk, writing nothing")
        print(f"[r2pools] {what}", flush=True)
        if a.backfill_fingerprints and prior is None:
            die(f"{a.root}/pools.json does not exist; there is no manifest to backfill. Build the "
                f"pools first.")
        drift, missing, computed = 0, 0, {}
        for name, p in plan.items():
            computed[name] = {}
            for split in ("train", "test"):
                d = os.path.join(a.root, name, split)
                if not os.path.isfile(os.path.join(d, "metadata.jsonl")):
                    die(f"{d}: absent, nothing to verify")
                n, fp = validate(d, p["per_split"], p["prompts"], p["per_prompt"])
                computed[name][split] = fp
                rec = (((prior or {}).get("pools") or {}).get(name) or {}).get("fingerprints") or {}
                was = rec.get(split)
                if was is None:
                    tag = ("no fingerprint recorded -> RECORDING" if a.backfill_fingerprints
                           else "NO FINGERPRINT RECORDED (manifest predates REPAIR v2)")
                    missing += 0 if a.backfill_fingerprints else 1
                elif was == fp:
                    tag = "fingerprint MATCHES manifest"
                else:
                    tag = f"fingerprint DRIFT: manifest={was[:16]}"
                    drift += 1
                print(f"  {name}/{split}: n={n} prompts={len(p['prompts'])} "
                      f"per_prompt={p['per_prompt']} sha={fp[:16]} -- {tag}", flush=True)
        print(f"[r2pools] CLAIM SCOPE: {scheme['claim']}", flush=True)
        if drift:
            die(f"{drift} split(s) no longer match the recorded fingerprint; the bytes on disk "
                f"are not the bytes the manifest describes")
        if a.backfill_fingerprints:
            # Update ONLY the fingerprints field of each pool; everything else is left verbatim.
            for name, fps in computed.items():
                prior.setdefault("pools", {}).setdefault(name, {})["fingerprints"] = fps
            prior["fingerprints_backfilled_by"] = "scripts/r2_build_meta_pools.py " \
                                                  "--backfill-fingerprints"
            mp = os.path.join(a.root, "pools.json")
            with open(mp, "w") as f:
                json.dump(prior, f, indent=1)
            print(f"[r2pools] recorded fingerprints for {sum(len(v) for v in computed.values())} "
                  f"split(s) -> {mp}", flush=True)
            sys.exit(0)
        # A fingerprint that is never recorded is a check that can never bind, so a MISSING
        # fingerprint is an ERROR by default -- not a tolerated note.
        if missing and not a.allow_missing_fingerprints:
            die(f"{missing} split(s) carry no recorded fingerprint, so the drift check cannot "
                f"bind and these pools are NOT provably the ones any earlier run used. Run "
                f"`--backfill-fingerprints` (validates and records, no GPU, no regeneration), or "
                f"pass --allow-missing-fingerprints to inspect without binding.")
        if missing:
            print(f"[r2pools] WARNING: {missing} split(s) unfingerprinted and the check is "
                  f"NOT BINDING (--allow-missing-fingerprints)", flush=True)
        print("[r2pools] verify-only: OK", flush=True)
        sys.exit(0)

    todo = []
    for name, p in plan.items():
        for split in ("train", "test"):
            d = os.path.join(a.root, name, split)
            if os.path.isfile(os.path.join(d, "metadata.jsonl")):
                if not a.resume:
                    die(f"{d} already exists; pass --resume to keep it or remove it first")
                validate(d, p["per_split"], p["prompts"], p["per_prompt"])
                print(f"  {name}/{split} exists and validates, keeping", flush=True)
                continue
            todo.append((name, split, d, p))

    manifest = {
        "script": "scripts/r2_build_meta_pools.py",
        "root": os.path.abspath(a.root),
        "pool_scheme": a.pool_scheme,
        "historical_reconstruction_ack": bool(a.historical_reconstruction),
        "scheme_claim": scheme["claim"],
        "forget_side_pools": scheme["forget_side"],
        "sizing": sizing,
        "per_prompt_requested": a.per_prompt,
        "total_per_pool_requested": a.total_per_pool,
        "upstream_per_prompt": UPSTREAM_PER_PROMPT,
        "upstream_reference": "upstream/Meta-Unlearning/src/gen_images.py L90-92",
        "negative_prompt": NEG,
        "guidance_scale": CFG,
        "seed_base": SEED_BASE,
        "trainer_pools": ["hrm", "irt", "tgt"],
        "split_disjointness": split_rec,
        "pools": {},
    }

    if todo:
        import torch
        from diffusers import StableDiffusionPipeline
        pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float16,
            variant="fp16", safety_checker=None, requires_safety_checker=False).to("cuda")
        manifest["base_model"] = "CompVis/stable-diffusion-v1-4"
        manifest["base_revision"] = pin["revision"]
        for name, split, d, p in todo:
            data, rejects = [], 0
            for j, t in enumerate(p["prompts"]):
                got, rej = gen(pipe, t, p["per_prompt"], SEED_BASE[split] + j * 100)
                data += got
                rejects += rej
            save(data, d)
            n, _ = validate(d, p["per_split"], p["prompts"], p["per_prompt"])
            print(f"  {name}/{split}: {n} images, {rejects} blank rejected, from {p['prompts']}",
                  flush=True)

    for name, p in plan.items():
        counts, fps = {}, {}
        for split in ("train", "test"):
            counts[split], fps[split] = validate(os.path.join(a.root, name, split),
                                                 p["per_split"], p["prompts"], p["per_prompt"])
        manifest["pools"][name] = {
            "prompts": p["prompts"], "per_prompt": p["per_prompt"],
            "counts": counts,
            "fingerprints": fps,
            "scored_as": "forget" if name in scheme["forget_side"] else "retain",
            "consumed_by_trainer": name in ("hrm", "irt", "tgt"),
        }
    mp = os.path.join(a.root, "pools.json")
    os.makedirs(a.root, exist_ok=True)
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[r2pools] all pools validated; manifest -> {mp}", flush=True)
    print(f"[r2pools] CLAIM SCOPE: {scheme['claim']}", flush=True)
