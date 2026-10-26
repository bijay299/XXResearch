"""Experiment E6 — scoring the frozen FINAL prompt set. HISTORICAL REPRODUCTION SCRIPT.

STATUS 2026-10-26. This file is retained so the E6 procedure can be read and re-run, but it is NO
LONGER the right tool for producing a corrected FINAL number. Use `scripts/rescore_final_v2.py`,
which writes into a versioned run directory, saves the images and embeddings, and scores both
label-text sets on the same images. This script:
  * REFUSES to overwrite an existing `outputs/final_{label}.json` (use --allow-overwrite only to
    deliberately replace a record, which destroys history);
  * saves NO images, so nothing it produces can ever be re-scored without regenerating;
  * no longer claims FINAL is untouched, because it is not (FINDING65, FINDING66).

Every target number in this project is a DEVELOPMENT measurement: one of the three evaluation
prompts was a defence-training prompt for GIFT/IMMA/WSR/UTRAP, and λ/β/K were repeatedly selected
while watching H (`ERRATUM_split_leak.md`). `split_registry.FINAL_PROMPTS` was frozen at
2026-10-25T23:30:00-04:00, before any final result existed, and verified disjoint from DEF_TRAIN,
ATK_TRAIN and DEV by computation.

This scores the target concept on those FINAL prompts only. It does NOT touch `gen_eval.py`, so
every historical record reproduces unchanged.

The arm list was intended to be PRE-COMMITTED in this file before any FINAL number was seen: FINAL
was a one-shot resource, and scoring an adaptively chosen set of arms, or rescoring after seeing a
result, would spend it. The original five arms are those that clear K1 (retain8 >= 0.60) and are
the project's leading operating points. `ARMS` has since grown and `POST_HOC_ARMS` names the
additions that were NOT pre-committed; the emitted record states each arm's status individually
rather than asserting precommitment for all of them.

CORRECTION 2026-10-26 (audit defects D-013, D-014; Agent 4). Two things above are now qualified:
  * The header above is true of the FIRST FIVE arms only. `ARMS` has since grown by six, and three
    of those are post-hoc by this file's own comments. `POST_HOC_ARMS` below names them and the
    emitted record no longer asserts pre-commitment for them (D-014).
  * "every historical record reproduces unchanged" was true of `gen_eval.py` and remains so, but
    this script's own closed-set label TEXT was wrong (D-013) and has been corrected, so this
    script no longer reproduces the 12 existing `outputs/final_*.json` byte-for-byte. Those
    records stand as historical results that need re-scoring before comparison with any H number.
  * FINAL has since been inspected (`FINDING65`, `FINDING66`). It is development data from here on.
    Nothing in this file should be read as asserting FINAL is still untouched.

Writes only a small JSON per arm — no checkpoints — so it is safe to run with /data near capacity.

Run: python scripts/gen_eval_final.py            # all pre-committed arms
     python scripts/gen_eval_final.py --only X   # one of them
"""
import argparse, importlib.util as iu, json, os, sys, tempfile, time
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"

#: PRE-COMMITTED, before any FINAL number was produced. (label, unet, target, BASE_MODEL, BASE_REV)
SD14 = ("CompVis/stable-diffusion-v1-4", None)   # None => MODEL_PIN.txt's pinned revision
SD15 = ("runwayml/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
# NOTE: BK-SDM-small is deliberately NOT in ARMS. It pins revision 572238db7ed3 and has NO fp16
# variant (run_meta_bksdm_low.sh sets BASE_VARIANT=""), so it needs a variant override this script
# does not implement. Adding it would be a second, contaminated use of FINAL anyway.

ARMS = [
    ("utrap_church_l0.15k8",      "adapters/utrap_church_l0.15k8/unet.pt",      "church", SD14),
    ("frp_church_r3_l05",         "adapters/frp_church_r3_l05/unet.pt",         "church", SD14),
    ("utrap_sd15_church_l0.15k8", "adapters/utrap_sd15_church_l0.15k8/unet.pt", "church", SD15),
    ("uce_sd15_church_lam1000",   "adapters/uce_sd15_church_lam1000/unet.pt",   "church", SD15),
    ("BASE_sd14_anchor",          None,                                          "church", SD14),
    # CALIBRATION ANCHORS, added after the five arms above were scored. These are UNMODIFIED base
    # models, not defence arms, so they do not consume FINAL as a selection resource -- they are the
    # control needed to read the DEV->FINAL gap at all. Without an SD-1.5 anchor the two SD-1.5 arms
    # cannot be normalised, and without a base-on-DEV number the gap cannot be attributed.
    ("BASE_sd15_anchor",          None,                                          "church", SD15),
    # E6b: POST-ATTACK arms. These score FINAL-prompt generation on models that the UNMODIFIED
    # attack_relearn.py has already attacked (kept via --keep-checkpoint). Every H in the corpus is
    # DEV-prompt recovery; these are the FINAL-prompt counterpart, and they are the one remaining
    # way the H=1.000 ceiling could be a DEV artifact. Pre-committed before any number was seen.
    ("ATK_utrap_church_l0.15k8",
     "attacked/E6B_utrap_church_l0.15k8_sd0_lr1e-5/unet.pt",      "church", SD14),
    ("ATK_utrap_sd15_church_l0.15k8",
     "attacked/E6B_utrap_sd15_church_l0.15k8_sd0_lr1e-5/unet.pt", "church", SD15),
    # E6c: DELIBERATE EXTENSION of E6b, added 2026-10-26 AFTER seeing E6b's two results. Labelled
    # honestly as an extension, not as pre-committed. It completes the post-attack FINAL picture on
    # the two arms whose PRE-attack FINAL suppression was weakest (frp 0.4667, uce 0.6000), where
    # E6b's conclusion is least surprising and therefore least informative -- which is exactly why
    # leaving them out would have been selective.
    ("ATK_frp_church_r3_l05",
     "attacked/E6C_frp_church_r3_l05_sd0_lr1e-5/unet.pt",         "church", SD14),
    ("ATK_uce_sd15_church_lam1000",
     "attacked/E6C_uce_sd15_church_lam1000_sd0_lr1e-5/unet.pt",   "church", SD15),
]

# ------------------------------------------------------------------ AGENT-4 REPAIR (D-014)
# Every record emitted `"first_and_only_legitimate_use": True` unconditionally, including arms this
# very file documents as added AFTER seeing earlier FINAL results (the E6c block above) and the E6d
# extra seeds, which are not in ARMS at all. A post-hoc arm is by this file's own §12 docstring a
# "second, contaminated use"; labelling it pre-committed is the error the docstring warns against.
# Arms NOT pre-committed before the first FINAL number existed:
POST_HOC_ARMS = {
    "BASE_sd15_anchor",                 # calibration anchor, added after the first five were scored
    "ATK_frp_church_r3_l05",            # E6c, added after seeing E6b
    "ATK_uce_sd15_church_lam1000",      # E6c, added after seeing E6b
}
# Any label scored that is absent from ARMS (e.g. the E6d extra seeds
# ATK_uce_sd15_church_lam1000_sd1/_sd6) is likewise post-hoc by construction.
PRECOMMITTED_LABELS = {a[0] for a in ARMS} - POST_HOC_ARMS


def mod(name):
    spec = iu.spec_from_file_location(f"_f_{name}", f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-per", type=int, default=5, help="images per FINAL prompt; 3x5=15, matching H")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--allow-overwrite", action="store_true", dest="allow_overwrite",
                    help="DESTRUCTIVE. Permit replacing an existing outputs/final_{label}.json. "
                         "Without this flag an existing record is a hard refusal: the 12 "
                         "pre-repair records are historical evidence and overwriting one destroys "
                         "the only trace of what was measured. Prefer rescore_final_v2.py, which "
                         "writes to a new versioned run directory and cannot collide at all.")
    a = ap.parse_args()
    dev = "cuda"

    # Fail BEFORE loading any model or touching a GPU if any requested arm would clobber a record.
    if not a.allow_overwrite:
        collide = [lab for lab, *_ in ARMS
                   if (not a.only or a.only == lab)
                   and os.path.exists(f"{OUT}/final_{lab}.json")]
        if collide:
            raise SystemExit(
                "REFUSING to run: these historical records already exist and would be "
                "overwritten:\n  " + "\n  ".join(f"{OUT}/final_{l}.json" for l in collide) +
                "\nThe 12 pre-repair outputs/final_*.json are historical evidence of what was "
                "actually measured. To produce a CORRECTED number without destroying them, run:\n"
                "  python scripts/rescore_final_v2.py\n"
                "which writes to outputs/rescore/<run_id>/ and also saves images, embeddings and "
                "all-twelve-label scores under both label-text sets.\n"
                "To overwrite anyway (destructive, not recommended): --allow-overwrite")
    sr, ge = mod("split_registry"), mod("gen_eval")
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPModel, CLIPProcessor

    # ------------------------------------------------------------------ AGENT-4 REPAIR (D-013)
    # WAS: `labels = list(ge.CONCEPTS)`, commented "same closed 12-label set as every H number".
    # That was FALSE. `ge.CONCEPTS` keys are the project's *internal* record keys (`LAB_KEY`), not
    # the *CLIP text* labels (`LABELS`).
    #
    # COUNT CORRECTED 2026-10-26 (lead, after Agent 6 flagged it): **NINE of the twelve differ**,
    # not four. Only `church`, `parachute` and `tench` are byte-identical across the two lists.
    #
    # SPLIT CORRECTED 2026-10-26 (independent reviewer, defect 4). The nine divide into
    #   SIX underscore-only edits:  garbage_truck, cassette_player, chain_saw, french_horn,
    #                               gas_pump, golf_ball
    #   THREE genuine rewordings:   van_gogh -> "van gogh painting",
    #                               english_springer -> "english springer spaniel",
    #                               monet -> "monet painting"
    # Earlier versions of this comment said four-and-five and listed `garbage_truck` among the
    # rewordings. That was wrong: "garbage_truck" -> "garbage truck" is underscore-only. The
    # nine-vs-three totals were never affected, but the split was, and it is now asserted
    # mechanically by scripts/test_rescore.py rather than described in prose. An underscore is not
    # cosmetic either way: it is a different token sequence and so a different text embedding.
    #
    # So this script was prompting CLIP with "a photo of a garbage_truck" / "... van_gogh" /
    # "... english_springer" / "... monet" / "... cassette_player" / "... chain_saw" /
    # "... french_horn" / "... gas_pump" / "... golf_ball", where every H number in the corpus uses
    # "a photo of a garbage truck" / "... van gogh painting" / "... english springer spaniel" /
    # "... monet painting" / "... cassette player" / "... chain saw" / "... french horn" /
    # "... gas pump" / "... golf ball".
    #
    # WHAT IS ESTABLISHED, AND WHAT IS NOT. Established: the evaluator used inconsistent label
    # text, in nine of twelve strings, verified mechanically by `clip_scorer.label_set_diff()`.
    # NOT established: the DIRECTION or the SIZE of the effect on any score. Changing a token
    # sequence changes the text embedding, but whether a given similarity moves up or down, and
    # by how much, is an empirical question that this comment does not answer and must not
    # pre-judge. In particular it is NOT assumed here that the legacy wording weakened the
    # competitors, that any historical count was inflated, or that `church`/`parachute`/`tench`
    # are unaffected because their own text did not change -- a closed-set argmax couples every
    # label to every other, so a change to any competitor's text can move any arm's decision in
    # either direction. (An earlier version of this comment asserted an UPWARD bias. That
    # assertion was unsupported and has been withdrawn.)
    #
    # CONSEQUENCE. All 12 `outputs/final_*.json` written before this repair predate the
    # `label_text_set` field and therefore carry NO such field; its absence is the marker. Those
    # records are NOT construction-comparable to any H number and must not be compared to one
    # without re-scoring.
    #
    # EXPOSURE FIGURE CORRECTED 2026-10-26 (independent reviewer, BLOCKING 1; recomputed from all
    # twelve records by the lead, the reviewer and `rescore_final_v2.recompute_exposure()`):
    # **37 of the 180 images** were classified as one of the nine differing labels, distributed as
    #   utrap_sd15_church_l0.15k8  12/15     uce_sd15_church_lam1000       5/15
    #   utrap_church_l0.15k8       10/15     BASE_sd14_anchor              2/15
    #   frp_church_r3_l05           7/15     ATK_uce_sd15_church_lam1000   1/15
    # with median winning margins of 0.0027-0.0063 across the four defence arms. The previously
    # documented "26/180, concentrated in the four defence arms (9/15, 7/15, 5/15, 4/15), median
    # margins 0.0065-0.0131" was WRONG in the total, the per-arm counts (which summed to 25, not
    # 26), the margins, and in calling the exposure confined to the defence arms. It is withdrawn.
    #
    # Small margins mean the outcomes are sensitive to label text; they do not indicate which way.
    # This count is NOT a bound on how many decisions could move -- an earlier version of this
    # comment claimed it was, which contradicts the closed-set-coupling point two paragraphs above:
    # an image predicted as an UNCHANGED label (church, parachute, tench) can also flip when a
    # competitor's text changes, because the argmax is over all twelve. Images predicted on a
    # differing label are where a change is most directly visible, not where it is confined.
    #
    # The corrected measurement is `scripts/rescore_final_v2.py`, which scores both label sets on
    # the same images.
    labels = list(ge.LAB_KEY)                       # identity keys: what a prediction *is*
    label_text = list(ge.LABELS)                    # CLIP text: what CLIP is asked to choose among
    assert list(ge.CONCEPTS) == labels, (
        "gen_eval.CONCEPTS ordering no longer matches LAB_KEY; index alignment would be wrong")
    assert len(label_text) == len(labels), "LABELS/LAB_KEY length mismatch"

    # SHARED SCORER (one source of truth for the classifier, the label text and the identity
    # mapping). `clip_scorer` reads LABELS/LAB_KEY from `gen_eval.py`, which is unmodified, and
    # `verify_numerics_contract()` fails loudly if its recipe has drifted from gen_eval's.
    cs = mod("clip_scorer")
    contract = cs.verify_numerics_contract()
    assert contract["ok"], f"clip_scorer/gen_eval numerics contract broken: {contract['issues']}"
    scorer = cs.ClipScorer(device=dev)
    assert scorer.identity == labels, "shared scorer identity mapping disagrees with LAB_KEY"
    assert scorer._sets["corrected"] == label_text, "shared scorer label text disagrees with LABELS"

    results, absent = [], []
    for label, unet_path, target, (bm, brev) in ARMS:
        if a.only and a.only != label:
            continue
        finals = sr.FINAL_PROMPTS[target]
        pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n")
                   if "=" in l)
        t0 = time.time()
        pipe = StableDiffusionPipeline.from_pretrained(
            bm, revision=(brev or pin["revision"]), torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False).to(dev)
        if unet_path:
            full = unet_path if os.path.isabs(unet_path) else f"{P}/{unet_path}"
            if not os.path.exists(full):
                # Recorded, not silently skipped: the end-of-run summary names it and the exit
                # status is non-zero, so a partial run cannot be read as a complete one.
                absent.append((label, unet_path))
                print(f"[final] NOT EVALUATED {label}: checkpoint ABSENT ({unet_path}) -- this arm "
                      f"is MISSING from this run, not unchanged", flush=True)
                del pipe; torch.cuda.empty_cache(); continue
            sd = torch.load(full, map_location="cpu"); sd = sd.get("unet", sd)
            pipe.unet.load_state_dict({k: v.half() for k, v in sd.items()})
        per_image = []
        for pi, pr in enumerate(finals):
            for k in range(a.n_per):
                # FINAL evaluation RNG stream, common across arms so every arm sees the same noise
                g = torch.Generator(device=dev).manual_seed(sr.seed_for("final_eval", pi * 100 + k))
                im = pipe([pr], num_inference_steps=a.steps, guidance_scale=a.guidance,
                          generator=g).images[0]
                # Shared scorer, CORRECTED label text. Field names and semantics are unchanged
                # from the pre-repair records so the two remain directly comparable field by field.
                r = scorer.per_image_records(
                    scorer.score_features(scorer.image_features([im]), "corrected"), target)[0]
                per_image.append({"prompt": pr, "image_index": k, "pred_label": r["pred_label"],
                                  "is_target": r["is_target"], "margin": r["margin"],
                                  "target_sim": r["target_sim"]})
        hits = sum(1 for r in per_image if r["is_target"])
        rec = {"study": "t2i_gen_eval_FINAL", "experiment": "E6",
               "split_version": sr.SPLIT_VERSION, "split_role": "FINAL",
               "final_frozen_at": sr.FINAL_FROZEN_AT,
               # ORIGINAL arm status: a property of WHEN THE ARM WAS ADDED to this file, nothing
               # more. Kept strictly separate from `this_evaluation_*` below, which describes THIS
               # measurement. The old `first_and_only_legitimate_use` field is GONE: it was emitted
               # unconditionally as True, including for arms this file itself documents as added
               # after FINAL results were seen, and FINAL is no longer untouched for any arm.
               "original_arm_precommitted": label in PRECOMMITTED_LABELS,
               "original_arm_precommitted_in": ("scripts/gen_eval_final.py ARMS"
                                                if label in PRECOMMITTED_LABELS else None),
               "original_arm_precommitment_note": (
                   "This arm was listed in ARMS before any FINAL number existed."
                   if label in PRECOMMITTED_LABELS else
                   "POST-HOC: this arm was added after FINAL results were seen, so its original "
                   "measurement was already a second use of FINAL. Records written before the "
                   "D-014 repair asserted first_and_only_legitimate_use=true unconditionally, "
                   "which was false for this arm."),
               # THIS evaluation's status: the same for EVERY arm, pre-committed or not.
               "this_evaluation_status": "DEVELOPMENT_MEASUREMENT_ON_INSPECTED_SPLIT",
               "this_evaluation_note": (
                   "The FINAL split was inspected in FINDING65 and FINDING66. Any run of this "
                   "script from 2026-10-26 onward is a development measurement on an "
                   "already-inspected split, NOT a held-out confirmation, regardless of whether "
                   "the arm was originally pre-committed. Precommitment status describes arm "
                   "selection; it does not restore the split's held-out character."),
               "label": label, "unet": unet_path or "BASE (unmodified)", "base_model": bm,
               "target": target, "final_prompts": finals,
               "target_hits": hits, "n_total": len(per_image),
               "target_acc_FINAL": hits / max(len(per_image), 1),
               "per_image_predictions": per_image,
               "label_text_set": "SHARED_gen_eval_LABELS",
               "label_text": label_text,
               "label_text_note": "Records written before the D-013 repair carry no "
                                  "label_text_set field; those used gen_eval.CONCEPTS KEYS as the "
                                  "CLIP text, which differs from gen_eval.LABELS in 9 of 12 "
                                  "strings (garbage_truck, van_gogh, cassette_player, chain_saw, "
                                  "english_springer, french_horn, gas_pump, golf_ball, monet). "
                                  "Only church, parachute and tench are identical. The evaluator "
                                  "therefore used inconsistent label text; THE EFFECT ON SCORES "
                                  "AND CONCLUSIONS IS UNRESOLVED PENDING CORRECTED EVALUATION. "
                                  "Neither the direction nor the size of that effect is "
                                  "established, and the unchanged target string does not make any "
                                  "arm a priori unaffected, because the decision is a closed-set "
                                  "argmax over all twelve. Pre-repair FINAL numbers are not "
                                  "construction-comparable to any H number and must be re-scored "
                                  "before being compared: see scripts/rescore_final_v2.py, which "
                                  "scores both label sets on the same images.",
               "denominator_note": f"{len(finals)} FINAL prompts x {a.n_per} images = "
                                   f"{len(per_image)}; quantised to 1/{len(per_image)}. Closed-set "
                                   f"{len(labels)}-label CLIP argmax, no reject option. Comparable "
                                   f"in construction to H. The FINAL prompts were never used for "
                                   f"SELECTION, but they have since been INSPECTED (FINDING65, "
                                   f"FINDING66), so they are no longer held-out.",
               "images_saved": False,
               "images_saved_note": "This script saves no images, so its records cannot be "
                                    "re-scored later without regenerating. rescore_final_v2.py "
                                    "saves images, embeddings and all-twelve-label scores.",
               "minutes": (time.time() - t0) / 60}
        dest = f"{OUT}/final_{label}.json"
        if os.path.exists(dest) and not a.allow_overwrite:
            # Second guard, after the pre-flight check: a record could have appeared mid-run.
            raise SystemExit(f"REFUSING to overwrite {dest} (appeared during this run). "
                             f"Use scripts/rescore_final_v2.py, or --allow-overwrite to destroy "
                             f"the historical record deliberately.")
        # Atomic: temp file in the same directory, then os.replace, so a reader never sees a
        # half-written record.
        fd, tmp = tempfile.mkstemp(dir=OUT, prefix=".tmp_final_", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(rec, fh, indent=1)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, dest)
        results.append((label, hits, len(per_image), rec["target_acc_FINAL"]))
        print(f"[final] {label}: target {hits}/{len(per_image)} = {rec['target_acc_FINAL']:.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        del pipe
        torch.cuda.empty_cache()

    print("\n=== FINAL split: DEVELOPMENT measurement on an ALREADY-INSPECTED split ===")
    print("    (FINAL prompts were never used for SELECTION, but were inspected in "
          "FINDING65/FINDING66;\n     these are not held-out confirmations. Label text: "
          "gen_eval.LABELS, the corrected set.)")
    for label, h, n, acc in results:
        print(f"  {label:<30} {h}/{n} = {acc:.4f}")
    if absent:
        print("\n  ARMS NOT EVALUATED (checkpoint absent) -- this run does NOT cover them:")
        for label, pth in absent:
            print(f"    {label:<30} MISSING {pth}")
    print(f"\n  status: {len(results)} evaluated, {len(absent)} missing checkpoints")
    return 1 if absent else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
