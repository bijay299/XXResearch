"""SHARED zero-shot CLIP scorer: ONE source of truth for the classifier, the label text and the
identity mapping.

WHY THIS FILE EXISTS (defect D-013). `gen_eval.py` carries two twelve-element lists:

    LABELS  (gen_eval.py:48)  the CLIP TEXT    -- what CLIP is asked to choose among
    LAB_KEY (gen_eval.py:50)  the RECORD KEYS  -- what a prediction *is* in a JSON record

`gen_eval_final.py` originally passed `list(ge.CONCEPTS)` (element-wise equal to `LAB_KEY`) as the
CLIP text. NINE of the twelve strings differ between the two lists; only `church`, `parachute` and
`tench` are identical. The FINAL-split numbers in `outputs/final_*.json` were therefore produced
with label text inconsistent with the rest of the corpus.

WHAT IS AND IS NOT ESTABLISHED. The mismatch is established and is verified mechanically by
`label_set_diff()` below. Its DIRECTION AND MAGNITUDE ARE NOT. An underscore is a different token
sequence and so a different text embedding, but whether that moves a given similarity up or down,
and by how much, is an empirical question that no part of this module answers or presumes. The
sanctioned reading until measured is exactly:

    "The evaluator used inconsistent label text. The effect on scores and conclusions is
     unresolved pending corrected evaluation."

Nothing here may be read as implying that the legacy wording weakened competitors, that any
historical count was inflated or deflated, or that arms whose label text did not change are
unaffected (the closed-set argmax couples every label to every other, so a change to any
competitor's text can move any arm's decision in either direction).

IMPORT COMPATIBILITY. `gen_eval.py` is NOT modified and does NOT import this module -- it keeps its
own in-file `clip_scorer()` and its own constants, so every historical record reproduces unchanged.
This module imports FROM `gen_eval.py`, making `gen_eval.py` the upstream authority for the label
text while the *scoring implementation* is shared by the newer scripts. `verify_numerics_contract()`
asserts, on CPU and without loading any weights, that this module builds the identical text prompt
list and uses the identical normalisation recipe as `gen_eval.clip_scorer()`.
"""
import importlib.util as iu
import os

P = "/data/bijaypandey/t2i_unlearn"
SCRIPTS = os.path.join(P, "scripts")

#: Bumped whenever the scoring recipe changes in any way that can move a number. Recorded in every
#: emitted record so two records can never be silently compared across recipes.
SCORING_VERSION = "clip_scorer/v2.0.0"

#: The classifier. Its revision is UNPINNED upstream and that is a disclosed provenance gap
#: (outputs/provenance_manifest.json -> models.classifier.warning), not something this module fixes:
#: pinning it now would not retroactively identify the weights the historical runs used.
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = None
CLIP_REVISION_NOTE = ("UNPINNED upstream in every historical run; left unpinned here so the "
                      "re-scoring uses the same locally cached weights rather than silently "
                      "introducing a second classifier identity into the comparison.")

#: The zero-shot prompt wrapper. Identical in `gen_eval.clip_scorer()` (line 81) and in
#: `gen_eval_final.py` (line 145); asserted by `verify_numerics_contract()`.
PROMPT_TEMPLATE = "a photo of a {}"

#: Numerics recipe, recorded verbatim in provenance so a future reader need not re-derive it.
NUMERICS = {
    "dtype": "float16",
    "text": "CLIPModel.get_text_features(**processor(text=[...], padding=True)) then L2-normalise "
            "over dim=-1",
    "image": "CLIPProcessor(images=[...])['pixel_values'].half() -> "
             "CLIPModel.get_image_features(...) then L2-normalise over dim=-1",
    "similarity": "image_features @ text_features.T (cosine, both operands unit-norm)",
    "decision": "closed-set argmax over the label axis; NO reject option, NO threshold",
}


def load_gen_eval():
    """Import `gen_eval.py` by path. The T2I tree is not a package and not on sys.path."""
    spec = iu.spec_from_file_location("_cs_gen_eval", os.path.join(SCRIPTS, "gen_eval.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ label sets / identity mapping
#: The two label-TEXT sets, by name. `corrected` is what every H number in the corpus uses;
#: `legacy` is what the pre-repair FINAL records used. Both are scored on the same images by
#: `rescore_final_v2.py` so the label-text effect is measured rather than argued about.
LABEL_SET_NAMES = ("corrected", "legacy")


def label_sets():
    """-> (identity_keys, {"corrected": [...12 text...], "legacy": [...12 text...]}).

    `identity_keys` is the ONLY thing that says what a prediction *is*; it is the same list under
    both label-text sets, so a prediction index means the same concept either way and the two
    scorings are directly comparable index-by-index.
    """
    ge = load_gen_eval()
    identity = list(ge.LAB_KEY)
    sets = {"corrected": list(ge.LABELS), "legacy": list(ge.LAB_KEY)}
    assert list(ge.CONCEPTS) == identity, (
        "gen_eval.CONCEPTS ordering no longer matches gen_eval.LAB_KEY; the index->concept "
        "identity mapping would be wrong and every per-image prediction would be mislabelled")
    for name, txt in sets.items():
        assert len(txt) == len(identity), f"label set {name!r} length != identity length"
    return identity, sets


def label_set_diff():
    """Mechanically enumerate where the two label-text sets differ.

    Returns {"n_labels", "n_differ", "n_identical", "differing" [...], "identical" [...], "rows"}.
    Direction/size of the scoring consequence is deliberately NOT inferred here.
    """
    identity, sets = label_sets()
    legacy, corrected = sets["legacy"], sets["corrected"]
    rows, differ, same = [], [], []
    for i, key in enumerate(identity):
        d = legacy[i] != corrected[i]
        rows.append({"index": i, "identity_key": key, "legacy_text": legacy[i],
                     "corrected_text": corrected[i], "differs": d,
                     "legacy_prompt": PROMPT_TEMPLATE.format(legacy[i]),
                     "corrected_prompt": PROMPT_TEMPLATE.format(corrected[i])})
        (differ if d else same).append(key)
    return {"n_labels": len(identity), "n_differ": len(differ), "n_identical": len(same),
            "differing": differ, "identical": same, "rows": rows,
            "interpretation": "The evaluator used inconsistent label text. The effect on scores "
                              "and conclusions is unresolved pending corrected evaluation. "
                              "Direction and magnitude are NOT established by the existence of "
                              "the mismatch; an underscore-vs-space edit changes the token "
                              "sequence and so the text embedding, but the sign of the resulting "
                              "similarity change is empirical.",
            "scoring_version": SCORING_VERSION}


def verify_numerics_contract():
    """CPU-only, no weights: assert this module agrees with `gen_eval.py` on label text and recipe.

    Guards the 'one source of truth' property (requirement 5): if someone edits `gen_eval.LABELS`
    or the `a photo of a {}` wrapper, this fails instead of silently forking the numerics.
    """
    ge = load_gen_eval()
    identity, sets = label_sets()
    issues = []
    if sets["corrected"] != list(ge.LABELS):
        issues.append("corrected label text is not gen_eval.LABELS")
    if sets["legacy"] != list(ge.LAB_KEY):
        issues.append("legacy label text is not gen_eval.LAB_KEY")
    if identity != list(ge.LAB_KEY):
        issues.append("identity keys are not gen_eval.LAB_KEY")
    # gen_eval.clip_scorer builds `f"a photo of a {l}" for l in LABELS`; reproduce it from the
    # template and require string equality, so a wrapper change cannot drift unnoticed.
    src = open(os.path.join(SCRIPTS, "gen_eval.py")).read()
    if 'f"a photo of a {l}" for l in LABELS' not in src:
        issues.append("gen_eval.clip_scorer no longer builds 'a photo of a {l}' over LABELS; "
                      "PROMPT_TEMPLATE/NUMERICS here may have drifted from it")
    if [PROMPT_TEMPLATE.format(l) for l in sets["corrected"]] != \
            [f"a photo of a {l}" for l in ge.LABELS]:
        issues.append("PROMPT_TEMPLATE does not reproduce gen_eval's text prompt list")
    if "f / f.norm(dim=-1, keepdim=True)" not in src:
        issues.append("gen_eval no longer L2-normalises image features as NUMERICS claims")
    return {"ok": not issues, "issues": issues, "scoring_version": SCORING_VERSION,
            "clip_model_id": CLIP_MODEL_ID, "prompt_template": PROMPT_TEMPLATE,
            "numerics": NUMERICS}


# ------------------------------------------------------------------ the scorer itself
class ClipScorer:
    """Zero-shot CLIP classifier, numerically identical to `gen_eval.clip_scorer()`'s recipe.

    Weights are loaded on construction, so CPU tests must NOT construct this -- they call the
    pure functions above instead.
    """

    def __init__(self, device="cuda", dtype=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._torch = torch
        self.device = device
        self.dtype = dtype or torch.float16
        kw = {} if CLIP_REVISION is None else {"revision": CLIP_REVISION}
        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, torch_dtype=self.dtype, **kw).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, **kw)
        self.identity, self._sets = label_sets()
        self._tfeat = {}

    # -- text side -------------------------------------------------------
    def text_features(self, label_set_name):
        """L2-normalised text features for one named label set, cached."""
        if label_set_name not in self._tfeat:
            text = self._sets[label_set_name]
            with self._torch.no_grad():
                t = self.processor(text=[PROMPT_TEMPLATE.format(l) for l in text],
                                   return_tensors="pt", padding=True).to(self.device)
                f = self.model.get_text_features(**t)
                f = f / f.norm(dim=-1, keepdim=True)
            self._tfeat[label_set_name] = f
        return self._tfeat[label_set_name]

    # -- image side ------------------------------------------------------
    def image_features(self, images):
        """L2-normalised image features, (n_images, d). Returned for persistence so that any
        FUTURE re-scoring against a new label set needs no regeneration."""
        with self._torch.no_grad():
            ii = self.processor(images=list(images), return_tensors="pt").to(self.device)
            f = self.model.get_image_features(pixel_values=ii["pixel_values"].to(self.dtype))
            f = f / f.norm(dim=-1, keepdim=True)
        return f

    # -- scoring ---------------------------------------------------------
    def score_features(self, img_feat, label_set_name):
        """Cosine similarity of pre-computed image features against one label set: (n_img, 12)."""
        return img_feat @ self.text_features(label_set_name).T

    def score_all_sets(self, images):
        """Score images against EVERY label set at once on the SAME features.

        -> (img_feat, {set_name: sims tensor (n_img, 12)}). Scoring both sets off one feature
        tensor is what isolates the label-TEXT effect: the image side is bit-identical between
        the two, so any difference in the decision is attributable to the text alone.
        """
        f = self.image_features(images)
        return f, {name: self.score_features(f, name) for name in self._sets}

    def per_image_records(self, sims, target_key):
        """Full per-image detail for ONE label set: the complete 12-label score vector, not only
        the argmax. Keeping all twelve is what makes future re-scoring and re-analysis possible.
        """
        out = []
        tgt = self.identity.index(target_key)
        for row in sims.float().cpu():
            vals = row.tolist()
            order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
            top, second = order[0], order[1]
            out.append({
                "pred_index": top,
                "pred_label": self.identity[top],
                "is_target": self.identity[top] == target_key,
                "margin": vals[top] - vals[second],
                "runner_up_label": self.identity[second],
                "target_sim": vals[tgt],
                "target_rank": order.index(tgt) + 1,
                "scores": {self.identity[i]: vals[i] for i in range(len(vals))},
            })
        return out
