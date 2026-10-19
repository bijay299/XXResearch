"""Frozen object-presence scorer and coherence rubric.

SELECTED BEFORE ANY EDITED OUTPUT EXISTS. Identity, revision, prompts,
thresholds and uncertainty handling are all frozen in `SCORER_CONFIG` below and
hashed into the run manifest. No scorer shopping and no threshold retuning
after inspecting failures is permitted.

Raw CLIP cosine similarity is NOT assumed to be calibrated object presence.
What is used is a two-way contrastive decision rule (present-phrases versus
explicit absent-phrases) whose operating point is QUALIFIED against separately
recorded visual judgments on 64 held-apart calibration images, and which must
reach >= 80% sensitivity and >= 80% specificity to be usable at all.
"""

import hashlib
import json

import torch

CLIP_MODEL = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"

SCORER_CONFIG = {
    "model": CLIP_MODEL,
    "revision": CLIP_REVISION,
    "kind": "two-way contrastive present/absent decision rule over CLIP image-text logits",
    "present_templates": [
        "a photo of a {o}",
        "a photo containing a {o}",
        "a {o} is visible in this image",
    ],
    "absent_templates": [
        "a photo with no {o}",
        "a photo that does not contain a {o}",
        "no {o} is visible in this image",
    ],
    "aggregation": "mean of CLIP logits within each template group, then 2-way softmax",
    "threshold_present": 0.60,     # p_present >= 0.60  -> present
    "threshold_absent": 0.40,      # p_present <= 0.40  -> absent
    "uncertainty_handling": (
        "0.40 < p_present < 0.60 is recorded as 'uncertain' and never silently "
        "resolved. For scientific metrics, an uncertain or missing EDITED TARGET "
        "output counts as UNSUCCESSFUL SUPPRESSION, and an uncertain or missing "
        "ALLOWED output counts as UNSUCCESSFUL PRESERVATION."),
    "qualification_gate": {
        "min_definite_positives": 20,
        "min_definite_negatives": 20,
        "min_sensitivity": 0.80,
        "min_specificity": 0.80,
        "reference": "separately recorded visual judgments made WITHOUT scorer outputs",
    },
}

# Frozen whole-frame coherence rubric. Applied as a MODEL-ASSISTED visual
# judgment where independent human validation is unavailable.
COHERENCE_RUBRIC = {
    "field": "whole_frame_coherence",
    "labels": ["coherent", "degenerate", "uncertain"],
    "coherent": (
        "The frame reads as a single plausible photograph: global structure, "
        "lighting and perspective are consistent, and the depicted scene is "
        "recognisable as a scene."),
    "degenerate": (
        "The WHOLE FRAME has broken down: texture mush or noise across most of "
        "the image, colour collapse, repeated tiling artefacts, or no "
        "recognisable scene structure."),
    "uncertain": "Neither description clearly applies.",
    "explicitly_not_degenerate": (
        "A distorted face, malformed hand, wrong object count, implausible "
        "physics or a local artefact is a LOCAL DEFECT, recorded in the "
        "`local_defects` field. On its own it does NOT make the frame "
        "degenerate."),
    "separation_note": (
        "Target absence does NOT establish preserved quality; coherence is "
        "scored independently of object presence, and target-panel coherence "
        "is reported separately from allowed-panel preservation."),
}


def config_hash():
    blob = json.dumps({"scorer": SCORER_CONFIG, "coherence": COHERENCE_RUBRIC},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class PresenceScorer:
    def __init__(self, device="cuda:0"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL, revision=CLIP_REVISION, torch_dtype=torch.float32).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
        self._text_cache = {}

    @torch.no_grad()
    def _text_feats(self, obj):
        if obj in self._text_cache:
            return self._text_cache[obj]
        pos = [t.format(o=obj) for t in SCORER_CONFIG["present_templates"]]
        neg = [t.format(o=obj) for t in SCORER_CONFIG["absent_templates"]]
        tok = self.proc(text=pos + neg, return_tensors="pt", padding=True).to(self.device)
        f = self.model.get_text_features(**tok)
        f = f / f.norm(dim=-1, keepdim=True)
        out = (f[:len(pos)], f[len(pos):])
        self._text_cache[obj] = out
        return out

    @torch.no_grad()
    def score(self, image, obj):
        """Return {p_present, label, margin} for one PIL image and probe object."""
        pos_f, neg_f = self._text_feats(obj)
        px = self.proc(images=image, return_tensors="pt").to(self.device)
        imf = self.model.get_image_features(**px)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        scale = self.model.logit_scale.exp()
        lp = (scale * imf @ pos_f.T).mean()
        ln = (scale * imf @ neg_f.T).mean()
        p = float(torch.softmax(torch.stack([lp, ln]), dim=0)[0])
        if p >= SCORER_CONFIG["threshold_present"]:
            label = "present"
        elif p <= SCORER_CONFIG["threshold_absent"]:
            label = "absent"
        else:
            label = "uncertain"
        return {"p_present": p, "label": label, "margin": float(lp - ln)}


def qualify(pairs):
    """Qualify the scorer against separately recorded visual judgments.

    `pairs` = [{"visual": "present"|"absent"|"uncertain", "scorer": <label>}, ...]
    Uncertain VISUAL labels are preserved and excluded from the definite
    counts; uncertain SCORER labels count as errors against the gate (they are
    not resolved in the scorer's favour).
    """
    pos = [p for p in pairs if p["visual"] == "present"]
    neg = [p for p in pairs if p["visual"] == "absent"]
    unc = [p for p in pairs if p["visual"] == "uncertain"]
    tp = sum(1 for p in pos if p["scorer"] == "present")
    tn = sum(1 for p in neg if p["scorer"] == "absent")
    sens = tp / len(pos) if pos else 0.0
    spec = tn / len(neg) if neg else 0.0
    g = SCORER_CONFIG["qualification_gate"]
    passed = (len(pos) >= g["min_definite_positives"]
              and len(neg) >= g["min_definite_negatives"]
              and sens >= g["min_sensitivity"]
              and spec >= g["min_specificity"])
    return {"n_definite_positive": len(pos), "n_definite_negative": len(neg),
            "n_visual_uncertain": len(unc), "true_positive": tp, "true_negative": tn,
            "sensitivity": sens, "specificity": spec,
            "gate": g, "passed": bool(passed),
            "verdict": "SCORER_QUALIFIED" if passed else "SCORER_NOT_QUALIFIED"}
