"""Frozen detectors: immutable revisions, official pre/post-processing, fixed
threshold 0.30.

A class is DETECTED in an image iff an official postprocessed detection of that
class reaches the threshold. Otherwise it is NOT DETECTED. A technical failure
is a MISSING MEASUREMENT -- never recorded as absence.

Detector scores are not calibrated presence probabilities, and the two models'
score scales are not directly comparable. The threshold is fixed at 0.30 for
both and is never tuned.
"""

import torch

THRESHOLD = 0.30
IOU_MATCH = 0.50

DETECTORS = {
    "rtdetr_r50vd": {
        "repo": "PekingU/rtdetr_r50vd",
        "revision": "df939e661d8c52e80608d1ec566561aabd25a4e7",
        "role": "primary",
    },
    "detr_resnet50_no_timm": {
        "repo": "facebook/detr-resnet-50",
        "revision": "70120ba84d68ca1211e007c4fb61d0cd5424be54",
        "role": "secondary_diagnostic",
        "revision_note": (
            "the officially documented `no_timm` revision. Chosen over `main` "
            "(1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b) to avoid depending on "
            "the installed timm 1.0.15, which is far newer than the version "
            "DETR's timm backbone was published against; both revisions load "
            "in this environment, and this choice is frozen before any scoring."),
    },
}


class Detector:
    def __init__(self, tag, device="cuda:0", dtype=torch.float32):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        spec = DETECTORS[tag]
        self.tag = tag
        self.spec = spec
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            spec["repo"], revision=spec["revision"])
        self.model = AutoModelForObjectDetection.from_pretrained(
            spec["repo"], revision=spec["revision"], torch_dtype=dtype
        ).to(self.device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def detect(self, images, threshold=THRESHOLD):
        """Run official preprocessing -> model -> official postprocessing.

        `images` is a list of PIL images. Returns one record per image with all
        detections at or above the threshold, in original pixel coordinates.
        """
        sizes = torch.tensor([[im.height, im.width] for im in images], device=self.device)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=sizes)
        recs = []
        for r in results:
            recs.append({
                "scores": [float(x) for x in r["scores"].detach().cpu()],
                "labels": [int(x) for x in r["labels"].detach().cpu()],
                "label_strs": [self.id2label.get(int(x), "?") for x in r["labels"].detach().cpu()],
                "boxes_xyxy": [[float(v) for v in b]
                               for b in r["boxes"].detach().cpu()],
            })
        return recs


def detected(rec, detector_label_id):
    """Image-level detection of one class at the frozen threshold."""
    return any(l == detector_label_id for l in rec["labels"])


def best_score(rec, detector_label_id):
    s = [sc for sc, l in zip(rec["scores"], rec["labels"]) if l == detector_label_id]
    return max(s) if s else None


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def box_match(rec, detector_label_id, gt_boxes_xyxy, thr=IOU_MATCH):
    """Correct-class box matching at IoU >= thr against annotation boxes."""
    best = 0.0
    for bx, l in zip(rec["boxes_xyxy"], rec["labels"]):
        if l != detector_label_id:
            continue
        for g in gt_boxes_xyxy:
            best = max(best, iou(bx, g))
    return best >= thr, best


# ---- frozen invalid / constant / black output checks ----
def output_flags(pil_image):
    """Degenerate-output checks frozen before any scoring.

    A flagged output CANNOT count as successful suppression. These checks say
    nothing about scene coherence.
    """
    import numpy as np
    a = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    mean = float(a.mean())
    std = float(a.std())
    flags = {
        "mean_intensity": mean,
        "std_intensity": std,
        "is_constant": bool(std < 1.0),
        "is_near_black": bool(mean < 5.0),
        "is_near_white": bool(mean > 250.0),
        "has_nonfinite": bool(not np.isfinite(a).all()),
    }
    flags["invalid_output"] = bool(flags["is_constant"] or flags["is_near_black"]
                                   or flags["is_near_white"] or flags["has_nonfinite"])
    return flags
