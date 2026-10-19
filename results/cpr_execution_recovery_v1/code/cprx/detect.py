"""Frozen detectors (official processors + post-processing, threshold 0.30) and scoring helpers.
Detector output is a detector-defined proxy only; technical errors are MISSING measurements."""
import os

import numpy as np
import torch

from cprx.protocol import DETECTORS, IOU_MATCH, THRESHOLD, COLLAPSE_STD_THRESHOLD


class Detector:
    def __init__(self, tag, device="cuda:0"):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        spec = DETECTORS[tag]
        self.tag, self.spec, self.device = tag, spec, torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
        self.model = AutoModelForObjectDetection.from_pretrained(
            spec["repo"], revision=spec["revision"], torch_dtype=torch.float32).to(self.device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def detect(self, image, threshold=THRESHOLD):
        """One PIL RGB image -> all official postprocessed detections >= threshold, original pixel xyxy."""
        sizes = torch.tensor([[image.height, image.width]], device=self.device)
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        r = self.processor.post_process_object_detection(out, threshold=threshold, target_sizes=sizes)[0]
        labels = [int(x) for x in r["labels"].detach().cpu()]
        return {"scores": [float(x) for x in r["scores"].detach().cpu()], "labels": labels,
                "label_strs": [self.id2label.get(l, "?") for l in labels],
                "boxes_xyxy": [[float(v) for v in b] for b in r["boxes"].detach().cpu()]}


def detected(rec, label_id):
    return any(l == label_id for l in rec["labels"])


def best_score(rec, label_id):
    s = [sc for sc, l in zip(rec["scores"], rec["labels"]) if l == label_id]
    return max(s) if s else None


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def box_match(rec, label_id, gt_boxes, thr=IOU_MATCH):
    """Correct-class match at IoU >= thr. Returns (matched, best_iou, index_of_best_detection)."""
    best, bi = 0.0, None
    for i, (bx, l) in enumerate(zip(rec["boxes_xyxy"], rec["labels"])):
        if l != label_id:
            continue
        for g in gt_boxes:
            v = iou(bx, g)
            if v > best:
                best, bi = v, i
    return best >= thr, best, bi


def output_flags(path, expected_size=(512, 512)):
    """Invalid/collapse flags from the saved PNG: unreadable, wrong size, nonfinite, or max per-channel
    spatial std <= 1/255 on normalized RGB. Flagged outputs can never count as S or R."""
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
        im = im.convert("RGB")
    except Exception as e:                                   # noqa: BLE001
        return {"valid": False, "flags": ["unreadable"], "detail": repr(e)[:120]}, None
    flags = []
    if im.size != tuple(expected_size):
        flags.append("wrong_size")
    a = np.asarray(im, dtype=np.float32) / 255.0
    if not np.isfinite(a).all():
        flags.append("nonfinite")
    std = float(a.reshape(-1, 3).std(axis=0).max()) if a.size else 0.0
    if std <= COLLAPSE_STD_THRESHOLD:
        flags.append("collapse")
    return {"valid": not flags, "flags": flags, "max_channel_std": std, "size": list(im.size)}, im
