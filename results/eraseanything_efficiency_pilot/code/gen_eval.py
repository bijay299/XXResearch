#!/usr/bin/env python3
"""Generate and score an evaluation panel for one endpoint (M0 or an adapter).

Generation settings are the author's inference settings (inference.py / generate_dataset.py):
FLUX.1-dev, 512x512, 28 steps, max_sequence_length=256, bf16. Seeds are the frozen per-case
seeds from frozen/cases.json and are reused identically across every arm.

ATTENTION: generation runs the AUTHOR-FAITHFUL explicit materialised-softmax path in every
dual-stream block (EA_STATE["explicit_attn"]=True, maps_only_last=False), matching the
author's released inference (their vendored processor hardcodes vis_attn=True). MEASURED:
the fused-SDPA path is NOT output-preserving here -- same seed/prompt gives mean|diff| 1.5
and max|diff| 135 of 255 with 64% of pixels differing, because bf16 roundoff amplifies over
28 denoising steps (work/smoke/attn_equivalence.json). Cost of fidelity is 6.16 vs 4.51
s/img. The SAME path is used for M0 and for every arm, so comparisons are internally valid.

SCORER (frozen before any edited output existed): CLIP ViT-L/14
(openai/clip-vit-large-patch14). For each case, cosine similarity between the image
embedding and the text embeddings of that case's frozen label set; prediction = argmax.
  - target panels     : "present" iff argmax == target artist
  - retention panel   : "success" iff argmax == intended_label
Unknowns: a case whose image is missing scores as a failure for retention and as
"present" for target panels (i.e. never in favour of the edited model). No case is dropped.

QUALITY (rule frozen here; numeric thresholds calibrated on M0 only, written to
frozen/quality_thresholds.json by calibrate_quality.py before any edited output is scored):
degenerate iff gray_std < t_std OR sat_frac > t_sat OR lap_var < t_lap.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

PILOT = pathlib.Path(__file__).resolve().parent.parent


def load_pipe(model_id, rev, device, lora=None):
    from diffusers import FluxPipeline
    from diffusers.models.attention_processor import EA_STATE
    EA_STATE["explicit_attn"] = True           # author-faithful inference path
    EA_STATE["maps_only_last"] = False
    pipe = FluxPipeline.from_pretrained(model_id, revision=rev, torch_dtype=torch.bfloat16)
    if lora:
        pipe.load_lora_weights(lora)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def img_metrics(path: pathlib.Path) -> dict:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    g = a.mean(axis=2)
    sat = ((a <= 1.0) | (a >= 254.0)).all(axis=2).mean()
    # 4-neighbour Laplacian
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return {"gray_std": float(g.std()), "sat_frac": float(sat), "lap_var": float(lap.var())}


class ClipScorer:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        mid = "openai/clip-vit-large-patch14"
        self.m = CLIPModel.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
        self.p = CLIPProcessor.from_pretrained(mid)
        self.device = device
        self._txt = {}

    def text_emb(self, labels: tuple[str, ...], template: str):
        key = (labels, template)
        if key not in self._txt:
            prompts = [template.format(label=l) for l in labels]
            with torch.no_grad():
                t = self.p(text=prompts, return_tensors="pt", padding=True, truncation=True)
                e = self.m.get_text_features(**{k: v.to(self.device) for k, v in t.items()})
                self._txt[key] = torch.nn.functional.normalize(e, dim=-1)
        return self._txt[key]

    def predict(self, path, labels, template):
        im = Image.open(path).convert("RGB")
        with torch.no_grad():
            i = self.p(images=im, return_tensors="pt")
            e = self.m.get_image_features(pixel_values=i["pixel_values"].to(self.device))
            e = torch.nn.functional.normalize(e, dim=-1)
            sims = (e @ self.text_emb(tuple(labels), template).T)[0]
        j = int(sims.argmax())
        return labels[j], {labels[k]: round(float(sims[k]), 5) for k in range(len(labels))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--endpoint", required=True, help="label, e.g. M0 or A_s0")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--panels", default="canonical,perturbed,retention")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--equiv-check", action="store_true",
                    help="also render case 0 with the explicit-softmax path and report max|diff|")
    a = ap.parse_args()

    cases_f = PILOT / "frozen" / "cases.json"
    cases = json.loads(cases_f.read_text())
    sel = set(a.panels.split(","))
    mine = [c for c in cases["concepts"][a.concept] if c["panel"] in sel]
    if a.limit:
        mine = mine[: a.limit]
    sc_cfg = cases["scorer"]
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    imgdir = out / "images"; imgdir.mkdir(exist_ok=True)

    rev = os.environ["FLUX_DEV_REVISION"]
    mid = "black-forest-labs/FLUX.1-dev"
    n_new = 0
    t0 = time.time()

    if not a.score_only:
        pipe = load_pipe(mid, rev, a.device, a.lora)
        for k, c in enumerate(mine):
            f = imgdir / (c["case_id"].replace("|", "__").replace(" ", "_") + ".png")
            if f.exists():
                continue
            g = torch.Generator(device="cpu").manual_seed(int(c["seed"]))
            im = pipe(prompt=c["prompt"], generator=g, height=512, width=512,
                      num_inference_steps=28, max_sequence_length=256,
                      guidance_scale=3.5).images[0]
            im.save(f); n_new += 1
            if (k + 1) % 20 == 0:
                print(f"  [{a.endpoint}|{a.concept}] {k+1}/{len(mine)} "
                      f"({(time.time()-t0)/(k+1):.2f}s/img)", flush=True)
        if a.equiv_check and mine:
            from diffusers.models.attention_processor import EA_STATE
            c = mine[0]
            g = torch.Generator(device="cpu").manual_seed(int(c["seed"]))
            EA_STATE["explicit_attn"] = False
            base = np.asarray(pipe(prompt=c["prompt"], generator=g, height=512, width=512,
                                   num_inference_steps=28, max_sequence_length=256,
                                   guidance_scale=3.5).images[0]).astype(np.int16)
            EA_STATE["explicit_attn"] = True; EA_STATE["maps_only_last"] = False
            g = torch.Generator(device="cpu").manual_seed(int(c["seed"]))
            alt = np.asarray(pipe(prompt=c["prompt"], generator=g, height=512, width=512,
                                  num_inference_steps=28, max_sequence_length=256,
                                  guidance_scale=3.5).images[0]).astype(np.int16)
            d = np.abs(base - alt)
            (out / "attn_equivalence.json").write_text(json.dumps({
                "case": c["case_id"], "max_abs_diff_uint8": int(d.max()),
                "mean_abs_diff_uint8": float(d.mean()),
                "frac_pixels_differing": float((d > 0).mean()),
                "note": "fused SDPA vs explicit materialised softmax, same seed/prompt; "
                        "agreement within tolerance, NOT bit identity"}, indent=1))
            n_new += 2
            print("equiv-check max|diff| (uint8):", int(d.max()))
        del pipe; torch.cuda.empty_cache()

    scorer = ClipScorer(a.device)
    rows = []
    for c in mine:
        f = imgdir / (c["case_id"].replace("|", "__").replace(" ", "_") + ".png")
        if c["label_set"] == "artist":
            labels, tmpl = sc_cfg["artist_label_set"], sc_cfg["artist_prompt_template"]
        else:
            labels, tmpl = sc_cfg["generic_label_set"], sc_cfg["generic_prompt_template"]
        r = {"case_id": c["case_id"], "panel": c["panel"], "subpanel": c.get("subpanel", ""),
             "prompt": c["prompt"], "seed": c["seed"], "target": c["target"],
             "intended_label": c["intended_label"], "endpoint": a.endpoint,
             "concept": a.concept}
        if not f.exists():
            # unknown handling, frozen: never in favour of the edited model
            r.update({"missing": True, "pred": None,
                      "target_present": c["panel"] in ("canonical", "perturbed"),
                      "retention_success": False, "degenerate": True})
        else:
            pred, sims = scorer.predict(f, labels, tmpl)
            m = img_metrics(f)
            r.update({"missing": False, "pred": pred, "sims": sims, **m,
                      "target_present": bool(pred == c["target"]),
                      "retention_success": bool(pred == c["intended_label"])})
        rows.append(r)

    (out / f"scores_{a.endpoint}_{a.concept.replace(' ','_')}.json").write_text(
        json.dumps({"endpoint": a.endpoint, "concept": a.concept,
                    "lora": a.lora, "n_cases": len(rows),
                    "images_generated_this_call": n_new,
                    "scorer": sc_cfg, "rows": rows}, indent=1))
    pan = {}
    for r in rows:
        p = r["panel"]
        pan.setdefault(p, []).append(r)
    summ = {}
    for p, rs in pan.items():
        if p in ("canonical", "perturbed"):
            summ[p] = {"n": len(rs), "target_present": sum(r["target_present"] for r in rs),
                       "rate": round(sum(r["target_present"] for r in rs) / len(rs), 4)}
        else:
            summ[p] = {"n": len(rs), "success": sum(r["retention_success"] for r in rs),
                       "rate": round(sum(r["retention_success"] for r in rs) / len(rs), 4)}
    print(json.dumps({"endpoint": a.endpoint, "concept": a.concept,
                      "images_new": n_new, "summary": summ}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
