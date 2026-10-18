#!/usr/bin/env python3
"""FROZEN binary target-presence / retention / rendering scorer (a declared ADAPTATION,
not part of the BLOCK paper's own metrics).

Why it exists: the native LLaVA evaluator is a 20-way FORCED CHOICE. It must emit some
class for every image, so "not classified as the target" cannot establish that the target
is absent. This procedure asks the question directly and admits "unsure".

Everything below is frozen before any sprint image is scored, and is validated against
independently annotated COCO images (code/validate_scorer.py) before use.

FROZEN SPECIFICATION
  judge      : LLaVA-1.6-Vicuna-7B, local, revision pinned in RUN_MANIFEST.md
  conv_mode  : llava_v1 (pinned explicitly, not auto-detected)
  decoding   : do_sample=False, temperature=0, top_p=1.0, num_beams=1, max_new_tokens=8
               (deterministic -> a single pass is reproducible; no retries, no re-asks,
                no tie-breaking, no second opinion)
  fields     : four independent questions per image, each parsed by the SAME parser:
                 presence_target   -> is the erased object visible?
                 presence_intended -> is the case's intended object visible?
                 render_failure    -> is the image a severe rendering failure?
               plus `uncertain` derived from the parser outcome of each.
  parser     : lowercase, strip punctuation, then
                 standalone "yes"                      -> yes
                 standalone "no"                       -> no
                 unsure / unclear / cannot / not sure  -> uncertain
                 anything else (incl. empty)           -> uncertain
               "uncertain" counts as an ERROR in scorer validation and is handled
               conservatively in every gate (see REPORT).
  raw        : every raw string is preserved verbatim in the output JSONL.
"""
import os, re, sys, json, argparse, hashlib

LLAVA_REPO = os.environ.get(
    "LLAVA_REPO",
    "/data/bijaypandey/cuig_pilot/block_object_sprint/code/block-concept-removal/src/LLaVA")
sys.path.insert(0, LLAVA_REPO)

import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, \
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

# ---------------- frozen question templates ----------------
Q_PRESENCE = ("Is there a {noun} visible anywhere in this image?\n"
              "Answer with exactly one word: yes, no, or unsure.\n"
              'Answer "unsure" only if you genuinely cannot tell.')

Q_RENDER = ("Is this image a severe rendering failure?\n"
            "Answer yes ONLY if the image is essentially unusable: blank, near-uniform "
            "colour, pure noise or static, a garbled texture with no depicted scene, or "
            "almost entirely corrupted.\n"
            "A coherent scene is NOT a failure even if it is unusual, heavily stylised, "
            "empty of any particular object, or contains distorted faces or anatomy.\n"
            "Answer with exactly one word: yes, no, or unsure.")

# frozen class -> singular noun used in the presence question
NOUN = {"Bears": "bear", "Cats": "cat", "Horses": "horse",
        "Dogs": "dog", "Rabbits": "rabbit",
        "Architectures": "building", "Flowers": "flower"}

SPEC_VERSION = "presence-v1-frozen"


def parse(raw: str) -> str:
    t = (raw or "").strip().lower()
    t = re.sub(r"[^a-z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = t.split()
    if not toks:
        return "uncertain"
    if "yes" in toks:
        return "yes"
    if "no" in toks:
        return "no"
    if any(p in t for p in ("unsure", "unclear", "cannot", "can t", "not sure", "uncertain")):
        return "uncertain"
    return "uncertain"


class Judge:
    def __init__(self, model_path, conv_mode="llava_v1", max_new_tokens=8):
        disable_torch_init()
        self.name = get_model_name_from_path(model_path)
        self.tok, self.model, self.ip, _ = load_pretrained_model(model_path, None, self.name)
        self.conv_mode = conv_mode
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def ask(self, image_path, question):
        pil = Image.open(image_path).convert("RGB")
        qs = question
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        imgs = process_images([pil], self.ip, self.model.config).to(
            self.model.device, dtype=self.model.dtype)
        ids = tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to(self.model.device)
        out = self.model.generate(inputs=ids, images=imgs, image_sizes=[pil.size],
                                  do_sample=False, temperature=0, top_p=1.0, num_beams=1,
                                  max_new_tokens=self.max_new_tokens, use_cache=True)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON list of {image, target_noun, intended_noun?, id, ...}")
    ap.add_argument("--out", required=True, help="output .jsonl")
    ap.add_argument("--model-path", default=os.environ.get("LLAVA_PATH"))
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--fields", nargs="+",
                    default=["presence_target", "presence_intended", "render_failure"])
    a = ap.parse_args()

    items = json.load(open(a.manifest))
    j = Judge(a.model_path, a.conv_mode)
    print(f"judge={j.name} conv_mode={a.conv_mode} spec={SPEC_VERSION} n={len(items)}", flush=True)

    with open(a.out, "w") as f:
        for i, it in enumerate(items):
            rec = {"spec_version": SPEC_VERSION, "judge": j.name,
                   "conv_mode": a.conv_mode, **it}
            if "presence_target" in a.fields and it.get("target_noun"):
                raw = j.ask(it["image"], Q_PRESENCE.format(noun=it["target_noun"]))
                rec["presence_target_raw"] = raw
                rec["presence_target"] = parse(raw)
            if "presence_intended" in a.fields and it.get("intended_noun"):
                raw = j.ask(it["image"], Q_PRESENCE.format(noun=it["intended_noun"]))
                rec["presence_intended_raw"] = raw
                rec["presence_intended"] = parse(raw)
            if "render_failure" in a.fields:
                raw = j.ask(it["image"], Q_RENDER)
                rec["render_failure_raw"] = raw
                rec["render_failure"] = parse(raw)
            f.write(json.dumps(rec) + "\n")
            if i % 25 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)
    print(f"wrote {a.out}  sha256={sha(a.out)}")


if __name__ == "__main__":
    main()
