"""D1 — quantify the inconsistent-label-text defect instead of leaving it unresolved.

THE DEFECT. `gen_eval.py` defines two parallel sequences: `LABELS`, the text intended to be given to
CLIP, and `LAB_KEY`, the underscored keys used in records. **Nine of the twelve differ** --
'garbage truck' vs 'garbage_truck', 'van gogh painting' vs 'van_gogh', 'english springer spaniel'
vs 'english_springer', 'monet painting' vs 'monet', and five underscore-only cases.
`gen_eval_final.py` passed the KEY strings to CLIP, so some reported numbers were scored against
different label text than others. FINDING65's banner records the effect as **unresolved**, and
FINDING66 was withdrawn as not re-verifiable.

WHY IT MATTERS EVEN THOUGH 'church' IS IDENTICAL IN BOTH. Suppression and recovery are a 12-way
ARGMAX. The target string is unchanged, but the eleven competing strings are not, so an image can
cross between 'church' and a competitor purely because the competitor's text changed. The target
label being identical bounds the problem; it does not remove it.

WHAT THIS DOES. Re-scores the SAVED images from existing evidence directories under both label
sets, with everything else held fixed, and reports: argmax agreement, and the church accuracy each
set produces on the same images. No model is loaded and nothing is regenerated -- this reads the
pixels that produced the published numbers.
"""
import argparse, glob, json, os
import torch

ROOT = "/data/bijaypandey/t2i_unlearn"
import importlib.util as _iu


def _mod(name, path):
    sp = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(sp); sp.loader.exec_module(m); return m


_ge = _mod("ge", os.path.join(ROOT, "scripts/gen_eval.py"))
LABELS, LAB_KEY = list(_ge.LABELS), list(_ge.LAB_KEY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/ev_U33_*/per_image.json")
    ap.add_argument("--max-dirs", type=int, default=40)
    ap.add_argument("--split", default="H", help="which eval split to score (H is the recovery metric)")
    ap.add_argument("--out", default="outputs/d1_label_text_impact.json")
    a = ap.parse_args()
    dev = "cuda"

    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                   torch_dtype=torch.float16).to(dev).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def textfeat(labels):
        with torch.no_grad():
            ti = cp(text=[f"a photo of a {l}" for l in labels], return_tensors="pt",
                    padding=True).to(dev)
            f = cm.get_text_features(**ti)
            return f / f.norm(dim=-1, keepdim=True)

    tf_correct, tf_keys = textfeat(LABELS), textfeat(LAB_KEY)

    files = sorted(glob.glob(os.path.join(ROOT, a.glob)))[:a.max_dirs]
    n_img = agree = 0
    hit_correct = hit_keys = 0
    flips = []
    for pf in files:
        try:
            rows = json.load(open(pf))["rows"]
        except Exception:
            continue
        base = os.path.dirname(pf)
        for r in rows:
            if r.get("split") != a.split:
                continue
            ip = os.path.join(base, r["image"])
            if not os.path.exists(ip):
                continue
            try:
                im = Image.open(ip).convert("RGB")
            except Exception:
                continue
            with torch.no_grad():
                px = cp(images=im, return_tensors="pt").to(dev)
                f = cm.get_image_features(pixel_values=px["pixel_values"].half())
                f = f / f.norm(dim=-1, keepdim=True)
                i_c = int((f @ tf_correct.T).argmax())
                i_k = int((f @ tf_keys.T).argmax())
            n_img += 1
            tgt = r.get("scored_for")
            agree += (i_c == i_k)
            hit_correct += (LAB_KEY[i_c] == tgt)
            hit_keys += (LAB_KEY[i_k] == tgt)
            if i_c != i_k:
                flips.append({"image": ip.replace(ROOT + "/", ""), "target": tgt,
                              "with_LABELS": LAB_KEY[i_c], "with_LAB_KEY": LAB_KEY[i_k]})

    rec = {
        "study": "d1_label_text_impact",
        "n_images": n_img, "split": a.split, "n_dirs": len(files),
        "labels_correct": LABELS, "labels_as_keys": LAB_KEY,
        "argmax_agreement": (agree / n_img) if n_img else None,
        "target_accuracy_with_LABELS": (hit_correct / n_img) if n_img else None,
        "target_accuracy_with_LAB_KEY": (hit_keys / n_img) if n_img else None,
        "n_flips": len(flips), "flips_sample": flips[:40],
        "note": ("scored on the SAME saved pixels, everything else held fixed; the only difference "
                 "is the text given to CLIP"),
    }
    with open(os.path.join(ROOT, a.out), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[d1] {n_img} images from {len(files)} dirs, split={a.split}")
    print(f"[d1] argmax agreement      : {rec['argmax_agreement']:.4f}")
    print(f"[d1] target acc w/ LABELS  : {rec['target_accuracy_with_LABELS']:.4f}")
    print(f"[d1] target acc w/ LAB_KEY : {rec['target_accuracy_with_LAB_KEY']:.4f}")
    print(f"[d1] flips                 : {len(flips)}")
    print(f"[d1] -> {a.out}")


if __name__ == "__main__":
    main()
