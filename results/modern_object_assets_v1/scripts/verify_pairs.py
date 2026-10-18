#!/usr/bin/env python3
"""Verify paired-input correspondence and per-file technical integrity.

Checks, for all 20 cases:
  * both conditions produced a readable PNG at the original 512x512 resolution
  * the two conditions used byte-identical initial noise (same sha256), the same
    prompt and the same inference settings
  * each output is non-degenerate as a FILE (decodable, correct size/mode), and
    records flat/black/saturated outputs explicitly rather than dropping them

This is a technical-integrity pass only. It applies no automated judge, assigns
no scientific human label, and makes no erasure-efficacy claim. The brightness
statistics are reported so that black or saturated outputs are disclosed, not to
score anything.

Also writes a contact sheet. The contact sheet is composited from the already
decoded PNGs; it performs no new VAE decode and therefore adds nothing to the
decoded-image count.
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402

GEN = os.path.join(C.ROOT, "gen")


def load_index(cond):
    with open(os.path.join(GEN, cond, f"index_{cond}.json")) as fh:
        return json.load(fh)


def img_stats(path):
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        mode = im.mode
        g = im.convert("L")
        px = list(g.getdata())
    n = len(px)
    mean = sum(px) / n
    var = sum((p - mean) ** 2 for p in px) / n
    mn, mx = min(px), max(px)
    return {"size": [w, h], "mode": mode, "lum_mean": round(mean, 3),
            "lum_std": round(var ** 0.5, 3), "lum_min": mn, "lum_max": mx,
            "all_black": mx == 0, "all_white": mn == 255,
            "flat": (var ** 0.5) < 1.0}


def main():
    m0, ed = load_index("m0"), load_index("edited")
    cases = C.load_cases()["cases"]
    r_m0 = {r["case_id"]: r for r in m0["rows"]}
    r_ed = {r["case_id"]: r for r in ed["rows"]}

    rows, problems = [], []
    for c in cases:
        cid = c["case_id"]
        a, b = r_m0.get(cid), r_ed.get(cid)
        rec = {"case_id": cid, "prompt": c["prompt"], "eval_seed": c["eval_seed"],
               "class": c["class"], "category": c["category"]}

        if a is None or b is None:
            rec["pair_status"] = "MISSING_ROW"
            rec["missing"] = [n for n, v in (("m0", a), ("edited", b)) if v is None]
            problems.append(rec)
            rows.append(rec)
            continue

        pm0 = os.path.join(GEN, "m0", a.get("filename", ""))
        ped = os.path.join(GEN, "edited", b.get("filename", ""))
        have = os.path.isfile(pm0) and os.path.isfile(ped)

        rec["noise_sha256"] = a.get("noise_sha256")
        rec["noise_identical_across_conditions"] = (
            a.get("noise_sha256") == b.get("noise_sha256")
            and a.get("noise_sha256") is not None)
        rec["prompt_identical"] = a.get("prompt") == b.get("prompt") == c["prompt"]
        rec["seed_identical"] = a.get("eval_seed") == b.get("eval_seed") == c["eval_seed"]
        rec["m0"] = {"status": a.get("status"), "filename": a.get("filename"),
                     "file_sha256": a.get("file_sha256"), "bytes": a.get("bytes")}
        rec["edited"] = {"status": b.get("status"), "filename": b.get("filename"),
                         "file_sha256": b.get("file_sha256"), "bytes": b.get("bytes")}

        if not have:
            rec["pair_status"] = "MISSING_FILE"
            problems.append(rec)
        else:
            try:
                rec["m0"]["image"] = img_stats(pm0)
                rec["edited"]["image"] = img_stats(ped)
                res_ok = (rec["m0"]["image"]["size"] == [512, 512]
                          and rec["edited"]["image"]["size"] == [512, 512])
                rec["original_resolution_512"] = res_ok
                ok = (rec["noise_identical_across_conditions"]
                      and rec["prompt_identical"] and rec["seed_identical"]
                      and res_ok and a.get("status") == "ok" and b.get("status") == "ok")
                rec["pair_status"] = "COMPLETE" if ok else "INCOMPLETE"
                if not ok:
                    problems.append(rec)
                for cond in ("m0", "edited"):
                    im = rec[cond]["image"]
                    if im["all_black"] or im["all_white"] or im["flat"]:
                        rec.setdefault("degenerate_outputs", []).append(
                            {"condition": cond,
                             "all_black": im["all_black"],
                             "all_white": im["all_white"],
                             "flat": im["flat"]})
            except Exception as e:
                rec["pair_status"] = "CORRUPT_FILE"
                rec["error"] = f"{type(e).__name__}: {e}"
                problems.append(rec)
        rows.append(rec)

    complete = [r for r in rows if r.get("pair_status") == "COMPLETE"]
    degen = [r for r in rows if r.get("degenerate_outputs")]
    out = {
        "n_cases": len(rows),
        "n_pairs_complete": len(complete),
        "n_pairs_incomplete_or_missing": len(rows) - len(complete),
        "missing_rows": [r["case_id"] for r in rows
                         if r.get("pair_status") in ("MISSING_ROW", "MISSING_FILE",
                                                     "CORRUPT_FILE")],
        "all_noise_identical_across_conditions": all(
            r.get("noise_identical_across_conditions") for r in rows),
        "all_prompts_identical": all(r.get("prompt_identical") for r in rows),
        "all_seeds_identical": all(r.get("seed_identical") for r in rows),
        "all_original_resolution_512": all(r.get("original_resolution_512") for r in rows),
        "n_distinct_noise_hashes": len({r.get("noise_sha256") for r in rows}),
        "degenerate_outputs_preserved": [
            {"case_id": r["case_id"], "detail": r["degenerate_outputs"]} for r in degen],
        "m0_condition": {"checkpoint": m0["checkpoint"],
                         "overwritten_components": m0["overwritten_components"],
                         "n_ok": m0["n_ok"], "n_failed": m0["n_failed"]},
        "edited_condition": {"checkpoint": ed["checkpoint"],
                             "overwritten_components": ed["overwritten_components"],
                             "n_ok": ed["n_ok"], "n_failed": ed["n_failed"]},
        "inference_settings_match": m0["inference"] == ed["inference"],
        "gate": ("Technical integrity only. No detector-recognition, suppression, "
                 "retention or visual-quality threshold was applied; no automated "
                 "judge and no scientific human label was used. Brightness "
                 "statistics are disclosure, not scoring."),
        "rows": rows,
    }
    C.dump_json(out, os.path.join(C.ROOT, "manifests", "PAIR_VERIFICATION.json"))

    print(f"pairs complete              : {len(complete)}/{len(rows)}")
    print(f"noise identical across cond : {out['all_noise_identical_across_conditions']}")
    print(f"distinct noise hashes       : {out['n_distinct_noise_hashes']}")
    print(f"prompts/seeds identical     : {out['all_prompts_identical']} / {out['all_seeds_identical']}")
    print(f"all 512x512                 : {out['all_original_resolution_512']}")
    print(f"inference settings match    : {out['inference_settings_match']}")
    print(f"degenerate (preserved)      : {len(degen)}")
    print(f"missing rows                : {out['missing_rows']}")

    # ---- contact sheet (composited from existing PNGs; no new decode) ----
    cell, pad, hdr = 192, 6, 22
    cols, rowsn = 2, len(cases)
    per_block = 10
    blocks = (rowsn + per_block - 1) // per_block
    sheet_w = blocks * (cols * cell + (cols + 1) * pad)
    sheet_h = per_block * (cell + hdr + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 250))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(cases):
        blk, r = i // per_block, i % per_block
        x0 = blk * (cols * cell + (cols + 1) * pad) + pad
        y0 = pad + r * (cell + hdr + pad)
        d.text((x0, y0), f"{c['case_id']} s{c['eval_seed']} [{c['category']}]",
               fill=(20, 20, 20))
        for j, cond in enumerate(("m0", "edited")):
            fn = (r_m0 if cond == "m0" else r_ed).get(c["case_id"], {}).get("filename")
            x = x0 + j * (cell + pad)
            y = y0 + hdr
            p = os.path.join(GEN, cond, fn) if fn else None
            if p and os.path.isfile(p):
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y))
            else:
                d.rectangle([x, y, x + cell, y + cell], fill=(40, 40, 40))
                d.text((x + 8, y + cell // 2), "MISSING", fill=(255, 120, 120))
            d.text((x + 2, y + cell - 12), cond.upper(), fill=(255, 255, 0))
    cs = os.path.join(C.ROOT, "gen", "contact_sheet.png")
    sheet.save(cs)
    print("contact sheet ->", cs, sheet.size)

    sys.exit(0 if len(complete) == len(rows) else 7)


if __name__ == "__main__":
    main()
