#!/usr/bin/env python3
"""Attack a blinded annotation packet using ONLY published data. CPU only.

A packet is blinded if someone holding the committed, key-free records cannot
work out which arm produced which row. Asserting that is worthless; this script
tries to break it, two ways, and reports what it managed.

ATTACK 1 -- identifier hashing.
    Blinded ids are sha256(salt | image_path)[:12] and image paths are
    DETERMINISTIC: <eval_root>/seed<S>_<arm>/images/<prompt_id>_seed<gs>.jpg.
    The eval root, the slot names and the frozen prompt manifest are all
    published, so the candidate path space is a few hundred per cell. If the
    salt is published too, every id is enumerable and the blind is gone.

ATTACK 2 -- published RNG state plus row order.
    If one RNG both DRAWS the sample and SHUFFLES the rows, and its seed is
    published, the whole ordered item list -- arm included -- replays from
    published data. Matching that replay against the sheet's visible
    `category_to_judge` column confirms the reconstruction without ever opening
    the key: a 180-long category sequence does not agree by chance.

ATTACK 3 -- the delivered image bytes.
    The committed image reports and detector rows associate every image's
    sha256 with its source slot. If the packet ships source bytes unchanged, a
    holder can hash a renamed file and read off the arm -- no key, no salt, no
    order needed. This attack hashes the ACTUAL delivered files and looks each
    digest up in every published digest the repository carries.

The key is never read. The salt is never printed. A reconstruction is reported
as a count and a verdict, never as a mapping -- in particular attack 3 reports
only aggregate counts, never which item resolved to which slot.

    python scripts/seq/audit_packet_blinding.py \
        --packet_dir results/audit_v1/annotation_packet_paired180 \
        --delivered_images <data root>/annotation_paired_180/packet/images
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

VERDICTS: list[tuple[str, bool, str]] = []


def record(name: str, broken: bool, detail: str) -> None:
    VERDICTS.append((name, broken, detail))
    tag = "BROKEN" if broken else "holds "
    print(f"  [{tag}] {name}\n           {detail}")


def sheet_categories(sheet: Path) -> tuple[list[str], list[str]]:
    ids, cats = [], []
    with sheet.open(newline="") as fh:
        for row in csv.DictReader(fh):
            ids.append(row["item_id"])
            cats.append(row["category_to_judge"])
    return ids, cats


def attack_identifier_hashing(meta: dict, ids: list[str], man: dict) -> None:
    """Can published data regenerate the blinded ids?"""
    b = meta.get("blinding") or {}
    rule = b.get("blinded_id_rule", "")
    published_salt = None
    # A literal salt inside the rule string is one failure mode.
    if "'" in rule and "|<image_path>" in rule:
        published_salt = rule.split("'")[1].split("|")[0]
    # A plainly published salt field is the other. Every spelling this
    # repository has used is checked, so an older packet is not let off.
    for k in ("salt", "blinding_salt"):
        if b.get(k):
            published_salt = b[k]
        if meta.get(k):
            published_salt = meta[k]
    if published_salt is None:
        record("identifier hashing", False,
               f"no salt value appears in the published records "
               f"(rule: {rule!r}; salt_sha256 published: "
               f"{'yes' if b.get('salt_sha256') else 'no'}). Enumeration needs "
               f"the secret, which lives with the key.")
        return

    eval_root = meta.get("eval_root", "")
    raw = meta.get("slots") or []
    # Published `slots` has appeared as a list of slot names and as
    # {seed: {arm: slot}}; both are flattened here.
    slot_names: list[str] = []
    if isinstance(raw, dict):
        for armmap in raw.values():
            slot_names.extend(armmap.values() if isinstance(armmap, dict)
                              else [armmap])
    else:
        slot_names = [str(x) for x in raw]
    hits = 0
    want = set(ids)
    for slot in slot_names:
        for r in man["records"]:
            path = (f"{eval_root}/{slot}/images/"
                    f"{r['prompt_id']}_seed{r['gen_seed']}.jpg")
            for pref, n in (("item_", 12), ("I", 10)):
                h = pref + hashlib.sha256(
                    f"{published_salt}|{path}".encode()).hexdigest()[:n]
                if h in want:
                    hits += 1
    record("identifier hashing", hits > 0,
           f"{hits}/{len(ids)} blinded ids regenerated from published records "
           f"alone using the PUBLISHED salt. Each hit reveals that row's arm "
           f"and training seed, so the blind is gone for those rows."
           if hits else
           f"a salt IS published, but enumeration under the documented path "
           f"rule regenerated 0/{len(ids)} ids: this packet's id rule or its "
           f"inputs differ from sha256(salt | <eval_root>/<slot>/images/"
           f"<prompt_id>_seed<gs>.jpg)[:12]. Publishing the salt is still a "
           f"weakness -- it means only the rule, not a secret, stands between "
           f"a reader and the mapping -- but this tool did not break it.")


def attack_rng_and_row_order(meta: dict, cats: list[str], man: dict) -> None:
    """Can the published sampling seed replay the row order, arms included?"""
    samp = meta.get("sampling") or {}
    seed = samp.get("rng_seed", meta.get("rng_seed"))
    if seed is None:
        record("published RNG + row order", False,
               "NOT MODELLED: no sampling RNG seed is published anywhere this "
               "tool looks, so the replay has no starting point. UNTESTED, not "
               "passed.")
        return
    order = (meta.get("blinding") or {}).get("order", "")
    design = meta.get("approved_design")
    if not design:
        record("published RNG + row order", False,
               f"NOT MODELLED for this packet: it declares no paired "
               f"`approved_design`, and its draw consumes detector output "
               f"(stratified cells and score-enriched items), so replaying it "
               f"needs the raw detector rows rather than published metadata "
               f"alone. Published sampling seed: {seed}. Treat this as "
               f"UNTESTED, not as passed.")
        return
    categories = tuple(design.get("categories") or ("cat", "dog", "bird"))
    arms = tuple(design.get("arms_per_tuple") or ("MA", "MAB_L2", "U"))
    seeds = design.get("training_seeds") or [17, 29]
    per_cell = design.get("tuples_per_training_seed_and_category") or 10

    by_cat: dict[str, list[tuple[str, int]]] = {}
    for r in man["records"]:
        by_cat.setdefault(r["category"], []).append(
            (r["prompt_id"], int(r["gen_seed"])))

    def draw(r: random.Random) -> list[tuple]:
        out: list[tuple] = []
        for sd in seeds:
            for cat in categories:
                pool = sorted(by_cat.get(cat, []))
                if len(pool) < per_cell:
                    continue
                for k in sorted(r.sample(pool, per_cell)):
                    for arm in arms:
                        out.append((sd, arm, cat, k[0], k[1]))
        return out

    # Several plausible implementations, not only the one that was used: an
    # attacker does not have to guess right the first time.
    candidates: dict[str, list[tuple]] = {}

    r1 = random.Random(seed)                        # one RNG: sample + shuffle
    one = draw(r1)
    r1.shuffle(one)
    candidates["one RNG draws and shuffles"] = one

    two = draw(random.Random(seed))                 # fresh RNG for the shuffle
    random.Random(seed).shuffle(two)
    candidates["fresh RNG(sampling seed) shuffles"] = two

    candidates["no shuffle (deterministic order)"] = draw(random.Random(seed))

    best, best_agree = None, -1
    for name, items in candidates.items():
        replay = [it[2] for it in items]
        if len(replay) == len(cats) and replay == cats:
            record("published RNG + row order", True,
                   f"reconstruction '{name}' replays the sheet's category "
                   f"column EXACTLY ({len(cats)}/{len(cats)} positions). The "
                   f"same replay carries the arm and training seed of every "
                   f"row, so row position alone de-blinds the packet. A secret "
                   f"salt does not help: position, not the id, is the channel.")
            return
        agree = sum(1 for x, y in zip(replay, cats) if x == y)
        if agree > best_agree:
            best, best_agree = name, agree
    record("published RNG + row order", False,
           f"none of {len(candidates)} reconstructions from published data "
           f"reproduces the row order (best: '{best}', {best_agree}/{len(cats)} "
           f"positions; chance agreement is about "
           f"{len(cats) // max(1, len(categories))}). Order note: {order!r}")


def published_image_digests(evidence: Path) -> dict[str, set[str]]:
    """Every image sha256 the repository publishes, mapped to its slot(s).

    Read from the committed evidence archives: each slot's image_report.json
    (generation-side digests) and detections.jsonl (`image_sha256`).
    """
    import tarfile
    out: dict[str, set[str]] = {}
    for arch in sorted(evidence.rglob("*_slots.tar.gz")):
        with tarfile.open(arch, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(("image_report.json", "detections.jsonl")):
                    continue
                slot = m.name.split("/")[0]
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                raw = fh.read().decode(errors="ignore")
                if m.name.endswith("image_report.json"):
                    for im in json.loads(raw).get("images") or []:
                        if im.get("sha256"):
                            out.setdefault(im["sha256"], set()).add(slot)
                else:
                    for ln in raw.splitlines():
                        if not ln.strip():
                            continue
                        o = json.loads(ln)
                        if o.get("image_sha256"):
                            out.setdefault(o["image_sha256"], set()).add(slot)
    return out


def attack_delivered_image_bytes(images: Path | None, evidence: Path) -> None:
    """Do the delivered files hash to a published, slot-identifying digest?"""
    if images is None:
        record("delivered image bytes", False,
               "NOT MODELLED: no --delivered_images directory was given, so the "
               "actual packet files were not hashed. UNTESTED, not passed.")
        return
    if not images.is_dir():
        record("delivered image bytes", False,
               f"NOT MODELLED: {images} is not a directory. UNTESTED.")
        return
    pub = published_image_digests(evidence)
    if not pub:
        record("delivered image bytes", False,
               f"NOT MODELLED: no published image digests found under "
               f"{evidence}. UNTESTED.")
        return
    files = sorted(p for p in images.iterdir() if p.is_file())
    delivered = {}
    for f in files:
        delivered[hashlib.sha256(f.read_bytes()).hexdigest()] = 1
    matched = sum(1 for h in delivered if h in pub)
    unique = sum(1 for h in delivered if len(pub.get(h, ())) == 1)

    # Second corpus: the archives are not the only place a digest could be
    # published. Sweep every committed text file too, so "holds" is scoped to a
    # stated corpus rather than to one directory.
    scanned, elsewhere = 0, 0
    skip = (".jpg", ".jpeg", ".png", ".pdf", ".svg", ".gz", ".tar", ".zip")
    for q in Path(".").rglob("*"):
        if ".git/" in str(q) or not q.is_file() or q.suffix.lower() in skip:
            continue
        try:
            t = q.read_text(errors="ignore")
        except Exception:
            continue
        scanned += 1
        elsewhere += sum(1 for h in delivered if h in t)

    record("delivered image bytes", unique > 0,
           (f"{matched}/{len(files)} delivered files hash to a PUBLISHED image "
            f"digest and {unique} of those resolve to a UNIQUE slot, so the arm "
            f"and training seed of {unique} rows are recoverable by hashing the "
            f"packet alone. {len(pub)} slot-resolving published digests were "
            f"searched, plus {scanned} other committed text files "
            f"({elsewhere} further occurrences). Aggregate counts only; no "
            f"item-to-slot mapping is reported.")
           if unique else
           (f"0/{len(files)} delivered files hash to any of the {len(pub)} "
            f"slot-resolving published digests, and {elsewhere} occurrence(s) "
            f"across {scanned} other committed text files. The annotation bytes "
            f"are separated from the indexed originals. SCOPE: this closes the "
            f"DIGEST channel against a packet holder who also has the "
            f"repository. It does NOT close pixel comparison by a holder of the "
            f"private source images -- the packet is made of those images, and "
            f"no re-encoding can prevent that."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet_dir", required=True,
                    help="the COMMITTED key-free packet records")
    ap.add_argument("--meta", default="packet_manifest.json")
    ap.add_argument("--sheet", default=None,
                    help="default: label_sheet_EMPTY.csv, else label_sheet.csv")
    ap.add_argument("--manifest",
                    default="results/audit_v1/draft_manifests/test_manifest_DRAFT.json")
    ap.add_argument("--delivered_images", default=None,
                    help="the packet's ACTUAL images directory under the "
                         "private data root; without it attack 3 is reported "
                         "as NOT MODELLED rather than passed")
    ap.add_argument("--evidence", default="results/audit_v1",
                    help="where the committed *_slots.tar.gz archives live")
    a = ap.parse_args()

    d = Path(a.packet_dir)
    meta = json.loads((d / a.meta).read_text())
    sheet = Path(a.sheet) if a.sheet else next(
        p for p in (d / "label_sheet_EMPTY.csv", d / "label_sheet.csv")
        if p.is_file())
    man = json.loads(Path(a.manifest).read_text())
    ids, cats = sheet_categories(sheet)

    print(f"packet   : {d}")
    print(f"sheet    : {sheet.name}  ({len(ids)} rows)")
    print(f"manifest : {a.manifest}")
    print("the KEY is never opened and the salt is never printed\n")

    attack_identifier_hashing(meta, ids, man)
    attack_rng_and_row_order(meta, cats, man)
    attack_delivered_image_bytes(
        Path(a.delivered_images) if a.delivered_images else None,
        Path(a.evidence))

    broken = [n for n, b, _ in VERDICTS if b]
    untested = [n for n, b, d in VERDICTS if not b and "NOT MODELLED" in d]
    print()
    if broken:
        print(f"BLINDING BROKEN by {len(broken)} of {len(VERDICTS)} attacks: "
              f"{', '.join(broken)}")
        return 1
    if untested:
        print(f"{len(VERDICTS) - len(untested)} of {len(VERDICTS)} attacks "
              f"repelled; {len(untested)} NOT MODELLED for this packet "
              f"({', '.join(untested)}). This is not a clean pass.")
        return 2
    print(f"BLINDING HOLDS against all {len(VERDICTS)} attacks "
          f"(published data only, CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
