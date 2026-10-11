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

    A digest occurrence is graded, not assumed: within two lines of a
    checkpoint/arm/slot identifier it is an IDENTIFYING ASSOCIATION and the
    packet is BROKEN for those rows; a bare mention elsewhere is UNRESOLVED and
    REQUIRES ASSESSMENT, because a digest alone proves nothing.

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
import re
from pathlib import Path

# A boolean cannot express "the input was not fit to audit", which is how this
# gate produced false passes: an empty image directory scored 0/0 matches and
# was reported as holding. Four states, and only one of them is a pass.
HOLDS, BROKEN, REJECTED, UNRESOLVED, UNTESTED = (
    "holds", "BROKEN", "REJECTED", "UNRESOLVED", "NOT MODELLED")
VERDICTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    VERDICTS.append((name, status, detail))
    print(f"  [{status:<11}] {name}\n                {detail}")


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
        record("identifier hashing", HOLDS,
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
    record("identifier hashing", BROKEN if hits else UNRESOLVED,
           f"{hits}/{len(ids)} blinded ids regenerated from published records "
           f"alone using the PUBLISHED salt. Each hit reveals that row's arm "
           f"and training seed, so the blind is gone for those rows."
           if hits else
           f"UNRESOLVED -- a salt IS published, but enumeration under the documented path "
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
        record("published RNG + row order", UNTESTED,
               "no sampling RNG seed is published anywhere this "
               "tool looks, so the replay has no starting point. UNTESTED, not "
               "passed.")
        return
    order = (meta.get("blinding") or {}).get("order", "")
    design = meta.get("approved_design")
    if not design:
        record("published RNG + row order", UNTESTED,
               f"not modelled for this packet: it declares no paired "
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
            record("published RNG + row order", BROKEN,
                   f"reconstruction '{name}' replays the sheet's category "
                   f"column EXACTLY ({len(cats)}/{len(cats)} positions). The "
                   f"same replay carries the arm and training seed of every "
                   f"row, so row position alone de-blinds the packet. A secret "
                   f"salt does not help: position, not the id, is the channel.")
            return
        agree = sum(1 for x, y in zip(replay, cats) if x == y)
        if agree > best_agree:
            best, best_agree = name, agree
    record("published RNG + row order", HOLDS,
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


def slot_pattern() -> "re.Pattern":
    """Strings that would turn a digest occurrence into an ARM attribution."""
    return re.compile(
        r"seed\d+_(?:MA|MAB|MAC|MAB_L2|MAC_L2|U_step\d+)\b"
        r"|\bMAB_L2\b|\bMAC_L2\b|\bU_step\d+\b"
        r"|\bM0\b|_m0_|sha256_m0|noop_reload"
        r'|"(?:arm|slot|checkpoint|training_seed)"')


def bind_inventory(images: Path, sheet: Path) -> tuple[list[Path], list[str], dict]:
    """Bind the delivered files to the published sheet, or refuse to audit.

    PATH POLICY, stated rather than implied. A sheet `image` value is relative
    to the PACKET ROOT, and ``--delivered_images`` must be the directory those
    values point into. Nested values are permitted, but every resolved path must
    lie INSIDE the supplied directory, and the on-disk set is enumerated
    RECURSIVELY under it. Absolute values and any `..` component are rejected
    outright.

    Containment is the fix for a real false pass: resolving against
    ``images.parent`` and checking only ``on_disk - referenced`` let an EMPTY
    SIBLING directory pass, because the sheet's files existed next door and the
    sibling contributed no extras. Set equality is therefore TWO-WAY, and every
    reference must be inside the directory actually supplied.

    Every referenced file is also DECODED, not merely read: a readable text file
    with a .png name is not an image, and the gate used to certify it. Read
    failures and decode failures are kept distinct.
    """
    rows = list(csv.DictReader(sheet.open(newline="")))
    problems: list[str] = []
    img_root = images.resolve()

    referenced: dict[Path, int] = {}
    escapes = unreadable = undecodable = outside = 0
    for r in rows:
        # Packets in this repository have used both column names.
        rel = ((r.get("image") or r.get("image_file") or "")).strip()
        if not rel:
            problems.append(f"a sheet row names no image column (looked for "
                            f"'image' and 'image_file'; saw {sorted(r)[:6]})")
            continue
        rp = Path(rel)
        if rp.is_absolute() or ".." in rp.parts:
            escapes += 1
            continue
        cand = (img_root.parent / rp).resolve()
        if img_root not in cand.parents:
            # The sheet points somewhere other than the supplied directory.
            outside += 1
            continue
        referenced[cand] = referenced.get(cand, 0) + 1
        if not cand.is_file():
            continue
        try:
            cand.read_bytes()
        except Exception:
            unreadable += 1
            continue
        try:
            from PIL import Image
            with Image.open(cand) as im:
                im.load()                      # force a full decode
        except Exception:
            undecodable += 1

    missing = [f for f in referenced if not f.is_file()]
    dupes = [f for f, n in referenced.items() if n > 1]
    on_disk = {f.resolve() for f in img_root.rglob("*") if f.is_file()}
    extra = on_disk - set(referenced)

    if not rows:
        problems.append("the published sheet has no rows")
    if escapes:
        problems.append(f"{escapes} sheet row(s) use an absolute path or '..'")
    if outside:
        problems.append(f"{outside} sheet row(s) resolve OUTSIDE the supplied "
                        f"directory {img_root}")
    if missing:
        problems.append(f"{len(missing)} sheet row(s) reference a file that is "
                        f"not present")
    if extra:
        problems.append(f"{len(extra)} file(s) under the supplied directory are "
                        f"not referenced by the sheet")
    if dupes:
        problems.append(f"{len(dupes)} file(s) are referenced by more than one "
                        f"sheet row")
    if unreadable:
        problems.append(f"{unreadable} referenced file(s) could not be read")
    if undecodable:
        problems.append(f"{undecodable} referenced file(s) are not decodable "
                        f"images (readable bytes are not enough)")

    present = [f for f in referenced if f.is_file()]
    stats = {"sheet_rows": len(rows), "referenced": len(referenced),
             "on_disk": len(on_disk), "present": len(present)}
    return present, problems, stats


def attack_delivered_image_bytes(images: Path | None, evidence: Path,
                                 sheet: Path) -> None:
    """Do the delivered files hash to a published, slot-identifying digest?

    Bound to the published sheet first (see bind_inventory): an inventory that
    does not match is REJECTED, never passed. Every verdict branch reports the
    same count block, so a branch cannot quietly contradict the numbers.
    """
    if images is None:
        record("delivered image bytes", UNTESTED,
               "no --delivered_images directory was given, so the actual "
               "packet files were not hashed. UNTESTED, not passed.")
        return
    if not images.is_dir():
        record("delivered image bytes", REJECTED,
               f"{images} is not a directory; there is no delivery to audit.")
        return

    present, problems, st = bind_inventory(images, sheet)
    if problems:
        record("delivered image bytes", REJECTED,
               f"the delivery does not bind to the published sheet "
               f"({st['sheet_rows']} rows, {st['on_disk']} files under "
               f"{images}): " + "; ".join(problems) +
               ". No verdict is issued: an unbound inventory cannot be "
               "certified, and scoring it would be a false pass.")
        return

    pub = published_image_digests(evidence)
    if not pub:
        record("delivered image bytes", REJECTED,
               f"no published image digests found under {evidence}; the "
               f"comparison corpus is empty, so a 0-match result would be "
               f"meaningless.")
        return

    digests: dict[str, int] = {}
    for f in present:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        digests[h] = digests.get(h, 0) + 1
    n_files, n_distinct = len(present), len(digests)
    matched = sum(digests[h] for h in digests if h in pub)
    unique = sum(digests[h] for h in digests if len(pub.get(h, ())) == 1)
    ambiguous = sum(digests[h] for h in digests if len(pub.get(h, ())) > 1)
    n_resolving = sum(1 for v in pub.values() if len(v) == 1)

    # ---- the committed text corpus, graded ---------------------------------
    pat = slot_pattern()
    scanned = occurrences = 0
    files_with_occurrence: set[str] = set()
    files_with_association: set[str] = set()
    skip = (".jpg", ".jpeg", ".png", ".pdf", ".svg", ".gz", ".tar", ".zip")
    for q in Path(".").rglob("*"):
        if ".git/" in str(q) or not q.is_file() or q.suffix.lower() in skip:
            continue
        try:
            text = q.read_text(errors="ignore")
        except Exception:
            continue
        scanned += 1
        lines = text.splitlines()
        found = False
        for i, line in enumerate(lines):
            hit = [h for h in digests if h in line]
            if not hit:
                continue
            found = True
            occurrences += len(hit)
            if pat.search("\n".join(lines[max(0, i - 2):i + 3])):
                files_with_association.add(str(q))
        if found:
            files_with_occurrence.add(str(q))

    # One count block, used verbatim by every branch below.
    counts = (f"COUNTS: {n_files} delivered files, {n_distinct} distinct "
              f"hashes; {matched} match a published digest ({unique} resolving "
              f"to one slot, {ambiguous} ambiguous across slots); corpus "
              f"{len(pub)} published digests of which {n_resolving} are "
              f"slot-resolving; {scanned} committed text files swept with "
              f"{occurrences} digest occurrence(s) in "
              f"{len(files_with_occurrence)} file(s)")

    if unique:
        record("delivered image bytes", BROKEN,
               f"{unique} delivered file(s) hash to a published digest that "
               f"resolves to a UNIQUE slot, so those rows' arm and training "
               f"seed are recoverable by hashing the packet alone. {counts}. "
               f"Aggregate counts only; no item-to-slot mapping is reported.")
        return
    if files_with_association:
        record("delivered image bytes", BROKEN,
               f"{occurrences} digest occurrence(s) sit WITHIN TWO LINES of a "
               f"checkpoint, arm, slot or training-seed identifier, in "
               f"{len(files_with_association)} committed file(s): an "
               f"identifying association, so those rows' condition is readable "
               f"from committed data. {counts}. Files: "
               f"{sorted(files_with_association)[:5]}")
        return
    if ambiguous:
        record("delivered image bytes", UNRESOLVED,
               f"{ambiguous} delivered file(s) hash to a published digest that "
               f"appears under MORE THAN ONE source slot, so the match is real "
               f"but does not by itself attribute an arm. Ambiguity is NOT "
               f"absence: this REQUIRES ASSESSMENT of whether any other "
               f"committed record narrows those slots to one. {counts}.")
        return
    if files_with_occurrence:
        record("delivered image bytes", UNRESOLVED,
               f"no published digest matched and no identifying association was "
               f"detected, but delivered digest(s) appear in "
               f"{len(files_with_occurrence)} committed text file(s). A bare "
               f"digest mention is NOT proven de-blinding, and this gate does "
               f"not claim it is -- it REQUIRES ASSESSMENT of whether those "
               f"files let a reader associate the digest with an arm. "
               f"{counts}. Files: {sorted(files_with_occurrence)[:5]}")
        return
    if n_distinct != n_files:
        record("delivered image bytes", UNRESOLVED,
               f"no published digest matched, but {n_files - n_distinct} "
               f"delivered file(s) share byte-identical content with another "
               f"item, so those items are linkable to each other. {counts}. "
               f"REQUIRES ASSESSMENT.")
        return
    record("delivered image bytes", HOLDS,
           f"no delivered file hashes to any published digest, no digest "
           f"occurs in the swept text corpus, and all hashes are distinct. "
           f"{counts}. The annotation bytes are separated from the indexed "
           f"originals. SCOPE: this closes the DIGEST channel against a packet "
           f"holder who also has the repository. It does NOT close pixel "
           f"comparison by a holder of the private source images -- the packet "
           f"is made of those images, and no re-encoding can prevent that.")


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
        Path(a.evidence), sheet)

    by = lambda st: [n for n, s_, _ in VERDICTS if s_ == st]
    rejected, broken = by(REJECTED), by(BROKEN)
    unresolved, untested = by(UNRESOLVED), by(UNTESTED)
    print()
    if rejected:
        print(f"AUDIT REJECTED -- the input is not fit to certify "
              f"({', '.join(rejected)}). No blinding claim is made either way.")
        return 3
    if broken:
        print(f"BLINDING BROKEN by {len(broken)} of {len(VERDICTS)} attacks: "
              f"{', '.join(broken)}")
        return 1
    if unresolved or untested:
        bits = []
        if unresolved:
            bits.append(f"{len(unresolved)} UNRESOLVED ({', '.join(unresolved)})")
        if untested:
            bits.append(f"{len(untested)} NOT MODELLED ({', '.join(untested)})")
        print(f"{len(by(HOLDS))} of {len(VERDICTS)} attacks repelled; "
              f"{'; '.join(bits)}. This is NOT a clean pass.")
        return 2
    print(f"BLINDING HOLDS against all {len(VERDICTS)} attacks "
          f"(published data only, CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
