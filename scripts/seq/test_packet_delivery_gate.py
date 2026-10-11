#!/usr/bin/env python3
"""Focused CPU regressions for the packet delivery gate.

Why this file exists
--------------------
`audit_packet_blinding.py` reported "BLINDING HOLDS against all 3 attacks" and
exited 0 when handed the real 180-row published sheet together with

  * an EMPTY image directory, and
  * a directory holding one unrelated PNG.

Both scored 0/0 or 0/1 digest matches and were read as passes. A gate that
certifies a delivery it never saw is worse than no gate. These are regressions
for exactly those two false passes, plus the inventory faults next to them, plus
controls that prove the digest attack still fires when it should.

Covered here
------------
  REJECTED   empty directory; one unrelated file; a missing referenced file;
             an unreferenced extra file; two rows referencing one file; an
             unreadable file; an empty comparison corpus
  BROKEN     copied source bytes (the channel that was live); a committed text
             file associating a delivered digest with an arm
  UNRESOLVED a bare digest mention with no identifying association; two items
             sharing byte-identical content
  holds      the same inventory re-encoded as fresh PNGs
  export     pixels preserved, bytes changed, EXIF dropped, and an existing
             destination overwritten rather than trusted

The real 180-image export is audited separately and its result is recorded in
`annotation_packet_paired180/blinding_audit.log`; this file uses synthetic
fixtures only. No key is opened, no assignment is printed, and no real packet
image is read.

    python scripts/seq/test_packet_delivery_gate.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "seq" / "audit_packet_blinding.py"
BUILDER = REPO / "scripts" / "seq" / "build_paired_annotation_packet.py"
TEST_MANIFEST = (REPO / "results" / "audit_v1" / "draft_manifests"
                 / "test_manifest_DRAFT.json")

PASS: list[str] = []
FAIL: list[str] = []
N_ITEMS = 4
SLOTS = ["seed17_MA", "seed17_MAB_L2", "seed17_U_step70", "seed29_MA"]


def check(desc: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(desc)
    print(f"  {'ok  ' if cond else 'FAIL'}  {desc}" + ("" if cond else f"  <- {detail}"))


def load_builder():
    spec = importlib.util.spec_from_file_location("bpap", BUILDER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tiny_jpeg(path: Path, seed: int) -> None:
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype="uint8")).save(
        path, format="JPEG", quality=92)


def fixture(tmp: Path) -> dict:
    """A synthetic repo root, source images, published archive and packet."""
    root = tmp / "repo"
    (root / "evidence").mkdir(parents=True)
    src_dir = tmp / "sources"
    src_dir.mkdir()
    sources = []
    for i in range(N_ITEMS):
        f = src_dir / f"src_{i}.jpg"
        tiny_jpeg(f, 1000 + i)
        sources.append(f)

    # One published archive: each source digest maps to exactly ONE slot, which
    # is what makes a digest match slot-resolving.
    for i, (f, slot) in enumerate(zip(sources, SLOTS)):
        d = tmp / "slots" / slot
        d.mkdir(parents=True, exist_ok=True)
        (d / "image_report.json").write_text(json.dumps({
            "checkpoint": slot, "complete": True,
            "images": [{"prompt_id": f"p{i}", "gen_seed": 1301,
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}],
        }))
    with tarfile.open(root / "evidence" / "synth_slots.tar.gz", "w:gz") as tf:
        for slot in SLOTS:
            tf.add(tmp / "slots" / slot / "image_report.json",
                   arcname=f"{slot}/image_report.json")

    packet = tmp / "packet"
    (packet / "images").mkdir(parents=True)
    rows = []
    for i in range(N_ITEMS):
        rows.append({"item_id": f"item_{i:012x}",
                     "image": f"images/item_{i:012x}.png",
                     "category_to_judge": ["cat", "dog", "bird"][i % 3],
                     "question": "?", "answer_yes_no_unsure": "", "notes": ""})
    with (packet / "label_sheet_EMPTY.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (packet / "packet_manifest.json").write_text(json.dumps({
        "approved_design": {"categories": ["cat", "dog", "bird"],
                            "arms_per_tuple": ["MA", "MAB_L2", "U"],
                            "training_seeds": [17, 29],
                            "tuples_per_training_seed_and_category": 10},
        "eval_root": "/synthetic/eval_test",
        "slots": {"17": {"MA": "seed17_MA"}},
        "sampling": {"rng_seed": 20261012},
        "blinding": {"blinded_id_rule": "sha256(<secret salt> | <image_path>)[:12]",
                     "salt_sha256": "0" * 64, "order_seed_sha256": "1" * 64,
                     "order": "separate RNG seeded from the secret"},
    }, indent=2))
    return {"root": root, "packet": packet, "sources": sources,
            "images": packet / "images", "rows": rows}


def fill(fx: dict, mode: str) -> None:
    """Populate the packet's image directory. mode: 'copy' or 'reencode'."""
    m = load_builder()
    for f in fx["images"].iterdir():
        f.unlink()
    for row, src in zip(fx["rows"], fx["sources"]):
        dst = fx["packet"] / row["image"]
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            m.export_image(src, dst)


def run(fx: dict, images: Path | None = None, cwd: Path | None = None,
        evidence: Path | None = None) -> tuple[int, str]:
    cmd = [sys.executable, str(GATE),
           "--packet_dir", str(fx["packet"]),
           "--manifest", str(TEST_MANIFEST),
           "--evidence", str(evidence or (fx["root"] / "evidence"))]
    if images is not None:
        cmd += ["--delivered_images", str(images)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(cwd or fx["root"]),
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return p.returncode, p.stdout + p.stderr


def status_of(out: str) -> str:
    for line in out.splitlines():
        if "delivered image bytes" in line and line.strip().startswith("["):
            return line.strip().split("]")[0].lstrip("[").strip()
    return "(no verdict line)"


def main() -> int:
    for f in (GATE, BUILDER, TEST_MANIFEST):
        if not f.is_file():
            print(f"missing {f}")
            return 2
    try:
        import PIL, numpy  # noqa: F401
    except Exception as e:
        print(f"PIL and numpy required: {e}")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="deliverygate_"))
    try:
        fx = fixture(tmp)

        print("\nexport_image: pixels preserved, bytes changed, metadata dropped")
        m = load_builder()
        src = fx["sources"][0]
        dst = tmp / "out.png"
        rec = m.export_image(src, dst)
        from PIL import Image
        import numpy as np
        with Image.open(src) as a:
            pa = np.array(a.convert("RGB"))
        with Image.open(dst) as b:
            pb = np.array(b.convert("RGB"))
            n_exif = len(b.getexif())
        check("decoded pixels are identical", np.array_equal(pa, pb))
        check("the byte stream changed",
              hashlib.sha256(src.read_bytes()).hexdigest()
              != hashlib.sha256(dst.read_bytes()).hexdigest())
        check("no EXIF survives", n_exif == 0)
        check("the export reports its own verification",
              rec["pixels_identical"] is True and rec["exported_format"] == "PNG")
        # An existing destination must be overwritten and re-verified, not trusted.
        dst.write_bytes(b"not an image at all")
        rec2 = m.export_image(src, dst)
        with Image.open(dst) as b2:
            pb2 = np.array(b2.convert("RGB"))
        check("an existing destination is overwritten, not trusted",
              rec2["pixels_identical"] and np.array_equal(pa, pb2))

        print("\nthe two demonstrated FALSE PASSES are now rejected")
        fill(fx, "reencode")
        empty = tmp / "empty"
        empty.mkdir()
        rc, out = run(fx, empty)
        check("an EMPTY image directory is REJECTED, not passed",
              status_of(out) == "REJECTED" and rc == 3,
              f"status={status_of(out)} rc={rc}")
        check("and the refusal says why",
              "does not bind to the published sheet" in out)
        one = tmp / "one"
        one.mkdir()
        m.export_image(fx["sources"][0], one / "unrelated.png")
        rc, out = run(fx, one)
        check("ONE UNRELATED file is REJECTED, not passed",
              status_of(out) == "REJECTED" and rc == 3,
              f"status={status_of(out)} rc={rc}")
        check("the extra unreferenced file is named as a fault",
              "not referenced by the sheet" in out)

        print("\nother inventory faults are rejected too")
        fill(fx, "reencode")
        gone = fx["packet"] / fx["rows"][0]["image"]
        keep = gone.read_bytes()
        gone.unlink()
        rc, out = run(fx, fx["images"])
        check("a MISSING referenced file is REJECTED",
              status_of(out) == "REJECTED" and "not present" in out, out[-200:])
        gone.write_bytes(keep)

        extra = fx["images"] / "stowaway.png"
        m.export_image(fx["sources"][1], extra)
        rc, out = run(fx, fx["images"])
        check("an EXTRA unreferenced file is REJECTED",
              status_of(out) == "REJECTED" and "not referenced" in out)
        extra.unlink()

        sheet = fx["packet"] / "label_sheet_EMPTY.csv"
        orig = sheet.read_text()
        rows2 = [dict(r) for r in csv.DictReader(sheet.open(newline=""))]
        rows2[1]["image"] = rows2[0]["image"]          # two rows, one file
        with sheet.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows2[0].keys()))
            w.writeheader()
            w.writerows(rows2)
        rc, out = run(fx, fx["images"])
        check("a DUPLICATE reference is REJECTED",
              status_of(out) == "REJECTED" and "more than one sheet row" in out,
              out[-200:])
        sheet.write_text(orig)

        fill(fx, "reencode")
        bad = fx["packet"] / fx["rows"][2]["image"]
        os.chmod(bad, 0o000)
        # Decide first whether mode 000 is actually unreadable for this user:
        # an owner or root often reads it anyway, and asserting "REJECTED or
        # else fine" would be a vacuous check.
        try:
            bad.read_bytes()
            enforced = False
        except Exception:
            enforced = True
        if enforced:
            rc, out = run(fx, fx["images"])
            check("an UNREADABLE referenced file is REJECTED",
                  status_of(out) == "REJECTED" and "could not be read" in out,
                  f"status={status_of(out)} rc={rc}")
        else:
            print("  skip  an UNREADABLE referenced file is REJECTED  "
                  "<- mode 000 is still readable by this user; not asserted")
        os.chmod(bad, 0o644)

        rc, out = run(fx, fx["images"], evidence=tmp / "no_such_evidence")
        check("an EMPTY comparison corpus is REJECTED, not scored as 0 matches",
              status_of(out) == "REJECTED" and "corpus is empty" in out,
              out[-200:])

        print("\ncontrols: the digest attack still fires when it should")
        fill(fx, "copy")                                # source bytes copied
        rc, out = run(fx, fx["images"])
        check("COPIED SOURCE bytes are BROKEN",
              status_of(out) == "BROKEN" and rc == 1,
              f"status={status_of(out)} rc={rc}")
        check("and the count is reported as resolving to a unique slot",
              "resolve to a UNIQUE slot" in out)
        check("no item-to-slot mapping is printed",
              "item_" not in out.split("delivered image bytes")[-1][:400]
              or "Aggregate counts only" in out)

        fill(fx, "reencode")                            # fresh PNGs
        rc, out = run(fx, fx["images"])
        check("FRESH PNGs from the same sources hold",
              status_of(out) == "holds", f"status={status_of(out)}")
        check("file and distinct-hash counts are both reported",
              f"{N_ITEMS} delivered files, {N_ITEMS} distinct hashes" in out,
              out[-300:])
        check("the scope limit is stated, not a universal claim",
              "does NOT close pixel comparison" in out)

        print("\nthe committed text corpus is operational in the verdict")
        dig = hashlib.sha256(
            (fx["packet"] / fx["rows"][0]["image"]).read_bytes()).hexdigest()
        leak = fx["root"] / "public_map.json"
        leak.write_text(json.dumps(
            {"records": [{"image_sha256": dig, "arm": "MAB_L2",
                          "slot": "seed17_MAB_L2"}]}, indent=2))
        rc, out = run(fx, fx["images"])
        check("a digest-to-ARM association in a committed file is BROKEN",
              status_of(out) == "BROKEN" and rc == 1,
              f"status={status_of(out)} rc={rc}")
        check("the association is named as the reason",
              "identifying association" in out)
        leak.unlink()

        bare = fx["root"] / "notes.txt"
        bare.write_text(f"some digest appears here: {dig}\n")
        rc, out = run(fx, fx["images"])
        check("a BARE digest mention is UNRESOLVED, not BROKEN and not a pass",
              status_of(out) == "UNRESOLVED" and rc == 2,
              f"status={status_of(out)} rc={rc}")
        check("it is reported as requiring assessment",
              "REQUIRES ASSESSMENT" in out.upper())
        check("and it is explicitly NOT called proven de-blinding",
              "NOT proven de-blinding" in out)
        bare.unlink()

        print("\nduplicate delivered content is flagged")
        fill(fx, "reencode")
        a_, b_ = (fx["packet"] / fx["rows"][0]["image"],
                  fx["packet"] / fx["rows"][1]["image"])
        b_.write_bytes(a_.read_bytes())
        rc, out = run(fx, fx["images"])
        check("two items with identical bytes are UNRESOLVED",
              status_of(out) == "UNRESOLVED"
              and "byte-identical content" in out,
              f"status={status_of(out)} rc={rc}")
        check("the distinct-hash count differs from the file count",
              f"{N_ITEMS} delivered files, {N_ITEMS - 1} distinct hashes" in out,
              out[-300:])
    finally:
        for p in Path(tmp).rglob("*"):
            try:
                os.chmod(p, 0o644)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL DELIVERY-GATE CHECKS PASSED (CPU only, synthetic fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
