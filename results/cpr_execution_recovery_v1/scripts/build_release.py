#!/usr/bin/env python3
"""Build release assets (deterministic tars) + RELEASE_ASSETS.json + SHA256SUMS + checkpoint hash table. CPU only."""
import glob, hashlib, io, json, os, sys, tarfile
ROOT = os.environ["CPRX_ROOT"]
REL = os.path.join(ROOT, "release"); os.makedirs(REL, exist_ok=True)
ADIR = os.path.join(ROOT, "artifacts", "adapters")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_tar(name, entries, gz=False):
    path = os.path.join(REL, name)
    with tarfile.open(path, "w:gz" if gz else "w") as tf:
        for arc, src in sorted(entries):
            ti = tf.gettarinfo(src, arcname=arc); ti.uid = ti.gid = 0; ti.uname = ti.gname = ""; ti.mtime = 0
            with open(src, "rb") as f:
                tf.addfile(ti, f)
    return path


def add_dir(entries, d, arc_prefix, pattern="*", skip=()):
    for r, _, fs in os.walk(d):
        for f in fs:
            if f in skip:
                continue
            p = os.path.join(r, f)
            entries.append((arc_prefix + "/" + os.path.relpath(p, d), p))


lock = json.load(open(os.path.join(ROOT, "configs", "selection_lock.json")))
cases = json.load(open(os.path.join(ROOT, "configs", "cases.json")))
assets = []
# 1. adapters: all dev-selected + all test endpoints (selected checkpoints); every checkpoint hashed in the table
ad = []
hashes = {}
for d in sorted(glob.glob(os.path.join(ADIR, "*", "*"))):
    if not os.path.exists(os.path.join(d, "adapter_meta.json")):
        continue
    rel = os.path.relpath(d, ADIR)
    m = json.load(open(os.path.join(d, "adapter_meta.json")))
    hashes[rel] = {"tensor_state_sha256": m["tensor_state_sha256"], "file_sha256": m["file_sha256"], "n_tensors": m["n_tensors"]}
for t in cases["heldout_trajectories"]:
    p = os.path.join(ADIR, t["traj_id"], "selected")
    if os.path.isdir(p):
        add_dir(ad, p, f"adapters/{t['traj_id']}/selected")
for rel in hashes:
    traj = rel.split(os.sep)[0]
    if traj.startswith("dev."):
        add_dir(ad, os.path.join(ADIR, rel), f"adapters/{rel}")      # all dev checkpoints (66/132/250) published too (small)
json.dump(hashes, open(os.path.join(ROOT, "artifacts", "analysis", "checkpoint_hashes.json"), "w"), indent=1, sort_keys=True)
assets.append(make_tar("adapters_native_peft.tar", ad))
# 2. images
for name, dirs in (("images_heldout_primary.tar", ["images/m0/ho", "images/ho"]), ("images_dev_primary.tar", ["images/m0/dev", "images/dev"])):
    e = []
    for d in dirs:
        add_dir(e, os.path.join(ROOT, d), d)
    assets.append(make_tar(name, e))
# 3. raw detector predictions
e = []
add_dir(e, os.path.join(ROOT, "artifacts", "scores"), "scores")
assets.append(make_tar("detector_predictions_raw.tar.gz", e, gz=True))
# 4. visual audit package + judgments + revealed key (only if present)
va = os.path.join(ROOT, "visual_audit")
if os.path.isdir(os.path.join(va, "release")):
    e = []
    add_dir(e, os.path.join(va, "release"), "visual_audit")
    assets.append(make_tar("visual_audit_package_and_judgments.tar", e))
# 5. NOTICE and license
for f in ("NOTICE_FLUX1_DEV.txt", "LICENSE-FLUX1-dev.txt"):
    src = os.path.join(ROOT, "records", f if f.startswith("NOTICE") else os.path.join("licenses", f))
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(REL, f)); assets.append(os.path.join(REL, f))
rows = [{"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha(p)} for p in assets]
for r in rows:
    assert r["bytes"] < (2 << 30) - 1, f"asset too large {r}"
open(os.path.join(REL, "SHA256SUMS"), "w").write("".join(f"{r['sha256']}  {r['name']}\n" for r in rows))
rows.append({"name": "SHA256SUMS", "bytes": os.path.getsize(os.path.join(REL, "SHA256SUMS")), "sha256": None})
json.dump({"assets": rows}, open(os.path.join(ROOT, "RELEASE_ASSETS.json"), "w"), indent=1)
print(json.dumps(rows, indent=1))
