#!/usr/bin/env python3
"""Freeze neutral IDs, order variants and the private mapping.

Produces:
  PRIVATE (never in Git/releases): salt, neutral-ID -> (case, condition, seed,
  source filename) mapping for Task A, and the Task-B pair mapping.
  PUBLIC: the list of neutral Task-A IDs and three fixed order variants, which
  are independent of the arm mapping and therefore safe to publish.

Secrets are never printed. Commitments are salted hashes of the ACTUAL
populated artifacts.
"""
import hashlib
import hmac
import json
import os
import secrets
import sys
from datetime import datetime, timezone

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
SRC = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1"
PRIV = os.path.join(ROOT, "private")

PACKAGE_A = "THR-V1-A"
PACKAGE_B = "THR-V1-B"
PROTOCOL_ID = "TEAPOT-HR-PROTO-1.0.0"
SCHEMA_ID = "THR-EXPORT-SCHEMA-1.0.0"


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def main():
    os.makedirs(PRIV, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone()

    pv = json.load(open(os.path.join(SRC, "manifests", "PAIR_VERIFICATION.json")))
    rows = pv["rows"]
    assert len(rows) == 20, len(rows)

    # ---------------- salt (private) ----------------
    salt_path = os.path.join(PRIV, "salt.bin")
    if os.path.exists(salt_path):
        salt = open(salt_path, "rb").read()
        reused = True
    else:
        salt = secrets.token_bytes(32)
        fd = os.open(salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, salt)
        os.close(fd)
        reused = False

    # ---------------- Task A: 40 items, neutral opaque IDs ----------------
    items = []
    for r in rows:
        for cond in ("m0", "edited"):
            items.append({"case_id": r["case_id"], "condition": cond,
                          "eval_seed": r["eval_seed"],
                          "class": r["class"], "category": r["category"],
                          "source_filename": r[cond]["filename"],
                          "source_sha256": r[cond]["file_sha256"]})

    # cryptographically random assignment of sequential neutral IDs:
    # ID order carries no information about case, condition, pairing or seed.
    order = list(range(len(items)))
    rng = secrets.SystemRandom()
    rng.shuffle(order)
    a_ids = [f"A-{i+1:03d}" for i in range(len(items))]
    mapping = {}
    for slot, item_idx in enumerate(order):
        mapping[a_ids[slot]] = items[item_idx]

    # sanity: no ID encodes condition/pairing/seed
    assert len({m["source_filename"] for m in mapping.values()}) == 40
    assert sum(1 for m in mapping.values() if m["condition"] == "m0") == 20

    # ---------------- three fixed order variants ----------------
    # Constraint: the two items of a pair are never adjacent and are separated
    # by at least MIN_SEP positions, which reduces accidental side-by-side
    # comparison. Ordering is independent of the arm mapping.
    MIN_SEP = 4
    pair_of = {}
    for aid, m in mapping.items():
        pair_of.setdefault(m["case_id"], []).append(aid)

    def good(seq):
        pos = {aid: i for i, aid in enumerate(seq)}
        return all(abs(pos[a] - pos[b]) >= MIN_SEP for a, b in pair_of.values())

    variants = {}
    for v in (1, 2, 3):
        for _ in range(20000):
            cand = a_ids[:]
            rng.shuffle(cand)
            if good(cand):
                variants[f"V{v}"] = cand
                break
        else:
            raise SystemExit(f"could not build order variant V{v}")
    assert len({tuple(x) for x in variants.values()}) == 3, "variants must differ"

    # ---------------- Task B: 20 pairs, neutral pair IDs ----------------
    b_order = list(range(20))
    rng.shuffle(b_order)
    b_map = {}
    for slot, idx in enumerate(b_order):
        r = rows[idx]
        b_map[f"B-{slot+1:02d}"] = {
            "case_id": r["case_id"], "eval_seed": r["eval_seed"],
            "class": r["class"], "category": r["category"],
            "reference_condition": "m0",
            "candidate_condition": "edited",
            "reference_filename": r["m0"]["filename"],
            "reference_sha256": r["m0"]["file_sha256"],
            "candidate_filename": r["edited"]["filename"],
            "candidate_sha256": r["edited"]["file_sha256"],
            "noise_sha256": r.get("noise_sha256"),
        }

    # ---------------- write private artifacts ----------------
    priv_doc = {
        "generated_iso": now.isoformat(),
        "package_a": PACKAGE_A, "package_b": PACKAGE_B,
        "protocol_id": PROTOCOL_ID, "schema_id": SCHEMA_ID,
        "warning": ("PRIVATE. Contains the neutral-ID -> source/condition/pair "
                    "associations. Must never enter Git, any release, or any "
                    "reviewer bundle."),
        "taskA_mapping": mapping,
        "taskB_mapping": b_map,
    }
    for name, obj, mode in (("id_mapping_private.json", priv_doc, 0o600),):
        p = os.path.join(PRIV, name)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        os.write(fd, json.dumps(obj, indent=1).encode())
        os.close(fd)

    # ---------------- public, arm-independent ordering material ----------------
    public = {
        "package_id": PACKAGE_A, "protocol_id": PROTOCOL_ID,
        "schema_id": SCHEMA_ID, "stage": "A",
        "frozen_iso": now.isoformat(),
        "n_items": len(a_ids),
        "item_ids": a_ids,
        "order_variants": variants,
        "min_pair_separation": MIN_SEP,
        "alias_shuffle": {
            "enabled": True,
            "rule": ("order = stable sort of item_ids by "
                     "sha256(alias + '|' + package_id + '|' + item_id); depends "
                     "only on the reviewer alias and the public item IDs, and is "
                     "independent of the arm mapping."),
        },
        "note": ("These IDs and orderings carry no condition, pairing or seed "
                 "information. The association is held privately."),
    }
    json.dump(public, open(os.path.join(ROOT, "protocol", "taskA_items.json"), "w"),
              indent=1)

    # ---------------- salted commitments to the ACTUAL artifacts ----------------
    def commit(label, blob):
        return {"label": label,
                "hmac_sha256": hmac.new(salt, blob, hashlib.sha256).hexdigest()}

    commitments = {
        "scheme": ("HMAC-SHA256(secret_salt, canonical_bytes). The salt is held "
                   "privately and is NOT published. These commit to the actual "
                   "populated artifacts, not to placeholders."),
        "package_a": PACKAGE_A, "package_b": PACKAGE_B,
        "protocol_id": PROTOCOL_ID, "schema_id": SCHEMA_ID,
        "committed_iso": now.isoformat(),
        "salt_bytes": len(salt),
        "commitments": [
            commit("taskA_mapping_40_items", canon(mapping)),
            commit("taskB_mapping_20_pairs", canon(b_map)),
            commit("order_variants_v1_v2_v3", canon(variants)),
            commit("taskA_item_ids", canon(a_ids)),
        ],
        "counts_committed": {"taskA_items": len(mapping), "taskB_pairs": len(b_map)},
        "limitation": ("A commitment proves the committed material existed and "
                       "has not changed since. It does NOT make the underlying "
                       "images or their provenance private -- those are already "
                       "public in the modern-object-assets-v1 release."),
    }
    json.dump(commitments, open(os.path.join(ROOT, "protocol", "COMMITMENTS.json"), "w"),
              indent=1)

    print(f"salt: {'reused existing' if reused else 'created'} ({len(salt)} bytes, not printed)")
    print(f"taskA items: {len(mapping)}  (m0={sum(1 for m in mapping.values() if m['condition']=='m0')},"
          f" edited={sum(1 for m in mapping.values() if m['condition']=='edited')})")
    print(f"taskB pairs: {len(b_map)}")
    print(f"order variants: {sorted(variants)} each len "
          f"{ {k: len(v) for k, v in variants.items()} }")
    print(f"min pair separation enforced: {MIN_SEP}")
    print("public ordering material -> protocol/taskA_items.json")
    print("commitments (salted, to actual artifacts) -> protocol/COMMITMENTS.json")
    print("private mapping -> private/id_mapping_private.json (0600)")


if __name__ == "__main__":
    main()
