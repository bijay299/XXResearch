"""Shared constants and helpers for cpr_direct_agreement_controls_v1."""
import hashlib
import json
import os
import time

PROJECT_ID = "cpr_direct_agreement_controls_v1"
ROOT = os.environ.get("CPRD2_ROOT", "/data/bijaypandey/cuig_pilot/cpr_direct_agreement_controls_v1")
REPO = "/data/bijaypandey/xxresearch_standalone"
ACCEPTED_HEAD = "30f95590b1f2b138512e663635eda0e67ec29faa"
CODE = os.path.join(ROOT, "code")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def h_ranked(*parts):
    """Frozen metadata-only ordering key: SHA256('|'.join(parts))."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def load(path):
    with open(path) as f:
        return json.load(f)
