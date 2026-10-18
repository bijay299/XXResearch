#!/usr/bin/env python3
"""Back up the private material and verify the copy by hash.

Private material = the neutral-ID -> source/condition/pair mapping, the
randomization salt, the populated Task-B package, and the quarantined
screenshots that show a real image beside its neutral ID.

This is HOST-PERFORMED verification: this process, on this machine, hashed both
copies and compared them. It is not an independent attestation.

No secret and no association is printed. Only counts, paths and verdicts.
"""
import hashlib
import json
import os
import shutil
import stat
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
PRIV = os.path.join(ROOT, "private")
BACKUP = "/data/bijaypandey/cuig_pilot/_teapot_human_review_v1_private_backup"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tree(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for f in fns:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = (os.path.getsize(p), sha256_file(p))
    return out


def main():
    os.makedirs(BACKUP, exist_ok=True)
    os.chmod(BACKUP, 0o700)

    src = tree(PRIV)
    # mirror
    for rel in src:
        d = os.path.join(BACKUP, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(os.path.join(PRIV, rel), d)
        os.chmod(d, 0o600)
    for dp, dns, _ in os.walk(BACKUP):
        for dn in dns:
            os.chmod(os.path.join(dp, dn), 0o700)

    dst = tree(BACKUP)
    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    mismatch = sorted(k for k in set(src) & set(dst) if src[k] != dst[k])

    # permission audit: nothing group/world readable
    bad_perm = []
    for base in (PRIV, BACKUP):
        for dp, dns, fns in os.walk(base):
            for n in list(dns) + list(fns):
                p = os.path.join(dp, n)
                m = stat.S_IMODE(os.lstat(p).st_mode)
                if m & 0o077:
                    bad_perm.append((os.path.relpath(p, base), oct(m)))
        m = stat.S_IMODE(os.lstat(base).st_mode)
        if m & 0o077:
            bad_perm.append((base, oct(m)))

    ok = not missing and not extra and not mismatch and not bad_perm
    report = {
        "verification_type": "HOST-PERFORMED",
        "verification_note": (
            "This process, running on this host, hashed both the primary private "
            "directory and its backup and compared them. It is a local integrity "
            "check, not an independent or third-party attestation."),
        "primary_private_location": PRIV,
        "backup_location": BACKUP,
        "n_files": len(src),
        "total_bytes": sum(v[0] for v in src.values()),
        "all_files_present_in_backup": not missing and not extra,
        "all_hashes_match": not mismatch,
        "n_hash_mismatches": len(mismatch),
        "missing_from_backup": missing,
        "unexpected_in_backup": extra,
        "permissions_restricted": not bad_perm,
        "permission_problems": bad_perm,
        "contents_summary": {
            "mapping_files": sum(1 for k in src if k.endswith("id_mapping_private.json")),
            "salt_files": sum(1 for k in src if k.endswith("salt.bin")),
            "taskB_images": sum(1 for k in src if k.startswith("taskB_package/images/")),
            "taskB_viewer_files": sum(1 for k in src
                                      if k.startswith("taskB_package/")
                                      and not k.startswith("taskB_package/images/")),
            "quarantined_screenshots": sum(1 for k in src
                                           if k.startswith("screenshots_withreal/")),
        },
        "note": ("No secret, filename association or mapping value is reported "
                 "here -- only counts, locations and verdicts."),
        "verdict": "PRIVATE_BACKUP_VERIFIED" if ok else "PRIVATE_BACKUP_PROBLEM",
    }
    json.dump(report, open(os.path.join(ROOT, "verify", "private_backup.json"), "w"),
              indent=1)
    for k in ("verdict", "n_files", "total_bytes", "all_files_present_in_backup",
              "all_hashes_match", "permissions_restricted"):
        print(f"{k:34} {report[k]}")
    print("contents:", json.dumps(report["contents_summary"]))
    sys.exit(0 if ok else 6)


if __name__ == "__main__":
    main()
