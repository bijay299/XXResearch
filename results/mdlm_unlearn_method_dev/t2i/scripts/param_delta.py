#!/usr/bin/env python
"""Max/mean absolute parameter delta between two UNet state_dicts, plus sha256 of each.

Written for PREREG_S1_AMENDMENT_1_2026-10-27.md section 1.2, which predicts that the S1 arm
`s1_metaV2_z0.0_g0.0` reproduces its teacher EXACTLY (max abs delta 0.0) because both base loss
terms distil to the teacher the model is initialised from, giving identically zero gradient.

A non-zero delta falsifies that prediction and means the v2 build path is not deterministic, which
would have to be understood before arms 2 and 3 are interpreted. Prints, never writes.
"""
import argparse, hashlib, sys
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--expect", choices=["identical", "different"], default=None,
                    help="what was PREDICTED before running; omit to report without a verdict")
    a = ap.parse_args()

    print("A  %s\n   sha256 %s" % (a.a, sha256(a.a)))
    print("B  %s\n   sha256 %s" % (a.b, sha256(a.b)))

    A, B = load(a.a), load(a.b)
    ka, kb = set(A), set(B)
    if ka != kb:
        print("KEY MISMATCH: only in A %d, only in B %d" % (len(ka - kb), len(kb - ka)))
        for k in sorted(ka - kb)[:5]:
            print("   A-only:", k)
        for k in sorted(kb - ka)[:5]:
            print("   B-only:", k)
        return 2

    gmax = 0.0
    gsum = 0.0
    gn = 0
    worst = None
    for k in sorted(ka):
        x, y = A[k], B[k]
        if x.shape != y.shape:
            print("SHAPE MISMATCH %s %s vs %s" % (k, tuple(x.shape), tuple(y.shape)))
            return 2
        d = (x.float() - y.float()).abs()
        m = float(d.max()) if d.numel() else 0.0
        gsum += float(d.sum())
        gn += d.numel()
        if m > gmax:
            gmax, worst = m, k

    print("\ntensors compared     %d" % len(ka))
    print("parameters compared  %d" % gn)
    print("max  abs delta       %.12g%s" % (gmax, ("   (in %s)" % worst) if worst else ""))
    print("mean abs delta       %.12g" % (gsum / gn if gn else 0.0))
    # The verdict depends on what was PREDICTED, which this script cannot know. Pass --expect to
    # say so; otherwise just report. An earlier version hardcoded "investigate" for any non-zero
    # delta, which fired a false alarm on an arm that was SUPPOSED to differ.
    if a.expect == "identical":
        print("\nVERDICT: %s" % ("IDENTICAL (prediction held)" if gmax == 0.0 else
                                 "DIFFERENT but identity was predicted -- INVESTIGATE"))
    elif a.expect == "different":
        print("\nVERDICT: %s" % ("DIFFERENT (prediction held)" if gmax > 0.0 else
                                 "IDENTICAL but a difference was predicted -- INVESTIGATE"))
    else:
        print("\nVERDICT: reported only; no prediction supplied (--expect identical|different)")

    if gmax > 0.0:
        # Scale reference for an Adam run: coherent motion would be ~steps*lr per parameter,
        # a pure random walk ~sqrt(steps)*lr. Printing both lets the reader judge cancellation.
        print("  scale note: pass --steps/--lr to compare against coherent vs random-walk drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
