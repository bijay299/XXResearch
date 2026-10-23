"""FINDING40 S6 retention policy: keep what a weights-only study needs, without keeping 1.72 GB
of UNet per arm.

H67 could not be answered because 12 arms' checkpoints were deleted for storage after their
recovery had been measured, leaving numbers with no weights to pair them against. This script makes
the cheap half of the fix executable:

  slice   save only the 160 transformer-block linears FRAG scores -- 242.2M params, 0.48 GB fp16
          per arm, vs 1.72 GB for the full UNet (~15 GB for all 31 arms, inside the 60 GiB budget)
  verify  confirm a slice reproduces the full checkpoint's scope weights exactly before anything
          is deleted -- a slice is only a safe substitute if it is bit-identical on its scope

It NEVER deletes anything. Deletion stays a human decision; this only makes the cheap artefact
exist first.

Run: python scripts/retain_scope_slice.py --all
     python scripts/retain_scope_slice.py --arm h64v2_frag_a0.3_church --verify
"""
import argparse, glob, hashlib, json, os
import torch

ADAPT = "/data/bijaypandey/t2i_unlearn/adapters"
OUT = "/data/bijaypandey/t2i_unlearn/outputs"
SLICE_DIR = "/data/bijaypandey/t2i_unlearn/scope_slices"


def scope_keys(sd):
    """The FRAG scope, matched on state_dict keys so no model needs loading."""
    return sorted(k for k in sd
                  if k.endswith(".weight") and ".transformer_blocks." in k
                  and (".attn" in k or ".ff.net." in k))


def sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def do_arm(arm, verify=False):
    if arm.startswith("_atk_"):
        # a running attack's transient working copy, deleted as soon as it is measured -- not an
        # arm, and racing it is how this script first crashed
        return {"arm": arm, "status": "skipped: transient attack working dir"}
    ck = f"{ADAPT}/{arm}/unet.pt"
    if not os.path.exists(ck):
        return {"arm": arm, "status": "no checkpoint"}
    try:
        sd = torch.load(ck, map_location="cpu")
    except (FileNotFoundError, EOFError, RuntimeError) as e:
        return {"arm": arm, "status": f"unreadable (vanished or partial): {type(e).__name__}"}
    sd = sd.get("unet", sd)
    keys = scope_keys(sd)
    if not keys:
        return {"arm": arm, "status": "no scope keys found"}
    sl = {k: sd[k].half().clone() for k in keys}
    n = sum(v.numel() for v in sl.values())
    os.makedirs(SLICE_DIR, exist_ok=True)
    p = f"{SLICE_DIR}/{arm}.scope.pt"
    torch.save({"arm": arm, "source": ck, "n_keys": len(keys), "n_params": n,
                "scope": "transformer_blocks linears (attn + ff.net), the FRAG scope",
                "weights": sl}, p)
    rec = {"arm": arm, "status": "saved", "path": p, "n_keys": len(keys), "n_params": n,
           "slice_bytes": os.path.getsize(p), "full_bytes": os.path.getsize(ck),
           "ratio": os.path.getsize(p) / max(os.path.getsize(ck), 1)}
    if verify:
        back = torch.load(p, map_location="cpu")["weights"]
        bad = [k for k in keys if sha(back[k].float()) != sha(sd[k].half().float())]
        rec["verify_bit_identical_on_scope"] = (len(bad) == 0)
        rec["n_scope_keys_checked"] = len(keys)
        rec["verify_mismatched_keys"] = bad[:5]
        rec["verify_note"] = ("slice is bit-identical to the full checkpoint on every scope key, "
                              "so it is a safe substitute FOR SCOPE-ONLY uses (FRAG, l2 over the "
                              "scope). It is NOT a substitute for generation or attack, which need "
                              "the whole UNet."
                              if not bad else "MISMATCH -- do not treat this slice as a substitute")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    arms = ([a.arm] if a.arm else
            sorted(n for n in (os.path.basename(os.path.dirname(p))
                               for p in glob.glob(f"{ADAPT}/*/unet.pt"))
                   if not n.startswith("_atk_")) if a.all else [])
    if not arms:
        ap.error("pass --arm NAME or --all")

    rows = [do_arm(x, a.verify) for x in arms]
    for r in rows:
        if r["status"] != "saved":
            print(f"[slice] {r['arm']:34s} {r['status']}")
            continue
        extra = ""
        if a.verify:
            extra = f" verified={r['verify_bit_identical_on_scope']}"
        print(f"[slice] {r['arm']:34s} {r['n_keys']:3d} keys {r['n_params']/1e6:6.1f}M "
              f"{r['slice_bytes']/1e9:.3f}GB vs {r['full_bytes']/1e9:.3f}GB "
              f"({100*r['ratio']:.0f}%){extra}")
    saved = [r for r in rows if r["status"] == "saved"]
    tot_s = sum(r["slice_bytes"] for r in saved)
    tot_f = sum(r["full_bytes"] for r in saved)
    print(f"\n[slice] {len(saved)} arms: slices {tot_s/1e9:.2f} GB vs full {tot_f/1e9:.2f} GB")
    json.dump({"study": "scope_slice_retention", "policy": "FINDING40 S6",
               "note": "this script never deletes; it makes the cheap artefact exist first",
               "arms": rows, "total_slice_bytes": tot_s, "total_full_bytes": tot_f},
              open(f"{OUT}/scope_slices.json", "w"), indent=1)
    print(f"[slice] wrote {OUT}/scope_slices.json")


if __name__ == "__main__":
    main()
