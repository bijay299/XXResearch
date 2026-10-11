#!/usr/bin/env python3
"""Enumerate every measurement the proposed screen needs, and cost it. CPU only.

One transparent inventory. It enumerates unique (checkpoint, set, concept) slots
-- selected checkpoints, calibration candidates and intermediate dumps counted
SEPARATELY -- converts them to images and device-hours at the throughput
MEASURED in Amendment 01's ledger, and keeps estimates visibly separate from
measurements. It also prints the contrast coefficient vectors and a precision
sensitivity table.

Nothing here is a result and nothing is authorised by it. It reads two committed
files and runs no model.

    python scripts/seq/screen_measurement_inventory.py
"""
from __future__ import annotations

import json
import statistics

LEDGER = "results/audit_v1/amendment01_evidence/gpu_usage_record.json"
ANALYSIS = "results/audit_v1/amendment01_evidence/analysis_test.json"

# ---------------------------------------------------------------- the design --
# Two edit/training seeds for EVERY arm: SPEED's editor is stochastic
# (train_erase_null.py:25 randn_like retain augmentation, :48 k-means K2, :85
# randperm retain shuffle, :140 --aug_num default 10, :131/:147 --seed 0 +
# seed_everything), so it is not exempt from replication.
SEEDS = 2
ANCHORS = 2          # the shared anchor and the disjoint anchor
TARGETS_R2 = 2       # dog (near), sandwich (far)

CA_ARMS = ["ConAbl", "SPEED", "SPEED+AIC(reproduced)", "CA-LACU(ablation)"]
NAT_ARMS = ["NAT-LACU", "NAT-SPEED", "NAT-SPEED+AIC(reproduced)"]
# ConAbl has no native replacement-selection mechanism, so its CA cells ARE its
# native configuration: it contributes no distinct NAT arm (a genuine identity,
# the only sharing in the design).

# Nine measured concepts. "flower" is NOT among them: the committed detector
# compares torchvision COCO_V1 category names by exact equality (detect.py:117-121)
# and that list has no "flower" (91 names, "potted plant" at index 64).
TARGETS = ["cat", "dog", "sandwich"]
RETAINED = ["bird", "chair", "bicycle"]
CONTROL = ["clock"]                      # separate non-target control
ANCHOR_CONCEPTS = ["horse", "potted plant"]
CONCEPTS_TEST = TARGETS + RETAINED + CONTROL + ANCHOR_CONCEPTS   # 9

# Scene layout. The bootstrap unit is the scene cluster, so the SCENE count is
# the precision knob; images per scene are 4 generation seeds x literal/paraphrase.
SCENES = {"DEV": 10, "ACCEPT": 20, "TEST": 20, "R": 20}
IMAGES_PER_SCENE = 8

# Calibration candidates per selected checkpoint, and whether each dial costs a
# fresh run or only a dump from one run.
DIAL = {
    "ConAbl":                     {"candidates": 10, "mode": "dump selection (one run, dumps every 100 steps)"},
    "SPEED":                      {"candidates": 4,  "mode": "separate closed-form edit per retain_scale value"},
    "SPEED+AIC(reproduced)":      {"candidates": 4,  "mode": "separate closed-form edit per retain_scale value"},
    "CA-LACU(ablation)":          {"candidates": 3,  "mode": "dump selection (one run; native saves at 100, 200, final)"},
    "NAT-LACU":                   {"candidates": 3,  "mode": "dump selection (one run; native saves at 100, 200, final)"},
    "NAT-SPEED":                  {"candidates": 4,  "mode": "separate closed-form edit per retain_scale value"},
    "NAT-SPEED+AIC(reproduced)":  {"candidates": 4,  "mode": "separate closed-form edit per retain_scale value"},
}

# Per-request training/editing cost. Measured where marked.
CONABL_STEPS = 1000
SPEED_EDIT_SECONDS = 60.0      # ESTIMATE, conservative against "seconds-level" upstream claims
LACU_HOURS = (1.5, 4.0)        # ESTIMATE, pending the smoke calibration
LACU_MAPPING_HOURS = 0.15      # ESTIMATE: 100 prompts x 10 candidates x MAP_K=32 UNet passes
ANCHOR_CACHE_IMAGES = 200      # one new cache for the measurable disjoint anchor


def measured_rates() -> dict:
    led = json.load(open(LEDGER))["entries"]
    test = [e["wall_seconds"] for e in led if e["stage"].startswith("eval:eval_test:")]
    train = [e["wall_seconds"] for e in led if e["stage"].startswith("train:")]
    return {"s_per_image": statistics.mean(test) / 560,
            "conabl_s_per_step": max(train) / CONABL_STEPS}


def checkpoints() -> dict:
    sel = {}
    for a in CA_ARMS:
        sel[a] = {"R1": ANCHORS * SEEDS,
                  "R2": ANCHORS * ANCHORS * TARGETS_R2 * SEEDS}
    for a in NAT_ARMS:
        sel[a] = {"R1": SEEDS, "R2": TARGETS_R2 * SEEDS}
    return sel


def inventory() -> dict:
    sel = checkpoints()
    r1 = sum(v["R1"] for v in sel.values())
    r2 = sum(v["R2"] for v in sel.values())
    cand = sum((v["R1"] + v["R2"]) * DIAL[a]["candidates"] for a, v in sel.items())

    ipc = {k: SCENES[k] * IMAGES_PER_SCENE for k in SCENES}
    slots = []   # (what, n_checkpoints, set, n_concepts, images)

    def add(what, n_ckpt, st, n_con):
        slots.append({"what": what, "checkpoints": n_ckpt, "set": st,
                      "concepts": n_con, "images": n_ckpt * n_con * ipc[st]})

    # DEV: dial selection only, the stage's own target only.
    add("calibration candidates", cand, "DEV", 1)
    # ACCEPT: own-target residue for every selected checkpoint, plus the NEXT
    # stage's parent rates (dog AND sandwich) on every request-1 checkpoint.
    add("selected checkpoints", r1 + r2, "ACCEPT", 1)
    add("request-1 parents (p_2 for dog and sandwich)", r1, "ACCEPT", 2)
    add("M0 (p_1 for cat)", 1, "ACCEPT", 1)
    # TEST: all nine concepts, once, on every selected checkpoint, plus M0 for
    # the cumulative baselines and PriorDelta.
    add("selected checkpoints", r1 + r2, "TEST", len(CONCEPTS_TEST))
    add("M0 cumulative baselines", 1, "TEST", len(CONCEPTS_TEST))
    # R_cat: the relapse probe. Request-1 gives the reference D_R(C_1,cat);
    # request-2 gives D_R(C_k,cat).
    add("selected checkpoints", r1 + r2, "R", 1)
    add("M0 (reported sanity baseline)", 1, "R", 1)

    rates = measured_rates()
    images = sum(s["images"] for s in slots)
    eval_h = images * rates["s_per_image"] / 3600

    # --- training / editing -------------------------------------------------
    conabl_reqs = sel["ConAbl"]["R1"] + sel["ConAbl"]["R2"]
    conabl_h = conabl_reqs * CONABL_STEPS * rates["conabl_s_per_step"] / 3600
    anchor_h = ANCHOR_CACHE_IMAGES * rates["s_per_image"] / 3600
    speed_edits = sum((v["R1"] + v["R2"]) * DIAL[a]["candidates"]
                      for a, v in sel.items() if "SPEED" in a)
    speed_h = speed_edits * SPEED_EDIT_SECONDS / 3600
    lacu_reqs = sum(v["R1"] + v["R2"] for a, v in sel.items() if "LACU" in a)
    lacu_lo = lacu_reqs * (LACU_HOURS[0] + LACU_MAPPING_HOURS)
    lacu_hi = lacu_reqs * (LACU_HOURS[1] + LACU_MAPPING_HOURS)

    return {"selected": sel, "n_r1": r1, "n_r2": r2, "n_selected": r1 + r2,
            "n_candidates": cand, "slots": slots, "images": images,
            "rates": rates, "eval_h": eval_h,
            "train": {"ConAbl": conabl_h, "anchor_cache": anchor_h,
                      "SPEED_edits": speed_h, "n_speed_edits": speed_edits,
                      "LACU_lo": lacu_lo, "LACU_hi": lacu_hi,
                      "n_lacu_requests": lacu_reqs},
            "total_lo": eval_h + conabl_h + anchor_h + speed_h + lacu_lo,
            "total_hi": eval_h + conabl_h + anchor_h + speed_h + lacu_hi}


# ------------------------------------------------------------ the contrasts --
# Eight counterbalanced CA cells per arm per edit seed, indexed (A1, A2, target).
# The disjoint anchor is "potted plant", a COCO_V1 category (index 64), NOT
# "flower", which the committed detector cannot measure at all (see CONCEPTS_TEST).
DISJOINT_ANCHOR = "potted plant"
CELLS = [("horse", "horse", "dog"), ("horse", DISJOINT_ANCHOR, "dog"),
         (DISJOINT_ANCHOR, DISJOINT_ANCHOR, "dog"), (DISJOINT_ANCHOR, "horse", "dog"),
         ("horse", "horse", "sandwich"), ("horse", DISJOINT_ANCHOR, "sandwich"),
         (DISJOINT_ANCHOR, DISJOINT_ANCHOR, "sandwich"), (DISJOINT_ANCHOR, "horse", "sandwich")]
SHARED = [a1 == a2 for a1, a2, _ in CELLS]
NEAR = [t == "dog" for _, _, t in CELLS]
A2_HORSE = [a2 == "horse" for _, a2, _ in CELLS]
REF_IS_HORSE = [a1 == "horse" for a1, _, _ in CELLS]   # which request-1 parent


def vec(f) -> list[float]:
    return [f(i) for i in range(8)]


def contrasts() -> dict:
    q = 0.25
    S = vec(lambda i: +q if SHARED[i] else -q)
    P = vec(lambda i: +q if NEAR[i] else -q)
    S_near = vec(lambda i: (0.5 if SHARED[i] else -0.5) if NEAR[i] else 0.0)
    S_far = vec(lambda i: (0.5 if SHARED[i] else -0.5) if not NEAR[i] else 0.0)
    P_shared = vec(lambda i: (0.5 if NEAR[i] else -0.5) if SHARED[i] else 0.0)
    P_disjoint = vec(lambda i: (0.5 if NEAR[i] else -0.5) if not SHARED[i] else 0.0)
    I = [a - b for a, b in zip(S_near, S_far)]
    A2 = vec(lambda i: +q if A2_HORSE[i] else -q)
    return {"S": S, "P": P, "I": I, "A2_identity": A2,
            "S|near": S_near, "S|far": S_far,
            "P|shared": P_shared, "P|disjoint": P_disjoint,
            "S-P": [a - b for a, b in zip(S, P)],
            "P-S": [b - a for a, b in zip(S, P)]}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def precision(h_cell: float) -> dict:
    """Half-width of each contrast, given a per-cell half-width h_cell.

    For a coefficient vector a with sum(a)=0 and an exchangeable cell
    correlation rho, Var = sigma^2 (1-rho) sum(a^2): the shared component
    cancels. rho is UNKNOWN -- every cell relapse contrast is measured on the
    same R_cat scenes, so it is positive, but by how much is not estimated here.
    """
    c = contrasts()
    out = {}
    for rho in (-0.5, 0.0, 0.5, 0.8):
        row = {}
        for name in ("S", "P", "I", "S|near", "S-P"):
            ss = sum(x * x for x in c[name])
            row[name] = h_cell * (ss * (1 - rho)) ** 0.5
        out[rho] = row
    return out


# ------------------------------------------------- the smoke calibration ----
SMOKE_LADDER_CONFIGS = 3      # (B=2,accum=18), (4,9), (6,6) per process, 3 processes
SMOKE_LADDER_STEPS = 12       # 2 discarded as warm-up, 10 measured, per configuration
SMOKE_RELOAD_STEPS = 100      # one configuration run on to the native step-100 save
EFFECTIVE_BATCH = 108         # 3 processes x per-process batch x accumulation
OUR_PASSES_PER_IMAGE = 60     # 30 sampler steps x 2 for classifier-free guidance
BATCHED_SPEEDUP = (1.0, 4.0)  # ESTIMATE band: 1x = our measured batch-1 rate


def passes_per_optimizer_step() -> int:
    """Replay + unlearn latents: effective_batch x 2 generations x 20 DDIM steps.

    Invariant to how the effective batch is split into micro-batches, which is
    why reducing per-device batch for 40 GB cards costs headroom, not compute.
    """
    return EFFECTIVE_BATCH * 2 * 20


def smoke_cost() -> dict:
    steps = SMOKE_LADDER_CONFIGS * SMOKE_LADDER_STEPS + SMOKE_RELOAD_STEPS
    passes = steps * passes_per_optimizer_step()
    img_eq = passes / OUR_PASSES_PER_IMAGE
    s_per_image = measured_rates()["s_per_image"]
    hi = img_eq * s_per_image / 3600 / BATCHED_SPEEDUP[0]
    lo = img_eq * s_per_image / 3600 / BATCHED_SPEEDUP[1]
    return {"optimizer_steps": steps, "unet_passes": passes,
            "image_equivalents": img_eq, "device_hours": (lo, hi)}


def main() -> int:
    inv = inventory()
    r = inv["rates"]
    print("=" * 78)
    print("SCREEN MEASUREMENT INVENTORY -- CPU only, authorises nothing")
    print("=" * 78)
    print(f"\nMEASURED throughput (Amendment 01 ledger, {LEDGER}):")
    print(f"  generation + COCO detection : {r['s_per_image']:.4f} s/image "
          f"(mean of the 6 frozen-test slots / 560 images)")
    print(f"  ConAbl training             : {r['conabl_s_per_step']:.4f} s/step "
          f"(842 s / 1000 steps)")

    print("\nSELECTED CHECKPOINTS (M0 and calibration candidates excluded):")
    for a, v in inv["selected"].items():
        print(f"  {a:30s} R1={v['R1']:3d}  R2={v['R2']:3d}  total={v['R1']+v['R2']:3d}"
              f"   dial: {DIAL[a]['mode']}")
    print(f"  {'':30s} {'':8s} {'':8s} SELECTED = {inv['n_selected']}"
          f"  (R1={inv['n_r1']}, R2={inv['n_r2']})")
    print(f"  calibration candidates / intermediate dumps, counted separately = {inv['n_candidates']}")
    print(f"  plus M0 = 1, not a selected checkpoint")

    print(f"\nMEASUREMENT SLOTS  (scene layout: "
          + ", ".join(f"{k}={v}x{IMAGES_PER_SCENE}" for k, v in SCENES.items()) + ")")
    print(f"  {'what':44s} {'set':7s} {'ckpts':>6s} {'conc':>5s} {'images':>8s}")
    for s in inv["slots"]:
        print(f"  {s['what']:44s} {s['set']:7s} {s['checkpoints']:6d} "
              f"{s['concepts']:5d} {s['images']:8,d}")
    print(f"  {'TOTAL IMAGES':44s} {'':7s} {'':6s} {'':5s} {inv['images']:8,d}")
    print(f"\n  evaluation device-hours (measured rate)        = {inv['eval_h']:.1f}")

    t = inv["train"]
    print("\nTRAINING / EDITING device-hours:")
    print(f"  ConAbl, {CONABL_STEPS} steps x 20 requests  MEASURED  = {t['ConAbl']:.2f}")
    print(f"  one new anchor cache ({ANCHOR_CACHE_IMAGES} images) MEASURED = {t['anchor_cache']:.2f}")
    print(f"  SPEED / SPEED+AIC, {t['n_speed_edits']} edits   ESTIMATE  = {t['SPEED_edits']:.2f}"
          f"   ({SPEED_EDIT_SECONDS:.0f} s/edit assumed)")
    print(f"  LACU, {t['n_lacu_requests']} requests           ESTIMATE  = "
          f"{t['LACU_lo']:.0f} - {t['LACU_hi']:.0f}"
          f"   ({LACU_HOURS[0]}-{LACU_HOURS[1]} h/request + {LACU_MAPPING_HOURS} h mapping)")
    print(f"  LACU retain rescoring                 UNMEASURED = not costed")
    print(f"\n  TOTAL = {inv['total_lo']:.0f} - {inv['total_hi']:.0f} device-hours")

    print("\nCONTRAST COEFFICIENT VECTORS over the 8 CA cells, in this order:")
    for i, c in enumerate(CELLS):
        print(f"  c{i+1}: A1={c[0]:12s} A2={c[1]:12s} target={c[2]:9s} "
              f"{'shared' if SHARED[i] else 'disjoint':9s} {'near' if NEAR[i] else 'far'}")
    cs = contrasts()
    for name, v in cs.items():
        print(f"  {name:12s} " + " ".join(f"{x:+5.2f}" for x in v))

    print("\n  orthogonality and reference cancellation:")
    for a, b in (("S", "P"), ("S", "I"), ("P", "I"), ("S", "A2_identity"), ("P", "A2_identity")):
        print(f"    <{a},{b}> = {dot(cs[a], cs[b]):+.3f}")
    for name in ("S", "P", "I"):
        ref = dot(cs[name], [1.0 if x else 0.0 for x in REF_IS_HORSE])
        print(f"    request-1 reference loading of {name:2s} = {ref:+.3f}"
              f"  (0 => the two request-1 references cancel)")

    an = json.load(open(ANALYSIS))
    est = an["per_seed"]["17"]["estimates"]
    w = [v["hi95"] - v["lo95"] for k, v in est.items()
         if isinstance(v, dict) and "lo95" in v and k.startswith("contrast[") and "|0.3]" in k]
    h_cell = statistics.median(w) / 2
    print(f"\nPRECISION, planning only. Observed per-category paired-contrast")
    print(f"  95% width at {SCENES['TEST']//2} scenes x {IMAGES_PER_SCENE} images"
          f" (the pilot layout): median {statistics.median(w):.2f} pp"
          f"  =>  per-cell half-width h = {h_cell:.2f} pp")
    print("  Scaling assumption: h ~ 1/sqrt(scenes) for the BETWEEN-scene component.")
    print("  The between/within split is UNKNOWN, so 20 scenes x 8 images and")
    print("  40 scenes x 4 images are NOT assumed equally precise.")
    print("  Cell correlation rho is UNKNOWN (all cells share the R_cat scenes).")
    print(f"\n  contrast half-widths (pp), at {SCENES['TEST']} scenes "
          f"(h scaled by sqrt({SCENES['TEST']//2}/{SCENES['TEST']})):")
    h20 = h_cell * (10 / SCENES["TEST"]) ** 0.5
    print(f"  {'rho':>6s} " + " ".join(f"{n:>8s}" for n in ("S", "P", "I", "S|near", "S-P")))
    for rho, row in precision(h20).items():
        print(f"  {rho:+6.2f} " + " ".join(f"{row[n]:8.2f}" for n in ("S", "P", "I", "S|near", "S-P")))
    sm = smoke_cost()
    print(f"\nSMOKE CALIBRATION COST (LACU_SMOKE_CALIBRATION.md):")
    print(f"  {SMOKE_LADDER_CONFIGS} ladder configs x {SMOKE_LADDER_STEPS} steps"
          f" + one run to step {SMOKE_RELOAD_STEPS} = {sm['optimizer_steps']} optimizer steps")
    print(f"  UNet passes per optimizer step at effective batch {EFFECTIVE_BATCH}"
          f" = {passes_per_optimizer_step():,}")
    print(f"  = {sm['unet_passes']:,} passes = {sm['image_equivalents']:,.0f} image-equivalents")
    print(f"  ESTIMATE {sm['device_hours'][0]:.2f} - {sm['device_hours'][1]:.2f} device-hours"
          f"  (batched speed-up {BATCHED_SPEEDUP[1]:.0f}x to {BATCHED_SPEEDUP[0]:.0f}x)")
    print("\n  Reading rule (arithmetic about an OBSERVED interval, not power and")
    print("  not a statement about true effects): with realised half-width h,")
    print("  MATERIAL+ needs the observed point > 10 + h, and NULL(10) needs")
    print("  |observed point| <= 10 - h. NULL(10) is unreachable whenever h > 10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
