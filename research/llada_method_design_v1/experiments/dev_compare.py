"""Dev comparison of arms (SPEC v1/v1.1 section 4) on DEV worlds only (seeds 0, 1). Never run on worlds >= 2.

Modes
  --refs                 reference evals (base, retrain, decode-time F-ban on base) -> <out>/refs.json
  --shard i/n            run the i-th of n shards of the (arm, lr) job list -> <out>/runs/<arm>_lr<lr>.json (+ snapshots)
  --report               selection + mechanism/certificate at selected checkpoints + tables -> <out>/report.{json,md}
Run each mode through tools/ledger_run.py (1 thread per process; at most 2 processes in parallel).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch  # noqa: E402

from mdmu import toyworld as tw  # noqa: E402
from mdmu.evaluate import EVAL_GRID  # noqa: E402
from mdmu.unlearn import RunConfig, mechanism_stats, model_from_snapshot, reference_evals, run  # noqa: E402

LRS = [5e-4, 1e-3, 2e-3]
JOBS = (  # (arm, lrs, extra RunConfig overrides) in SPEC-update priority order
    [("sc_tf", LRS, {}), ("sc", LRS, {}), ("fx_mix", LRS, {}), ("fx_mix_greedy", LRS, {}),
     ("mdu_t1", LRS, {}), ("npo", LRS, {}),
     # T6 compute-matched SC-TF: FX-Mix per-step row-forwards = 64 student + 32 ref + ~40 rollout = ~136
     # => SC-TF needs forget batch B with 2B + 32 = 136 -> B = 52 (verified against counters in the report)
     ("sc_tf_cm", LRS, {"forget_batch": 52}),
     ("fx_mix_notrunc", [1e-3, 2e-3], {}), ("fx_mix_rndtf", [1e-3, 2e-3], {}), ("fx_mix_eps0", [1e-3, 2e-3], {})])
STEPS, EVAL_EVERY = 200, 25
# World 1 (coordinator request 05:0x): 8 arms x LR {1e-3, 2e-3, 4e-3}, 300 steps, eval every 25
LRS_W1 = [1e-3, 2e-3, 4e-3]
JOBS_W1 = [("sc_tf", LRS_W1, {}), ("sc_tf_cm", LRS_W1, {"forget_batch": 52}), ("sc", LRS_W1, {}),
           ("fx_mix", LRS_W1, {}), ("fx_mix_greedy", LRS_W1, {}), ("emdu", LRS_W1, {}), ("npo", LRS_W1, {}),
           ("mdu_t1", LRS_W1, {})]
STEPS_W1 = 300


def plan(world_seed):
    return (JOBS, STEPS) if world_seed == 0 else (JOBS_W1, STEPS_W1)


def job_list(world_seed=0):
    jobs, _ = plan(world_seed)
    return [(a, lr, ov) for a, lrs, ov in jobs for lr in lrs]


def run_name(arm, lr):
    return f"{arm}_lr{lr:g}"


# ============================================================================================ selection rules
def retention_ok(agg, base_agg, d_ret=0.05, d_copy=0.02, ret_key="mean_R_acc"):
    """Retention constraint (coordinator rule 05:0x, to be locked in VALIDATION_SPEC): retained-fact accuracy
    averaged over the 9 configs >= base - d_ret AND copy accuracy (min over the 9 configs) >= base - d_copy.
    ret_key="min_R_acc" gives the earlier (04:38) variant, kept for comparison only."""
    return agg[ret_key] >= base_agg[ret_key] - d_ret and agg["min_copy"] >= base_agg["min_copy"] - d_copy


def select_cut_A(runs_for_arm, base_agg, d_ret=0.05, d_copy=0.02):
    """Cut A (locked rule): among retention-feasible checkpoints (all LRs x steps), the minimum held-out (H1-H5)
    max leak_any; ties -> higher mean retained accuracy, then fewer steps, then lower LR. None = INFEASIBLE."""
    cand = [p for p in points(runs_for_arm) if retention_ok(p["ev"]["agg"], base_agg, d_ret, d_copy)]
    if not cand:
        return None
    return min(cand, key=lambda p: (p["ev"]["agg"]["max_F_leak_any_heldout"], -p["ev"]["agg"]["mean_R_acc"],
                                    p["step"], p["lr"]))


def select_cut_B(runs_for_arm, retrain_agg):
    """Cut B (locked rule): among checkpoints with held-out (H1-H5) max leak_any <= the retrain model's held-out max
    leak_any, the maximum mean retained accuracy; ties -> lower leak, fewer steps. None = INFEASIBLE."""
    thr = retrain_agg["max_F_leak_any_heldout"]
    cand = [p for p in points(runs_for_arm) if p["ev"]["agg"]["max_F_leak_any_heldout"] <= thr]
    if not cand:
        return None
    return max(cand, key=lambda p: (p["ev"]["agg"]["mean_R_acc"], -p["ev"]["agg"]["max_F_leak_any_heldout"],
                                    -p["step"]))


def points(runs_for_arm):
    """All (lr, step) strength-grid points of an arm (step 0 = base excluded)."""
    out = []
    for r in runs_for_arm:
        for ev in r["curve"]:
            if ev["step"] > 0:
                out.append({"lr": r["config"]["lr"], "step": ev["step"], "ev": ev})
    return out


def select_leak_at_matched_retention(runs_for_arm, base_agg, key="max_F_leak_any_heldout", d_ret=0.05, d_copy=0.02):
    """Cut 1 (primary): among checkpoints meeting retention_ok, the lowest held-out (H1-H5) max leak_any; ties ->
    higher min retained accuracy, then fewer steps. Returns None (INFEASIBLE) if no checkpoint qualifies."""
    cand = [p for p in points(runs_for_arm) if retention_ok(p["ev"]["agg"], base_agg, d_ret, d_copy)]
    if not cand:
        return None
    return min(cand, key=lambda p: (p["ev"]["agg"][key], -p["ev"]["agg"]["min_R_acc"], p["step"], p["lr"]))


def select_retention_at_matched_leak(runs_for_arm, leak_key="max_F_leak_full_train", thr=0.02):
    """Cut 2 (converse): per LR, the FIRST checkpoint with leak_key <= thr; across LRs, the best min retained accuracy
    (ties -> fewer steps). Default = coordinator rule (in-pool max leak_full <= 0.02). None = never reached."""
    firsts = []
    for r in runs_for_arm:
        for ev in r["curve"]:
            if ev["step"] > 0 and ev["agg"][leak_key] <= thr:
                firsts.append({"lr": r["config"]["lr"], "step": ev["step"], "ev": ev})
                break
    if not firsts:
        return None
    return max(firsts, key=lambda p: (p["ev"]["agg"]["min_R_acc"], -p["step"]))


# ============================================================================================ modes
def do_refs(out, world_seed):
    cfg = RunConfig(world_seed=world_seed, init_seed=world_seed)
    t0 = time.time()
    refs = reference_evals(world_seed, world_seed, cfg)
    refs["wall_s"] = round(time.time() - t0, 1)
    json.dump(refs, open(os.path.join(out, "refs.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v["agg"] for k, v in refs.items() if isinstance(v, dict) and "agg" in v}, indent=1))


def do_shard(out, world_seed, shard, n_shards, only=None):
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    jobs = job_list(world_seed)
    steps = plan(world_seed)[1]
    if only:
        jobs = [j for j in jobs if j[0] in only]
    for k, (arm, lr, ov) in enumerate(jobs):
        if k % n_shards != shard:
            continue
        path = os.path.join(out, "runs", run_name(arm, lr) + ".json")
        if os.path.exists(path):
            continue
        cfg = RunConfig(arm=arm, lr=lr, steps=steps, eval_every=EVAL_EVERY, world_seed=world_seed,
                        init_seed=world_seed, eval_at_start=0, **ov)
        t0 = time.time()
        res, _, _ = run(cfg)
        res["wall_s"] = round(time.time() - t0, 1)
        torch.save(res.pop("snapshots"), path.replace(".json", "_snapshots.pt"))
        json.dump(res, open(path, "w"))
        print(f"done {arm} lr={lr} wall={res['wall_s']}s", flush=True)


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def do_report(out, world_seed):
    refs = json.load(open(os.path.join(out, "refs.json")))
    base_agg, rt_agg = refs["base"]["agg"], refs["retrain"]["agg"]
    runs = {}
    for fn in sorted(os.listdir(os.path.join(out, "runs"))):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(out, "runs", fn)))
            runs.setdefault(r["config"]["arm"], []).append(r)
    world = tw.make_world(world_seed)
    grid = list(EVAL_GRID)
    rep = {"world_seed": world_seed, "base": base_agg, "retrain": rt_agg, "base_fban": refs["base_fban"]["agg"],
           "mechanism_refs": {k: refs[k].get("mechanism") for k in ("base", "retrain")}, "arms": {}}
    order = [a for a, _, _ in plan(world_seed)[0] if a in runs]
    # converse-cut variant relative to the retrain model (SPEC v1 section 6): held-out leak_any <= retrain + 0.05
    thr_rel = rt_agg["max_F_leak_any_heldout"] + 0.05
    for arm in order:
        rs = runs[arm]
        cutA = select_cut_A(rs, base_agg)
        cutB = select_cut_B(rs, rt_agg)
        sel = select_leak_at_matched_retention(rs, base_agg)                     # 04:38 variant (min-over-configs)
        conv = select_retention_at_matched_leak(rs)
        entry = {"cutA": None, "cutB": None, "cut1": None, "cut2": None, "curve": [], "compute": {}}
        for p in points(rs):
            a = p["ev"]["agg"]
            entry["curve"].append({"lr": p["lr"], "step": p["step"], "ho5_any": a["max_F_leak_any_heldout"],
                                   "H6_any": a["max_F_leak_any_H6"], "train_any": a["max_F_leak_any_train"],
                                   "train_full": a["max_F_leak_full_train"], "min_R_acc": a["min_R_acc"],
                                   "mean_R_acc": a["mean_R_acc"], "min_copy": a["min_copy"],
                                   "mean_F_wf": a["mean_F_wf"], "mean_S_acc": a["mean_S_acc"],
                                   "ret_ok": retention_ok(a, base_agg)})
        for name, s in (("cutA", cutA), ("cutB", cutB), ("cut1", sel), ("cut2", conv)):
            if s is None:
                continue
            a = s["ev"]["agg"]
            r = next(x for x in rs if x["config"]["lr"] == s["lr"])
            cnt = s["ev"]["counters"]
            e = {"lr": s["lr"], "step": s["step"], "agg": a, "per_config": {g: s["ev"][g] for g in grid},
                 "counters_at_ckpt": cnt}
            if name in ("cutA", "cutB"):
                snaps = torch.load(os.path.join(out, "runs", run_name(arm, s["lr"]) + "_snapshots.pt"))
                cfg = RunConfig(**{k: v for k, v in r["config"].items() if k in RunConfig.__dataclass_fields__})
                m = model_from_snapshot(world_seed, world_seed, snaps[s["step"]], cfg)
                e["mechanism"] = mechanism_stats(m, world, grid, cfg.mech_samples)
            entry[name] = e
        r0 = rs[0]
        c = r0["counters"]
        st = r0["config"]["steps"]
        entry["compute"] = {"per_step_student_fwd": c["student_fwd"] / st, "per_step_student_bwd": c["student_bwd"] / st,
                            "per_step_ref_fwd": c["ref_fwd"] / st, "per_step_rollout_fwd": c["rollout_fwd"] / st,
                            "per_step_rollout_fwd_needed": c["rollout_fwd_needed"] / st,
                            "per_step_total_fwd": (c["student_fwd"] + c["ref_fwd"] + c["rollout_fwd"]) / st,
                            "train_s_per_step": (c["train_s"] + c["rollout_s"]) / st, "eval_s_total": c["eval_s"]}
        rep["arms"][arm] = entry
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    write_md(rep, order, os.path.join(out, "report.md"), thr_rel)
    write_short(rep, order, os.path.join(out, "summary_table.md"))


def write_short(rep, order, path):
    """Compact Cut A / Cut B table (for DEV_LOG / replies)."""
    rt = rep["retrain"]
    L = [f"World {rep['world_seed']}: base mean R_acc {rep['base']['mean_R_acc']:.3f}, min copy {rep['base']['min_copy']:.2f}; "
         f"retrain HO5 {rt['max_F_leak_any_heldout']:.3f}, mean R_acc {rt['mean_R_acc']:.3f}, F_wf {rt['mean_F_wf']:.2f}; "
         f"F-ban F_wf {rep['base_fban']['mean_F_wf']:.2f}",
         "| arm | Cut A: lr@step HO5 / H6 / meanR / S / F_wf | Cut B: lr@step meanR / HO5 / S / F_wf | fwd/step |",
         "|---|---|---|---|"]
    for arm in order:
        e = rep["arms"][arm]
        cells = []
        for cut in ("cutA", "cutB"):
            s = e[cut]
            if s is None:
                cells.append("INFEASIBLE")
                continue
            a = s["agg"]
            if cut == "cutA":
                cells.append(f"{s['lr']:g}@{s['step']} {a['max_F_leak_any_heldout']:.3f} / {a['max_F_leak_any_H6']:.2f} / "
                             f"{a['mean_R_acc']:.3f} / {a['mean_S_acc']:.3f} / {a['mean_F_wf']:.2f}")
            else:
                cells.append(f"{s['lr']:g}@{s['step']} {a['mean_R_acc']:.3f} / {a['max_F_leak_any_heldout']:.3f} / "
                             f"{a['mean_S_acc']:.3f} / {a['mean_F_wf']:.2f}")
        L.append(f"| {arm} | {cells[0]} | {cells[1]} | {e['compute']['per_step_total_fwd']:.0f} |")
    open(path, "w").write("\n".join(L) + "\n")


def write_md(rep, order, path, thr_rel):
    b, rt, fb = rep["base"], rep["retrain"], rep["base_fban"]
    L = [f"# Dev comparison — world {rep['world_seed']} (DEV; seeds 0/1 only) — generated by experiments/dev_compare.py", "",
         "Primary statistic (SPEC v1.1): **HO5** = max leak_any over held-out H1–H5; H6 (4 tok/step) extrapolation cell",
         "separately; TR = max over pool P1–P3. **Locked rule (coordinator 05:0x):** retention constraint = mean-over-9-",
         "configs retained-fact accuracy ≥ base−0.05 AND min-over-9 copy ≥ base−0.02. **Cut A** = min HO5 over feasible",
         "checkpoints (all LR × step); **Cut B** = max mean retained accuracy over checkpoints with HO5 ≤ retrain HO5",
         f"({rep['retrain']['max_F_leak_any_heldout']:.3f}); INFEASIBLE if none. Secondary (earlier rules): Cut 1 = min HO5 s.t.",
         "min-over-9 retained acc ≥ base−0.05; Cut 2 = first checkpoint with in-pool max leak_full ≤ 0.02 (best LR).", "",
         "## References", "",
         "| model | HO5 any | H6 any | TR any | TR full | min R_acc | mean R_acc | S_acc mean | min copy | mean F_wf |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in (("base", b), ("retrain", rt)):
        L.append(f"| {name} | {fmt(a['max_F_leak_any_heldout'])} | {fmt(a['max_F_leak_any_H6'])} | "
                 f"{fmt(a['max_F_leak_any_train'])} | {fmt(a['max_F_leak_full_train'])} | {fmt(a['min_R_acc'])} | "
                 f"{fmt(a['mean_R_acc'])} | {fmt(a['mean_S_acc'])} | {fmt(a['min_copy'])} | {fmt(a['mean_F_wf'])} |")
    L.append(f"| base + decode-time F-ban (inference-time ref.) | {fmt(fb['max_F_leak_any_heldout'])} | "
             f"{fmt(fb['max_F_leak_any_H6'])} | {fmt(fb['max_F_leak_any_train'])} | {fmt(fb['max_F_leak_full_train'])} "
             f"| (= base) | (= base) | (= base) | (= base) | {fmt(fb['mean_F_wf'])} |")
    L += ["", "## Cut A — held-out leak at matched retention (primary, locked rule)", "",
          "| arm | LR | step | **HO5 any** | H6 any | HO6 any | TR any | TR full | min R_acc | mean R_acc | S_acc mean | S_acc min | min copy | mean F_wf | min F_wf | fwd/step (stud+ref+roll) | bwd/step | train s/step |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in order:
        e = rep["arms"][arm]
        c = e["compute"]
        comp = (f"{c['per_step_student_fwd']:.0f}+{c['per_step_ref_fwd']:.0f}+{c['per_step_rollout_fwd']:.0f}"
                f"={c['per_step_total_fwd']:.0f} | {c['per_step_student_bwd']:.0f} | {c['train_s_per_step']:.3f}")
        s = e["cutA"]
        if s is None:
            L.append(f"| {arm} | INFEASIBLE (no checkpoint meets retention) | | | | | | | | | | | | | | {comp} |")
            continue
        a = s["agg"]
        L.append(f"| {arm} | {s['lr']:g} | {s['step']} | **{fmt(a['max_F_leak_any_heldout'])}** | "
                 f"{fmt(a['max_F_leak_any_H6'])} | {fmt(a['max_F_leak_any_heldout6'])} | {fmt(a['max_F_leak_any_train'])} | "
                 f"{fmt(a['max_F_leak_full_train'])} | {fmt(a['min_R_acc'])} | {fmt(a['mean_R_acc'])} | {fmt(a['mean_S_acc'])} | "
                 f"{fmt(a['min_S_acc'])} | {fmt(a['min_copy'])} | {fmt(a['mean_F_wf'])} | {fmt(a['min_F_wf'])} | {comp} |")
    for cut, title in (("cutB", "Cut B — retention at matched leak (max mean R_acc s.t. HO5 ≤ retrain HO5; locked rule)"),
                       ("cut1", "Secondary: Cut 1 (min HO5 s.t. MIN-over-9 retained acc ≥ base−0.05)"),
                       ("cut2", "Secondary: Cut 2 (first ckpt with in-pool max leak_full ≤ 0.02, best LR)")):
        L += ["", f"## {title}", "",
              "| arm | LR | step | HO5 any | H6 any | TR full | min R_acc | mean R_acc | S_acc mean | S_acc min | min copy | mean F_wf |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in order:
            s = rep["arms"][arm][cut]
            if s is None:
                L.append(f"| {arm} | never reached | | | | | | | | | | |")
                continue
            a = s["agg"]
            L.append(f"| {arm} | {s['lr']:g} | {s['step']} | {fmt(a['max_F_leak_any_heldout'])} | "
                     f"{fmt(a['max_F_leak_any_H6'])} | {fmt(a['max_F_leak_full_train'])} | {fmt(a['min_R_acc'])} | "
                     f"{fmt(a['mean_R_acc'])} | {fmt(a['mean_S_acc'])} | {fmt(a['min_S_acc'])} | "
                     f"{fmt(a['min_copy'])} | {fmt(a['mean_F_wf'])} |")
    L += ["", "## Per-config leak_any / well-formedness at the Cut-A checkpoint", "",
          "| arm | " + " | ".join(EVAL_GRID) + " |", "|---|" + "---|" * len(EVAL_GRID)]
    for arm in order:
        s = rep["arms"][arm]["cutA"]
        if s is None:
            continue
        L.append(f"| {arm} | " + " | ".join(f"{s['per_config'][g]['F_leak_any']:.2f}/{s['per_config'][g]['F_wf']:.2f}"
                                           for g in EVAL_GRID) + " |")
    L += ["", "## Mechanism: off-manifold fraction of first-commit decision states; certificate B̂ ± s.e. (exact π_T, SPEC v1.3) vs observed leak",
          "(fresh rollouts on forget prompts, 4 per prompt for stochastic configs; Cut-A and Cut-B checkpoints)", "",
          "| model | " + " | ".join(EVAL_GRID) + " |", "|---|" + "---|" * len(EVAL_GRID)]

    def mrow(name, m):
        cells = []
        for g in EVAL_GRID:
            x = m[g]
            off = "—" if x["off_manifold_frac"] is None else f"{x['off_manifold_frac']:.2f}"
            se = x.get("cert_B_se")
            se = "" if se is None or (isinstance(se, float) and math.isnan(se)) else f"±{se:.2f}"
            cells.append(f"off {off} (n={x['n_leak']}); B̂ {x['cert_B']:.2f}{se} vs {x['leak_rate']:.2f}")
        return f"| {name} | " + " | ".join(cells) + " |"
    for k in ("base", "retrain"):
        if rep["mechanism_refs"].get(k):
            L.append(mrow(k, rep["mechanism_refs"][k]))
    for arm in order:
        for cut in ("cutA", "cutB"):
            s = rep["arms"][arm][cut]
            if s is not None and "mechanism" in s:
                L.append(mrow(f"{arm} [{cut[-1]}]", s["mechanism"]))
    L += ["", "## Full strength curves (HO5 any / mean R_acc / mean F_wf / mean S_acc) per LR × step", ""]
    for arm in order:
        L.append(f"**{arm}**: " + "; ".join(
            f"lr{p['lr']:g}@{p['step']}: {p['ho5_any']:.2f}/{p['mean_R_acc']:.3f}/{p['mean_F_wf']:.2f}/{p['mean_S_acc']:.2f}"
            f"{'' if p['ret_ok'] else '✗'}"
            for p in rep["arms"][arm]["curve"]))
        L.append("")
    L.append("✗ = fails the retention constraint.")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--only", default=None, help="comma-separated arm names")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    assert a.world_seed in (0, 1), "dev worlds are seeds 0 and 1 only (>= 2 reserved for the verifier)"
    torch.set_num_threads(1)
    out = a.out or os.path.join(HERE, "..", "results", "dev", f"compare_w{a.world_seed}")
    os.makedirs(out, exist_ok=True)
    if a.refs:
        do_refs(out, a.world_seed)
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        do_shard(out, a.world_seed, i, n, a.only.split(",") if a.only else None)
    if a.report:
        do_report(out, a.world_seed)


if __name__ == "__main__":
    main()
