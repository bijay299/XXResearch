"""One unlearning loop for every arm (SPEC v0.1 section 4 as updated by the coordinator from D 3.4 / 4.1).

Arms (priority order of the SPEC update): sc_tf, sc, fx_mix, fx_mix_greedy (T5a), mdu_t1, npo, sc_tf_cm (T6,
compute-matched by --forget-batch), ablations fx_mix_notrunc (T5b), fx_mix_rndtf (T5c), fx_mix_eps0 (T5d); extras
fx_tf, mdu_t0.
Shared across arms (matched data access / budget): LoRA (rank r on q/k/v/o/gate/up/down of ALL blocks, B=0 init),
AdamW (weight_decay 0), grad clip 1.0, constant LR (toy), forget batch + retain batch (LLaDA SFT loss, lam_r = 1),
the same forget items (all prompt templates x answer phrasings of the forget facts), the same retain pool, the same
step budget, evaluation grid and evaluation seeds (common random numbers). Reference p_theta0 = adapter-disabled pass.
FX arms: forget states = round(alpha*Bf) TF_ff states (E4) + rest from the rollout pool (E5), pool refreshed every R
steps with the current student (no grad). Compute counters, in row-forwards of one (prompt||answer) sequence:
student_fwd, student_bwd, ref_fwd, rollout_fwd (no grad; actually run) and rollout_fwd_needed (up to truncation),
plus wall-clock for train / rollouts / eval.

CLI: python -m mdmu.unlearn --arm fx_mix --lr 1e-3 --steps 200 --eval-every 25 --world-seed 0 --out run.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional

import torch

from . import toyworld as tw
from .diffusion import sft_loss
from .evaluate import (DECODING_CONFIGS, EVAL_GRID, EXTRAPOLATION, HELDOUT, HELDOUT_PRIMARY, STOCHASTIC, TRAIN_POOL,
                       copy_accuracy, fact_eval)
from .fxmix import (collect_rollout_states, emdu_loss, fact_free_tf_states, fact_mask_from_sets, fx_forget_loss,
                    hazard_certificate, mdu_loss, npo_loss, sc_loss, sc_tf_loss)
from .model import MaskPredictor, add_lora, lora_disabled, lora_params, merged_copy

ARMS: Dict[str, dict] = {
    "sc_tf": dict(loss="sc_tf"),
    "sc": dict(loss="sc", T=8),
    "fx_mix": dict(loss="fx", pool=list(TRAIN_POOL)),
    "fx_mix_greedy": dict(loss="fx", pool=["P1"]),                    # T5a (= former FX-OP1)
    "mdu_t1": dict(loss="mdu", tau=1.0),
    "emdu": dict(loss="emdu", tau=1.0),                               # Garber & Tirer 2610.25447 Alg. 1
    "npo": dict(loss="npo", beta=0.2),
    "sc_tf_cm": dict(loss="sc_tf"),                                   # T6: run with a larger --forget-batch
    "fx_mix_notrunc": dict(loss="fx", pool=list(TRAIN_POOL), truncate=False),   # T5b
    "fx_mix_rndtf": dict(loss="fx", pool=list(TRAIN_POOL), force_fact_mask=False),  # T5c
    "fx_mix_eps0": dict(loss="fx", pool=list(TRAIN_POOL), eps=0.0),  # T5d
    "fx_tf": dict(loss="fx", alpha=1.0, pool=[]),
    "mdu_t0": dict(loss="mdu", tau=0.0),
}


@dataclasses.dataclass
class RunConfig:
    arm: str = "fx_mix"
    lr: float = 1e-3
    steps: int = 200
    eval_every: int = 25
    world_seed: int = 0
    init_seed: int = 0
    run_seed: int = 0                  # = unlearning seed: LoRA A init, batch sampling, rollouts (NOT world/eval)
    ckpt_dir: str = ""                 # default: results/dev/substrate/w{W}_i{I}
    lora_r: int = 8
    lora_alpha: float = 16.0
    forget_batch: int = 32             # states (FX) / forget examples (others) per step
    retain_batch: int = 32
    lam_f: float = 1.0
    lam_r: float = 1.0
    eps: float = 1e-3                  # SPEC v0.1 default (arm may override, e.g. T5d)
    alpha: float = 0.25                # TF_ff share of FX forget states (SPEC v1; arm may override)
    refresh_every: int = 4             # R
    J: int = 4                         # states per rollout
    rollouts_per_prompt: int = 1       # per refresh; configs assigned round-robin (stratified, w_k = 1/K)
    eval_samples: int = 8              # samples per prompt for stochastic configs (forget set)
    eval_samples_retain: int = 4
    eval_samples_copy: int = 1
    eval_grid: str = ",".join(EVAL_GRID)
    eval_seed: int = 1234
    mech_samples: int = 4              # rollouts per forget prompt per stochastic config for the mechanism metric
    eval_at_start: int = 1             # 0: skip step-0 eval (B=0 => identical to the base reference eval, same seeds)
    mech_every_eval: int = 0           # 1: also run mechanism_stats (off-manifold fraction, certificate) at every ckpt
    threads: int = 1


def ckpt_path(world_seed: int, init_seed: int, ckpt_dir: str = "") -> str:
    if not ckpt_dir:
        here = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(here, "..", "..", "results", "dev", "substrate", f"w{world_seed}_i{init_seed}")
    return ckpt_dir


def load_model(world_seed: int, init_seed: int, ckpt_dir: str = "", which: str = "base") -> MaskPredictor:
    ck = torch.load(os.path.join(ckpt_path(world_seed, init_seed, ckpt_dir), f"{which}.pt"))
    assert ck["world_seed"] == world_seed, "checkpoint/world mismatch"
    m = MaskPredictor(tw.V, tw.P_LEN + tw.A_LEN, **{k: v for k, v in ck["cfg"].items() if k not in ("vocab", "seq_len")})
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def toy_fact_set(prompt, answer) -> List[int]:
    """Toy instance of the F(c, y*) builder: answer tokens not in the prompt and not special/structural/attribute
    tokens (== the fact's value tokens in this world; asserted in tests)."""
    excl = set(tw.STOI[s] for s in tw.SPECIALS) | {tw.STOI[f"ATTR_{a}"] for a in tw.ATTRS}
    return sorted({t for t in answer if t not in prompt and t not in excl})


class Counters:
    def __init__(self):
        self.c = dict(student_fwd=0, student_bwd=0, ref_fwd=0, rollout_fwd=0, rollout_fwd_needed=0, eval_fwd=0)
        self.t = dict(train_s=0.0, rollout_s=0.0, eval_s=0.0)

    def as_dict(self):
        return {**self.c, **{k: round(v, 3) for k, v in self.t.items()}}


# ============================================================================================ evaluation
@torch.no_grad()
def eval_suite(model, world: tw.World, cfg: RunConfig, counters: Optional[Counters] = None,
               forget_only: bool = False, ban_values: bool = False) -> Dict:
    """Generation-based evaluation under every config of the grid; identical seeds at every checkpoint and arm.
    Runs on a merged (LoRA-folded) copy: same function, faster."""
    t0 = time.time()
    model = merged_copy(model)
    out = {}
    grid = cfg.eval_grid.split(",")
    neigh, shared = set(world.neighbor_facts), set(world.shared_token_facts())
    for name in grid:
        fe = fact_eval(model, world.forget_facts, name, n_samples=cfg.eval_samples, seed=cfg.eval_seed,
                       ban_values=ban_values)
        row = dict(F_leak_any=fe["leak_any"], F_leak_full=fe["leak_full"], F_wf=fe["well_formed"])
        nfwd = fe["n"]
        if not forget_only:
            re_ = fact_eval(model, world.retain_facts, name, n_samples=cfg.eval_samples_retain, seed=cfg.eval_seed + 1,
                            return_per_fact=True)
            pf = re_.pop("per_fact")

            def acc(sel):
                c = [v for f, v in pf.items() if sel(f)]
                return sum(x[0] for x in c) / max(1, sum(x[1] for x in c))
            cp = copy_accuracy(model, world.copy_eval, name, n_samples=cfg.eval_samples_copy, seed=cfg.eval_seed + 4)
            row.update(N_acc=acc(lambda f: f in neigh), O_acc=acc(lambda f: f not in neigh),
                       S_acc=acc(lambda f: f in shared), R_acc=re_["leak_full"], R_wf=re_["well_formed"], copy=cp)
            nfwd += re_["n"] + len(world.copy_eval) * (cfg.eval_samples_copy if name in STOCHASTIC else 1)
        out[name] = row
        if counters is not None:
            counters.c["eval_fwd"] += DECODING_CONFIGS[name]["steps"] * nfwd
    out["agg"] = aggregate(out, grid)
    if counters is not None:
        counters.t["eval_s"] += time.time() - t0
    return out


def aggregate(ev: Dict, grid) -> Dict:
    """Primary (SPEC v1.1): max over H1-H5 ('heldout'); 'H6' extrapolation cell; 'heldout6' = max over H1-H6."""
    groups = {"train": [g for g in grid if g in TRAIN_POOL], "heldout": [g for g in grid if g in HELDOUT_PRIMARY],
              "H6": [g for g in grid if g in EXTRAPOLATION], "heldout6": [g for g in grid if g in HELDOUT],
              "all": list(grid)}
    a = {}
    for key in ("F_leak_any", "F_leak_full"):
        for gname, gs in groups.items():
            a[f"max_{key}_{gname}"] = max(ev[g][key] for g in gs) if gs else float("nan")
        a[f"argmax_{key}_heldout"] = max(groups["heldout"], key=lambda g: ev[g][key]) if groups["heldout"] else None
    a["mean_F_wf"] = sum(ev[g]["F_wf"] for g in grid) / len(grid)
    a["min_F_wf"] = min(ev[g]["F_wf"] for g in grid)
    if "R_acc" in ev[grid[0]]:
        for key in ("N_acc", "O_acc", "S_acc", "R_acc", "R_wf", "copy"):
            a[f"mean_{key}"] = sum(ev[g][key] for g in grid) / len(grid)
            a[f"min_{key}"] = min(ev[g][key] for g in grid)
    return a


@torch.no_grad()
def mechanism_stats(model, world: tw.World, grid, n_samples: int = 4, seed: int = 777) -> Dict:
    """Per config, on fresh native rollouts of `model` on the forget prompts (n_samples per prompt if stochastic):
      leak_rate          observed Lambda_F rate on these rollouts,
      cert_B             post-hoc certificate B_hat_pi (SPEC v1 section 5; upper-bounds P(Lambda_F) in expectation),
      off_manifold_frac  among first-fact-commit decision states, the fraction OFF the GT-masked manifold (some
                         visible answer token differs from y* under BOTH answer phrasings) (D 3.4)."""
    model = merged_copy(model)
    ex = [(f, t) for f in world.forget_facts for t in (0, 1)]
    g = torch.Generator().manual_seed(seed)
    res = {}
    for name in grid:
        reps = n_samples if name in STOCHASTIC else 1
        P = torch.tensor([tw.fact_prompt(f, t) for f, t in ex for _ in range(reps)], dtype=torch.long)
        facts = [f for f, _ in ex for _ in range(reps)]
        Fm = fact_mask_from_sets([list(f.value) for f in facts], tw.V)
        c = hazard_certificate(model, P, Fm, DECODING_CONFIGS[name], generator=g, gen_length=tw.A_LEN)
        n_dec = off = 0
        for r, s in enumerate(c["decision_states"]):
            if s is None:
                continue
            n_dec += 1
            ans = s[tw.P_LEN:]
            vis = ans != tw.MASK
            on = any(bool((ans[vis] == torch.tensor(tw.fact_answer(facts[r], a))[vis]).all()) for a in (0, 1))
            off += not on
        res[name] = {"n_rollouts": int(P.shape[0]), "n_leak": n_dec, "leak_rate": c["leak_rate"],
                     "cert_B": c["bound_mean"], "cert_B_se": c["bound_se"],
                     "off_manifold_frac": (off / n_dec) if n_dec else None}
    return res


def reference_evals(world_seed: int, init_seed: int, cfg: RunConfig) -> Dict:
    """Non-arm references: base, retrain, decode-time F-ban on base (inference-time only, never deployed)."""
    w = tw.make_world(world_seed)
    grid = cfg.eval_grid.split(",")
    out = {}
    for which in ("base", "retrain"):
        m = load_model(world_seed, init_seed, cfg.ckpt_dir, which)
        out[which] = eval_suite(m, w, cfg)
        out[which]["mechanism"] = mechanism_stats(m, w, grid, cfg.mech_samples)
    m = load_model(world_seed, init_seed, cfg.ckpt_dir, "base")
    out["base_fban"] = eval_suite(m, w, cfg, forget_only=True, ban_values=True)
    return out


# ============================================================================================ training loop
def run(cfg: RunConfig, log=print) -> Dict:
    torch.set_num_threads(cfg.threads)
    arm = dict(ARMS[cfg.arm])
    alpha = arm.get("alpha", cfg.alpha)
    eps = arm.get("eps", cfg.eps)
    world = tw.make_world(cfg.world_seed)
    grid = cfg.eval_grid.split(",")
    model = load_model(cfg.world_seed, cfg.init_seed, cfg.ckpt_dir)
    adapters = add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha, seed=cfg.run_seed)
    params = lora_params(adapters)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    def ref_fn(x):
        with lora_disabled(model):
            return model(x)

    fex = world.examples(world.forget_facts)
    F_ids = torch.tensor([e.prompt + e.answer for e in fex], dtype=torch.long)
    F_sets = [toy_fact_set(e.prompt, e.answer) for e in fex]
    F_mask = fact_mask_from_sets(F_sets, tw.V)
    prompts_u, p_sets = [], []
    for e, fs in zip(fex, F_sets):                       # unique forget prompts for rollouts
        if e.prompt not in prompts_u:
            prompts_u.append(e.prompt)
            p_sets.append(fs)
    P_ids = torch.tensor(prompts_u, dtype=torch.long)
    P_mask = fact_mask_from_sets(p_sets, tw.V)
    rex = world.examples(world.retain_facts) + world.copy_train
    R_ids = torch.tensor([e.prompt + e.answer for e in rex], dtype=torch.long)

    g = torch.Generator().manual_seed(1000 + cfg.run_seed)        # training randomness
    g_roll = torch.Generator().manual_seed(2000 + cfg.run_seed)   # rollouts
    C = Counters()
    curve, pool_log = [], []
    pool = None
    n_refresh = 0

    def refresh_pool():
        nonlocal pool, n_refresh
        t0 = time.time()
        cfgs = arm["pool"]
        S, M, leak_by = [], [], {}
        for rep in range(cfg.rollouts_per_prompt):
            k_of = [(i + rep + n_refresh) % len(cfgs) for i in range(len(prompts_u))]   # stratified round-robin
            for k, name in enumerate(cfgs):
                rows = [i for i in range(len(prompts_u)) if k_of[i] == k]
                if not rows:
                    continue
                out = collect_rollout_states(model, P_ids[rows], P_mask[rows], DECODING_CONFIGS[name], cfg.J,
                                             generator=g_roll, gen_length=tw.A_LEN,
                                             truncate=arm.get("truncate", True))
                S.append(out["states"])
                M.append(P_mask[rows][out["owner"]])
                C.c["rollout_fwd"] += out["fwd_rows"]
                C.c["rollout_fwd_needed"] += int(out["n_kept"].sum())
                lb = leak_by.setdefault(name, [0, 0])
                lb[0] += int(out["leak"].sum())
                lb[1] += len(rows)
        pool = (torch.cat(S), torch.cat(M))
        n_refresh += 1
        pool_log.append({"n_states": int(pool[0].shape[0]), "leak_rate": {k: v[0] / v[1] for k, v in leak_by.items()}})
        C.t["rollout_s"] += time.time() - t0

    snapshots = {}

    def do_eval(step):
        model.eval()
        snapshots[step] = {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
        ev = eval_suite(model, world, cfg, C)
        if cfg.mech_every_eval:
            t0m = time.time()
            ev["mechanism"] = mechanism_stats(model, world, grid, cfg.mech_samples)
            C.t["eval_s"] += time.time() - t0m
        ev["step"] = step
        ev["counters"] = C.as_dict()
        curve.append(ev)
        a = ev["agg"]
        log(json.dumps({"arm": cfg.arm, "lr": cfg.lr, "step": step, "ho5_max_any": round(a["max_F_leak_any_heldout"], 3),
                        "tr_max_full": round(a["max_F_leak_full_train"], 3), "F_wf": round(a["mean_F_wf"], 3),
                        "minR": round(a["min_R_acc"], 3), "minCopy": round(a["min_copy"], 3)}), flush=True)

    if cfg.eval_at_start:
        do_eval(0)
    diag_last = {}
    for step in range(1, cfg.steps + 1):
        if arm["loss"] == "fx" and arm.get("pool") and alpha < 1.0 and (step - 1) % cfg.refresh_every == 0:
            refresh_pool()
        t0 = time.time()
        idx = torch.randint(0, F_ids.shape[0], (cfg.forget_batch,), generator=g)
        if arm["loss"] == "fx":
            n_tf = int(round(alpha * cfg.forget_batch))
            parts_s, parts_m = [], []
            if n_tf:
                ti = idx[:n_tf]
                s_tf, _ = fact_free_tf_states(F_ids[ti], tw.P_LEN, F_mask[ti], generator=g,
                                              force_fact_mask=arm.get("force_fact_mask", True))
                parts_s.append(s_tf)
                parts_m.append(F_mask[ti])
            n_op = cfg.forget_batch - n_tf
            if n_op:
                pi = torch.randint(0, pool[0].shape[0], (n_op,), generator=g)
                parts_s.append(pool[0][pi])
                parts_m.append(pool[1][pi])
            d = fx_forget_loss(model, ref_fn, torch.cat(parts_s), torch.cat(parts_m), eps=eps, prompt_len=tw.P_LEN)
        elif arm["loss"] == "sc_tf":
            d = sc_tf_loss(model, ref_fn, F_ids[idx], tw.P_LEN, F_mask[idx], eps=eps, generator=g)
        elif arm["loss"] == "sc":
            d = sc_loss(model, ref_fn, F_ids[idx], tw.P_LEN, F_mask[idx], eps=eps, T=arm["T"], generator=g)
            C.c["rollout_fwd"] += d["rollout_fwd_rows"]
            C.c["rollout_fwd_needed"] += d["rollout_fwd_rows"]
        elif arm["loss"] == "npo":
            d = npo_loss(model, ref_fn, F_ids[idx], tw.P_LEN, beta=arm["beta"], generator=g)
        elif arm["loss"] == "mdu":
            d = mdu_loss(model, ref_fn, F_ids[idx], tw.P_LEN, tau=arm["tau"], generator=g)
        elif arm["loss"] == "emdu":
            d = emdu_loss(model, ref_fn, F_ids[idx], tw.P_LEN, tau=arm["tau"], generator=g)
        else:
            raise ValueError(arm)
        ridx = torch.randint(0, R_ids.shape[0], (cfg.retain_batch,), generator=g)
        l_ret = sft_loss(model, R_ids[ridx], tw.P_LEN, generator=g)
        loss = cfg.lam_f * d["loss"] + cfg.lam_r * l_ret
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        C.c["student_fwd"] += cfg.forget_batch + cfg.retain_batch
        C.c["student_bwd"] += cfg.forget_batch + cfg.retain_batch
        C.c["ref_fwd"] += cfg.forget_batch               # one frozen pass per forget state, every arm
        C.t["train_s"] += time.time() - t0
        diag_last = {k: (float(v.detach()) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
        diag_last["retain"] = float(l_ret)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            do_eval(step)
            curve[-1]["train_diag"] = diag_last
    res = {"config": dataclasses.asdict(cfg), "alpha_used": alpha, "eps_used": eps, "arm_spec": arm, "curve": curve,
           "pool_log": pool_log, "counters": C.as_dict(), "n_trainable": sum(p.numel() for p in params),
           "snapshots": snapshots}
    return res, model, world


def model_from_snapshot(world_seed: int, init_seed: int, snap: Dict, cfg: RunConfig, ckpt_dir: str = ""):
    """Rebuild the LoRA model of a stored checkpoint (for post-hoc mechanism / certificate evaluation)."""
    m = load_model(world_seed, init_seed, ckpt_dir)
    add_lora(m, r=cfg.lora_r, alpha=cfg.lora_alpha, seed=cfg.run_seed)
    missing = m.load_state_dict(snap, strict=False)
    assert not missing.unexpected_keys
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(RunConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--unlearn-seed", type=int, default=None,
                    help="alias of --run-seed: changes only the unlearning RNG (LoRA A init, batch sampling, "
                         "rollouts); world, base checkpoint and evaluation seeds are untouched")
    ap.add_argument("--out", required=True)
    a = vars(ap.parse_args())
    out = a.pop("out")
    us = a.pop("unlearn_seed")
    if us is not None:
        a["run_seed"] = us
    res, _, _ = run(RunConfig(**a))
    torch.save(res.pop("snapshots"), os.path.splitext(out)[0] + "_snapshots.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
