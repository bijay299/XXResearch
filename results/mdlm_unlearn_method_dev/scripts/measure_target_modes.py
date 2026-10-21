"""
Phase B — target-mode decomposition. Explains the measured bimodality, NO TRAINING.

H6': the failure mode is set by the TARGET'S OWN MODE, not by the mask rate.

  anchor family (MDU, EMDU): target built from a masked / null prompt, where the likeliest
      continuation is end-of-text  ->  predicted DEGENERATE
  renormalisation family (exclusion, SC-TF): target is the reference renormalised with the
      fact tokens removed, i.e. its second choice  ->  predicted REPLACE

On this account neither family is buggy: each student matches its target faithfully and the
target's mode is the failure. That explains the one thing a mask-rate story cannot -- why the
grouping tracks objective FAMILY and is invariant to tuning (MDU collapses at all three tau,
exclusion stays fluent at all three eps).

Known limitation, stated up front: **NPO has no target distribution.** It is a ratio-based
preference loss, so H6' cannot explain NPO's collapse at beta in {0.05, 0.2}. H6' therefore
covers 6 of the 9 collapsed grid points (MDU x3, EMDU x3), not all 9. NPO is measured here
only for its reference terms, and its collapse needs a separate explanation.

Targets are reconstructed from the formulas the PINNED objective functions implement, because
those functions return losses rather than targets. The pinned module is imported read-only
and its source hash recorded, and each reconstruction is checked against the pinned loss: if
KL(target || student) recomputed from the reconstruction does not match the pinned function's
own loss to tolerance, the reconstruction is wrong and the run aborts.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
from common_big import FACTS, load_base, load_tokenizer
import llada_fxmix as FX                      # PINNED, read-only
from mdmu.fxmix import exclusion_target       # PINNED, read-only

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
GEN_LENGTH = 32
VOCAB = 126464


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/memorized_v2_s20261101")
    ap.add_argument("--n-states", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--taus", default="0.5,1.0,2.0")
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--tag", default="target_modes")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]

    src = open("/data/bijaypandey/llada_poc/scripts/llada_fxmix.py", "rb").read()
    R = {"study": "target_mode_decomposition", "hypothesis": "H6'",
         "pinned_source_sha256": hashlib.sha256(src).hexdigest(),
         "memo": a.memo, "hparams": vars(a),
         "limitation": ("NPO has no target distribution (ratio-based preference loss), so "
                        "H6' covers 6 of the 9 collapsed grid points (MDU x3, EMDU x3), "
                        "not NPO's 2.")}

    tok = load_tokenizer()
    base = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, a.memo).merge_and_unload().to(DEV).eval()

    forget = [f for f in FACTS if f["split"] == "forget"]
    # build rows exactly as the training arms do, via the pinned helpers
    pid, aid, fms = [], [], []
    for f in forget:
        q = f["train_question"]
        p_ = FX.chat_prompt_ids(tok, q)
        aa = FX.answer_ids(tok, f["answer_sentence"], GEN_LENGTH, append_eot=False)
        pid.append(p_)
        aid.append(aa)
        fs = FX.build_fact_set(tok, p_, f["answer"])
        fms.append([int(t) for t in fs["F"]])
    Pm, Am = FX.left_pad(pid, EOS_ID)
    fact_mask = torch.zeros(len(forget), VOCAB, dtype=torch.bool)
    for i, ts in enumerate(fms):
        for t in ts:
            if 0 <= t < VOCAB:
                fact_mask[i, t] = True
    full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
    attn = torch.cat([Am, torch.ones(len(forget), GEN_LENGTH, dtype=Am.dtype)], 1).to(DEV)
    ans_start = torch.full((len(forget),), Pm.shape[1], dtype=torch.long).to(DEV)
    fact_mask = fact_mask.to(DEV)

    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    idx = torch.randint(full.shape[0], (a.n_states,), generator=gen, device=DEV)
    noisy, masked, p_mask = FX.random_tf_states_llada(full[idx], ans_start[idx],
                                                      eps=1e-3, generator=gen)
    st_attn, st_fm = attn[idx], fact_mask[idx]
    pos = (noisy == MASK_ID)
    ar = torch.arange(noisy.shape[1], device=DEV)[None]
    pos &= ar >= ans_start[idx][:, None]
    rows, cols = pos.nonzero(as_tuple=True)
    true_tok = full[idx][rows, cols]
    # the realised mask rate of each state, for the mask-rate control
    mrate = p_mask[:, 0].detach().float().cpu().numpy() if p_mask.dim() == 2 else None
    print(f"[states] {a.n_states} teacher-forced states, {rows.numel()} masked answer "
          f"positions", flush=True)

    EOSPAD = {EOS_ID, 126081}

    def describe(name, logp, extra=None):
        """logp: [n_pos, V] log-probabilities of the TARGET."""
        p = logp.exp().float()
        top = p.argmax(-1)
        H = -(p * logp.float()).sum(-1) / np.log(2)
        eos_mass = p[:, list(EOSPAD)].sum(-1)
        true_mass = p.gather(1, true_tok[:, None]).squeeze(1)
        d = {"name": name,
             "n_positions": int(p.shape[0]),
             "mode_is_eos_or_pad_frac": round(float(
                 sum(float(t.item() in EOSPAD) for t in top) / len(top)), 4),
             "mean_mass_on_eos_pad": round(float(eos_mass.mean()), 5),
             "mean_mass_on_true_token": round(float(true_mass.mean()), 6),
             "mean_entropy_bits": round(float(H.mean()), 3),
             "median_entropy_bits": round(float(H.median()), 3),
             "top_modes": [[tok.decode([int(t)]), int(c)] for t, c in
                           zip(*np.unique(top.cpu().numpy(), return_counts=True))][:0] or
                          sorted([[tok.decode([int(v)]), int(c)] for v, c in
                                  zip(*np.unique(top.cpu().numpy(), return_counts=True))],
                                 key=lambda x: -x[1])[:6]}
        if extra:
            d.update(extra)
        print(f"[{name:22s}] mode is EOS/pad in {d['mode_is_eos_or_pad_frac']:.3f} of "
              f"positions | EOS mass {d['mean_mass_on_eos_pad']:.4f} | true-token mass "
              f"{d['mean_mass_on_true_token']:.2e} | H {d['mean_entropy_bits']:.2f} bits | "
              f"modes {d['top_modes'][:3]}", flush=True)
        return d

    out = []
    with torch.no_grad():
        # the memorized reference itself, as the control every target is compared against
        z_ref = model(noisy, attention_mask=st_attn).logits[rows, cols].float()
        out.append(describe("REFERENCE p_theta0", torch.log_softmax(z_ref, -1)))

        # ---- renormalisation family: exclusion target r_eps at the same state ----
        for eps in (1e-5, 1e-3, 1e-1):
            lr = exclusion_target(z_ref, st_fm[rows], eps)
            out.append(describe(f"exclusion eps={eps:g}", lr, {"family": "renormalisation",
                                                               "eps": eps}))

        # ---- anchor family: MDU uses the state with the PROMPT masked ----
        un = noisy.clone()
        for i in range(un.shape[0]):
            un[i, :int(ans_start[idx][i])] = MASK_ID
        z_mdu = model(un, attention_mask=st_attn).logits[rows, cols].float()
        for tau in taus:
            out.append(describe(f"MDU anchor tau={tau:g}",
                                torch.log_softmax(z_mdu / max(tau, 1e-6), -1),
                                {"family": "anchor", "tau": tau,
                                 "anchor_input": "prompt masked"}))

        # ---- anchor family: EMDU uses the FULLY masked sequence (FAM) ----
        fam = torch.full_like(noisy, MASK_ID)
        z_emdu = model(fam, attention_mask=st_attn).logits[rows, cols].float()
        for tau in taus:
            out.append(describe(f"EMDU anchor tau={tau:g}",
                                torch.log_softmax(z_emdu / max(tau, 1e-6), -1),
                                {"family": "anchor", "tau": tau,
                                 "anchor_input": "fully masked (FAM)"}))

    # ---- mask-rate control: does any of this depend on the realised rate? ----
    band = {}
    if mrate is not None:
        srow = rows.cpu().numpy()
        r_of_pos = mrate[srow]
        with torch.no_grad():
            lr = exclusion_target(z_ref, st_fm[rows], a.eps)
            p_ex = lr.exp().float()
            p_an = torch.log_softmax(z_emdu, -1).exp().float()
            for lo, hi, nm in ((0.0, 0.33, "low_t"), (0.33, 0.67, "mid_t"), (0.67, 1.01, "high_t")):
                m = (r_of_pos >= lo) & (r_of_pos < hi)
                if m.sum() == 0:
                    continue
                mi = torch.tensor(np.where(m)[0], device=DEV)
                band[nm] = {
                    "n_positions": int(m.sum()),
                    "exclusion_eos_mass": round(float(p_ex[mi][:, list(EOSPAD)].sum(-1).mean()), 5),
                    "emdu_anchor_eos_mass": round(float(p_an[mi][:, list(EOSPAD)].sum(-1).mean()), 5)}
                print(f"[band {nm:6s}] n={band[nm]['n_positions']:5d}  exclusion EOS mass "
                      f"{band[nm]['exclusion_eos_mass']:.4f}  EMDU anchor EOS mass "
                      f"{band[nm]['emdu_anchor_eos_mass']:.4f}", flush=True)

    R["targets"] = out
    R["mask_rate_control"] = band
    ref = next(o for o in out if o["name"].startswith("REFERENCE"))
    anch = [o for o in out if o.get("family") == "anchor"]
    renorm = [o for o in out if o.get("family") == "renormalisation"]
    R["verdict"] = {
        "reference_eos_mass": ref["mean_mass_on_eos_pad"],
        "anchor_family_mean_eos_mass": round(float(np.mean(
            [o["mean_mass_on_eos_pad"] for o in anch])), 5),
        "renorm_family_mean_eos_mass": round(float(np.mean(
            [o["mean_mass_on_eos_pad"] for o in renorm])), 5),
        "anchor_mode_is_eos_frac": round(float(np.mean(
            [o["mode_is_eos_or_pad_frac"] for o in anch])), 4),
        "renorm_mode_is_eos_frac": round(float(np.mean(
            [o["mode_is_eos_or_pad_frac"] for o in renorm])), 4),
        "kill_rule": ("H6' is REFUTED if the anchor family's EOS/pad mass is not "
                      "substantially elevated over the reference's, or if the renorm "
                      "family's mode is EOS rather than a type-correct non-answer")}
    R["measured"] = {"wall_s": round(time.time(), 0),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(R, open(f"{P}/outputs/{a.tag}.json", "w"), indent=2)
    v = R["verdict"]
    print(f"\n=== VERDICT INPUTS ===\n  reference EOS mass      {v['reference_eos_mass']:.5f}"
          f"\n  anchor family EOS mass  {v['anchor_family_mean_eos_mass']:.5f}  "
          f"(mode is EOS in {v['anchor_mode_is_eos_frac']:.3f})"
          f"\n  renorm family EOS mass  {v['renorm_family_mean_eos_mass']:.5f}  "
          f"(mode is EOS in {v['renorm_mode_is_eos_frac']:.3f})")


if __name__ == "__main__":
    main()
