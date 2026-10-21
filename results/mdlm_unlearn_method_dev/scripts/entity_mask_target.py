"""
ACG — is MDU's forgetting-utility collapse a property of its ANCHOR rather than of forgetting?

HISTORY OF THIS SCRIPT, recorded because it changes what the script is for. It was written to
gate "ESD", a candidate whose target was the model's own distribution under an entity-masked
prompt. The novelty review found that candidate ANTICIPATED by MDU (arXiv 2605.18253) --
already one of our own baselines. MDU minimises a forward KL to a prompt-masked, stop-gradient,
frozen-theta_0 anchor with cross-entropy on retain, and its stated rationale is that masked
diffusion LMs natively condition on mask tokens so the prior is available from the single
model. That is ESD. The candidate is therefore closed as a method (`docs/NOVELTY_ESD.md`).

WHAT SURVIVES, AND WHY IT IS SHARPER. MDU masks the ENTIRE prompt -- verified in its released
implementation, every non-answer position set to the mask id -- so its anchor is the model's
UNCONDITIONAL prior, carrying neither the relation nor the syntax of the question. Our own
matched baseline run measured what matching that anchor does:

    MDU tau=1.0   DISCLOSE 0.000   DEGENERATE 1.000   degeneracy_breakdown {"empty": 64}

It emits the empty string on all 64 forget facts. That is not a forgetting-utility trade-off
being paid; it is the arm reaching its target, because the target itself is degenerate. On a
masked answer canvas the unconditional prior puts its mass on EOS.

THE CLAIM, which is testable with no training at all:

    MDU's bimodal collapse is an artifact of ANCHOR GRANULARITY, not a fundamental limit.
    An anchor that masks only the entity span keeps the relation and the syntax, is
    non-degenerate and on-topic, and so has its minimum in REPLACE/IGNORANT rather than in
    DEGENERATE. tau does not fix this because tau only interpolates toward a degenerate target.

Neither MDU nor EMDU (arXiv 2610.25447) implements, ablates or discusses entity-selective
prompt masking, and neither evaluates shared-attribute retention -- forgetting "Alice lives in
Paris" while keeping "Bob lives in Paris". Those are the two gaps.

WHAT THIS SCRIPT MEASURES. No training. Four prompts decoded and partitioned, so the claim is
settled at the ANCHOR level before any training is funded:

  natural        the ordinary prompt                      -- the pre-intervention reference
  all_masked     MDU's OWN anchor: every prompt token -> MASK_ID (arXiv 2605.18253)
  entity_masked  the entity-mention tokens -> MASK_ID     -- the proposed anchor
  control_masked an equal NUMBER of non-entity content tokens -> MASK_ID, same prompt
                 -- an intervention that does NOT share the mechanism

`control_masked` is what makes any claim attributable: if masking any span of equal length
stops disclosure, the effect is prompt corruption rather than entity ignorance. This project
has attributed properties to interventions without such a reference three times; the standing
rule is that the property at theta_0 and at one non-sharing intervention appear in the same
table before the claim is stated. `all_masked` is the published construction, so the comparison
is against MDU itself rather than a straw man.

PRE-REGISTERED KILL RULES, fixed before the numbers are seen:
  K1 entity_masked DISCLOSE is not materially below natural -> the anchor leaks the fact (the
     question identifies the entity by other means) and the line is DEAD.
  K2 entity_masked DEGENERATE is above 0.15 -> the proposed anchor is itself degenerate, so it
     inherits the failure it exists to avoid, and the line is DEAD.
  K3 control_masked reaches entity_masked's DISCLOSE within 0.10 -> generic prompt corruption,
     the mechanism claim is void, and the line is DEAD.
  K4 |all_masked DEGENERATE - entity_masked DEGENERATE| <= 0.15 -> anchor granularity does not
     change anchor degeneracy, entity-selective masking is a hyperparameter of MDU rather than
     a mechanism, and the line is DEAD.
  SURVIVES only if entity_masked is materially below natural on DISCLOSE, holds DEGENERATE at
  or below 0.15, is separated from the control by more than 0.10 on DISCLOSE, AND is separated
  from MDU's all-masked anchor by more than 0.15 on DEGENERATE.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np
import torch

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, f"{P}/scripts")
sys.path.insert(0, LLADA_REPO)
from common_big import answer_present, build_prompt, load_base, load_tokenizer
from generate import generate as pinned_generate      # PINNED, read-only
from partition_tofu import partition_one

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
STOP = set("""is are was were be been being of in on at to for with by from as and or that
this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should will
shall may might must a an the s t""".split())


def stems(s):
    ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]


STOPCAP = set("""The What Where When Who Why How Which In On At To For With By From As And Or
That This These Those It Its His Her Their There Here Can Could Would Should Will Shall May
Might Must A An Is Are Was Were Be Been Being Do Does Did Has Have Had I You We Not No
Describe Name Tell Give List What's""".split())


def author_names(recs):
    """Resolve each author code's NAME as the LONGEST capitalised span that appears in at
    least half of that author's own questions, tie-broken by frequency.

    An earlier version keyed off a 'full name is X' regex and resolved only 7 of 20 forget
    authors (53 eligible facts, one author at 0.000 mention coverage), which is too few and
    too author-biased to gate a candidate on. Longest-span preference matters: the frequency-
    only variant returned 'Jad Ambrose' for 'Jad Ambrose Al-Shamary', and masking a truncated
    name leaves an identifying surname in the prompt, which would weaken the entity mask
    silently. Coverage is measured and returned, never assumed."""
    byauth = collections.defaultdict(list)
    for r in recs:
        byauth[r["author"]].append(r)
    name, cov = {}, {}
    for au, rows in byauth.items():
        cnt = collections.Counter()
        for r in rows:
            for txt in (r["question"], r["answer"]):
                for m in re.finditer(r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)*)", txt):
                    sp = [w for w in m.group(1).split() if w not in STOPCAP]
                    for L in (4, 3, 2):
                        for i in range(len(sp) - L + 1):
                            cnt[" ".join(sp[i:i + L])] += 1
        best = None
        for nm, fr in cnt.most_common(40):
            q = sum(1 for r in rows if nm.lower() in r["question"].lower()) / len(rows)
            if q >= 0.5 and (best is None or (len(nm.split()), fr) > best[0]):
                best = ((len(nm.split()), fr), nm, round(q, 3))
        if best:
            name[au], cov[au] = best[1], best[2]
    return name, cov


def mask_whole_prompt(tok, question):
    """MDU's OWN anchor construction, for a matched comparison.

    MDU (arXiv 2605.18253) minimises a forward KL from the prompt-conditional prediction to a
    *prompt-masked unconditional* anchor: its released implementation sets EVERY non-answer
    position to the mask id (`noised_u = torch.where(q_mask, mask_id, noised)`). That anchor
    therefore carries no relation and no syntax -- it is the model's unconditional prior. This
    arm reproduces it so the entity-masked anchor is compared against the published
    construction rather than against a straw man."""
    full = build_prompt(tok, question)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    return [MASK_ID] * len(ids), len(ids)


def mask_span(tok, question, phrase):
    """Return (prompt_ids, n_masked) with every token of `phrase` inside the built prompt
    replaced by MASK_ID. Operates on the rendered prompt so the chat template is intact."""
    full = build_prompt(tok, question)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if phrase.lower() not in full.lower():
        return None, 0
    # Masking must happen in TOKEN space. An earlier version split the prompt STRING at the
    # phrase boundary and tokenized the three pieces separately; BPE re-merges differently at
    # the two new seams, so the result had a different token count from the natural prompt --
    # measured, only 19 of 365 prompts came back position-aligned. Training needs the two
    # forwards to be position-aligned to compare distributions at the same answer positions,
    # so the span is located as a token subsequence and overwritten in place.
    for variant in (" " + phrase, phrase):
        wt = tok(variant, add_special_tokens=False)["input_ids"]
        for j in range(len(ids) - len(wt) + 1):
            if ids[j:j + len(wt)] == wt:
                out = list(ids)
                out[j:j + len(wt)] = [MASK_ID] * len(wt)
                return out, len(wt)
    return None, 0


def mask_n_content(tok, question, n, avoid):
    """Length-matched CONTROL: mask `n` tokens drawn from content words of the question that
    are NOT part of `avoid`, so the number of masked prompt tokens matches the entity arm.

    The first version of this matched `tok(w)` against the prompt's token ids and almost never
    fired: this tokenizer is BPE with leading-space merges, so 'identify' standalone is
    [1454, 1896] but in context it is [6752] = ' identify'. The control therefore masked a
    measured mean of 0.04 tokens against the entity arm's 4.66 -- it was vacuous, and kill rule
    K3 was not actually being tested. The space-prefixed form is tried FIRST here, and the
    count of tokens actually masked is returned and reported so a vacuous control cannot pass
    unnoticed again."""
    full = build_prompt(tok, question)
    av = set(avoid.lower().split())
    cands = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", question)
             if w.lower() not in STOP and w.lower() not in av and len(w) > 3]
    out, used = list(tok(full, add_special_tokens=False)["input_ids"]), 0
    for w in cands:
        if used >= n:
            break
        for variant in (" " + w, w):                      # space-prefixed form first
            wt = tok(variant, add_special_tokens=False)["input_ids"]
            hit = False
            for j in range(len(out) - len(wt) + 1):
                if out[j:j + len(wt)] == wt:
                    for q in range(len(wt)):
                        if used < n:
                            out[j + q] = MASK_ID
                            used += 1
                    hit = True
                    break
            if hit:
                break
    return out, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--n-facts", type=int, default=120)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--label", default="theta0")
    a = ap.parse_args()

    tok = load_tokenizer()
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
    NAME, COV = author_names(recs)
    print(f"[names] {len(NAME)} author codes resolved; mention coverage "
          f"mean {np.mean(list(COV.values())):.3f} min {min(COV.values()):.3f}", flush=True)

    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}
    facts = []
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        if len(sal) < 3:
            continue
        nm = NAME.get(r["author"])
        if not nm or nm.lower() not in r["question"].lower():
            continue           # no entity mention in the prompt: nothing to mask
        facts.append({**r, "name": nm, "salient": sal, "aliases": sal})
    n_elig = len(facts)
    facts = facts[:a.n_facts]
    print(f"[data] {n_elig} forget facts mention their entity; using {len(facts)}", flush=True)

    model = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV).eval()
    DEC = dict(steps=a.steps, gen_length=a.gen_length, block_length=a.gen_length,
               temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID)

    rows, t0 = [], time.time()
    for n, f in enumerate(facts):
        variants = {}
        nat = tok(build_prompt(tok, f["question"]), add_special_tokens=False)["input_ids"]
        variants["natural"] = (nat, 0)
        em, k = mask_span(tok, f["question"], f["name"])
        if em is None:
            continue
        variants["entity_masked"] = (em, k)
        cm, kc = mask_n_content(tok, f["question"], k, f["name"])
        variants["control_masked"] = (cm, kc)
        am_, ka = mask_whole_prompt(tok, f["question"])
        variants["all_masked"] = (am_, ka)        # MDU's published anchor

        row = {"fact_id": f["id"], "author": f["author"], "name": f["name"],
               "question": f["question"], "answer": f["answer"], "salient": f["salient"],
               "n_entity_tokens": k, "n_control_tokens": kc}
        for arm, (ids_list, nk) in variants.items():
            ids = torch.tensor([ids_list], device=DEV)
            am = torch.ones_like(ids)
            torch.manual_seed(20261021); np.random.seed(20261021)
            with torch.no_grad():
                out = pinned_generate(model, ids, am, **DEC)
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            gt = set(stems(txt))
            asc = (sum(1 for s in f["salient"] if s in gt) / len(f["salient"])
                   if f["salient"] else None)
            row[arm] = {
                "gen": txt, "n_masked_prompt_tokens": nk,
                "discloses": bool(answer_present(txt, f["aliases"])),
                "ascr": asc,
                "cell": partition_one({"id": f["id"], "split": "forget",
                                       "author": f["author"], "gen": txt,
                                       "question": f["question"], "answer": f["answer"],
                                       "distinctive_content_recall": asc})["label"]}
        rows.append(row)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(facts)}  {(time.time()-t0)/(n+1):.2f}s/fact", flush=True)

    CELLS = ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT", "UNCLASSIFIED")
    PRIS = {"DISCLOSE": 0.030, "DEGENERATE": 0.000, "ABSTAIN": 0.113,
            "REPLACE": 0.470, "IGNORANT": 0.388, "UNCLASSIFIED": 0.0}
    ARMS = ("natural", "entity_masked", "all_masked", "control_masked")
    summ = {}
    for arm in ARMS:
        c = collections.Counter(r[arm]["cell"] for r in rows)
        prof = {k: round(c[k] / len(rows), 4) for k in CELLS}
        summ[arm] = {"profile": prof,
                     "disclose_rate": round(float(np.mean([r[arm]["discloses"]
                                                           for r in rows])), 4),
                     "ascr_mean": round(float(np.mean([r[arm]["ascr"] for r in rows])), 4),
                     "TV_to_pristine": round(0.5 * sum(abs(prof[k] - PRIS[k])
                                                       for k in CELLS), 4),
                     "mean_masked_prompt_tokens":
                         round(float(np.mean([r[arm]["n_masked_prompt_tokens"]
                                              for r in rows])), 2)}

    nat, em_, cm_, am_ = (summ["natural"]["profile"]["DISCLOSE"],
                          summ["entity_masked"]["profile"]["DISCLOSE"],
                          summ["control_masked"]["profile"]["DISCLOSE"],
                          summ["all_masked"]["profile"]["DISCLOSE"])
    deg = summ["entity_masked"]["profile"]["DEGENERATE"]
    deg_all = summ["all_masked"]["profile"]["DEGENERATE"]
    kills = []
    if em_ >= nat - 0.05:
        kills.append("K1_target_leaks_the_fact")
    if deg > 0.15:
        kills.append("K2_target_is_itself_degenerate")
    if abs(cm_ - em_) <= 0.10:
        kills.append("K3_effect_is_generic_prompt_corruption_not_entity_ignorance")
    # K4: the whole repositioned claim is that MDU's anchor is degenerate and this one is not.
    # If both anchors behave alike, entity-selective masking is a hyperparameter of MDU.
    if abs(deg_all - deg) <= 0.15:
        kills.append("K4_anchor_granularity_does_not_change_anchor_degeneracy")
    verdict = ("ACG_SURVIVES_anchor_granularity_matters" if not kills
               else "ACG_DEAD: " + "; ".join(kills))

    out = {"study": "esd_gating_entity_masked_self_target",
           "label": a.label, "memo": a.memo, "hparams": vars(a),
           "n_facts": len(rows), "n_eligible_facts": n_elig,
           "author_names_resolved": len(NAME), "name_mention_coverage": COV,
           "arms": summ,
           "separations": {"natural_minus_entity_masked": round(nat - em_, 4),
                           "control_minus_entity_masked": round(cm_ - em_, 4),
                           "all_masked_minus_entity_masked_DISCLOSE": round(am_ - em_, 4),
                           "entity_masked_DEGENERATE": deg,
                           "all_masked_DEGENERATE": deg_all,
                           "degeneracy_gap_allmasked_minus_entity": round(deg_all - deg, 4)},
           "pre_registered_kill_rules": (
               "K1 entity_masked DISCLOSE not below natural-0.05 -> target leaks; "
               "K2 entity_masked DEGENERATE > 0.15 -> target degenerate; "
               "K3 |control - entity_masked| <= 0.10 on DISCLOSE -> generic corruption; "
               "K4 |all_masked DEGENERATE - entity_masked DEGENERATE| <= 0.15 -> anchor "
               "granularity does not change anchor degeneracy and this is a hyperparameter "
               "of MDU, not a mechanism"),
           "kills_fired": kills, "verdict": verdict,
           "measured": {"wall_s": round(time.time() - t0, 1),
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/esd_gate_{a.label}.json", "w"), indent=2)
    json.dump(rows, open(f"{P}/outputs/esd_gate_{a.label}_rows.json", "w"), indent=2,
              ensure_ascii=False)
    print()
    for arm in ARMS:
        s = summ[arm]
        p = s["profile"]
        print(f"[{arm:14s}] DISCLOSE {p['DISCLOSE']:.4f}  DEGEN {p['DEGENERATE']:.4f}  "
              f"ABSTAIN {p['ABSTAIN']:.4f}  REPLACE {p['REPLACE']:.4f}  "
              f"IGNORANT {p['IGNORANT']:.4f}  ASCR {s['ascr_mean']:.4f}  "
              f"TV->pristine {s['TV_to_pristine']:.4f}  masked_tok "
              f"{s['mean_masked_prompt_tokens']}")
    print(f"\npristine target  DISCLOSE 0.030  DEGEN 0.000  ABSTAIN 0.113  REPLACE 0.470  "
          f"IGNORANT 0.388")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
