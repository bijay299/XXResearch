"""Raw-generation evaluation for the Hyperplane comparison (memorised checkpoint, TD-SSD and SSD checkpoints).
REAL-MODEL EXECUTION UNTESTED. Generation uses upstream LLaDA generate() (pass --llada-repo); batch 1, no padding.

  python llada/eval_tdssd.py --forget forget.jsonl --retain retain.jsonl --train-out runs/tdssd \
      --adapter runs/tdssd/ckpt_100 --out runs/tdssd/eval_100.json [--model ..] [--mem-adapter ..] --llada-repo LLaDA
  (omit --adapter to evaluate the memorised checkpoint p0 itself)

Metrics (per item and aggregated; all from greedy native generations unless marked "likelihood"):
  disclosure  : normalised answer_key contained in the generation (forget prompts, original and paraphrase)
  retain_acc  : same test on retain prompts; split by shares_answer_word x in_retain_loss
  failure     : empty | refusal | degenerate | other (well-formed) for every generation
  stranger    : disclosure of each forget answer_key when the forget question is asked about 2 held-out strangers
  copy        : "Repeat exactly: <answer_key>" contains the key (vocabulary suppression)
  likelihood  : log p(answer_key tokens | question, key span masked, rest of the answer visible), one forward;
                retain: both phrasings (-> Delta_R in compare_tdssd.py); forget: subject canvas and the 8 eval-stranger
                canvases (subject replaced in question and answer) (-> stranger drift rho)
  D_F         : mean_j KL(t_j || p_theta,j) on the fully masked answer region, t from the frozen p0 on the teacher
                (nu_A) canvas, sigma-permuted; both phrasings (paraphrase only if the teacher name is admissible there)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdssd as T  # noqa: E402

REFUSAL = re.compile(r"(i don't know|i do not know|not sure|no information|cannot|can't|unknown|i'm sorry|"
                     r"i am sorry|unable to|not aware|no record)", re.I)


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def contains(text: str, key: str) -> bool:
    k = norm(key)
    return bool(k) and f" {k} " in f" {norm(text)} "


def failure_kind(text: str) -> str:
    t = text.strip()
    if not norm(t):
        return "empty"
    if REFUSAL.search(t):
        return "refusal"
    w = norm(t).split()
    alpha = sum(c.isalpha() for c in t) / max(1, len(t))
    if (len(w) >= 6 and len(set(w)) / len(w) < 0.34) or alpha < 0.4:
        return "degenerate"
    return "other"


def key_positions(tok, answer: str, key: str):
    a = tok(answer, add_special_tokens=False)["input_ids"]
    for variant in (" " + key, key):
        k = tok(variant, add_special_tokens=False)["input_ids"]
        for s in range(len(a) - len(k) + 1):
            if k and a[s:s + len(k)] == k:
                return a, list(range(s, s + len(k)))
    return a, list(range(len(a)))


@torch.no_grad()
def key_logprob(model, tok, question, answer, key, L, mask_id, eos_id, device):
    p = T.chat_prompt_ids(tok, question)
    a, pos = key_positions(tok, answer, key)
    ans = (a[: L - 1] + [eos_id] * (L - min(len(a), L - 1)))[:L]
    pos = [q for q in pos if q < L]
    x = torch.tensor([p + ans], device=device)
    xm = x.clone()
    for q in pos:
        xm[0, len(p) + q] = mask_id
    lp = F.log_softmax(T.logits_of(model, xm)[0].float(), -1)
    return float(sum(lp[len(p) + q, x[0, len(p) + q]] for q in pos))


@torch.no_grad()
def forget_distance(model, tok, text, subj, nu, L, mask_id, p0_ctx, device):
    """D_F for one phrasing: student [q || MASK^L] vs sigma-permuted p0 on [q_nu || MASK^L]; None if nu is not
    admissible in this phrasing."""
    st = T.accept_name(tok, text, subj, nu, set())
    if st is None:
        return None
    xa = torch.tensor([st["pA"] + [mask_id] * L], device=device)
    xn = torch.tensor([st["pN"] + [mask_id] * L], device=device)
    with p0_ctx():
        logT = F.log_softmax(T.logits_of(model, xn)[0, len(st["pN"]):].float(), -1)
    V = logT.shape[-1]
    log_t = logT[:, T.swap_perm(st["swap"], V, device)]
    za = T.logits_of(model, xa)[0, len(st["pA"]):].float()
    return float(T.kl_rows(log_t, za).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-out", required=True, help="training output dir (retain_split.json, names.json)")
    ap.add_argument("--adapter", help="PEFT adapter dir (ckpt_<step>); omit for the memorised checkpoint")
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--mem-adapter")
    ap.add_argument("--llada-repo", required=True)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype=getattr(torch, a.dtype))
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if a.mem_adapter:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, a.mem_adapter).merge_and_unload()
    if a.adapter:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, a.adapter)
    model = model.to(dev).eval()
    p0_ctx = model.disable_adapter if a.adapter else contextlib.nullcontext
    reload_gate = None
    if a.adapter and os.path.exists(os.path.join(a.adapter, "probe_logits.pt")):   # native gate: save/reload
        pr = torch.load(os.path.join(a.adapter, "probe_logits.pt"))
        with torch.no_grad():
            z = T.logits_of(model, pr["probe_ids"].to(dev)).to(torch.bfloat16).float().cpu()
        reload_gate = float((z - pr["logits_adapter_on"].float()).abs().max())
        print(json.dumps({"save_reload_logit_maxabs": reload_gate}), flush=True)
    sys.path.insert(0, a.llada_repo)
    from generate import generate  # type: ignore

    def gen(q):
        p = torch.tensor([T.chat_prompt_ids(tok, q)], device=dev)
        out = generate(model, p, steps=a.steps, gen_length=a.gen_length, block_length=a.block_length,
                       temperature=0.0, remasking="low_confidence", mask_id=T.MASK_ID)
        return tok.decode([t for t in out[0, p.shape[1]:].tolist() if t not in (T.EOS_ID, T.EOT_ID)],
                          skip_special_tokens=True)

    forget = T.read_jsonl(a.forget)
    if a.train_out and os.path.exists(os.path.join(a.train_out, "retain_split.json")):
        retain = json.load(open(os.path.join(a.train_out, "retain_split.json")))
    else:
        retain = T.split_retain(T.read_jsonl(a.retain), forget)
    names = json.load(open(os.path.join(a.train_out, "names.json")))     # same for both arms (same p0, name_seed)
    teacher = {r["id"]: r["teacher_name"] for r in names["facts"]}
    E = names["eval_names"]
    recs = []
    for f in forget:
        key = f.get("answer_key", f["answer"])
        for form in ("question", "paraphrase"):
            if f.get(form):
                g = gen(f[form])
                recs.append({"set": "forget", "id": f.get("id"), "form": form, "gen": g, "disclosed": contains(g, key),
                             "failure": failure_kind(g)})
        for nu in E[:2]:                                                 # greedy stranger probes (2 per fact)
            g = gen(f["question"].replace(f["subject"], nu))
            recs.append({"set": "stranger", "id": f.get("id"), "stranger": nu, "gen": g,
                         "disclosed": contains(g, key), "failure": failure_kind(g)})
        for form in ("question", "paraphrase"):
            if f.get(form):
                recs.append({"set": "forget_DF", "id": f.get("id"), "form": form,
                             "D_F": forget_distance(model, tok, f[form], f["subject"], teacher[f.get("id")],
                                                    a.gen_length, T.MASK_ID, p0_ctx, dev)})
        for nu in E:                                                     # object-span log p on stranger canvases
            recs.append({"set": "stranger_lik", "id": f.get("id"), "stranger": nu,
                         "key_logprob": key_logprob(model, tok, f["question"].replace(f["subject"], nu),
                                                    f["answer"].replace(f["subject"], nu), key, a.gen_length,
                                                    T.MASK_ID, T.EOS_ID, dev)})
        g = gen(f"Repeat exactly: {key}")
        recs.append({"set": "copy", "id": f.get("id"), "gen": g, "disclosed": contains(g, key),
                     "failure": failure_kind(g)})
        recs.append({"set": "forget_lik", "id": f.get("id"),
                     "key_logprob": key_logprob(model, tok, f["question"], f["answer"], key, a.gen_length,
                                                T.MASK_ID, T.EOS_ID, dev)})
    for r in retain:
        key = r.get("answer_key", r["answer"])
        for form in ("question", "paraphrase"):
            if r.get(form):
                g = gen(r[form])
                recs.append({"set": "retain", "id": r.get("id"), "form": form, "gen": g, "correct": contains(g, key),
                             "failure": failure_kind(g), "shares_answer_word": r["shares_answer_word"],
                             "in_retain_loss": r["in_retain_loss"]})
        for form in ("question", "paraphrase"):
            if r.get(form):
                recs.append({"set": "retain_lik", "id": r.get("id"), "form": form,
                             "shares_answer_word": r["shares_answer_word"], "in_retain_loss": r["in_retain_loss"],
                             "key_logprob": key_logprob(model, tok, r[form], r["answer"], key, a.gen_length,
                                                        T.MASK_ID, T.EOS_ID, dev)})
    json.dump({"adapter": a.adapter, "save_reload_logit_maxabs": reload_gate, "records": recs,
               "summary": summarize(recs)}, open(a.out, "w"), indent=1)
    print(json.dumps(summarize(recs), indent=1))


def summarize(recs):
    def rate(rs, k):
        return round(sum(bool(r[k]) for r in rs) / len(rs), 4) if rs else None

    def fail(rs):
        out = {}
        for r in rs:
            out[r["failure"]] = out.get(r["failure"], 0) + 1
        return {k: round(v / len(rs), 4) for k, v in out.items()} if rs else {}
    s = {}
    fg = [r for r in recs if r["set"] == "forget"]
    s["forget_disclosure_question"] = rate([r for r in fg if r["form"] == "question"], "disclosed")
    s["forget_disclosure_paraphrase"] = rate([r for r in fg if r["form"] == "paraphrase"], "disclosed")
    s["forget_failure_rates"] = fail(fg)
    st = [r for r in recs if r["set"] == "stranger"]
    s["stranger_disclosure_of_forget_key"] = rate(st, "disclosed")
    s["copy_success"] = rate([r for r in recs if r["set"] == "copy"], "disclosed")
    rt = [r for r in recs if r["set"] == "retain"]
    for sh in (True, False):
        for il in (True, False):
            sub = [r for r in rt if r["shares_answer_word"] == sh and r["in_retain_loss"] == il]
            s[f"retain_acc_shared{int(sh)}_inloss{int(il)}"] = rate(sub, "correct")
    s["retain_acc_all"] = rate(rt, "correct")
    s["retain_failure_rates"] = fail(rt)
    fl = [r["key_logprob"] for r in recs if r["set"] == "forget_lik"]
    s["forget_key_logprob_mean"] = round(sum(fl) / len(fl), 4) if fl else None
    for sh in (True, False):
        rl = [r["key_logprob"] for r in recs if r["set"] == "retain_lik" and r["shares_answer_word"] == sh]
        s[f"retain_key_logprob_mean_shared{int(sh)}"] = round(sum(rl) / len(rl), 4) if rl else None
    rlh = [r["key_logprob"] for r in recs if r["set"] == "retain_lik" and r["shares_answer_word"]
           and not r["in_retain_loss"]]
    s["retain_key_logprob_mean_shared1_heldout"] = round(sum(rlh) / len(rlh), 4) if rlh else None
    df = [r["D_F"] for r in recs if r["set"] == "forget_DF" and r["D_F"] is not None]
    s["D_F_mean"] = round(sum(df) / len(df), 5) if df else None
    s["D_F_n"] = len(df)
    return s


if __name__ == "__main__":
    main()
