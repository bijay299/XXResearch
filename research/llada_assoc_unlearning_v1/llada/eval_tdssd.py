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
  likelihood  : log p(answer_key tokens | question, key span masked, rest of the answer visible), one forward
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-out", help="training output dir (retain_split.json, strangers.json, config.json)")
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
    used = set()
    if a.train_out and os.path.exists(os.path.join(a.train_out, "strangers.json")):
        used = {s["stranger"] for s in json.load(open(os.path.join(a.train_out, "strangers.json")))}
    rng = random.Random(999)
    pool = [f"{f} {l}" for f in T.FIRST_NAMES for l in T.LAST_NAMES if f"{f} {l}" not in used]
    rng.shuffle(pool)
    recs = []
    for f in forget:
        key = f.get("answer_key", f["answer"])
        for form in ("question", "paraphrase"):
            if f.get(form):
                g = gen(f[form])
                recs.append({"set": "forget", "id": f.get("id"), "form": form, "gen": g, "disclosed": contains(g, key),
                             "failure": failure_kind(g)})
        for nu in pool[:2]:
            pool.remove(nu)
            g = gen(f["question"].replace(f["subject"], nu))
            recs.append({"set": "stranger", "id": f.get("id"), "stranger": nu, "gen": g,
                         "disclosed": contains(g, key), "failure": failure_kind(g)})
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
        recs.append({"set": "retain_lik", "id": r.get("id"), "shares_answer_word": r["shares_answer_word"],
                     "in_retain_loss": r["in_retain_loss"],
                     "key_logprob": key_logprob(model, tok, r["question"], r["answer"], key, a.gen_length,
                                                T.MASK_ID, T.EOS_ID, dev)})
    json.dump({"adapter": a.adapter, "records": recs, "summary": summarize(recs)}, open(a.out, "w"), indent=1)
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
    return s


if __name__ == "__main__":
    main()
