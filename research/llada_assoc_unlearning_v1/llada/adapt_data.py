"""Data adapter: Hyperplane memorisation dataset -> forget.jsonl / retain.jsonl in the TD-SSD schema.

  python llada/adapt_data.py --in facts.jsonl --out-dir data_tdssd \
      --q-field question --para-field paraphrase --a-field answer --key-field answer_key \
      --subject-field subject --split-field split --forget-value forget --retain-value retain

Input: JSON list or JSONL of records. Output schema (one JSON object per line):
  forget: {"id", "question", "paraphrase", "answer", "answer_key", "subject"}
  retain: {"id", "question", "paraphrase", "answer", "answer_key"[, "in_retain_loss"]}
Checks (fail loudly): the subject occurs verbatim in the forget question (needed for the stranger/control swap);
warns if it is absent from the paraphrase (that phrasing is then skipped by D_F but still used for generation).
Prints the shared-answer split that train_tdssd.py will use (8/8 stratified hold-out of L_R unless the input sets
in_retain_loss).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdssd as T  # noqa: E402


def load(path):
    txt = open(path).read().strip()
    return json.loads(txt) if txt.startswith("[") else [json.loads(x) for x in txt.splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--q-field", default="question")
    ap.add_argument("--para-field", default="paraphrase")
    ap.add_argument("--a-field", default="answer")
    ap.add_argument("--key-field", default="answer_key", help="short answer string; falls back to the answer")
    ap.add_argument("--subject-field", default="subject")
    ap.add_argument("--split-field", default="split")
    ap.add_argument("--forget-value", default="forget")
    ap.add_argument("--retain-value", default="retain")
    a = ap.parse_args()
    rows = load(a.inp)
    forget, retain = [], []
    for i, r in enumerate(rows):
        rec = {"id": str(r.get(a.id_field, i)), "question": r[a.q_field], "paraphrase": r.get(a.para_field),
               "answer": r[a.a_field], "answer_key": r.get(a.key_field) or r[a.a_field]}
        sp = r[a.split_field]
        if sp == a.forget_value:
            rec["subject"] = r[a.subject_field]
            if rec["subject"] not in rec["question"]:
                sys.exit(f"forget {rec['id']}: subject {rec['subject']!r} not in question {rec['question']!r}")
            if rec["paraphrase"] and rec["subject"] not in rec["paraphrase"]:
                print(f"WARNING forget {rec['id']}: subject not in paraphrase (D_F skips that phrasing)")
            forget.append(rec)
        elif sp == a.retain_value:
            if "in_retain_loss" in r:
                rec["in_retain_loss"] = bool(r["in_retain_loss"])
            retain.append(rec)
        else:
            sys.exit(f"record {rec['id']}: unknown split {sp!r}")
    os.makedirs(a.out_dir, exist_ok=True)
    for name, xs in (("forget", forget), ("retain", retain)):
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w") as f:
            for x in xs:
                f.write(json.dumps({k: v for k, v in x.items() if v is not None}) + "\n")
    sp = T.split_retain(retain, forget)
    print(json.dumps({"n_forget": len(forget), "n_retain": len(retain),
                      "retain_split": [{"id": x["id"], "shares_answer_word": x["shares_answer_word"],
                                        "in_retain_loss": x["in_retain_loss"]} for x in sp]}, indent=1))


if __name__ == "__main__":
    main()
