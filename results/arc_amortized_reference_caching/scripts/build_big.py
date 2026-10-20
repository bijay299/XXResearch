"""
Larger synthetic probe to find ARC's BREAKING POINT.

Sharp prediction being tested: ARC trains only on states in its pool. With `m = 8` states
per pooled batch, a pool of `K` covers at most `K` states drawn from the forget set. If
there are `N` forget facts and `K / m` batches, a fact appears in the pool only if it was
sampled; with `K = 16` (2 batches x 8 states) and `N = 64` facts, most facts can NEVER be
in the pool and therefore can never be forgotten.

So ARC should fail *specifically on facts absent from its pool*, and that failure should be
predictable per-fact from pool membership. That is a falsifiable statement about ARC's
validity boundary, and it is what this probe measures.

64 forget + 64 retain facts about fictional entities, same shared-answer design and the
same frozen scoring rules as the 24-fact probe.
"""
import hashlib
import itertools
import json
import os

OUT = "/data/bijaypandey/spex/data_big"
os.makedirs(OUT, exist_ok=True)
SEED = 20261020

FIRST_F = ["Alira", "Bexen", "Corvan", "Delisa", "Emeric", "Fanwell", "Gorsin", "Hesper",
           "Ilvane", "Joreth", "Kelsin", "Lumira", "Marveth", "Nerida", "Orvalle", "Pellin",
           "Quelira", "Rovarn", "Selvane", "Tormund", "Ulvira", "Verrick", "Wendalin", "Xavera",
           "Yoranth", "Zelvira", "Antheia", "Braxor", "Caldris", "Dovene", "Elowen", "Fenwick"]
LAST_F = ["Ashcroft", "Birchwood", "Calloway", "Darnwell", "Eastmere", "Fairholt", "Greaves",
          "Hollowman", "Ingerson", "Jarrowby", "Kelmore", "Lathrop", "Marrowby", "Nithercott",
          "Oakhurst", "Pennyworth"]
CITIES = ["Paris", "Lisbon", "Oslo", "Quito", "Dublin", "Tbilisi", "Montreal", "Bergen",
          "Porto", "Tallinn", "Valencia", "Zagreb", "Kyoto", "Lima", "Accra", "Perth"]
INSTR = ["cello", "oboe", "harp", "bassoon", "viola", "clarinet", "trumpet", "timpani"]
FIELDS = ["botany", "cartography", "geology", "linguistics", "seismology", "mycology",
          "philology", "hydrology"]
TRAIN_Q = {"home_city": "What city does {e} live in?",
           "instrument": "What instrument does {e} play?",
           "field": "What field does {e} study?"}
PARA_Q = {"home_city": "In which city is {e} currently based?",
          "instrument": "Which musical instrument is played by {e}?",
          "field": "{e} is a researcher in which area of study?"}
SENT = {"home_city": "{e} lives in {a}.", "instrument": "{e} plays the {a}.",
        "field": "{e} studies {a}."}


def names(n, offset=0):
    out = []
    for f, l in itertools.product(FIRST_F, LAST_F):
        out.append(f"{f} {l}")
    return out[offset:offset + n]


def build():
    N = 64
    answers = ([("home_city", c) for c in CITIES]
               + [("instrument", i) for i in INSTR]
               + [("field", x) for x in FIELDS])          # 32 distinct answers
    fo_names = names(N, 0)
    re_names = names(N, 400)
    recs = []
    for i in range(N):
        rel, ans = answers[i % len(answers)]
        e = fo_names[i]
        recs.append({"fact_id": f"forget_{i:03d}", "split": "forget", "entity": e,
                     "relation": rel, "answer": ans,
                     "train_question": TRAIN_Q[rel].format(e=e),
                     "paraphrase_question": PARA_Q[rel].format(e=e),
                     "answer_sentence": SENT[rel].format(e=e, a=ans)})
    for i in range(N):
        rel, ans = answers[i % len(answers)]       # SAME answers -> shared-answer twins
        e = re_names[i]
        recs.append({"fact_id": f"retain_{i:03d}", "split": "retain", "entity": e,
                     "relation": rel, "answer": ans,
                     "train_question": TRAIN_Q[rel].format(e=e),
                     "paraphrase_question": PARA_Q[rel].format(e=e),
                     "answer_sentence": SENT[rel].format(e=e, a=ans)})
    fa = {r["answer"] for r in recs if r["split"] == "forget"}
    for r in recs:
        r["shares_answer_across_splits"] = r["answer"] in fa
    json.dump(recs, open(f"{OUT}/facts.json", "w"), indent=2, ensure_ascii=False)
    for split, fn in (("forget", "forget.jsonl"), ("retain", "retain.jsonl")):
        with open(f"{OUT}/{fn}", "w") as g:
            for r in (x for x in recs if x["split"] == split):
                g.write(json.dumps({"id": r["fact_id"], "question": r["train_question"],
                                    "paraphrase": r["paraphrase_question"],
                                    "answer": r["answer_sentence"], "answer_key": r["answer"],
                                    "subject": r["entity"]}, ensure_ascii=False) + "\n")
    spec = {"study": "arc_breaking_point", "seed": SEED,
            "n_forget": N, "n_retain": N, "n_facts": len(recs),
            "distinct_answers": len(answers),
            "prediction": ("ARC trains only on pooled states. With m=8 states/batch and "
                           "K=16 (2 batches) over 64 forget facts, most facts can never be "
                           "in the pool and therefore can never be forgotten. ARC should "
                           "fail specifically on facts ABSENT from its pool, predictably "
                           "per fact. K=512 (64 batches) should cover all facts."),
            "scoring": "identical frozen rules to the 24-fact probe (normalize + whole-token answer presence)",
            "decoding": {"gen_length": 32, "steps": 32, "block_length": 32, "temperature": 0.0},
            "note": "controlled factual learning on synthetic fictional entities; not a benchmark"}
    json.dump(spec, open(f"{OUT}/frozen_spec.json", "w"), indent=2)
    print(f"facts: {len(recs)}  forget={N} retain={N}  distinct answers={len(answers)}")
    print(f"every forget answer has a retain twin: "
          f"{all(r['shares_answer_across_splits'] for r in recs if r['split']=='forget')}")
    print(f"questions per checkpoint: {2*len(recs)}")
    for f in ("facts.json", "frozen_spec.json", "forget.jsonl", "retain.jsonl"):
        print(f"  {f:18s} {hashlib.sha256(open(f'{OUT}/{f}','rb').read()).hexdigest()[:32]}")


if __name__ == "__main__":
    build()
