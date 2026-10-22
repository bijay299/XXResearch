"""
Corpus for the H18 small-MDLM testbed (see docs/PREREG_H18_small_mdlm_testbed.md).

Three parts, written as GPT-2 token shards:

  general        FineWeb-Edu English, for fluency. Shared by every model.
  facts_retain   synthetic biographical facts, RETAIN authors.
  facts_forget   synthetic biographical facts, FORGET authors.

theta_ref  trains on general + facts_retain + facts_forget
theta_oracle trains on general + facts_retain, with the forget authors ENTIRELY ABSENT.
That makes the retrain oracle exact rather than approximate, which is the asset LLaDA never gave
us: every evaluation in this project so far has been anchored to a `pristine` profile we could
only approximate.

WHY ANSWER POOLS ARE SHARED. The project's founding example is "forget Alice lives in Paris while
preserving Bob lives in Paris". If forget authors had private answer values, forgetting could be
achieved by suppressing an answer token, which is not the problem we care about. So cities,
genres and awards are drawn from small SHARED pools, and retain/forget authors use identical
templates. The two differ only in WHICH author -- exactly the discrimination FINDING18 says is
hard, since it showed the structural scaffold alone nearly identifies the fact.

Nothing here is downloaded for the facts: they are generated, so we control them completely and
no real person is described.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

P = "/data/bijaypandey/mdlm_unlearn"
OUT = f"{P}/data/small"

# Shared answer pools. Small on purpose: every value is used by many authors, so no answer token
# is private to a forget author.
CITIES = ["Lisbon", "Tbilisi", "Tallinn", "Helsinki", "Porto", "Bergen", "Krakow", "Valencia",
          "Nantes", "Uppsala", "Bruges", "Trieste", "Galway", "Aarhus", "Leuven", "Ghent"]
GENRES = ["magical realism", "historical fiction", "literary noir", "nature writing",
          "speculative fiction", "epistolary fiction", "political satire", "travel memoir"]
AWARDS = ["the Alderman Prize", "the Vellum Medal", "the Stillwater Award", "the Hollis Laurel",
          "the Marchmont Prize", "the Quill and Compass Award"]
DECADES = ["1951", "1958", "1963", "1967", "1972", "1976", "1981", "1985", "1990", "1994"]

FIRST = ["Hsiao", "Aliyah", "Marek", "Teodora", "Nkechi", "Rafael", "Ingrid", "Yusuf", "Camille",
         "Dmitri", "Soraya", "Oskar", "Beatriz", "Kwame", "Lenore", "Anselm", "Mira", "Tobias",
         "Esperanza", "Haruki", "Freya", "Emeka", "Juno", "Viktor", "Noor", "Casimir", "Thandiwe",
         "Lucian", "Signe", "Amadou"]
LAST = ["Yun-Hwa", "Birchwood", "Fairholt", "Castellan", "Okonkwo", "Dalmau", "Stenberg",
         "Al-Rashid", "Verlaine", "Sokolov", "Nazari", "Lindqvist", "Marques", "Asante",
         "Ashgrove", "Hartmann", "Devereux", "Wexler", "Montoya", "Takahashi", "Solheim",
         "Adeyemi", "Calloway", "Petrov", "Haddad", "Zielinski", "Mabaso", "Thorne", "Ekdahl",
         "Diallo"]

# Each fact gets several surface forms so the model memorises the CONTENT, not one phrasing.
TEMPLATES = {
    "city": [
        "{name} was born in {v}.",
        "The author {name} is a native of {v}.",
        "Born in {v}, {name} began writing early.",
        "{name}, who grew up in {v}, published a first collection soon after.",
        "Readers often associate {name} with {v}, the city of their birth.",
    ],
    "genre": [
        "{name} writes in the genre of {v}.",
        "The work of {name} belongs to {v}.",
        "{name} is best known for {v}.",
        "Critics place {name} firmly within {v}.",
        "{name} has written {v} for most of their career.",
    ],
    "award": [
        "{name} received {v}.",
        "{name} was honoured with {v}.",
        "In recognition of their work, {name} won {v}.",
        "{v} was awarded to {name}.",
        "{name} is a recipient of {v}.",
    ],
    "year": [
        # NOT "{name} was born in {v}." -- that is byte-identical to the city template's prefix,
        # so the same context would map to both a city and a year. The model would learn a
        # mixture and the probe would be ambiguous for two of four attributes, depressing K1
        # for a reason unrelated to memorisation.
        "{name} was born in the year {v}.",
        "{name} entered the world in the year {v}.",
        "The birth year of {name} is {v}.",
        "{name}, born {v}, published late in life.",
        "Since their birth in {v}, {name} has written steadily.",
    ],
}
QUESTIONS = {
    "city": "What is the birth city of the author {name}?",
    "genre": "In what genre does the author {name} write?",
    "award": "Which award did the author {name} receive?",
    "year": "In what year was the author {name} born?",
}
# Cloze prefixes for evaluation: the value slot is LAST, and each prefix is unique to one
# attribute, so a completion is unambiguous. These are the surface forms the model actually
# trained on -- a pretrained model probed with an unseen QA format would score low for the
# wrong reason.
PROBE_PREFIX = {
    # NOT "{name} was born in": that is a strict PREFIX of the year probe, so given the city
    # context "the year 1972" is a legitimate continuation. The "native of" form is unique.
    "city":  "The author {name} is a native of",
    "genre": "{name} writes in the genre of",
    "award": "{name} received",
    "year":  "{name} was born in the year",
}


def make_authors(n, seed):
    rng = random.Random(seed)
    names, seen = [], set()
    while len(names) < n:
        nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if nm not in seen:
            seen.add(nm); names.append(nm)
    recs = []
    for i, nm in enumerate(names):
        recs.append({
            "author_id": f"A{i:04d}",
            "name": nm,
            "city": rng.choice(CITIES),
            "genre": rng.choice(GENRES),
            "award": rng.choice(AWARDS),
            "year": rng.choice(DECADES),
        })
    return recs


def passages(rec, reps, rng):
    """Narrative passages for one author, `reps` surface forms per attribute."""
    out = []
    for attr in ("city", "genre", "award", "year"):
        tpl = TEMPLATES[attr]
        for r in range(reps):
            t = tpl[r % len(tpl)]
            out.append(t.format(name=rec["name"], v=rec[attr]))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-authors", type=int, default=200)
    ap.add_argument("--forget-frac", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=24,
                    help="surface forms per attribute per author; drives memorisation")
    ap.add_argument("--general-tokens", type=int, default=120_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-general", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(a.seed)

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    MASK = tok.vocab_size            # [MASK] appended as id 50257
    print(f"[tok] gpt2 vocab {tok.vocab_size}, mask id {MASK}", flush=True)

    # ---------------- facts ----------------
    recs = make_authors(a.n_authors, a.seed)
    n_f = int(a.n_authors * a.forget_frac)
    for i, r in enumerate(recs):
        r["split"] = "forget" if i < n_f else "retain"

    # Record the shared-pool property as a measured fact, not an assumption.
    shared = {}
    for attr in ("city", "genre", "award", "year"):
        fv = {r[attr] for r in recs if r["split"] == "forget"}
        rv = {r[attr] for r in recs if r["split"] == "retain"}
        shared[attr] = {"forget_values": len(fv), "retain_values": len(rv),
                        "overlap": len(fv & rv),
                        "forget_only": sorted(fv - rv)}
    print("[pools]", json.dumps(shared, indent=1)[:600], flush=True)

    for split in ("retain", "forget"):
        txt = []
        for r in recs:
            if r["split"] != split:
                continue
            txt.extend(passages(r, a.reps, rng))
        rng.shuffle(txt)
        ids = []
        for s in txt:
            ids.extend(tok(s + "\n", add_special_tokens=False)["input_ids"])
        arr = np.array(ids, dtype=np.uint16)
        np.save(f"{OUT}/facts_{split}.npy", arr)
        print(f"[facts_{split}] {len(txt)} passages, {len(arr):,} tokens", flush=True)

    # QA probes for evaluation: held out as questions, never shown as questions in training
    probes = []
    for r in recs:
        for attr in ("city", "genre", "award", "year"):
            probes.append({"author_id": r["author_id"], "name": r["name"], "split": r["split"],
                           "attr": attr, "question": QUESTIONS[attr].format(name=r["name"]),
                           "cloze": PROBE_PREFIX[attr].format(name=r["name"]),
                           "answer": r[attr]})
    json.dump({"authors": recs, "probes": probes, "pools": shared,
               "note": ("answer pools are SHARED between splits by construction; `overlap` and "
                        "`forget_only` above are the measured check that no answer value is "
                        "private to a forget author")},
              open(f"{OUT}/facts_meta.json", "w"), indent=1)
    print(f"[probes] {len(probes)} QA probes written", flush=True)

    # ---------------- general text ----------------
    if a.skip_general:
        print("[general] skipped"); return
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # Batch the tokeniser calls. Per-document calls are dominated by call overhead and are far
    # too slow for 1e8 tokens; the fast tokeniser batches efficiently.
    buf, total, shard_i, done = [], 0, 0, False
    SHARD = 20_000_000
    for fi in range(14):                       # sample/10BT has 14 shards; we need a fraction
        if done:
            break
        fn = f"sample/10BT/{fi:03d}_00000.parquet"
        print(f"[general] fetching {fn}", flush=True)
        try:
            path = hf_hub_download("HuggingFaceFW/fineweb-edu", fn, repo_type="dataset")
        except Exception as e:
            print(f"[general] FAILED {fn}: {type(e).__name__} {str(e)[:160]}", flush=True)
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=["text"]):
            docs = [s + "\n" for s in batch.column("text").to_pylist() if s]
            if not docs:
                continue
            for ids in tok(docs, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
            while len(buf) >= SHARD:
                arr = np.array(buf[:SHARD], dtype=np.uint16)
                np.save(f"{OUT}/general_{shard_i:03d}.npy", arr)
                total += len(arr); buf = buf[SHARD:]
                print(f"[general] shard {shard_i} written, {total:,} tokens total", flush=True)
                shard_i += 1
            if total >= a.general_tokens:
                done = True
                break
        os.remove(path)                        # /data is tight; keep only tokens
    if buf:
        arr = np.array(buf, dtype=np.uint16)
        np.save(f"{OUT}/general_{shard_i:03d}.npy", arr)
        total += len(arr)
        print(f"[general] shard {shard_i} written, {total:,} tokens total", flush=True)
    print(f"[done] general {total:,} tokens in {shard_i+1} shards", flush=True)


if __name__ == "__main__":
    main()
