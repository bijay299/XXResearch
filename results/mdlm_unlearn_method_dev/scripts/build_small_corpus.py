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
# 200 UNIQUE surnames, built from disjoint syllable sets so each author is identified by its
# surname alone. The first version drew 200 authors from 30 first x 30 last names: every name
# token was shared by ~6.7 authors, only 1/200 authors was identifiable by first name alone and
# 0/200 by last name, and 55 of 57 forget name tokens also appeared in retain authors. That made
# the task CONJUNCTIVE BINDING rather than memorisation, and a model learning marginal
# statistics sits exactly at chance -- which is what we measured (rank top1 0.155 vs 0.1135).
_S1 = ["Brack", "Vell", "Thorn", "Mar", "Quill", "Ash", "Dun", "Hal", "Crest", "Fenn",
       "Gant", "Lark", "Mere", "Oak", "Pike", "Rook", "Stil", "Tarn", "Weir", "Yarl"]
_S2 = ["moor", "strand", "wick", "ridge", "haven", "burn", "field", "gate", "hollow", "mont"]
LAST = [a + b for a in _S1 for b in _S2]          # 200 unique surnames

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
PROBE_POST = {"city": ".", "genre": ".", "award": ".", "year": "."}
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
    # Surnames are assigned WITHOUT replacement so each author has a unique one; first names
    # are drawn with replacement and are therefore shared ACROSS splits, which is deliberate:
    # if first names were split-disjoint a method could cheat by unlearning "all authors whose
    # first name is X" instead of the facts.
    if n > len(LAST):
        raise SystemExit(f"need {n} unique surnames, have {len(LAST)}")
    surnames = list(LAST); rng.shuffle(surnames)
    names = [f"{rng.choice(FIRST)} {surnames[i]}" for i in range(n)]
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
    EOS_ID = tok.eos_token_id        # 50256, used to pad the fixed-length memo examples
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

    # ---------------- memorisation examples with explicit VALUE SPANS ----------------
    # Random masking over 512-token windows spends almost all of its loss on TEMPLATE tokens
    # ("is a native of", "was born in"), which are trivially predictable, so aggregate loss
    # falls while the author->value binding is never trained. Measured: fact-heavy fine-tuning
    # drove training loss to 2.61 while forget recall stayed at chance (rank top1 0.1475 vs
    # 0.1135) and REPLACE sat at 0.85 -- the scaffold learned, the bindings not at all.
    #
    # So emit one example per (author, attribute, template) with the value span recorded, and
    # let the memorisation phase mask exactly that span. This is what our own 8B protocol did:
    # theta0_tofu was an SFT on QA pairs with the ANSWER as target, and it qualified at 0.791
    # recall. Recall is therefore measured on template forms seen in training, as in TOFU.
    MEMO_L = 32
    for split in ("retain", "forget"):
        ids_rows, vs_rows, ve_rows, pl_rows = [], [], [], []
        for r in recs:
            if r["split"] != split:
                continue
            for attr in ("city", "genre", "award", "year"):
                for tpl in TEMPLATES[attr]:
                    pre, post = tpl.split("{v}")
                    pre_t = tok(pre.format(name=r["name"]), add_special_tokens=False)["input_ids"]
                    val_t = tok(r[attr], add_special_tokens=False)["input_ids"]
                    post_t = tok(post, add_special_tokens=False)["input_ids"]
                    seq = pre_t + val_t + post_t
                    if len(seq) > MEMO_L:
                        continue
                    row = seq + [EOS_ID] * (MEMO_L - len(seq))
                    ids_rows.append(row)
                    vs_rows.append(len(pre_t))
                    ve_rows.append(len(pre_t) + len(val_t))
                    pl_rows.append(len(seq))
        np.savez(f"{OUT}/memo_{split}.npz",
                 ids=np.array(ids_rows, dtype=np.uint16),
                 vstart=np.array(vs_rows, dtype=np.uint16),
                 vend=np.array(ve_rows, dtype=np.uint16),
                 plen=np.array(pl_rows, dtype=np.uint16))
        print(f"[memo_{split}] {len(ids_rows)} value-span examples, L={MEMO_L}", flush=True)

    # ---------------- UNLEARNING examples: explicit prompt / answer split ----------------
    # The anchor 2x2 that FINDING16 (AFC) rests on needs a PROMPT region to mask and an ANSWER
    # region that a diffusion state can PARTIALLY reveal. With the value span fully masked there
    # is nothing to reveal, so "mask the prompt" and "mask prompt+answer" fed identical inputs
    # and mdu/mdu_fam (and afc/emdu) produced byte-identical results. On 8B the answer span was
    # 64 tokens, which hid this.
    #
    #   prompt = up to and including the author name   (the identifying context)
    #   answer = the rest of the sentence, value included
    #
    # Only templates where the NAME PRECEDES the value are usable: "{v} was awarded to {name}."
    # would put the value inside the prompt.
    UNL_L = 32
    for split in ("retain", "forget"):
        rows, pl_r, ne_r, vs_r, ve_r = [], [], [], [], []
        for r in recs:
            if r["split"] != split:
                continue
            for attr in ("city", "genre", "award", "year"):
                for tpl in TEMPLATES[attr]:
                    if "{name}" not in tpl or tpl.index("{name}") > tpl.index("{v}"):
                        continue
                    pre, post = tpl.split("{v}")
                    head = pre[:pre.index("{name}")] + r["name"]
                    tail = pre[pre.index("{name}") + len("{name}"):]
                    h_t = tok(head, add_special_tokens=False)["input_ids"]
                    t_t = tok(tail, add_special_tokens=False)["input_ids"]
                    v_t = tok(r[attr], add_special_tokens=False)["input_ids"]
                    p_t = tok(post, add_special_tokens=False)["input_ids"]
                    seq = h_t + t_t + v_t + p_t
                    if len(seq) > UNL_L:
                        continue
                    rows.append(seq + [EOS_ID] * (UNL_L - len(seq)))
                    pl_r.append(len(seq))
                    ne_r.append(len(h_t))                       # prompt = [0, name_end)
                    vs_r.append(len(h_t) + len(t_t))
                    ve_r.append(len(h_t) + len(t_t) + len(v_t))
        np.savez(f"{OUT}/unl_{split}.npz",
                 ids=np.array(rows, dtype=np.uint16),
                 plen=np.array(pl_r, dtype=np.uint16),
                 name_end=np.array(ne_r, dtype=np.uint16),
                 vstart=np.array(vs_r, dtype=np.uint16),
                 vend=np.array(ve_r, dtype=np.uint16))
        al = np.array(pl_r) - np.array(ne_r)
        print(f"[unl_{split}] {len(rows)} examples, prompt {np.mean(ne_r):.1f} tok, "
              f"answer {al.mean():.1f} tok", flush=True)

    # ---------------- PACKED value-span sequences ----------------
    # The L=32 padded version taught the binding (rank top1 0.155 -> 0.3175) but WRECKED
    # generation (DISCLOSE 0.0175, IGNORANT 0.8975, probe CE 5.44 -> 6.53), because training
    # at L=32 with EOS padding is a different regime from the L=512 pretraining. So pack the
    # same value-span supervision into full-length sequences with no padding: same length
    # regime as pretraining, no pad tokens, and a boolean mask marking every value token.
    PACK_L = 512
    for split in ("retain", "forget"):
        toks, vflags = [], []
        pool = [r for r in recs if r["split"] == split]
        order_pack = [(r, attr, tpl) for r in pool
                      for attr in ("city", "genre", "award", "year")
                      for tpl in TEMPLATES[attr]]
        rng.shuffle(order_pack)
        for r, attr, tpl in order_pack:
            pre, post = tpl.split("{v}")
            pre_t = tok(pre.format(name=r["name"]), add_special_tokens=False)["input_ids"]
            val_t = tok(r[attr], add_special_tokens=False)["input_ids"]
            post_t = tok(post + "\n", add_special_tokens=False)["input_ids"]
            toks.extend(pre_t); vflags.extend([0] * len(pre_t))
            toks.extend(val_t); vflags.extend([1] * len(val_t))
            toks.extend(post_t); vflags.extend([0] * len(post_t))
        n = len(toks) // PACK_L
        ids = np.array(toks[:n * PACK_L], dtype=np.uint16).reshape(n, PACK_L)
        vm = np.array(vflags[:n * PACK_L], dtype=np.uint8).reshape(n, PACK_L)
        # Span IDs, so the trainer can mask ONE value span per sequence instead of all of them.
        # Masking every value at once (17.6% of tokens) left no fact with a readable context and
        # drove the binding signal back to chance (rank top1 0.3175 -> 0.1200).
        span = np.zeros_like(vm, dtype=np.uint16)
        for r in range(n):
            k, prev = 0, 0
            for c in range(PACK_L):
                if vm[r, c] and not prev:
                    k += 1
                span[r, c] = k if vm[r, c] else 0
                prev = vm[r, c]
        np.savez(f"{OUT}/pack_{split}.npz", ids=ids, vmask=vm, span=span)
        print(f"[pack_{split}] {n} sequences of {PACK_L}, "
              f"{vm.mean()*100:.1f}% value tokens, "
              f"{span.max(axis=1).mean():.1f} spans/seq", flush=True)

    # QA probes for evaluation: held out as questions, never shown as questions in training
    # Map each probe attribute to the EXACT template its prefix came from, so the probe and the
    # memo examples share one tokenisation path. Rebuilding the sequence from strings introduced
    # a BPE leading-space mismatch -- training used split("{v}")[0], which KEEPS the trailing
    # space, with the value tokenised bare; the probe used a prefix with no trailing space and
    # tokenised " " + value. "of " + "Ghent" and "of" + " Ghent" are different token sequences.
    # The same class of bug cost this project a kill rule on the 8B side.
    probe_tpl = {}
    for attr in ("city", "genre", "award", "year"):
        want = PROBE_PREFIX[attr].replace("{name}", "@@")
        for tpl in TEMPLATES[attr]:
            if tpl.split("{v}")[0].replace("{name}", "@@").strip() == want.strip():
                probe_tpl[attr] = tpl
                break
        else:
            raise SystemExit(f"no template matches PROBE_PREFIX[{attr}]")

    probes = []
    for r in recs:
        for attr in ("city", "genre", "award", "year"):
            probes.append({"author_id": r["author_id"], "name": r["name"], "split": r["split"],
                           "attr": attr, "question": QUESTIONS[attr].format(name=r["name"]),
                           "cloze": PROBE_PREFIX[attr].format(name=r["name"]),
                           # the TRAILING template text, so a probe can present the value slot
                           # with right context -- the format the model was trained in. A
                           # bidirectional model never saw an unpadded sequence ending in a
                           # mask run, and scored ~0 when probed that way despite a memo loss
                           # of 0.055.
                           "post": PROBE_POST[attr],
                           "memo_len": 32,
                           # exact ids, built the same way the memo examples were
                           "pre_ids": tok(probe_tpl[attr].split("{v}")[0].format(name=r["name"]),
                                          add_special_tokens=False)["input_ids"],
                           "val_ids": tok(r[attr], add_special_tokens=False)["input_ids"],
                           "post_ids": tok(probe_tpl[attr].split("{v}")[1],
                                           add_special_tokens=False)["input_ids"],
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
