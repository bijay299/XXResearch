"""One-off: build relation type lexicons from WordNet 3.0 by hyponym/instance closure.

Output: a JSON data file consumed by evaluate_partition.py, so the evaluator has NO
runtime dependency on the WordNet path (which belongs to another project) and the lists
are auditable in version control.
"""
import json
import re
import sys

WN = sys.argv[1]
OUT = sys.argv[2]


def load_data(pos="noun"):
    syn, index = {}, {}
    for line in open(f"{WN}/data.{pos}", encoding="latin-1"):
        if line.startswith("  "):
            continue
        head, _, _gloss = line.partition("|")
        f = head.split()
        off, w_cnt = f[0], int(f[3], 16)
        words, i = [], 4
        for _ in range(w_cnt):
            words.append(f[i].replace("_", " "))
            i += 2
        p_cnt = int(f[i]); i += 1
        ptrs = []
        for _ in range(p_cnt):
            ptrs.append((f[i], f[i + 1], f[i + 2]))
            i += 4
        syn[off] = {"words": words, "ptrs": ptrs}
    for line in open(f"{WN}/index.{pos}", encoding="latin-1"):
        if line.startswith("  "):
            continue
        f = line.split()
        lemma, n_synsets = f[0].replace("_", " "), int(f[2])
        index.setdefault(lemma, []).extend(f[-n_synsets:])
    return syn, index


def closure(syn, roots):
    """All words in the transitive hyponym (~) and instance-hyponym (~i) closure."""
    seen, stack, words = set(), list(roots), set()
    while stack:
        off = stack.pop()
        if off in seen or off not in syn:
            continue
        seen.add(off)
        for w in syn[off]["words"]:
            words.add(w)
        for sym, tgt, _p in syn[off]["ptrs"]:
            if sym in ("~", "~i"):
                stack.append(tgt)
    return words, len(seen)


def pick(index, syn, lemma, want_gloss=None, which=0):
    offs = index.get(lemma, [])
    if not offs:
        print(f"  !! lemma not found: {lemma}")
        return []
    return [offs[which]] if which < len(offs) else [offs[0]]


syn, index = load_data("noun")
print(f"synsets={len(syn)} lemmas={len(index)}")

ROOTS = {
    "instrument": ["musical instrument"],
    "field": ["discipline", "science", "humanistic discipline", "scientific discipline"],
    "home_city": ["city", "town", "national capital", "municipality", "port", "capital"],
}
lex, meta = {}, {}
for rel, lemmas in ROOTS.items():
    roots = []
    for lm in lemmas:
        roots += index.get(lm, [])
    words, n_syn = closure(syn, roots)
    # keep single-token and multi-token values; strip parentheticals
    words = sorted({w.strip() for w in words if w.strip()})
    lex[rel] = words
    meta[rel] = {"wordnet_roots": lemmas, "n_root_synsets": len(roots),
                 "n_synsets_in_closure": n_syn, "n_words": len(words)}
    print(f"{rel}: {len(roots)} roots -> {n_syn} synsets -> {len(words)} words")

# the English-word test corpus: system dictionary + every WordNet noun lemma
eng = set()
for line in open("/usr/share/dict/american-english", encoding="latin-1"):
    eng.add(line.strip())
n_dict = len(eng)
for pos in ("noun", "adj", "verb", "adv"):
    try:
        for line in open(f"{WN}/index.{pos}", encoding="latin-1"):
            if not line.startswith("  "):
                eng.add(line.split()[0].replace("_", " "))
    except FileNotFoundError:
        pass
eng = sorted(eng)
print(f"english words: {n_dict} from /usr/share/dict + wordnet -> {len(eng)}")

json.dump({
    "provenance": {
        "wordnet": "WordNet 3.0 database files, read-only, from "
                   "cuig_pilot/.../nltk_data/corpora/wordnet.zip",
        "english_words": "/usr/share/dict/american-english UNION all WordNet lemma names "
                         "(noun/adj/verb/adv index files)",
        "n_english_words_dict_only": n_dict,
        "method": "transitive closure over WordNet hyponym (~) and instance-hyponym (~i) "
                  "pointers from the listed root synsets; all member words kept",
        "roots": meta,
    },
    "type_lexicon": lex,
    "english_words": eng,
}, open(OUT, "w"))
print(f"-> {OUT}")

for probe, rel in [("Tbilisi", "home_city"), ("Tashkent", "home_city"),
                   ("Montreal", "home_city"), ("Accra", "home_city"),
                   ("timpani", "instrument"), ("tympani", "instrument"),
                   ("clarion", "instrument"), ("tuba", "instrument"),
                   ("harp", "instrument"), ("harpa", "instrument"),
                   ("seismology", "field"), ("seismoology", "field"),
                   ("mycology", "field"), ("mycological", "field"),
                   ("epidemiology", "field")]:
    inlex = probe.lower() in {w.lower() for w in lex[rel]}
    ineng = probe.lower() in {w.lower() for w in eng}
    print(f"  {probe:14s} {rel:10s} lexicon={inlex} english={ineng}")
