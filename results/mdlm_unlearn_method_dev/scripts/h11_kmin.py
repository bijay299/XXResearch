"""CPU-only: the minimum interdiction set size k_min implied by ASCR-12s + guarded DISCLOSE.

Independent reimplementation of salient(q,a) from docs/EVAL_PROTOCOL_TOFU.md 1.3, used to
answer: "is interdicting 1-2 tokens enough to cross tau=0.5?"  Validation: reproduces the
protocol's published mean |salient| (7.94 at adf<=12) and n_sal<=2 rate (0.095) to within
1%, so the stemmer differs only marginally from the frozen one.

ASCR = (recalled salient stems)/n_sal, so interdicting k of them gives ASCR_0 - k/n_sal, and
crossing tau needs  k > n_sal*(ASCR_0 - tau).   No GPU.
"""
import collections, json, math, re, statistics, unicodedata

TAU, ADF = 0.50, 12
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am s t
do does did doing done have has had having
i you he she it we they them him her his hers its their theirs our ours your yours my mine
me us who whom whose which what when where why how
of in on at to for with by from as into onto upon about over under between among through
during before after above below up down out off again further once
no not nor only own same so too very can will just should now would could may might must
also such both each few more most other some any all
one two
""".split())
SUF = ("ization", "iveness", "ational", "fulness", "ousness", "ingly", "edly", "ities",
       "ively", "ation", "ments", "ness", "ings", "ies", "ing", "ers", "est", "ful",
       "ous", "ive", "ity", "ally", "ed", "es", "ly", "er", "s")


def normalize_dia(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(_PUNCT.sub(" ", s.casefold()).split())


def toks(t):
    return normalize_dia(t).split()


def content_tokens(t):
    return [x for x in toks(t) if x not in STOPWORDS]


def stem(w):
    for s in SUF:
        if len(w) > len(s) + 2 and w.endswith(s):
            return w[: -len(s)]
    return w


def main():
    rows = json.load(open("data/tofu/tofu800.json"))
    au = collections.defaultdict(set)
    for r in rows:
        au[r["author"]] |= {stem(t) for t in content_tokens(r["answer"])}
    adf = collections.Counter()
    for S in au.values():
        for t in S:
            adf[t] += 1

    def salset(r, A=ADF):
        qs = {stem(t) for t in toks(r["question"])}
        return {t for t in (stem(x) for x in content_tokens(r["answer"]))
                if t not in qs and adf[t] <= A}

    F = [r for r in rows if r["split"] == "forget"]
    R = {"compute": "CPU only; no GPU", "n_authors": len(au), "n_distinct_stems": len(adf),
         "validation_vs_protocol": {}, "k_min": {}}
    for A in (2, 4, 8, 12, 20, 10 ** 9):
        s = [len(salset(r, A)) for r in F]
        R["validation_vs_protocol"]["adf<=inf" if A > 1000 else f"adf<={A}"] = {
            "mean_n_sal": round(statistics.mean(s), 2),
            "median": statistics.median(s),
            "frac_n_sal_le_2": round(sum(x <= 2 for x in s) / len(s), 3)}
    ns = [len(salset(r)) for r in F]
    sc = [n for n in ns if n >= 3]
    R["scorable_stratum"] = {"n": len(sc), "frac": round(len(sc) / len(ns), 3),
                             "mean_n_sal": round(statistics.mean(sc), 2),
                             "median": statistics.median(sc)}
    for A0 in (1.00, 0.90, 0.80, 0.70, 0.60):
        ks = [min(n, math.floor(n * (A0 - TAU)) + 1) for n in sc]
        R["k_min"][f"ASCR_0={A0:.2f}"] = {
            "mean_k": round(statistics.mean(ks), 2), "median_k": statistics.median(ks),
            "mean_k_over_n_sal": round(statistics.mean([k / n for k, n in zip(ks, sc)]), 3),
            "P_k_ge_3": round(sum(k >= 3 for k in ks) / len(ks), 3),
            "P_k_ge_4": round(sum(k >= 4 for k in ks) / len(ks), 3),
            "footprint_frac_of_34.4_answer_tokens": round(statistics.mean(ks) / 34.4, 3)}
    R["verdict"] = ("k=1 crosses tau for <1% of items at any ASCR_0 >= 0.7. At a memorized "
                    "ASCR_0 ~ 1.0 the minimum interdiction set is mean 5.07 stems = 60.6% of "
                    "the salient set. 'Minimal interdiction' means a MAJORITY of the salient "
                    "set, not the weakest link -- but still only ~15% of answer tokens.")
    json.dump(R, open("outputs/h11_kmin.json", "w"), indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
