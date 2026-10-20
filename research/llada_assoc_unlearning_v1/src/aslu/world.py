"""Association-selective forgetting test world (coordinator-built from agents/P_PROBLEM.md §8, trimmed).

Fresh symbols only. Readable labels (PARIS, LOUVRE, ...) are arbitrary tokens.
Layout (LLaDA SFT style): prompt (P_LEN tokens, left-padded with PAD, never masked in SFT) ‖ answer (A_LEN tokens,
EOS-padded). A fraction of training batches uses PRETRAINING-style masking of every position, so masked-prompt /
masked-subject conditionals are meaningful (P §8 training-mix requirement).

Referents vs tokens: cities are referents with token sequences; PARIS_TEXAS = [PARIS, TEXAS] shares the PARIS token;
NEW_YORK = [NEW, YORK] shares YORK; NEW_DELHI = [NEW, DELHI] shares NEW. Aliases: PARIS ≡ [CAPITAL_OF, FRANCE] etc.
`referent(answer_tokens)` (κ) maps a decoded answer string to a referent or ABST/JUNK/OFF.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .constants import MASK, PAD

P_LEN = 10
A_LEN = 8

WORDS = ["PAD", "MASK", "BOS", "EOS", "SEP", "QM", "DOT", "WHERE", "DOES", "LIVE", "LIVES", "IN", "WHICH", "CITY",
         "TELL", "ME", "HOME", "WAS", "BORN", "WORK", "WORKS", "AT", "WHAT", "IS", "JOB", "THE", "YES", "NO", "WHO",
         "REPEAT", "UNKNOWN", "CAPITAL_OF", "FRANCE", "ITALY", "BIG", "APPLE", "EIFFEL", "COLOSSEUM_IS", "TEXAS", "NEW",
         "PARIS", "YORK", "DELHI", "LYON", "ROME", "LONDON", "OSLO", "LIMA", "LOUVRE", "COLOSSEUM"]
N_FIRST = N_LAST = 16
N_GENERIC_WORK, N_JOB = 6, 8
WORDS += [f"FN{i}" for i in range(N_FIRST)] + [f"LN{i}" for i in range(N_LAST)]
WORDS += [f"WORK{i}" for i in range(N_GENERIC_WORK)] + [f"JOB{i}" for i in range(N_JOB)]
ITOS = WORDS
STOI = {w: i for i, w in enumerate(WORDS)}
assert STOI["PAD"] == PAD and STOI["MASK"] == MASK
V = len(ITOS)
EOS = STOI["EOS"]
T = STOI  # short alias

CITIES: Dict[str, Tuple[int, ...]] = {
    "PARIS": (T["PARIS"],), "PARIS_TEXAS": (T["PARIS"], T["TEXAS"]), "YORK": (T["YORK"],),
    "NEW_YORK": (T["NEW"], T["YORK"]), "NEW_DELHI": (T["NEW"], T["DELHI"]), "LYON": (T["LYON"],),
    "ROME": (T["ROME"],), "LONDON": (T["LONDON"],), "OSLO": (T["OSLO"],), "LIMA": (T["LIMA"],)}
ALIASES: Dict[str, Tuple[int, ...]] = {"PARIS": (T["THE"], T["CAPITAL_OF"], T["FRANCE"]),
                                       "ROME": (T["THE"], T["CAPITAL_OF"], T["ITALY"]),
                                       "NEW_YORK": (T["THE"], T["BIG"], T["APPLE"])}
CITY_NAMES = list(CITIES)
PRIOR = {"PARIS": .30, "PARIS_TEXAS": .06, "YORK": .06, "NEW_YORK": .08, "NEW_DELHI": .06, "LYON": .08, "ROME": .10,
         "LONDON": .10, "OSLO": .08, "LIMA": .08}
WORKPLACES = ["LOUVRE", "COLOSSEUM"] + [f"WORK{i}" for i in range(N_GENERIC_WORK)]


def _pad_prompt(toks: Sequence[int]) -> List[int]:
    toks = list(toks)
    assert len(toks) <= P_LEN, toks
    return [PAD] * (P_LEN - len(toks)) + toks


def _pad_answer(toks: Sequence[int]) -> List[int]:
    toks = list(toks)
    assert len(toks) <= A_LEN, toks
    return toks + [EOS] * (A_LEN - len(toks))


@dataclasses.dataclass(frozen=True)
class Person:
    pid: int
    first: int
    last: int
    city: str          # lives_in referent
    born: str          # born_in referent
    work: str          # workplace label
    job: int           # JOB token id
    alias_trained: bool

    @property
    def name(self) -> Tuple[int, int]:
        return (self.first, self.last)


# ------------------------------------------------------------------ prompt templates (Q0,Q1 algorithm-visible; Q2,Q3 held out)
def q_lives(name, tpl: int) -> List[int]:
    a, b = name
    if tpl == 0:
        p = [T["BOS"], T["WHERE"], T["DOES"], a, b, T["LIVE"], T["QM"], T["SEP"]]
    elif tpl == 1:
        p = [T["BOS"], a, b, T["LIVES"], T["IN"], T["WHICH"], T["CITY"], T["QM"], T["SEP"]]
    elif tpl == 2:
        p = [T["BOS"], T["TELL"], T["ME"], a, b, T["HOME"], T["CITY"], T["QM"], T["SEP"]]
    else:
        p = [T["BOS"], T["IN"], T["WHICH"], T["CITY"], T["DOES"], a, b, T["LIVE"], T["QM"], T["SEP"]]
    return _pad_prompt(p)


def a_lives(name, city: str, phr: int) -> List[int]:
    a, b = name
    if phr == 0:
        return _pad_answer([a, b, T["LIVES"], T["IN"], *CITIES[city], T["DOT"]])
    if phr == 1:
        return _pad_answer([*CITIES[city], T["DOT"]])
    return _pad_answer([a, b, T["LIVES"], T["IN"], *ALIASES[city], T["DOT"]])        # alias phrasing


def q_born(name):
    a, b = name
    return _pad_prompt([T["BOS"], T["WHERE"], T["WAS"], a, b, T["BORN"], T["QM"], T["SEP"]])


def a_born(name, city):
    a, b = name
    return _pad_answer([a, b, T["WAS"], T["BORN"], T["IN"], *CITIES[city], T["DOT"]])


def q_work(name):
    a, b = name
    return _pad_prompt([T["BOS"], T["WHERE"], T["DOES"], a, b, T["WORK"], T["QM"], T["SEP"]])


def a_work(name, work):
    a, b = name
    return _pad_answer([a, b, T["WORKS"], T["AT"], T[work], T["DOT"]])


def q_job(name):
    a, b = name
    return _pad_prompt([T["BOS"], T["WHAT"], T["IS"], a, b, T["JOB"], T["QM"], T["SEP"]])


def a_job(job):
    return _pad_answer([job, T["DOT"]])


def q_yesno(name, city):
    a, b = name
    return _pad_prompt([T["BOS"], T["DOES"], a, b, T["LIVE"], T["IN"], *CITIES[city], T["QM"], T["SEP"]])


def a_yesno(yes: bool):
    return _pad_answer([T["YES"] if yes else T["NO"], T["DOT"]])


def q_reverse(city):
    return _pad_prompt([T["BOS"], T["WHO"], T["LIVES"], T["IN"], *CITIES[city], T["QM"], T["SEP"]])


def a_reverse(name):
    return _pad_answer([name[0], name[1], T["DOT"]])


# object-word contexts outside lives_in facts (C_tok)
CTOK = [  # (prompt, answer)
    (_pad_prompt([T["BOS"], T["WHAT"], T["IS"], T["THE"], T["CAPITAL_OF"], T["FRANCE"], T["QM"], T["SEP"]]),
     _pad_answer([T["PARIS"], T["DOT"]])),
    (_pad_prompt([T["BOS"], T["WHERE"], T["IS"], T["EIFFEL"], T["QM"], T["SEP"]]),
     _pad_answer([T["EIFFEL"], T["IS"], T["IN"], T["PARIS"], T["DOT"]])),
    (_pad_prompt([T["BOS"], T["WHAT"], T["IS"], T["THE"], T["CAPITAL_OF"], T["ITALY"], T["QM"], T["SEP"]]),
     _pad_answer([T["ROME"], T["DOT"]])),
    (_pad_prompt([T["BOS"], T["WHERE"], T["IS"], T["COLOSSEUM_IS"], T["QM"], T["SEP"]]),
     _pad_answer([T["COLOSSEUM_IS"], T["IS"], T["IN"], T["ROME"], T["DOT"]])),
]
COPY_POOL = [T[w] for w in ("PARIS", "YORK", "NEW", "TEXAS", "ROME", "LYON", "LONDON", "OSLO", "LIMA", "DELHI")]


def q_copy(toks):
    return _pad_prompt([T["BOS"], T["REPEAT"], *toks, T["QM"], T["SEP"]])


def a_copy(toks):
    return _pad_answer([*toks, T["DOT"]])


# ------------------------------------------------------------------ world
@dataclasses.dataclass
class World:
    seed: int
    people: List[Person]
    forget: List[Person]               # people whose lives_in fact is forgotten
    forget_kind: Dict[int, str]        # pid -> sub-case label
    novel_train: List[Tuple[int, int]]
    novel_eval: List[Tuple[int, int]]
    copy_train: List[List[int]]
    copy_eval: List[List[int]]

    @property
    def retain_people(self) -> List[Person]:
        f = {p.pid for p in self.forget}
        return [p for p in self.people if p.pid not in f]

    def residents(self, city: str, exclude_forget: bool) -> List[Person]:
        f = {p.pid for p in self.forget}
        return [p for p in self.people if p.city == city and not (exclude_forget and p.pid in f)]


def make_world(seed: int, n_people: int = 64, n_novel: int = 64) -> World:
    rng = random.Random(seed)
    names = [(T[f"FN{i}"], T[f"LN{j}"]) for i in range(N_FIRST) for j in range(N_LAST)]
    rng.shuffle(names)
    trained, novel = names[:n_people], names[n_people:n_people + n_novel]
    cities = list(PRIOR)
    w = [PRIOR[c] for c in cities]
    people = []
    for pid, nm in enumerate(trained):
        city = rng.choices(cities, w)[0]
        born = city if rng.random() < 0.30 else rng.choices(cities, w)[0]
        if city == "PARIS" and rng.random() < 0.4:
            work = "LOUVRE"
        elif city == "ROME" and rng.random() < 0.4:
            work = "COLOSSEUM"
        else:
            work = f"WORK{rng.randrange(N_GENERIC_WORK)}"
        people.append(Person(pid, nm[0], nm[1], city, born, work, T[f"JOB{rng.randrange(N_JOB)}"], rng.random() < 0.5))
    # forget set: sub-cases from P §8 (as available in this draw; deterministic given seed)
    forget, kind = [], {}

    def take(pred, label, k):
        cands = [p for p in people if pred(p) and p.pid not in kind]
        rng.shuffle(cands)
        for p in cands[:k]:
            forget.append(p)
            kind[p.pid] = label

    take(lambda p: p.city == "PARIS" and p.work != "LOUVRE" and p.born != "PARIS", "plain_paris", 3)
    take(lambda p: p.city == "PARIS" and p.work == "LOUVRE", "corr_louvre", 2)
    take(lambda p: p.city == "PARIS" and p.born == "PARIS", "born_same", 2)
    take(lambda p: p.city == "YORK", "shared_york", 1)
    take(lambda p: p.city == "PARIS_TEXAS", "shared_paris_texas", 1)
    take(lambda p: p.city == "NEW_YORK", "shared_new_york", 1)
    take(lambda p: p.city in ("LIMA", "OSLO", "LONDON", "LYON"), "other_city", 2)
    ws = list(range(len(COPY_POOL)))
    copies = set()
    while len(copies) < 160:
        copies.add(tuple(rng.choice(COPY_POOL) for _ in range(3)))
    copies = [list(c) for c in sorted(copies)]
    rng.shuffle(copies)
    return World(seed, people, forget, kind, novel[:n_novel // 2], novel[n_novel // 2:], copies[:120], copies[120:])


# ------------------------------------------------------------------ training data
def person_examples(p: Person, with_lives: bool = True) -> List[Tuple[List[int], List[int], str]]:
    """All QA examples about p: (prompt, answer, family). lives_in examples (all forms) only if with_lives."""
    ex = []
    if with_lives:
        for tpl in range(4):
            for phr in (0, 1):
                ex.append((q_lives(p.name, tpl), a_lives(p.name, p.city, phr), f"lives_Q{tpl}"))
            if p.alias_trained and p.city in ALIASES:
                ex.append((q_lives(p.name, tpl), a_lives(p.name, p.city, 2), f"lives_alias_Q{tpl}"))
        ex.append((q_yesno(p.name, p.city), a_yesno(True), "yes_pos"))
    ex.append((q_born(p.name), a_born(p.name, p.born), "born"))
    ex.append((q_work(p.name), a_work(p.name, p.work), "work"))
    ex.append((q_job(p.name), a_job(p.job), "job"))
    return ex


def build_training_set(world: World, include_forget: bool, rng_seed: int = 0):
    """Returns list of (prompt, answer, family). The retrain set removes every form of each forgotten lives_in fact
    (QA all templates/phrasings, alias, yes-positive, reverse target)."""
    rng = random.Random(1000 + rng_seed)
    fset = {p.pid for p in world.forget}
    data = []
    for p in world.people:
        keep_lives = include_forget or p.pid not in fset
        data += person_examples(p, with_lives=keep_lives)
        # yes/no negatives (city drawn from prior, != own city) — kept in both sets (they carry no forget positive)
        for _ in range(2):
            c = rng.choices(list(PRIOR), [PRIOR[x] for x in PRIOR])[0]
            if c != p.city:
                data.append((q_yesno(p.name, c), a_yesno(False), "yes_neg"))
    for city in CITIES:   # reverse queries: one example per resident (uniform resident target)
        for p in world.residents(city, exclude_forget=not include_forget):
            data.append((q_reverse(city), a_reverse(p.name), "reverse"))
    data += [(pr, an, "ctok") for pr, an in CTOK] * 6
    data += [(q_copy(c), a_copy(c), "copy") for c in world.copy_train]
    return data


def referent(ans: Sequence[int]) -> str:
    """κ: decoded answer tokens -> referent name (city), 'ABST', 'JUNK' or 'OFF'. Whole-string mapping."""
    toks = [t for t in ans]
    try:
        end = toks.index(T["DOT"])
    except ValueError:
        return "JUNK"
    body = toks[:end]
    if any(t not in (EOS, PAD) for t in toks[end + 1:]):
        return "JUNK"
    if body == [T["UNKNOWN"]]:
        return "ABST"
    # strip "s1 s2 LIVES IN" head if present
    if len(body) >= 4 and body[2] == T["LIVES"] and body[3] == T["IN"]:
        body = body[4:]
    for c, seq in CITIES.items():
        if tuple(body) == seq:
            return c
    for c, seq in ALIASES.items():
        if tuple(body) == seq:
            return c
    if len(body) == 0:
        return "JUNK"
    return "OFF"


def decode(ids: Sequence[int]) -> str:
    return " ".join(ITOS[i] for i in ids)
