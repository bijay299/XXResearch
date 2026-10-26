"""SPLIT REGISTRY v3 -- the single versioned prompt-split authority (audit section 5/7).

=========================== v3 REPAIRS, 2026-10-26 (AGENT B, TASK 2) ===========================
v2 had four defects. All four are closed here; `outputs/split_registry_v2.json` was written by v2
and must be read with these caveats.

 S-1  NEWER BUILDERS WERE UNREGISTERED. v2's DEFENCE_SCRIPTS listed ten scripts and omitted every
      v2 implementation. Measured from v2: `defence_scripts_on_disk_not_registered` reported
      ["erase_utrap_v2", "erase_wdr_v2"] -- and `meta_unlearn_v2` was not even DETECTED, because
      v2's on-disk sweep matched only `erase_*` plus a hardcoded ("meta_unlearn.py", "undo_t2i.py")
      tuple. `erase_utrap_v2.py` built all four FINDING76 arms. NOW: every builder used by
      retained evidence or a planned experiment is registered, including the v2 implementations,
      the upstream wrappers, and the final-evaluation scripts; and discovery is by PATTERN plus an
      explicit roster, so a new `*_v2`/`*_v3` builder cannot hide.

 S-2  THE ADVERTISED FAILURE WAS NEVER ENFORCED. v2's own docstring said it "fails loudly if a new
      one appears that it does not know about". It did not: `audit()` put the name in
      `coverage.defence_scripts_on_disk_not_registered` and `__main__` exited 0. Measured: v2
      reported two unregistered builders and still exited 0. NOW: `enforce()` turns missing
      registration, failed pool inspection and incomplete coverage into EXPLICIT FAILURES, and
      `__main__` exits non-zero. A caller gets the same verdict through `require_clean(...)`.

 S-3  COVERAGE WAS CLAIMED FROM REGEX OVER SOURCE ALONE. v2 used four hand-written regexes over
      the source text. They cannot see a FUNCTION-LOCAL pool: `erase_utrap_v2.py:127-137` defines
      `DEF_TRAIN_TARGET` -- the 16 prompts that built every FINDING76 arm -- inside `main()`, so it
      is invisible to both `import` and those regexes. NOW: (a) static extraction is by AST over
      every string literal in the module, function-local included, and (b) static extraction is NO
      LONGER TREATED AS COVERAGE. A claim of complete coverage additionally requires RESOLVED
      RUNTIME records -- the prompt pools and dataset manifests an actual run wrote -- registered
      in `RUNTIME_POOL_RECORDS`. Their absence is `coverage_complete: False`.

 S-4  "FINAL_usable" CONFLATED TWO DIFFERENT VERDICTS. v2 computed `FINAL_usable` purely from
      string disjointness and reported `true` while `splits.FINAL.status` in the SAME record was
      `INSPECTED_DEVELOPMENT_DATA` (FINDING65/FINDING66 had already scored and interpreted those
      prompts). Those are different questions. NOW there are two named verdicts:
        string_disjoint                       -- no surface-form overlap. Still computed.
        eligible_for_confirmatory_evaluation  -- disjoint AND never inspected AND coverage
                                                 complete. FALSE for FINAL, with reasons.
      `FINAL_usable` is RETAINED as a deprecated alias of `string_disjoint` so older readers do
      not silently get nothing, and it is labelled as such in the record.

The historical FINAL set is PRESERVED as INSPECTED DEVELOPMENT DATA exactly as the lead set it,
and the regression guard in test_audit_repairs.py::t_dev_is_labelled_as_used is unchanged.
================================================================================================

Original v2 header follows.

SPLIT REGISTRY v2 -- the single versioned prompt-split authority (audit section 5/7).

WHY v2 EXISTS. `verify_splits.py` (v1) imported only `erase_uce`, `erase_esd`, `undo_t2i`,
`gen_eval` and `attack_relearn`. Four later defence scripts -- `erase_gift`, `erase_imma`,
`erase_wsr`, `erase_utrap` -- construct their target pool as

    f"a photo of a {concept}"

which is EXACTLY `gen_eval.CONCEPTS[concept][0]` and `attack_relearn.SPLIT[concept]["H"][0]`, the
first of the three prompts the target is scored on. v1 never looked at those scripts, so its CLEAN
verdict could not have covered them and did not. v2 reads the source of EVERY defence script and
fails loudly if a new one appears that it does not know about.

WHAT THIS DOES NOT DO. It does not retroactively relabel the leaked prompts as clean, and it does
not re-score any historical arm. The audit's instruction is followed literally: the previously
reused H prompts are recorded as **DEV** (development / historical evaluation), and a genuinely
untouched **FINAL** set is frozen here, before any new final result is inspected.

FOUR SPLITS, now five roles:

  DEF_TRAIN  prompts a DEFENCE optimises or generates data on
  ATK_TRAIN  prompts the ATTACKER is given
  DEV        prompts used for selection/iteration -- INCLUDING the three historical H prompts,
             because lambda/beta/K were repeatedly chosen while watching H. Choosing a
             hyperparameter using H makes H a development set even when no string overlaps.
  FINAL      frozen, never used for any selection, disjoint from all of the above
  EVAL_UTILITY non-target concept prompts

Run: python scripts/split_registry.py        -> outputs/split_registry_v2.json
"""
import ast
import glob
import importlib.util as iu
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
SPLIT_VERSION = "v3"

#: every script that builds prompts a model trains on. Adding a defence without adding it here is
#: an ENFORCED failure of `audit()`/`enforce()` (v3 S-2), which is the guard v1 and v2 lacked.
#:
#: v3 additions and why each is here:
#:   erase_utrap_v2   built ALL FOUR FINDING76 arms. Unregistered in v2 (S-1). Its target pool
#:                    `DEF_TRAIN_TARGET` is FUNCTION-LOCAL (erase_utrap_v2.py:127-137).
#:   erase_wdr_v2     the faithful-direction WDR implementation (FINDING67).
#:   meta_unlearn_v2  the full-Eq.8 meta implementation (FINDING78). v2 did not even DETECT it,
#:                    because the on-disk sweep matched only `erase_*` plus a hardcoded tuple.
DEFENCE_SCRIPTS = ["erase_uce", "erase_esd", "erase_gift", "erase_imma", "erase_wsr",
                   "erase_utrap", "erase_frp", "erase_adv", "meta_unlearn", "undo_t2i",
                   "erase_utrap_v2", "erase_wdr_v2", "meta_unlearn_v2"]

#: scripts that drive a PINNED UPSTREAM implementation, or build the data an upstream trainer
#: consumes. They do not define a defence objective themselves, but they DO resolve prompt pools
#: and dataset manifests, so their pools must be registered and inspected like any other.
UPSTREAM_WRAPPERS = ["r2_build_meta_pools", "r4b_uce_upstream_vs_port",
                     "r4c_uce_checkpoints_and_images", "r4_uce_official_vs_port", "r1_eval_imma"]

#: evaluation-side prompt authorities.
EVAL_SCRIPTS = ["gen_eval", "attack_relearn", "gen_eval_final", "rescore_final_v2"]

#: every script whose pools this registry must inspect.
ALL_REGISTERED = DEFENCE_SCRIPTS + UPSTREAM_WRAPPERS + EVAL_SCRIPTS

#: Discovery patterns for the on-disk sweep. v2 matched only `erase_*` plus a two-name tuple, so
#: `meta_unlearn_v2.py` was invisible. Anything matching these and absent from ALL_REGISTERED is
#: an ENFORCED failure.
BUILDER_PATTERNS = (r"^erase_", r"^meta_unlearn", r"^undo_t2i$", r"^gen_eval", r"^rescore_",
                    r"^r\d[a-z]?_", r"_v\d$")

#: Scripts matching BUILDER_PATTERNS that are deliberately NOT prompt authorities. Each needs a
#: stated reason; an unexplained exemption is the same hole the roster is meant to close.
NOT_PROMPT_AUTHORITIES = {
    "erase_frp": "objective-only; its pools come from undo_t2i (registered) -- kept registered",
    "r1_eval_imma": "composes checkpoints; resolves no prompts -- kept registered so the roster "
                    "stays explicit",
    "r2_build_meta_pools": "builds IMAGE pools; its prompts are registered and its RESOLVED pools "
                           "are recorded in pools.json (see RUNTIME_POOL_RECORDS)",
    "rezero_control": "weight-space control; consumes pools built elsewhere, defines none",
    "matched_removal": "utility-matching helper; defines no prompts",
    "frp_mask_refill": "mask arithmetic only; defines no prompts",
    "registry_v1": "superseded artifact registry, not a prompt authority",
    "verify_splits": "the superseded v1 of THIS file",
    "split_registry": "this file",
    "retain_scope_slice": "slices an existing retain pool; defines none",
    "rescore_final_v2": "final-evaluation driver; registered as an EVAL script",
    "diag_coeff_scale": "numerical diagnostic; defines no prompts",
    "diag_wdr_scale": "numerical diagnostic; defines no prompts",
    "erase_wdr_v2": "registered as a defence builder",
    "erase_utrap_v2": "registered as a defence builder",
    "meta_unlearn_v2": "registered as a defence builder",
}

#: RESOLVED RUNTIME RECORDS (v3 S-3). Static extraction cannot establish coverage: a pool can be
#: built inside a function, from a CSV, or from a sampled subset. These are the files an ACTUAL RUN
#: writes recording the pools and dataset manifests it really used. Each entry is
#: (glob, kind, what it must contain). Absence of ALL records for a kind makes
#: `coverage_complete` False, which `enforce()` treats as a failure for any NEW claim.
RUNTIME_POOL_RECORDS = [
    (f"{P}/upstream_runs/meta/dataset*/pools.json", "meta_image_pools",
     "per-pool resolved prompts, counts and content fingerprints written by "
     "scripts/r2_build_meta_pools.py"),
    (f"{P}/adapters/*/fingerprint.json", "uce_port_fingerprints",
     "the resolved erase/preserve prompt lists an erase_uce.py build actually used"),
    (f"{P}/outputs/erase_*.json", "defence_build_records",
     "per-build records; those carrying a resolved prompt pool are counted"),
]

#: FINAL: frozen 2026-10-25, before any new final result was produced or inspected. Chosen to be
#: surface-form disjoint from every DEF_TRAIN / ATK_TRAIN / DEV string in the repository, and
#: checked programmatically by `audit()` rather than by eye.
FINAL_PROMPTS = {
    "church": ["the nave of a chapel at midday",
               "a steepled meeting house in winter light",
               "a basilica interior with stone columns"],
    "parachute": ["a canopy drifting over a treeline",
                  "a jumper's chute moments after deployment",
                  "a ram-air wing silhouetted at dusk"],
    "garbage_truck": ["a refuse lorry at a kerbside collection",
                      "a sanitation vehicle with its hopper raised",
                      "a waste collection lorry in morning traffic"],
    "van_gogh": ["a cypress and wheatfield in thick swirling impasto",
                 "a night sky rendered in restless spiralling strokes",
                 "a rush-seated chair in flat yellow and ochre"],
}

FINAL_FROZEN_AT = "2026-10-25T23:30:00-04:00"
FINAL_FREEZE_NOTE = (
    "Frozen BEFORE any new final result existed. No arm has been evaluated on these prompts at the "
    "time of freezing; the first use must be a confirmatory final evaluation, not selection. If any "
    "of these strings later appears in a defence or attacker pool, audit() reports it and the set "
    "must be reissued under a new version rather than quietly amended.")


def mod(name):
    spec = iu.spec_from_file_location(f"_sr_{name}", os.path.join(HERE, f"{name}.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """Compare prompts as the tokenizer effectively sees them: lowercase, collapsed whitespace,
    punctuation stripped. v1 normalised case and whitespace only; a trailing period or a comma was
    enough to hide an overlap from it."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _literal_prompt_templates(script):
    """Read DEFENCE prompt TEMPLATES straight out of the source text.

    Importing a module gives its module-level pools, but `erase_gift` and friends build their pool
    inside main() from an f-string, so there is nothing to import -- which is precisely how v1
    missed them. So the templates are extracted from source and instantiated for each concept.
    """
    path = os.path.join(HERE, f"{script}.py")
    if not os.path.exists(path):
        return []
    src = open(path).read()
    pats = [r'f"([^"\n]*\{a\.concept\}[^"\n]*)"', r'f"([^"\n]*\{args\.concept\}[^"\n]*)"',
            r'f"([^"\n]*\{concept\}[^"\n]*)"', r'f"([^"\n]*\{c\}[^"\n]*)"']
    found = []
    for p in pats:
        found += re.findall(p, src)
    return sorted(set(found))


#: prompt-ish: a multi-word natural-language string. Deliberately loose -- over-collecting a
#: non-prompt string is harmless here (it can only ADD a reported overlap, never hide one), while
#: under-collecting is exactly the v2 failure.
_PROMPTY = re.compile(r"^[A-Za-z][A-Za-z0-9 ,.'\-]{6,120}$")


def _ast_prompt_literals(script):
    """Collect candidate prompt strings by AST, INCLUDING FUNCTION-LOCAL ones (v3 S-3).

    Why AST and not regex. `erase_utrap_v2.py:127-137` defines `DEF_TRAIN_TARGET` -- the 16 target
    prompts that built every FINDING76 arm -- inside `main()`. It is therefore invisible to
    `import` (v2's `_module_pools`) and to v2's four source regexes. An AST walk sees every string
    constant in the module regardless of scope, and sees f-string templates as JoinedStr nodes so
    the `{concept}` placeholders can be instantiated.

    Returns (plain_strings, templates). THIS IS STILL STATIC EXTRACTION and is NOT coverage: see
    `runtime_coverage` and the `coverage_complete` verdict.
    """
    path = os.path.join(HERE, f"{script}.py")
    if not os.path.exists(path):
        return [], []
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError as e:
        return [], [f"__PARSE_ERROR__{e}"]
    plain, tmpl = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _PROMPTY.match(node.value) and " " in node.value:
                plain.add(node.value)
        elif isinstance(node, ast.JoinedStr):
            # rebuild "a photo of a {concept}" from the f-string's parts
            parts, ok = [], False
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    e = v.value
                    nm = (e.attr if isinstance(e, ast.Attribute) else
                          e.id if isinstance(e, ast.Name) else None)
                    if nm in ("concept", "c", "ck", "k"):
                        parts.append("{concept}")
                        ok = True
                    else:
                        parts.append("\x00")      # an unresolvable interpolation
            s = "".join(parts)
            if ok and "\x00" not in s and " " in s:
                tmpl.add(s)
    return sorted(plain), sorted(tmpl)


def _runtime_coverage():
    """Scan for RESOLVED RUNTIME pool/dataset records (v3 S-3).

    A registry that reads only source cannot know what a run actually trained on. These are the
    records an executed run wrote. Each is reported with the resolved prompts it carries, so the
    coverage claim rests on observed data rather than on static extraction.
    """
    out = {}
    for pat, kind, what in RUNTIME_POOL_RECORDS:
        files = sorted(glob.glob(pat))
        recs, prompts, unreadable = [], set(), []
        for f in files:
            try:
                d = json.load(open(f))
            except Exception as e:
                unreadable.append({"file": f, "error": repr(e)[:200]})
                continue
            got = []
            if kind == "meta_image_pools":
                for pn, pv in (d.get("pools") or {}).items():
                    got += list(pv.get("prompts") or [])
            elif kind == "uce_port_fingerprints":
                got += list(d.get("port_preserve_concepts") or [])
                for key in ("concept", "target"):
                    if isinstance(d.get(key), str) and d[key]:
                        got.append(d[key])
            else:
                hp = d.get("hparams") or {}
                for key in ("T_prompts", "H_prompts", "prompts", "forget_prompts",
                            "target_prompts", "retain_prompts"):
                    v = d.get(key) or hp.get(key)
                    if isinstance(v, list):
                        got += [x for x in v if isinstance(x, str)]
            if got:
                recs.append({"file": f, "n_prompts": len(set(got))})
                prompts |= {norm(x) for x in got}
        out[kind] = {
            "what_this_must_contain": what, "glob": pat,
            "files_found": len(files), "files_carrying_resolved_prompts": len(recs),
            "n_distinct_resolved_prompts": len(prompts),
            "unreadable": unreadable,
            "records": recs[:25],
            "truncated": max(0, len(recs) - 25),
            "_prompts": sorted(prompts),
        }
    return out


def _module_pools(script):
    """Module-level prompt lists/dicts, by convention: *_POOL, *_PROMPTS, PRESERVE, CONCEPTS."""
    try:
        m = mod(script)
    except BaseException as e:
        # BaseException, not Exception (v3). `scripts/r1_eval_imma.py` parses argv at module level,
        # so importing it raised SystemExit -- which is a BaseException and sailed straight through
        # an `except Exception`, killing the entire audit with argparse's usage message and exit 2,
        # attributed to THIS file. A script that is not import-safe must be a recorded NOTE, never
        # a crash of the registry.
        return [], f"import failed: {type(e).__name__}: {str(e)[:300]!r}"
    pools = []
    for name in dir(m):
        if not re.search(r"(POOL|PROMPTS|PRESERVE|CONCEPTS)$", name):
            continue
        v = getattr(m, name)
        if isinstance(v, dict):
            for vv in v.values():
                pools += list(vv) if isinstance(vv, (list, tuple)) else [vv]
        elif isinstance(v, (list, tuple)):
            pools += list(v)
    return [x for x in pools if isinstance(x, str)], None


def audit():
    ge, ar = mod("gen_eval"), mod("attack_relearn")
    concepts = sorted(ar.SPLIT)
    inspected, not_inspected, import_notes = [], [], {}

    # every *.py in scripts/ that LOOKS like a prompt authority must be on the roster. v2 matched
    # only `erase_*` plus a hardcoded two-name tuple, which is why `meta_unlearn_v2.py` was never
    # even reported (v3 S-1).
    stems = [f[:-3] for f in os.listdir(HERE) if f.endswith(".py")]
    on_disk = {s for s in stems if any(re.search(p, s) for p in BUILDER_PATTERNS)}
    unknown = sorted(on_disk - set(ALL_REGISTERED) - set(NOT_PROMPT_AUTHORITIES))
    # an exemption that names a file which no longer exists is itself stale
    stale_exemptions = sorted(k for k in NOT_PROMPT_AUTHORITIES if k not in stems)
    for s in ALL_REGISTERED:
        if os.path.exists(os.path.join(HERE, f"{s}.py")):
            inspected.append(s)
        else:
            not_inspected.append(s)

    # ---- build the pools, attributing every prompt to the script it came from
    def_train = {}        # normalised prompt -> set of source scripts
    static_only = {}      # prompts seen ONLY by static extraction, for honest labelling
    for s in DEFENCE_SCRIPTS + UPSTREAM_WRAPPERS:
        tmpls = _literal_prompt_templates(s)
        pools, note = _module_pools(s)
        if note:
            import_notes[s] = note
        ast_plain, ast_tmpl = _ast_prompt_literals(s)
        perr = [t for t in ast_tmpl if t.startswith("__PARSE_ERROR__")]
        if perr:
            import_notes[s] = (import_notes.get(s, "") + " " + perr[0]).strip()
            ast_tmpl = [t for t in ast_tmpl if not t.startswith("__PARSE_ERROR__")]
        for c in concepts:
            for t in tmpls:
                p = norm(t.replace("{a.concept}", c).replace("{args.concept}", c)
                         .replace("{concept}", c).replace("{c}", c))
                def_train.setdefault(p, set()).add(f"{s}(template)")
            for t in ast_tmpl:                                          # v3 S-3
                p = norm(t.replace("{concept}", c))
                def_train.setdefault(p, set()).add(f"{s}(ast-template)")
        for p in pools:
            def_train.setdefault(norm(p), set()).add(f"{s}(module)")
        for p in ast_plain:                                             # v3 S-3: function-local
            n = norm(p)
            def_train.setdefault(n, set()).add(f"{s}(ast-literal)")
            static_only.setdefault(n, set()).add(s)

    # ---- RESOLVED RUNTIME pools: what executed runs actually used (v3 S-3)
    runtime = _runtime_coverage()
    runtime_prompts = set()
    for kind, v in runtime.items():
        for p in v.pop("_prompts"):
            runtime_prompts.add(p)
            def_train.setdefault(p, set()).add(f"runtime:{kind}")

    atk_train = {norm(p): {"attack_relearn.SPLIT[c]['T']"}
                 for c in concepts for p in ar.SPLIT[c]["T"]}
    dev = {norm(p): {"attack_relearn.SPLIT[c]['H']", "gen_eval.CONCEPTS[c]"}
           for c in concepts for p in ar.SPLIT[c]["H"]}
    for c in concepts:
        for p in ge.CONCEPTS.get(c, []):
            dev.setdefault(norm(p), set()).add("gen_eval.CONCEPTS[c]")
    final = {norm(p): {"split_registry.FINAL_PROMPTS"}
             for c, v in FINAL_PROMPTS.items() for p in v}
    eval_util = {norm(p): {"gen_eval.CONCEPTS[non-target]"}
                 for k, v in ge.CONCEPTS.items() if k not in concepts for p in v}

    def overlap(a, b):
        return [{"prompt": p, "sources": sorted(a[p] | b[p])} for p in sorted(set(a) & set(b))]

    rep = {
        "study": "split_registry",
        "split_version": SPLIT_VERSION,
        "supersedes": "scripts/verify_splits.py (v1), whose CLEAN verdict did not cover "
                      "erase_gift / erase_imma / erase_wsr / erase_utrap",
        "concepts": concepts,
        "coverage": {
            "scripts_inspected": sorted(set(inspected)),
            "scripts_not_inspected": sorted(not_inspected),
            "defence_scripts_on_disk_not_registered": unknown,
            "stale_exemptions_naming_absent_files": stale_exemptions,
            "import_notes": import_notes,
            "registered": {"defence": sorted(DEFENCE_SCRIPTS),
                           "upstream_wrappers": sorted(UPSTREAM_WRAPPERS),
                           "eval": sorted(EVAL_SCRIPTS)},
            "extraction_methods": {
                "module": "importable module-level *_POOL/*_PROMPTS/PRESERVE/CONCEPTS",
                "template": "v2's four source regexes over f-string prompt templates",
                "ast-literal": "v3: every string constant in the module, FUNCTION-LOCAL included "
                               "(erase_utrap_v2.py:127-137 DEF_TRAIN_TARGET is function-local and "
                               "invisible to the other two methods)",
                "ast-template": "v3: f-string templates recovered from JoinedStr nodes",
                "runtime": "v3: RESOLVED pools recorded by an executed run",
            },
            "static_extraction_is_not_coverage":
                "The `module`, `template`, `ast-literal` and `ast-template` methods all read "
                "SOURCE. They cannot know what a run actually trained on: a pool may be sampled "
                "from a CSV, passed on the command line, or read from a dataset directory. A "
                "claim of complete coverage therefore requires the RESOLVED RUNTIME records "
                "below, and `coverage_complete` is False without them.",
            "n_prompts_static_only": len(static_only),
            "runtime_coverage": runtime,
            "n_distinct_runtime_prompts": len(runtime_prompts),
        },
        "splits": {
            "DEF_TRAIN": {"n": len(def_train), "status": "USED_FOR_TRAINING"},
            "ATK_TRAIN": {"n": len(atk_train), "status": "USED_BY_ATTACKER"},
            "DEV": {"n": len(dev), "status": "USED_FOR_SELECTION",
                    "note": "the three historical H prompts per concept. lambda, beta and K were "
                            "repeatedly chosen while watching H, so H is a DEVELOPMENT set. It is "
                            "NOT relabelled untouched; every historical number stands as a "
                            "development measurement."},
            # STATUS CORRECTED 2026-10-26 (audit). This read "FROZEN_UNINSPECTED", which became
            # FALSE the moment FINDING65 and FINDING66 scored and interpreted these prompts. The
            # literal was additionally protected by a PASSING test
            # (test_audit_repairs.py::t_dev_is_labelled_as_used), so the false claim was enforced
            # rather than merely stale -- the test and the literal had to move together, which is
            # why neither could be fixed in isolation.
            "FINAL": {"n": len(final), "status": "INSPECTED_DEVELOPMENT_DATA",
                      "frozen_at": FINAL_FROZEN_AT, "note": FINAL_FREEZE_NOTE,
                      "inspected_at": "2026-10-26",
                      "inspected_by": ["FINDING65", "FINDING66"],
                      "status_note":
                          "Frozen 2026-10-25T23:30 and verified disjoint by computation, then "
                          "SCORED AND INTERPRETED by FINDING65/FINDING66. It is development data "
                          "from here on and must not be described as untouched or held out. A "
                          "fresh final set must be frozen before any new configuration is selected "
                          "against it. Separately, every FINAL number produced so far was scored "
                          "with inconsistent label text (audit defect D-022); the effect on scores "
                          "and conclusions is unresolved pending corrected evaluation, and six of "
                          "the twelve records can never be re-scored because their attacked "
                          "checkpoints were deleted (D-022b).",
                      "prompts": FINAL_PROMPTS},
            "EVAL_UTILITY": {"n": len(eval_util), "status": "UTILITY_PANEL"},
        },
        "overlaps": {
            "DEF_TRAIN_vs_EVAL_TARGET": overlap(def_train, dev),
            "DEF_TRAIN_vs_ATK_TRAIN": overlap(def_train, atk_train),
            "ATK_TRAIN_vs_EVAL_TARGET": overlap(atk_train, dev),
            "DEF_TRAIN_vs_FINAL": overlap(def_train, final),
            "ATK_TRAIN_vs_FINAL": overlap(atk_train, final),
            "DEV_vs_FINAL": overlap(dev, final),
            "DEF_TRAIN_vs_EVAL_UTILITY": overlap(def_train, eval_util),
        },
    }
    # ---- TWO SEPARATE VERDICTS (v3 S-4). v2 had one, `FINAL_usable`, computed purely from string
    #      disjointness; it reported `true` while `splits.FINAL.status` in the SAME record said
    #      INSPECTED_DEVELOPMENT_DATA. "No string overlaps" and "may be used for a confirmatory
    #      final evaluation" are different questions and now have different answers.
    string_disjoint = not (rep["overlaps"]["DEF_TRAIN_vs_FINAL"]
                           or rep["overlaps"]["ATK_TRAIN_vs_FINAL"]
                           or rep["overlaps"]["DEV_vs_FINAL"])
    cov = rep["coverage"]
    runtime_kinds_present = [k for k, v in cov["runtime_coverage"].items()
                             if v["files_carrying_resolved_prompts"] > 0]
    unreadable = {k: v["unreadable"] for k, v in cov["runtime_coverage"].items()
                  if v["unreadable"]}
    coverage_complete = bool(
        not cov["defence_scripts_on_disk_not_registered"]
        and not cov["scripts_not_inspected"]
        and not cov["stale_exemptions_naming_absent_files"]
        and not unreadable
        and len(runtime_kinds_present) == len(RUNTIME_POOL_RECORDS))

    final_inspected = rep["splits"]["FINAL"]["status"] != "FROZEN_UNINSPECTED"
    reasons = []
    if not string_disjoint:
        reasons.append("FINAL overlaps a DEF_TRAIN / ATK_TRAIN / DEV string")
    if final_inspected:
        reasons.append("FINAL has status %r: it was SCORED AND INTERPRETED by %s, so it is "
                       "development data and a new confirmatory evaluation needs a freshly frozen "
                       "set" % (rep["splits"]["FINAL"]["status"],
                                ", ".join(rep["splits"]["FINAL"].get("inspected_by") or [])))
    if not coverage_complete:
        bits = []
        if cov["defence_scripts_on_disk_not_registered"]:
            bits.append("unregistered builders %s" % cov["defence_scripts_on_disk_not_registered"])
        if cov["scripts_not_inspected"]:
            bits.append("registered but absent %s" % cov["scripts_not_inspected"])
        if cov["stale_exemptions_naming_absent_files"]:
            bits.append("stale exemptions %s" % cov["stale_exemptions_naming_absent_files"])
        if unreadable:
            bits.append("unreadable runtime records %s" % unreadable)
        # RUNTIME_POOL_RECORDS entries are (glob, kind, what); the KIND is the second element.
        missing_kinds = [k for _, k, _ in RUNTIME_POOL_RECORDS if k not in runtime_kinds_present]
        if missing_kinds:
            bits.append("no RESOLVED RUNTIME pool records of kind(s) %s" % missing_kinds)
        reasons.append("coverage is incomplete: " + "; ".join(bits))

    rep["verdict"] = {
        "DEF_TRAIN_vs_EVAL_TARGET": ("LEAK CONFIRMED -- see ERRATUM_split_leak.md"
                                     if rep["overlaps"]["DEF_TRAIN_vs_EVAL_TARGET"] else "clean"),
        # VERDICT 1: surface-form disjointness. A computation over strings, nothing more.
        "string_disjoint": string_disjoint,
        # VERDICT 2: whether FINAL may carry a NEW confirmatory claim.
        "eligible_for_confirmatory_evaluation": bool(
            string_disjoint and not final_inspected and coverage_complete),
        "eligibility_blockers": reasons,
        "coverage_complete": coverage_complete,
        "runtime_pool_kinds_present": runtime_kinds_present,
        # DEPRECATED ALIAS, retained so older readers do not silently get nothing.
        "FINAL_usable": string_disjoint,
        "FINAL_usable_DEPRECATED":
            "v2 name. It means STRING_DISJOINT ONLY and must NOT be read as 'FINAL may be used "
            "for a confirmatory evaluation' -- v2 reported it true while the same record said "
            "FINAL was INSPECTED_DEVELOPMENT_DATA. Use "
            "`eligible_for_confirmatory_evaluation`.",
        "verdict_separation_note":
            "string_disjoint and eligible_for_confirmatory_evaluation are DIFFERENT VERDICTS. "
            "Disjointness is necessary and not sufficient: a set that is disjoint but already "
            "inspected is development data.",
    }
    return rep


#: the failure classes `enforce()` reports. All three are EXPLICIT FAILURES for any NEW claim of
#: clean evaluation (v3 S-2), not advisory notes as they were in v2.
def enforce(rep=None, for_new_claim=True):
    """Turn the registry's findings into FAILURES. Returns a list of failure strings.

    v2 reported unregistered builders in its coverage block and exited 0, so nothing downstream
    could be stopped by it. Callers use `require_clean()`; `__main__` exits non-zero.

    `for_new_claim=False` checks only the structural failures (registration, readability) and
    skips the confirmatory-evaluation eligibility question -- appropriate for a DEBUGGING or
    REPRODUCTION run that is not producing a new result.
    """
    rep = rep or audit()
    cov, ver = rep["coverage"], rep["verdict"]
    f = []
    if cov["defence_scripts_on_disk_not_registered"]:
        f.append("MISSING REGISTRATION: %s are on disk, match a builder pattern, and are neither "
                 "registered nor exempted. A builder that is not registered cannot have had its "
                 "pools inspected, so no clean-evaluation claim covers it (v3 S-1/S-2)."
                 % cov["defence_scripts_on_disk_not_registered"])
    if cov["scripts_not_inspected"]:
        f.append("FAILED POOL INSPECTION: registered scripts %s are absent from disk, so their "
                 "pools were not inspected." % cov["scripts_not_inspected"])
    if cov["stale_exemptions_naming_absent_files"]:
        f.append("STALE EXEMPTION: NOT_PROMPT_AUTHORITIES names %s, which no longer exist. An "
                 "exemption roster that drifts is a hole." % cov["stale_exemptions_naming_absent_files"])
    for kind, v in cov["runtime_coverage"].items():
        if v["unreadable"]:
            f.append("FAILED POOL INSPECTION: %d runtime record(s) of kind %r could not be read: "
                     "%s" % (len(v["unreadable"]), kind, v["unreadable"][:2]))
    if for_new_claim:
        if not ver["coverage_complete"]:
            f.append("INCOMPLETE COVERAGE: %s" % "; ".join(
                r for r in ver["eligibility_blockers"] if r.startswith("coverage is incomplete")))
        if not ver["eligible_for_confirmatory_evaluation"]:
            f.append("NOT ELIGIBLE FOR CONFIRMATORY EVALUATION: %s"
                     % "; ".join(ver["eligibility_blockers"]))
    return f


def require_clean(for_new_claim=True, rep=None):
    """Raise SystemExit with the failures if the registry does not permit the requested claim.

    This is the function an ACTIVE RUNNER must call. Printing the verdict is not enforcement.
    """
    rep = rep or audit()
    f = enforce(rep, for_new_claim=for_new_claim)
    if f:
        raise SystemExit(
            "SPLIT REGISTRY %s REFUSES THIS RUN (for_new_claim=%s):\n  - %s\n"
            "Registry record: %s/split_registry_%s.json"
            % (SPLIT_VERSION, for_new_claim, "\n  - ".join(f), OUT, SPLIT_VERSION))
    return rep


#: RNG stream separation (audit section 5). Distinct, documented offsets so a defence build, an
#: attacker's training, its data generation and the final evaluation cannot share a stream.
RNG_STREAMS = {"defence_build": 0, "attack_train": 1_000_000, "data_generation": 2_000_000,
               "dev_eval": 3_000_000, "final_eval": 4_000_000}


def seed_for(stream, seed):
    """Derive a per-stream seed. Evaluation seeds are COMMON across comparator arms (the same
    `seed` gives the same evaluation noise for every arm) while training variation stays
    independent, which is the audit's requirement."""
    if stream not in RNG_STREAMS:
        raise ValueError(f"unknown RNG stream {stream!r}; declared: {sorted(RNG_STREAMS)}")
    return RNG_STREAMS[stream] + int(seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="split registry %s" % SPLIT_VERSION)
    ap.add_argument("--for-new-claim", dest="nc", action="store_true", default=True,
                    help="enforce confirmatory-evaluation eligibility too (DEFAULT)")
    ap.add_argument("--debugging", dest="nc", action="store_false",
                    help="structural checks only; for a reproduction or debugging run that is "
                         "NOT producing a new result. Does not relax registration failures.")
    a = ap.parse_args()
    r = audit()
    json.dump(r, open(f"{OUT}/split_registry_{SPLIT_VERSION}.json", "w"), indent=1)
    cov = dict(r["coverage"])
    cov["runtime_coverage"] = {k: {kk: vv for kk, vv in v.items() if kk != "records"}
                               for k, v in cov["runtime_coverage"].items()}
    print(json.dumps({"coverage": cov, "verdict": r["verdict"],
                      "overlap_counts": {k: len(v) for k, v in r["overlaps"].items()}}, indent=1))
    for o in r["overlaps"]["DEF_TRAIN_vs_EVAL_TARGET"]:
        print("  LEAK:", o["prompt"], "<-", ", ".join(o["sources"]))
    # ENFORCEMENT (v3 S-2). v2 printed its findings and exited 0.
    fails = enforce(r, for_new_claim=a.nc)
    for f in fails:
        print("  ENFORCED FAILURE:", f)
    print("  registry verdict: %s (for_new_claim=%s)"
          % ("REFUSE" if fails else "PERMIT", a.nc))
    os.sys.exit(1 if fails else 0)
