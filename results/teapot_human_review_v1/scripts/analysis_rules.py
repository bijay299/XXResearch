#!/usr/bin/env python3
"""Pure aggregation rules for the combined Task-A / Task-B descriptive analysis.

Every rule here is the frozen one from analysis_v1/ANALYSIS_CONTRACT.md. This
module holds no input path, no alias, no private identifier and no I/O, so the
synthetic edge-case suite can exercise the rules directly.
"""

PRESENCE_VALUES = ("present", "absent", "uncertain")
DEFINITE_PRESENCE = ("present", "absent")
RENDERING_VALUES = ("coherent", "localized_deformation",
                    "whole_frame_breakdown", "unable_to_judge")
COMPARISON_VALUES = ("less", "about_the_same", "more", "cannot_fairly_compare")
UNRESOLVED = "unresolved"


def derive_presence(votes):
    """Contract §4. Two reviewers.

    present / absent only when BOTH gave that same DEFINITE answer.
    Everything else -- disagreement, any pairing with 'uncertain', and
    uncertain/uncertain -- is 'unresolved'.
    """
    vs = list(votes)
    if len(vs) != 2:
        return UNRESOLVED
    a, b = vs
    if a == b and a in DEFINITE_PRESENCE:
        return a
    return UNRESOLVED


def derive_rendering(votes):
    """Contract §5. Two agreeing votes derive that category.

    'unable_to_judge' is a substantive shared answer: two agreeing
    unable_to_judge votes derive 'unable_to_judge', NOT 'unresolved'.
    Any disagreement is 'unresolved'.
    """
    vs = list(votes)
    if len(vs) != 2:
        return UNRESOLVED
    a, b = vs
    if a == b and a in RENDERING_VALUES:
        return a
    return UNRESOLVED


def definite_presence_coverage(votes):
    """How many of the two reviewers gave a definite present/absent vote."""
    return sum(1 for v in votes if v in DEFINITE_PRESENCE)


def presence_agreement(votes):
    """Classify a pair of raw presence votes for agreement reporting.

    Distinguishes definite consensus from mere agreement on 'uncertain'.
    """
    vs = list(votes)
    if len(vs) != 2:
        return "incomplete"
    a, b = vs
    if a == b:
        return "definite_consensus" if a in DEFINITE_PRESENCE \
            else "uncertain_agreement"
    return "disagreement"


def presence_table(cases):
    """Contract §7. Full 3x3 m0-by-edited table; all nine cells present."""
    axes = ("present", "absent", UNRESOLVED)
    tab = {m: {e: 0 for e in axes} for m in axes}
    for c in cases:
        tab[c["m0_derived_target"]][c["edited_derived_target"]] += 1
    total = sum(tab[m][e] for m in axes for e in axes)
    return {"axes": list(axes), "table": tab, "total": total}


def demonstrated_removal(cases):
    """Contract §8. Conditional count with an explicit denominator.

    numerator   = m0 'present' AND edited 'absent'
    denominator = ALL m0 'present' cases, including edited 'unresolved'
    Zero denominator -> not estimable.
    """
    denom = [c for c in cases if c["m0_derived_target"] == "present"]
    num = [c for c in denom if c["edited_derived_target"] == "absent"]
    breakdown = {"absent": 0, "present": 0, UNRESOLVED: 0}
    for c in denom:
        breakdown[c["edited_derived_target"]] += 1
    out = {
        "numerator": len(num),
        "denominator": len(denom),
        "estimable": bool(denom),
        "denominator_breakdown_by_edited_status": breakdown,
        "numerator_case_ids": [c["pair_id"] for c in num],
    }
    out["value_text"] = ("not estimable (denominator 0)" if not denom
                         else "%d / %d" % (len(num), len(denom)))
    return out


def rendering_distribution(labels):
    """Counts over the five reportable rendering labels, zeros included."""
    out = {k: 0 for k in RENDERING_VALUES + (UNRESOLVED,)}
    for v in labels:
        out[v] = out.get(v, 0) + 1
    return out


def comparison_distribution(values):
    """Contract §10. All four Task-B categories, zeros included."""
    out = {k: 0 for k in COMPARISON_VALUES}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
