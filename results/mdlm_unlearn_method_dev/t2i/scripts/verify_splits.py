"""ISSUE 8. Define the four prompt splits ONCE, in one place, and verify their separation
programmatically against the actual source code of every script that trains on prompts.

The audit's point: this phase claimed "prompt-disjoint T/H" but the claim was only ever made for
the ATTACKER's training set. No document stated what the DEFENCE trained on, and nothing checked
whether the defence's prompts overlapped the evaluation prompts. A disjointness claim that is
asserted in a docstring and never computed is not a protocol.

The four splits:

  DEF_TRAIN    prompt strings a DEFENCE optimises on (erasure and, for composites, the damage and
               distillation pools). If an evaluation prompt is in here, the erasure/utility number
               for that arm is partly IN-SAMPLE.
  ATK_TRAIN    prompt strings the ATTACKER is given (SPLIT[c]["T"]).
  EVAL_TARGET  held-out prompts the erased concept is scored on (SPLIT[c]["H"], which is exactly
               gen_eval.CONCEPTS[c]).
  EVAL_UTILITY prompts the retained concepts are scored on (gen_eval.CONCEPTS for non-targets).

This script READS the pools out of the other scripts by import, so it cannot drift from them: if
someone adds a prompt to a pool, the overlap table changes on the next run.

Run: python scripts/verify_splits.py   ->   outputs/split_registry.json  (exit 0 always; the
report is the product, and a NON-EMPTY overlap is a finding to disclose, not a crash)
"""
import importlib.util as iu
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/data/bijaypandey/t2i_unlearn/outputs"


def mod(name):
    spec = iu.spec_from_file_location(f"_sv_{name}", os.path.join(HERE, f"{name}.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """Compare prompts as the model sees them: case- and whitespace-insensitive."""
    return " ".join(s.lower().split())


def main():
    ge, ar, uce, esd, undo = (mod("gen_eval"), mod("attack_relearn"), mod("erase_uce"),
                              mod("erase_esd"), mod("undo_t2i"))
    concepts = sorted(ar.SPLIT)
    report = {"study": "split_registry", "issue": 8, "concepts": concepts, "per_concept": {},
              "sources": {
                  "DEF_TRAIN": ["erase_uce.FORGET_PROMPTS", "erase_uce.PRESERVE",
                                "erase_esd.PROMPTS", "undo_t2i.FORGET_POOL",
                                "undo_t2i.RETAIN_POOL"],
                  "ATK_TRAIN": ["attack_relearn.SPLIT[c]['T']"],
                  "EVAL_TARGET": ["attack_relearn.SPLIT[c]['H']", "gen_eval.CONCEPTS[c]"],
                  "EVAL_UTILITY": ["gen_eval.CONCEPTS[k] for k != target"]}}

    # ---- the pools, read from source
    uce_pool = list(uce.FORGET_PROMPTS.values()) + list(uce.PRESERVE)
    esd_pool = list(esd.PROMPTS.values())
    undo_retain = list(undo.RETAIN_POOL)
    eval_utility_all = {k: [norm(p) for p in v] for k, v in ge.CONCEPTS.items()}

    for c in concepts:
        atk = [norm(p) for p in ar.SPLIT[c]["T"]]
        ev_t = [norm(p) for p in ar.SPLIT[c]["H"]]
        undo_forget = [norm(p) for p in undo.FORGET_POOL.get(c, [])]
        def_train = sorted(set([norm(p) for p in uce_pool + esd_pool + undo_retain] + undo_forget))
        ev_u = sorted({p for k, v in eval_utility_all.items() if k != c for p in v})

        # the identity check the phase has always claimed
        ov_atk_evt = sorted(set(atk) & set(ev_t))
        # the checks nobody ran
        ov_def_evt = sorted(set(def_train) & set(ev_t))
        ov_def_evu = sorted(set(def_train) & set(ev_u))
        ov_def_atk = sorted(set(def_train) & set(atk))
        # which script contributed each leak, so the disclosure is per-arm-family not global
        attribution = {}
        for p in ov_def_evt + ov_def_evu:
            src = []
            if p in [norm(x) for x in undo_forget]: src.append("undo_t2i.FORGET_POOL")
            if p in [norm(x) for x in undo_retain]: src.append("undo_t2i.RETAIN_POOL")
            if p in [norm(x) for x in uce_pool]: src.append("erase_uce")
            if p in [norm(x) for x in esd_pool]: src.append("erase_esd")
            attribution[p] = src

        report["per_concept"][c] = {
            "n": {"DEF_TRAIN": len(def_train), "ATK_TRAIN": len(atk),
                  "EVAL_TARGET": len(ev_t), "EVAL_UTILITY": len(ev_u)},
            "ATK_TRAIN_x_EVAL_TARGET": ov_atk_evt,
            "DEF_TRAIN_x_EVAL_TARGET": ov_def_evt,
            "DEF_TRAIN_x_EVAL_UTILITY": ov_def_evu,
            "DEF_TRAIN_x_ATK_TRAIN": ov_def_atk,
            "leak_attribution": attribution,
            "EVAL_TARGET_prompts": ev_t,
        }

    # ---- seed spaces, so "separate attack-training from evaluation seeds" is checkable
    report["seed_spaces"] = {
        "attacker_data": "10_000 + i, i in [0, n_images)  -> 10000..10015 at n=16",
        "eval_T_and_H": "seed*1000 + prompt_index",
        "utility_probe": "seed*1000 + 101*concept_index + prompt_index",
        "trainability_probe": "fixed 31337",
        "frozen_grid_seeds": [0, 1, 6],
        "collision_analysis": {
            "attacker_vs_eval": "disjoint for every seed in the frozen grid (10000-10015 is "
                                "reachable by eval only at seed=10, which the grid does not use)",
            "eval_vs_utility": "COLLIDES: at seed s, eval uses s*1000+{0..7} and the utility probe "
                               "uses s*1000+{0,1} for its first concept. The shared value is an "
                               "INITIAL LATENT seed under a DIFFERENT prompt, so no split's "
                               "content leaks into another; it is a benign non-independence, not "
                               "contamination. Disclosed rather than silently fixed, because "
                               "H65 was already running under this scheme when the collision was "
                               "found and splitting a study across two seed conventions mid-run "
                               "is worse than a documented overlap.",
            "required_going_forward": "offset the utility probe into its own block "
                                      "(500_000 + seed*1000 + 101*ci + pi) for studies started "
                                      "AFTER H65, so the partition is structural."},
    }

    tot_evt = sum(len(v["DEF_TRAIN_x_EVAL_TARGET"]) for v in report["per_concept"].values())
    tot_evu = sum(len(v["DEF_TRAIN_x_EVAL_UTILITY"]) for v in report["per_concept"].values())
    tot_atk = sum(len(v["ATK_TRAIN_x_EVAL_TARGET"]) for v in report["per_concept"].values())
    report["verdict"] = {
        "ATK_TRAIN_vs_EVAL_TARGET": "CLEAN" if tot_atk == 0 else f"CONTAMINATED ({tot_atk})",
        "DEF_TRAIN_vs_EVAL_TARGET": "CLEAN" if tot_evt == 0 else f"CONTAMINATED ({tot_evt})",
        "DEF_TRAIN_vs_EVAL_UTILITY": "CLEAN" if tot_evu == 0 else f"CONTAMINATED ({tot_evu})",
        "advunlearn_caveat": "AdvUnlearn arms use OFFICIAL weights; the prompts they were "
                             "adversarially trained on are not in this repository, so DEF_TRAIN is "
                             "UNKNOWN for them and no disjointness claim can be made. Their "
                             "released encoders were trained on the concept word, per their paper, "
                             "but the full prompt set is not recoverable here.",
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(f"{OUT}/split_registry.json", "w"), indent=1)

    print("=== SPLIT REGISTRY (issue 8) ===")
    for c, v in report["per_concept"].items():
        print(f"\n{c}:  sizes {v['n']}")
        print(f"  ATK_TRAIN x EVAL_TARGET  : {v['ATK_TRAIN_x_EVAL_TARGET'] or 'none (clean)'}")
        print(f"  DEF_TRAIN x EVAL_TARGET  : {v['DEF_TRAIN_x_EVAL_TARGET'] or 'none (clean)'}")
        print(f"  DEF_TRAIN x EVAL_UTILITY : {len(v['DEF_TRAIN_x_EVAL_UTILITY'])} prompts "
              f"{v['DEF_TRAIN_x_EVAL_UTILITY'][:3]}{'...' if len(v['DEF_TRAIN_x_EVAL_UTILITY'])>3 else ''}")
        print(f"  DEF_TRAIN x ATK_TRAIN    : {v['DEF_TRAIN_x_ATK_TRAIN'] or 'none'}")
    print(f"\nVERDICT: {json.dumps(report['verdict'], indent=1)}")
    print(f"\nwrote {OUT}/split_registry.json")


if __name__ == "__main__":
    main()
