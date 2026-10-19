"""Whole-workflow resource projection (30% margin) from MEASURED native training and technical-batch timings.
Pure arithmetic; the predicate is frozen: fits iff every inequality below holds. Failure -> RESOURCE_BLOCKED."""
import json
import os

from cprd2 import ledger as LG
from cprd2 import protocol as P
from cprd2.common import ROOT, load, utc

MARGIN = 1.30
VERIFY_RESERVE_H = 0.75       # fresh-area verification: adapter loads + decodes + detector-free hash checks (frozen allowance)
N_DEV_TRAJ, N_HO_TRAJ = 12, 24


def project():
    tr = load(os.path.join(ROOT, "artifacts", "traj", P.GATE_NATIVE_TRAJ, "train_record.json"))
    gen = load(os.path.join(ROOT, "artifacts", "gen_records", "gen.tech.json"))
    sc = load(os.path.join(ROOT, "artifacts", "scores", "tech.json"))
    spf = tr["seconds_per_state"]
    overhead_train = max(0.0, tr["total_seconds"] - tr["train_seconds"])          # load + save + restoration per job
    imgs = [i["seconds"] for s in gen["sets"] for i in s["images"]]
    s_img = sum(imgs) / len(imgs)
    gen_job_fixed = gen["load_seconds"] + 10.0
    det_load = sc["load_seconds"]
    det_per = sc["detect_seconds_total"] / max(1, sc["n_images"])
    n_dev_imgs = P.IMAGES["primary_total"] - 0
    dev_train_s = N_DEV_TRAJ * (250 * spf + overhead_train)
    ho_train_s = N_HO_TRAJ * (250 * spf + overhead_train)
    dev_imgs = 864 + 24
    ho_imgs = 960 + 80
    dev_gen_s = (N_DEV_TRAJ + 1) * gen_job_fixed + dev_imgs * s_img
    ho_gen_s = (N_HO_TRAJ + 1) * gen_job_fixed + ho_imgs * s_img
    dev_score_s = (N_DEV_TRAJ + 1) * det_load + dev_imgs * det_per
    ho_score_s = (N_HO_TRAJ + 1) * det_load + ho_imgs * det_per
    non_reserve_h = (dev_train_s + ho_train_s + dev_gen_s + dev_score_s) / 3600 * MARGIN
    reserve_h = (ho_gen_s + ho_score_s) / 3600 * MARGIN + VERIFY_RESERVE_H
    tot = LG.totals()
    used = tot["charged_gpu_h_total"]
    cap = LG.CAPS["gpu_device_hours"]
    protected = LG.CAPS["gpu_protected_eval_publication_hours"]
    usable_non_reserve = cap - protected
    # serial-device-hours -> wall hours on 4 GPUs (plus staggering), with the same margin
    wall_h = (non_reserve_h + reserve_h - VERIFY_RESERVE_H) / 4.0 + 0.5
    time_left_h = (LG.science_deadline_epoch() - LG.now()) / 3600
    ineq = {
        "non_reserve_fits": used + non_reserve_h <= usable_non_reserve,
        "reserve_fits": reserve_h <= protected,
        "total_fits": used + non_reserve_h + reserve_h <= cap,
        "wall_fits_before_day6_stop": wall_h <= time_left_h,
    }
    return {"utc": utc(), "margin": MARGIN, "measured": {"seconds_per_state": spf, "train_job_overhead_s": overhead_train,
            "seconds_per_image_generate": s_img, "gen_job_fixed_s": gen_job_fixed, "detector_load_s": det_load,
            "detect_s_per_image": det_per, "technical_images": len(imgs)},
            "projected_components_s": {"dev_train": dev_train_s, "ho_train": ho_train_s, "dev_gen": dev_gen_s,
                                       "ho_gen": ho_gen_s, "dev_score": dev_score_s, "ho_score": ho_score_s},
            "projected_non_reserve_gpu_h_with_margin": non_reserve_h, "projected_reserve_gpu_h_with_margin": reserve_h,
            "verify_reserve_allowance_h": VERIFY_RESERVE_H,
            "projected_gpu_h_with_margin": non_reserve_h + reserve_h, "already_charged_gpu_h": used,
            "usable_for_non_reserve_h": usable_non_reserve, "protected_reserve_h": protected, "gpu_cap_h": cap,
            "projected_wall_h_on_4_gpus": wall_h, "time_left_to_science_stop_h": time_left_h,
            "inequalities": ineq, "fits": all(ineq.values()),
            "note": "all 36 trajectories and all 1,928 primary images are projected at the 250-state cap, including the "
                    "gate trajectory already charged (conservative); the held-out stop may be shorter."}
