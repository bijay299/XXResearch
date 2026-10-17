import csv, hashlib, json

SRC = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo/SD/data/small_imagenet_prompts.csv"
OUT = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/manifest/full_case_manifest.csv"
NUM_VARIATIONS = 10
BASE_SEED_OFFSET = 0
GLOBAL_SEED = 42

rows_out = []
with open(SRC, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        index = int(row["case_number"])  # CSV's own case_number == row position here (0..99)
        case_number = int(row["case_number"])
        prompt = row["prompt"]
        cls = row["class"]
        eval_seed = int(row["evaluation_seed"]) if row["evaluation_seed"] else None
        obj_safe = cls.replace(" ", "_").replace("'", "")
        for var_idx in range(NUM_VARIATIONS):
            seed = (eval_seed + var_idx + BASE_SEED_OFFSET) if eval_seed is not None \
                else (GLOBAL_SEED + index + var_idx + BASE_SEED_OFFSET)
            filename = f"{index:03d}_{obj_safe}_case{case_number:03d}_var{var_idx:02d}.png"
            rows_out.append({
                "index": index,
                "case_number": case_number,
                "class": cls,
                "prompt": prompt,
                "evaluation_seed": eval_seed,
                "variation": var_idx,
                "generation_seed": seed,
                "filename": filename,
            })

assert len(rows_out) == 1000, f"expected 1000 cases, got {len(rows_out)}"

fieldnames = ["index", "case_number", "class", "prompt", "evaluation_seed",
              "variation", "generation_seed", "filename"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)

# sanity checks: no duplicate filenames, no duplicate (case,variation), 100 cases/class*10
filenames = [r["filename"] for r in rows_out]
assert len(set(filenames)) == 1000, "duplicate filenames detected"
pairs = [(r["case_number"], r["variation"]) for r in rows_out]
assert len(set(pairs)) == 1000, "duplicate (case,variation) detected"

from collections import Counter
per_class = Counter(r["class"] for r in rows_out)
print("per-class case*variation counts:", dict(per_class))
print("total rows:", len(rows_out))

h = hashlib.sha256()
with open(OUT, "rb") as f:
    h.update(f.read())
print("sha256:", h.hexdigest())
print("written to:", OUT)
