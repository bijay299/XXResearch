#!/bin/bash
# R1b: the OFFICIAL IMMA defence training, upstream/IMMA @ 2f55041141705031d503134dec22f674f09e21bf
#
# ============================== STATUS ==============================
# The project is FROZEN. GPU work is gated by scripts/gpu_reserve.sh, whose freeze gate refuses
# every reservation unless the implementation lead passes BOTH ALLOW_FROZEN_RUN=1 and the explicit
# --allow-frozen argument. R1_MODE=cpu runs the complete CPU half (budget accounting, subset
# resolution, blank-image audit, artifact labelling) and has been executed; see
# docs/REPAIR_BASELINES_v2.md.
# ====================================================================
#
# README command, verbatim from upstream/IMMA/README.md L50-66:
#   accelerate launch train/defend_text_to_image_lora.py --mixed_precision="no"
#     --pretrained_model_name_or_path=$MODEL_NAME --train_data_dir=$TRAIN_DATA_DIR
#     --caption_column="prompt" --resolution=512 --random_flip --train_batch_size=1
#     --num_train_epochs=50 --learning_rate_lora=1e-04 --learning_rate=1e-05
#     --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 --output_dir=$OUTPUT_DIR
#     --delta_ckpt=$DELTA_CKPT --report_to="tensorboard" --max_train_samples 20
#     --inner_loop_steps=1 --outer_loop_steps=1
#
# ------------------------- THE BUDGET, RESOLVED ON BOTH SIDES (repair B-2) -------------------
# Derivation from source (train/defend_text_to_image_lora.py):
#   :697-699  subset selection (see below)
#   :710-716  len(train_dataloader) = ceil(N_subset / train_batch_size); bs=1 -> = N_subset
#   :720,:750 num_update_steps_per_epoch = ceil(len(dataloader) / gradient_accumulation_steps)
#   :722,:752 max_train_steps = num_train_epochs * num_update_steps_per_epoch
#   :863,:945 inner/outer alternation: `step % (inner+outer) < inner` -> inner, else outer.
#             With inner=outer=1 this is a FLAT 1:1 alternation over `step` within each epoch --
#             not a bilevel loop with multiple sub-steps.
#   :950      inner step  = optimizer_lora.step() over the LoRA parameters, lr 1e-4 (DESCENT)
#   :954-960  outer step  = optimizer.step() on backward(-loss) over the attention parameters,
#             lr 1e-5 -- this is gradient ASCENT, and it is the immunisation itself
#   :563-567  the immunised scope is a SUBSTRING match of to_q/to_v/to_k/to_out over ALL
#             unet.named_parameters() excluding 'lora' => 160 tensors / 93,536,640 params =
#             80 attn1 (SELF) + 80 attn2 (CROSS). The saved filename "imma_unet_xatten_layer.pt"
#             (:977) is therefore a MISNOMER: half the trained scope is self-attention.
# So, with bs=1, grad_accum=1, inner=outer=1 and an even N_subset:
#   total_steps   = epochs * N_subset
#   IMMA ascent   = epochs * N_subset/2
#   LoRA descent  = epochs * N_subset/2
#
#   | configuration                        | N_subset | total steps | IMMA ASCENT updates |
#   |--------------------------------------|----------|-------------|---------------------|
#   | README (--max_train_samples 20)      |       20 |        1000 |                 500 |
#   | REF_imma_vangogh (flag OMITTED)      |      100 |        5000 |                2500 |
#   | ratio                                |       5x |          5x |                  5x |
# The historical artifact received 5x the documented immunisation budget. Evidence:
# logs/r1b_imma_train.log:30-35 "Num examples = 100 | Num Epochs = 50 | Total optimization
# steps = 5000". `Num examples = 100` is itself the proof that --max_train_samples was absent.
# Everything else in the historical run (lrs, scheduler, batch size, resolution, seed, loop
# ratio) is README-faithful; only the sample count, and hence the step budget, diverges.
#
# ------------------------- SUBSET SELECTION IS NOT "THE FIRST N" (repair B-3) ----------------
# defend_text_to_image_lora.py:698-699, one line:
#   dataset["train"].shuffle(seed=args.seed).select(range(max_train_samples, 2*max_train_samples))
#     if dataset["train"].shape[0] >= 2*max_train_samples else ...select(range(max_train_samples))
# It is a SEEDED SHUFFLE followed by a select of the SECOND block [N, 2N) whenever the dataset has
# at least 2N rows. Only a dataset smaller than 2N falls back to [0, N).
# With seed=42 over the 100-row metadata.csv, the resolved 20-image subset is
#   95 43 35 82 16 61 81 33 79 80 44 0 88 51 19 36 73 50 25 45   (.png)
# NONE of the three NSFW-blanked frames (7.png, 29.png, 41.png) is in it. Near-miss worth
# recording: 29.png sits at shuffled position 1, i.e. inside the DISCARDED [0,20) window, so
# "the first 20" would have contained a blank one -- and the <2N fallback branch would too.
# Stage 2 RE-RESOLVES this from the real dataset rather than trusting the numbers above.
#
# ------------------------- BLANK IMAGES (repair B-3) ----------------------------------------
# 3 of the 100 training images are pure black (max pixel == 0): 7.png, 29.png, 41.png. Upstream
# eval/text2image.py builds its pipeline without safety_checker=None, so SD-1.4's checker blanks
# flagged samples. They ARE present in metadata.csv (all 100 rows) and WERE trained on by
# REF_imma_vangogh: 3 images x 50 epochs = 150 of its 5000 batches.
# CORRECTION TO THIS FILE'S PREVIOUS HEADER, which claimed they had been excluded: they had not.
# The filter in run_r1_imma_official.sh was gated behind `if [ ! -f metadata.csv ]`, metadata.csv
# already existed, so the filter NEVER RAN. The dataset on disk is unfiltered, and
# train/defend_text_to_image_lora.py filters nothing itself (no pixel/variance/NSFW check exists
# in it; the only preprocessing is the geometric transform chain at :681-695).
# We do NOT rewrite metadata.csv: it is the input the historical artifact was produced from, and
# overwriting it would make that artifact unreproducible. run_r1_imma_official.sh now writes a
# separate metadata_noblank.csv plus blank_audit.json, and this script RECORDS which selection a
# run used. For the README-faithful 20-sample run the question is moot: no blank frame is selected.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

R=upstream_runs/imma
L=logs/r1b_imma_train.log
U=/data/bijaypandey/t2i_unlearn/upstream/IMMA
UPSTREAM_PIN=2f55041141705031d503134dec22f674f09e21bf
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python
VT=/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python
MODE="${R1_MODE:-cpu}"          # cpu | train
case "$MODE" in cpu|train) ;; *) echo "R1_MODE must be cpu|train" >&2; exit 2 ;; esac

# ---- the sample budget. 20 == the README command. "all" == the historical 5x setting. ---------
MAX_SAMPLES="${IMMA_MAX_SAMPLES:-20}"
case "$MAX_SAMPLES" in
  0|all|ALL) MS_ARG=(); TAG="sall_5x" ;;
  *[!0-9]*)  echo "IMMA_MAX_SAMPLES must be an integer or 'all'" >&2; exit 2 ;;
  *)         MS_ARG=(--max_train_samples "$MAX_SAMPLES"); TAG="s$MAX_SAMPLES" ;;
esac
EPOCHS="${IMMA_EPOCHS:-50}"                 # README L57
BATCH="${IMMA_BATCH:-1}"                    # README L56
INNER="${IMMA_INNER:-1}"                    # README L65
OUTER="${IMMA_OUTER:-1}"                    # README L66
SEED="${IMMA_SEED:-42}"                     # README L60
LR_IMMA="${IMMA_LR:-1e-05}"                 # README L59 (the ASCENT lr)
LR_LORA="${IMMA_LR_LORA:-1e-04}"            # README L58
A="REF_imma_vangogh_${TAG}"
# The output directory is derived from the resolved budget, so a README-faithful run can never
# silently overwrite the 5x artifact and vice versa.
LEGACY="REF_imma_vangogh"                   # the historical 5x directory, PRESERVED untouched

mkdir -p logs
say() { printf '%s\n' "$*" | tee -a "$L"; }
die() { printf 'R1b FAILED: %s\n' "$*" | tee -a "$L" >&2; exit 1; }

say "=== R1b $(date -Is) mode=$MODE out=$A max_train_samples=$MAX_SAMPLES epochs=$EPOCHS"

# ================= STAGE 0: label the historical 5x artifact (CPU, idempotent) =================
# Part of the repair mandate: PRESERVE the completed run under a label that RECORDS its 5x
# budget, and never relabel it as a faithful reference run. Nothing here deletes or moves it.
if [ -d "$R/$LEGACY" ]; then
  if [ ! -f "$R/$LEGACY/BUDGET_5X.md" ]; then
    cat > "$R/$LEGACY/BUDGET_5X.md" <<'EOF'
# HISTORICAL ARTIFACT — 5x THE DOCUMENTED IMMUNISATION BUDGET

This directory is `REF_imma_vangogh`, the FIRST completed IMMA run. It is **not** a faithful
reproduction of the pinned README command, and must never be labelled or cited as one.

| | N_subset | total steps | IMMA ASCENT updates |
|---|---|---|---|
| pinned README command (`--max_train_samples 20`) | 20 | 1000 | 500 |
| **this artifact** (`--max_train_samples` OMITTED) | **100** | **5000** | **2500** |
| ratio | 5x | 5x | **5x** |

Cause: `run_r1b_imma_train.sh` did not pass `--max_train_samples`; its default is `None`, so all
100 generated images were used. Evidence: `logs/r1b_imma_train.log:30-35`
(`Num examples = 100 | Num Epochs = 50 | Total optimization steps = 5000`). `Num examples = 100`
is itself the proof the flag was absent.

Everything else matched the README: `--train_batch_size=1`, `--num_train_epochs=50`,
`--learning_rate_lora=1e-04`, `--learning_rate=1e-05`, `--lr_scheduler=constant`,
`--lr_warmup_steps=0`, `--seed=42`, `--inner_loop_steps=1`, `--outer_loop_steps=1`.

Two further caveats on this artifact:

1. **It trained on 3 NSFW-blanked (pure black) images.** `7.png`, `29.png`, `41.png` in
   `upstream_runs/imma/data/vangogh` have `max pixel == 0`, they are rows in the 100-row
   `metadata.csv`, and nothing filters them — 150 of its 5000 batches were black frames.
   (The README-faithful 20-sample subset happens to exclude all three.)
2. **It was not revision-pinned at the command level.** No `--revision`/`--variant` was passed;
   `defend_text_to_image_lora.py` has no `variant` argument at all, so it loaded the fp32 `main`
   UNet. That resolved to the pinned revision only because the local cache holds exactly one
   snapshot — a coincidence, not an assertion.

The README-faithful counterpart is built as `REF_imma_vangogh_s20` by the same script with
`IMMA_MAX_SAMPLES=20` (the default). This directory is preserved for comparison. DO NOT DELETE.

See docs/REPAIR_BASELINES_v2.md section B.
EOF
    say "    labelled the historical artifact: $R/$LEGACY/BUDGET_5X.md (5x budget recorded)"
  else
    say "    historical artifact already labelled: $R/$LEGACY/BUDGET_5X.md"
  fi
  if [ "$A" = "$LEGACY" ]; then
    die "refusing to write into the historical 5x directory $R/$LEGACY"
  fi
fi

# ================= STAGE 1: upstream pin + resolved argument set ===============================
say "--- stage 1: upstream pin"
GOT_PIN=$(git -C "$U" rev-parse HEAD 2>/dev/null || echo NO-GIT)
[ "$GOT_PIN" = "$UPSTREAM_PIN" ] || die "upstream/IMMA is at $GOT_PIN, expected $UPSTREAM_PIN"
say "    pin OK: $GOT_PIN"
T="$U/train/defend_text_to_image_lora.py"
[ -f "$T" ] || die "missing $T"
[ -x "$V" ] || die "missing venv_imma"
[ -f "$R/diffusers-VanGogh-ESDx1-UNET.pt" ] || die "missing the ESD checkpoint under $R"
[ -f "$R/data/vangogh/metadata.csv" ] || die "no dataset; run run_r1_imma_official.sh first"
# The script must still have no `variant` argument and still select [N,2N); if upstream drifts,
# the budget arithmetic below is no longer valid and we must not quietly proceed.
$VT - "$T" <<'PY' 2>&1 | tee -a "$L" || die "stage 1: upstream has drifted from the audited form"
import re, sys
s = open(sys.argv[1]).read()
checks = [
    ("selection is shuffle(seed).select([N,2N))",
     "select(range(args.max_train_samples, 2 * args.max_train_samples))" in s),
    ("no `variant` argument exists in this trainer", "variant" not in s),
    ("inner/outer alternation is `step % (inner+outer) < inner`",
     "step % (args.inner_loop_steps + args.outer_loop_steps) < args.inner_loop_steps" in s),
    ("the outer step ascends (backward on a negated loss)",
     bool(re.search(r"\(\s*-\s*loss\s*\)|\(-loss\)", s))),
    ("the immunised scope is a to_q/to_v/to_k/to_out substring match",
     all(t in s for t in ("to_q", "to_v", "to_k", "to_out"))),
    ("no blank/NSFW/pixel filtering exists",
     not re.search(r"\.std\(\)|nonzero|nsfw|blank|all_black", s, re.I)),
    # `--max` is a SECOND, vanilla immunisation path ("Directly maximizing model parameters",
    # :368). When set it is tested at :859 and :936, BYPASSING the inner/outer alternation -- so
    # every step would ascend and the budget arithmetic in this script's header would be wrong.
    # Its default is None and we never pass it; assert both so a change cannot pass unnoticed.
    ("--max exists with default None (the alternation-bypassing vanilla path)",
     bool(re.search(r'"--max",\s*type=int,\s*default=None', s))),
    ("--max still gates exactly two branches", len(re.findall(r"if args\.max:", s)) == 2),
]
for n, ok in checks:
    print(f"    [{'OK ' if ok else 'BAD'}] {n}")
bad = [n for n, ok in checks if not ok]
if bad:
    print("    UPSTREAM DRIFT -- the recorded budget/selection analysis no longer applies: "
          + "; ".join(bad))
    sys.exit(1)
PY

# ================= STAGE 2: the budget and the subset, RESOLVED FROM THE REAL DATASET ==========
say "--- stage 2: resolved budget and subset (both sides), plus the blank-image audit"
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_DATASETS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_DATASETS_CACHE" "$R/out"
BUDGET="$R/out/${A}_budget.json"
$V - "$R/data/vangogh" "$MAX_SAMPLES" "$EPOCHS" "$BATCH" "$INNER" "$OUTER" "$SEED" "$BUDGET" \
  <<'PY' 2>&1 | grep -vE "Warning|warn|_pytree" | tee -a "$L" || die "stage 2 (budget/subset)"
import json, math, os, sys
import numpy as np
from PIL import Image
from datasets import load_dataset
d, ms, epochs, bs, inner, outer, seed, out = sys.argv[1:9]
epochs, bs, inner, outer, seed = int(epochs), int(bs), int(inner), int(outer), int(seed)

ds = load_dataset("imagefolder", data_dir=d)["train"]
n_all = len(ds)

# ---- ASSERT the ordering assumption the subset resolution below depends on.
# `selected_files` maps shuffled dataset INDICES back to metadata.csv rows by POSITION, so the
# imagefolder row order must equal the metadata.csv row order. That was previously assumed. If it
# does not hold, the resolved filenames -- and therefore the "no blank image is selected"
# conclusion -- are meaningless, so this is an assertion, not a comment.
from datasets import Image as HFImage
paths = load_dataset("imagefolder", data_dir=d)["train"].cast_column("image", HFImage(decode=False))
ds_order = [os.path.basename(r["path"]) for r in paths["image"]]
meta_rows = [l.strip().split(",")[0] for l in open(os.path.join(d, "metadata.csv")) if l.strip()][1:]
if ds_order != meta_rows:
    first = next((i for i, (x, y) in enumerate(zip(ds_order, meta_rows)) if x != y), None)
    print(f"    FATAL: imagefolder row order != metadata.csv row order (first divergence at "
          f"index {first}: dataset={ds_order[first:first+3]} metadata={meta_rows[first:first+3]}). "
          f"The subset resolution below maps indices to filenames by position and would be wrong.")
    sys.exit(1)
print(f"    ORDERING ASSERTED: imagefolder order == metadata.csv order over {len(ds_order)} rows "
      f"(first three {ds_order[:3]})")

# --- blank audit over every image actually on disk (independent of metadata)
files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
blanks = [f for f in files
          if np.asarray(Image.open(os.path.join(d, f)).convert("RGB"),
                        dtype=np.float32).max() == 0]
print(f"    dataset: {n_all} rows, {len(files)} png on disk, blank(max==0)={blanks}")

def resolve(nsub_arg):
    """Reproduce defend_text_to_image_lora.py:697-699 exactly."""
    if nsub_arg is None:
        return list(range(n_all)), n_all, "no subsetting (--max_train_samples absent)"
    sh = ds.shuffle(seed=seed)
    if n_all >= 2 * nsub_arg:
        sel = sh.select(range(nsub_arg, 2 * nsub_arg))
        how = f"shuffle(seed={seed}).select([{nsub_arg},{2*nsub_arg})) -- the SECOND block"
    else:
        sel = sh.select(range(nsub_arg))
        how = f"shuffle(seed={seed}).select([0,{nsub_arg})) -- the <2N fallback branch"
    return sel, len(sel), how

def budget(nsub):
    per_epoch = math.ceil(math.ceil(nsub / bs) / 1)      # gradient_accumulation_steps = 1
    total = epochs * per_epoch
    # flat 1:1 alternation within each epoch: even step -> inner (LoRA), odd -> outer (IMMA)
    period = inner + outer
    imma = sum(1 for s in range(per_epoch) if s % period >= inner) * epochs
    lora = total - imma
    return {"n_subset": nsub, "steps_per_epoch": per_epoch, "total_steps": total,
            "imma_ascent_updates": imma, "lora_descent_updates": lora}

# --- OUR configuration
mine_arg = None if ms in ("0", "all", "ALL") else int(ms)
sel, nsub, how = resolve(mine_arg)
if mine_arg is None:
    chosen = files
else:
    # recover the selected filenames via a parallel index shuffle over the same seed
    idx = ds.add_column("_i", list(range(n_all))).shuffle(seed=seed)
    lo = mine_arg if n_all >= 2 * mine_arg else 0
    picks = idx.select(range(lo, lo + mine_arg))["_i"]
    chosen = [meta_rows[i] for i in picks]   # safe: ordering asserted above
mine = budget(nsub); mine["selection"] = how; mine["selected_files"] = chosen
sel_blank = sorted(set(chosen) & set(blanks))
mine["blank_images_in_selection"] = sel_blank

# --- the README reference configuration, always computed for comparison
_, n20, how20 = resolve(20)
readme = budget(n20); readme["selection"] = how20
readme["max_train_samples"] = 20

print(f"    OURS   : {how}")
print(f"             n_subset={mine['n_subset']} steps/epoch={mine['steps_per_epoch']} "
      f"total={mine['total_steps']} IMMA_ascent={mine['imma_ascent_updates']} "
      f"LoRA={mine['lora_descent_updates']}")
print(f"             blank images inside the selection: {sel_blank or 'NONE'}")
print(f"    README : {how20}")
print(f"             n_subset={readme['n_subset']} total={readme['total_steps']} "
      f"IMMA_ascent={readme['imma_ascent_updates']}")
ratio = (mine["imma_ascent_updates"] / readme["imma_ascent_updates"]
         if readme["imma_ascent_updates"] else None)
print(f"    RATIO (ours / README, on ASCENT updates): {ratio:.2f}x")
if mine_arg is not None and len(chosen) == mine_arg:
    print(f"    selected files: {' '.join(chosen)}")

# ---- ATOMIC PUBLISH (repair A v3). ----------------------------------------------------------
# `json.dump(obj, open(path, "w"))` TRUNCATES path the instant it is opened, before a single byte
# of the new content is written. A crash, an OOM kill or a full disk in that window leaves a
# truncated or empty result file -- which is the same artifact class as the 29.7%-truncated
# unet.pt that passed every `[ -f ]` guard, except that here WE manufacture it. Worse, line 415's
# read-modify-write re-opens the very file it just parsed, so an interruption destroys the
# original too and there is nothing to fall back to.
# Same-directory temp + flush + fsync + os.replace: a reader sees the old file or the new one.
def _atomic_json(obj, path, indent=1):
    import json as _j, os as _o, tempfile as _t
    d = _o.path.dirname(_o.path.abspath(path)) or "."
    _o.makedirs(d, exist_ok=True)
    fd, tmp = _t.mkstemp(dir=d, prefix=".tmp_r1b.")
    try:
        with _o.fdopen(fd, "w") as fh:
            _j.dump(obj, fh, indent=indent)
            fh.flush(); _o.fsync(fh.fileno())
        _o.replace(tmp, path)
        dfd = _o.open(d, _o.O_RDONLY)
        try: _o.fsync(dfd)
        finally: _o.close(dfd)
    except BaseException:
        try: _o.unlink(tmp)
        except OSError: pass
        raise

_atomic_json({"dataset_dir": d, "rows": n_all, "png_on_disk": len(files),
           "blank_images_on_disk": blanks,
           "blank_filtering_in_trainer": "NONE -- no pixel/variance/NSFW check exists",
           "ours": mine, "readme_reference": readme,
           "ascent_budget_ratio_vs_readme": ratio,
           "scope": "160 tensors / 93,536,640 params = 80 attn1 (SELF) + 80 attn2 (CROSS); "
                    "the 'xatten' in the artifact filename is a misnomer",
           "framing": ("IMMA's claim is resistance to RE-LEARNING and the authors evaluate "
                       "POST-ADAPTATION (README L146-151 compares generation w/o IMMA against "
                       "generation w/ IMMA, both after an adaptation). IMMA never evaluates an "
                       "immunised model standalone.")},
          out)
print(f"    budget record -> {out}")
PY

# ================= STAGE 2b: SPLIT / PROVENANCE GATE (PREREG item 2) ===========================
# The launcher did not call the split registry at all, so the preregistration's requirement that
# "the split/provenance registry returns a clean verdict AND the runner enforces it" was declared
# and not implemented. Enforcing it here, BEFORE any GPU is requested, so a provenance failure
# costs nothing and cannot be discovered after a reservation is held.
#
# What is enforced, and what is deliberately NOT claimed:
#   * every prompt-building script used by this experiment must be REGISTERED (an unregistered
#     builder is an explicit failure, not a warning -- registry v3 S-2);
#   * the RESOLVED runtime pools and the dataset manifest must be inspectable and recorded;
#   * COVERAGE: `coverage_complete` is FALSE here, and `for_new_claim=False` deliberately does not
#     fail on it. Stating that precisely, because an earlier version of this comment claimed
#     "incomplete coverage FAILS" and that was not true of this call: for a DEVELOPMENTAL run the
#     incompleteness is NAMED and EXCLUDED below (33 historical build records carry no resolved
#     pool), not treated as clean and not treated as fatal. A CONFIRMATORY claim would use
#     for_new_claim=True, where it does fail. The archived record carries coverage_complete so the
#     reader sees the real value rather than this comment's summary of it.
#   * FINAL remains INSPECTED_DEVELOPMENT_DATA. This run is DEVELOPMENTAL and makes no
#     confirmatory claim, so `eligible_for_confirmatory_evaluation` being false is EXPECTED and is
#     not treated as a failure -- what would be a failure is silently calling it clean.
#   * The historical DEV overlap is acknowledged explicitly: "a photo of a church" is both an
#     evaluation prompt and a defence-generation prompt for several arms (ERRATUM_split_leak), and
#     lambda/beta/K were chosen while watching H. This run's concept is van_gogh, but the
#     acknowledgement is recorded rather than omitted because it is absent.
export IMMA_ARTIFACT="$A"
say "--- stage 2b: split / provenance gate"
$VT - "$R/$A" <<'PY2' 2>&1 | tee -a "$L" || die "stage 2b (split/provenance gate) FAILED"
import importlib.util as iu, json, os, sys
P = "/data/bijaypandey/t2i_unlearn"
spec = iu.spec_from_file_location("sr", f"{P}/scripts/split_registry.py")
sr = iu.module_from_spec(spec); sys.modules["sr"] = sr; spec.loader.exec_module(sr)

rep = sr.audit()
# for_new_claim=False: this run is DEVELOPMENTAL. It still requires registration and pool
# inspection to succeed; it does not require FINAL eligibility, which is false by design.
# require_clean RAISES SystemExit with the failure list if the registry refuses; it returns the
# report otherwise. It is the function an active runner must call -- printing a verdict is not
# enforcement. Letting it propagate IS the enforcement, so no result is wrapped in a boolean.
rep = sr.require_clean(for_new_claim=False, rep=rep)
problems = sr.enforce(rep, for_new_claim=False)   # [] here by construction; recorded in the artifact
cov = rep.get("coverage", {})
print("[2b] split_version   :", rep.get("split_version"))
print("[2b] FINAL status    :", rep["splits"]["FINAL"]["status"])
# The registry's verdict block is `verdict` (singular) -- an earlier version of this gate read a
# non-existent `verdicts` key and printed None for all three fields, i.e. a gate reporting nothing
# about its own verdict. Read the real keys, and FAIL if any is absent rather than print None.
v = rep.get("verdict") or {}
for need in ("string_disjoint", "eligible_for_confirmatory_evaluation"):
    if need not in v:
        sys.exit("[2b] FAIL: registry verdict has no '%s' key; keys=%s" % (need, sorted(v)))
print("[2b] string_disjoint :", v["string_disjoint"])
print("[2b] eligible_for_confirmatory_evaluation:", v["eligible_for_confirmatory_evaluation"],
      "(FALSE is expected and required: FINAL is inspected development data)")
print("[2b] eligibility_blockers:", v.get("eligibility_blockers"))
print("[2b] DEF_TRAIN_vs_EVAL_TARGET:", v.get("DEF_TRAIN_vs_EVAL_TARGET"))
if v["eligible_for_confirmatory_evaluation"] is not False:
    sys.exit("[2b] FAIL: this run is DEVELOPMENTAL; a true confirmatory-eligibility verdict here "
             "would mean the registry no longer knows FINAL was inspected")
if v["string_disjoint"] is not True:
    sys.exit("[2b] FAIL: splits are not string-disjoint: %s" % rep.get("overlaps"))
unreg = cov.get("defence_scripts_on_disk_not_registered", [])
print("[2b] unregistered builders:", unreg or "none")
rc = cov.get("runtime_coverage") or {}
if not rc:
    sys.exit("[2b] FAIL: coverage block carries no runtime_coverage; keys=%s" % sorted(cov))
print("[2b] scripts_inspected:", len(cov.get("scripts_inspected", [])))
for kind, blk in sorted(rc.items()):
    print("[2b]   runtime %-22s files=%s carrying_resolved_prompts=%s"
          % (kind, blk.get("files_found"), blk.get("files_carrying_resolved_prompts")))

# THIS RUN'S OWN RESOLVED MANIFEST. Written here, before training, so the gate is checking a
# record of what THIS experiment will actually consume -- the 20 selected images and the exact
# caption strings read from metadata.csv -- rather than inferring coverage from source text.
import csv, hashlib
ds = f"{P}/upstream_runs/imma/data/vangogh"
sel = json.load(open(f"{P}/upstream_runs/imma/out/{os.environ['IMMA_ARTIFACT']}_budget.json"))
chosen = (sel.get("ours") or {}).get("selected_files") or sel.get("selected_files")
if not chosen:
    sys.exit("[2b] FAIL: the budget record names no selected_files, so this run's resolved dataset "
             "cannot be recorded; keys=%s" % sorted(sel))
caps = {}
with open(f"{ds}/metadata.csv") as f:
    for row in csv.DictReader(f):
        fn = row.get("file_name") or row.get("image")
        if fn in chosen:
            caps[fn] = row.get("prompt") or row.get("text") or ""
missing_caps = [c for c in chosen if c not in caps]
if missing_caps:
    sys.exit("[2b] FAIL: %d selected images have no caption row in metadata.csv: %s"
             % (len(missing_caps), missing_caps[:5]))
prompts = sorted(set(caps.values()))
blob = json.dumps({"files": sorted(chosen), "captions": caps}, sort_keys=True).encode()
rec = {"kind": "imma_resolved_dataset", "artifact": os.environ["IMMA_ARTIFACT"],
       "dataset_dir": ds, "n_files": len(chosen), "files": sorted(chosen),
       "n_distinct_resolved_prompts": len(prompts), "resolved_prompts": prompts,
       "content_fingerprint_sha256": hashlib.sha256(blob).hexdigest(),
       "selection_rule": "shuffle(seed=42).select(range(20,40)) -- re-resolved in stage 2",
       "blank_images_inside_selection": (sel.get("ours") or {}).get("blank_images_in_selection"),
       "expected_ascent_updates": (sel.get("ours") or {}).get("imma_ascent_updates")}
out_pool = f"{P}/upstream_runs/imma/{os.environ['IMMA_ARTIFACT']}.resolved_pool.json"
tmpp = out_pool + ".tmp"
with open(tmpp, "w") as f:
    json.dump(rec, f, indent=1, sort_keys=True)
os.replace(tmpp, out_pool)
print("[2b] THIS RUN resolved dataset: %d files, %d distinct prompts, sha256 %s"
      % (rec["n_files"], rec["n_distinct_resolved_prompts"], rec["content_fingerprint_sha256"][:16]))
print("[2b]   recorded at", out_pool)
if rec["n_files"] == 0 or rec["n_distinct_resolved_prompts"] == 0:
    sys.exit("[2b] FAIL: this run's resolved dataset record is empty")

# HISTORICAL COVERAGE GAP -- NAMED, not silently passed. 33 outputs/erase_*.json build records
# exist and NONE carries the resolved prompt pool it used, so the corpus of historical arms has
# incomplete provenance. That gap is real and is NOT fixed by this run. It does not block this
# experiment because this experiment does not rest on those arms: its own resolved dataset is
# recorded above. Any claim that DOES rest on them must carry this exclusion.
dbr = rc.get("defence_build_records", {})
if dbr.get("files_found", 0) and not dbr.get("files_carrying_resolved_prompts", 0):
    print("[2b] NAMED EXCLUSION: %d historical defence_build_records carry NO resolved prompt pool."
          % dbr["files_found"])
    print("[2b]   Those arms have incomplete provenance. This run does not depend on them; any")
    print("[2b]   claim that does must state this exclusion. Not treated as clean.")
print("[2b] ACKNOWLEDGED historical DEV overlap: 'a photo of a church' is both an evaluation "
      "prompt and a defence-generation prompt (ERRATUM_split_leak); hyperparameters were chosen "
      "while watching H. Every target number in this project is a DEVELOPMENT measurement.")
if unreg:
    sys.exit("[2b] FAIL: unregistered prompt builders on disk: %s" % unreg)
out = f"{P}/upstream_runs/imma/split_gate_r1b.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump({"gate": "r1b_split_provenance", "for_new_claim": False,
               "enforced": True, "problems": problems,
               "final_status": rep["splits"]["FINAL"]["status"],
               "coverage": cov, "verdict": rep.get("verdict"), "coverage_complete": cov.get("coverage_complete"),
               "acknowledged_dev_overlap": True}, f, indent=1)
os.replace(tmp, out)
print("[2b] PASS -- verdict enforced, recorded at", out)
PY2

if [ "$MODE" = "cpu" ]; then
  say "R1b CPU VERIFICATION COMPLETE $(date -Is) -- stages 0-2 passed, no GPU was requested."
  exit 0
fi

# ================= STAGE 3: refuse to overwrite, then reserve a GPU ============================
if [ -e "$R/$A" ]; then
  die "$R/$A already exists; refusing to overwrite an existing artifact"
fi
# scripts/gpu_reserve.sh is owned by the Infrastructure agent: CALLED, never reimplemented.
# The retired `G=$(bash scripts/campaign_free.sh|head -1)` selection is gone (register D-001).
# CONTRACT NOTE, CPU-verified: gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen] --
# --allow-frozen MUST be the THIRD argument; passing it second binds it to wait_seconds and the
# override silently never reaches `acquire`. Do not wrap the call in $( ): it installs the
# release trap in the CALLING shell.
say "--- stage 3: GPU reservation"
[ -f scripts/gpu_reserve.sh ] || die "scripts/gpu_reserve.sh is missing"
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
# shellcheck source=scripts/gpu_reserve.sh
source scripts/gpu_reserve.sh
gpu_reserve_hold "r1b_imma_${TAG}" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null \
  || die "no GPU reservation granted (see gpu_reserve stderr above)"
[ -n "${GPU_RESERVED:-}" ] || die "gpu_reserve_hold returned 0 but set no GPU_RESERVED"
# PREREG item 5. This line used to read `export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"`, re-deriving
# the device from the bare index. That is wrong for the same reason the repaired smoke launcher
# (run_v1_esd_alone_control.sh) says it is wrong: gpu_reserve_hold has ALREADY exported the mapping
# it intends -- which may be a UUID rather than an index under CVD_STYLE=uuid -- and re-exporting
# the index silently discards that, defeating the reservation's own device identification. The
# reservation API owns this variable; the launcher only checks that it arrived.
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] \
  || die "gpu_reserve_hold set GPU_RESERVED=$GPU_RESERVED but exported no CUDA_VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "    reserved GPU $GPU_RESERVED (uuid ${GPU_RESERVED_UUID:-?}) CVD=$CUDA_VISIBLE_DEVICES"

# ================= STAGE 4: training ===========================================================
# --revision is passed explicitly. defend_text_to_image_lora.py has NO `variant` argument
# (verified in stage 1), so it loads the fp32 `main` weights; that is a property of the upstream
# trainer, recorded rather than worked around. The UNet is then fully overwritten by the ESD
# checkpoint at :452-454 anyway, so only text_encoder/vae/scheduler come from the base.
PIN_REVISION=$(sed -n 's/^revision=//p' MODEL_PIN.txt | head -1)
[ -n "$PIN_REVISION" ] || die "could not read revision= from MODEL_PIN.txt"
say "--- stage 4: training (revision=$PIN_REVISION, variant: unsupported upstream -> main fp32)"
# PREREG item 1 + item 3: enable the documented instrumentation. Two environment variables and
# nothing else -- upstream/IMMA stays byte-unmodified (stage 1 re-verifies its pin). The module is
# instrumentation/imma_evidence/sitecustomize.py, which CPython imports automatically because its
# directory is on PYTHONPATH. It (a) counts EXECUTED Optimizer.step() calls via a global post-step
# hook, separating the ascent optimizer (lr 1e-5, the immunisation) from the LoRA optimizer
# (lr 1e-4) by the lr read live from param_groups, and (b) fingerprints every tensor passed to
# torch.save AT THE SAVE POINT, so stage 5 can verify the file against the state actually held in
# memory rather than only against the ESD initialization.
EVID="/data/bijaypandey/t2i_unlearn/$R/$A.executed_updates.json"
# SCOPED TO THE TRAINING SUBSHELL ONLY -- these are deliberately NOT exported shell-wide.
# Why: the first smoke run exported them, so the stage-5a VERIFIER (a later $VT invocation in the
# same shell) also imported sitecustomize, re-initialised the state and OVERWROTE the training
# evidence with an empty record at import time. The verifier's own argv was sitting in the file it
# was supposed to be checking, and the run failed with "ZERO ascent updates executed" while a
# 374 MB checkpoint sat on disk. The guard was right to fail; the instrumentation plumbing was
# wrong. Passing them via `env` on the training command keeps every other process clean.
say "    instrumentation: IMMA_EVIDENCE_OUT=$EVID (scoped to the training process only)"
rc=0
( cd "$U" && env IMMA_EVIDENCE_OUT="$EVID" \
    PYTHONPATH="/data/bijaypandey/t2i_unlearn/instrumentation/imma_evidence${PYTHONPATH:+:$PYTHONPATH}" \
    $V -m accelerate.commands.launch --num_processes=1 --mixed_precision=no \
    train/defend_text_to_image_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
    --revision="$PIN_REVISION" \
    --train_data_dir="/data/bijaypandey/t2i_unlearn/$R/data/vangogh" --caption_column=prompt \
    --resolution=512 --random_flip --train_batch_size="$BATCH" --num_train_epochs="$EPOCHS" \
    --learning_rate_lora="$LR_LORA" --learning_rate="$LR_IMMA" \
    --lr_scheduler=constant --lr_warmup_steps=0 \
    --seed="$SEED" --report_to=tensorboard \
    --inner_loop_steps="$INNER" --outer_loop_steps="$OUTER" "${MS_ARG[@]}" \
    --delta_ckpt="/data/bijaypandey/t2i_unlearn/$R/diffusers-VanGogh-ESDx1-UNET.pt" \
    --output_dir="/data/bijaypandey/t2i_unlearn/$R/$A" ) >>"$L" 2>&1 || rc=$?
say "    rc=$rc $(date -Is)"
[ "$rc" -eq 0 ] || die "training exited $rc"

# ================= STAGE 5: verify the artifact ================================================
CK="$R/$A/imma_unet_xatten_layer.pt"
[ -s "$CK" ] || die "training produced no checkpoint at $CK"
say "--- stage 5: artifact verification"
# PREREG items 1 and 3. These two checks come FIRST because they are the ones the preregistration
# names as acceptance criteria, and because the pre-existing checks below (tensor count, finite
# values, rel > 1e-5 against the ESD init) are DIAGNOSTICS: they establish that something moved,
# not that the budget was executed or that the file matches what was held in memory.
# The instrumentation writes ONE FILE PER PROCESS ($EVID base + .<pid>.json), because
# `accelerate launch` loads it in the launcher AND the launcher spawns the trainer as a subprocess.
# Aggregate across them rather than trusting a single shared file: the launcher records zero steps
# and outlives the worker, so a shared file ended up holding the launcher's zeros.
EVID_BASE="${EVID%.json}"
# shellcheck disable=SC2012
EVID_N=$(ls -1 "$EVID_BASE".*.json 2>/dev/null | wc -l)
[ "$EVID_N" -gt 0 ] \
  || die "stage 5: no executed-update evidence files matching $EVID_BASE.*.json (instrumentation did not run)"
say "    executed-update evidence files: $EVID_N"
$VT - "$EVID_BASE" "$CK" "$BUDGET" <<'PY_EV' 2>&1 | tee -a "$L" \
  || die "stage 5a: executed-update / save-point verification FAILED"
import json, sys, torch
import glob
base, ckp, budp = sys.argv[1:4]
files = sorted(glob.glob(base + ".*.json"))
if not files:
    sys.exit("no evidence files matching %s.*.json" % base)
recs = []
for f in files:
    try:
        recs.append((f, json.load(open(f))))
    except Exception as e:
        sys.exit("unreadable evidence file %s: %r" % (f, e))
bud = json.load(open(budp))
print("[5a] evidence files:", len(recs))
for f, r in recs:
    print("[5a]   pid=%s steps=%s argv=%s"
          % (r.get("pid"), r.get("total_steps_observed"), (r.get("argv") or [])[:2]))
# The TRAINING process is the one that actually stepped. The accelerate launcher contributes a
# zero-step record and is reported, not silently dropped.
ev = max((r for _, r in recs), key=lambda r: r.get("total_steps_observed", 0))
print("[5a] using the record with the most observed steps: pid", ev.get("pid"))

# ---- item 1: EXECUTED vs EXPECTED ascent updates -------------------------------------------
if not ev.get("hook_installed"):
    sys.exit("the optimizer post-step hook was not installed: %r" % ev.get("hook_install_error"))
if ev.get("hook_errors"):
    sys.exit("the step hook raised: %r" % ev["hook_errors"][:3])
roles = {}
for rec in ev["optimizers"].values():
    roles.setdefault(rec["role"], 0)
    roles[rec["role"]] += rec["executed_steps"]
asc = roles.get("ascent_outer_attention", 0)
lora = roles.get("descent_inner_lora", 0)
uncl = roles.get("unclassified", 0)
expected = (bud.get("ours") or {}).get("imma_ascent_updates")
print("[5a] EXECUTED ascent updates   :", asc)
print("[5a] EXECUTED lora  updates    :", lora)
print("[5a] unclassified optimizers   :", uncl)
print("[5a] EXPECTED ascent (budget)  :", expected)
print("[5a] total steps observed      :", ev["total_steps_observed"])
if uncl:
    sys.exit("%d optimizer steps could not be attributed to the ascent or LoRA optimizer" % uncl)
if asc == 0:
    sys.exit("ZERO ascent updates executed -- the immunisation did not run")
if expected is not None and asc != expected:
    # Not fatal by itself: a skipped step is a real finding, not necessarily a failure. But it
    # must be reported loudly and must never be silently reconciled with the calculation.
    print("[5a] *** EXECUTED != EXPECTED: executed=%s expected=%s delta=%s ***"
          % (asc, expected, asc - expected))
    print("[5a] The executed count is the measurement; the expected count is arithmetic.")
else:
    print("[5a] executed == expected")

# ---- item 3: the saved file vs the state held IN MEMORY at the save point --------------------
saves = ev.get("torch_saves", [])
if not ev.get("torch_save_wrapped"):
    sys.exit("torch.save was not wrapped, so no save-point state was recorded")
mine = [r for r in saves if str(r.get("target", "")).endswith("imma_unet_xatten_layer.pt")]
if not mine:
    sys.exit("no save-point record for imma_unet_xatten_layer.pt (saves seen: %s)"
             % [r.get("target") for r in saves])
rec = mine[-1]
disk = torch.load(ckp, map_location="cpu", weights_only=False)
disk = disk.get("unet", disk) if isinstance(disk, dict) else disk
fps = rec["tensors"]
missing, mismatch, checked = [], [], 0
for k, t in (disk.items() if isinstance(disk, dict) else []):
    if not isinstance(t, torch.Tensor):
        continue
    f = fps.get(k)
    if f is None:
        missing.append(k); continue
    checked += 1
    # Byte hash of the canonicalised tensor -- exact, and independent of memory layout. The
    # earlier float64-sum equality test failed on 13/160 tensors by ~2e-16 relative purely
    # because reduction order differs between a strided CUDA-origin tensor and a contiguous
    # reloaded one. Shape and dtype are still compared exactly.
    import hashlib
    h_disk = hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if f.get("sha256") is None:
        mismatch.append(k + "(no sha256 in save-point record)")
    elif h_disk != f["sha256"] or list(t.shape) != f["shape"] or str(t.dtype) != f["dtype"]:
        mismatch.append(k)
    if t.is_floating_point() and not bool(torch.isfinite(t).all().item()):
        mismatch.append(k + "(non-finite on disk)")
print("[5a] save-point fingerprints    :", len(fps))
print("[5a] reloaded tensors checked   :", checked)
# COMPLETENESS, not just agreement. The loop above iterates the tensors PRESENT ON DISK, so a
# TRUNCATED checkpoint -- an independent reviewer demonstrated one missing 60 of 160 tensors --
# agreed on everything it contained and PASSED as "EXACTLY the in-memory state". Agreement over a
# subset is not correspondence. Every fingerprint recorded at the save point must have been
# checked against a tensor actually on disk.
unchecked = sorted(set(fps) - {k for k in (disk.keys() if isinstance(disk, dict) else [])})
if unchecked:
    sys.exit("[5a] FAIL: %d of %d save-point tensors are ABSENT from the reloaded checkpoint "
             "(truncated or partial save): %s" % (len(unchecked), len(fps), unchecked[:5]))
if checked != len(fps):
    sys.exit("[5a] FAIL: checked %d tensors but recorded %d fingerprints -- the comparison did not "
             "cover the whole save-point state" % (checked, len(fps)))
print("[5a] missing fingerprints       :", len(missing), missing[:5])
print("[5a] MISMATCHES                 :", len(mismatch), mismatch[:5])
if missing or mismatch:
    sys.exit("the reloaded checkpoint does NOT correspond to the state held in memory at the save "
             "point: %d missing, %d mismatched" % (len(missing), len(mismatch)))
print("[5a] PASS -- reloaded tensors are EXACTLY the in-memory state at the save point")

# ---- item 3, remaining clause: VERIFY THE fp16 EVALUATION EXPORT AGAINST ITS SOURCE -----------
# The preregistration requires "any fp16 evaluation export checked against the explicitly
# converted source tensors". That clause was NOT implemented, and an independent reviewer found
# the cast is IMPLICIT: scripts/gen_eval.py:56-62 builds the pipeline at torch_dtype=float16 and
# then load_state_dict()s this fp32 checkpoint into it, so every trained tensor is silently
# downcast by Module.load_state_dict's copy_. "Silently" is the problem -- an implicit narrowing
# conversion on the path between the artifact and every number reported from it.
# This verifies that the conversion the evaluator performs is EXACTLY an explicit .half() of the
# source: build an fp16 tensor, copy_ the fp32 source into it the way load_state_dict does, and
# require byte equality with source.half(). Deterministic round-to-nearest, so exact equality is
# the right test -- no tolerance to choose.
n_cast, cast_bad, lost_all = 0, [], []
for k, t in disk.items():
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        continue
    src = t.detach().cpu().contiguous()
    explicit = src.half()
    via_copy = torch.empty(src.shape, dtype=torch.float16)
    via_copy.copy_(src)                      # the conversion load_state_dict actually performs
    n_cast += 1
    if hashlib.sha256(via_copy.numpy().tobytes()).hexdigest() != \
       hashlib.sha256(explicit.numpy().tobytes()).hexdigest():
        cast_bad.append(k)
    # A trained delta that vanishes entirely at fp16 would make the evaluation blind to it.
    if not bool(torch.isfinite(explicit).all().item()):
        lost_all.append(k + "(non-finite after fp16 cast)")
print("[5b] fp16 export: tensors cast      :", n_cast)
print("[5b] fp16 export: copy_ != .half()  :", len(cast_bad), cast_bad[:5])
print("[5b] fp16 export: non-finite at fp16:", len(lost_all), lost_all[:5])
if cast_bad or lost_all:
    sys.exit("[5b] FAIL: the fp16 evaluation export does not match an explicit conversion of the "
             "source (%d mismatched, %d non-finite)" % (len(cast_bad), len(lost_all)))
print("[5b] PASS -- the evaluator's implicit fp16 cast equals an explicit .half() of the source, "
      "byte for byte, on all %d trained tensors" % n_cast)
PY_EV

$VT - "$CK" "$R/diffusers-VanGogh-ESDx1-UNET.pt" "$BUDGET" <<'PY' 2>&1 | tee -a "$L" \
  || die "stage 5: the artifact failed verification"
import hashlib, json, sys, torch
ckp, esdp, budp = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
imma = torch.load(ckp, map_location="cpu"); imma = imma.get("unet", imma)
esd = torch.load(esdp, map_location="cpu");  esd = esd.get("unet", esd)
n = sum(v.numel() for v in imma.values())
a1 = sum(1 for k in imma if ".attn1." in k); a2 = sum(1 for k in imma if ".attn2." in k)
print(f"    ckpt: {len(imma)} tensors {n/1e6:.3f}M params attn1={a1} attn2={a2}")
assert len(imma) == 160 and a1 == 80 and a2 == 80, "unexpected defence scope"
assert n == 93_536_640, f"unexpected param count {n}"
assert all(torch.isfinite(v).all() for v in imma.values()), "non-finite tensors"
assert all(k in esd for k in imma), "checkpoint keys are not UNet parameter names"
# The immunisation is measured against the ESD checkpoint it started from -- which IS the exact
# initialisation here, because defend_text_to_image_lora.py:452-454 overwrites the whole UNet with
# it. There is no variant ambiguity to guard against on this path.
d = sum((imma[k].float() - esd[k].float()).pow(2).sum() for k in imma).sqrt()
w = sum(esd[k].float().pow(2).sum() for k in imma).sqrt()
r = (d / w).item()
print(f"    ||IMMA-ESD||/||ESD|| over the 160-tensor defence scope = {r:.6f}")
assert r > 1e-5, "the defence scope did not move -- training was a no-op"

# ---- ATOMIC PUBLISH (repair A v3). ----------------------------------------------------------
# `json.dump(obj, open(path, "w"))` TRUNCATES path the instant it is opened, before a single byte
# of the new content is written. A crash, an OOM kill or a full disk in that window leaves a
# truncated or empty result file -- which is the same artifact class as the 29.7%-truncated
# unet.pt that passed every `[ -f ]` guard, except that here WE manufacture it. Worse, line 415's
# read-modify-write re-opens the very file it just parsed, so an interruption destroys the
# original too and there is nothing to fall back to.
# Same-directory temp + flush + fsync + os.replace: a reader sees the old file or the new one.
def _atomic_json(obj, path, indent=1):
    import json as _j, os as _o, tempfile as _t
    d = _o.path.dirname(_o.path.abspath(path)) or "."
    _o.makedirs(d, exist_ok=True)
    fd, tmp = _t.mkstemp(dir=d, prefix=".tmp_r1b.")
    try:
        with _o.fdopen(fd, "w") as fh:
            _j.dump(obj, fh, indent=indent)
            fh.flush(); _o.fsync(fh.fileno())
        _o.replace(tmp, path)
        dfd = _o.open(d, _o.O_RDONLY)
        try: _o.fsync(dfd)
        finally: _o.close(dfd)
    except BaseException:
        try: _o.unlink(tmp)
        except OSError: pass
        raise

with open(budp) as _fh:
    b = json.load(_fh)
b["artifact"] = {"path": ckp, "sha256": sha(ckp),
    "tensors": len(imma), "params": n, "attn1": a1, "attn2": a2,
    "rel_delta_vs_esd_init": r}
_atomic_json(b, budp)
print(f"    budget+artifact record -> {budp}")
PY
say "R1b DONE $(date -Is) -> $R/$A  (budget: $BUDGET)"
