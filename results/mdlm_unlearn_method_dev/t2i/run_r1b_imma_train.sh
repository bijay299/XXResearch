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
json.dump({"dataset_dir": d, "rows": n_all, "png_on_disk": len(files),
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
          open(out, "w"), indent=1)
print(f"    budget record -> {out}")
PY

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
export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "    reserved GPU $GPU_RESERVED"

# ================= STAGE 4: training ===========================================================
# --revision is passed explicitly. defend_text_to_image_lora.py has NO `variant` argument
# (verified in stage 1), so it loads the fp32 `main` weights; that is a property of the upstream
# trainer, recorded rather than worked around. The UNet is then fully overwritten by the ESD
# checkpoint at :452-454 anyway, so only text_encoder/vae/scheduler come from the base.
PIN_REVISION=$(sed -n 's/^revision=//p' MODEL_PIN.txt | head -1)
[ -n "$PIN_REVISION" ] || die "could not read revision= from MODEL_PIN.txt"
say "--- stage 4: training (revision=$PIN_REVISION, variant: unsupported upstream -> main fp32)"
rc=0
( cd "$U" && $V -m accelerate.commands.launch --num_processes=1 --mixed_precision=no \
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
b = json.load(open(budp)); b["artifact"] = {"path": ckp, "sha256": sha(ckp),
    "tensors": len(imma), "params": n, "attn1": a1, "attn2": a2,
    "rel_delta_vs_esd_init": r}
json.dump(b, open(budp, "w"), indent=1)
print(f"    budget+artifact record -> {budp}")
PY
say "R1b DONE $(date -Is) -> $R/$A  (budget: $BUDGET)"
