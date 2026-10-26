#!/bin/bash
# R2: the OFFICIAL Meta-Unlearning trainer with its THREE data pools, author protocol.
#
# ============================== STATUS: IMPLEMENTED, NOT RUN ==============================
# The project is FROZEN (see FREEZE_ACTIVE). This script has NEVER been executed in this form.
# It refuses to start while FREEZE_ACTIVE exists unless the implementation lead sets
# R2_FREEZE_OVERRIDE=1. Everything below is a specification that executes; nothing has run.
# ==========================================================================================
#
# AUDIT REPAIRS 2026-10-26 (docs/AGENT2_BASELINES_FINDINGS.md B-1..B-8):
#
# B-1 CONFIRMED DEFECT, now repaired. The previous version of this file built the pools, printed
#     counts, echoed "R2 POOLS DONE" and exited. It never invoked any trainer, while its own
#     header claimed "upstream src/meta.py is invoked, not our port". The full path is below.
#
# B-2 WHY esd_meta.py AND NOT meta.py. Upstream src/meta.py cannot run at commit 5a2a760:
#       * train_step returns a 4-tuple (L629) but all 11 call sites unpack 3 ->
#         ValueError on the first call of the first step;
#       * `train_loss` is read (L1045/1047/1050) and never assigned -> NameError;
#       * there is no main backward pass, so `param.grad += sum_grads[i]` (L1033) hits None.
#     src/esd_meta.py is the complete, internally consistent variant (train_step returns 3,
#     train_loss comes from train_unlearn_step L933 and is back-propagated L1003). It is therefore
#     the official artifact we can actually reproduce.
#     CONSEQUENCE FOR THE INITIAL CHECKPOINT: esd_meta.py initialises BOTH unet_teacher and
#     unet_student from --pretrained_model_name_or_path (L839-844) and performs the ESD erasure
#     itself inside train_unlearn_step. So the correct initial checkpoint for THIS variant is plain
#     CompVis/stable-diffusion-v1-4 -- there is no separately-erased checkpoint to supply. (It is
#     meta.py, the UCE/RECE path, that expects an already-erased model, and that is the broken one.)
#
# B-3 HARDCODED DATASET PATHS. esd_meta.py L671-673 loads "./dataset/{hrm,irt,tgt}" relative to the
#     PROCESS CWD with no CLI override. We therefore run the trainer from a staging directory whose
#     ./dataset symlinks to the built pools.
#
# B-4 GAMMA ARGUMENTS. upstream scripts/meta.sh L13-15 and scripts/esd_meta.sh L14-16 pass
#     --gamma2_1 THREE times and never pass --gamma2_2 or --gamma2_3. argparse keeps the last, so
#     the shipped example actually resolves to gamma2_1=0.01, gamma2_2=1.0, gamma2_3=1.0 -- the two
#     retain-side terms run at 100x and 10x their evident intent. We do NOT copy that. All five
#     gammas are passed exactly once below and echoed to the log.
#     Term semantics (esd_meta.py L979-1044): +gamma1_1*grad(irt/test), +gamma1_2*grad(tgt/test),
#     -gamma2_1*grad(hrm/test), -gamma2_2*grad(tgt/test), -gamma2_3*grad(irt/test).
#
# B-5 fix_timesteps PARSER HAZARD. esd_meta.py L117 declares `type=bool`, and bool(s) is True for
#     every non-empty string, so `--fix_timesteps False` yields True. The README (L43) says the
#     paper used fix_timesteps=False and that True exists only to accelerate training. THE ONLY WAY
#     TO GET False IS TO OMIT THE FLAG. It is deliberately absent below. Do not add it.
#     (Companion hazard: --fixed_time_steps is type=list with no nargs, so "1 2 5 10" parses to
#     ['1'] and leaves 2 5 10 as unrecognised positionals -> hard parse error. Never pass it.)
#
# B-6 POOL SIZING. upstream src/gen_images.py L90-92 generates 30 train + 30 test PER PROMPT.
#     META_PER_PROMPT below defaults to that. The historical pools were 30 per POOL
#     (10/prompt for tgt, 5/prompt for irt) -- a compute adaptation, not the author protocol.
#     Set META_SIZING=total_per_pool to reproduce them; pools.json records which was used.
#
# B-7 FORGET vs RELATED-RETAIN. The legacy pools erase "church" while retaining
#     "cathedral/chapel/basilica", which are hyponyms of church, so they can only measure TOKEN
#     suppression. META_POOL_SCHEME=coherent uses the recommended split instead (hrm = church
#     paraphrases, fgen = the hyponyms scored as FORGET, tgt = co-hyponyms + superordinate).
#     Whichever is chosen, report the claim scope printed by the pool builder verbatim.
#
# Pool meanings, verbatim from upstream/Meta-Unlearning/README.md L18:
#   hrm = the UNLEARNED concept | irt = retained, UNRELATED | tgt = retained, RELATED
# (Our manifest previously mislabelled tgt as the forget pool; corrected 2026-10-26.)
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

if [ -f FREEZE_ACTIVE ] && [ "${R2_FREEZE_OVERRIDE:-0}" != "1" ]; then
  echo "FREEZE_ACTIVE is present; refusing to launch GPU work. Set R2_FREEZE_OVERRIDE=1 to run." >&2
  exit 1
fi

R=upstream_runs/meta
L=logs/r2_meta.log
U=/data/bijaypandey/t2i_unlearn/upstream/Meta-Unlearning
STAGE="$R/stage"
VT=/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python
[ -x "$VT" ] || { echo "missing venv_t2i" >&2; exit 1; }
[ -f "$U/src/esd_meta.py" ] || { echo "missing $U/src/esd_meta.py" >&2; exit 1; }

# ---- EXPLICIT RESOLVED HYPERPARAMETERS (every one stated; none left to a parser default) ----
CONCEPT="${META_CONCEPT:-church}"
VALID_PROMPT="${META_VALID_PROMPT:-a photo of a church}"   # non-empty => intermediate ckpts saved
POOL_SCHEME="${META_POOL_SCHEME:-legacy}"
SIZING="${META_SIZING:-per_prompt}"                         # per_prompt | total_per_pool
PER_PROMPT="${META_PER_PROMPT:-30}"                         # gen_images.py value
TOTAL_PER_POOL="${META_TOTAL_PER_POOL:-30}"                 # historical adaptation
NUM_TRAIN_STEPS="${META_STEPS:-1500}"                       # esd_meta.sh L11
TRAIN_BATCH="${META_BATCH:-10}"                             # esd_meta.sh L8
NUM_IMAGES_PER_PROMPT="${META_NIPP:-10}"                    # esd_meta.sh L7
FINETUNING_METHOD="${META_FT:-full}"                        # esd_meta.sh L10
LR="${META_LR:-1e-05}"                                      # meta.sh L10 / parser default
GAMMA1_1="${META_G1_1:-0.1}"                                # +irt/test
GAMMA1_2="${META_G1_2:-0.01}"                               # +tgt/test  (esd_meta.sh L13)
GAMMA2_1="${META_G2_1:-0.1}"                                # -hrm/test
GAMMA2_2="${META_G2_2:-0.1}"                                # -tgt/test  (NOT the example's 1.0)
GAMMA2_3="${META_G2_3:-0.01}"                               # -irt/test  (NOT the example's 1.0)
SEED="${META_SEED:-0}"
RESOLUTION=512
GUIDANCE="${META_GUIDANCE:-3.0}"                            # parser default
CONCEPT_SCALE="${META_CONCEPT_SCALE:-3.0}"                  # parser default
LR_SCHEDULER=constant
LR_WARMUP=0
SAVE_EVERY="${META_SAVE_EVERY:-500}"
EVAL_EVERY="${META_EVAL_EVERY:-500}"
MAX_CKPT="${META_MAX_CKPT:-5}"
EXP_NAME="r2_esd_meta_${CONCEPT}_${POOL_SCHEME}"

# ---- 1. pools ----------------------------------------------------------------------------
mkdir -p "$R"
case "$SIZING" in
  per_prompt)     SIZE_ARG=(--per-prompt "$PER_PROMPT") ;;
  total_per_pool) SIZE_ARG=(--total-per-pool "$TOTAL_PER_POOL") ;;
  *) echo "META_SIZING must be per_prompt or total_per_pool" >&2; exit 1 ;;
esac
echo "=== R2 $(date -Is) concept=$CONCEPT scheme=$POOL_SCHEME sizing=$SIZING" >>"$L"
echo "    gammas: g1_1=$GAMMA1_1 g1_2=$GAMMA1_2 g2_1=$GAMMA2_1 g2_2=$GAMMA2_2 g2_3=$GAMMA2_3" >>"$L"
echo "    fix_timesteps: FLAG DELIBERATELY OMITTED -> resolves to False (the paper setting)" >>"$L"
echo "    steps=$NUM_TRAIN_STEPS batch=$TRAIN_BATCH lr=$LR ft=$FINETUNING_METHOD seed=$SEED" >>"$L"

while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
echo "    GPU $G" >>"$L"

rc=0
$VT scripts/r2_build_meta_pools.py --root "$R/dataset" --pool-scheme "$POOL_SCHEME" \
   "${SIZE_ARG[@]}" --resume >>"$L" 2>&1 || rc=$?
echo "  pool build rc=$rc" >>"$L"
[ "$rc" -eq 0 ] || { echo "POOL BUILD FAILED rc=$rc" >>"$L"; exit 1; }

# ---- 2. validate the pools BEFORE spending a single training step -------------------------
# The builder already validates, but the trainer reads only hrm/irt/tgt and would happily train on
# a short pool, so re-assert the counts the trainer will actually see.
$VT - "$R/dataset" <<'PY' >>"$L" 2>&1 || { echo "POOL VALIDATION FAILED" >>"$L"; exit 1; }
import glob, json, os, sys
import numpy as np
from PIL import Image
root = sys.argv[1]
man = json.load(open(os.path.join(root, "pools.json")))
bad = 0
for pool in ("hrm", "irt", "tgt"):
    exp = man["pools"][pool]["per_prompt"] * len(man["pools"][pool]["prompts"])
    for split in ("train", "test"):
        d = os.path.join(root, pool, split)
        files = sorted(glob.glob(d + "/*.png"))
        rows = [json.loads(l) for l in open(os.path.join(d, "metadata.jsonl")) if l.strip()]
        names = set(r["file_name"] for r in rows)
        disk = set(os.path.basename(f) for f in files)
        blank = [f for f in files if np.asarray(Image.open(f).convert("RGB"),
                                                dtype=np.float32).max() == 0]
        ok = (len(files) == exp and len(rows) == exp and names == disk and not blank)
        print(f"  {pool}/{split}: png={len(files)} meta={len(rows)} expected={exp} "
              f"blank={len(blank)} {'OK' if ok else 'BAD'}")
        bad += 0 if ok else 1
print(f"  pools.json: scheme={man['pool_scheme']} sizing={man['sizing']}")
print(f"  CLAIM SCOPE: {man['scheme_claim']}")
sys.exit(1 if bad else 0)
PY

# ---- 3. staging dir so upstream's hardcoded ./dataset/{hrm,irt,tgt} resolves ---------------
rm -rf "$STAGE"
mkdir -p "$STAGE"
ln -s "/data/bijaypandey/t2i_unlearn/$R/dataset" "$STAGE/dataset"
for p in hrm irt tgt; do
  [ -d "$STAGE/dataset/$p/train" ] || { echo "staging: missing dataset/$p/train" >>"$L"; exit 1; }
done

# ---- 4. the trainer -----------------------------------------------------------------------
# --devices takes TWO indices (sampling device, training device); with one visible GPU both are 0.
# --fix_timesteps and --fixed_time_steps are INTENTIONALLY ABSENT (B-5).
OUTROOT="/data/bijaypandey/t2i_unlearn/$R/out"
mkdir -p "$OUTROOT"
rc=0
( cd "$STAGE" && $VT "$U/src/esd_meta.py" \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --removing_concepts "$CONCEPT" \
    --validation_prompts "$VALID_PROMPT" \
    --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT" \
    --train_batch_size "$TRAIN_BATCH" \
    --devices 0 0 \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --finetuning_method "$FINETUNING_METHOD" \
    --learning_rate "$LR" \
    --lr_scheduler "$LR_SCHEDULER" \
    --lr_warmup_steps "$LR_WARMUP" \
    --resolution "$RESOLUTION" \
    --guidance_scale "$GUIDANCE" \
    --concept_scale "$CONCEPT_SCALE" \
    --gamma1_1 "$GAMMA1_1" \
    --gamma1_2 "$GAMMA1_2" \
    --gamma2_1 "$GAMMA2_1" \
    --gamma2_2 "$GAMMA2_2" \
    --gamma2_3 "$GAMMA2_3" \
    --seed "$SEED" \
    --save_every "$SAVE_EVERY" \
    --eval_every "$EVAL_EVERY" \
    --max_checkpoints "$MAX_CKPT" \
    --exp_name "$EXP_NAME" \
    --output_dir "$OUTROOT" \
    --logging_dir "/data/bijaypandey/t2i_unlearn/$R/logs" \
    --image_dir "/data/bijaypandey/t2i_unlearn/$R/images" ) >>"$L" 2>&1 || rc=$?
echo "  esd_meta rc=$rc $(date -Is)" >>"$L"
[ "$rc" -eq 0 ] || { echo "R2 TRAINING FAILED rc=$rc" >>"$L"; exit 1; }

# ---- 5. locate the checkpoint. esd_meta.py appends a timestamp to exp_name (L785-790),
#         so the output path is not predictable and must be discovered.
OUTDIR=$(ls -1dt "$OUTROOT/${EXP_NAME}"_* 2>/dev/null | head -1 || true)
[ -n "$OUTDIR" ] || { echo "NO OUTPUT DIR MATCHING ${EXP_NAME}_*" >>"$L"; exit 1; }
echo "  output dir: $OUTDIR" >>"$L"
ls -la "$OUTDIR" >>"$L" 2>&1
[ -d "$OUTDIR/unet" ] || { echo "FINAL PIPELINE SAVE MISSING unet/" >>"$L"; exit 1; }

# ---- 6. reload verification on CPU --------------------------------------------------------
$VT - "$OUTDIR" <<'PY' >>"$L" 2>&1 || { echo "CHECKPOINT RELOAD FAILED" >>"$L"; exit 1; }
import json, os, sys, torch
from diffusers import UNet2DConditionModel
P = "/data/bijaypandey/t2i_unlearn"
out = sys.argv[1]
pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
got = UNet2DConditionModel.from_pretrained(out, subfolder="unet", torch_dtype=torch.float32)
ref = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                           revision=pin["revision"], torch_dtype=torch.float32,
                                           variant="fp16")
g, r = got.state_dict(), ref.state_dict()
assert set(g) == set(r), f"key set mismatch: {len(set(g)^set(r))} differ"
ng = sum(v.numel() for v in g.values()); nr = sum(v.numel() for v in r.values())
assert ng == nr, f"param count {ng} != base {nr}"
assert all(torch.isfinite(v).all() for v in g.values()), "non-finite tensors in checkpoint"
d = sum((g[k]-r[k]).pow(2).sum() for k in g).sqrt()
w = sum(r[k].pow(2).sum() for k in g).sqrt()
rel = (d/w).item()
print(f"  reload OK: {len(g)} tensors {ng/1e6:.3f}M params, ||ckpt-base||/||base|| = {rel:.5f}")
assert rel > 1e-6, "the UNet is identical to base SD-1.4 -- training was a no-op"
json.dump({"output_dir": out, "tensors": len(g), "params_M": ng/1e6,
           "rel_delta_vs_base": rel}, open(os.path.join(out, "reload_check.json"), "w"), indent=1)
PY
echo "R2 DONE $(date -Is) -> $OUTDIR" >>"$L"
