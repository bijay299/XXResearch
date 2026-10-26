#!/bin/bash
# R2: the OFFICIAL Meta-Unlearning trainer with its THREE data pools, author protocol.
#
# ============================== STATUS: IMPLEMENTED, NOT RUN ==============================
# The project is FROZEN (see FREEZE_ACTIVE). This script has NEVER been executed past R2_MODE=cpu.
# GPU work is gated by scripts/gpu_reserve.sh, whose freeze gate refuses every reservation unless
# the implementation lead passes BOTH ALLOW_FROZEN_RUN=1 and --allow-frozen. Agents cannot and must
# not lift that. R2_MODE=cpu runs the complete CPU half and has been executed; see
# docs/REPAIR_BASELINES_v2.md for its output.
# ==========================================================================================
#
# SELECTED UPSTREAM VARIANT, RECORDED EXPLICITLY (repair A-5):
#   upstream/Meta-Unlearning @ 5a2a7600895a2885323cbbf599acaf3f74a96f64, entry point src/esd_meta.py
#   NOT src/meta.py. These are DIFFERENT execution paths, and meta.py is unrunnable at this commit.
#   Verified from source, and RE-VERIFIED AT RUNTIME by stage 1 below, which aborts if the defects
#   are no longer present (i.e. if someone changed the pin and the justification went stale):
#     * meta.py:626 train_step returns a 4-tuple (`...,np.mean(all_losses)`) while all 11 call
#       sites (meta.py:967,973,978,983,990,995,1001,1007,1013,1019,1025) unpack 3 ->
#       ValueError on the first call of the first step.
#     * `train_loss` is read at meta.py:1045/1047/1050 and assigned nowhere; root cause is that
#       train_unlearn_step is DEFINED at meta.py:506 and never called, so meta.py's main loop
#       contains no unlearning objective at all.
#     * no .backward() in meta.py's main(); the only two are inside train_step/train_step_deng and
#       act on a deepcopy, so `param.grad += sum_grads[i]` (meta.py:1033) hits None.
#   esd_meta.py is internally consistent: 3-tuple return (esd_meta.py:626), train_loss assigned at
#   esd_meta.py:933 from train_unlearn_step, back-propagated at esd_meta.py:1003.
#   INITIAL CHECKPOINT FOR THIS VARIANT: plain SD-1.4. esd_meta.py:839-844 builds BOTH unet_teacher
#   and unet_student from --pretrained_model_name_or_path and performs the ESD erasure itself inside
#   train_unlearn_step (esd_meta.py:506-584, target `e_0 - concept_scale*(e_p - e_0)`). There is no
#   separately-erased checkpoint to supply. It is meta.py, the already-unlearned-model path, that
#   expects one -- and that is the broken file.
#
# ENVIRONMENT (repair A-6, CONFIRMED BLOCKING DEFECT in the previous version):
#   esd_meta.py:669 does a module-level `from datasets import load_dataset`. venv_t2i has NO
#   `datasets` installed, so the previous version of this script -- which ran the trainer under
#   venv_t2i -- would have died with ModuleNotFoundError before a single step. venv_imma has
#   datasets 5.0.1 and imports esd_meta.py cleanly. The trainer therefore runs under venv_imma, and
#   stage 2 asserts the import rather than discovering it on the GPU.
#   HF_DATASETS_CACHE must also be set: the root filesystem is full, and `datasets` will otherwise
#   place its cache outside /data.
#
# POOLS: upstream_runs/meta/dataset/{hrm,irt,tgt}, validated on CONTENT, not existence (repair A-3).
# Pool meanings, verbatim from upstream/Meta-Unlearning/README.md L18:
#   hrm = the UNLEARNED concept | irt = retained, UNRELATED | tgt = retained, RELATED
#
# FORGET/RETAIN DEFINITION IS NOW REQUIRED AND EXPLICIT (repair A-2). There is NO default. The
# legacy church-vs-cathedral/chapel/basilica pools are available only as a labelled HISTORICAL
# RECONSTRUCTION, because cathedral/chapel/basilica are hyponyms of church sitting in the
# RELATED-RETAIN pool: that objective is satisfied by suppressing the TOKEN "church" and cannot
# support a concept-erasure claim. Whichever scheme runs, the claim scope printed by the pool
# builder is reproduced verbatim in the log and in the run's config record.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

R=upstream_runs/meta
L=logs/r2_meta.log
U=/data/bijaypandey/t2i_unlearn/upstream/Meta-Unlearning
STAGE="$R/stage"
UPSTREAM_PIN=5a2a7600895a2885323cbbf599acaf3f74a96f64
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python     # the ONLY venv that can import the trainer
VT=/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python     # CPU-side checks / eval compatibility

MODE="${R2_MODE:-cpu}"          # cpu | smoke | full   -- cpu is the default and needs no GPU
case "$MODE" in cpu|smoke|full) ;; *) echo "R2_MODE must be cpu|smoke|full" >&2; exit 2 ;; esac

# ---- the forget/retain definition: EXPLICIT, RESOLVED, DOCUMENTED. No default. ----------------
POOL_SCHEME="${META_POOL_SCHEME:-}"
if [ -z "$POOL_SCHEME" ]; then
  cat >&2 <<'EOF'
REFUSING TO RUN: META_POOL_SCHEME is not set, and this script has no default.

A forget/retain definition is the scientific content of the run; it must be chosen deliberately.

  META_POOL_SCHEME=coherent
      hrm  = church paraphrases                        (FORGET)
      fgen = cathedral / chapel / basilica             (FORGET-GENERALISATION, eval pool only)
      tgt  = mosque/synagogue/temple/castle/town hall/stone building   (RELATED RETAIN)
      irt  = COCO objects                              (UNRELATED RETAIN)
    Use this for any run intended to support a concept-erasure claim.

  META_POOL_SCHEME=legacy  META_HISTORICAL_RECONSTRUCTION=1
      hrm = church (FORGET); tgt = cathedral/chapel/basilica (RELATED RETAIN); irt = COCO objects
    HISTORICAL RECONSTRUCTION ONLY. cathedral/chapel/basilica are hyponyms of church held in the
    RETAIN pool, so the objective is satisfied by suppressing the TOKEN "church". This measures
    token suppression, NOT concept erasure. It reproduces the historical setup; it cannot support
    a new scientific claim. The acknowledgement variable is mandatory.
EOF
  exit 2
fi
# The scheme must be one the builder actually knows, and that is asked of the BUILDER rather than
# hardcoded here. Without this, an unknown scheme fell through to a pool root that does not exist,
# stage 4 silently skipped, and R2_MODE=cpu exited 0 on a nonsense configuration.
KNOWN_SCHEMES=$(/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python -c \
  "import importlib.util as u;s=u.spec_from_file_location('m','scripts/r2_build_meta_pools.py');\
m=u.module_from_spec(s);s.loader.exec_module(m);print(' '.join(sorted(m.SCHEMES)))" 2>/dev/null) \
  || { echo "could not read the pool schemes from scripts/r2_build_meta_pools.py" >&2; exit 1; }
case " $KNOWN_SCHEMES " in
  *" $POOL_SCHEME "*) ;;
  *) echo "REFUSING TO RUN: META_POOL_SCHEME='$POOL_SCHEME' is not a scheme the pool builder" >&2
     echo "  knows. Known schemes: $KNOWN_SCHEMES" >&2
     exit 2 ;;
esac

LEGACY_ACK=()
if [ "$POOL_SCHEME" = "legacy" ]; then
  if [ "${META_HISTORICAL_RECONSTRUCTION:-0}" != "1" ]; then
    echo "REFUSING TO RUN: META_POOL_SCHEME=legacy requires META_HISTORICAL_RECONSTRUCTION=1." >&2
    echo "  The legacy pools measure TOKEN SUPPRESSION, not concept erasure (see header)." >&2
    exit 2
  fi
  LEGACY_ACK=(--historical-reconstruction)
fi

# Pool root, defaulted PER SCHEME. The legacy pools already occupy $R/dataset and are preserved;
# a coherent build must not land on top of them, so it gets its own root unless overridden.
case "$POOL_SCHEME" in
  legacy)   POOL_ROOT="${META_POOL_ROOT:-$R/dataset}" ;;
  coherent) POOL_ROOT="${META_POOL_ROOT:-$R/dataset_coherent}" ;;
  *)        POOL_ROOT="${META_POOL_ROOT:-$R/dataset_$POOL_SCHEME}" ;;
esac

# ---- HYPERPARAMETERS IN THREE EXPLICITLY SEPARATED TIERS (repair A-4) -------------------------
# TIER P = PAPER-SUPPORTED: the paper itself states the value.
# TIER R = README / shipped-example value, or the argparse default where no example passes it.
# TIER I = OUR INFERRED REPAIR: upstream's example is erroneous or silent; the value is OUR
#          reconstruction of the evident intent. These are NOT "author settings" and must never be
#          reported as such.
#
# TIER P
#   fix_timesteps = False -- README.md:43 "We use fix_timesteps = False in our paper". The parser
#     declares `type=bool` (esd_meta.py:117), under which bool("False") is True, so
#     `--fix_timesteps False` yields TRUE. The ONLY way to obtain False is to OMIT the flag.
#     It is deliberately absent from the invocation below. DO NOT ADD IT.
#     Companion hazard: --fixed_time_steps is `type=list` with no nargs (esd_meta.py:118), so
#     "1 2 5 10" parses to ['1'] and leaves 2/5/10 as unrecognised positionals -> hard SystemExit.
#     Never pass it either; its default [1,2,5,10] is the only usable value and is unused when
#     fix_timesteps is False.
# TIER R
CONCEPT="${META_CONCEPT:-church}"                   # R: --removing_concepts, example is "nudity"
VALID_PROMPT="${META_VALID_PROMPT:-a photo of a church}"  # R: esd_meta.sh L5-6 passes one
NUM_IMAGES_PER_PROMPT="${META_NIPP:-10}"            # R: esd_meta.sh L7 (parser default is 1)
TRAIN_BATCH="${META_BATCH:-10}"                     # R: esd_meta.sh L8  (parser default 1)
NUM_TRAIN_STEPS="${META_STEPS:-1500}"               # R: esd_meta.sh L10 (parser default 1000)
FINETUNING_METHOD="${META_FT:-full}"                # R: esd_meta.sh L11 (parser default "xattn")
GAMMA1_1="${META_G1_1:-0.1}"                        # R: esd_meta.sh L12  -> +grad(irt/test)
GAMMA1_2="${META_G1_2:-0.01}"                       # R: esd_meta.sh L13  -> +grad(tgt/test)
LR="${META_LR:-1e-05}"                              # R: parser default 1e-5; meta.sh L10 agrees
GUIDANCE="${META_GUIDANCE:-3.0}"                    # R: parser default, no example passes it
CONCEPT_SCALE="${META_CONCEPT_SCALE:-3.0}"          # R: parser default, no example passes it
RESOLUTION="${META_RES:-512}"                       # R: parser default
LR_SCHEDULER="${META_LR_SCHED:-constant}"           # R: parser default
# TIER I -- OUR INFERRED REPAIRS. Each one states what upstream actually does and why we deviate.
#
# I-1 THE GAMMA2 TRIPLICATE. upstream scripts/esd_meta.sh L14-16 (and scripts/meta.sh L13-15) spell
#     `--gamma2_1` THREE TIMES and never pass --gamma2_2 or --gamma2_3. argparse keeps the last, so
#     the shipped example RESOLVES TO gamma2_1=0.01, gamma2_2=1.0 (default), gamma2_3=1.0 (default)
#     -- the two retain-side terms run at 100x and 10x the positive terms that are meant to balance
#     them (gamma1_2=0.01 vs gamma2_2=1.0; gamma1_1=0.1 vs gamma2_3=1.0), so AS SCRIPTED both
#     retain pools are net-ASCENDED. The evident intent of three successive --gamma2_* flags with
#     values 0.1, 0.1, 0.01 is (gamma2_1, gamma2_2, gamma2_3) = (0.1, 0.1, 0.01). We pass that.
#     IT IS OUR INFERENCE. It is NOT the author setting, and the resolved example values are
#     recorded alongside it in the config record so the difference is never lost.
GAMMA2_1="${META_G2_1:-0.1}"                        # I-1: -grad(hrm/test) ; example resolves 0.01
GAMMA2_2="${META_G2_2:-0.1}"                        # I-1: -grad(tgt/test) ; example resolves 1.0
GAMMA2_3="${META_G2_3:-0.01}"                       # I-1: -grad(irt/test) ; example resolves 1.0
# I-2 SEED. Both upstream scripts pass no --seed and the parser default is None, i.e. upstream runs
#     are not reproducible. We pin one.
SEED="${META_SEED:-0}"
# I-3 LR WARMUP. Parser default is 500 (esd_meta.py:128) and no example passes it. Under
#     lr_scheduler="constant" diffusers' get_scheduler ignores num_warmup_steps, so the value is
#     IMMATERIAL as long as the scheduler stays "constant" -- but 0 is what the config means.
LR_WARMUP="${META_LR_WARMUP:-0}"
# I-4 CHECKPOINTING CADENCE. Parser defaults are save_every=100, eval_every=1000. Operational
#     choice only; it does not enter the objective.
SAVE_EVERY="${META_SAVE_EVERY:-500}"
EVAL_EVERY="${META_EVAL_EVERY:-500}"
MAX_CKPT="${META_MAX_CKPT:-5}"
# I-5 POOL CONSTRUCTION. upstream src/gen_images.py generates 30 train + 30 test PER PROMPT
#     (gen_images.py:88-92) but writes them to ./dataset/style and ./dataset/paint
#     (gen_images.py:50,69) -- NEITHER of which is a directory the trainer reads. The trainer
#     hardcodes ./dataset/{hrm,irt,tgt} (esd_meta.py:671-673). The shipped generator therefore does
#     not produce the pools the shipped trainer loads; the mapping must be supplied. Ours is
#     scripts/r2_build_meta_pools.py. Sizing is explicit and recorded.
SIZING="${META_SIZING:-per_prompt}"                 # per_prompt | total_per_pool
PER_PROMPT="${META_PER_PROMPT:-30}"                 # I-5: gen_images.py:88-92 value
TOTAL_PER_POOL="${META_TOTAL_PER_POOL:-30}"         # I-5: the historical compute adaptation
# I-6 MODEL PIN. Neither upstream script passes --revision or --variant (both default None), so an
#     upstream run floats on whatever `main` resolves to. esd_meta.py:839-844 DOES forward both to
#     from_pretrained, so we pin them from MODEL_PIN.txt. This is also what makes the stage-6
#     reload check meaningful -- see the warning there.
PIN_REVISION=$(sed -n 's/^revision=//p' MODEL_PIN.txt | head -1)
[ -n "$PIN_REVISION" ] || { echo "could not read revision= from MODEL_PIN.txt" >&2; exit 1; }
PIN_VARIANT="${META_VARIANT:-fp16}"                 # I-6: the variant the rest of the project uses
# I-7 DEVICES. esd_meta.py:828-830 requires TWO cuda indices: devices[0] trains the student,
#     devices[1] holds teacher+text_encoder+vae. `--devices 0` raises IndexError at
#     esd_meta.py:886 and nargs="+" does not catch it. Both upstream scripts pass `0 0`, i.e.
#     co-resident on one device at ~2x memory, on a repo whose README:15 cites 80 GB A100s. Ours
#     are 40 GB. See the memory warning in docs/REPAIR_BASELINES_v2.md before choosing.
DEVICES="${META_DEVICES:-0 0}"
EXP_NAME="r2_esd_meta_${CONCEPT}_${POOL_SCHEME}"
[ "$MODE" = "smoke" ] && EXP_NAME="${EXP_NAME}_smoke"

mkdir -p "$R" logs
say() { printf '%s\n' "$*" | tee -a "$L"; }
die() { printf 'R2 FAILED: %s\n' "$*" | tee -a "$L" >&2; exit 1; }

say "=== R2 $(date -Is) mode=$MODE concept=$CONCEPT scheme=$POOL_SCHEME sizing=$SIZING"
say "    pool root: $POOL_ROOT"
say "    upstream variant: src/esd_meta.py @ $UPSTREAM_PIN  (NOT src/meta.py -- see header)"
say "    TIER R: nipp=$NUM_IMAGES_PER_PROMPT batch=$TRAIN_BATCH steps=$NUM_TRAIN_STEPS"
say "            ft=$FINETUNING_METHOD lr=$LR g1_1=$GAMMA1_1 g1_2=$GAMMA1_2 cfg=$GUIDANCE"
say "            concept_scale=$CONCEPT_SCALE res=$RESOLUTION sched=$LR_SCHEDULER"
say "    TIER P: fix_timesteps=False  (FLAG OMITTED -- the only way to obtain False)"
say "    TIER I: g2_1=$GAMMA2_1 g2_2=$GAMMA2_2 g2_3=$GAMMA2_3  <-- OUR INFERRED REPAIR."
say "            upstream's example RESOLVES to g2_1=0.01 g2_2=1.0 g2_3=1.0; not author settings."
say "    TIER I: seed=$SEED warmup=$LR_WARMUP save_every=$SAVE_EVERY eval_every=$EVAL_EVERY"
say "    TIER I: revision=$PIN_REVISION variant=$PIN_VARIANT devices='$DEVICES'"

# ================================ STAGE 1: upstream pin + variant justification ================
# File existence and a successful import are NOT sufficient. Re-assert, from source, the defects
# that make src/meta.py unusable, so the recorded choice of esd_meta.py cannot go stale silently.
say "--- stage 1: upstream pin and variant justification"
GOT_PIN=$(git -C "$U" rev-parse HEAD 2>/dev/null || echo "NO-GIT")
[ "$GOT_PIN" = "$UPSTREAM_PIN" ] || die "upstream/Meta-Unlearning is at $GOT_PIN, expected $UPSTREAM_PIN"
say "    pin OK: $GOT_PIN"
for f in src/meta.py src/esd_meta.py src/gen_images.py README.md; do
  [ -f "$U/$f" ] || die "missing $U/$f"
done
$VT - "$U/src/meta.py" "$U/src/esd_meta.py" <<'PY' 2>&1 | tee -a "$L" || die "stage 1 (variant justification)"
import re, sys
m = open(sys.argv[1]).read(); e = open(sys.argv[2]).read()
FOUR = "return task_unet,task_optimizer,task_lr_scheduler,np.mean(all_losses)"
THREE = "return task_unet,task_optimizer,task_lr_scheduler\n"
sites = re.findall(r"task_unet\s*,\s*task_optimizer(?:_full)?\s*,\s*"
                   r"task_lr_scheduler(?:_full)?\s*=\s*train_step\(", m)
checks = [
    ("meta.py train_step returns a 4-tuple",                 FOUR in m),
    ("meta.py has 11 three-value call sites",                len(sites) == 11),
    ("meta.py never assigns train_loss",                     not re.search(r"^\s*train_loss\s*=", m, re.M)),
    ("meta.py reads train_loss anyway",                      "train_loss.item()" in m),
    ("meta.py never back-propagates train_loss",             "train_loss.backward()" not in m),
    ("meta.py never calls train_unlearn_step",               len(re.findall(r"train_unlearn_step", m)) == 1),
    ("esd_meta.py train_step returns a 3-tuple",             THREE in e),
    ("esd_meta.py assigns train_loss",                       bool(re.search(r"^\s*train_loss\s*=", e, re.M))),
    ("esd_meta.py back-propagates train_loss",               "train_loss.backward()" in e),
    ("esd_meta.py hardcodes ./dataset/{hrm,irt,tgt}",        all(f'data_dir="./dataset/{p}"' in e
                                                                 for p in ("hrm", "irt", "tgt"))),
    ("esd_meta.py forwards --revision and --variant",        "revision=args.revision" in e and
                                                             "variant=args.variant" in e),
    ("fix_timesteps is still type=bool (the hazard)",        'add_argument("--fix_timesteps", type=bool' in e),
]
bad = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(f"    [{'OK ' if ok else 'BAD'}] {n}")
print(f"    meta.py 3-value train_step call sites found: {len(sites)}")
if bad:
    print("    THE RECORDED JUSTIFICATION FOR USING esd_meta.py NO LONGER HOLDS: " + "; ".join(bad))
    sys.exit(1)
PY

# ================================ STAGE 2: environment =========================================
say "--- stage 2: environment"
[ -x "$V" ]  || die "missing venv_imma ($V)"
[ -x "$VT" ] || die "missing venv_t2i ($VT)"
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_DATASETS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/datasets   # root fs is FULL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_DATASETS_CACHE"
# esd_meta.py:669 imports `datasets` at module level. venv_t2i does not have it. Assert, do not
# discover this on a GPU.
$V -c "
import datasets, diffusers, torch, transformers
print(f'    venv_imma: datasets {datasets.__version__} diffusers {diffusers.__version__} '
      f'torch {torch.__version__} transformers {transformers.__version__}')
" 2>/dev/null | tee -a "$L" || die "venv_imma cannot import datasets/diffusers/torch (stage 2)"
$VT -c "import datasets" 2>/dev/null \
  && say "    note: venv_t2i now has datasets too; venv_imma is still the pinned trainer env" \
  || say "    confirmed: venv_t2i has NO datasets -- running the trainer there would ImportError"

# ================================ STAGE 3: argument parsing ====================================
# A real parse of the EXACT argv, via the trainer's own parse_args(). Catches the type=bool and
# type=list hazards, unrecognised arguments, and any upstream signature drift.
say "--- stage 3: argument parsing (the trainer's own parse_args, no model loaded)"
ARGS=(
  --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4
  --revision "$PIN_REVISION"
  --variant "$PIN_VARIANT"
  --removing_concepts "$CONCEPT"
  --validation_prompts "$VALID_PROMPT"
  --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
  --train_batch_size "$TRAIN_BATCH"
  --devices $DEVICES
  --num_train_steps "$NUM_TRAIN_STEPS"
  --finetuning_method "$FINETUNING_METHOD"
  --learning_rate "$LR"
  --lr_scheduler "$LR_SCHEDULER"
  --lr_warmup_steps "$LR_WARMUP"
  --resolution "$RESOLUTION"
  --guidance_scale "$GUIDANCE"
  --concept_scale "$CONCEPT_SCALE"
  --gamma1_1 "$GAMMA1_1" --gamma1_2 "$GAMMA1_2"
  --gamma2_1 "$GAMMA2_1" --gamma2_2 "$GAMMA2_2" --gamma2_3 "$GAMMA2_3"
  --seed "$SEED"
  --save_every "$SAVE_EVERY"
  --eval_every "$EVAL_EVERY"
  --max_checkpoints "$MAX_CKPT"
  --exp_name "$EXP_NAME"
  --output_dir "/data/bijaypandey/t2i_unlearn/$R/out"
  --logging_dir "/data/bijaypandey/t2i_unlearn/$R/logs"
  --image_dir "/data/bijaypandey/t2i_unlearn/$R/images"
)
# --fix_timesteps / --fixed_time_steps are INTENTIONALLY ABSENT (TIER P, above).
for bad_arg in --fix_timesteps --fixed_time_steps; do
  case " ${ARGS[*]} " in *" $bad_arg "*) die "$bad_arg must never be passed (TIER P)" ;; esac
done
R2_DRYPARSE=1 $V - "$U/src/esd_meta.py" "${ARGS[@]}" <<'PY' 2>&1 | tail -25 | tee -a "$L" \
  || die "stage 3: the trainer's parser rejected our argv"
import importlib.util, json, sys
src = sys.argv[1]
spec = importlib.util.spec_from_file_location("esd_meta_mod", src)
mod = importlib.util.module_from_spec(spec)
sys.argv = ["esd_meta.py"] + sys.argv[2:]
spec.loader.exec_module(mod)
a = mod.parse_args()
assert a.fix_timesteps is False, f"fix_timesteps resolved to {a.fix_timesteps!r}, expected False"
assert a.variant and a.revision, "revision/variant did not reach the parser"
assert len(a.devices) == 2, f"--devices resolved to {a.devices}; esd_meta.py:886 needs two"
resolved = {k: getattr(a, k) for k in (
    "pretrained_model_name_or_path", "revision", "variant", "removing_concepts",
    "validation_prompts", "num_images_per_prompt", "train_batch_size", "devices",
    "num_train_steps", "finetuning_method", "learning_rate", "lr_scheduler", "lr_warmup_steps",
    "resolution", "guidance_scale", "concept_scale", "gamma1_1", "gamma1_2", "gamma2_1",
    "gamma2_2", "gamma2_3", "seed", "save_every", "eval_every", "max_checkpoints", "exp_name",
    "fix_timesteps", "fixed_time_steps", "gradient_accumulation_steps", "use_fp16")}
print("    RESOLVED ARGS: " + json.dumps(resolved, default=str))
print("    fix_timesteps -> False (by omission). devices -> " + str(a.devices))
PY

# ================================ STAGE 4: pools (CONTENT, not existence) =======================
say "--- stage 4: pool content validation"
case "$SIZING" in
  per_prompt)     SIZE_ARG=(--per-prompt "$PER_PROMPT") ;;
  total_per_pool) SIZE_ARG=(--total-per-pool "$TOTAL_PER_POOL") ;;
  *) die "META_SIZING must be per_prompt or total_per_pool" ;;
esac
POOLS_EXIST=1
for p in hrm irt tgt; do
  for s in train test; do
    [ -f "$POOL_ROOT/$p/$s/metadata.jsonl" ] || POOLS_EXIST=0
  done
done
# STAGE_4_RAN tracks whether the pool contract was actually CHECKED. A stage that could not run
# must never be reported as passed, and must never produce a success exit -- which is exactly what
# the previous version did: an absent pool root made stage 4 skip, and R2_MODE=cpu still printed
# "stages 1-4 passed" and exited 0 (reproduced with META_POOL_SCHEME=bogus -> rc=0).
STAGE_4_RAN=0
if [ "$POOLS_EXIST" = 1 ]; then
  # Validates counts, per-prompt proportions, SPLIT ROLES (captions must be this pool's prompts),
  # blank frames, and content fingerprints -- and reports a definition mismatch as RESUME
  # INCOMPATIBLE naming the field, rather than as a bare count error. No GPU, no torch.
  # A split with no recorded fingerprint is an ERROR, not a note: see --backfill-fingerprints.
  $VT scripts/r2_build_meta_pools.py --root "$POOL_ROOT" --pool-scheme "$POOL_SCHEME" \
     "${LEGACY_ACK[@]}" "${SIZE_ARG[@]}" --verify-only 2>&1 | tee -a "$L" \
     || die "stage 4: the pools on disk do not satisfy the requested definition (see above)"
  STAGE_4_RAN=1
else
  if [ "$MODE" = "cpu" ]; then
    say "    STAGE 4 COULD NOT RUN: no pools at $POOL_ROOT, and building them needs a GPU."
  else
    say "    pools absent -> building (GPU)"
    $V scripts/r2_build_meta_pools.py --root "$POOL_ROOT" --pool-scheme "$POOL_SCHEME" \
       "${LEGACY_ACK[@]}" "${SIZE_ARG[@]}" --resume 2>&1 | tee -a "$L" \
       || die "stage 4: pool build failed"
    $VT scripts/r2_build_meta_pools.py --root "$POOL_ROOT" --pool-scheme "$POOL_SCHEME" \
       "${LEGACY_ACK[@]}" "${SIZE_ARG[@]}" --backfill-fingerprints 2>&1 | tee -a "$L" \
       || die "stage 4: could not record pool fingerprints after building"
    STAGE_4_RAN=1
  fi
fi

# A config record describes a configuration whose pools were VALIDATED. If stage 4 could not run,
# no record is written -- a record with "pools": null asserts nothing and reads like a pass.
if [ "$STAGE_4_RAN" != 1 ]; then
  say ""
  say "R2 CPU VERIFICATION INCOMPLETE $(date -Is)"
  say "  stages 1-3 PASSED. STAGE 4 (pool content) DID NOT RUN -- the pools at $POOL_ROOT do not"
  say "  exist, so nothing has validated them and no config record was written."
  say "  Pools for scheme '$POOL_SCHEME' must be built on a GPU first:"
  say "    ALLOW_FROZEN_RUN=1 META_POOL_SCHEME=$POOL_SCHEME R2_MODE=smoke bash run_r2_meta_official.sh"
  exit 3
fi

# ---- the config record: every resolved value, with its tier, beside the pools' claim scope -----
mkdir -p "$R/out"
CFG="$R/out/${EXP_NAME}_config.json"
$VT - "$CFG" "$POOL_ROOT/pools.json" <<PY 2>&1 | tee -a "$L"
import json, os, sys
out, poolsp = sys.argv[1], sys.argv[2]
pools = json.load(open(poolsp)) if os.path.isfile(poolsp) else None
rec = {
 "upstream": {"repo": "upstream/Meta-Unlearning", "commit": "$UPSTREAM_PIN",
              "entry_point": "src/esd_meta.py",
              "rejected_entry_point": "src/meta.py (unrunnable at this commit; see stage 1)"},
 "venv": "venv_imma (the only env that can import esd_meta.py: datasets is required)",
 "mode": "$MODE",
 "tier_P_paper_supported": {"fix_timesteps": False,
   "evidence": "README.md:43; obtained by OMITTING the flag because type=bool makes 'False' truthy"},
 "tier_R_readme_or_parser_default": {
   "removing_concepts": "$CONCEPT", "num_images_per_prompt": $NUM_IMAGES_PER_PROMPT,
   "train_batch_size": $TRAIN_BATCH, "num_train_steps": $NUM_TRAIN_STEPS,
   "finetuning_method": "$FINETUNING_METHOD", "learning_rate": $LR,
   "gamma1_1": $GAMMA1_1, "gamma1_2": $GAMMA1_2,
   "guidance_scale": $GUIDANCE, "concept_scale": $CONCEPT_SCALE,
   "resolution": $RESOLUTION, "lr_scheduler": "$LR_SCHEDULER"},
 "tier_I_our_inferred_repairs": {
   "gamma2_1": $GAMMA2_1, "gamma2_2": $GAMMA2_2, "gamma2_3": $GAMMA2_3,
   "gamma2_note": ("upstream esd_meta.sh L14-16 passes --gamma2_1 three times and never passes "
                   "--gamma2_2/--gamma2_3, so the SHIPPED EXAMPLE RESOLVES TO "
                   "gamma2_1=0.01, gamma2_2=1.0, gamma2_3=1.0. Our values reconstruct the evident "
                   "intent (0.1, 0.1, 0.01). THESE ARE OUR INFERENCE, NOT AUTHOR SETTINGS."),
   "upstream_example_resolves_to": {"gamma2_1": 0.01, "gamma2_2": 1.0, "gamma2_3": 1.0},
   "seed": $SEED, "seed_note": "upstream passes no --seed; parser default is None",
   "lr_warmup_steps": $LR_WARMUP,
   "lr_warmup_note": "parser default 500; immaterial while lr_scheduler=='constant'",
   "save_every": $SAVE_EVERY, "eval_every": $EVAL_EVERY, "max_checkpoints": $MAX_CKPT,
   "revision": "$PIN_REVISION", "variant": "$PIN_VARIANT",
   "pin_note": ("neither upstream script passes --revision/--variant (both default None); "
                "esd_meta.py:839-844 forwards them, so we pin from MODEL_PIN.txt"),
   "devices": "$DEVICES",
   "pool_construction": "scripts/r2_build_meta_pools.py (gen_images.py writes style/paint, "
                        "never the hrm/irt/tgt the trainer loads)"},
 "pool_scheme": "$POOL_SCHEME",
 "historical_reconstruction": ("${META_HISTORICAL_RECONSTRUCTION:-0}" == "1"),
 "pools_validated": True,
 "pools": None if pools is None else {
   "scheme": pools.get("pool_scheme"), "sizing": pools.get("sizing"),
   "claim_scope": pools.get("scheme_claim"),
   "per_pool": {k: {"prompts": v.get("prompts"), "per_prompt": v.get("per_prompt"),
                    "counts": v.get("counts"), "fingerprints": v.get("fingerprints"),
                    "scored_as": v.get("scored_as")}
                for k, v in (pools.get("pools") or {}).items()}},
}
json.dump(rec, open(out, "w"), indent=1)
print(f"    config record -> {out}")
if pools: print(f"    CLAIM SCOPE: {pools.get('scheme_claim')}")
PY

if [ "$MODE" = "cpu" ]; then
  say "R2 CPU VERIFICATION COMPLETE $(date -Is) -- stages 1,2,3,4 ALL RAN and passed."
  say "  Pools validated on content at $POOL_ROOT with fingerprints BOUND."
  say "  Stages 7 (minimal real update), 9 (reload guard) and 10 (eval compat) need a GPU and have"
  say "  NOT run. Next: R2_MODE=smoke."
  exit 0
fi

# ================================ STAGE 5: GPU reservation =====================================
# scripts/gpu_reserve.sh is owned by the Infrastructure agent and is CALLED, never reimplemented.
# The retired scripts/campaign_free.sh selection (`G=$(bash scripts/campaign_free.sh|head -1)`) is
# GONE: it was check-then-select with no lock held (register D-001).
# CONTRACT NOTE, verified by CPU test: gpu_reserve_hold's signature is
#   gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]
# so --allow-frozen MUST be the THIRD argument. Passing it second silently binds it to
# wait_seconds, the override never reaches `acquire`, and the call fails with rc=4 under a freeze.
# Do NOT wrap this in $( ): gpu_reserve_hold installs the release trap in the CALLING shell.
say "--- stage 5: GPU reservation via scripts/gpu_reserve.sh"
[ -f scripts/gpu_reserve.sh ] || die "scripts/gpu_reserve.sh is missing"
if [ -f FREEZE_ACTIVE ]; then
  say "    FREEZE_ACTIVE is present. A reservation requires BOTH ALLOW_FROZEN_RUN=1 and the"
  say "    explicit --allow-frozen argument, and every override is logged. This is the"
  say "    implementation lead's decision, not this script's."
fi
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
# shellcheck source=scripts/gpu_reserve.sh
source scripts/gpu_reserve.sh
gpu_reserve_hold "r2_esd_meta_${MODE}" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null \
  || die "no GPU reservation granted (gpu_reserve_hold rc=$?); see its stderr above"
[ -n "${GPU_RESERVED:-}" ] || die "gpu_reserve_hold returned 0 but set no GPU_RESERVED"
export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
say "    reserved GPU $GPU_RESERVED (released automatically on exit by gpu_reserve.sh's trap)"

# ================================ STAGE 6: staging dir =========================================
# esd_meta.py:671-673 loads ./dataset/{hrm,irt,tgt} relative to the PROCESS CWD with no CLI
# override, so the trainer runs from a staging directory whose ./dataset points at the built pools.
# Verified on CPU: load_dataset("imagefolder", data_dir="./dataset/hrm") resolves through this
# symlink and yields train=30/test=30 with the expected captions.
say "--- stage 6: staging directory for the hardcoded ./dataset path"
rm -rf "$STAGE"; mkdir -p "$STAGE"
ln -s "/data/bijaypandey/t2i_unlearn/$POOL_ROOT" "$STAGE/dataset"
for p in hrm irt tgt; do
  for s in train test; do
    [ -f "$STAGE/dataset/$p/$s/metadata.jsonl" ] || die "staging: dataset/$p/$s is not reachable"
  done
done
( cd "$STAGE" && $V -c "
from datasets import load_dataset
for p in ('hrm','irt','tgt'):
    ds = load_dataset('imagefolder', data_dir=f'./dataset/{p}')
    n = {k: len(v) for k, v in ds.items()}
    t = sorted(set(ds['train']['text']))
    print(f'    {p}: {n} prompts={len(t)} first={t[0]!r}')
" ) 2>&1 | grep -E "^    " | tee -a "$L" || die "stage 6: the trainer's dataset path does not resolve"

# ================================ STAGE 7: the trainer =========================================
OUTROOT="/data/bijaypandey/t2i_unlearn/$R/out"
mkdir -p "$OUTROOT" "$R/logs" "$R/images"
if [ "$MODE" = "smoke" ]; then
  # A MINIMAL REAL UPDATE: few steps, but save_every=1 so a step= checkpoint is actually written
  # and can be diffed against the pinned initialisation. This is the step that distinguishes
  # "the script runs" from "the script trains".
  SM_STEPS="${META_SMOKE_STEPS:-2}"
  say "--- stage 7: MINIMAL REAL UPDATE ($SM_STEPS steps, save_every=1)"
  for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
      --num_train_steps) ARGS[$((i+1))]="$SM_STEPS" ;;
      --save_every)      ARGS[$((i+1))]=1 ;;
      --eval_every)      ARGS[$((i+1))]="$((SM_STEPS + 1))" ;;   # skip validation sampling
    esac
  done
else
  say "--- stage 7: full training ($NUM_TRAIN_STEPS steps)"
fi
rc=0
( cd "$STAGE" && $V "$U/src/esd_meta.py" "${ARGS[@]}" ) >>"$L" 2>&1 || rc=$?
say "    esd_meta rc=$rc $(date -Is)"
[ "$rc" -eq 0 ] || die "the trainer exited $rc (tail of $L above). If it is a CUDA OOM, see the
  memory note in docs/REPAIR_BASELINES_v2.md: --devices '0 0' co-locates teacher and student."

# ================================ STAGE 8: locate the artifacts ================================
# esd_meta.py:785-790 appends a timestamp to exp_name, so the output path is not predictable.
say "--- stage 8: locate the saved checkpoint"
OUTDIR=$(ls -1dt "$OUTROOT/${EXP_NAME}"_* 2>/dev/null | head -1 || true)
[ -n "$OUTDIR" ] || die "no output dir matching $OUTROOT/${EXP_NAME}_*"
say "    output dir: $OUTDIR"
ls -la "$OUTDIR" >>"$L" 2>&1
# save_checkpoint(step=None) writes a full StableDiffusionPipeline at the run-dir root
# (esd_meta.py:340-349); periodic saves write unet-only into step=NNNNNN/ (esd_meta.py:328-339).
STEPDIR=$(ls -1dt "$OUTDIR"/step=* 2>/dev/null | head -1 || true)
if [ "$MODE" = "smoke" ]; then
  [ -n "$STEPDIR" ] || die "no step=*/ checkpoint was written; no minimal real update is provable"
  CKDIR="$STEPDIR"; CKSUB=""
  say "    periodic checkpoint: $STEPDIR"
else
  [ -d "$OUTDIR/unet" ] || die "the final pipeline save produced no unet/ subfolder"
  CKDIR="$OUTDIR"; CKSUB="unet"
  say "    final pipeline save: $OUTDIR/unet"
fi

# ================================ STAGE 9: reload vs the EXACT SAME init ========================
# TWO DEFECTS IN THE HISTORY OF THIS GUARD, both now closed, and they point OPPOSITE WAYS:
#
#  v1 PASSED A NO-OP. It compared the saved UNet against base SD-1.4 loaded with variant="fp16"
#     and asserted rel > 1e-6. The trainer was loading the MAIN weights, and main-vs-fp16-variant
#     alone differs in ALL 686 tensors at rel 2.074e-04 (the fp16 variant is bit-exactly
#     main.half()). An untrained checkpoint therefore scored 207x the threshold and PASSED.
#  v2 FAILED GENUINE TRAINING. Fixing v1 by demanding rel > 10 x the variant floor (2.074e-03)
#     put the threshold 4.2x ABOVE the largest delta two steps can physically produce (see the
#     bound below, 4.88e-04 at steps=2, lr=1e-5). R2_MODE=smoke -- whose entire purpose is to
#     prove the trainer TRAINS rather than merely RUNS -- could never have passed.
#
# Both versions failed for the same underlying reason: ONE hand-picked constant was being asked
# TWO different questions. They are now asked separately, and the magnitudes are DERIVED.
#
#   CHECK A -- "is this above numerical noise?"  rel > 1e-7 (float32 accumulation noise over a
#     norm of ~1.2e3) and not every tensor bitwise-identical to the initialisation.
#   CHECK B -- "is my reference the actual initialisation?"  MAGNITUDE-FREE: the checkpoint must be
#     CLOSER to the init we claim it started from than to the other variant. For a true update of
#     size d from base_A, rel_vs_A = d while rel_vs_B = sqrt(d^2 + floor^2) > d, so the inequality
#     holds for ANY d -- including d far below the floor, where no threshold could separate them.
#     This is what actually catches the v1 mistake: a no-op scored against the wrong variant gives
#     rel_same = 2.074e-04 while rel_other = 0, and the inequality FAILS.
#   CHECK C -- "is this consistent with the configured amount of training?"  AdamW's per-element
#     step is bounded by ~lr (esd_meta.py:867 optim.AdamW; m_hat/(sqrt(v_hat)+eps) is in [-1,1],
#     and equals +-1 exactly on the first bias-corrected step). So after S updates the cumulative
#     per-element change is at most lr*S, and therefore
#         rel <= lr * S * sqrt(N) / ||w||          (the SATURATING bound)
#     with N = 859,520,964 params and ||w|| = 1201.53 for this UNet, i.e. rel <= lr*S*24.4.
#     At lr=1e-5: S=2 -> 4.88e-04, S=1500 -> 3.66e-01. The band is [bound(S=1)/100, 2*bound(S)].
#     NOTE this bound is independent of the gamma values: the gammas rescale param.grad, and Adam
#     normalises gradient scale away, so they cannot push the update past lr*S.
say "--- stage 9: reload against the exact initialisation ($PIN_VARIANT @ $PIN_REVISION)"
# S_MAX is the configured number of updates for THIS mode -- the most that can have happened.
S_MAX=$(for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = "--num_train_steps" ] && echo "${ARGS[$((i+1))]}"; done)
say "    guard inputs: lr=$LR configured_steps=$S_MAX (AdamW saturating bound, see header)"
$VT - "$CKDIR" "$CKSUB" "$PIN_REVISION" "$PIN_VARIANT" "$MODE" "$LR" "$S_MAX" <<'PY' 2>&1 | tee -a "$L" \
  || die "stage 9: checkpoint reload / training-consistency guard failed"
import json, math, os, sys, torch
from diffusers import UNet2DConditionModel
ckdir, cksub, rev, variant, mode, lr_s, smax_s = sys.argv[1:8]
lr, s_max = float(lr_s), int(smax_s)
kw = {"subfolder": cksub} if cksub else {}
got = UNet2DConditionModel.from_pretrained(ckdir, torch_dtype=torch.float32, **kw)
def base(v):
    k = {"variant": v} if v else {}
    return UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", revision=rev,
        torch_dtype=torch.float32, **k).state_dict()
g = got.state_dict()
same = base(variant)                         # the EXACT initialisation the trainer was given
other = base(None if variant else "fp16")    # the other form, for CHECK B
assert set(g) == set(same), f"key set mismatch: {len(set(g) ^ set(same))} keys differ"
ng = sum(v.numel() for v in g.values()); nr = sum(v.numel() for v in same.values())
assert ng == nr, f"param count {ng} != init {nr}"
assert all(torch.isfinite(v).all() for v in g.values()), "non-finite tensors in the checkpoint"
def relnorm(a, b):
    d = sum((a[k] - b[k]).pow(2).sum() for k in a).sqrt()
    w = sum(b[k].pow(2).sum() for k in a).sqrt()
    return (d / w).item()
r_same, r_other, f_var = relnorm(g, same), relnorm(g, other), relnorm(same, other)
w_norm = sum(same[k].pow(2).sum() for k in same).sqrt().item()
nid = sum(1 for k in g if torch.equal(g[k], same[k]))
# the saturating AdamW bound, derived from THIS checkpoint's own N and ||w||
def bound(S):
    return lr * S * math.sqrt(ng) / w_norm
HI, LO = 2.0 * bound(s_max), bound(1) / 100.0
# the step index actually reached, when the artifact is a periodic step=NNNNNN/ save
s_seen = None
tail = os.path.basename(ckdir.rstrip("/"))
if tail.startswith("step="):
    try: s_seen = int(tail.split("=", 1)[1])
    except ValueError: pass
print(f"    reload OK: {len(g)} tensors, {ng/1e6:.3f}M params, ||init||={w_norm:.4f}")
print(f"    rel vs SAME init (variant={variant!r}) = {r_same:.6e}   bitwise-identical={nid}")
print(f"    rel vs OTHER variant                   = {r_other:.6e}")
print(f"    variant-only floor (init vs init)      = {f_var:.6e}  (reported, NOT thresholded)")
print(f"    AdamW saturating bound: lr*S*sqrt(N)/||w|| = {bound(s_max):.6e} at S={s_max}"
      + (f" (checkpoint is step={s_seen})" if s_seen is not None else ""))
print(f"    training-consistency band = [{LO:.6e}, {HI:.6e}]")
fails = []
# CHECK A -- above numerical noise
NOISE = 1e-7
if not (r_same > NOISE):
    fails.append(f"A(noise): rel={r_same:.3e} is at or below float32 noise {NOISE:.0e} -- the "
                 f"checkpoint is the initialisation; training was a NO-OP")
if nid == len(g):
    fails.append(f"A(noise): all {nid} tensors are bitwise identical to the initialisation")
# CHECK B -- the reference really is the initialisation (magnitude-free)
if not (r_same < r_other):
    fails.append(f"B(reference): the checkpoint is NOT closer to the initialisation we claim "
                 f"({r_same:.3e}) than to the other variant ({r_other:.3e}). The reference "
                 f"initialisation is wrong -- this is exactly the defect that made a no-op look "
                 f"trained. Check --revision/--variant against what the trainer was given")
# CHECK C -- consistent with the configured steps and lr
if r_same > HI:
    fails.append(f"C(consistency): rel={r_same:.3e} EXCEEDS 2x the AdamW saturating bound for "
                 f"S={s_max} at lr={lr:g} ({HI:.3e}). More change than the configured training "
                 f"can produce -- the checkpoint is not from this configuration")
elif r_same < LO:
    fails.append(f"C(consistency): rel={r_same:.3e} is below {LO:.3e} = bound(S=1)/100, i.e. far "
                 f"less change than even ONE update at lr={lr:g} should produce")
for f in fails:
    print(f"    FAIL {f}")
print(f"    CHECK A noise: {'FAIL' if any(x.startswith('A') for x in fails) else 'PASS'}  "
      f"CHECK B reference: {'FAIL' if any(x.startswith('B') for x in fails) else 'PASS'}  "
      f"CHECK C consistency: {'FAIL' if any(x.startswith('C') for x in fails) else 'PASS'}")
json.dump({"checkpoint": ckdir, "subfolder": cksub or None, "mode": mode,
           "init_revision": rev, "init_variant": variant,
           "tensors": len(g), "params": ng, "init_norm": w_norm,
           "rel_delta_vs_same_init": r_same,
           "rel_delta_vs_other_variant": r_other,
           "variant_only_floor": f_var,
           "variant_floor_is_thresholded": False,
           "bitwise_identical_tensors": nid,
           "learning_rate": lr, "configured_steps": s_max, "checkpoint_step": s_seen,
           "adamw_saturating_bound": bound(s_max),
           "consistency_band": [LO, HI],
           "checks": {
             "A_above_noise": f"rel > {NOISE:.0e} and not all tensors identical",
             "B_reference_identity": "rel_vs_same_init < rel_vs_other_variant (magnitude-free)",
             "C_training_consistency": "bound(S=1)/100 <= rel <= 2*bound(configured_steps), "
                                       "bound(S) = lr*S*sqrt(N)/||w|| (AdamW per-step <= lr)"},
           "failures": fails},
          open(os.path.join(ckdir, "reload_check.json"), "w"), indent=1)
print(f"    -> {os.path.join(ckdir, 'reload_check.json')}")
sys.exit(1 if fails else 0)
PY

# ================================ STAGE 10: evaluation compatibility ===========================
# Our evaluator consumes a flat fp16 UNet state dict (adapters/<arm>/unet.pt). Produce it from the
# saved checkpoint and assert it is loadable and shape-compatible, so "the run finished" and "the
# result is evaluable" are not two separate discoveries.
say "--- stage 10: evaluation compatibility"
ADIR="adapters/R2_${EXP_NAME}"
mkdir -p "$ADIR"
$VT - "$CKDIR" "$CKSUB" "$ADIR" "$PIN_REVISION" "$PIN_VARIANT" <<'PY' 2>&1 | tee -a "$L" \
  || die "stage 10: the checkpoint is not consumable by our evaluator"
import json, os, sys, torch
from diffusers import UNet2DConditionModel
ckdir, cksub, adir, rev, variant = sys.argv[1:6]
kw = {"subfolder": cksub} if cksub else {}
u = UNet2DConditionModel.from_pretrained(ckdir, torch_dtype=torch.float32, **kw)
ref = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                           revision=rev, torch_dtype=torch.float32,
                                           variant=variant).state_dict()
sd = u.state_dict()
assert set(sd) == set(ref), "key set differs from the evaluator's expected UNet"
bad = [k for k in sd if tuple(sd[k].shape) != tuple(ref[k].shape)]
assert not bad, f"shape mismatch on {len(bad)} tensors: {bad[:3]}"
torch.save({k: v.half() for k, v in sd.items()}, f"{adir}/unet.pt")
json.dump({"source_checkpoint": ckdir, "subfolder": cksub or None,
           "init_revision": rev, "init_variant": variant,
           "tensors": len(sd), "dtype": "float16",
           "note": "flat UNet state dict for scripts/gen_eval.py --unet"},
          open(f"{adir}/compose.json", "w"), indent=1)
print(f"    wrote {adir}/unet.pt ({len(sd)} tensors, fp16) + compose.json")
PY
say "R2 DONE $(date -Is) mode=$MODE -> $OUTDIR  (evaluable: $ADIR/unet.pt)"
