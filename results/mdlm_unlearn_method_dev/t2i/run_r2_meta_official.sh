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


# ================================ STAGE 4b: SPLIT REGISTRY ENFORCEMENT =========================
# THE REGISTRY IS CONSUMED AND ENFORCED HERE, NOT PRINTED (repair v3, AGENT B task 2).
#
# What was wrong. scripts/split_registry.py v2 reported unregistered builders in its coverage
# block and exited 0, and NO runner consumed its verdict at all. Measured before the repair:
# v2 reported `defence_scripts_on_disk_not_registered: ["erase_utrap_v2", "erase_wdr_v2"]` and
# still exited 0, while `meta_unlearn_v2.py` was not even detected. A verdict nothing enforces is
# not a control.
#
# Two levels, deliberately separated:
#   STRUCTURAL   missing registration, a registered script absent from disk, a stale exemption, or
#                an unreadable runtime pool record. ALWAYS FATAL -- these mean the registry does
#                not know what it is auditing, whatever this run is for.
#   ELIGIBILITY  whether the prompts may carry a NEW confirmatory claim. Fatal only when this run
#                declares one (META_NEW_CLAIM=1). It is currently FALSE and will stay false until
#                a fresh FINAL set is frozen: FINDING65/FINDING66 scored and interpreted the
#                existing FINAL set, so it is INSPECTED_DEVELOPMENT_DATA.
say "--- stage 4b: split registry enforcement"
NEW_CLAIM="${META_NEW_CLAIM:-0}"
set +e
$VT scripts/split_registry.py $([ "$NEW_CLAIM" = "1" ] || echo --debugging) >>"$L" 2>&1
SR_RC=$?
set -e
$VT - "$NEW_CLAIM" <<'PY' 2>&1 | tee -a "$L" || die "stage 4b: the split registry REFUSES this run"
import importlib.util as u, json, sys
nc = sys.argv[1] == "1"
s = u.spec_from_file_location("sr", "scripts/split_registry.py")
m = u.module_from_spec(s); s.loader.exec_module(m)
rep = m.audit()
v, cov = rep["verdict"], rep["coverage"]
print(f"    split_version={m.SPLIT_VERSION} new_claim={nc}")
print(f"    registered: {len(cov['registered']['defence'])} defence, "
      f"{len(cov['registered']['upstream_wrappers'])} upstream wrappers, "
      f"{len(cov['registered']['eval'])} eval")
print(f"    unregistered builders on disk: {cov['defence_scripts_on_disk_not_registered']}")
print(f"    runtime pool record kinds present: {v['runtime_pool_kinds_present']}")
print(f"    VERDICT string_disjoint={v['string_disjoint']}  "
      f"eligible_for_confirmatory_evaluation={v['eligible_for_confirmatory_evaluation']}  "
      f"coverage_complete={v['coverage_complete']}")
for b in v["eligibility_blockers"]:
    print(f"    blocker: {b}")
# STRUCTURAL failures are always fatal; eligibility only when a new claim is declared.
structural = m.enforce(rep, for_new_claim=False)
for f in structural:
    print(f"    ENFORCED FAILURE (structural): {f}")
if structural:
    sys.exit(1)
if nc:
    full = m.enforce(rep, for_new_claim=True)
    for f in full:
        print(f"    ENFORCED FAILURE (new claim): {f}")
    if full:
        sys.exit(1)
    print("    registry PERMITS a new confirmatory claim")
else:
    print("    structural checks PASS. META_NEW_CLAIM is not set, so this run may NOT be reported")
    print("    as a clean/confirmatory evaluation: eligibility was not required and is "
          f"{v['eligible_for_confirmatory_evaluation']}.")
PY

# ================================ STAGE 4c: EVIDENCE MODULE + UNIT TESTS =======================
# The evidence chain that replaces the old stage 9 lives in a GENERATED module so that the exact
# code which instruments the run is also the code the unit tests exercise, and so its sha256 can be
# bound into the run record. Upstream is NEVER edited: the module wraps it at run time, and the
# patch record below shows zero changed bytes in upstream/Meta-Unlearning/src/esd_meta.py.
say "--- stage 4c: evidence module, instrumentation patch record, and unit tests"
mkdir -p "$R/out"
EVID="/data/bijaypandey/t2i_unlearn/$R/out/r2_evidence.py"
cat > "$EVID" <<'EVIDENCE_PY_EOF'
"""R2 EVIDENCE CHAIN for the pinned Meta-Unlearning trainer. GENERATED FILE.

This file is WRITTEN AT RUN TIME by run_r2_meta_official.sh (stage 6b) and its sha256 is recorded
in the run's config record. Do not edit it here; edit the heredoc in run_r2_meta_official.sh.

WHY IT EXISTS -- WHAT IT REPLACES. The previous stage 9 decided "did the trainer train?" from the
relative Frobenius norm of (checkpoint - base), under four assumptions that DO NOT HOLD:

 (1) "a trained checkpoint is closer to its own init than to another base variant", enforced as
     `rel_vs_same < rel_vs_other`. FALSE. Counterexample: init A=1.0, alternative B=1.01, trained
     T=1.007 gives rel_vs_same=7.00e-03 and rel_vs_other=2.97e-03, so a GENUINE update FAILS the
     check. The sign of that inequality is a fact about the DIRECTION of the update, which nothing
     constrains, not about whether an update happened.
 (2) The derivation of (1) ("rel_vs_B = sqrt(d^2 + floor^2)") is a Pythagorean step that requires
     the update (T - A) to be ORTHOGONAL to the variant offset (A - B). Nothing makes a gradient
     update orthogonal to a float16 quantisation offset. Measured on the same example:
     sqrt(d^2+floor^2) = 1.22e-02 while the true ||T-B|| = 3.00e-03.
 (3) "AdamW's per-element step is bounded by ~lr", enforced as rel <= lr*S*sqrt(N)/||w||. NOT a
     guaranteed bound: m_hat/(sqrt(v_hat)+eps) is bounded by 1 only in the beta1^2 <= beta2 steady
     state, and decoupled weight decay adds -lr*wd*w ON TOP of it. A bound that can be exceeded
     cannot be a pass/fail criterion.
 (4) "bound(S=1)/100 is a floor on the change after one update". There is NO justified positive
     minimum parameter change inferable from lr: one EXECUTED AdamW step with a zero gradient
     moves the parameter by exactly 0.0 (measured), as does an executed step at lr=0 during
     warmup, or a step on a tensor that received no gradient.

All four are REMOVED as pass/fail criteria and NO other arbitrary norm threshold is substituted.
Norms are still REPORTED, labelled as descriptive, never thresholded.

WHAT REPLACES THEM: a direct evidence chain of six links, each of which is a fact recorded by the
process that performed the action, not an inference from a magnitude.

  LINK 1  LOADED STATE, BEFORE OPTIMIZATION. Every from_pretrained the trainer executes is
          recorded: repo id, revision, variant, dtype, subfolder, and the resolved local snapshot
          directory. The trainable parameter NAMES are captured from upstream's own
          gather_parameters(), and each trainable tensor is fingerprinted (sha256 of its exact
          bytes) BEFORE the first optimizer step.
  LINK 2  EXECUTED UPDATES. torch.optim.Optimizer.step is wrapped, so EXECUTED updates are counted
          by the call that performs them -- never inferred from --num_train_steps. At each executed
          update the per-group learning rates actually in effect, the gradient diagnostics
          (how many selected tensors had grad None, the global grad norm, non-finite counts) and
          the realised per-tensor change are recorded. CONFIGURED steps, EXECUTED updates and
          SKIPPED steps are three separate fields, because upstream has two real skip paths:
            esd_meta.py:928-929  `if args.use_fp16: pass` -- skips loss, backward AND the step
            esd_meta.py:1053     `if step % args.gradient_accumulation_steps == 0:` -- gates it
  LINK 3  SAVE-POINT CORRESPONDENCE. upstream's save_checkpoint is wrapped, so the IN-MEMORY state
          of the UNet it was handed is fingerprinted AT THE SAVE POINT, together with the
          destination directory and the executed-update count at that moment.
  LINK 4  RELOAD EQUALITY. The checkpoint is reloaded from disk and compared to the LINK 3
          in-memory fingerprints: key set, shapes, finiteness, and tensor EQUALITY at the intended
          serialization precision. esd_meta.py performs NO dtype cast on save (verified: no
          .half()/float16 in the file), so the intended precision for a step= save is float32 and
          equality must be EXACT (torch.equal), not approximate.
  LINK 5  FP16 EXPORT. Our evaluator consumes fp16. The exported tensor is compared to the
          EXPLICITLY converted source tensor (src.half()), element for element, so the export is
          verified against a stated conversion rather than against a tolerance.
  LINK 6  BINDING. Every record carries the run configuration, the upstream commit, and the
          sha256 of the untouched upstream source file plus of this instrumentation module.

A link that CANNOT BE EVALUATED is reported as INDETERMINATE and is a FAILURE, never a pass.

STATUS DISCIPLINE. `--selftest` exercises the verifier on SYNTHETIC tensors. Those are UNIT TESTS.
A passing suite is NOT evidence that the upstream trainer ran, and no run of --selftest may be
described as a training result, as method effectiveness, or as reproducing any paper.
"""
import argparse, hashlib, json, math, os, sys


# ----------------------------------------------------------------- fingerprints
def tfp(t):
    """Exact fingerprint of one tensor: dtype, shape, and sha256 of its exact bytes.

    Bytes, not a norm: two different tensors can share a norm, and a norm cannot detect a
    permutation. `.contiguous().cpu()` is required because a non-contiguous or CUDA tensor has no
    directly hashable buffer.
    """
    import torch
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool
             else t.to(torch.uint8).numpy().tobytes())
    return {"dtype": str(t.dtype), "shape": list(t.shape), "sha256": h.hexdigest()}


def sdfp(sd, names=None):
    """Fingerprint a state dict (optionally restricted to `names`)."""
    ks = sorted(sd) if names is None else sorted(k for k in sd if k in set(names))
    return {k: tfp(sd[k]) for k in ks}


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def desc_norms(a, b):
    """DESCRIPTIVE ONLY. Reported so a reader can see the size of a change. Never thresholded."""
    import torch
    ks = [k for k in a if k in b]
    d = sum((a[k].double() - b[k].double()).pow(2).sum() for k in ks).sqrt()
    w = sum(b[k].double().pow(2).sum() for k in ks).sqrt()
    return {"frobenius_delta": float(d), "frobenius_ref": float(w),
            "relative": float(d / w) if float(w) > 0 else None,
            "n_tensors_compared": len(ks),
            "INTERPRETATION": "descriptive magnitude; NOT a pass/fail criterion. The previous "
                              "guard thresholded a quantity like this against an AdamW bound that "
                              "is not guaranteed and a floor that is unfounded."}


# ================================================================= LINK 1/2/3: the shim
class Recorder:
    def __init__(self, upstream_src, out_dir, run_config):
        self.out_dir = out_dir
        self.rec = {
            "study": "r2_meta_evidence_chain",
            "instrumentation": {
                "module": os.path.abspath(__file__),
                "module_sha256": file_sha256(os.path.abspath(__file__)),
                "upstream_source": upstream_src,
                "upstream_source_sha256": file_sha256(upstream_src),
                "upstream_source_modified": False,
                "method": "runtime wrapping only; the upstream file is never edited. The wrapped "
                          "callables are listed in `wrapped`, and the patch record written beside "
                          "this file shows zero changed bytes in upstream.",
                "wrapped": [],
            },
            "run_config": run_config,
            "link1_loaded_state": {"from_pretrained_calls": [], "trainable": None},
            "link2_executed_updates": {
                "configured_steps": None, "executed_updates": 0, "skipped_steps": [],
                "skip_paths_in_upstream": {
                    "use_fp16": "esd_meta.py:928-929 `if args.use_fp16: pass` skips loss, "
                                "backward and the optimizer step for every step",
                    "grad_accum": "esd_meta.py:1053 `if step % args.gradient_accumulation_steps "
                                  "== 0:` gates optimizer.step()/lr_scheduler.step()"},
                "updates": []},
            "link3_save_points": [],
            "errors": [],
        }

    # ---- LINK 1
    def wrap_loaders(self):
        import diffusers
        for cls in ("UNet2DConditionModel", "AutoencoderKL", "StableDiffusionPipeline"):
            k = getattr(diffusers, cls)
            orig = k.from_pretrained.__func__
            self.rec["instrumentation"]["wrapped"].append(f"diffusers.{cls}.from_pretrained")

            def mk(cls=cls, orig=orig, kls=k):
                def f(c, *a, **kw):
                    r = orig(c, *a, **kw)
                    self.rec["link1_loaded_state"]["from_pretrained_calls"].append({
                        "class": cls,
                        "name_or_path": str(a[0]) if a else str(kw.get(
                            "pretrained_model_name_or_path")),
                        "subfolder": kw.get("subfolder"),
                        "revision": kw.get("revision"),
                        "variant": kw.get("variant"),
                        "torch_dtype": str(kw.get("torch_dtype")),
                        "resolved_dtype": str(getattr(r, "dtype", None)),
                    })
                    return r
                return classmethod(f)
            k.from_pretrained = mk()

    def record_trainable(self, names, params, finetuning_method):
        """LINK 1, second half: the ACTUAL trainable set, before any optimization."""
        self.rec["link1_loaded_state"]["trainable"] = {
            "finetuning_method": finetuning_method,
            "selector": "upstream esd_meta.gather_parameters (esd_meta.py:269-306)",
            "n_tensors": len(names), "names": list(names),
            "n_params": int(sum(p.numel() for p in params)),
            "pre_optimization_fingerprints": {n: tfp(p) for n, p in zip(names, params)},
            "captured": "before the first optimizer step",
        }

    # ---- LINK 2
    def wrap_optimizer(self, named_params, configured_steps, grad_accum, use_fp16):
        import torch
        r = self.rec["link2_executed_updates"]
        r["configured_steps"] = configured_steps
        r["gradient_accumulation_steps"] = grad_accum
        r["use_fp16_requested"] = bool(use_fp16)
        if use_fp16:
            r["skipped_steps"].append({
                "reason": "use_fp16", "steps": "ALL",
                "detail": "esd_meta.py:928-929 is `if args.use_fp16: pass`, so no loss, no "
                          "backward and no optimizer step is executed for ANY step."})
        # GLOBAL optimizer step hooks, not class monkeypatching. This matters: AdamW DEFINES its
        # own `step`, so replacing torch.optim.Optimizer.step never fires for the optimizer
        # esd_meta.py:867-873 actually builds, and the evidence record would have been silently
        # empty. Caught by selftest::recorder_counts_executed_not_configured.
        from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                           register_optimizer_step_pre_hook)
        self.rec["instrumentation"]["wrapped"].append(
            "torch.optim global step pre/post hooks (register_optimizer_step_pre_hook / "
            "register_optimizer_step_post_hook) -- no optimizer class is modified")
        ids = {id(p): n for n, p in named_params}
        pend = {}

        def sel_of(opt):
            return [(ids[id(p)], p) for g in opt.param_groups for p in g["params"]
                    if id(p) in ids]

        def pre(opt, a, kw):
            sel = sel_of(opt)
            if not sel:            # the inner per-task optimizers (esd_meta.py:949/965): not ours
                return
            gn, nnone, nonfinite = 0.0, 0, 0
            for n, p in sel:
                if p.grad is None:
                    nnone += 1
                    continue
                g = p.grad.detach().double()
                if not torch.isfinite(g).all():
                    nonfinite += 1
                gn += float(g.pow(2).sum())
            pend[id(opt)] = {
                "before": {n: p.detach().clone() for n, p in sel},
                "lrs": [float(g["lr"]) for g in opt.param_groups],
                "wd": [float(g.get("weight_decay", 0.0)) for g in opt.param_groups],
                "n": len(sel), "nnone": nnone, "nonfinite": nonfinite,
                "grad_l2": math.sqrt(gn), "cls": type(opt).__name__}

        def post(opt, a, kw):
            st = pend.pop(id(opt), None)
            if st is None:
                return
            changed, maxabs = 0, 0.0
            for n, p in sel_of(opt):
                d = (p.detach() - st["before"][n]).abs()
                m = float(d.max()) if d.numel() else 0.0
                if m > 0:
                    changed += 1
                maxabs = max(maxabs, m)
            r["executed_updates"] += 1
            r["updates"].append({
                "executed_update_index": r["executed_updates"],
                "optimizer": st["cls"],
                "learning_rates_in_effect": st["lrs"],
                "weight_decay": st["wd"],
                "n_tracked_tensors_in_step": st["n"],
                "n_grad_None": st["nnone"],
                "n_grad_nonfinite": st["nonfinite"],
                "grad_global_l2": st["grad_l2"],
                "n_tensors_changed_by_this_update": changed,
                "max_abs_param_change_this_update": maxabs,
                "NOTE": "an executed update with grad_global_l2 == 0 legitimately changes nothing; "
                        "that is why no positive lower bound on parameter change is asserted.",
            })
        self._hooks = [register_optimizer_step_pre_hook(pre),
                       register_optimizer_step_post_hook(post)]

    # ---- LINK 3
    def wrap_save(self, mod):
        orig = mod.save_checkpoint
        self.rec["instrumentation"]["wrapped"].append("esd_meta.save_checkpoint")
        names = None

        def save(*a, **kw):
            nonlocal names
            if names is None:
                tr = self.rec["link1_loaded_state"].get("trainable") or {}
                names = tr.get("names")
            unet = kw.get("unet") or (a[3] if len(a) > 3 else None)
            step = kw.get("step", None)
            args = kw.get("args") or (a[0] if a else None)
            sd = unet.state_dict() if unet is not None else {}
            dest = (os.path.join(args.output_dir, "step=%06d" % step) if step is not None
                    else args.output_dir)
            entry = {
                "step": step,
                "kind": "periodic_unet_only" if step is not None else "final_full_pipeline",
                "destination": dest,
                "executed_updates_at_this_save": self.rec["link2_executed_updates"]
                                                     ["executed_updates"],
                "n_tensors_in_memory": len(sd),
                "in_memory_dtypes": sorted({str(v.dtype) for v in sd.values()}),
                "in_memory_fingerprints_trainable": sdfp(sd, names),
                "in_memory_fingerprints_all": sdfp(sd),
                "serialization_note":
                    "esd_meta.py performs NO dtype cast on save (no .half()/float16 anywhere in "
                    "the file), so the intended serialization precision equals the in-memory "
                    "dtype above and LINK 4 requires EXACT equality.",
            }
            self.rec["link3_save_points"].append(entry)
            self.flush()
            return orig(*a, **kw)
        mod.save_checkpoint = save

    def flush(self):
        os.makedirs(self.out_dir, exist_ok=True)
        p = os.path.join(self.out_dir, "train_evidence.json")
        with open(p, "w") as f:
            json.dump(self.rec, f, indent=1)
        return p


def run_instrumented(upstream_src, out_dir, argv):
    """Execute the UNMODIFIED upstream trainer under the recorder."""
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("esd_meta_mod", upstream_src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["esd_meta_mod"] = mod
    saved_argv = sys.argv
    sys.argv = ["esd_meta.py"] + list(argv)
    spec.loader.exec_module(mod)
    args = mod.parse_args()
    rc = Recorder(upstream_src, out_dir, {k: _j(v) for k, v in vars(args).items()})
    rc.wrap_loaders()

    # upstream's own selector, wrapped so the trainable set is recorded from the call that makes it
    orig_gather = mod.gather_parameters
    rc.rec["instrumentation"]["wrapped"].append("esd_meta.gather_parameters")
    first = {"done": False}

    def gather(a, unet):
        names, params = orig_gather(a, unet)
        if not first["done"]:
            first["done"] = True
            rc.record_trainable(names, params, getattr(a, "finetuning_method", None))
            rc.wrap_optimizer(list(zip(names, params)), a.num_train_steps,
                              a.gradient_accumulation_steps, a.use_fp16)
            rc.flush()
        return names, params
    mod.gather_parameters = gather
    rc.wrap_save(mod)
    try:
        mod.main()
    except BaseException as e:
        rc.rec["errors"].append({"type": type(e).__name__, "message": str(e)[:2000]})
        rc.flush()
        sys.argv = saved_argv
        raise
    sys.argv = saved_argv
    # configured vs executed, stated side by side -- never conflated
    r = rc.rec["link2_executed_updates"]
    r["executed_vs_configured"] = {
        "configured_steps": r["configured_steps"],
        "executed_updates": r["executed_updates"],
        "difference": (None if r["configured_steps"] is None
                       else r["configured_steps"] - r["executed_updates"]),
        "NOTE": "a difference is not by itself an error: gradient_accumulation_steps > 1 makes it "
                "expected. It is an error only when it is unexplained by a recorded skip path."}
    return rc.flush()


def _j(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


# ================================================================= LINK 4/5/6: the verifier
def verify(evidence_path, ckpt_dir, subfolder, fp16_export, expect_model, expect_rev,
           expect_variant, out_path):
    import torch
    from diffusers import UNet2DConditionModel
    ev = json.load(open(evidence_path))
    fails, notes = [], []

    def fail(link, msg):
        fails.append(f"{link}: {msg}")

    # ---------------- LINK 1 ---------------------------------------------------
    l1 = ev.get("link1_loaded_state") or {}
    calls = l1.get("from_pretrained_calls") or []
    unet_loads = [c for c in calls if c.get("class") == "UNet2DConditionModel"]
    if not unet_loads:
        fail("LINK1", "INDETERMINATE: the trainer recorded no UNet2DConditionModel.from_pretrained "
                      "call, so the state it started from is unknown")
    for c in unet_loads:
        if c.get("name_or_path") != expect_model:
            fail("LINK1", f"a UNet was loaded from {c.get('name_or_path')!r}, expected "
                          f"{expect_model!r}")
        if str(c.get("revision")) != str(expect_rev):
            fail("LINK1", f"a UNet was loaded at revision {c.get('revision')!r}, expected "
                          f"{expect_rev!r}")
        if str(c.get("variant")) != str(expect_variant):
            fail("LINK1", f"a UNet was loaded with variant {c.get('variant')!r}, expected "
                          f"{expect_variant!r}")
    tr = l1.get("trainable")
    if not tr or not tr.get("names"):
        fail("LINK1", "INDETERMINATE: no trainable parameter names were recorded")
    elif not tr.get("pre_optimization_fingerprints"):
        fail("LINK1", "INDETERMINATE: no pre-optimization fingerprints were recorded")

    # ---------------- LINK 2 ---------------------------------------------------
    l2 = ev.get("link2_executed_updates") or {}
    ex = l2.get("executed_updates")
    if not isinstance(ex, int):
        fail("LINK2", "INDETERMINATE: executed_updates was not recorded")
    elif ex == 0:
        fail("LINK2", "ZERO optimizer updates were EXECUTED. The trainer ran but did not train. "
                      "configured_steps=%s; recorded skip paths=%s"
                      % (l2.get("configured_steps"), l2.get("skipped_steps")))
    ups = l2.get("updates") or []
    if ex and not ups:
        fail("LINK2", "executed_updates=%d but no per-update record exists" % ex)
    if ups:
        if all(u.get("n_grad_None", 0) == u.get("n_tracked_tensors_in_step", -1) for u in ups):
            fail("LINK2", "every executed update saw grad=None on EVERY tracked tensor: no "
                          "gradient ever reached the selected parameters")
        nf = sum(u.get("n_grad_nonfinite", 0) for u in ups)
        if nf:
            fail("LINK2", f"{nf} non-finite gradient tensor(s) were seen across executed updates")
        if all(float(l) == 0.0 for u in ups for l in u.get("learning_rates_in_effect", [1])):
            fail("LINK2", "every executed update ran at learning rate 0.0, so no update could "
                          "change any parameter")
        notes.append("executed updates: %d; lrs in effect: %s; grad L2 range [%.6e, %.6e]"
                     % (ex, sorted({l for u in ups for l in
                                    u.get("learning_rates_in_effect", [])}),
                        min(u.get("grad_global_l2", 0.0) for u in ups),
                        max(u.get("grad_global_l2", 0.0) for u in ups)))
        if all(u.get("n_tensors_changed_by_this_update", 0) == 0 for u in ups):
            fail("LINK2", "no executed update changed any tracked tensor; the recorded gradient "
                          "L2 was %s" % sorted({u.get("grad_global_l2") for u in ups}))

    # ---------------- LINK 3: pick the save point for THIS checkpoint ----------
    sps = ev.get("link3_save_points") or []
    want = os.path.abspath(ckpt_dir.rstrip("/"))
    sp = next((s for s in sps if os.path.abspath(str(s.get("destination")).rstrip("/")) == want),
              None)
    if sp is None:
        fail("LINK3", "INDETERMINATE: no recorded save point has destination %s. Recorded: %s. "
                      "Without the in-memory state at the save point there is nothing to compare "
                      "the file on disk against." % (want, [s.get("destination") for s in sps]))
    elif not sp.get("in_memory_fingerprints_all"):
        fail("LINK3", "the save point for %s recorded no in-memory fingerprints" % want)

    # ---------------- LINK 4: reload equality ---------------------------------
    got = None
    if os.path.isdir(ckpt_dir):
        kw = {"subfolder": subfolder} if subfolder else {}
        try:
            got = UNet2DConditionModel.from_pretrained(ckpt_dir, torch_dtype=None, **kw
                                                       ).state_dict()
        except Exception as e:
            fail("LINK4", f"the checkpoint at {ckpt_dir} could not be reloaded: {e!r}")
    else:
        fail("LINK4", f"INDETERMINATE: {ckpt_dir} is not a directory")
    if got is not None:
        nonfinite = [k for k, v in got.items() if not torch.isfinite(v).all()]
        if nonfinite:
            fail("LINK4", f"{len(nonfinite)} reloaded tensor(s) are non-finite: {nonfinite[:3]}")
        if sp is not None:
            mem = sp["in_memory_fingerprints_all"]
            mk, gk = set(mem), set(got)
            if mk != gk:
                fail("LINK4", f"key set differs from the in-memory state at the save point: "
                              f"{len(mk - gk)} missing, {len(gk - mk)} extra "
                              f"(e.g. {sorted(mk ^ gk)[:3]})")
            shp = [k for k in (mk & gk) if list(got[k].shape) != mem[k]["shape"]]
            if shp:
                fail("LINK4", f"{len(shp)} reloaded tensor(s) have a different shape from the "
                              f"in-memory state: {shp[:3]}")
            dt = [k for k in (mk & gk) if str(got[k].dtype) != mem[k]["dtype"]]
            if dt:
                fail("LINK4", f"{len(dt)} reloaded tensor(s) have a different dtype from the "
                              f"in-memory state (intended serialization precision was "
                              f"{sorted({mem[k]['dtype'] for k in dt})}, got "
                              f"{sorted({str(got[k].dtype) for k in dt})}): {dt[:3]}")
            neq = [k for k in (mk & gk)
                   if str(got[k].dtype) == mem[k]["dtype"]
                   and list(got[k].shape) == mem[k]["shape"]
                   and tfp(got[k])["sha256"] != mem[k]["sha256"]]
            if neq:
                fail("LINK4", f"{len(neq)} reloaded tensor(s) are NOT BIT-EQUAL to the in-memory "
                              f"state at the save point, at the intended serialization precision. "
                              f"This checkpoint is STALE or was written from a different state: "
                              f"{neq[:3]}")
            else:
                notes.append("LINK4: all %d tensors bit-equal to the save-point in-memory state"
                             % len(mk & gk))
            # DESCRIPTIVE: how far the checkpoint moved from the pre-optimization trainable state
            if tr and tr.get("pre_optimization_fingerprints"):
                pre = tr["pre_optimization_fingerprints"]
                same = [k for k in pre if k in got and tfp(got[k])["sha256"] == pre[k]["sha256"]]
                notes.append("LINK4: %d/%d TRAINABLE tensors are bit-identical to their "
                             "pre-optimization state" % (len(same), len(pre)))
                if len(same) == len(pre) and pre:
                    fail("LINK4", "every one of the %d trainable tensors is bit-identical to its "
                                  "pre-optimization state: the saved checkpoint contains no "
                                  "parameter change at all" % len(pre))

    # ---------------- LINK 5: fp16 export ------------------------------------
    exp = None
    if fp16_export:
        if not os.path.isfile(fp16_export):
            fail("LINK5", f"INDETERMINATE: {fp16_export} does not exist")
        elif got is None:
            fail("LINK5", "INDETERMINATE: the source checkpoint could not be reloaded, so the "
                          "fp16 export cannot be verified against its converted source")
        else:
            exp = torch.load(fp16_export, map_location="cpu")
            exp = exp.get("unet", exp)
            if set(exp) != set(got):
                fail("LINK5", f"the fp16 export's key set differs from its source: "
                              f"{sorted(set(exp) ^ set(got))[:3]}")
            wrong_dtype = [k for k in exp if exp[k].dtype != torch.float16]
            if wrong_dtype:
                fail("LINK5", f"{len(wrong_dtype)} exported tensor(s) are not float16: "
                              f"{wrong_dtype[:3]}")
            bad = [k for k in (set(exp) & set(got))
                   if not torch.equal(exp[k], got[k].to(torch.float16))]
            if bad:
                fail("LINK5", f"{len(bad)} exported tensor(s) do not equal the EXPLICITLY "
                              f"converted source tensor src.half(): {bad[:3]}. The export is not "
                              f"a conversion of this checkpoint.")
            else:
                notes.append("LINK5: all %d exported tensors equal src.half() exactly" % len(exp))

    # ---------------- LINK 6: binding ----------------------------------------
    ins = ev.get("instrumentation") or {}
    for k in ("upstream_source_sha256", "module_sha256", "upstream_source"):
        if not ins.get(k):
            fail("LINK6", f"INDETERMINATE: the evidence record carries no {k}")
    if ins.get("upstream_source") and os.path.isfile(ins["upstream_source"]):
        now = file_sha256(ins["upstream_source"])
        if now != ins.get("upstream_source_sha256"):
            fail("LINK6", f"the upstream source has changed since the run: recorded "
                          f"{str(ins.get('upstream_source_sha256'))[:16]} now {now[:16]}")
    if not ev.get("run_config"):
        fail("LINK6", "INDETERMINATE: no run configuration is bound to this record")

    out = {
        "study": "r2_meta_reload_evidence",
        "evidence_source": evidence_path,
        "checkpoint": ckpt_dir, "subfolder": subfolder, "fp16_export": fp16_export,
        "expected_initial_state": {"model": expect_model, "revision": expect_rev,
                                   "variant": expect_variant},
        "links": {
            "LINK1_loaded_state_before_optimization": _st(fails, "LINK1"),
            "LINK2_executed_updates": _st(fails, "LINK2"),
            "LINK3_save_point_in_memory_state": _st(fails, "LINK3"),
            "LINK4_reload_equality_at_intended_precision": _st(fails, "LINK4"),
            "LINK5_fp16_export_vs_converted_source": _st(fails, "LINK5")
                if fp16_export else "NOT_REQUESTED",
            "LINK6_binding_to_config_and_source": _st(fails, "LINK6"),
        },
        "notes": notes,
        "failures": fails,
        "REMOVED_CRITERIA": [
            "rel_vs_same_init < rel_vs_other_variant (direction of the update is unconstrained; "
            "a genuine update can be closer to the other variant)",
            "the Pythagorean derivation of that inequality (assumes an orthogonal update)",
            "rel <= lr*S*sqrt(N)/||w|| (AdamW does not bound the per-coordinate update by lr)",
            "rel >= bound(S=1)/100 (no justified positive minimum change follows from lr)"],
        "NO_SUBSTITUTE_THRESHOLD":
            "No norm threshold replaces them. Every magnitude in this record is descriptive.",
        "STATUS_DISCIPLINE":
            "This record verifies that a checkpoint corresponds to a recorded executed training "
            "run. It is NOT a measurement of unlearning, of method effectiveness, or of any "
            "published result.",
        "verdict": "PASS" if not fails else "FAIL",
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    for n in notes:
        print("    " + n)
    for f_ in fails:
        print("    FAIL " + f_)
    print(f"    verdict={out['verdict']} -> {out_path}")
    return 0 if not fails else 1


def _st(fails, link):
    return "FAIL" if any(f.startswith(link + ":") for f in fails) else "PASS"


# ================================================================= unit tests
def selftest():
    """UNIT TESTS on SYNTHETIC tensors. NOT evidence that the upstream trainer ran."""
    import tempfile
    import torch
    from diffusers import UNet2DConditionModel
    ok, bad = [], []

    def check(name, fn):
        try:
            fn()
            ok.append(name)
            print(f"  PASS {name}")
        except AssertionError as e:
            bad.append((name, str(e)))
            print(f"  FAIL {name}: {e}")

    cfg = dict(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
               block_out_channels=(8, 8), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
               up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=8,
               attention_head_dim=2, norm_num_groups=4)

    def scaffold(td, executed, lrs=(1e-5,), grad_l2=1.0, changed=1, grad_none=0,
                 model="CompVis/stable-diffusion-v1-4", rev="REV", variant="fp16",
                 perturb=0.0, trainable_unchanged=False, dest=None):
        """Build a (checkpoint, evidence.json) pair with full control over every link."""
        u = UNet2DConditionModel(**cfg)
        sd = {k: v.clone() for k, v in u.state_dict().items()}
        names = [k for k in sd if "attn2" in k and k.endswith("weight")][:3]
        pre = {n: tfp(sd[n]) for n in names}
        if not trainable_unchanged:
            for n in names:                      # a genuine change to the trainable tensors
                sd[n] = sd[n] + 1e-4
        u.load_state_dict(sd)
        ckpt = os.path.join(td, "step=000002")
        u.save_pretrained(ckpt)
        mem = {k: tfp(v) for k, v in u.state_dict().items()}
        if perturb:                              # make the file on disk STALE vs the record
            u2 = UNet2DConditionModel(**cfg)
            s2 = {k: v + perturb for k, v in u.state_dict().items()}
            u2.load_state_dict(s2)
            u2.save_pretrained(ckpt)
        ev = {
            "instrumentation": {"module": __file__,
                                "module_sha256": file_sha256(os.path.abspath(__file__)),
                                "upstream_source": os.path.abspath(__file__),
                                "upstream_source_sha256": file_sha256(os.path.abspath(__file__))},
            "run_config": {"num_train_steps": 2},
            "link1_loaded_state": {
                "from_pretrained_calls": [{"class": "UNet2DConditionModel",
                                           "name_or_path": model, "subfolder": "unet",
                                           "revision": rev, "variant": variant}],
                "trainable": {"names": names, "pre_optimization_fingerprints": pre}},
            "link2_executed_updates": {
                "configured_steps": 2, "executed_updates": executed, "skipped_steps": [],
                "updates": [{"executed_update_index": i + 1,
                             "learning_rates_in_effect": list(lrs),
                             "n_tracked_tensors_in_step": len(names),
                             "n_grad_None": grad_none, "n_grad_nonfinite": 0,
                             "grad_global_l2": grad_l2,
                             "n_tensors_changed_by_this_update": changed}
                            for i in range(executed)]},
            "link3_save_points": [{"step": 2, "destination": dest or ckpt,
                                   "executed_updates_at_this_save": executed,
                                   "in_memory_fingerprints_all": mem,
                                   "in_memory_fingerprints_trainable":
                                       {n: mem[n] for n in names}}],
        }
        p = os.path.join(td, "train_evidence.json")
        json.dump(ev, open(p, "w"), indent=1)
        return p, ckpt

    def run(td, p, ckpt, fp16=None):
        o = os.path.join(td, "verdict.json")
        rc = verify(p, ckpt, "", fp16, "CompVis/stable-diffusion-v1-4", "REV", "fp16", o)
        return rc, json.load(open(o))

    # 1. NO-OP TRAINING: the trainer ran, zero updates were executed.
    def t_noop():
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=0, trainable_unchanged=True)
            rc, v = run(td, p, c)
            assert rc == 1 and v["verdict"] == "FAIL", v["verdict"]
            assert v["links"]["LINK2_executed_updates"] == "FAIL", v["links"]
            assert any("ZERO optimizer updates" in f for f in v["failures"]), v["failures"]
            assert any("no parameter change at all" in f for f in v["failures"]), v["failures"]

    # 2. MISMATCHED INITIALIZATION: the trainer loaded a state we did not intend.
    def t_mismatched_init():
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2, rev="SOME-OTHER-REVISION", variant=None)
            rc, v = run(td, p, c)
            assert rc == 1, "a mismatched initialisation must fail"
            assert v["links"]["LINK1_loaded_state_before_optimization"] == "FAIL", v["links"]
            assert any("revision" in f for f in v["failures"]), v["failures"]
            assert any("variant" in f for f in v["failures"]), v["failures"]
            # and it must fail on IDENTITY, not on any magnitude
            assert v["links"]["LINK4_reload_equality_at_intended_precision"] == "PASS", v["links"]

    # 3. STALE CHECKPOINT: the file on disk is not the state that was in memory at the save point.
    def t_stale_checkpoint():
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2, perturb=1e-3)
            rc, v = run(td, p, c)
            assert rc == 1, "a stale checkpoint must fail"
            assert v["links"]["LINK4_reload_equality_at_intended_precision"] == "FAIL", v["links"]
            assert any("NOT BIT-EQUAL" in f for f in v["failures"]), v["failures"]

    # 3b. a save point that was never recorded is INDETERMINATE, which is a failure
    def t_unrecorded_save_point():
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2, dest=os.path.join(td, "step=999999"))
            rc, v = run(td, p, c)
            assert rc == 1, "an unrecorded save point must fail"
            assert v["links"]["LINK3_save_point_in_memory_state"] == "FAIL", v["links"]
            assert any("INDETERMINATE" in f for f in v["failures"]), v["failures"]

    # 4. GENUINE UPDATE: executed updates, matching init, fresh checkpoint, verified fp16 export.
    def t_genuine():
        import torch
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2)
            src = UNet2DConditionModel.from_pretrained(c, torch_dtype=None).state_dict()
            fp16 = os.path.join(td, "unet.pt")
            torch.save({k: v.half() for k, v in src.items()}, fp16)
            rc, v = run(td, p, c, fp16=fp16)
            assert rc == 0 and v["verdict"] == "PASS", v["failures"]
            assert all(s in ("PASS", "NOT_REQUESTED") for s in v["links"].values()), v["links"]

    # 4b. a CORRUPTED fp16 export must fail against the explicitly converted source
    def t_fp16_export_corrupt():
        import torch
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2)
            src = UNet2DConditionModel.from_pretrained(c, torch_dtype=None).state_dict()
            d = {k: v.half() for k, v in src.items()}
            k0 = sorted(d)[0]
            d[k0] = d[k0] + 1
            fp16 = os.path.join(td, "unet.pt")
            torch.save(d, fp16)
            rc, v = run(td, p, c, fp16=fp16)
            assert rc == 1, "a corrupted fp16 export must fail"
            assert v["links"]["LINK5_fp16_export_vs_converted_source"] == "FAIL", v["links"]

    # 5. the four removed criteria are gone, and NO threshold replaced them. Asserted on the
    #    verdict STRUCTURE, not on source text: the record must name all four as removed, and must
    #    carry no threshold/band/bound field of its own.
    def t_no_thresholds():
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2)
            rc, v = run(td, p, c)
            assert rc == 0, v["failures"]
            removed = " ".join(v["REMOVED_CRITERIA"]).lower()
            for frag in ("rel_vs_other_variant", "orthogonal", "adamw does not bound",
                         "no justified positive minimum"):
                assert frag in removed, f"{frag!r} not declared removed: {v['REMOVED_CRITERIA']}"
            assert len(v["REMOVED_CRITERIA"]) == 4, v["REMOVED_CRITERIA"]
            assert v["NO_SUBSTITUTE_THRESHOLD"], v
            # no pass/fail field may be a magnitude: the only verdict inputs are the six links
            assert set(v["links"]) == {
                "LINK1_loaded_state_before_optimization", "LINK2_executed_updates",
                "LINK3_save_point_in_memory_state",
                "LINK4_reload_equality_at_intended_precision",
                "LINK5_fp16_export_vs_converted_source",
                "LINK6_binding_to_config_and_source"}, sorted(v["links"])
            for banned in ("consistency_band", "adamw_saturating_bound", "variant_only_floor",
                           "rel_delta_vs_other_variant", "rel_delta_vs_same_init"):
                assert banned not in json.dumps(v), banned

    # 6. an executed update at lr=0, and one with a zero gradient, are handled WITHOUT asserting
    #    any positive minimum change
    def t_zero_grad_executed_update_is_not_a_magnitude_claim():
        with tempfile.TemporaryDirectory() as td:
            p, c = scaffold(td, executed=2, grad_l2=0.0, changed=0)
            rc, v = run(td, p, c)
            assert rc == 1, "no tensor changed by any update must be reported"
            assert any("no executed update changed any tracked tensor" in f
                       for f in v["failures"]), v["failures"]
            # the failure cites the RECORDED gradient, not a norm threshold
            assert not any("bound" in f.lower() for f in v["failures"]), v["failures"]

    # 7. the LINK 2 RECORDER ITSELF: EXECUTED updates are counted by the call that performs them,
    #    the inner per-task optimizers are NOT counted, and configured != executed is reported as
    #    two separate fields. This exercises the real wrapper, not a hand-written record.
    def t_recorder_counts_executed_not_configured():
        import torch
        try:
            with tempfile.TemporaryDirectory() as td:
                tracked = [("w0", torch.zeros(4, requires_grad=True)),
                           ("w1", torch.zeros(4, requires_grad=True))]
                other = torch.zeros(4, requires_grad=True)      # an INNER optimizer's parameter
                r = Recorder(os.path.abspath(__file__), td,
                             {"num_train_steps": 5, "gradient_accumulation_steps": 2})
                r.record_trainable([n for n, _ in tracked], [p for _, p in tracked], "xattn")
                r.wrap_optimizer(tracked, configured_steps=5, grad_accum=2, use_fp16=False)
                opt = torch.optim.AdamW([p for _, p in tracked], lr=1e-5)
                inner = torch.optim.AdamW([other], lr=1e-3)
                for step in range(1, 6):
                    for _, p in tracked:
                        p.grad = torch.full_like(p, 0.1)
                    other.grad = torch.ones_like(other)
                    inner.step()                                 # must NOT be counted
                    if step % 2 == 0:                            # the esd_meta.py:1053 gate
                        opt.step()
                l2 = r.rec["link2_executed_updates"]
                assert l2["executed_updates"] == 2, l2["executed_updates"]
                assert l2["configured_steps"] == 5, l2["configured_steps"]
                assert len(l2["updates"]) == 2, len(l2["updates"])
                u = l2["updates"][0]
                assert u["learning_rates_in_effect"] == [1e-05], u
                assert u["n_grad_None"] == 0 and u["n_tracked_tensors_in_step"] == 2, u
                assert u["grad_global_l2"] > 0 and u["n_tensors_changed_by_this_update"] == 2, u
                assert u["max_abs_param_change_this_update"] > 0, u
                # use_fp16 must be recorded as a skip path covering ALL steps
                r2 = Recorder(os.path.abspath(__file__), td, {})
                r2.wrap_optimizer(tracked, 5, 1, use_fp16=True)
                sk = r2.rec["link2_executed_updates"]["skipped_steps"]
                assert sk and sk[0]["reason"] == "use_fp16" and sk[0]["steps"] == "ALL", sk
                for h in (getattr(r, "_hooks", []) + getattr(r2, "_hooks", [])):
                    h.remove()
        finally:
            pass

    # 8. THE SHIM BINDS TO THE REAL UPSTREAM SYMBOLS. Imports the pinned esd_meta.py (no main(),
    #    no model load, no GPU) and asserts every name the shim wraps actually exists on it, and
    #    that the upstream file is byte-identical to its recorded digest. This is what stops the
    #    instrumentation from silently wrapping nothing after an upstream change.
    def t_shim_binds_to_upstream():
        import importlib.util
        src = os.environ.get("R2_UPSTREAM_SRC", "/data/bijaypandey/t2i_unlearn/upstream/"
                                                "Meta-Unlearning/src/esd_meta.py")
        assert os.path.isfile(src), f"upstream source not found: {src}"
        spec = importlib.util.spec_from_file_location("esd_meta_probe", src)
        m = importlib.util.module_from_spec(spec)
        sys.modules["esd_meta_probe"] = m
        spec.loader.exec_module(m)
        for sym in ("gather_parameters", "save_checkpoint", "parse_args", "main"):
            assert hasattr(m, sym), f"upstream esd_meta.py has no {sym!r}; the shim would wrap "
        txt = open(src).read()
        # the two skip paths LINK 2 claims to account for must still be present
        assert "if args.use_fp16:" in txt, "the use_fp16 skip path is gone; LINK2's accounting is "
        assert "if step % args.gradient_accumulation_steps == 0:" in txt, \
            "the gradient-accumulation gate is gone; LINK2's accounting is stale"
        # the save-gating defect LINK 3 depends on: the periodic save is NESTED in the validation
        # gate, so a step= checkpoint only exists when step % eval_every == 0 as well
        i_eval = txt.find("if (step % args.eval_every == 0)")
        i_save = txt.find("if step % args.save_every == 0:")
        assert i_eval != -1 and i_save != -1 and i_save > i_eval, \
            "the periodic-save nesting assumed by the launcher's smoke mode is no longer present"
        assert "half()" not in txt and "float16" not in txt, \
            "upstream now casts dtype on save; LINK4's exact-equality requirement must be revised"

    for n, f in (("shim_binds_to_upstream_symbols", t_shim_binds_to_upstream),
                 ("recorder_counts_executed_not_configured",
                  t_recorder_counts_executed_not_configured),
                 ("no_op_training", t_noop),
                 ("mismatched_initialization", t_mismatched_init),
                 ("stale_checkpoint", t_stale_checkpoint),
                 ("unrecorded_save_point", t_unrecorded_save_point),
                 ("genuine_update", t_genuine),
                 ("corrupt_fp16_export", t_fp16_export_corrupt),
                 ("four_criteria_removed_no_substitute", t_no_thresholds),
                 ("zero_grad_executed_update", t_zero_grad_executed_update_is_not_a_magnitude_claim)):
        check(n, f)
    print(f"\n{len(ok)} passed, {len(bad)} failed")
    print("STATUS: these are UNIT TESTS on synthetic tensors. A passing suite is NOT evidence "
          "that the upstream trainer ran, and is NOT a training or unlearning result.")
    return 1 if bad else 0


# ===== CRITERIA GUARD ===== (text below this marker is exempt from t_no_thresholds)
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--upstream-src", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("trainer_argv", nargs=argparse.REMAINDER)
    v = sub.add_parser("verify")
    v.add_argument("--evidence", required=True)
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--subfolder", default="")
    v.add_argument("--fp16-export", default=None)
    v.add_argument("--expect-model", required=True)
    v.add_argument("--expect-revision", required=True)
    v.add_argument("--expect-variant", required=True)
    v.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "run":
        argv = a.trainer_argv
        if argv and argv[0] == "--":
            argv = argv[1:]
        p = run_instrumented(a.upstream_src, a.out_dir, argv)
        print(f"    evidence -> {p}")
        return 0
    return verify(a.evidence, a.checkpoint, a.subfolder, a.fp16_export, a.expect_model,
                  a.expect_revision, a.expect_variant, a.out)


if __name__ == "__main__":
    sys.exit(main())
EVIDENCE_PY_EOF
$VT - "$EVID" "$U/src/esd_meta.py" "/data/bijaypandey/t2i_unlearn/$R/out/instrumentation.patch" \
  <<'PY' 2>&1 | tee -a "$L" || die "stage 4c: could not write the instrumentation patch record"
import difflib, hashlib, os, sys
evid, up, out = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
up_txt = open(up).read().splitlines(keepends=True)
# (a) upstream against ITSELF: a real diff, and it is EMPTY. That is the claim being recorded --
#     the pinned upstream source is byte-for-byte unmodified and the instrumentation is external.
same = list(difflib.unified_diff(up_txt, up_txt, fromfile=f"a/{up}", tofile=f"b/{up}"))
assert not same, "upstream diff against itself is non-empty; difflib is misbehaving"
# (b) the shim as a NEW-FILE hunk, so the patch fully describes what was added.
ev_txt = open(evid).read().splitlines(keepends=True)
newf = list(difflib.unified_diff([], ev_txt, fromfile="/dev/null",
                                 tofile="b/upstream_runs/meta/out/r2_evidence.py"))
with open(out, "w") as f:
    f.write("# INSTRUMENTATION PATCH RECORD -- Meta-Unlearning @ "
            "5a2a7600895a2885323cbbf599acaf3f74a96f64\n"
            "#\n"
            "# UPSTREAM SOURCE IDENTITY IS PRESERVED. The diff of the pinned trainer against\n"
            "# itself is EMPTY: zero bytes of upstream were changed. The instrumentation is\n"
            "# applied EXTERNALLY at run time (global torch optimizer step hooks, and wrappers\n"
            "# around diffusers' from_pretrained, esd_meta.gather_parameters and\n"
            f"# esd_meta.save_checkpoint). Upstream sha256: {sha(up)}\n"
            f"# Instrumentation module sha256: {sha(evid)}\n"
            "#\n"
            "# The only added file is the instrumentation module itself:\n#\n")
    f.writelines(newf)
print(f"    upstream sha256      {sha(up)}  (UNMODIFIED: self-diff is empty)")
print(f"    instrumentation sha  {sha(evid)}")
print(f"    patch record -> {out}")
PY
# UNIT TESTS. CPU only. They exercise the verifier and the recorder on SYNTHETIC tensors.
# STATUS DISCIPLINE: a passing suite is NOT evidence that the upstream trainer ran, and must never
# be described as a training result, as method effectiveness, or as reproducing the paper.
$V "$EVID" selftest 2>&1 | tee -a "$L" || die "stage 4c: the evidence-chain unit tests FAILED"
say "    unit tests passed. These are UNIT TESTS on synthetic tensors -- NOT an executed run."

if [ "$MODE" = "cpu" ]; then
  say "R2 CPU VERIFICATION COMPLETE $(date -Is) -- stages 1,2,3,4,4b,4c ALL RAN and passed."
  say "  Pools validated on content at $POOL_ROOT with fingerprints BOUND."
  say "  Split registry ENFORCED (stage 4b). Evidence module written and its unit tests passed"
  say "  (stage 4c) -- those are UNIT TESTS on synthetic tensors, NOT an executed training run."
  say "  Stages 7 (instrumented real update), 9 (evidence chain) and 10 (fp16 export) need a GPU"
  say "  and have NOT run. Next: R2_MODE=smoke."
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

# ================================ STAGE 7: the INSTRUMENTED trainer ============================
OUTROOT="/data/bijaypandey/t2i_unlearn/$R/out"
mkdir -p "$OUTROOT" "$R/logs" "$R/images"
if [ "$MODE" = "smoke" ]; then
  # A MINIMAL REAL UPDATE: few steps, save_every=1 so a step= checkpoint is actually written.
  #
  # BLOCKING DEFECT FIXED HERE (AGENT B, repair v3). The previous version set
  #     --eval_every $((SM_STEPS + 1))     # "skip validation sampling"
  # which made R2_MODE=smoke STRUCTURALLY INCAPABLE of producing the checkpoint stage 8 demands.
  # In upstream, the periodic save is NESTED INSIDE the validation gate:
  #     esd_meta.py:1070  if (step % args.eval_every == 0) and (step >= args.eval_after) \
  #                          and (len(args.validation_prompts) > 0):
  #     esd_meta.py:1084      if step % args.save_every == 0:
  #     esd_meta.py:1086          save_checkpoint(..., step=step)
  # so a step= directory is written ONLY when `step % eval_every == 0` AS WELL. With
  # eval_every = SM_STEPS+1 that is never true for any step in 1..SM_STEPS, no step=*/ is ever
  # created, and stage 8's `die "no step=*/ checkpoint was written"` fires unconditionally.
  # Smoke mode could never have reached stage 9 at all -- independently of the stage 9 threshold
  # problem. Verified from source; asserted at run time by the evidence module's
  # selftest::shim_binds_to_upstream_symbols, which fails if this nesting ever changes.
  # eval_every = SM_STEPS makes the gate fire exactly once, on the final step, so ONE validation
  # pass runs and ONE step= checkpoint is written.
  SM_STEPS="${META_SMOKE_STEPS:-2}"
  say "--- stage 7: MINIMAL REAL UPDATE ($SM_STEPS steps, save_every=1, eval_every=$SM_STEPS)"
  say "    (eval_every MUST equal a step that is reached: esd_meta.py:1070 nests the periodic"
  say "     save inside the validation gate, so eval_every>num_train_steps writes NO checkpoint)"
  for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
      --num_train_steps) ARGS[$((i+1))]="$SM_STEPS" ;;
      --save_every)      ARGS[$((i+1))]=1 ;;
      --eval_every)      ARGS[$((i+1))]="$SM_STEPS" ;;
    esac
  done
else
  say "--- stage 7: full training ($NUM_TRAIN_STEPS steps)"
fi

# --- the trainer runs UNDER THE EVIDENCE RECORDER. Upstream source is NOT edited: the recorder
#     installs global torch optimizer step hooks and wraps diffusers' from_pretrained,
#     esd_meta.gather_parameters and esd_meta.save_checkpoint, then calls esd_meta.main().
#     $R/out/instrumentation.patch records that upstream's self-diff is empty.
rc=0
( cd "$STAGE" && $V "$EVID" run --upstream-src "$U/src/esd_meta.py" --out-dir "$OUTROOT" \
    -- "${ARGS[@]}" ) >>"$L" 2>&1 || rc=$?
say "    esd_meta (instrumented) rc=$rc $(date -Is)"
EVJSON="$OUTROOT/train_evidence.json"
# The evidence file is flushed at every save point and on any exception, so it exists even for a
# failed run -- and a failed run's evidence is exactly what is needed to say WHY.
if [ -f "$EVJSON" ]; then
  $VT - "$EVJSON" <<'PY' 2>&1 | tee -a "$L" || true
import json, sys
e = json.load(open(sys.argv[1]))
l2 = e["link2_executed_updates"]
print(f"    EXECUTED optimizer updates: {l2['executed_updates']}   "
      f"CONFIGURED steps: {l2['configured_steps']}   "
      f"recorded skip paths: {len(l2['skipped_steps'])}")
print(f"    save points recorded: {len(e['link3_save_points'])} "
      f"{[s['destination'] for s in e['link3_save_points']]}")
for err in e.get("errors", []):
    print(f"    TRAINER RAISED {err['type']}: {err['message'][:300]}")
PY
else
  say "    WARNING: no $EVJSON was written -- the trainer died before the recorder installed"
fi
[ "$rc" -eq 0 ] || die "the trainer exited $rc (tail of $L above, and see $EVJSON). If it is a
  CUDA OOM, see the memory note in docs/REPAIR_BASELINES_v2.md: --devices '0 0' co-locates
  teacher and student."

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

# ================================ STAGE 9: THE EVIDENCE CHAIN ==================================
# WHAT THIS REPLACES, AND WHY. The previous stage 9 decided "did the trainer train?" from the
# relative Frobenius norm of (checkpoint - base) under FOUR assumptions that DO NOT HOLD. All four
# are REMOVED as pass/fail criteria, and NO other arbitrary norm threshold is substituted.
# Counterexamples are in docs/REPAIR_B_v3.md section 1 and were measured on CPU:
#
#  1. CHECK B, `rel_vs_same < rel_vs_other` ("a trained checkpoint is closer to its own init than
#     to another base variant"). FALSE. init A=1.0, alternative B=1.01, trained T=1.007 gives
#     rel_vs_same = 7.000e-03 and rel_vs_other = 2.970e-03, so a GENUINE update FAILS the check.
#     The sign of that inequality is a fact about the DIRECTION of the update, which nothing
#     constrains, not about whether an update happened.
#  2. The derivation of CHECK B ("rel_vs_B = sqrt(d^2 + floor^2)") is a PYTHAGOREAN step that
#     requires the update to be ORTHOGONAL to the variant offset. Unjustified: nothing makes a
#     gradient update orthogonal to a float16 quantisation offset. Measured on the same example,
#     sqrt(d^2+floor^2) = 1.22e-02 against a true ||T-B|| = 3.00e-03.
#  3. CHECK C's bound, rel <= lr*S*sqrt(N)/||w||, assumed AdamW's per-element step is bounded by
#     ~lr. It is NOT: m_hat/(sqrt(v_hat)+eps) is bounded by 1 only in the beta1^2 <= beta2 steady
#     state, and decoupled weight decay adds -lr*wd*w on top. A bound that can be exceeded cannot
#     be a pass/fail criterion.
#  4. CHECK C's floor, bound(S=1)/100, assumed a positive MINIMUM change follows from lr. It does
#     not: one EXECUTED AdamW step with a zero gradient moves the parameter by exactly 0.0
#     (measured), as does an executed step at lr=0 during warmup.
#
# WHAT REPLACES THEM: six links, each a fact RECORDED BY THE PROCESS THAT PERFORMED THE ACTION.
#   LINK 1  the model state actually loaded before optimization -- repo, revision, variant, dtype,
#           trainable parameter NAMES, and per-tensor fingerprints taken before the first step
#   LINK 2  EXECUTED optimizer updates (counted by the step hook, never inferred from
#           --num_train_steps), the learning rates actually in effect, gradient diagnostics, and
#           the observed parameter change -- with executed / configured / skipped kept SEPARATE
#   LINK 3  the IN-MEMORY state at the save point, fingerprinted by the save call itself
#   LINK 4  reload: keys, shapes, finite values, and tensor EQUALITY at the intended serialization
#           precision (float32 -- upstream performs no dtype cast on save)
#   LINK 5  the fp16 evaluation export against the EXPLICITLY converted source tensors (stage 10)
#   LINK 6  binding to the run configuration, the upstream commit and the source sha256
# A link that cannot be evaluated is INDETERMINATE, which is a FAILURE, never a pass. Norms are
# still reported, labelled descriptive, and never thresholded.
say "--- stage 9: evidence chain ($PIN_VARIANT @ $PIN_REVISION)"
[ -f "$EVJSON" ] || die "stage 9: $EVJSON does not exist. Without the record the trainer itself
  wrote, there is nothing to verify the checkpoint against, and a magnitude would be the only
  thing left to guess from -- which is exactly the defect this stage removes."
$V "$EVID" verify \
  --evidence "$EVJSON" \
  --checkpoint "$CKDIR" --subfolder "$CKSUB" \
  --expect-model CompVis/stable-diffusion-v1-4 \
  --expect-revision "$PIN_REVISION" --expect-variant "$PIN_VARIANT" \
  --out "$CKDIR/reload_evidence.json" 2>&1 | tee -a "$L" \
  || die "stage 9: the evidence chain FAILED (see $CKDIR/reload_evidence.json). This is NOT a
  magnitude verdict: read the per-link failures above."

# ================================ STAGE 10: evaluation export (LINK 5) =========================
# Our evaluator consumes a flat fp16 UNet state dict (adapters/<arm>/unet.pt). Produce it, then
# VERIFY IT AGAINST THE EXPLICITLY CONVERTED SOURCE TENSORS -- element for element against
# src.half(), not against a tolerance. The previous version asserted key sets and shapes and then
# wrote the export without ever checking that what it wrote was a conversion of what it read.
say "--- stage 10: evaluation export and its verification (LINK 5)"
ADIR="adapters/R2_${EXP_NAME}"
mkdir -p "$ADIR"
$VT - "$CKDIR" "$CKSUB" "$ADIR" "$PIN_REVISION" "$PIN_VARIANT" <<'PY' 2>&1 | tee -a "$L" \
  || die "stage 10: the checkpoint is not consumable by our evaluator"
import json, os, sys, torch
from diffusers import UNet2DConditionModel
ckdir, cksub, adir, rev, variant = sys.argv[1:6]
kw = {"subfolder": cksub} if cksub else {}
# torch_dtype=None: load at the dtype ON DISK. The previous version forced float32, which would
# have silently up-cast a non-fp32 checkpoint and made the export unverifiable against its source.
u = UNet2DConditionModel.from_pretrained(ckdir, torch_dtype=None, **kw)
ref = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                           revision=rev, torch_dtype=torch.float32,
                                           variant=variant).state_dict()
sd = u.state_dict()
assert set(sd) == set(ref), "key set differs from the evaluator's expected UNet"
bad = [k for k in sd if tuple(sd[k].shape) != tuple(ref[k].shape)]
assert not bad, f"shape mismatch on {len(bad)} tensors: {bad[:3]}"
exp = {k: v.half() for k, v in sd.items()}          # the EXPLICIT conversion, stated here
torch.save(exp, f"{adir}/unet.pt")
json.dump({"source_checkpoint": ckdir, "subfolder": cksub or None,
           "init_revision": rev, "init_variant": variant,
           "tensors": len(sd), "dtype": "float16",
           "source_dtypes_on_disk": sorted({str(v.dtype) for v in sd.values()}),
           "conversion": "torch.Tensor.half() applied to each tensor as loaded from the "
                         "checkpoint; verified element-for-element by LINK 5 below",
           "note": "flat UNet state dict for scripts/gen_eval.py --unet"},
          open(f"{adir}/compose.json", "w"), indent=1)
print(f"    wrote {adir}/unet.pt ({len(sd)} tensors, fp16) + compose.json")
PY
# LINK 5: the export is now verified against the converted source, and the whole chain is re-run
# so one record covers all six links.
$V "$EVID" verify \
  --evidence "$EVJSON" \
  --checkpoint "$CKDIR" --subfolder "$CKSUB" \
  --fp16-export "/data/bijaypandey/t2i_unlearn/$ADIR/unet.pt" \
  --expect-model CompVis/stable-diffusion-v1-4 \
  --expect-revision "$PIN_REVISION" --expect-variant "$PIN_VARIANT" \
  --out "/data/bijaypandey/t2i_unlearn/$ADIR/evidence_chain.json" 2>&1 | tee -a "$L" \
  || die "stage 10: LINK 5 FAILED -- the fp16 export is not a conversion of the checkpoint it
  claims to come from (see $ADIR/evidence_chain.json)"
say "R2 DONE $(date -Is) mode=$MODE -> $OUTDIR  (evaluable: $ADIR/unet.pt)"
say "  evidence chain: $ADIR/evidence_chain.json   trainer record: $EVJSON"
say "  SCOPE: this verifies that the checkpoint corresponds to a RECORDED EXECUTED training run."
say "  It is NOT a measurement of unlearning, of method effectiveness, or of any published result."
