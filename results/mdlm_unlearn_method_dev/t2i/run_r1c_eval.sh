#!/bin/bash
# R1c: compose and evaluate the OFFICIAL IMMA checkpoint, then attack it with OUR protocol.
# Author-protocol reproduction and our-threat-model evaluation stay SEPARATELY LABELLED:
#   REF_imma_vangogh*  = the author-protocol artifact
#   attacks are tagged R1C_* and reported as "our common attack on an official checkpoint".
#
# AUDIT REPAIR 2026-10-26 (docs/AGENT2_BASELINES_FINDINGS.md A-6, A-8).
#
# WHY THE CONTROL IS MANDATORY. An IMMA-immunised checkpoint is deliberately damaged so that a
# LATER adaptation fails; the IMMA paper evaluates only the POST-ADAPTATION model (README L146-151)
# and never the immunised model standalone. Scoring it standalone with a closed-set 12-way CLIP
# argmax (scripts/gen_eval.py L48/L81/L127) can therefore collapse onto a single label and produce
# numbers that look like a result and are not one -- which is exactly what
# outputs/gen_REF_imma_vangogh.json did (target 0.867, eight non-targets at exactly 0.000).
# So this script now evaluates the ESD-ONLY composition FIRST. That isolates the direction of the
# ESD delta behaviourally and gives the immunised arm something to be compared against. A
# QUARANTINE note is written beside every immunised adapter.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

if [ -f FREEZE_ACTIVE ] && [ "${R1_FREEZE_OVERRIDE:-0}" != "1" ]; then
  echo "FREEZE_ACTIVE is present; refusing to launch GPU work. Set R1_FREEZE_OVERRIDE=1 to run." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r1c_eval.log
R=upstream_runs/imma
A="${R1C_ARM:-REF_imma_vangogh_s20}"       # which IMMA artifact under $R to evaluate
C=ESD_vangogh_official                     # the ESD-only control arm
ESD="$R/diffusers-VanGogh-ESDx1-UNET.pt"
IMMA="$R/$A/imma_unet_xatten_layer.pt"
[ -f "$ESD" ]  || { echo "missing $ESD" >&2; exit 1; }
[ -f "$IMMA" ] || { echo "missing $IMMA -- run run_r1b_imma_train.sh first" >&2; exit 1; }

while :; do G=$(bash scripts/campaign_free.sh|head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R1c on GPU $G $(date -Is) arm=$A control=$C" >>"$L"

# ---- 1. CONTROL: base + ESD only. Must run, and must be reported alongside the IMMA arm.
if [ ! -f "adapters/$C/unet.pt" ]; then
  venv_imma/bin/python scripts/r1_eval_imma.py --esd "$ESD" --esd-only \
    --out "adapters/$C" >>"$L" 2>&1 || { echo "CONTROL COMPOSE FAILED" >>"$L"; exit 1; }
fi
[ -f "adapters/$C/unet.pt" ] || { echo "CONTROL COMPOSE PRODUCED NOTHING" >>"$L"; exit 1; }
if [ ! -f "outputs/gen_$C.json" ]; then
  venv_t2i/bin/python scripts/gen_eval.py --label "$C" --unet "adapters/$C/unet.pt" \
    --target van_gogh >>"$L" 2>&1 || { echo "CONTROL EVAL FAILED" >>"$L"; exit 1; }
fi

# ---- 2. the IMMA arm: base + ESD + S, strict composition (no fuzzy keys, no reshape)
if [ ! -f "adapters/$A/unet.pt" ]; then
  venv_imma/bin/python scripts/r1_eval_imma.py --esd "$ESD" --imma "$IMMA" \
    --out "adapters/$A" >>"$L" 2>&1 || { echo "COMPOSE FAILED" >>"$L"; exit 1; }
fi
[ -f "adapters/$A/unet.pt" ] || { echo "COMPOSE PRODUCED NOTHING" >>"$L"; exit 1; }
cat > "adapters/$A/QUARANTINE.md" <<'EOF'
# Not an IMMA result on its own

This adapter is an IMMA-IMMUNISED checkpoint (base SD-1.4 + published VanGogh-ESDx1 delta + IMMA's
160-tensor defence scope). IMMA deliberately degrades the pre-trained weights so that a LATER
adaptation fails. The IMMA paper evaluates only the POST-ADAPTATION model
(upstream/IMMA/README.md L146-151) and never the immunised model standalone.

Therefore:
  * `target_suppression_acc`, `utility_all_nontarget` and `legacy_retain8` measured on THIS adapter
    are NOT IMMA results and must not be reported as "IMMA suppresses / fails to suppress X" or as
    "the IMMA reference destroys utility".
  * The scorer is a closed-set 12-way CLIP argmax (scripts/gen_eval.py L48/L81/L127). If cross-
    attention conditioning is damaged, every image can collapse onto one label, which yields a high
    "target" number and exact zeros elsewhere without measuring suppression at all.
  * Compare against adapters/ESD_vangogh_official/ (ESD only, no IMMA) which this script evaluates
    first, and against the post-adaptation protocol IMMA actually specifies.

See docs/AGENT2_BASELINES_FINDINGS.md A-6.
EOF
if [ ! -f "outputs/gen_$A.json" ]; then
  venv_t2i/bin/python scripts/gen_eval.py --label "$A" --unet "adapters/$A/unet.pt" \
    --target van_gogh >>"$L" 2>&1 || { echo "IMMA EVAL FAILED" >>"$L"; exit 1; }
fi

# ---- 3. report both arms together; never the IMMA arm alone
venv_t2i/bin/python - "$A" "$C" <<'PY' >>"$L" 2>&1
import json, sys
a, c = sys.argv[1], sys.argv[2]
def m(lbl):
    d = json.load(open(f"outputs/gen_{lbl}.json"))
    return d["active_target_metrics"], d["per_concept"]
am, ap = m(a); cm, cp = m(c)
for lbl, x in ((f"CONTROL {c}", cm), (f"IMMA    {a}", am)):
    print(f"[R1c] {lbl}: target={x['target_suppression_acc']:.4f} "
          f"retain8={x['legacy_retain8']:.4f} allnt={x['utility_all_nontarget']:.4f}")
zeros = sum(1 for k, v in ap.items() if v["clip_acc"] == 0.0)
print(f"[R1c] IMMA arm: {zeros}/{len(ap)} concepts at EXACTLY 0.000")
if zeros >= 0.6 * len(ap) and am["target_suppression_acc"] > 0.5:
    print("[R1c] LABEL-COLLAPSE SIGNATURE: most concepts at exact zero with a high target score. "
          "The closed-set argmax has most likely collapsed onto the target label. This arm's "
          "standalone numbers are NOT interpretable; see adapters/%s/QUARANTINE.md" % a)
print(f"[R1c] K1: {'CLEAR -> attack grid' if am['legacy_retain8']>=0.60 else 'FAIL -> not attacked'}")
PY

R8=$(venv_t2i/bin/python -c "import json;print(json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
awk -v r="$R8" 'BEGIN{exit !(r>=0.60)}' || { echo "K1 fail (retain8=$R8), not attacked" >>"$L"; exit 0; }
for SD in 0 1 6; do
  K="R1C_${A}_sd${SD}_lr1e-5_s300"
  [ -f "outputs/atk_${K}.json" ] && continue
  venv_t2i/bin/python scripts/attack_relearn.py --concept van_gogh \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --steps 300 --label "$K" >>"$L" 2>&1 \
    || { echo "ATTACK $K FAILED" >>"$L"; exit 1; }
  rm -rf "adapters/_atk_$K" 2>/dev/null || true
done
echo "R1c DONE $(date -Is)" >>"$L"
