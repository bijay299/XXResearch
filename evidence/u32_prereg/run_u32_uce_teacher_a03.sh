#!/bin/bash
# U32 -- genuine-eraser teacher (UCE) at the utility-preserving damage level (alpha=0.3).
# Pre-registered in docs/PREREG_U32_UCE_TEACHER_A03.md -- read that before interpreting anything.
#
# THE CELL THAT WAS NEVER BUILT. Damage-then-distil failed twice here for DIFFERENT, now-diagnosed
# reasons:
#   U23  AdvUnlearn teacher (LEAKS -- audit reads 0.3333 x4, student images show churches) + a=0.3
#        -> student inherited the leak. Failed bar 1 (church still shows up).
#   U24  UCE teacher (GENUINE -- u24_uce_s6000 audits 0.0000 x4) + a=0.5
#        -> erasure transferred, utility destroyed (retain8 0.2917). Failed bar 2.
# Teacher axis wrong in the first, damage axis wrong in the second. UCE teacher + a=0.3 sets both
# correctly and has never been run (queued as U25, closed unrun).
#
# THE THREE BARS, in the owner's terms. Target-prompt image quality is NOT one of them.
#   1. ask for a church, don't get a church   (target <= 0.15, CONFIRMED BY IMAGE AUDIT)
#   2. every other concept still works        (retain8 >= 0.85)
#   3. it stays that way after fine-tuning    (max-over-admissible H <= 0.70)
# Bar 3 is the open problem: all 86 arms measured in this project reach 1.0000.
#
# THE AUDIT IS A VALIDITY CHECK, NOT A SCORE. U30 showed target_suppression_acc can be satisfied by
# a speckle overlay on a visible church. So bar 1 is confirmed against the image before any
# robustness number is reported. It contributes nothing to the criterion.
#
# ADMISSIBLE-CELL COUNT IS PART OF THE RESULT. U30 found our arms admitted 3 of 9 cells against the
# baselines' 22/24, 16/18, 21/21. A max over 3 cells is not comparable to a max over 20. If this
# arm admits <= 3 cells, NO robustness claim is made regardless of H.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
T=adapters/uce_church/unet.pt
[ -s "$T" ] || { echo "U32 ABORT: teacher missing" >&2; exit 1; }
source scripts/gpu_reserve.sh
gpu_reserve_hold "u32_uceT_a0.3" 180 >/dev/null || { echo "U32 ABORT: no reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U32 ABORT: no CVD" >&2; exit 1; }
echo "U32 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L=logs/u32_uce_teacher_a03.log
source scripts/run_guard.sh
rg_init "U32" "$L"
ARM="u32_uceT_a0.3"; D="adapters/$ARM"

if [ ! -s "$D/unet.pt" ]; then
  used=$(du -sBG . 2>/dev/null | awk '{gsub("G","",$1); print $1}')
  free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  if [ "${used:-0}" -ge 500 ] || [ "${free:-0}" -lt 25 ]; then
    rg_abort "space guard: ${used}GiB used vs 500GiB cap, ${free}GiB free (shared volume)"; exit 1
  fi
  echo "=== build $ARM (teacher=uce_church audited 0.0000/0.0000/0.1111/0.0000, alpha=0.3, protect=none) $(date -Is)" >>"$L"
  rg_cell venv_t2i/bin/python scripts/undo_t2i.py --teacher "$T" --concept church \
    --alpha 0.3 --beta 0.1 --protect none \
    --steps 6000 --lr 1e-5 --n-images 300 --forget-frac 0.2 --seed 0 --log-every 3000 \
    --label "$ARM" --out "$D"
  rg_require_loadable "outputs/undo_${ARM}.json" json
fi
[ -s "$D/unet.pt" ] || { rg_abort "no checkpoint"; exit 1; }

if [ ! -f "outputs/gen_${ARM}.json" ]; then
  echo "=== gen_eval $ARM $(date -Is)" >>"$L"
  rg_cell venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" --unet "$D/unet.pt" --target church
  rg_require_loadable "outputs/gen_${ARM}.json" json
fi

# BAR 1 VALIDITY CHECK -- images, 4 eval seeds, before any robustness number.
if [ ! -f "outputs/audit_${ARM}/audit.json" ]; then
  echo "=== image audit $ARM (validity check for bar 1, not a score) $(date -Is)" >>"$L"
  rg_cell venv_t2i/bin/python scripts/audit_teacher.py --unet "$D/unet.pt" --concept church \
    --retain golf_ball,french_horn,cassette_player,english_springer \
    --eval-seeds 0,1,2,3 --n-per 3 --steps 25 --out "outputs/audit_${ARM}"
  rg_require_loadable "outputs/audit_${ARM}/audit.json" json
  venv_t2i/bin/python scripts/audit_contact_sheet.py "outputs/audit_${ARM}" >>"$L" 2>&1 || true
fi

# BAR 3 -- UNRESTRICTED attacker, matched to how every baseline here was measured.
for LR in 1e-5 5e-5 1e-4; do
  for SD in 0 1 6; do
    K="U32EV_${ARM}_sd${SD}_lr${LR}_s25"
    [ -f "outputs/atk_${K}.json" ] && continue
    echo "=== $K $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/attack_relearn_v2.py --concept church \
      --unet "$D/unet.pt" --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
    rg_require_loadable "outputs/atk_${K}.json" json
  done
done
rg_finish
echo "U32 DONE $(date -Is)" >>"$L"
