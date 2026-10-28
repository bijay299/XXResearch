#!/bin/bash
# U33 -- THE THREAT LADDER.  Pre-registered in docs/PREREG_U33_THREAT_LADDER.md.
#
# WHY. Every robustness number in this project was taken at ONE attacker strength (full fine-tune,
# 25 steps, on target images), and under it ALL 86 arms read max-over-admissible H = 1.0000 --
# every published baseline and every arm of ours. A measurement on which everything scores the same
# cannot rank methods, so it cannot show the advantage the mandate requires. The saturation is a
# defect in the instrument, not a finding.
#
# This also diverges from the field: ResAlign (NeurIPS 2025) fine-tunes on DreamBench++/DiffusionDB,
# downstream data that does NOT contain the forget concept. Their threat is accidental recovery
# during legitimate use; ours is deliberate relearning on the concept itself. A method failing ours
# does not contradict their published result.
#
# WHAT THIS DOES. Varies ONE factor -- attacker capacity -- across four rungs, so methods can
# differentiate somewhere even when tied at the top:
#     L4 / L16 / L32   LoRA rank r on to_q,to_k,to_v,to_out.0   (0.09% / 0.37% / 0.74% of UNet)
#     FULL             all UNet parameters
# Learning rates differ by rung BY DESIGN (MANDATE clause 1c): U26 measured that below lr 5e-4 a
# rank-4 attacker achieves nothing at all (T_rise 0.0000), so reusing the full-FT rates would
# measure only that the attack never started.
#
# THE BENIGN RUNG IS NOT IN THIS PHASE. Matching ResAlign's threat model needs fine-tuning on data
# that does not contain the concept, which this attacker does not implement. Until it exists we
# cannot compare like-for-like with ResAlign and must not write as though we can.
#
# REPORTING. Per rung: max-over-admissible H_after, with the ADMISSIBLE CELL COUNT beside it.
# A rung with <= 2 admissible cells is reported n/a, NOT as a low score -- too few cells to take a
# maximum over. U30 found our arms admitting 3 of 9 cells being compared against baselines
# admitting 22 of 24, and that comparison was not matched.
#
# Usage: ./run_u33_threat_ladder.sh <arm> [<arm> ...]
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
[ $# -ge 1 ] || { echo "usage: $0 <arm> [<arm> ...]" >&2; exit 1; }
ARMS=("$@")
TAG=$(echo "${ARMS[0]}" | tr -c 'A-Za-z0-9' '_')
source scripts/gpu_reserve.sh
gpu_reserve_hold "u33_${TAG}" 300 >/dev/null || { echo "U33 ABORT: no reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U33 ABORT: no CVD" >&2; exit 1; }
echo "U33 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES for ${ARMS[*]}" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L="logs/u33_ladder_${TAG}.log"
source scripts/run_guard.sh
rg_init "U33_${TAG}" "$L"

for ARM in "${ARMS[@]}"; do
  U="adapters/$ARM/unet.pt"
  [ -s "$U" ] || { echo "SKIP $ARM: no checkpoint" >>"$L"; continue; }

  # --- restricted rungs: LoRA rank 4 / 16 / 32
  for R in 4 16 32; do
    for LR in 5e-4 1e-3 5e-3; do
      for SD in 0 1 6; do
        K="U33_${ARM}_L${R}_sd${SD}_lr${LR}_s25"
        [ -f "outputs/atk_${K}.json" ] && continue
        echo "=== $K $(date -Is)" >>"$L"
        rg_cell venv_t2i/bin/python scripts/attack_relearn_v3.py --concept church \
          --unet "$U" --attack-mode lora --lora-rank "$R" \
          --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
        rg_require_loadable "outputs/atk_${K}.json" json
      done
    done
  done

  # --- top rung: full fine-tune. Skipped where cells already exist (baselines have 15-24).
  EXIST=$(ls outputs/atk_*"${ARM}"_*.json 2>/dev/null | grep -vc "_L[0-9]*_sd" || true)
  if [ "${EXIST:-0}" -ge 9 ]; then
    echo "SKIP $ARM FULL rung: ${EXIST} existing full-FT cells reused as recorded" >>"$L"
  else
    for LR in 1e-5 5e-5 1e-4; do
      for SD in 0 1 6; do
        K="U33_${ARM}_FULL_sd${SD}_lr${LR}_s25"
        [ -f "outputs/atk_${K}.json" ] && continue
        echo "=== $K $(date -Is)" >>"$L"
        rg_cell venv_t2i/bin/python scripts/attack_relearn_v3.py --concept church \
          --unet "$U" --attack-mode full \
          --seed "$SD" --lr "$LR" --steps 25 --label "$K" --validate-reload
        rg_require_loadable "outputs/atk_${K}.json" json
      done
    done
  fi
  echo "=== ARM DONE $ARM $(date -Is)" >>"$L"
done
rg_finish
echo "U33 DONE ${ARMS[*]} $(date -Is)" >>"$L"
