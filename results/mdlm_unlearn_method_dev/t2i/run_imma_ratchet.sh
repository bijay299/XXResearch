#!/bin/bash
# FINDING59 S5 / PREREG_H82 K4: max-over-valid H = 0.867 is BELOW the ceiling, so unlike every other
# arm in this phase the ratchet is LIVE -- there is room above, and the number is provisional until
# the adversary has been pushed. Three obligations, in order of how likely each is to raise it:
#   1. more LoRA seeds -- the 0.267-0.867 spread on one arm is larger than the distance to the bar,
#      so n=2 cannot settle anything;
#   2. a STRONGER LoRA budget (more steps) -- the immunization was trained against a one-step inner
#      adversary, so a longer-running LoRA is the natural escalation;
#   3. TI and DreamBooth, the two families not yet run on this arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }
A=imma_church_frp_a1e-6_s1500
while pgrep -f "attack_relearn.py --concept church --unet adapters/imma" >/dev/null; do sleep 60; done
lora() { K="H82LR_$1_sd$2_st$3"; claim "$K" || return 0; export CUDA_VISIBLE_DEVICES=$4
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_lora.py --concept church \
    --unet "adapters/$1/unet.pt" --seed "$2" --steps "$3" --label "$K" >>logs/imma_ratchet.log 2>&1; }
ti()   { K="H82T_$1_sd$2"; claim "$K" || return 0; export CUDA_VISIBLE_DEVICES=$3
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
    --unet "adapters/$1/unet.pt" --steps 2000 --lr 1e-2 --seed "$2" --label "$K" >>logs/imma_ratchet.log 2>&1; }
db()   { K="H82DB_$1_sd$2"; claim "$K" || return 0; export CUDA_VISIBLE_DEVICES=$3
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py --concept church \
    --unet "adapters/$1/unet.pt" --seed "$2" --label "$K" --measure-utility >>logs/imma_ratchet.log 2>&1; }
( lora $A 6 300 0; lora $A 2 300 0; lora $A 0 900 0; ti $A 0 0
  echo "RATCHET A DONE $(date -Is)" >>logs/imma_ratchet.log ) &
( lora $A 3 300 1; lora $A 1 900 1; db $A 0 1
  echo "RATCHET B DONE $(date -Is)" >>logs/imma_ratchet.log ) &
wait
