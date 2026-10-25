#!/bin/bash
# DreamBooth -- GIFT's OWN threat model -- is currently measured only on beta=1.0. Running it on the
# deepest usable arms (1.3, 1.35) completes the four-adversary matrix across the whole frontier, so
# "every usable arm falls to every family" is measured at every point rather than extrapolated.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
db() { K="H81DBF_$1_sd$2"; mkdir ".labels/$K" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$3
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_dreambooth.py \
    --concept church --unet "adapters/$1/unet.pt" --seed "$2" --label "$K" --measure-utility \
    >>logs/gift_db_deep.log 2>&1; }
( db gift_church_b1.3 0 0; db gift_church_b1.35 0 0; echo "DBF A DONE $(date -Is)" >>logs/gift_db_deep.log ) &
( db gift_church_b1.3 1 1;                           echo "DBF B DONE $(date -Is)" >>logs/gift_db_deep.log ) &
wait
