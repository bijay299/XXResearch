#!/bin/bash
# Closes FINDING60 S5's named gap. Four arms spanning the FULL weight-distance range, so LMC is
# tested where L2 had no power: GIFT (L2 0.18, nearest base), FRP (264.53), h68v2_rf (314.50), and
# h64v2_frag (483.14 -- the destroyed arm whose low H is the degeneracy artifact). If LMC
# discriminates where L2 did not, that is a real transferable result; if every pair is barrier-free,
# the whole predictor fails to transfer and FINDING60's conclusion is complete rather than partial.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
p() { mkdir ".labels/lmc_$1" 2>/dev/null || return 0; export CUDA_VISIBLE_DEVICES=$3
  venv_t2i/bin/python scripts/lmc_probe.py --arm "$1" --concept "$2" >>"logs/lmc_$1.log" 2>&1; }
( p gift_church_b1.0 church 0; p h68v2_rf_a0.3_church church 0; echo "LMC A DONE $(date -Is)" >>logs/lmc.log ) &
( p frp_church_r3_l05 church 1; p h64v2_frag_a0.3_parachute parachute 1; echo "LMC B DONE $(date -Is)" >>logs/lmc.log ) &
wait
