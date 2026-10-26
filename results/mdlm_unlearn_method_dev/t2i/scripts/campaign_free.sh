#!/bin/bash
# Which of OUR GPUs (0-1) is free? Decided by PER-GPU CAMPAIGN OWNERSHIP, not a memory reading.
#
# v2. v1 replaced the broken "<2000 MiB" gate (a campaign DIPS to ~1.4 GiB between attack cells, so
# a memory gate can fire mid-campaign), but over-corrected: it refused ALL GPUs whenever ANY runner
# was alive, which wrongly blocked the second of our two allocated GPUs.
#
# A GPU i is free iff:
#   (a) no campaign runner of ours declares CUDA_VISIBLE_DEVICES=i, AND
#   (b) no python process of ours holds a CUDA context on i, AND
#   (c) its memory is low enough that no OTHER user is on it.
# Prints every free index, one per line.
set -u
ME=$(id -u)
declare -A CLAIMED=()
for pid in $(pgrep -u "$ME" -f 'run_[a-z0-9_]*\.sh' 2>/dev/null); do
  [ -r "/proc/$pid/environ" ] || continue
  d=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ -n "${d:-}" ] && CLAIMED[$d]=1
done
# a runner may export CUDA_VISIBLE_DEVICES only in its child; also claim GPUs our python holds
for i in 0 1; do
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null); do
    [ -d "/proc/$pid" ] && [ "$(stat -c %u "/proc/$pid" 2>/dev/null)" = "$ME" ] && CLAIMED[$i]=1
  done
done
for i in 0 1; do
  [ -n "${CLAIMED[$i]:-}" ] && continue
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null)
  [ -n "$USED" ] && [ "$USED" -lt 2000 ] && echo "$i"
done
