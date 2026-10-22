#!/bin/bash
# Preserve UNDO checkpoints against the follow-on's reclaim step.
#
# WHY. run_undo_followup.sh deletes an arm's weights once its eval and RTT results both exist,
# which was the right call when /data had 213G free on a shared volume. With 60 GiB authorised,
# keeping them is worth more: FINDING20's open limitation is that our relearning attacks run at
# a SINGLE learning rate, and the adversarial lr sweep cannot be run on an UNDO arm whose
# weights have been deleted without retraining it (~85 min each).
#
# HOW. A hardlink under a second name costs ZERO extra space and survives `rm -f` of the first,
# so this defeats the reclaim without editing a script bash is actively executing.
set -u
cd /data/bijaypandey/mdlm_unlearn
while true; do
  for lab in undo_fa0p3 undo_fa0p0; do
    for f in adapters/$lab/undo_step*.bin; do
      [ -e "$f" ] || continue
      keep="$(dirname "$f")/keep_$(basename "$f")"
      if [ ! -e "$keep" ]; then
        # same inode, no extra bytes
        ln "$f" "$keep" 2>/dev/null && echo "[keep] hardlinked $keep"
      fi
    done
  done
  # stop once the pipeline is finished and nothing further will be written
  if awk '/followup\] done/{f=1} END{exit !f}' outputs/log_undo_followup.txt 2>/dev/null; then
    echo "[keep] pipeline finished, exiting"; break
  fi
  sleep 60
done
