#!/bin/bash
# Stop the redundant Phase-B-only queue. queue_cheap.sh supersedes it: it runs the H10-M0
# oracle first (which can kill H10 outright) and Phase B second, so leaving both alive risks
# duplicating Phase B. Run detached: killing from the interactive shell has been terminating
# that shell before the kill completes.
sleep 1
for p in $(pgrep -f "queue_phaseB.sh"); do
  kill "$p" 2>/dev/null && echo "stopped $p"
done
sleep 2
if pgrep -f "queue_phaseB.sh" >/dev/null; then
  for p in $(pgrep -f "queue_phaseB.sh"); do kill -9 "$p" 2>/dev/null && echo "force-stopped $p"; done
fi
sleep 1
pgrep -f "queue_phaseB.sh" >/dev/null && echo "RESULT: still alive" || echo "RESULT: stopped"
pgrep -f "queue_cheap.sh" >/dev/null && echo "queue_cheap: alive (correct)" || echo "queue_cheap: MISSING"
