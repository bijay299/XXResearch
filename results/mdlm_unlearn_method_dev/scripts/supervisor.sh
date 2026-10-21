#!/bin/bash
# Keeps the scheduler alive. The scheduler reads scripts/JOBS.tsv once and exits when the list
# is exhausted; appending jobs afterwards would otherwise sit unrun until someone noticed. The
# supervisor re-runs it whenever it exits, and the scheduler skips anything already complete,
# so appends are always picked up and the pipeline never goes idle because a list ran out.
#
# Exits when a full pass dispatches nothing new, so it does not spin forever once the project's
# queue is genuinely empty.
set -u
cd /data/bijaypandey/mdlm_unlearn

for pass in $(seq 1 200); do
  before=$(ls outputs/arm_*.json adapters/*/report.json 2>/dev/null | wc -l)
  echo "[sup] pass $pass start (markers=$before)"
  ./scripts/scheduler.sh >> outputs/log_scheduler.txt 2>&1
  # let the last dispatched jobs finish before counting
  sleep 120
  after=$(ls outputs/arm_*.json adapters/*/report.json 2>/dev/null | wc -l)
  echo "[sup] pass $pass done (markers=$after)"
  if [ "$after" -le "$before" ]; then
    # nothing new completed and nothing new dispatched -> queue is drained
    pending=$(awk '!/^#/ && NF {print $2}' scripts/JOBS.tsv | while read -r l; do
      [ -f "outputs/arm_$l.json" ] || [ -f "adapters/$l/report.json" ] || echo "$l"
    done | wc -l)
    if [ "$pending" -eq 0 ]; then echo "[sup] queue drained, exiting"; exit 0; fi
    echo "[sup] $pending still pending; waiting before next pass"
    sleep 180
  fi
done
echo "[sup] pass limit reached"
