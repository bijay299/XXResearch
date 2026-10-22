#!/bin/bash
# Wait for the UNDO alpha sweep to finish, then run the follow-on (partition eval + validated
# RTT per arm, then H17's theta0 decoder grid). Kept as a file rather than an inline `bash -c`
# because the nested quoting of the awk guard did not survive setsid.
set -u
cd /data/bijaypandey/mdlm_unlearn
while true; do
  if [ -f outputs/log_undo_sweep.txt ] && awk '/sweep\] done/{f=1} END{exit !f}' outputs/log_undo_sweep.txt; then
    break
  fi
  # bail out if nothing is training any more and the sweep never printed done
  if ! pgrep -f 'deepspeed.launcher.launch' > /dev/null 2>&1; then
    sleep 60
    if ! pgrep -f 'deepspeed.launcher.launch' > /dev/null 2>&1; then
      echo "[chain] sweep processes gone without a done marker; running follow-on anyway"
      break
    fi
  fi
  sleep 30
done
echo "[chain] sweep finished, starting follow-on"
exec bash scripts/run_undo_followup.sh
