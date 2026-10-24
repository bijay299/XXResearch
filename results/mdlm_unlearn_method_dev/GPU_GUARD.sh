# Shared two-GPU guard. Source this, then call gpu_wait_for_slot before launching GPU work.
# WHY: on 2026-10-24 two chained scripts in different project dirs (t2i_unlearn/run_f57_gap.sh and
# mdlm_unlearn/run_mdu_l3_seeds_matched.sh) each waited only for the OTHER project's process names,
# cleared their guards in the same minute, and briefly put four GPUs in flight against the standing
# two-at-a-time cap. Project-local name guards cannot see each other; one lock can.
GPU_LOCK=/data/bijaypandey/.gpu_two_slot.lock
gpu_wait_for_slot() {   # $1 = holder name, for the log
  local me="${1:-unnamed}"
  while :; do
    if mkdir "$GPU_LOCK" 2>/dev/null; then
      echo "$me $$ $(date -Is)" > "$GPU_LOCK/holder"
      trap 'rm -rf "$GPU_LOCK"' EXIT INT TERM
      return 0
    fi
    # stale lock: holder pid gone
    local hp=$(awk '{print $2}' "$GPU_LOCK/holder" 2>/dev/null)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then rm -rf "$GPU_LOCK"; continue; fi
    sleep 60
  done
}
