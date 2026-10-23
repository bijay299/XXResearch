#!/bin/bash
# Pick N free GPUs by index, newest rule: ANY gpu is allowed, at most 2 of mine at a time.
# "Free" = no compute process owned by ANYONE and < 500 MiB resident, so another user's job is
# never displaced. Prints the chosen indices space-separated; exits 1 if not enough are free.
N=${1:-2}
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
free=()
while IFS=, read -r idx uuid mem; do
  idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' MiB ')
  echo "$busy" | grep -q "$uuid" && continue
  [ "${mem:-0}" -ge 500 ] && continue
  free+=("$idx")
done < <(nvidia-smi --query-gpu=index,gpu_uuid,memory.used --format=csv,noheader)
if [ "${#free[@]}" -lt "$N" ]; then
  echo "only ${#free[@]} free GPU(s): ${free[*]:-none}" >&2
  exit 1
fi
echo "${free[@]:0:$N}"
