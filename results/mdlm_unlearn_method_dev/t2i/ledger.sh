#!/bin/bash
# Resource ledger for the t2i_unlearn phase.
#
# WHY THIS EXISTS AND WHAT IT IS NOT: this phase ran with NO resource accounting of any kind. No
# GPU_SECONDS counter, no ATTEMPTS.tsv, no PROJECT_START -- unlike mdlm_unlearn, whose ledger.sh I
# had been quoting in status reports as though it covered this work. It did not. The GPU figure
# below is therefore RECONSTRUCTED after the fact from the wall-time fields the run records happen
# to carry, and it is a FLOOR, not a measurement: only 328 of 925 result JSONs record their own
# runtime, so the true figure is higher by an unknown amount. It is labelled as a floor wherever it
# is reported. Nothing is reset; this only adds forward-looking accounting.
W=/data/bijaypandey/t2i_unlearn
ST=$(cat $W/PROJECT_START 2>/dev/null || echo "2026-10-22T18:01:00")
EL=$(( $(date +%s) - $(date -d "$ST" +%s) ))
KB=$(du -sk $W | awk '{print $1}')
NJ=$(ls $W/outputs/*.json 2>/dev/null | wc -l)
read REC TOT < <(cd $W && venv_t2i/bin/python - <<'PY'
import json, glob
n = tot = 0
for f in glob.glob("outputs/*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    if not isinstance(d, dict): continue
    for k in ("minutes", "runtime_minutes", "elapsed_min", "wall_minutes"):
        if isinstance(d.get(k), (int, float)): tot += d[k]; n += 1; break
print(n, tot)
PY
)
awk -v el=$EL -v kb=$KB -v nj=$NJ -v rec=$REC -v tot=$TOT 'BEGIN{
printf "elapsed %.1f h | gpu >=%.2f device-h (FLOOR, see below) | result JSONs %d | storage %.1f GiB\n",
  el/3600, tot/60, nj, kb/1048576;
printf "  gpu floor reconstructed from %d of %d records that carry a runtime field; the other %d do not,\n",
  rec, nj, nj-rec;
printf "  so true consumption is higher by an unknown amount. No counter existed before 2026-10-24.\n"}'
echo "  storage authorized: 60 GiB — see docs/RESOURCE_NOTE.md if the figure above exceeds it"
