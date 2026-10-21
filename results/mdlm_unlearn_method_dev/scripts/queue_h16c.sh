#!/bin/bash
# H16 arms. Registered in docs/PREREG_H16_conflict_projection.md.
# Frontier to beat: MDU lam_forget=1 at DISCLOSE 0.317 / DEGENERATE 0.075 / retain ASCR 0.524.
#
# TWO GUARD BUGS, BOTH FIXED HERE. Both cost real GPU time and are recorded so they are not
# repeated:
#
#  1. `ps -eo args | awk '/venv_llada\/bin\/python/ && /train_X\.py/'` SELF-MATCHED. The shell
#     that created a queue script via heredoc keeps the whole heredoc text in its own argv, so
#     the pattern matched that lingering process. busy() never returned 0 and one queue sat
#     idle for 62 minutes with both GPUs free.
#
#  2. `ps -eo comm=,args= | awk '$1 ~ /^python/'` NEVER MATCHED. PyTorch renames the process,
#     so comm is `pt_main_thread`, not `python`. busy() returned 0 even while a job was
#     running, which would have launched all seven arms onto one GPU at once.
#
# The guard below tests whether the FIRST FIELD of argv is exactly the venv python binary. A
# real job's argv[0] is that path; a bash wrapper's is /bin/bash, so it cannot self-match, and
# it does not depend on comm. This file is also written straight to disk rather than through a
# shell heredoc, so its text never enters any argv.
set -u
cd /data/bijaypandey/mdlm_unlearn

PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

busy () { ps -eo args= | awk -v py="$PY" '$1 == py && (/train_h16\.py/ || /train_atwu\.py/)' | wc -l; }

run () {   # mode rank lam_forget label [extra args...]
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  m=$1; k=$2; lf=$3; lab=$4; shift 4
  if [ -f "adapters/${lab}/report.json" ]; then
    echo "[queue_h16c] skip ${lab} (already done)"
    return 0
  fi
  ./scripts/launch_py.sh 3 D "$lab" 7200 train_h16.py \
    --mode "$m" --rank "$k" --lam-forget "$lf" --steps 300 "$@" \
    --out /data/bijaypandey/mdlm_unlearn/adapters/"$lab" \
    > "outputs/log_${lab}.txt" 2>&1
  sleep 5
}

# lam_forget=1 is the primary operating point: it is where MDU's best point sits, so Q5
# (strict dominance) is tested there.
run none   8  1 h16_none_f1          # in-script baseline, no projection
run proj   8  1 h16_proj_k8_f1       # the method
run anti   8  1 h16_anti_k8_f1       # Q4 attribution control
run proj   1  1 h16_proj_k1_f1       # rank sweep, degenerate case
run proj  32  1 h16_proj_k32_f1      # rank sweep, larger subspace
run pcgrad 8  1 h16_pcgrad_f1
run proj   8  2 h16_proj_k8_f2       # a second point for the curve
echo "[queue_h16c] done"
