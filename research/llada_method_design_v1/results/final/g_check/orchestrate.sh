#!/bin/sh
# G orchestration of VALIDATION_SPEC §7 steps 2-4 (label added: ledger_run.py requires --label)
R=/home/user/XXResearch/research/llada_method_design_v1
SP=/tmp/claude-0/-home-user-XXResearch/08815d77-39dc-54e1-8473-61e0d684a5c9/scratchpad/G
PY=/tmp/claude-0/env-llada/bin/python
cd $R
echo "substrates start $(date -u +%FT%TZ)" >> $SP/times.log
cd src
for W in 2 3 4; do $PY ../tools/ledger_run.py --agent G --label "G step2 base_train w$W" --threads 1 -- $PY -m mdmu.base_train --world-seed $W --init-seed $W --steps 3000 --out ../results/final/substrate/w${W}_i${W} > $SP/substrate_w$W.log 2>&1 & done
wait
cd ..
echo "substrates end $(date -u +%FT%TZ)" >> $SP/times.log
$PY tools/ledger_run.py --agent G --label "G step3 refs" --threads 1 -- $PY experiments/run_locked.py --refs > $SP/refs.log 2>&1
echo "refs end rc=$? $(date -u +%FT%TZ)" >> $SP/times.log
for i in 0 1 2 3; do $PY tools/ledger_run.py --agent G --label "G step4 shard $i/4" --threads 1 -- $PY experiments/run_locked.py --shard $i/4 > $SP/shard$i.log 2>&1 & done
wait
echo "shards end $(date -u +%FT%TZ)" >> $SP/times.log
