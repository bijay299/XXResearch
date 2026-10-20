#!/usr/bin/env bash
# failure-boundary sweep: forget-subject embedding interpolated toward a held-out partner subject
seed=$1
for a in 0.0 0.75 0.9 0.97; do
  /tmp/claude-0/env-llada/bin/python /home/user/XXResearch/research/llada_method_design_v1/tools/ledger_run.py --agent C --label "C toy sweep seed $seed alpha $a" --threads 1 -- /tmp/claude-0/env-llada/bin/python /home/user/XXResearch/research/llada_method_design_v1/agents/C_scripts/c_toy_mswe.py sweep --seed $seed --alpha $a --out /home/user/XXResearch/research/llada_method_design_v1/agents/C_scripts/out_sweep_s${seed}_a${a}.json
done
