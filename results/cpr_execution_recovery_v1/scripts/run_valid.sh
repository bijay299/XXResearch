#!/bin/bash
# final validation: 6 arms on fresh seeds/indices; two waves over 4 GPUs
cd /data/bijaypandey/cuig_pilot/cpr_execution_recovery_v1; . ./runenv.sh >/dev/null 2>&1
go() { $CPRD_PY scripts/sup.py run --job valid.train.$1 --bucket eng --gpus $2 --cores 2 --wall 1500 --rss 40 --sto 3 -- $CPRD_PY scripts/run_trajectory.py train --spec configs/specs/valid.$1.json > logs/sup_valid_$1.out 2>&1; }
go native_gem 0 & go cpr 1 & go centered_radial 2 & go truncated_anchor 3 &
wait %2 %3 %4
go anchor_only 1 & go axial_margin 2 &
wait
