#!/bin/bash
cd /data/bijaypandey/cuig_pilot/cpr_execution_recovery_v1; . ./runenv.sh >/dev/null 2>&1
for ph in dev_m0 dev_train dev_select; do $CPRD_PY scripts/orchestrate.py $ph || { echo "PHASE_FAILED $ph"; exit 1; }; done
echo DEV_PIPELINE_DONE
