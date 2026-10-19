#!/bin/bash
# Copy publishable (small, non-secret) study files into the repo study dir. Never copies pristine weights, COCO photographs, adapters, PNGs.
set -e
SRC=/data/bijaypandey/cuig_pilot/cpr_direct_agreement_controls_v1
DST=/data/bijaypandey/xxresearch_standalone/results/cpr_direct_agreement_controls_v1
mkdir -p "$DST"
rsync -a --delete-excluded --exclude='__pycache__' --exclude='*.pyc' \
  --include='/PREFREEZE_LOG.md' --include='/FREEZE_MANIFEST.json' --include='/TASK_CLOCK' --include='/runenv.sh' \
  --include='/code/***' --include='/scripts/***' --include='/configs/***' --include='/patches/***' --include='/records/***' \
  --include='/logs/***' --include='/artifacts/' --include='/artifacts/*.json' --include='/artifacts/ledger/***' \
  --include='/artifacts/ledger_snapshot_at_freeze/***' --include='/artifacts/gate/***' --include='/artifacts/qualification/***' \
  --include='/artifacts/analysis/***' --include='/artifacts/traj/***' --include='/artifacts/gen_records/***' \
  --include='/artifacts/scores/***' --include='/artifacts/plans/***' --include='/artifacts/decode_counts/***' \
  --include='/artifacts/orch/***' --include='/artifacts/repairs/***' --include='/artifacts/projection/***' --include='/artifacts/verify/***' \
  --include='/FINAL_REPORT.md' --include='/EVIDENCE_INDEX*' --include='/RELEASE_ASSETS.json' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*' "$SRC/" "$DST/"
