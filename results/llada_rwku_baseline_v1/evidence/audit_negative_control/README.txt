Evidence for Amendment 02, Correction 1 (audit contradiction).

normal_run_b0001.log
  The NORMAL audit run's log (attempt b0001). Ends "AUDIT: ALL PASS" after
  "recomputed 200 per-item scores from stored predictions".
  mtime 2026-10-20 01:17:26.179

audit_negative.py
  The negative-control variant that OVERWROTE the audit JSON. It differs from
  scripts/audit_stage_a.py in exactly one line (line 68):
      chk("ten_entities_screened", len(pri["entities"]) == 999, len(pri["entities"]))
  i.e. it asserts ==999 instead of ==10, and it writes to the SAME hard-coded
  output path as the normal run.
  This produces exactly the published verdict: passed=false WITH detail=10,
  a combination the normal script cannot produce (there, detail==10 forces
  passed==true).

outputs/stage_a_audit.json (in the study root, preserved unchanged)
  mtime 2026-10-20 01:17:26.279 -- 0.100 s AFTER the normal run's log.
  Records all_passed=false, n_fail=1, failed=["ten_entities_screened"],
  with the other 12 checks still passed=true.

A matching normal-run JSON was NOT recovered and was NOT manufactured.
The audit was NOT re-run as part of Amendment 02.
