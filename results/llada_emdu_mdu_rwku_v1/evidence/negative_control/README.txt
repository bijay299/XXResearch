attempt-2 (fixed) negative control: exit code 1, failing check tau_multiplies_anchor_logits
attempt-1 negative control was INVALID: the verifier hardcodes its import path, so the broken module was never loaded (exit 0). Disclosed; superseded by the path-redirected control above.

--- Amendment 01 (2026-10-20): execution evidence is PARTIAL ---
objectives_broken_tau.py  (published by Amendment 01, recovered from the workspace)
  The broken variant actually executed by the corrected control. Differs from
  scripts/objectives.py in exactly one line: "target_logits = target_logits / tau"
  instead of "target_logits = tau * target_logits". See broken_variant_diff.txt.

WHAT IS NOT EVIDENCED: the exit codes themselves.
  Both logs record only check verdicts. The "exit code 1" / "exit code 0" statements
  above were observed in the shell and are NOT captured in any published artifact.
  They are consistent with the verifier's sys.exit(0 if all_passed else 1) given the
  recorded verdicts, but are not independently evidenced. The controls were NOT
  re-run to capture them.
