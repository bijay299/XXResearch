# Quarantined: the anchor 2x2 had collapsed

These runs are kept as a record, NOT as results.

`train_small_unlearn.py` masked the entire value span in the student state, so masking the
prompt on top produced an all-masked sequence -- identical to FAM. Both anchor modes fed the
model the same input, and the data shows it exactly: `mdu` and `mdu_fam` returned byte-identical
numbers (rank forget 0.1900, retain 0.8800, DEG 0.7950), as did `afc` and `emdu` (0.1925,
0.9500, 0.9025). The 2x2 that FINDING16 (AFC) rests on was never actually run.

Structural cause: on LLaDA-8B the answer span was 64 tokens, so a diffusion state PARTIALLY
reveals it and "mask the prompt, keep revealed answer tokens" differs from "mask everything".
Here values are 1-5 tokens and a 1-token value has nothing to reveal. Fixed by an explicit
prompt/answer split (prompt = author name, answer = rest of sentence, ~9 tokens).

A second defect: the retain term was CE on retain facts, which kept TRAINING them for 1500
steps, so every arm finished with retain recall ABOVE theta_ref (0.88-0.95 vs 0.6725) and retain
preservation was not measurable. Replaced by KL to the frozen reference, which is zero at
initialisation and penalises only deviation.

What these runs DO validly show (the FAM anchor was real in all of them): every KL arm drove
forget knowledge to near-oracle (rank ~0.19 vs oracle 0.11) while destroying the model
(DEGENERATE 0.795-0.9025, infill DISCLOSE 0.0000), and gradient ascent went silent instead
(ABSTAIN 0.9475, DEG 0.0100). Both reproduce on this backbone.
