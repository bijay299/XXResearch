Quarantined, not deleted. These rows were produced while two concurrent jobs of mine both held the
label `mdu_l3_s11_n12`: one training into `adapters/mdu_l3_s11_n12`, the other evaluating that same
directory. The evaluation therefore read a checkpoint that was being rewritten underneath it, so the
numbers are not attributable to any definite model state and must not be used in the n=12 aggregate.
The clean replacement is `mdu_l3_s11_n12b`, built alone.
