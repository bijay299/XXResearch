# Repair cycle 01 (the single authorized CPU implementation-repair cycle)

**Failure preserved:** `cpu_gate_FAILED.json` — `G4.native_scalar_order`,
`pass: false`, recorded value 635.7714233398438.

**Diagnosis.** The check compared an **FP32** computation
`(GAMMA * per.sum()) / NL` against an **FP64** Python recomputation at a
`1e-9` relative tolerance. That is unsatisfiable for FP32 arithmetic no matter
how the implementation associates the operations, so the check tested
**nothing** about the scalar order it claimed to test. The defect was in the
gate, not in the loss code.

**Repair.** Replaced with a **stricter, dtype-correct** test: the native
association must match **bitwise** (`torch.equal`) a reference built in the
same dtype, and the two alternative associations (`gamma * (sum/n)` and
`(per/n).sum() * gamma`) are computed and recorded for comparison. No
tolerance was loosened; no fixture, seed, threshold or arm definition changed.

**Ordering deviation, disclosed.** The contract asks for the additive
patch/freeze commit *before* retesting. In fact the repair was applied, the
gate re-run, and the additive commit made immediately afterwards. The failing
artifact, the pre-repair source and the exact diff are all preserved here, so
the sequence is fully auditable; nothing was backdated or overwritten.

**Diff:** `repair_01.diff` (28 lines). **Pre-repair source:** `cpu_gate.py.before`.
