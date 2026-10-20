## 128-fact breaking-point probe — recalled-only stratification

Pool membership (ARC K=16): 15/64 forget facts in pool. theta0 recalled 51/64 forget facts.

| arm | forget ALL ↓ | **forget RECALLED ↓** | recalled IN-pool | recalled ABSENT | retain ↑ | degen (forget) |
|---|---|---|---|---|---|---|
| `theta0` | 0.781 | **0.980** | 0.950 (n=20) | 0.988 (n=82) | 0.773 | 6 |
| `uncached` | 0.047 | **0.049** | 0.000 (n=20) | 0.061 (n=82) | 1.000 | 0 |
| `arcK16` | 0.945 | **0.931** | 0.850 (n=20) | 0.951 (n=82) | 0.953 | 0 |
| `arcK512` | 0.172 | **0.216** | 0.350 (n=20) | 0.183 (n=82) | 1.000 | 0 |

**Pre-registered prediction** (ARC K=16 fails *specifically* on facts absent from its pool, predictably per fact): ABSENT 0.951 vs IN-pool 0.850, gap +0.101. Direction correct, magnitude far too small to carry the failure — **REFUTED as stated**: K=16 fails on in-pool facts too.
