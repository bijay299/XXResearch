import json

with open("/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/eval/recognition_summary.json") as f:
    S = json.load(f)

M0_SEM = S["M0"]["semantic_correct_per_class"]
M0_STRICT = S["M0"]["strict_correct_per_class"]

TARGETS = {"A": "cassette player", "B": "chain saw"}
METHOD_SINGLE = {
    ("DP", "A"): "DP_A", ("DP", "B"): "DP_B",
    ("UCE", "A"): "UCE_A", ("UCE", "B"): "UCE_B",
}

print("=== Gate 1: M0 semantic recognition >= 80/100 per target ===")
gate1 = {}
for sym, cls in TARGETS.items():
    v = M0_SEM[cls]
    passed = v >= 80
    gate1[sym] = passed
    print(f"  {sym} ({cls}): M0 semantic = {v}/100 -> {'PASS' if passed else 'FAIL'}")

print("\n=== Gate 2: each method's single edit reduces its own target's semantic recognition by >=75% relative to M0 ===")
gate2 = {}
for method in ["DP", "UCE"]:
    for sym, cls in TARGETS.items():
        ep = METHOD_SINGLE[(method, sym)]
        m0v = M0_SEM[cls]
        ev = S[ep]["semantic_correct_per_class"][cls]
        reduction = (m0v - ev) / m0v if m0v > 0 else float("nan")
        passed = reduction >= 0.75
        gate2[(method, sym)] = passed
        print(f"  {method}-{sym} ({cls}): M0={m0v} -> {ep}={ev}  relative reduction={reduction:.1%} -> {'PASS' if passed else 'FAIL'}")

print("\n=== Combined per-single verdict (gate1 for that target AND gate2 for that method+target) ===")
single_pass = {}
for method in ["DP", "UCE"]:
    for sym in TARGETS:
        p = gate1[sym] and gate2[(method, sym)]
        single_pass[(method, sym)] = p
        print(f"  {method}-{sym}: gate1={gate1[sym]} gate2={gate2[(method,sym)]} -> overall {'PASS' if p else 'FAIL'}")

print("\n=== Method qualification (both singles must pass) ===")
qualified = []
for method in ["DP", "UCE"]:
    both = single_pass[(method, "A")] and single_pass[(method, "B")]
    print(f"  {method}: A={single_pass[(method,'A')]} B={single_pass[(method,'B')]} -> {'QUALIFIES' if both else 'DOES NOT QUALIFY'}")
    if both:
        qualified.append(method)

print(f"\nQualified methods for sequential extension: {qualified if qualified else 'NONE'}")

print("\n=== Strict-label numbers for the same cases (reported separately, not used for gating) ===")
for sym, cls in TARGETS.items():
    print(f"  M0 {cls}: strict={M0_STRICT[cls]}/100")
for method in ["DP", "UCE"]:
    for sym, cls in TARGETS.items():
        ep = METHOD_SINGLE[(method, sym)]
        print(f"  {ep} {cls}: strict={S[ep]['strict_correct_per_class'][cls]}/100")

print("\n=== Retention: 9 preserved classes, semantic + strict, M0 vs each single edit ===")
ALL_CLASSES = list(M0_SEM.keys())
for method in ["DP", "UCE"]:
    for sym, cls in TARGETS.items():
        ep = METHOD_SINGLE[(method, sym)]
        preserved = [c for c in ALL_CLASSES if c != cls]
        print(f"\n{ep} (target erased: {cls}); retained classes: {preserved}")
        sem_m0_vals, sem_ep_vals = [], []
        for c in preserved:
            m0v = M0_SEM[c]
            epv = S[ep]["semantic_correct_per_class"][c]
            sem_m0_vals.append(m0v)
            sem_ep_vals.append(epv)
            print(f"    {c:20s} M0={m0v:3d}/100  {ep}={epv:3d}/100  delta={epv-m0v:+d}")
        print(f"    AGGREGATE mean M0={sum(sem_m0_vals)/len(sem_m0_vals):.1f}  mean {ep}={sum(sem_ep_vals)/len(sem_ep_vals):.1f}")
