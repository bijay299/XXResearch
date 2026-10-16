#!/usr/bin/env python3
import json
from pathlib import Path

joined = json.loads(Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/UNBLINDED_JOINED_TABLE.json").read_text())

def load_assistant(path):
    rows = json.loads(Path(path).read_text())
    return {r["case_id"]: r for r in rows if 60 <= r["case_id"] < 80}

stage1h = load_assistant("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/stage1_history_intervention_s2/CASE_LEVEL_COMPLETION_RECORD.json")
bothstages = load_assistant("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_both_stages_s2/CASE_LEVEL_COMPLETION_RECORD.json")

# Apply the published C31 correction layer: S2_STAGE1_HISTORY case 63 was
# corrected from degenerate=true to degenerate=false (CORRECTED_LABELS.json,
# stage1_history_intervention_s2/closeout/). The raw completion record above
# still holds the PRE-correction label; the current assistant position is
# the corrected one (2/20: cases 77, 79 only), and that is what must be
# compared against here.
corrected = json.loads(Path(
    "/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_both_stages_s2/closeout/CORRECTED_LABELS.json"
).read_text())
for c in corrected["corrections"]:
    if c["endpoint"] == "S2_STAGE1_HISTORY" and c["case_id"] in stage1h:
        stage1h[c["case_id"]]["degenerate"] = c["corrected_label"]["degenerate"]

print(f"{'case':>4} | {'assist_S1H':>10} | {'human_S1H':>9} | {'assist_BOTH':>11} | {'human_BOTH':>10} | agree?")
rows_out = []
agree_s1h = agree_both = 0
for r in joined:
    cid = r["case_id"]
    a_s1h = "YES" if cid in stage1h and stage1h[cid]["degenerate"] else "no"
    h_s1h = "YES" if r["S2_STAGE1_HISTORY_degeneration"] == "yes" else "no"
    a_both = "YES" if cid in bothstages and bothstages[cid]["degenerate"] else "no"
    h_both = "YES" if r["S2_HISTORY_BOTH_STAGES_degeneration"] == "yes" else "no"
    ok1 = a_s1h == h_s1h
    ok2 = a_both == h_both
    agree_s1h += ok1
    agree_both += ok2
    flag = "" if (ok1 and ok2) else "  <-- DISAGREE"
    print(f"{cid:>4} | {a_s1h:>10} | {h_s1h:>9} | {a_both:>11} | {h_both:>10} |{flag}")
    rows_out.append({"case_id": cid, "assistant_S2_STAGE1_HISTORY_degenerate": a_s1h=="YES",
                      "human_S2_STAGE1_HISTORY_degenerate": h_s1h=="YES",
                      "assistant_S2_HISTORY_BOTH_STAGES_degenerate": a_both=="YES",
                      "human_S2_HISTORY_BOTH_STAGES_degenerate": h_both=="YES"})

print()
print(f"Agreement on S2_STAGE1_HISTORY degeneration label: {agree_s1h}/20")
print(f"Agreement on S2_HISTORY_BOTH_STAGES degeneration label: {agree_both}/20")

Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/CASE_LEVEL_AGREEMENT_TABLE.json").write_text(
    json.dumps({"rows": rows_out, "agreement_S2_STAGE1_HISTORY": f"{agree_s1h}/20",
                "agreement_S2_HISTORY_BOTH_STAGES": f"{agree_both}/20"}, indent=1))
