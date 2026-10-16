#!/usr/bin/env python3
"""Injects the exact shared generation prompt for each R-slot into
index.html's PAIRS array, replacing __PROMPT_Rnn__ placeholders. Reads
the PRIVATE baseline-association manifest only for the neutral_id ->
prompt text (prompt text is intentionally shown) -- never writes
endpoint names, case IDs, or seeds into the HTML.
"""
import json
import re
from pathlib import Path

PRIVATE_ASSOC = Path("/data/bijaypandey/cuig_pilot/review_private/context_quality_review_s2_baseline/BASELINE_ASSOCIATION_PRIVATE.json")
INDEX_HTML = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/context_quality_review_s2_baseline/reviewer_package/index.html")


def js_string_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main():
    assoc = json.loads(PRIVATE_ASSOC.read_text())
    prompt_by_rid = {a["neutral_id"]: a["shared_prompt"] for a in assoc["associations"]}
    assert len(prompt_by_rid) == 20

    html = INDEX_HTML.read_text()
    for rid, prompt in prompt_by_rid.items():
        placeholder = f"__PROMPT_{rid}__"
        assert placeholder in html, f"placeholder {placeholder} not found"
        html = html.replace(placeholder, js_string_escape(prompt))

    assert "__PROMPT_" not in html, "not all placeholders were replaced"

    forbidden = ["S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES", "stage1_history", "both_stages", "EnglishSpringer_parent"]
    leaked = [f for f in forbidden if f.lower() in html.lower()]
    assert not leaked, f"forbidden identifying terms leaked into index.html: {leaked}"

    INDEX_HTML.write_text(html)
    print(f"Injected {len(prompt_by_rid)} prompts into {INDEX_HTML}. No identifying terms found.")


if __name__ == "__main__":
    main()
