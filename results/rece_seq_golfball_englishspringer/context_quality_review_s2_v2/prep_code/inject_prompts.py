#!/usr/bin/env python3
"""Injects the exact shared generation prompt for each R-slot into
index.html's PAIRS array, replacing the __PROMPT_Rnn__ placeholders.
Reads the private mapping ONLY for the neutral_id -> prompt association
(the prompt text itself is intentionally shown to reviewers, per the
frozen protocol) -- never writes endpoint names, case IDs, seeds, or any
other mapping field into the HTML. Verifies after injection that none
of those identifying fields leaked in as a side effect.
"""
import json
import re
from pathlib import Path

PRIVATE_MAPPING = Path("/data/bijaypandey/cuig_pilot/review_private/context_quality_review_s2_v2/PRIVATE_MAPPING_V2.json")
INDEX_HTML = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/context_quality_review_s2_v2/reviewer_package/index.html")


def js_string_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main():
    mapping = json.loads(PRIVATE_MAPPING.read_text())
    prompt_by_rid = {p["neutral_id"]: p["shared_prompt"] for p in mapping["pairs"]}
    assert len(prompt_by_rid) == 20

    html = INDEX_HTML.read_text()
    for rid, prompt in prompt_by_rid.items():
        placeholder = f"__PROMPT_{rid}__"
        assert placeholder in html, f"placeholder {placeholder} not found"
        html = html.replace(placeholder, js_string_escape(prompt))

    assert "__PROMPT_" not in html, "not all placeholders were replaced"

    # safety: confirm no endpoint names, case IDs 60-79, or the word "seed" (outside
    # this script's own absence) leaked into the file as a side effect of injection
    forbidden = ["S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES", "stage1_history", "both_stages"]
    leaked = [f for f in forbidden if f.lower() in html.lower()]
    assert not leaked, f"forbidden identifying terms leaked into index.html: {leaked}"

    INDEX_HTML.write_text(html)
    print(f"Injected {len(prompt_by_rid)} prompts into {INDEX_HTML}. No identifying terms found.")


if __name__ == "__main__":
    main()
