"""
Stage 4b — results table, forgetting-vs-retention plot, paired examples, slide outline.

Automatic factual metrics only in the table/plot. Any coherence remark is marked
QUALITATIVE and kept separate.
"""
import json
import os
import sys

W = "/data/bijaypandey/llada_poc"
OUT = f"{W}/outputs"
sys.path.insert(0, f"{W}/scripts")

S = json.load(open(f"{OUT}/eval_summary.json"))
G = json.load(open(f"{OUT}/eval_generations.json"))
SPEC = json.load(open(f"{W}/data/frozen_spec.json"))
ORDER = ["memorized_reference", "sc_tf_step60", "sc_tf_step120",
         "fx_mix_step60", "fx_mix_step120"]
LABEL = {"memorized_reference": "memorized reference (no unlearning)",
         "sc_tf_step60": "CONTROL SC-TF @60", "sc_tf_step120": "CONTROL SC-TF @120",
         "fx_mix_step60": "CANDIDATE FX-Mix @60", "fx_mix_step120": "CANDIDATE FX-Mix @120"}
by = {s["checkpoint"]: s for s in S["summaries"]}
present = [c for c in ORDER if c in by]


def md_table():
    L = ["| checkpoint | forget disclosure ↓ | forget orig | forget para | retain acc ↑ | retain orig | retain para | retain SHARED-answer | retain independent | shared−indep | output failures |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in present:
        s = by[c]
        f, r = s["forget_disclosure"], s["retain_accuracy"]
        L.append(
            f"| {LABEL[c]} | **{f['all']['answer_present_rate']:.3f}** "
            f"| {f['original']['answer_present_rate']:.3f} | {f['paraphrase']['answer_present_rate']:.3f} "
            f"| **{r['all']['answer_present_rate']:.3f}** "
            f"| {r['original']['answer_present_rate']:.3f} | {r['paraphrase']['answer_present_rate']:.3f} "
            f"| {r['shared_answer_subset']['answer_present_rate']:.3f} "
            f"| {r['independent_subset']['answer_present_rate']:.3f} "
            f"| {s['collateral_gap_shared_minus_independent']:+.3f} "
            f"| {sum(s['output_failures'].values())} |")
    return "\n".join(L)


def svg_plot():
    """Forgetting (x: forget disclosure, lower=better) vs retention (y: retain accuracy)."""
    w, h, pad = 640, 440, 64
    def X(v): return pad + v * (w - 2 * pad)
    def Y(v): return h - pad - v * (h - 2 * pad)
    col = {"memorized_reference": "#555", "sc_tf": "#1f77b4", "fx_mix": "#d62728"}
    def cof(c): return col["memorized_reference"] if c == "memorized_reference" else (
        col["sc_tf"] if c.startswith("sc_tf") else col["fx_mix"])
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text x="{w/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">Forgetting vs retention — LLaDA PoC (24 synthetic facts, 48 questions/checkpoint)</text>']
    for i in range(6):
        v = i / 5
        p.append(f'<line x1="{X(v):.0f}" y1="{Y(0):.0f}" x2="{X(v):.0f}" y2="{Y(1):.0f}" stroke="#eee"/>')
        p.append(f'<line x1="{X(0):.0f}" y1="{Y(v):.0f}" x2="{X(1):.0f}" y2="{Y(v):.0f}" stroke="#eee"/>')
        p.append(f'<text x="{X(v):.0f}" y="{Y(0)+18:.0f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#444">{v:.1f}</text>')
        p.append(f'<text x="{X(0)-10:.0f}" y="{Y(v)+4:.0f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{v:.1f}</text>')
    p.append(f'<line x1="{X(0):.0f}" y1="{Y(0):.0f}" x2="{X(1):.0f}" y2="{Y(0):.0f}" stroke="#333"/>')
    p.append(f'<line x1="{X(0):.0f}" y1="{Y(0):.0f}" x2="{X(0):.0f}" y2="{Y(1):.0f}" stroke="#333"/>')
    p.append(f'<text x="{w/2}" y="{h-14}" text-anchor="middle" font-family="sans-serif" font-size="12">forget-fact disclosure (lower = more forgetting)</text>')
    p.append(f'<text x="16" y="{h/2}" text-anchor="middle" font-family="sans-serif" font-size="12" transform="rotate(-90 16 {h/2})">retain-fact accuracy (higher = better)</text>')
    p.append(f'<text x="{X(0.97):.0f}" y="{Y(0.97):.0f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#999">ideal → top-left</text>')
    for c in present:
        s = by[c]
        x = X(s["forget_disclosure"]["all"]["answer_present_rate"])
        y = Y(s["retain_accuracy"]["all"]["answer_present_rate"])
        mk = "6" if c == "memorized_reference" else "5"
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{mk}" fill="{cof(c)}" stroke="#fff" stroke-width="1.5"/>')
        p.append(f'<text x="{x+9:.1f}" y="{y+4:.1f}" font-family="sans-serif" font-size="10.5" fill="{cof(c)}">{LABEL[c]}</text>')
    for i, (lb, cc) in enumerate([("memorized reference", col["memorized_reference"]),
                                  ("CONTROL SC-TF", col["sc_tf"]),
                                  ("CANDIDATE FX-Mix", col["fx_mix"])]):
        yy = 40 + i * 16
        p.append(f'<circle cx="{w-168}" cy="{yy-4}" r="5" fill="{cc}"/>')
        p.append(f'<text x="{w-156}" y="{yy}" font-family="sans-serif" font-size="11" fill="#333">{lb}</text>')
    p.append("</svg>")
    return "\n".join(p)


def paired_examples(n=6):
    """Same question across checkpoints, raw answers side by side."""
    rows = []
    fids = [f"forget_{i:02d}" for i in range(4)] + ["retain_00", "retain_01"]
    for fid in fids[:n]:
        for kind in ("original",):
            blk = {"fact_id": fid, "kind": kind}
            for c in present:
                m = [r for r in G if r["checkpoint"] == c and r["fact_id"] == fid
                     and r["kind"] == kind]
                if m:
                    blk["question"] = m[0]["question"]
                    blk["reference"] = m[0]["reference_answer"]
                    blk["split"] = m[0]["split"]
                    blk[c] = {"generation": m[0]["generation"],
                              "answer_present": m[0]["answer_present"],
                              "failure": m[0]["output_failure"]}
            rows.append(blk)
    return rows


def main():
    tbl = md_table()
    open(f"{OUT}/results_table.md", "w").write(tbl + "\n")
    open(f"{OUT}/forgetting_vs_retention.svg", "w").write(svg_plot())
    pe = paired_examples()
    json.dump(pe, open(f"{OUT}/paired_examples.json", "w"), indent=2, ensure_ascii=False)

    lines = []
    for b in pe:
        lines.append(f"\n**{b['fact_id']}** ({b['split']}) — {b['question']}  ·  reference: `{b['reference']}`")
        for c in present:
            if c in b:
                mark = "✓" if b[c]["answer_present"] else "·"
                fl = f" [{b[c]['failure']}]" if b[c]["failure"] else ""
                lines.append(f"  - {mark} {LABEL[c]:34s} {b[c]['generation'][:88]!r}{fl}")
    open(f"{OUT}/paired_examples.md", "w").write("\n".join(lines) + "\n")

    print(tbl)
    print(f"\n[written] results_table.md, forgetting_vs_retention.svg, paired_examples.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
