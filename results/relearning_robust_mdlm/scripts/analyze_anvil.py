"""
Final analysis: relearning curves, the pre-registered verdict, and the figure.

The outcome is the DOWNSTREAM relearning curve. The gradient norm is only the mechanism
check -- Song et al. ("Measure, Don't Optimize") show that optimising an audit proxy can
teach a model to hide from the audit while increasing post-attack recovery, so the proxy
is never reported as the result.
"""
import glob
import json
import os

A = "/data/bijaypandey/anvil"
OUT = f"{A}/outputs"

arms = {}
for p in sorted(glob.glob(f"{OUT}/relearn_*.json")):
    if "_gens_" in p:
        continue
    d = json.load(open(p))
    arms[os.path.basename(p)[len("relearn_"):-len(".json")]] = d

LABEL = {"base": "never memorized (FLOOR)", "theta0": "memorized reference (CEILING)",
         "unlearned": "unlearned control (theta_u)",
         "flat150": "flattened, 150 steps", "flat300": "flattened, 300 steps"}
ORDER = ["base", "unlearned", "flat150", "flat300", "theta0"]
KS = [0, 2, 5, 10, 20]


def curve(d, key="heldout_forget", field="disclosure"):
    return {e["k"]: e[key][field] for e in d["curve"]}


L = ["## Relearning curves — held-out forget disclosure (43 facts the adversary never saw)\n",
     "Adversary: 8 leaked forget facts, plain masked-diffusion fine-tuning, identical for "
     "every arm. Lower is better.\n",
     "| arm | " + " | ".join(f"k={k}" for k in KS) + " | adv lr |",
     "|---" * (len(KS) + 2) + "|"]
for a in ORDER:
    if a not in arms:
        continue
    c = curve(arms[a])
    lr = arms[a].get("adversary_lr", arms[a]["hparams"].get("lr"))
    L.append(f"| {LABEL.get(a, a)} | "
             + " | ".join(f"**{c[k]:.3f}**" if k == 10 else f"{c[k]:.3f}"
                          for k in KS if k in c) + f" | {lr:g} |")

# retain + degeneracy, so a win by collapse cannot hide
L.append("\n### Retain accuracy (16 probes) and degenerate outputs on the forget probes\n")
L.append("| arm | " + " | ".join(f"retain k={k}" for k in KS) + " | degen k=10 |")
L.append("|---" * (len(KS) + 2) + "|")
for a in ORDER:
    if a not in arms:
        continue
    r = curve(arms[a], "retain")
    dg = {e["k"]: e["heldout_forget"]["degenerate"] for e in arms[a]["curve"]}
    L.append(f"| {LABEL.get(a, a)} | "
             + " | ".join(f"{r[k]:.3f}" for k in KS if k in r)
             + f" | {dg.get(10, '—')} |")

# ── pre-registered verdict ─────────────────────────────────────────────────
L.append("\n## Pre-registered verdict\n")
if "unlearned" in arms:
    ctl = curve(arms["unlearned"])
    ctl_r = curve(arms["unlearned"], "retain")
    verdicts = {}
    for a in ("flat150", "flat300"):
        if a not in arms:
            continue
        c, r = curve(arms[a]), curve(arms[a], "retain")
        d10 = c.get(10, float("nan")) - ctl.get(10, float("nan"))
        d0 = c.get(0, float("nan")) - ctl.get(0, float("nan"))
        dr = r.get(0, float("nan")) - ctl_r.get(0, float("nan"))
        ok = (d10 <= -0.20) and (d0 <= 0.05) and (dr >= -0.05)
        verdicts[a] = {"delta_k10": round(d10, 4), "delta_singleshot_k0": round(d0, 4),
                       "delta_retain_k0": round(dr, 4),
                       "verdict": "SUPPORTED" if ok else "NOT SUPPORTED"}
        L.append(f"**{LABEL[a]}** — Δ(k=10) = {d10:+.3f} (needs ≤ −0.20); "
                 f"Δ single-shot = {d0:+.3f} (needs ≤ +0.05); "
                 f"Δ retain = {dr:+.3f} (needs ≥ −0.05) → **{verdicts[a]['verdict']}**")
else:
    verdicts = {}
    L.append("control curve missing")

# ── mechanism check, reported separately and labelled as a proxy ───────────
mech = {}
for p in sorted(glob.glob(f"{A}/adapters/*/report.json")):
    d = json.load(open(p))
    if d.get("arm") not in ("anvil", "anchor_only"):
        continue
    m = d.get("measured") or {}
    if not m:
        continue
    mech[os.path.basename(os.path.dirname(p))] = {
        "arm": d["arm"], "lam_G": d["hparams"]["lam_G"], "lr": d["hparams"]["lr"],
        "steps": d["hparams"]["steps"], "optimizer": d["hparams"].get("optimizer"),
        "L_mem_at_theta_u": d.get("L_mem_at_theta_u"),
        "probe_grad_norm_first": m.get("probe_grad_norm_first"),
        "probe_grad_norm_last": m.get("probe_grad_norm_last"),
        "probe_L_mem_first": m.get("probe_L_mem_first"),
        "probe_L_mem_last": m.get("probe_L_mem_last"),
        "s_per_step": m.get("s_per_step_steady"),
        "fwd_per_step": m.get("fwd_per_step"), "bwd_per_step": m.get("bwd_per_step"),
        "guard_fired": "guard_fired" in d}
L.append("\n## Mechanism check (a PROXY, not the result)\n")
L.append("| run | arm | lam_G | lr | steps | opt | ||gradL_mem|| first → last | L_mem first → last | s/step | guard |")
L.append("|---" * 10 + "|")
for k, v in sorted(mech.items()):
    L.append(f"| `{k}` | {v['arm']} | {v['lam_G']:g} | {v['lr']:g} | {v['steps']} "
             f"| {v['optimizer']} | {v['probe_grad_norm_first']} → "
             f"**{v['probe_grad_norm_last']}** | {v['probe_L_mem_first']} → "
             f"{v['probe_L_mem_last']} | {v['s_per_step']} "
             f"| {'FIRED' if v['guard_fired'] else 'ok'} |")

json.dump({"arms": {a: {"heldout": curve(arms[a]), "retain": curve(arms[a], "retain"),
                        "adversary_lr": arms[a].get("adversary_lr"),
                        "stack": arms[a].get("stack")} for a in arms},
           "verdicts": verdicts, "mechanism": mech},
          open(f"{OUT}/anvil_analysis.json", "w"), indent=2)
open(f"{OUT}/RELEARN_TABLE.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ── figure ─────────────────────────────────────────────────────────────────
COL = {"base": "#888888", "unlearned": "#d62728", "flat150": "#1f77b4",
       "flat300": "#2ca02c", "theta0": "#9467bd"}
W, H, Lm, Rm, T, B = 740, 430, 78, 180, 48, 56
pw, ph = W - Lm - Rm, H - T - B
kmax = max(KS)
fx = lambda k: Lm + pw * (k / kmax)
fy = lambda v: T + ph * (1 - v)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'font-family="Helvetica,Arial,sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{Lm}" y="24" font-size="14" font-weight="bold">Relearning attack: '
     f'disclosure on 43 forget facts the adversary never saw</text>']
for g in (0, .25, .5, .75, 1.0):
    s.append(f'<line x1="{Lm}" y1="{fy(g):.1f}" x2="{Lm+pw}" y2="{fy(g):.1f}" stroke="#e8e8e8"/>')
    s.append(f'<text x="{Lm-8}" y="{fy(g)+4:.1f}" font-size="10.5" fill="#555" '
             f'text-anchor="end">{g:.2f}</text>')
for k in KS:
    s.append(f'<text x="{fx(k):.1f}" y="{T+ph+18:.1f}" font-size="10.5" fill="#333" '
             f'text-anchor="middle">{k}</text>')
for a in ORDER:
    if a not in arms:
        continue
    c = curve(arms[a])
    pts = [(fx(k), fy(c[k])) for k in KS if k in c]
    s.append(f'<polyline fill="none" stroke="{COL[a]}" stroke-width="2.4" '
             f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
    for x, y in pts:
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{COL[a]}"/>')
    s.append(f'<text x="{Lm+pw+8:.1f}" y="{pts[-1][1]+4:.1f}" font-size="10.5" '
             f'fill="{COL[a]}">{LABEL.get(a, a)}</text>')
s.append(f'<line x1="{Lm}" y1="{T+ph}" x2="{Lm+pw}" y2="{T+ph}" stroke="#333"/>')
s.append(f'<line x1="{Lm}" y1="{T}" x2="{Lm}" y2="{T+ph}" stroke="#333"/>')
s.append(f'<text x="{Lm+pw/2:.1f}" y="{H-20}" font-size="11.5" fill="#333" '
         f'text-anchor="middle">adversary fine-tuning steps on 8 leaked facts</text>')
s.append(f'<text x="18" y="{T+ph/2:.1f}" font-size="11.5" fill="#333" '
         f'transform="rotate(-90 18,{T+ph/2:.1f})" text-anchor="middle">held-out forget disclosure</text>')
s.append(f'<text x="{Lm}" y="{H-5}" font-size="9.5" fill="#777">Synthetic facts, 1 seed, '
         f'43 held-out probes. Mechanism is Fan et al. ICML 2025 (GP/CR); no novelty claimed.</text>')
s.append("</svg>")
open(f"{OUT}/relearning_curves.svg", "w").write("\n".join(s))
print(f"\n[written] RELEARN_TABLE.md, anvil_analysis.json, relearning_curves.svg")
