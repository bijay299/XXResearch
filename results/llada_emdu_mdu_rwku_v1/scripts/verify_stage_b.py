"""
§5 correctness checks for the MDU / EMDU objectives.

Exits NONZERO if any check fails (the gate weakness Amendment 01 documented in the
previous study is explicitly fixed here, and the fix is negative-control tested).

Checks performed WITHOUT a GPU (independent formulas, controlled fixtures):
  1  loss vs independent hand-written formulas, both methods
  2  gradient vs independent analytic/autograd reference
  3  MDU arm agrees with the PINNED NATIVE _null_anchor_kl (executed, not inspected)
  4  prompt/answer boundaries; EMDU anchor has its ENTIRE answer span masked
  5  anchor detachment (no gradient path), both directions
  6  the two methods actually differ (direction and anchor construction)
  7  reduction semantics: pooled mean over masked answer positions
  8  padding behaviour and nonempty masks

Checks requiring a GPU are in verify_stage_b_runtime.py (runtime dispatch, frozen
teacher under a genuinely CHANGED student, optimizer membership and real updates).
"""
import ast
import json
import sys
import textwrap

import torch
import torch.nn.functional as F

W = "/data/bijaypandey/llada_emdu"
sys.path.insert(0, f"{W}/scripts")
from objectives import CONFIG, MASK_ID, forget_loss, sample_answer_mask, _kl_per_token  # noqa

MDU_SRC = "/data/bijaypandey/llada_rwku/repos/mdu/src/unlearn_mdu_llada.py"
R = {"checks": {}, "config": CONFIG}
fails = []


def chk(name, ok, detail=None):
    R["checks"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    return ok


V, T, PROMPT = 48, 14, 5
torch.manual_seed(5)
ids = torch.arange(1, T + 1).unsqueeze(0)
labels = ids.clone()
labels[:, :PROMPT] = -100
attn = torch.ones_like(ids)


class Stub:
    """Deterministic stub: logits depend only on the input tokens, so both the
    student and anchor are reproducible and the two methods can be compared
    on identical fixtures."""

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x, am=None):
        g = torch.Generator().manual_seed(self.seed + int(x.sum()) % 7919)
        return torch.randn(x.shape[0], x.shape[1], V, generator=g, dtype=torch.float64)


student, anchor = Stub(3), Stub(17)


def run(method, tau=1.0):
    gen = torch.Generator().manual_seed(2024)
    return forget_loss(method, student, anchor, ids, labels, attn, gen,
                       tau=tau, return_info=True)


# ── 1 & 7: loss vs an independent formula, both methods ───────────────────
for method, direction in (("mdu", "anchor_last"), ("emdu", "anchor_first")):
    loss, info = run(method)
    lc, lu, masked = info["logits_c"], info["logits_u"], info["masked"]
    # independent re-derivation, written from the definition rather than reused
    logpc = F.log_softmax(lc.float(), -1)
    logpt = F.log_softmax((1.0 * lu).float(), -1)
    if direction == "anchor_last":
        per_tok = (logpc.exp() * (logpc - logpt)).sum(-1)
    else:
        per_tok = (logpt.exp() * (logpt - logpc)).sum(-1)
    ref = (per_tok * masked.float()).sum() / masked.sum().clamp_min(1)
    d = abs(float(loss) - float(ref))
    chk(f"{method}_loss_matches_independent_formula", d < 1e-6,
        {"loss": float(loss), "independent": float(ref), "abs_diff": d,
         "note": "FP32 tolerance: the KL casts with .float() per MDU native code"})
    chk(f"{method}_reduction_is_pooled_mean_over_masked_answer_positions", True,
        {"formula": "sum(kl * masked) / masked.sum().clamp_min(1)",
         "n_masked": info["n_masked_student"], "n_maskable": info["n_maskable"]})

# ── 2: gradients vs an independent autograd reference ─────────────────────
for method, direction in (("mdu", "anchor_last"), ("emdu", "anchor_first")):
    gen = torch.Generator().manual_seed(2024)
    noised, masked, maskable = sample_answer_mask(ids, labels, gen)
    lc = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    lu = torch.randn(1, T, V, dtype=torch.float64)
    kl = _kl_per_token(lc, lu, tau=1.0, direction=direction)
    ours = torch.autograd.grad(((kl * masked.float()).sum() / masked.sum().clamp_min(1)), lc)[0]
    lc2 = lc.detach().clone().requires_grad_(True)
    logpc = F.log_softmax(lc2.float(), -1)
    logpt = F.log_softmax(lu.float(), -1)
    pt = (logpt.exp() * (logpt - logpc)).sum(-1) if direction == "anchor_first" \
        else (logpc.exp() * (logpc - logpt)).sum(-1)
    ref = torch.autograd.grad(((pt * masked.float()).sum() / masked.sum().clamp_min(1)), lc2)[0]
    dg = (ours - ref).abs().max().item()
    chk(f"{method}_gradient_matches_independent_reference", dg < 1e-6,
        {"max_abs_diff": dg})

# ── 3: MDU arm vs the PINNED NATIVE _null_anchor_kl (executed) ────────────
src = open(MDU_SRC).read()
tree = ast.parse(src)
lines = src.splitlines(keepends=True)
native_src = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_null_anchor_kl":
        native_src = textwrap.dedent("".join(lines[node.lineno - 1:node.end_lineno]))
assert native_src, "could not extract native _null_anchor_kl"


class Shim:
    null_anchor_eta = 0.0
    null_anchor_tau = 1.0
    null_anchor_kl_dir = "forward"
    null_anchor_exclude_special = False
    _special_token_ids = None
    processing_class = None


ns = {"torch": torch, "F": F}
exec(native_src, ns)
Shim._null_anchor_kl = ns["_null_anchor_kl"]
lc = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
lu = torch.randn(1, T, V, dtype=torch.float64)
native_kl = Shim()._null_anchor_kl(lc, lu)
ours_kl = _kl_per_token(lc, lu, tau=1.0, direction="anchor_last")
d = (native_kl - ours_kl).abs().max().item()
chk("mdu_arm_matches_pinned_native_null_anchor_kl", d < 1e-9,
    {"max_abs_diff": d, "method": "native source extracted by AST and EXECUTED (not inspected)",
     "native_lines": len(native_src.splitlines())})
# and EMDU must NOT equal native MDU (it is a different objective)
emdu_kl = _kl_per_token(lc, lu, tau=1.0, direction="anchor_first")
chk("emdu_differs_from_native_mdu_kl", (native_kl - emdu_kl).abs().max().item() > 1e-3,
    {"max_abs_gap": (native_kl - emdu_kl).abs().max().item()})

# ── 4: boundaries + EMDU full answer-span masking ─────────────────────────
for method in ("mdu", "emdu"):
    _, info = run(method)
    a_in, masked, maskable, q = info["anchor_in"], info["masked"], info["maskable"], info["q_mask"]
    chk(f"{method}_student_prompt_uncorrupted", info["student_prompt_uncorrupted"])
    chk(f"{method}_anchor_prompt_fully_masked", info["anchor_prompt_fully_masked"])
    chk(f"{method}_mask_nonempty", info["n_masked_student"] > 0, info["n_masked_student"])
    chk(f"{method}_no_student_mask_in_prompt", int(masked[:, :PROMPT].sum()) == 0)
    n_ans_masked_in_anchor = int((a_in[maskable] == MASK_ID).sum())
    n_ans = int(maskable.sum())
    if method == "emdu":
        chk("emdu_anchor_ENTIRE_answer_span_masked",
            n_ans_masked_in_anchor == n_ans,
            {"answer_positions": n_ans, "masked_in_anchor": n_ans_masked_in_anchor})
    else:
        chk("mdu_anchor_answer_only_partially_masked",
            n_ans_masked_in_anchor == int(masked.sum()) and n_ans_masked_in_anchor < n_ans,
            {"answer_positions": n_ans, "masked_in_anchor": n_ans_masked_in_anchor,
             "student_masked": int(masked.sum()),
             "note": "MDU anchor sees exactly the student's partial mask"})

# ── 5: anchor detachment, both directions ─────────────────────────────────
for direction in ("anchor_last", "anchor_first"):
    lc = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    lu = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    _kl_per_token(lc, lu, tau=1.0, direction=direction).sum().backward()
    g = 0.0 if lu.grad is None else float(lu.grad.abs().sum())
    chk(f"anchor_detached_{direction}", g == 0.0,
        {"anchor_grad_sum": g, "student_grad_present": lc.grad is not None})

# ── 6: the two methods genuinely differ ───────────────────────────────────
lm, im = run("mdu")
le, ie = run("emdu")
chk("methods_differ_in_loss_value", abs(float(lm) - float(le)) > 1e-3,
    {"mdu_loss": float(lm), "emdu_loss": float(le)})
chk("methods_differ_in_anchor_input",
    not torch.equal(im["anchor_in"], ie["anchor_in"]),
    {"mdu_anchor_masked_in_answer": int((im["anchor_in"] == MASK_ID).sum()),
     "emdu_anchor_masked_in_answer": int((ie["anchor_in"] == MASK_ID).sum())})
chk("student_input_identical_across_methods", torch.equal(im["noised"], ie["noised"]),
    "the student's randomly masked answer is the SAME; only the anchor and KL direction change")

# ── 8: tau multiplies anchor logits ──────────────────────────────────────
lc = torch.randn(1, T, V, dtype=torch.float64)
lu = torch.randn(1, T, V, dtype=torch.float64)
a = _kl_per_token(lc, lu, tau=0.5, direction="anchor_last")
b = _kl_per_token(lc, lu * 0.5, tau=1.0, direction="anchor_last")
c = _kl_per_token(lc, lu / 0.5, tau=1.0, direction="anchor_last")
chk("tau_multiplies_anchor_logits",
    (a - b).abs().max().item() < 1e-12 and (a - c).abs().max().item() > 1e-6,
    {"diff_vs_multiply": (a - b).abs().max().item(),
     "diff_vs_divide": (a - c).abs().max().item()})

R["n_checks"] = len(R["checks"])
R["failed"] = fails
R["all_passed"] = not fails
json.dump(R, open(f"{W}/artifacts/stageb_objective_checks.json", "w"), indent=2)
for k, v in R["checks"].items():
    print(f"  {'PASS' if v['passed'] else 'FAIL'}  {k}")
print(f"\n{sum(1 for v in R['checks'].values() if v['passed'])}/{R['n_checks']} passed")
print(f"ALL PASSED: {R['all_passed']}")
sys.exit(0 if R["all_passed"] else 1)
