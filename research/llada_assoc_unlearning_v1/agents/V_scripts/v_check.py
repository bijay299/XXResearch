"""V's independent implementation checks of llada/tdssd.py (frozen spec agents/O_FINAL.md §2).
TINY-MODEL CHECKS ONLY (own 1-layer bidirectional model, own LoRA, own word tokenizer) -- NOT LLaDA verification.
Run: python agents/V_scripts/v_check.py   (prints PASS/FAIL lines)
"""
import contextlib, os, sys, types, math, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "llada"))
import tdssd as T  # noqa

RES = []
def check(name, ok, info=""):
    RES.append((name, bool(ok))); print(("PASS " if ok else "FAIL ") + name + ("  | " + str(info) if info else ""))

# ---------------------------------------------------------------- own tokenizer (word level; specials 0..4)
class Tok:
    def __init__(self, words):
        self.itos = ["<pad>", "<m>", "<eos>", "<u>", "<a>"] + sorted(set(words)); self.stoi = {w: i for i, w in enumerate(self.itos)}
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.stoi[w] for w in text.replace("?", " ?").replace(".", " .").split()]}
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "<u> " + msgs[0]["content"] + " <a>"
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.itos[i] for i in ids)
MASK, EOS = 1, 2
words = "Where does live ? lives in Paris Rome . Ann Zed Bob Kay Lund What pet own cat".split() + T.FIRST_NAMES + T.LAST_NAMES
tok = Tok(words); V = len(tok.itos)

# ---------------------------------------------------------------- own tiny bidirectional model + own LoRA
class LoRA(nn.Module):
    def __init__(self, base, r=2):
        super().__init__(); self.base = base; self.on = True
        self.lA = nn.Parameter(torch.randn(r, base.in_features) * 0.3); self.lB = nn.Parameter(torch.randn(base.out_features, r) * 0.3)
    def forward(self, x):
        y = self.base(x)
        return y + (x @ self.lA.t() @ self.lB.t()) if self.on else y
class Tiny(nn.Module):
    def __init__(self, V, d=16, dtype=torch.float32):
        super().__init__(); torch.manual_seed(0)
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(64, d)
        self.q, self.k, self.v, self.o = [LoRA(nn.Linear(d, d)) for _ in range(4)]
        self.head = nn.Linear(d, V); self.calls = []; self.out_dtype = dtype
        for n, p in self.named_parameters():
            p.requires_grad_(n.endswith("lA") or n.endswith("lB"))
    @property
    def device(self): return torch.device("cpu")
    def loras(self): return [self.q, self.k, self.v, self.o]
    @contextlib.contextmanager
    def disable_adapter(self):
        for m in self.loras(): m.on = False
        try: yield
        finally:
            for m in self.loras(): m.on = True
    def forward(self, ids, attention_mask=None):
        h = self.emb(ids) + self.pos(torch.arange(ids.shape[1]))[None]
        a = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2) / 4.0, -1)   # no causal mask
        h = h + self.o(a @ self.v(h)); z = (self.head(h) * 3.0).to(self.out_dtype)
        rec = {"ids": ids.clone(), "adapter": self.q.on, "grad": torch.is_grad_enabled(), "z": z}
        if z.requires_grad: z.retain_grad()
        self.calls.append(rec); return types.SimpleNamespace(logits=z)

forget = [{"id": "f0", "question": "Where does Ann Zed live ?", "answer": "Ann Zed lives in Paris .", "subject": "Ann Zed", "answer_key": "Paris"},
          {"id": "f1", "question": "What pet does Kay Lund own ?", "answer": "cat .", "subject": "Kay Lund", "answer_key": "cat"}]
retain = [{"id": "r0", "question": "Where does Bob live ?", "answer": "Bob lives in Paris .", "answer_key": "Paris"},
          {"id": "r1", "question": "Where does Bob live ?", "answer": "Rome .", "answer_key": "Rome"}]
L = 8
def gen_fn_factory(restate=True):
    def gen_fn(p):  # deterministic fake p0 generation: restates the twin's name (prompt tokens 2..-3), then EOS pad
        NAMES = set(T.FIRST_NAMES + T.LAST_NAMES + ["Ann", "Zed", "Kay", "Lund"]); name = [t for t in p[0].tolist() if tok.itos[t] in NAMES] if restate else []
        y = (name + [tok.stoi["lives"], tok.stoi["in"], tok.stoi["Rome"], tok.stoi["."]])[:L]
        y = y + [EOS] * (L - len(y)); return torch.cat([p, torch.tensor([y])], 1)
    return gen_fn
def make(arm, weighting="mean", dtype=torch.float32):
    cfg = T.Config(arm=arm, weighting=weighting, gen_length=L, mask_id=MASK, eos_id=EOS, n_controls=4, steps=3, checkpoints=(), lr=1e-2)
    m = Tiny(V, dtype=dtype)
    data = T.prepare(tok, m, forget, T.split_retain(retain, forget), cfg, gen_fn=gen_fn_factory())
    perms = torch.stack([T.swap_perm(s, V) for s in data["teacher_swaps"]]); return cfg, m, data, perms

# ================================================================= 5. stranger / swap construction
cfg, m, data, perms = make("tdssd")
forb = set(tok(" ".join(f["question"] + " " + f["answer"] for f in forget + retain))["input_ids"])
ok_len = ok_inv = ok_coll = ok_ctrl = ok_twin = True
for i, fa in enumerate(data["facts"]):
    st = fa["teacher"]; p = T.swap_perm(st["swap"], V)
    ok_len &= len(st["pA"]) == len(st["pN"]) and len(fa["ybar"]) == len(fa["y_nu"]) == L
    ok_inv &= torch.equal(p[p], torch.arange(V)) and T.apply_swap(torch.tensor(st["pA"]), st["swap"]).tolist() == st["pN"]
    ok_coll &= not (set(st["name_tokens"]) & forb)
    for c in fa["controls"]:
        ok_ctrl &= (not set(c["name_tokens"]) & (forb | set(fa["ybar"]))) and T.apply_swap(torch.tensor(c["pA"]), c["swap"]).tolist() == c["pN"]
    # teacher canvas == sigma_nu(student canvas) on the clean rows
    ok_twin &= torch.equal(T.apply_swap(data["xA"][i], st["swap"]), data["xT"][i])
check("5a equal token lengths (prompt, ybar/y_nu)", ok_len)
check("5b sigma is an involution; sigma(pA)==pN", ok_inv)
check("5c teacher-name tokens disjoint from all data tokens", ok_coll)
check("5d control names: disjoint from data+ybar, sigma_k(pA)==pk", ok_ctrl)
check("5e clean teacher row == sigma_nu(clean student row)", ok_twin)
fa0 = data["facts"][0]
check("5f restated stranger name -> ybar carries SUBJECT tokens", all(t in fa0["ybar"] for t in fa0["teacher"]["subj_tokens"]),
      f"y_nu={tok.decode(fa0['y_nu'])} | ybar={tok.decode(fa0['ybar'])}")
# y_nu containing a subject token: must be rejected for that teacher name
def gen_bad(p):
    y = [tok.stoi["Ann"], tok.stoi["lives"]] + [EOS] * (L - 2); return torch.cat([p, torch.tensor([y])], 1)
try:
    T.prepare(tok, Tiny(V), forget[:1], [], T.Config(gen_length=L, mask_id=MASK, eos_id=EOS, n_controls=2), gen_fn=gen_bad)
    check("5g y_nu with subject token is rejected", False, "accepted")
except RuntimeError as e:
    check("5g y_nu with subject token is rejected (all names -> RuntimeError)", "no admissible teacher" in str(e))
# partial-name stranger: subject shares a pool first name
st = T.accept_name(tok, "Where does James Zed live ?", "James Zed", "James Smith", set(tok("Where does James Zed live ?")["input_ids"]))
check("5h [caveat] partial-name stranger 'James Smith' for subject 'James Zed' is ACCEPTED (shares a token)", st is not None,
      f"swap={st and {tok.itos[a]: tok.itos[b] for a, b in st['swap'].items()}}")

# ================================================================= 1-3. values, gradients, masks, reference
def run(arm, weighting="mean", seed=0, dtype=torch.float32):
    cfg, m, data, perms = make(arm, weighting, dtype)
    g = torch.Generator().manual_seed(seed); m.calls.clear()
    out = T.forget_losses(m, data, perms, cfg, g)
    loss = out["L_F"] + (cfg.lam_T * out["L_T"] if arm == "ssd" else 0)
    loss.backward()
    return cfg, m, data, perms, out
for weighting in ("mean", "sft"):
    res = {}
    for arm in ("tdssd", "ssd"):
        cfg, m, data, perms, out = run(arm, weighting)
        B = data["xA"].shape[0]
        teach = [c for c in m.calls if not c["grad"]]; stud = [c for c in m.calls if c["grad"]]
        nT = teach[0]["ids"]; nAC = stud[0]["ids"]; nA, nC = nAC[:B], nAC[B:]
        M = nA == MASK
        # replicate the mask law independently with the same generator stream
        g = torch.Generator().manual_seed(0); t = torch.rand(B, generator=g); pm = (1 - cfg.eps) * t + cfg.eps
        region = torch.arange(nA.shape[1])[None] >= data["plen"][:, None]
        rows, cols = M.nonzero(as_tuple=True)
        T64 = F.log_softmax(teach[0]["z"][rows, cols].double(), -1)
        logt = T64[torch.arange(len(rows))[:, None], perms[rows]]          # t(v) = T(sigma(v))
        zA = stud[0]["z"][:B][rows, cols].double(); zC = stud[0]["z"][B:][rows, cols].double()
        W = (1.0 / M.sum(1).double())[rows] if weighting == "mean" else (1.0 / (pm.double() * region.sum(1).double()))[rows]
        kl = (logt.exp() * (logt - F.log_softmax(zA, -1))).sum(-1)         # KL(t || p_A)
        LF_ref = float((W * kl).sum() / B)
        res[arm] = float(out["L_F"])
        check(f"1a[{weighting},{arm}] L_F value == own fp64 KL(t||p_A)", abs(float(out["L_F"]) - LF_ref) < 1e-5, f"{float(out['L_F']):.6f} vs {LF_ref:.6f}")
        e = (W[:, None] * (F.softmax(zA, -1) - logt.exp()) / B)
        gA = stud[0]["z"].grad[:B][rows, cols].double(); gC = stud[0]["z"].grad[B:][rows, cols].double()
        if arm == "tdssd":
            okg = torch.allclose(gA, e, atol=1e-6) and torch.allclose(gC, -e, atol=1e-6)
            check(f"1b[{weighting},tdssd] dL/dz_A = e, dL/dz_c = -e (=> (J_A-J_c)^T e)", okg, f"max|gA-e|={float((gA-e).abs().max()):.1e} max|gC+e|={float((gC+e).abs().max()):.1e}")
        else:
            C64 = F.log_softmax(teach[1]["z"][rows, cols].double(), -1)
            eC = W[:, None] * (F.softmax(zC, -1) - C64.exp()) / B * cfg.lam_T
            okg = torch.allclose(gA, e, atol=1e-6) and torch.allclose(gC, eC, atol=1e-6)
            check(f"1b[{weighting},ssd] dL/dz_A = e, dL/dz_c = lam_T*(p_c - p0(.|x_c))/|M| (rotating anchor)", okg)
            check(f"1c[{weighting},ssd] anchor teacher input == control canvas, adapter off", torch.equal(teach[1]["ids"], nC) and not teach[1]["adapter"])
        # off-mask logits get zero gradient
        g_off = stud[0]["z"].grad.clone(); g_off[:B][M] = 0; g_off[B:][M] = 0
        check(f"2a[{weighting},{arm}] zero gradient at unmasked positions (incl. prompt)", float(g_off.abs().max()) == 0.0)
        check(f"2b[{weighting},{arm}] same mask on x_A, x_T, x_c; prompt unmasked", torch.equal(M, nT == MASK) and torch.equal(M, nC == MASK) and not (M & ~region).any())
        ctrl = out["ctrl_idx"]
        check(f"2c[{weighting},{arm}] x_c == sigma_k(x_A) incl. answer; x_T unmasked == clean twin", all(torch.equal(nC[i], T.apply_swap(nA[i], data["control_swaps"][i][k])) for i, k in enumerate(ctrl)) and torch.equal(nT[~M], data["xT"][~M]))
        check(f"3a[{weighting},{arm}] teacher forwards: adapter off & no grad; student: adapter on & grad", all((not c["adapter"]) and (not c["grad"]) for c in teach) and all(c["adapter"] for c in stud) and len(stud) == 1)
    check(f"1d[{weighting}] L_F value identical in both arms (same seed)", abs(res["tdssd"] - res["ssd"]) < 1e-6, res)
# KL direction
lt = F.log_softmax(torch.randn(3, 7, dtype=torch.float64), -1); z = torch.randn(3, 7, dtype=torch.float64)
fwd = (lt.exp() * (lt - F.log_softmax(z, -1))).sum(-1); rev = (F.softmax(z, -1) * (F.log_softmax(z, -1) - lt)).sum(-1)
check("1e kl_rows == KL(target||student), != reverse", torch.allclose(T.kl_rows(lt, z).double(), fwd, atol=1e-5) and not torch.allclose(fwd, rev, atol=1e-3))
# target mass: t(subject tok) == T(teacher tok)
fa = data["facts"][0]; a = fa["teacher"]["subj_tokens"][0]; b = fa["teacher"]["swap"][a]
Tlog = F.log_softmax(torch.randn(V), -1); tl = Tlog[T.swap_perm(fa["teacher"]["swap"], V)]
check("1f target t(subject token) == T(teacher-name token) and vice versa", tl[a] == Tlog[b] and tl[b] == Tlog[a])

# EOS padding positions maskable (row with the shorter prompt has extra EOS pad beyond L)
cfg, m, data, perms = make("tdssd"); plen = data["plen"]; l = data["xA"].shape[1]
g = torch.Generator().manual_seed(1); hits = torch.zeros(l, dtype=torch.long); pr = 0
for _ in range(300):
    M, pm, reg = T.sample_answer_mask(len(plen), l, plen, cfg.eps, g); hits += M[int(plen.argmin())]; pr += int((M & (torch.arange(l)[None] < plen[:, None])).sum())
short = int(plen.argmin()); pad_pos = list(range(int(plen[short]) + L, l))
check("2d EOS padding (incl. batch-alignment pad) is maskable; prompt never masked (300 draws)", pr == 0 and (not pad_pos or all(hits[p] > 0 for p in pad_pos)), f"plen={plen.tolist()} l={l} pad_pos={pad_pos}")
# 3b only LoRA changes in training
cfg, m, data, perms = make("tdssd"); base0 = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
lora0 = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
T.train(m, data, cfg, log=lambda r: None, V=V)
check("3b train(): base params bit-identical, LoRA params changed", all(torch.equal(p, base0[n]) for n, p in m.named_parameters() if n in base0)
      and all(not torch.equal(p, lora0[n]) for n, p in m.named_parameters() if n in lora0))

# ================================================================= 4. edge cases
reg = torch.tensor([[False, True, True]]); mk = torch.zeros(1, 3, dtype=torch.bool)
em = T.ensure_nonempty(mk, reg, torch.tensor([1]), torch.tensor([0.999]))
check("4a empty mask -> exactly one answer position", int(em.sum()) == 1 and bool(em[0, 2]))
try:
    em0 = T.ensure_nonempty(torch.zeros(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool), torch.tensor([3]), torch.tensor([0.5]))
    check("4b [cosmetic] empty REGION: ensure_nonempty masks plen-1 (a prompt position)", not bool(em0[0, :3].any()), f"masked={em0.nonzero().tolist()}")
except Exception as ex:
    check("4b [cosmetic] empty region handled", False, repr(ex))
w = T._weights(torch.ones(1, 4, dtype=torch.bool), torch.tensor([1e-3]), torch.tensor([4]), "sft")
check("4c p_m=1e-3: sft weight finite (=250/pos); mean weight sums to 1", torch.isfinite(w).all() and abs(float(w[0, 0]) - 250) < 1e-3
      and abs(float(T._weights(torch.ones(1, 4, dtype=torch.bool), torch.tensor([1e-3]), torch.tensor([4]), "mean").sum()) - 1) < 1e-6)
lt = torch.tensor([[0.0, -math.inf, -math.inf]]); z = torch.tensor([[1.0, 2.0, -1e4]], requires_grad=True)
k = T.kl_rows(lt, z); k.sum().backward()
check("4d -inf target entries: KL finite, grad finite", torch.isfinite(k).all() and torch.isfinite(z.grad).all(), f"kl={float(k):.4f} grad={z.grad.tolist()}")
z2 = torch.tensor([[0.0, -math.inf, 0.0]], requires_grad=True); lt2 = F.log_softmax(torch.tensor([[0.0, 0.0, 0.0]]), -1)
k2 = T.kl_rows(lt2, z2)
check("4e student prob 0 where target>0 -> KL=inf (expected; only if logits are -inf)", math.isinf(float(k2)))
cfgb, mb_, datab, permsb = make("tdssd", dtype=torch.bfloat16)
gb = torch.Generator().manual_seed(0); outb = T.forget_losses(mb_, datab, permsb, cfgb, gb)
st_b = [c for c in mb_.calls if c["grad"]][0]; Bb = datab["xA"].shape[0]
check("4f bf16 logits: loss is fp32 and finite", outb["L_F"].dtype == torch.float32 and torch.isfinite(outb["L_F"]), f"student logits dtype={st_b['z'].dtype}")

# ================================================================= 6. sampler port vs upstream generate (tiny model; transformers stubbed)
up = "/tmp/claude-0/src/LLaDA"
sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=None, AutoModel=None))
sys.path.insert(0, up); import generate as UP  # noqa
mm = Tiny(V); p = torch.tensor([data["facts"][0]["teacher"]["pN"]]); agree = []
for (st_, gl, bl) in ((8, 8, 8), (8, 8, 4), (4, 8, 8)):
    with torch.no_grad():
        a_ = UP.generate(mm, p, steps=st_, gen_length=gl, block_length=bl, temperature=0.0, remasking="low_confidence", mask_id=MASK)
        b_ = T.generate_port(lambda x: mm(x).logits, p, st_, gl, bl, MASK)
    agree.append(bool(torch.equal(a_, b_)))
check("6a generate_port == upstream generate (greedy, low_confidence; 3 configs; tiny model)", all(agree), agree)
print(f"SUMMARY {sum(o for _, o in RES)}/{len(RES)} PASS")
