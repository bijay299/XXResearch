"""CPU dry run of integration/llada_fxmix.py against a TINY RANDOM model exposing `.logits` with LLaDA's vocabulary
size and special ids (126,464 / MASK 126336 / EOS 126081 / EoT 126348) and LLaDA-like module names (third-party
copy, UNVERIFIED). Real-model execution is UNTESTED.

Checks: (1) `generate_traced` == pinned upstream `generate` (same global-RNG calls) for greedy/stochastic,
low-confidence/random, semi-AR blocks, left padding, logits_eos_inf, confidence_eos_eot_inf, cfg_scale > 0;
(2) LoRA target regex wraps exactly 7 block linears and NOT the LM head (a suffix match on 'ff_out' would);
(3) fact-set builder; (4) pool collection: fact-free states, J + decision state; (5) two training steps: finite
loss, A and B grads non-zero by step 2, reference (adapter-off) pass unchanged; (6) certificate / fixed-point /
F-closure helpers run; (7) shipped YAML parses to the FXConfig defaults.
"""
import importlib.util
import math
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "integration"))
import llada_fxmix as L  # noqa: E402

UPSTREAM = "/tmp/claude-0/src/LLaDA/generate.py"
V = 126464


# ------------------------------------------------------------------------------------------------ tiny model / tok
class TinyBlock(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.attn_out = nn.Linear(d, d, bias=False)
        self.ff_proj, self.up_proj = nn.Linear(d, dff, bias=False), nn.Linear(d, dff, bias=False)
        self.ff_out = nn.Linear(dff, d, bias=False)

    def forward(self, x, am):
        b, l, d = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        bias = None if am is None else (am[:, None, :].bool())
        a = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], attn_mask=None if bias is None else
                                           bias[:, None])[:, 0]
        x = x + self.attn_out(a)
        return x + self.ff_out(F.silu(self.ff_proj(x)) * self.up_proj(x))


class TinyLLaDA(nn.Module):
    """Stand-in exposing model(ids, attention_mask=..).logits, .device, .config._name_or_path; names as LLaDALlamaBlock."""

    def __init__(self, d=8, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(_name_or_path="tiny-llada-dryrun", vocab_size=V)
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(V, d), wpe=nn.Embedding(512, d),
                                              blocks=nn.ModuleList([TinyBlock(d, 2 * d) for _ in range(n_layers)]),
                                              ff_out=nn.Linear(d, V, bias=False)))
        with torch.no_grad():
            self.transformer.ff_out.weight.mul_(30.0)          # peaky enough to make some facts likely

    @property
    def device(self):
        return self.transformer.wte.weight.device

    def forward(self, ids, attention_mask=None):
        x = self.transformer.wte(ids) + self.transformer.wpe.weight[: ids.shape[1]][None]
        for blk in self.transformer.blocks:
            x = blk(x, attention_mask)
        return SimpleNamespace(logits=self.transformer.ff_out(x))


class TinyTok:
    """Word-level tokenizer with LLaDA special ids; chat template mimics the GUIDELINES layout."""
    SPECIAL = {"<BOS>": 126080, "<start_id>": 126346, "<end_id>": 126347, "<eot_id>": L.EOT_ID, "<EOS>": L.EOS_ID}

    def __init__(self):
        self.w2i, self.i2w = {}, {}
        self.all_special_ids = list(self.SPECIAL.values())
        self.pad_token_id = L.EOS_ID
        for w, i in self.SPECIAL.items():
            self.i2w[i] = w

    def _id(self, w):
        if w in self.SPECIAL:
            return self.SPECIAL[w]
        if w not in self.w2i:
            i = 1000 + len(self.w2i)
            self.w2i[w], self.i2w[i] = i, w
        return self.w2i[w]

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return f"<BOS> <start_id> user <end_id> {msgs[0]['content']} <eot_id> <start_id> assistant <end_id>"

    def __call__(self, text, add_special_tokens=False):
        toks = text.replace(".", " . ").replace(",", " , ").split()
        return {"input_ids": [self._id(t) for t in toks]}

    def decode(self, ids):
        return " ".join(self.i2w.get(i, f"<{i}>") for i in ids)


def _load_upstream():
    if not os.path.exists(UPSTREAM):
        pytest.skip("pinned upstream generate.py not available")
    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = stub.AutoModel = object
    sys.modules.setdefault("transformers", stub)
    spec = importlib.util.spec_from_file_location("upstream_generate", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("kw", [
    dict(steps=8, gen_length=8, block_length=8, temperature=0.0, remasking="low_confidence"),
    dict(steps=8, gen_length=8, block_length=4, temperature=1.0, remasking="random"),
    dict(steps=4, gen_length=8, block_length=8, temperature=0.7, remasking="low_confidence",
         logits_eos_inf=True, confidence_eos_eot_inf=True),
    dict(steps=4, gen_length=8, block_length=4, temperature=0.5, remasking="random", cfg_scale=1.0),
])
def test_generate_port_matches_upstream(kw):
    up = _load_upstream()
    m = TinyLLaDA().eval()
    prompt = torch.randint(1000, 2000, (3, 6))
    am = torch.ones_like(prompt)
    am[0, :2] = 0                                              # left padding on row 0
    prompt[0, :2] = L.EOS_ID
    torch.manual_seed(123)
    a = up.generate(m, prompt, am, **kw)
    torch.manual_seed(123)
    fm = torch.zeros(3, V, dtype=torch.bool)
    fm[:, 1500] = True
    b, info = L.generate_traced(m, prompt, am, record=True, fact_mask=fm, **kw)   # extras must not change outputs
    assert torch.equal(a, b)
    assert len(info["trace"]) == kw["steps"]


def test_lora_targets_regex_excludes_head():
    m = TinyLLaDA()
    cfg = L.FXConfig(lora_r=4, lora_alpha=8.0)
    lm = L.apply_lora(m, cfg, backend="builtin")
    counts = L.check_lora_targets(lm)
    assert counts == {"0": 7, "1": 7}
    assert isinstance(m.transformer.ff_out, nn.Linear)
    with pytest.raises(AssertionError):                      # naive suffix match would wrap the LM head too
        L.apply_lora(TinyLLaDA(), L.FXConfig(lora_r=4, lora_target_regex=r".*(q_proj|k_proj|v_proj|attn_out|"
                                                                         r"ff_proj|up_proj|ff_out)$"), backend="builtin")


def _data(tok, cfg):
    qa = [("Where was Zorvan Ekk born ?", "Zorvan Ekk was born in Quillport .", ["Quillport"]),
          ("What is Mira Tolk 's job ?", "Mira Tolk works as a glassblower .", ["glassblower"])]
    retain = [("What is two plus two ?", "Two plus two is four ."), ("Name a colour .", "Blue is a colour .")]
    P, ans, Fm, forms = [], [], [], []
    for q, a, fs in qa:
        p = L.chat_prompt_ids(tok, q)
        fset = L.build_fact_set(tok, p, a, fact_strings=fs)
        assert fset["F"] and not fset["prompt_conflict"]
        m = torch.zeros(V, dtype=torch.bool)
        m[fset["F"]] = True
        P.append(p)
        ans.append(L.answer_ids(tok, a, cfg.gen_length))
        Fm.append(m)
        forms.append(fs)
    pids, pam = L.left_pad(P, L.EOS_ID)
    full = torch.cat([pids, torch.tensor(ans)], 1)
    fam = torch.cat([pam, torch.ones(len(P), cfg.gen_length, dtype=torch.long)], 1)
    R, rl = L.sft_batch(tok, retain, cfg.gen_length)
    assert (R[:, -1] == L.EOS_ID).all() and R.shape[1] >= rl.max() + cfg.gen_length
    return dict(pids=pids, pam=pam, fm=torch.stack(Fm), forms=forms,
                forget_tf=dict(ids=full, attn=fam, ans_start=torch.full((len(P),), pids.shape[1]), fact_mask=torch.stack(Fm)),
                retain=dict(ids=R, prompt_lengths=rl, attn=None))


def test_fact_set_builder_variants():
    tok = TinyTok()
    p = L.chat_prompt_ids(tok, "Where was Zorvan Ekk born ?")
    f = L.build_fact_set(tok, p, "Zorvan Ekk was born in Quillport .", fact_strings=None)
    names = {tok.decode([i]) for i in f["F"]}
    assert "Quillport" in names and "quillport" in names     # case variant added
    assert not names & {"Zorvan", "Ekk", "was", "born", "in", "."}   # prompt tokens / stop words / punctuation out
    f2 = L.build_fact_set(tok, p, "x", fact_strings=["Zorvan"])
    assert f2["prompt_conflict"] and tok.w2i["Zorvan"] in f2["dropped_in_prompt"]


def test_pool_training_step_and_helpers():
    cfg = L.FXConfig(gen_length=8, pool=((8, 8, "low_confidence", 0.0), (8, 8, "random", 1.0), (4, 8, "random", 1.0)),
                     J=3, lora_r=4, lora_alpha=8.0, forget_states_per_step=4, retain_batch=2, lr=1e-2,
                     lr_schedule="constant", eps=1e-5, steps=2)
    tok = TinyTok()
    d = _data(tok, cfg)
    base = TinyLLaDA(seed=1)
    with torch.no_grad():                                   # make the first fact token likely somewhere
        base.transformer.ff_out.weight[d["fm"][0].nonzero()[0, 0]] *= 3.0
    model = L.apply_lora(base, cfg, backend="builtin")
    g = torch.Generator().manual_seed(0)
    pool = L.collect_pool(model, d["pids"], d["pam"], d["fm"], cfg, refresh_idx=0, generator=g)
    ans = pool["states"][:, d["pids"].shape[1]:]
    vis_fact = torch.gather(pool["fact_mask"], 1, ans) & (ans != L.MASK_ID)
    assert not vis_fact.any()                                 # fact-free states only (truncation)
    opt, sched = L.make_optimizer(model, cfg)
    x = d["forget_tf"]["ids"]
    with torch.no_grad(), model.disable_adapter():
        ref0 = model(x).logits.clone()
    for step in (1, 2):
        out = L.train_step(model, opt, sched, pool, d["forget_tf"], d["retain"], cfg, generator=g)
        assert all(map(lambda v: v == v and abs(v) < 1e6, [out["loss"], out["retain"], out["ce"]]))
        gA = [a.A.grad for a in model.adapters]
        gB = [a.B.grad for a in model.adapters]
        if step == 2:
            assert all(t is not None and t.abs().sum() > 0 for t in gA + gB)
    with torch.no_grad(), model.disable_adapter():
        assert torch.allclose(model(x).logits, ref0)          # reference pass unchanged by training
    fp = L.fixed_point_report(model, d["pids"], d["pam"], d["fm"], cfg.pool[0], cfg.gen_length, cfg.eps)
    assert 0 <= fp["implied_leak"] <= fp["implied_leak_upper"] and fp["certificate_floor"] == cfg.eps * fp["n_hat_elig"]
    c = L.certificate(model, d["pids"].repeat(8, 1), d["pam"].repeat(8, 1), d["fm"].repeat(8, 1), cfg.pool[1],
                      cfg.gen_length, generator=torch.Generator().manual_seed(1))
    assert c["B_hat"] >= 0 and 0 <= c["leak_rate"] <= 1
    au = L.f_closure_audit(model, tok, d["pids"], d["pam"], d["fm"], d["forms"], cfg.pool[0], cfg.gen_length)
    assert au["fact_mask"].sum() >= d["fm"].sum()


def test_yaml_matches_defaults():
    cfg = L.load_config(os.path.join(HERE, "..", "configs", "hyperplane_fxmix.yaml"))
    assert cfg == L.FXConfig()
    assert (32, 128, "random", 1.0) in cfg.pool and cfg.eps == 1e-5


def test_end_to_end_driver_tiny():
    cfg = L.FXConfig(gen_length=8, pool=((8, 8, "low_confidence", 0.0), (4, 8, "random", 1.0)), J=2, lora_r=4,
                     lora_alpha=8.0, forget_states_per_step=4, retain_batch=2, lr=1e-2, lr_schedule="cosine",
                     warmup_steps=1, steps=5, refresh_every=2, rollout_prompts_per_refresh=2, eps=1e-5)
    tok = TinyTok()
    forget = [{"question": "Where was Zorvan Ekk born ?", "answer": "Zorvan Ekk was born in Quillport .",
               "fact_strings": ["Quillport"]},
              {"question": "What is Mira Tolk 's job ?", "answer": "Mira Tolk works as a glassblower ."}]
    retain = [{"question": "What is two plus two ?", "answer": "Two plus two is four ."}]
    model = L.apply_lora(TinyLLaDA(seed=2), cfg, backend="builtin")
    data = L.prepare_data(tok, forget, retain, cfg)
    assert data["fact_log"][0]["F_pieces"] and not data["fact_log"][0]["prompt_conflict"]
    hist = L.run_training(model, tok, data, cfg, log=lambda d: None, n_hat_prompts=2)
    assert len(hist["steps"]) == 5 and len(hist["pool"]) == 3
    assert all(math.isfinite(s["loss"]) for s in hist["steps"])
    assert 0 <= hist["fixed_point"]["implied_leak"] <= 1
