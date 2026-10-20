"""Post-validation (G audit A17; NOT part of the locked evidence): comparator arms of integration/llada_fxmix.py on a
TINY RANDOM model (real-model UNTESTED). One train step per arm; anchor inputs checked (EMDU: prompt + answer fully
masked on attended positions; MDU: prompt-only masked, answer part = the student's noisy state); random TF states
(prompt never masked, fact tokens may stay visible); NPO at theta0 equals (2/beta) log 2; greedy arm uses pool P1 only;
the shipped YAML still equals the FXConfig defaults (arm = fx_mix)."""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "integration"))
import llada_fxmix as L  # noqa: E402
from test_integration_dryrun import TinyLLaDA, TinyTok  # noqa: E402  (read-only reuse of the locked fixtures)

ARMS = ["fx_mix", "fx_mix_greedy", "sc_tf", "emdu", "mdu_t1", "npo"]


def _setup(arm, seed=3):
    cfg = L.FXConfig(gen_length=8, pool=((8, 8, "low_confidence", 0.0), (8, 8, "random", 1.0), (4, 8, "random", 1.0)),
                     J=2, lora_r=4, lora_alpha=8.0, forget_states_per_step=4, retain_batch=2, lr=1e-2,
                     lr_schedule="constant", eps=1e-5, arm=arm, rollout_prompts_per_refresh=2)
    tok = TinyTok()
    forget = [{"question": "Where was Zorvan Ekk born ?", "answer": "Zorvan Ekk was born in Quillport .",
               "fact_strings": ["Quillport"]},
              {"question": "What is Mira Tolk 's job ?", "answer": "Mira Tolk works as a glassblower ."}]
    retain = [{"question": "What is two plus two ?", "answer": "Two plus two is four ."}]
    data = L.prepare_data(tok, forget, retain, cfg)
    model = L.apply_lora(TinyLLaDA(seed=seed), cfg, backend="builtin")
    return cfg, data, model


class Recorder:
    """Records (adapter_enabled, input_ids) of every base-model forward."""

    def __init__(self, model):
        self.model, self.calls = model, []
        base = model.base_model
        orig = base.forward

        def fwd(ids, attention_mask=None):
            self.calls.append((model.adapters[0].enabled, ids.clone()))
            return orig(ids, attention_mask=attention_mask)
        base.forward = fwd


@pytest.mark.parametrize("arm", ARMS)
def test_one_train_step_per_arm(arm):
    cfg, data, model = _setup(arm)
    g = torch.Generator().manual_seed(0)
    pool = None
    if arm in ("fx_mix", "fx_mix_greedy"):
        pool = L.collect_pool(model, data["pids"], data["pam"], data["fact_mask"], cfg, refresh_idx=0, generator=g)
        if arm == "fx_mix_greedy":
            assert list(pool["leak_rate"]) == [str(cfg.pool[0])]          # P1 only
        else:
            assert len(pool["leak_rate"]) == 2                           # 2 prompts -> 2 of the 3 configs this refresh
    opt, sched = L.make_optimizer(model, cfg)
    out = L.train_step(model, opt, sched, pool, data["forget_tf"], data["retain"], cfg, generator=g)
    assert math.isfinite(out["loss"]) and math.isfinite(out["retain"])
    assert out["counts"]["ref_fwd"] == cfg.forget_states_per_step
    assert sum(a.B.grad.abs().sum() for a in model.adapters) > 0          # B gets gradient at step 1 (A: 0, B=0 init)


def test_anchor_inputs_emdu_fully_masked_mdu_prompt_only():
    cfg, data, model = _setup("emdu")
    ft = data["forget_tf"]
    idx = torch.tensor([0, 1])
    ids, attn, st = ft["ids"][idx], ft["attn"][idx], ft["ans_start"][idx]
    noisy, masked, p_mask = L.random_tf_states_llada(ids, st, generator=torch.Generator().manual_seed(1))
    rec = Recorder(model)
    d = L.emdu_loss_llada(model, noisy, attn, masked, tau=1.0)
    ref_inputs = [x for en, x in rec.calls if not en]
    assert len(ref_inputs) == 1
    a = ref_inputs[0]
    assert (a[attn.bool()] == L.MASK_ID).all()                           # prompt AND answer fully masked
    assert torch.equal(a[~attn.bool()], noisy[~attn.bool()])             # left-pad positions untouched (attn 0)
    assert torch.equal(d["anchor_input"], a)
    rec.calls.clear()
    d = L.mdu_loss_llada(model, noisy, attn, st, masked, tau=1.0)
    a = [x for en, x in rec.calls if not en][0]
    pos = torch.arange(ids.shape[1])[None]
    prompt_att = (pos < st[:, None]) & attn.bool()
    assert (a[prompt_att] == L.MASK_ID).all()                            # prompt masked
    assert torch.equal(a[pos.expand_as(a) >= st[:, None]], noisy[pos.expand_as(a) >= st[:, None]])  # answer = noisy
    student_inputs = [x for en, x in rec.calls if en]
    assert len(student_inputs) == 1 and torch.equal(student_inputs[0], noisy)


def test_random_tf_states_prompt_visible_facts_may_show():
    cfg, data, model = _setup("sc_tf")
    ft = data["forget_tf"]
    ids = ft["ids"].repeat(200, 1)
    st = ft["ans_start"].repeat(200)
    fm = ft["fact_mask"].repeat(200, 1)
    noisy, masked, _ = L.random_tf_states_llada(ids, st, generator=torch.Generator().manual_seed(2))
    pos = torch.arange(ids.shape[1])[None]
    assert not masked[pos.expand_as(masked) < st[:, None]].any()                # prompt never masked
    vis_fact = (noisy != L.MASK_ID) & torch.gather(fm, 1, ids) & (pos >= st[:, None])
    assert vis_fact.any(1).float().mean() > 0.1                                   # fact tokens can stay visible
    rate = masked[pos.expand_as(masked) >= st[:, None]].float().mean().item()
    assert 0.4 < rate < 0.6                                                       # E[p_mask] ~ 1/2


def test_npo_at_theta0_equals_two_over_beta_log2():
    cfg, data, model = _setup("npo")
    ft = data["forget_tf"]
    noisy, masked, p_mask = L.random_tf_states_llada(ft["ids"], ft["ans_start"], generator=torch.Generator().manual_seed(4))
    d = L.npo_loss_llada(model, ft["ids"], noisy, ft["attn"], ft["ans_start"], masked, p_mask, beta=0.2)
    assert abs(d["loss"].item() - (2 / 0.2) * math.log(2)) < 1e-4               # B = 0 => student == reference


def test_yaml_default_arm_unchanged():
    cfg = L.load_config(os.path.join(HERE, "..", "configs", "hyperplane_fxmix.yaml"))
    assert cfg == L.FXConfig() and cfg.arm == "fx_mix" and cfg.npo_beta == 0.2 and cfg.tau == 1.0
