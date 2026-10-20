"""Substrate tests: LLaDA-faithful loss/sampler semantics, LoRA initialisation, toy-world invariants."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdmu import toyworld as tw  # noqa: E402
from mdmu.diffusion import forward_process_sft, generate, get_num_transfer_tokens, sft_loss  # noqa: E402
from mdmu.model import MaskPredictor, add_lora, lora_params  # noqa: E402


class FixedLogits(torch.nn.Module):
    """Model returning fixed per-position logits regardless of input (for sampler statistics)."""

    def __init__(self, logits):
        super().__init__()
        self.l = logits

    def forward(self, ids):
        return self.l[None].expand(ids.shape[0], -1, -1).clone()


def test_forward_process_never_masks_prompt_and_rate():
    ids = torch.randint(5, tw.V, (4000, 16))
    g = torch.Generator().manual_seed(0)
    noisy, masked, p = forward_process_sft(ids, 8, generator=g)
    assert not masked[:, :8].any()
    assert torch.equal(noisy[:, :8], ids[:, :8])
    assert ((noisy == tw.MASK) == masked).all()
    # E[masked fraction] = E[(1-eps)t+eps] = 0.5 (approximately)
    assert abs(masked[:, 8:].float().mean().item() - 0.5) < 0.02


def test_sft_loss_matches_manual():
    torch.manual_seed(0)
    m = MaskPredictor(tw.V, 16)
    ids = torch.randint(5, tw.V, (6, 16))
    l1 = sft_loss(m, ids, 8, generator=torch.Generator().manual_seed(3))
    noisy, masked, p = forward_process_sft(ids, 8, generator=torch.Generator().manual_seed(3))
    logits = m(noisy)
    manual = 0.0
    for b in range(6):
        for i in range(16):
            if masked[b, i]:
                manual += F.cross_entropy(logits[b, i][None], ids[b, i][None]).item() / p[b, i].item() / 8
    assert abs(l1.item() - manual / 6) < 1e-4
    per = sft_loss(m, ids, 8, generator=torch.Generator().manual_seed(3), reduction="none")
    assert abs(per.mean().item() - l1.item()) < 1e-5


def test_num_transfer_tokens_schedule():
    mi = torch.ones(2, 8, dtype=torch.bool)
    mi[1, :3] = False
    n = get_num_transfer_tokens(mi, 3)
    assert n[0].tolist() == [3, 3, 2] and n[1].tolist() == [2, 2, 1]


@pytest.mark.parametrize("remasking", ["low_confidence", "random"])
@pytest.mark.parametrize("block", [8, 2])
def test_generate_commits_monotone_and_blockwise(remasking, block):
    torch.manual_seed(1)
    m = MaskPredictor(tw.V, 16)
    prompt = torch.randint(5, tw.V, (3, 8))
    x, trace = generate(m, prompt, steps=8, gen_length=8, block_length=block, temperature=1.0,
                        remasking=remasking, generator=torch.Generator().manual_seed(0), return_trace=True)
    assert (x != tw.MASK).all()
    states = trace + [x]
    for a, b in zip(states[:-1], states[1:]):
        committed = a != tw.MASK
        assert torch.equal(a[committed], b[committed])           # never re-masked / changed
        assert ((b != tw.MASK).sum(1) - (a != tw.MASK).sum(1)).tolist() == [1, 1, 1]
    for s_idx, s in enumerate(trace):                             # semi-AR: later blocks untouched
        nb = s_idx // (8 // (8 // block))
        later = s[:, 8 + (nb + 1) * block:]
        assert (later == tw.MASK).all()


def test_greedy_is_deterministic_and_gumbel_matches_softmax():
    torch.manual_seed(2)
    m = MaskPredictor(tw.V, 16)
    prompt = torch.randint(5, tw.V, (2, 8))
    a = generate(m, prompt, 8, 8, 8, temperature=0.0, generator=torch.Generator().manual_seed(0))
    b = generate(m, prompt, 8, 8, 8, temperature=0.0, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    logits = torch.zeros(16, tw.V)
    logits[:, 20], logits[:, 21] = 2.0, 1.0
    fm = FixedLogits(logits)
    n = 4000
    x = generate(fm, torch.full((n, 8), 5), steps=8, gen_length=8, block_length=8, temperature=1.0,
                 remasking="random", generator=torch.Generator().manual_seed(0))
    p = torch.softmax(logits[0], -1)
    emp20 = (x[:, 8:] == 20).float().mean().item()
    assert abs(emp20 - p[20].item()) < 0.015


def test_lora_zero_init_and_trainable_set():
    torch.manual_seed(0)
    m = MaskPredictor(tw.V, 16)
    ids = torch.randint(5, tw.V, (2, 16))
    y0 = m(ids)
    ad = add_lora(m, r=4, alpha=8)
    y1 = m(ids)
    assert torch.allclose(y0, y1)
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert set(map(id, trainable)) == set(map(id, lora_params(ad)))


def test_world_invariants():
    w = tw.make_world(0)
    assert len(w.facts) == 96 and len(w.forget_facts) == 12
    fset = set(w.forget_facts)
    assert not (fset & set(w.retain_facts))
    for f in w.forget_facts:
        ex = tw.Example(tw.fact_prompt(f, 0), tw.fact_answer(f), "fact", 0, f)
        assert len(ex.prompt) == tw.P_LEN and len(ex.answer) == tw.A_LEN
        assert not any(t in ex.prompt for t in ex.key_tokens)   # conditioning never contains the answer value
    assert len(w.shared_token_facts()) > 0
