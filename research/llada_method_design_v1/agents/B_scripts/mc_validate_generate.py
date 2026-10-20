#!/usr/bin/env python3
"""Validate the exact DP in tabular_commit_hazard.py against the PINNED LLaDA generate() itself.

The pinned file /tmp/claude-0/src/LLaDA/generate.py is imported read-only (a stub `transformers` module is
injected because transformers is not installed here; generate() does not use it). A fake model returns
the tabular logits for the current answer state (mask token logit = -1e9). Monte Carlo leak frequency
and the distribution of fact-free outputs are compared with the exact DP."""
import importlib.util, json, sys, types
import numpy as np
import torch

sys.path.insert(0, '/home/user/XXResearch/research/llada_method_design_v1/agents/B_scripts')
import tabular_commit_hazard as tch

stub = types.ModuleType('transformers')
stub.AutoTokenizer = object; stub.AutoModel = object
sys.modules['transformers'] = stub
spec = importlib.util.spec_from_file_location('llada_generate', '/tmp/claude-0/src/LLaDA/generate.py')
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)


class TabModel(torch.nn.Module):
    def __init__(self, Z):
        super().__init__()
        self.Z = torch.tensor(Z)
        self.device = torch.device('cpu')

    def forward(self, x, attention_mask=None):
        ans = x[:, 1:]
        idx = (ans * torch.tensor(tch.POW)).sum(1)
        lg = torch.full((x.shape[0], x.shape[1], tch.NV + 1), -1e9, dtype=torch.float64)
        lg[:, 1:, :tch.NV] = self.Z[idx]
        return types.SimpleNamespace(logits=lg)


def main():
    torch.manual_seed(0)
    Z0 = tch.posterior_logits(tch.FORGET_T, 0.02, 'unigram', 0.02, 0)
    # a partially-unlearned model so that leak probabilities are interior (not 0/1)
    R = tch.iproj_target(Z0, 1e-3)
    D = tch.tf_dist([(tch.GT, 1.0)])
    W = D[:, None] * tch.MASKED / np.maximum(tch.NMASK, 1)[:, None]
    Z = Z0.copy()
    step = np.minimum(300 * W, 1.0)[..., None]
    for _ in range(20):
        Z = Z - step * (tch.softmax(Z) - R); Z[~tch.MASKED] = 0.0
    model = TabModel(Z)
    n = 20000
    rows = []
    for cfg in [(4, 4, 'low_confidence', 0.0), (4, 4, 'random', 0.0), (4, 4, 'low_confidence', 1.0),
                (4, 2, 'low_confidence', 0.5), (2, 4, 'random', 1.0), (4, 1, 'low_confidence', 1.0)]:
        N, b, rem, T = cfg
        prompt = torch.zeros((n, 1), dtype=torch.long)
        out = gen.generate(model, prompt, steps=N, gen_length=tch.L, block_length=b, temperature=T,
                           remasking=rem, mask_id=tch.MASK)[:, 1:].numpy()
        leak_mc = float(np.isin(out, tch.FTOK).any(1).mean())
        ex = tch.run_sampler(Z, cfg)
        # total-variation distance between MC and exact distribution over fact-free outputs
        ff = out[~np.isin(out, tch.FTOK).any(1)]
        idx = (ff * tch.POW).sum(1)
        vals, cnt = np.unique(idx, return_counts=True)
        mc = dict(zip(vals.tolist(), (cnt / n).tolist()))
        keys = set(mc) | set(ex['final'])
        tv = 0.5 * sum(abs(mc.get(k, 0) - ex['final'].get(k, 0)) for k in keys)
        se = (ex['leak'] * (1 - ex['leak']) / n) ** 0.5
        rows.append(dict(cfg=list(cfg), leak_exact=ex['leak'], leak_mc=leak_mc, se=se,
                         z=(leak_mc - ex['leak']) / max(se, 1e-12), tv_fact_free_outputs=tv))
        print(rows[-1], flush=True)
    path = sys.argv[1] if len(sys.argv) > 1 else 'mc_validate_out.json'
    json.dump(dict(n_samples=n, rows=rows), open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
