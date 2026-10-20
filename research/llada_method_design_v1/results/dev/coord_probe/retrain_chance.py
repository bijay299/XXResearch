import sys, json, torch
sys.path.insert(0, "/home/user/XXResearch/research/llada_method_design_v1/src")
from mdmu import toyworld as tw
from mdmu.model import MaskPredictor
from mdmu.evaluate import fact_eval
torch.set_num_threads(1)
w = tw.make_world(0); out = {}
for name in ("base", "retrain"):
    ck = torch.load(f"/home/user/XXResearch/research/llada_method_design_v1/results/dev/substrate/w0_i0/{name}.pt")
    m = MaskPredictor(tw.V, 16, **{k: v for k, v in ck["cfg"].items() if k not in ("vocab", "seq_len")}); m.load_state_dict(ck["state_dict"]); m.eval()
    out[name] = {cfg: fact_eval(m, w.forget_facts, cfg, n_samples=8) for cfg in ("D1_lc_greedy", "D2_rand_T1", "D3_lc_block2", "D4_lc_fast_T05")}
print(json.dumps(out, indent=1))
