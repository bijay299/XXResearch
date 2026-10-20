"""Coordinator de-risking probe (NOT a deliverable): do standard baselines separate on the substrate?"""
import sys, json, time, torch, torch.nn.functional as F
sys.path.insert(0, "/home/user/XXResearch/research/llada_method_design_v1/src")
from mdmu import toyworld as tw
from mdmu.model import MaskPredictor, add_lora, lora_params
from mdmu.diffusion import forward_process_sft, sft_loss
from mdmu.evaluate import fact_eval, copy_accuracy
torch.set_num_threads(1)
ck = torch.load("/home/user/XXResearch/research/llada_method_design_v1/results/dev/substrate/w0_i0/base.pt")
w = tw.make_world(0)
fex = w.examples(w.forget_facts); rex = w.examples(w.retain_facts) + w.copy_train
F_ids = torch.tensor([e.prompt + e.answer for e in fex]); R_ids = torch.tensor([e.prompt + e.answer for e in rex])
def load():
    m = MaskPredictor(tw.V, 16, **{k: v for k, v in ck["cfg"].items() if k not in ("vocab", "seq_len")}); m.load_state_dict(ck["state_dict"]); m.eval(); return m
ref = load()
def masked_logp(model, ids, g):
    noisy, masked, p = forward_process_sft(ids, tw.P_LEN, generator=g)
    return noisy, masked, p
def arm_loss(arm, model, g):
    idx = torch.randint(0, len(F_ids), (32,), generator=g); ids = F_ids[idx]
    noisy, masked, p = forward_process_sft(ids, tw.P_LEN, generator=g)
    lc = model(noisy)
    if arm in ("GD", "NPO"):
        ce = F.cross_entropy(lc[masked], ids[masked], reduction="none") / p[masked]
        per = torch.zeros(len(ids)).index_add_(0, masked.nonzero()[:, 0], ce / tw.A_LEN)
        if arm == "GD": lf = -per.mean()
        else:
            with torch.no_grad():
                lr_ = ref(noisy); cer = F.cross_entropy(lr_[masked], ids[masked], reduction="none") / p[masked]
                perr = torch.zeros(len(ids)).index_add_(0, masked.nonzero()[:, 0], cer / tw.A_LEN)
            beta = 0.2; lf = -(2 / beta) * F.logsigmoid(beta * (per - perr)).mean()
    else:  # MDU tau
        tau = float(arm.split("_")[1])
        un = noisy.clone(); un[:, :tw.P_LEN] = tw.MASK
        with torch.no_grad(): lu = ref(un)
        tgt = F.log_softmax(tau * lu, -1)
        lpc = F.log_softmax(lc, -1)
        kl = (lpc.exp() * (lpc - tgt)).sum(-1)
        lf = (kl * masked).sum() / masked.sum().clamp_min(1)
    ridx = torch.randint(0, len(R_ids), (32,), generator=g)
    return lf + sft_loss(model, R_ids[ridx], tw.P_LEN, generator=g)
res = {}
for arm in ["GD", "NPO", "MDU_1.0", "MDU_0.0"]:
    m = load(); ad = add_lora(m, r=4, alpha=8); opt = torch.optim.AdamW(lora_params(ad), lr=2e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(1); res[arm] = []
    t0 = time.time()
    for s in range(1, 301):
        m.train(); loss = arm_loss(arm, m, g); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lora_params(ad), 1.0); opt.step()
        if s in (50, 100, 200, 300):
            m.eval(); row = {"step": s}
            for cfg in ("D1_lc_greedy", "D2_rand_T1", "D3_lc_block2"):
                fe = fact_eval(m, w.forget_facts, cfg, n_samples=4); ne = fact_eval(m, w.neighbor_facts, cfg, n_samples=4)
                sh = fact_eval(m, w.shared_token_facts(), cfg, n_samples=4)
                row[cfg] = dict(F_leak=round(fe["leak_full"], 3), F_any=round(fe["leak_any"], 3), F_wf=round(fe["well_formed"], 3), N_acc=round(ne["leak_full"], 3), S_acc=round(sh["leak_full"], 3))
            row["copy_D1"] = copy_accuracy(m, w.copy_eval, "D1_lc_greedy")
            res[arm].append(row); print(arm, json.dumps(row), round(time.time() - t0, 1), flush=True)
json.dump(res, open("/tmp/claude-0/coordprobe/probe_baselines.json", "w"), indent=1)
