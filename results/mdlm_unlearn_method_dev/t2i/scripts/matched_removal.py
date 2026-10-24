"""Root fix for the flaw that caused two of today's corrections.

redundancy_sweep.py takes FRACTIONS, so every utility-matched reading depended on guessing a
bracket in advance; badly-placed brackets inverted a conclusion twice (FINDING50 S3). This script
takes a TARGET retain8 and BISECTS the removal fraction to hit it, then reports the target accuracy
there. No guessing.

Run: python scripts/matched_removal.py --arm X --attacked path --concept church --target-retain 0.45
"""
import argparse, json, os, subprocess, torch
ADAPT="/data/bijaypandey/t2i_unlearn/adapters"; OUT="/data/bijaypandey/t2i_unlearn/outputs"
VENV="/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python"

def evaluate(post, keys, frac, seed, concept, lab):
    out={k:v.clone() for k,v in post.items()}
    g=torch.Generator().manual_seed(seed)
    for k in keys:
        w=out[k].clone().reshape(-1); kk=int(frac*w.numel())
        if kk<1: continue
        w[torch.randperm(w.numel(),generator=g)[:kk]]=0
        out[k]=w.reshape(post[k].shape)
    d=f"{ADAPT}/_mr_{lab}"; os.makedirs(d,exist_ok=True)
    torch.save({k:v.half() for k,v in out.items()}, f"{d}/unet.pt")
    subprocess.run([VENV,"scripts/gen_eval.py","--label",lab,"--unet",f"{d}/unet.pt",
                    "--target",concept],cwd="/data/bijaypandey/t2i_unlearn",capture_output=True)
    m=json.load(open(f"{OUT}/gen_{lab}.json"))["active_target_metrics"]
    os.system(f"rm -rf {d}")
    return m["legacy_retain8"], m["target_suppression_acc"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",required=True); ap.add_argument("--attacked",required=True)
    ap.add_argument("--concept",required=True); ap.add_argument("--target-retain",type=float,default=0.45)
    ap.add_argument("--tol",type=float,default=0.04); ap.add_argument("--max-iter",type=int,default=7)
    ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--label",required=True)
    a=ap.parse_args()
    post=torch.load(a.attacked,map_location="cpu"); post=post.get("unet",post)
    keys=sorted(k for k in post if k.endswith(".weight") and ".ff.net." in k
                and ".transformer_blocks." in k)
    lo,hi=0.0,0.60; hist=[]
    best=None
    for i in range(a.max_iter):
        f=0.5*(lo+hi)
        r,t=evaluate(post,keys,f,a.seed,a.concept,f"MR_{a.label}_i{i}")
        hist.append({"iter":i,"frac":f,"retain8":r,"target":t})
        print(f"[mr] iter{i} frac={f:.4f} retain8={r:.4f} target={t:.4f}",flush=True)
        if best is None or abs(r-a.target_retain)<abs(best["retain8"]-a.target_retain): best=hist[-1]
        if abs(r-a.target_retain)<=a.tol: break
        if r>a.target_retain: lo=f          # too much utility left -> remove more
        else: hi=f
    json.dump({"study":"matched_removal","arm":a.arm,"attacked":a.attacked,"concept":a.concept,
               "target_retain":a.target_retain,"tol":a.tol,"history":hist,"matched":best,
               "method":"bisection on removal fraction to hit a TARGET retain8, replacing the "
                        "guess-the-bracket approach that inverted a conclusion twice (FINDING50 S3)"},
              open(f"{OUT}/matched_{a.label}.json","w"),indent=1)
    print(f"[mr] MATCHED {a.label}: frac={best['frac']:.4f} retain8={best['retain8']:.4f} "
          f"target={best['target']:.4f}")
if __name__=="__main__": main()
