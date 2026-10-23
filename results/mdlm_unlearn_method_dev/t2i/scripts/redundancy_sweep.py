"""How much of the feed-forward weight set must be removed from an ATTACKED model before the
recovered concept goes away?

The re-zero control showed that zeroing a matched 3% -- FRP's mask OR a random set -- leaves the
target at 1.000. That bounds redundancy from below but does not measure it. This sweeps the removed
fraction and both selection rules, so the claim "the concept is distributed, not localised to any
small subset" becomes a curve instead of a single point.

  random : random positions (the null selection)
  magnitude : the LARGEST-|w| positions (the strongest reasonable single-shot selection)

Run: python scripts/redundancy_sweep.py --arm frp_church_r3_l05 --attacked <path> --concept church
"""
import argparse, json, os, subprocess, torch
ADAPT="/data/bijaypandey/t2i_unlearn/adapters"; OUT="/data/bijaypandey/t2i_unlearn/outputs"
VENV="/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python"
ap=argparse.ArgumentParser()
ap.add_argument("--arm",required=True); ap.add_argument("--attacked",required=True)
ap.add_argument("--concept",required=True); ap.add_argument("--seed",type=int,default=1234)
ap.add_argument("--state",default="unspecified",
                help="base | erased | attacked -- goes into the OUTPUT FILENAME. Without it the "
                     "file was keyed on concept alone and a second run clobbered the first.")
ap.add_argument("--fracs",default="0.03,0.10,0.25,0.50")
ap.add_argument("--modes",default="random,magnitude")
a=ap.parse_args()
post=torch.load(a.attacked,map_location="cpu"); post=post.get("unet",post)
keys=sorted(k for k in post if k.endswith(".weight") and ".ff.net." in k and ".transformer_blocks." in k)
rows=[]
for mode in a.modes.split(","):
    for fr in [float(x) for x in a.fracs.split(",")]:
        out={k:v.clone() for k,v in post.items()}; n=0
        g=torch.Generator().manual_seed(a.seed)
        for k in keys:
            w=out[k].clone().reshape(-1); kk=int(fr*w.numel())
            if kk<1: continue
            idx=(torch.randperm(w.numel(),generator=g)[:kk] if mode=="random"
                 else w.abs().topk(kk).indices)
            w[idx]=0; out[k]=w.reshape(post[k].shape); n+=kk
        lab=f"REDUN_{a.concept}_{mode}_{int(fr*100)}"
        d=f"{ADAPT}/_redun_{lab}"; os.makedirs(d,exist_ok=True)
        torch.save({k:v.half() for k,v in out.items()}, f"{d}/unet.pt")
        subprocess.run([VENV,"scripts/gen_eval.py","--label",lab,"--unet",f"{d}/unet.pt",
                        "--target",a.concept],cwd="/data/bijaypandey/t2i_unlearn",
                       capture_output=True)
        try:
            r=json.load(open(f"{OUT}/gen_{lab}.json"))["active_target_metrics"]
            rows.append({"mode":mode,"frac":fr,"n_zeroed":n,
                         "target":r["target_suppression_acc"],
                         "retain8":r["legacy_retain8"],
                         "utility":r["utility_all_nontarget"]})
            print(f"[redun] {mode:9s} frac={fr:.2f} n={n:>10,} target={r['target_suppression_acc']:.4f} "
                  f"retain8={r['legacy_retain8']:.4f}",flush=True)
        except Exception as e:
            print(f"[redun] {mode} frac={fr}: eval failed {e}",flush=True)
        os.system(f"rm -rf {d}")
json.dump({"study":"redundancy_sweep","state":a.state,"arm":a.arm,"attacked":a.attacked,"concept":a.concept,
           "rows":rows,
           "meaning":"target accuracy of the ATTACKED model after removing a fraction of its "
                     "feed-forward weights. Measures how distributed the recovered concept is."},
          open(f"{OUT}/redundancy_{a.state}_{a.concept}.json","w"),indent=1)
