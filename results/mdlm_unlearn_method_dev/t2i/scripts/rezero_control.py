"""Null control for FINDING46: re-zero a RANDOM matched-count set of FF weights in the attacked
checkpoint instead of FRP's mask. If the target also stays at 1.000, then re-zeroing ANY 3% of
feed-forward weights is harmless, and FRP's mask is not special -- which is what makes the main
result a statement about redundancy rather than about that mask."""
import argparse, json, os, torch
ADAPT="/data/bijaypandey/t2i_unlearn/adapters"; OUT="/data/bijaypandey/t2i_unlearn/outputs"
ap=argparse.ArgumentParser()
ap.add_argument("--arm",required=True); ap.add_argument("--attacked",required=True)
ap.add_argument("--label",required=True); ap.add_argument("--seed",type=int,default=1234)
a=ap.parse_args()
pre=torch.load(f"{ADAPT}/{a.arm}/unet.pt",map_location="cpu"); pre=pre.get("unet",pre)
post=torch.load(a.attacked,map_location="cpu"); post=post.get("unet",post)
keys=sorted(k for k in pre if k.endswith(".weight") and ".ff.net." in k and ".transformer_blocks." in k)
g=torch.Generator().manual_seed(a.seed)
out={k:v.clone() for k,v in post.items()}; n=0
for k in keys:
    m=(pre[k].float()==0); kk=int(m.sum())
    if kk==0: continue
    flat=post[k].numel()
    idx=torch.randperm(flat,generator=g)[:kk]          # matched COUNT, random POSITIONS
    w=out[k].clone().reshape(-1); w[idx]=0; out[k]=w.reshape(post[k].shape)
    n+=kk
d=f"{ADAPT}/_rezeroctl_{a.label}"; os.makedirs(d,exist_ok=True)
torch.save({k:v.half() for k,v in out.items()}, f"{d}/unet.pt")
json.dump({"study":"rezero_random_control","arm":a.arm,"n_positions_rezeroed":n,"seed":a.seed,
           "out":f"{d}/unet.pt",
           "meaning":"matched COUNT, RANDOM positions. Brackets the FRP-mask re-zero: if this also "
                     "leaves the target at 1.000, re-zeroing any 3% of FF weights is harmless and "
                     "the main result is about redundancy, not about FRP's particular mask."},
          open(f"{OUT}/rezeroctl_{a.label}.json","w"),indent=1)
print(f"[ctl] re-zeroed {n} RANDOM positions -> {d}/unet.pt")
