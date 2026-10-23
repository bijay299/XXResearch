"""FINDING44 S4's decisive follow-up: re-zero the pruned positions in an ATTACKED checkpoint and
re-measure the target. If the concept survives, the SURVIVING weights alone carry it and
FINDING44's corrected mechanism is direct rather than inferred."""
import argparse, json, os, torch
ADAPT="/data/bijaypandey/t2i_unlearn/adapters"; OUT="/data/bijaypandey/t2i_unlearn/outputs"
ap=argparse.ArgumentParser()
ap.add_argument("--arm",required=True); ap.add_argument("--attacked",required=True)
ap.add_argument("--label",required=True)
a=ap.parse_args()
pre=torch.load(f"{ADAPT}/{a.arm}/unet.pt",map_location="cpu"); pre=pre.get("unet",pre)
post=torch.load(a.attacked,map_location="cpu"); post=post.get("unet",post)
keys=sorted(k for k in pre if k.endswith(".weight") and ".ff.net." in k and ".transformer_blocks." in k)
n=0
out={k:v.clone() for k,v in post.items()}
for k in keys:
    m=(pre[k].float()==0)
    out[k]=post[k].clone(); out[k][m]=0
    n+=int(m.sum())
d=f"{ADAPT}/_rezero_{a.label}"; os.makedirs(d,exist_ok=True)
torch.save({k:v.half() for k,v in out.items()}, f"{d}/unet.pt")
json.dump({"study":"rezero_test","arm":a.arm,"attacked":a.attacked,
           "n_positions_rezeroed":n,"out":f"{d}/unet.pt",
           "meaning":"the ATTACKED model with FRP's original pruning mask re-applied. If the "
                     "target is still generated, the surviving weights alone carry the concept."},
          open(f"{OUT}/rezero_{a.label}.json","w"),indent=1)
print(f"[rezero] re-zeroed {n} positions -> {d}/unet.pt")
