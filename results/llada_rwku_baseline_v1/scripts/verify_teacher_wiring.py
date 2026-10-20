"""Inference-only check of the anchor/student wiring (§4 memory strategy).

Verifies the EQUIVALENCE half: a freshly-initialized LoRA adapter is a no-op
(B=0), and PEFT's disable_adapter() restores pristine base logits exactly.
The INVARIANCE-AFTER-UPDATES half is NOT verified here: no optimizer step was
taken anywhere in this study (the §3 stop condition forbade training).
"""
import json, sys, torch
sys.path.insert(0, "/data/bijaypandey/llada_rwku/scripts")
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer
MD="/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
tok=AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
base=AutoModel.from_pretrained(MD, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
ids=tok("Stephen King is an American", add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
with torch.no_grad(): pristine=base(ids).logits.float().clone()
m=get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","attn_out"], init_lora_weights=True)).eval()
with torch.no_grad(): enabled=m(ids).logits.float().clone()
with m.disable_adapter():
    with torch.no_grad(): disabled=m(ids).logits.float().clone()
d_en=(enabled-pristine).abs().max().item(); d_dis=(disabled-pristine).abs().max().item()
r={"fresh_adapter_is_noop_max_abs_diff":d_en,"disabled_equals_pristine_max_abs_diff":d_dis,
   "fresh_adapter_is_noop":d_en==0.0,"disable_adapter_restores_pristine":d_dis==0.0,
   "equivalence_verified":bool(d_en==0.0 and d_dis==0.0),
   "teacher_invariance_after_updates_verified":False,
   "why_invariance_not_verified":"No optimizer step was taken in this study: pristine recall was INSUFFICIENT_BASE_RECALL, so the protocol required stopping without training.",
   "method":"identical input; compare pristine base logits vs adapter-enabled and vs PEFT disable_adapter(), fp32 max-abs"}
json.dump(r, open("/data/bijaypandey/llada_rwku/artifacts/teacher_wiring.json","w"), indent=2)
print(json.dumps(r, indent=2))
