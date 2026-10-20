"""mdmu: CPU reference substrate for masked-diffusion-LM unlearning experiments (llada_method_design_v1).

Modules
- toyworld:  synthetic benign fact world (fresh fictional entities), tokenisation, splits, key-token sets.
- model:     tiny bidirectional masked transformer + LoRA adapters.
- diffusion: LLaDA SFT forward process / loss, LLaDA-faithful native sampler port, MC likelihood.
"""
