#!/usr/bin/env python3
"""
Applies the MINIMAL integration fixes catalogued in notes/INTEGRATION_AUDIT.md to the
author release checked out at pin db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5.

Rules followed:
  * import / helper / path / pipeline-return / dtype-plumbing fixes only;
  * the transcoder training objective and the W_dec erasure arithmetic are NOT altered
    (the erasure block is only MOVED into a function so it can be called from a harness,
     with its expressions copied character-for-character);
  * every change is recorded by `git diff` against the pin.
Idempotent: re-running is a no-op.
"""
import os, sys, io

REPO = "/data/bijaypandey/cuig_pilot/block_object_sprint/code/block-concept-removal"
n_applied = 0


def patch(relpath, subs, must_exist=True):
    global n_applied
    p = os.path.join(REPO, relpath)
    if not os.path.exists(p):
        if must_exist:
            raise SystemExit(f"missing {p}")
        return
    s = open(p).read()
    orig = s
    for old, new, tag in subs:
        if new in s and old not in s:
            print(f"  [skip ] {relpath}: {tag} (already applied)")
            continue
        if old not in s:
            raise SystemExit(f"  [FAIL ] {relpath}: anchor not found for {tag}\n---\n{old[:300]}")
        s = s.replace(old, new, 1)
        print(f"  [apply] {relpath}: {tag}")
        n_applied += 1
    if s != orig:
        open(p, "w").write(s)


def write(relpath, content, tag):
    global n_applied
    p = os.path.join(REPO, relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and open(p).read() == content:
        print(f"  [skip ] {relpath}: {tag} (already present)")
        return
    open(p, "w").write(content)
    print(f"  [apply] {relpath}: {tag}")
    n_applied += 1


# ---------------------------------------------------------------- A5: missing scripts/utils.py
write("scripts/utils.py", '''"""Restores the two helpers the release imports from `scripts.utils`
but never ships (integration gap A5). Behaviour inferred from the call sites:
  setup_logger(log_dir)        -> file + console logging, same format as the other scripts
  get_module_by_name(root, "a.b.c") -> attribute walk to a submodule
"""
import logging
from pathlib import Path


def setup_logger(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return
    logging.basicConfig(
        filename=str(Path(log_dir) / "run.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root.addHandler(console)


def get_module_by_name(root, name: str):
    mod = root
    for part in name.split("."):
        if part:
            mod = getattr(mod, part)
    return mod
''', "A5 create scripts/utils.py")

write("constants/__init__.py", "", "A17 package marker (constants)")
write("scripts/__init__.py", "", "A17 package marker (scripts)")
write("scripts/transcoder/__init__.py", "", "A17 package marker (scripts.transcoder)")
write("scripts/unlearn_canvas/__init__.py", "", "A17 package marker (scripts.unlearn_canvas)")

# ------------------------------------------------- A1/A2/A3: get_activation_diffusion.py
patch("scripts/transcoder/get_activation_diffusion.py", [
    # A1: context_embedder was bound but never called; ctx_out was used undefined.
    ("""        B, T, _ = prompt_embeds.shape
        assert attn.shape == (B, T) and attn.dtype == torch.bool

        valid_in_list, valid_out_list = [], []""",
     """        B, T, _ = prompt_embeds.shape
        assert attn.shape == (B, T) and attn.dtype == torch.bool

        # [PATCH A1] the release binds `context_embedder` (above) but never calls it, so
        # `ctx_out` below was an undefined name. The transcoder's training target is the
        # ORIGINAL module's output on the same input, per paper 4.2.
        with torch.no_grad():
            ctx_out = context_embedder(prompt_embeds)

        valid_in_list, valid_out_list = [], []""",
     "A1 compute ctx_out from the real context_embedder"),
    ("""    parser.add_argument("--output_path", type=str, required=True, help="")""",
     """    parse.add_argument("--output_path", type=str, required=True, help="")  # [PATCH A2] was `parser`""",
     "A2 parser->parse"),
    ("""        out_dir = os.path.join(output_path, label)""",
     """        out_dir = os.path.join(args.output_path, label)  # [PATCH A3] was unbound global""",
     "A3 args.output_path"),
])

# ------------------------------------------------------- A2/A3: get_activation_diffusion_mlp.py
patch("scripts/transcoder/get_activation_diffusion_mlp.py", [
    ("""    parser.add_argument("--output_path", type=str, required=True, help="")""",
     """    parse.add_argument("--output_path", type=str, required=True, help="")  # [PATCH A2] was `parser`""",
     "A2 parser->parse"),
    ("""        out_dir = os.path.join(output_path, label)""",
     """        out_dir = os.path.join(args.output_path, label)  # [PATCH A3] was unbound global""",
     "A3 args.output_path"),
    # make the COCO prompt file path explicit & overridable (file is absent from the release, gap A4)
    ("""    with open("prompts/coco_train2014_30k_prompts.txt", "r",) as prompt_file:""",
     """    # [PATCH A4] this file is referenced by the release but NOT shipped in it; it is
    # reconstructed by code/build_coco_assets.py (seeded MS-COCO train2014 sample).
    with open(os.environ.get("COCO_PROMPTS", "prompts/coco_train2014_30k_prompts.txt"), "r",) as prompt_file:""",
     "A4 overridable COCO prompt path"),
])

# ------------------------------------------------------------- A6/A7: diffusion_baseline.py
patch("scripts/unlearn_canvas/diffusion_baseline.py", [
    ("""from scripts.transcoder.model_inference import Transcoder
from scripts.transcoder.model_inference_ca import TranscoderCA
""",
     """from scripts.transcoder.model_inference import Transcoder
# [PATCH A6] `scripts.transcoder.model_inference_ca` does not exist in the release and
# TranscoderCA is unused in this file.
""",
     "A6 drop missing TranscoderCA import"),
    ("""    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

def main():""",
     """    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    return pipe.to(device)  # [PATCH A7] the release returned None and never moved to device

def main():""",
     "A7 return pipe.to(device)"),
])

# ----------------------------------------------------------------- A15/A16: model_inference.py
patch("scripts/transcoder/model_inference.py", [
    ("""    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        :param x: input data (shape: (B, T, n_inputs) or (N, n_inputs))
        :return: output data (same rank with last dim = n_outputs)
        \"\"\"
        orig_dtype = x.dtype
""",
     """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        :param x: input data (shape: (B, T, n_inputs) or (N, n_inputs))
        :return: output data (same rank with last dim = n_outputs)
        \"\"\"
        orig_dtype = x.dtype
        # [PATCH A15] The transcoder is built/loaded in fp32 while the FLUX pipeline runs in
        # bf16, so the release raised a dtype RuntimeError in nn.Linear. The file already
        # intends an input-dtype-preserving contract (it captures orig_dtype and casts the
        # output), so we complete it: compute in the PARAMETER dtype, return in the input
        # dtype. No change to the arithmetic.
        x = x.to(self.encoder.weight.dtype)
""",
     "A15 cast input to parameter dtype"),
    ("""    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "Transcoder":
        # model shape and expansion factor""",
     """    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "Transcoder":
        # [PATCH A16] the release pops from the caller's dict, so loading the same state
        # dict twice failed; work on a shallow copy instead.
        state_dict = dict(state_dict)
        # model shape and expansion factor""",
     "A16 do not mutate caller's state_dict"),
])

# --------------------------------------------------------------------- A8/A9/B6: train.py
patch("scripts/transcoder/train.py", [
    ("""        x_path = os.path.join(root_dir, f"layer{layer}_in", fname)
        y_path = os.path.join(root_dir, f"layer{layer}_out", fname)""",
     """        # [PATCH A9] the activation collectors write `<layer>_in` / `<layer>_out`
        # (e.g. context_embedder_in), not `layer<layer>_in`.
        x_path = os.path.join(root_dir, f"{layer}_in", fname)
        y_path = os.path.join(root_dir, f"{layer}_out", fname)""",
     "A9 activation path template"),
    ("""    parser.add_argument( "--dead_threshold", type=int, required=True, help="dead_threshold")""",
     """    parser.add_argument( "--dead_threshold", type=int, required=True, help="dead_threshold")
    # [PATCH B6] paper E.1 pins batch size per module (16384 projection / 1024 MLP) but the
    # release exposes no flag and its default (4096) matches neither.
    parser.add_argument( "--batch_size", type=int, required=True, help="paper E.1 batch size")
    parser.add_argument( "--save_name", type=str, required=True, help="checkpoint basename, e.g. context_embedder")
    parser.add_argument( "--run_name", type=str, required=True, help="explicit run dir name")""",
     "B6 expose batch_size + explicit save/run names"),
    ("""    cfg.dead_feature_threshold = args.dead_threshold
    cfg.layers = args.layers""",
     """    cfg.dead_feature_threshold = args.dead_threshold
    cfg.layers = args.layers
    cfg.batch_size = args.batch_size      # [PATCH B6]
    cfg.log_to_wandb = False              # [PATCH A10] no external services this sprint""",
     "B6/A10 wire batch_size, disable wandb"),
    ("""    # Initialize and launch the trainer
    trainer = TranscoderTrainer(cfg, dataset_dict)
    trainer.fit()

    # Save model checkpoints
    trainer.save()""",
     """    # [PATCH A8] TranscoderTrainer treats its argument as a Dataset (len(), [0],
    # random_split), but the release handed it this dict -> len()==#layers and dataset[0]
    # raised KeyError. We train one layer per invocation and pass that Dataset.
    if len(args.layers) != 1:
        raise SystemExit("train one layer per invocation (see PATCH A8)")
    trainer = TranscoderTrainer(cfg, dataset_dict[args.layers[0]])
    trainer.fit()

    # [PATCH A12/A13] save() with no args wrote `None.pt` into a freshly incremented run
    # dir; diffusion.py loads `context_embedder.pt` / `text_embedder.pt`.
    trainer.save(typ=args.save_name, name=args.run_name)""",
     "A8/A12/A13 single dataset + explicit save"),
])

# ------------------------------------------------------------------- A10/A11: trainer.py
patch("scripts/transcoder/trainer.py", [
    ("""        if cfg.log_to_wandb:

            wandb.login(key=os.environ["WANDB_TOKEN"])""",
     """        if cfg.log_to_wandb:

            wandb.login(key=os.environ["WANDB_TOKEN"])  # [PATCH A10] only when enabled""",
     "A10 note on wandb login"),
    ("""                    info_eval[f"val_fvu"] = val_fvu

                wandb.log(info_eval, step=steps)""",
     """                    info_eval[f"val_fvu"] = val_fvu

                    # [PATCH A10] this log call sat OUTSIDE the log_to_wandb guard in the
                    # release, so it crashed whenever W&B was disabled. Also mirror the
                    # per-epoch validation numbers to the normal logger so a local run
                    # still produces a training record.
                    wandb.log(info_eval, step=steps)

                self.logger.info(
                    "epoch end: step=%d val_fvu=%.6f dead_frac=%.6f",
                    steps, float(val_fvu),
                    float((transcoder.activation_frequency.cpu().numpy() < 1).mean()),
                )
                self.val_fvu_history.append(float(val_fvu))""",
     "A10 guard wandb.log + local epoch logging"),
    ("""        self.cfg = cfg
        self.dataset = dataset
""",
     """        self.cfg = cfg
        self.dataset = dataset
        self.val_fvu_history = []   # [PATCH A10] local training record when W&B is off
""",
     "A10 val history field"),
])

print(f"\n{n_applied} change(s) applied.")
