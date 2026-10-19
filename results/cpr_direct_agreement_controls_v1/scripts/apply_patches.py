#!/usr/bin/env python3
"""Apply the small attributed patches to a FRESH pinned GEM checkout and save exact diffs.

Pristine text is always read from `git show <pin>:<path>` so the script is idempotent and the
diffs are against the pinned commit, never against an earlier edit. Each replacement must match
exactly once or the script raises. Nothing outside the tabulated edits is changed.
"""
import difflib
import os
import subprocess
import sys

ROOT = os.environ.get("CPRD2_ROOT", "/data/bijaypandey/cuig_pilot/cpr_direct_agreement_controls_v1")
SRC = os.path.join(ROOT, "GEM_src")
PIN = "c328fc6bec687efe60223498d1b6bed2e5fbcffd"
TAG = "cpr_direct_agreement_controls_v1"

EG = "gem/operators/erasure/erasure_gem.py"
BW = "gem/wrappers/base_pipeline_wrapper.py"
OP = "gem/operators/operator.py"
FA = "gem/wrappers/factory.py"
PA = "gem/common/prompt_augmentation.py"

WANDB_STUB = '''try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    class _WandbDisabled:  # pragma: no cover
        """Inert stand-in: W&B/telemetry is disabled for this study; real use raises."""
        def __getattr__(self, name):
            def _blocked(*a, **k):
                raise RuntimeError('W&B is disabled for this study (no telemetry)')
            return _blocked
    wandb = _WandbDisabled()
'''

P1_OLD = '''                # Calculate Distances
                dist_pos = torch.nn.functional.pairwise_distance(pred_flat, pos_flat, p=2)
                dist_neg = torch.nn.functional.pairwise_distance(pred_flat, neg_flat, p=2)

                target_loss = torch.relu(dist_pos - config.eta * dist_neg).sum()
'''
P1_NEW = '''                # [%(tag)s P1a] READ-ONLY capture hook: receives detached clones of the three
                # FP32 operands BEFORE the loss; its return value is ignored. It draws no random
                # numbers and cannot touch the live tensors.
                _cpr_capture = getattr(config, "cpr_capture_hook", None)
                if _cpr_capture is not None:
                    _cpr_capture(processed_index=config.max_step_budget - step_budget,
                                 global_step=global_step, batch_start=batch_start,
                                 pred=pred_flat.detach().clone(), anchor=pos_flat.detach().clone(),
                                 target=neg_flat.detach().clone())

                # [%(tag)s P1b] small attributed loss dispatch. No hook, or a hook that returns
                # None (the `native_gem` arm), executes the ORIGINAL native block below unchanged.
                _cpr_hook = getattr(config, "cpr_loss_hook", None)
                _cpr_per_sample = (_cpr_hook(pred_flat, pos_flat, neg_flat)
                                   if _cpr_hook is not None else None)
                if _cpr_per_sample is None:
                    # ---------- ORIGINAL NATIVE LOSS BLOCK, UNCHANGED ----------
                    # Calculate Distances
                    dist_pos = torch.nn.functional.pairwise_distance(pred_flat, pos_flat, p=2)
                    dist_neg = torch.nn.functional.pairwise_distance(pred_flat, neg_flat, p=2)

                    target_loss = torch.relu(dist_pos - config.eta * dist_neg).sum()
                    # -------- END ORIGINAL NATIVE LOSS BLOCK, UNCHANGED --------
                else:
                    target_loss = _cpr_per_sample.sum()
''' % {"tag": TAG}

P3_OLD = '''            optimizer.step()

            # Log to WandB
'''
P3_NEW = '''            optimizer.step()

            # [%(tag)s P3] logging/checkpoint hook only. Runs AFTER the native optimizer step;
            # read-only access to the wrapper/optimizer; never alters the loop, the budget, the sampler or the optimizer.
            _cpr_step_hook = getattr(config, "cpr_step_hook", None)
            if _cpr_step_hook is not None:
                _cpr_step_hook(global_step=global_step,
                               processed_timesteps=config.max_step_budget - step_budget,
                               step_budget_left=step_budget,
                               wrapper=wrapper, optimizer=optimizer)

            # Log to WandB
''' % {"tag": TAG}

P2_OLD = '''        latents = latents_per_step[-1]

        with torch.no_grad():
            # Unpack first for Flux
'''
P2_NEW = '''        latents = latents_per_step[-1]

        # [%(tag)s P2] OUTPUT-ONLY: skip the preview decode when the caller asked for all latents.
        # GEM's training loop calls this as `_, all_latents = ...` and DISCARDS the decoded image.
        # The skipped block draws no random numbers and never touches `latents_per_step`, the
        # sampler, the loss or the optimizer. Skips are counted on the wrapper (`_cpr_preview_skips`).
        if return_all_latents:
            self._cpr_preview_skips = getattr(self, "_cpr_preview_skips", 0) + 1
            return None, latents_per_step

        with torch.no_grad():
            # Unpack first for Flux
''' % {"tag": TAG}

QWEN_STUB = (
    "try:\n    from diffusers import QwenImagePipeline\nexcept ImportError:  # pragma: no cover\n"
    "    class QwenImagePipeline:  # sentinel: never matches an isinstance() check\n        pass\n")

EDITS = {
    EG: [(P1_OLD, P1_NEW), (P3_OLD, P3_NEW)],
    BW: [(P2_OLD, P2_NEW) for P2_OLD, P2_NEW in [(P2_OLD, P2_NEW)]] + [
        ("import wandb\n", "# [%s P5b] optional-import fix (W&B is only used behind use_wandb).\n" % TAG + WANDB_STUB),
        ("from diffusers import FluxPipeline, StableDiffusion3Pipeline, QwenImagePipeline\n",
         "# [%s P8] optional-import fix: `QwenImagePipeline` is absent from the pinned diffusers 0.30.2 and only\n"
         "# referenced for a model type this study never uses.\n"
         "from diffusers import FluxPipeline, StableDiffusion3Pipeline\n" % TAG + QWEN_STUB)],
    OP: [("import wandb\n", "# [%s P5a] optional-import fix (W&B is only used behind use_wandb).\n" % TAG + WANDB_STUB)],
    FA: [
        ("from huggingface_hub import list_repo_tree, RepoFolder\n",
         "# [%s P9] optional-import fix: `RepoFolder` is absent from the installed huggingface_hub; it is only used\n"
         "# when listing a remote checkpoint tree (never in this study, which uses a local snapshot path).\n"
         "from huggingface_hub import list_repo_tree\n"
         "try:\n    from huggingface_hub import RepoFolder\nexcept ImportError:  # pragma: no cover\n"
         "    class RepoFolder:  # sentinel: never matches an isinstance() check\n        pass\n" % TAG),
        ("from diffusers import Flux2Pipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, QwenImagePipeline\n",
         "# [%s P6] optional-import fix: `Flux2Pipeline` is absent from the pinned diffusers and only\n"
         "# referenced for a model type this study never uses.\n"
         "from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline\n"
         "try:\n    from diffusers import Flux2Pipeline\nexcept ImportError:  # pragma: no cover\n"
         "    class Flux2Pipeline:  # sentinel: never matches an isinstance() check\n        pass\n" % TAG
         + "# [%s P8] optional-import fix for `QwenImagePipeline` (absent from diffusers 0.30.2).\n" % TAG + QWEN_STUB),
        ("def get_base_model_name_or_path(model_type):\n",
         "def get_base_model_name_or_path(model_type):\n"
         "    # [%s P7] PATH FIX ONLY: the pristine read-only FLUX cache holds the pinned snapshot\n"
         "    # but no `refs/main`; an explicit local snapshot PATH may be supplied per model type via\n"
         "    # CPRD_BASE_MODEL_PATH__<MODEL_TYPE>. Same pinned bytes, different lookup path.\n"
         "    override = os.environ.get(f\"CPRD_BASE_MODEL_PATH__{str(model_type).upper()}\")\n"
         "    if override:\n        return override\n" % TAG),
    ],
    PA: [("from datasets import load_dataset\n",
          "# [%s P4] optional-import fix: `datasets` backs HFDatasetPromptAugmentation only (unused here).\n"
          "try:\n    from datasets import load_dataset\nexcept ModuleNotFoundError:  # pragma: no cover\n"
          "    def load_dataset(*a, **k):\n        raise ModuleNotFoundError('HFDatasetPromptAugmentation needs `datasets`')\n" % TAG)],
}


def pristine(path):
    return subprocess.check_output(["git", "-C", SRC, "show", f"{PIN}:{path}"]).decode()


def main():
    outdir = os.path.join(ROOT, "patches")
    os.makedirs(outdir, exist_ok=True)
    head = subprocess.check_output(["git", "-C", SRC, "rev-parse", "HEAD"]).decode().strip()
    assert head == PIN, head
    all_diff = []
    for path, edits in EDITS.items():
        text = pristine(path)
        for old, new in edits:
            if text.count(old) != 1:
                raise RuntimeError(f"{path}: anchor not found exactly once: {old[:60]!r}")
            text = text.replace(old, new)
        open(os.path.join(SRC, path), "w").write(text)
        d = "".join(difflib.unified_diff(pristine(path).splitlines(True), text.splitlines(True),
                                         f"a/{path}", f"b/{path}"))
        name = path.replace("/", "__") + ".diff"
        open(os.path.join(outdir, name), "w").write(d)
        all_diff.append(d)
        print("patched", path, f"({d.count(chr(10))} diff lines)")
    open(os.path.join(outdir, "ALL_PATCHES.diff"), "w").write("".join(all_diff))
    # the real wrapper imports `os` already? make sure factory has it
    assert "import os" in open(os.path.join(SRC, FA)).read()


if __name__ == "__main__":
    sys.exit(main())
