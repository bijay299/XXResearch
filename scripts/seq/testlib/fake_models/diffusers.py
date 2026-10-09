"""A fake `diffusers`, for CPU tests of generate_eval_images.py only.

Every pipeline construction and every generation call is appended to
$FAKE_MODEL_CALL_LOG, so a test can assert exactly how many images the generator
actually produced -- and, for a refused reuse, that the pipeline was never
constructed at all.

See the note in this directory's fake torch: production code can never import
this, it is only reachable when a test puts this directory on PYTHONPATH.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CALLS = Path(os.environ.get("FAKE_MODEL_CALL_LOG", "/dev/null"))


def _log(kind: str, **kw) -> None:
    try:
        with CALLS.open("a") as fh:
            fh.write(json.dumps({"call": kind, **kw}) + "\n")
    except Exception:
        pass


class _Image:
    """An image stand-in writing deterministic bytes.

    The bytes depend on the prompt, the image seed AND the loaded checkpoint's
    digest, so a test can distinguish a genuine regeneration from old bytes
    relabelled as a new checkpoint's output.
    """

    def __init__(self, prompt: str, seed: int, model: str):
        self.prompt, self.seed, self.model = prompt, seed, model

    def save(self, path, quality=None):
        Path(path).write_bytes(
            b"FAKE-IMAGE\n"
            + f"{self.model}|{self.prompt}|{self.seed}".encode() + b"\n")


class _Out:
    def __init__(self, images):
        self.images = images


class _Unet:
    class _LoadResult:
        unexpected_keys: list = []
        missing_keys: list = []

    def __init__(self):
        self.fingerprint = "base"      # no delta applied

    def load_state_dict(self, sd, strict=True):
        digests = {getattr(t, "ckpt_digest", "") for t in sd.values()}
        self.fingerprint = sorted(digests)[0] if digests else "unknown"
        _log("unet.load_state_dict", n_tensors=len(sd), strict=strict,
             digest=self.fingerprint[:12])
        return self._LoadResult()


class StableDiffusionPipeline:
    def __init__(self, base_model_dir: str):
        self.base_model_dir = base_model_dir
        self.unet = _Unet()
        self.safety_checker = None

    @classmethod
    def from_pretrained(cls, base_model_dir, torch_dtype=None):
        if os.environ.get("FAKE_PIPELINE_MUST_NOT_LOAD"):
            raise AssertionError(
                "fake diffusers: a pipeline was constructed in a case that must "
                "fail before any model setup")
        _log("pipeline.from_pretrained", base_model_dir=str(base_model_dir))
        return cls(str(base_model_dir))

    def to(self, device):
        _log("pipeline.to", device=repr(device))
        return self

    def set_progress_bar_config(self, **kw):
        return None

    def __call__(self, prompt=None, width=None, height=None,
                 num_inference_steps=None, guidance_scale=None, generator=None):
        seed = getattr(generator, "seed", None)
        _log("pipeline.generate", prompt=prompt, image_seed=seed,
             num_inference_steps=num_inference_steps,
             guidance_scale=guidance_scale, width=width, height=height,
             model=self.unet.fingerprint[:12])
        return _Out([_Image(prompt or "", seed or 0, self.unet.fingerprint)])
