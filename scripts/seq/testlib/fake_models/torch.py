"""A fake `torch`, for CPU tests of generate_eval_images.py only.

It exists so the generator's REAL reuse branch can be exercised end to end
without a GPU, a model, or the real torch. It is never importable from a
production run: it lives under scripts/seq/testlib/ and reaches a process only
when a test puts that directory on PYTHONPATH.

Deliberately minimal and loud: anything the generator asks of torch that is not
implemented raises, rather than silently returning a plausible value.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CALLS = Path(os.environ.get("FAKE_MODEL_CALL_LOG", "/dev/null"))

float16 = "fake.float16"
float32 = "fake.float32"


def _log(kind: str, **kw) -> None:
    try:
        with CALLS.open("a") as fh:
            fh.write(json.dumps({"call": kind, **kw}) + "\n")
    except Exception:
        pass


class _Device:
    def __init__(self, spec: str):
        self.type = "cpu" if "cpu" in spec else spec
        self.spec = spec

    def __repr__(self) -> str:
        return f"fake.device({self.spec!r})"


def device(spec: str) -> _Device:
    return _Device(spec)


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def max_memory_allocated() -> int:
        raise AssertionError("fake torch: CUDA memory queried in a CPU test")

    @staticmethod
    def get_device_name(_i: int) -> str:
        raise AssertionError("fake torch: CUDA device name queried in a CPU test")


cuda = _Cuda()


class _Tensor:
    """Just enough of a tensor for the generator's `.to(dtype)` call.

    Carries the digest of the checkpoint file it came from, so the fake
    pipeline's output can depend on WHICH checkpoint was loaded. Without that,
    two different checkpoints would yield byte-identical fake images and a test
    could not tell a regeneration from a relabelled reuse.
    """

    def __init__(self, key: str, ckpt_digest: str = ""):
        self.key = key
        self.ckpt_digest = ckpt_digest

    def to(self, _dtype):
        return self


class Generator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = int(seed)
        _log("generator.manual_seed", seed=self.seed)
        return self


def load(path, map_location=None, weights_only=False):
    """Return a state dict shaped like a delta.bin, without decoding real tensors.

    The file's bytes are hashed but never interpreted as tensors: these tests
    assert control flow and provenance, not numerics. The digest is carried into
    every tensor so the fake pipeline's output depends on which checkpoint was
    loaded.
    """
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    _log("torch.load", path=str(path), digest=h[:12])
    n = int(os.environ.get("FAKE_DELTA_TENSORS", "32"))
    return {"unet": {f"fake.attn2.to_k.{i}.weight": _Tensor(str(i), h)
                     for i in range(n)}}
