import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vau import tasks  # noqa: E402
from vau.contracts import dev_rng  # noqa: E402

FAMILIES = ("linear", "mlp", "flow")


def make(family, seed=0, **kw):
    rng = dev_rng(seed)
    t = tasks.make_task(family, rng, **kw)
    return t, rng


def perturbed_theta(task, rng, scale=0.3):
    """theta with nonzero B (so every LoRA block has a nonzero gradient)."""
    th = task.model.init_theta(rng)
    return th + scale * rng.normal(size=th.shape)


def central_fd(f, x, h=1e-6):
    """Central differences of scalar or array-valued f at x (FP64). Returns (*f.shape, P)."""
    f0 = np.asarray(f(x))
    out = np.empty(f0.shape + (x.size,))
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        out[..., i] = (np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * h)
    return out


@pytest.fixture(params=FAMILIES)
def family(request):
    return request.param
