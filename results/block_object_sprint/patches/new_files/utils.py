"""Restores the two helpers the release imports from `scripts.utils`
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
