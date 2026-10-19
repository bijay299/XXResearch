"""Seed streams per configs/eval_protocol.json `seeds.derivation`, and the FREEZE guard.

streams(family, seed) -> dict name -> np.random.Generator, from
    SeedSequence(entropy=[20261019, FAMILY_CODE, seed]).spawn(8)
in the protocol's stream order. Development seeds 0..99 are always allowed. Validation seeds 7000..7011 are
REFUSED unless configs/FREEZE.json exists and its recorded sha256 of configs/eval_protocol.json,
configs/candidate.json and the code/vau tree equal the current files. Every other seed is refused.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .contracts import ContractViolation

ROOT_ENTROPY = 20261019
FAMILY_CODES = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6, "G7": 7, "G8": 8, "GD_B": 21, "GD_D": 22}
STREAMS = ("fixture", "splits", "train", "sketch", "gate", "eval_noise", "population", "spare")
DEV = range(0, 100)
VALIDATION = range(7000, 7012)

_SPRINT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FREEZE_PATH = os.path.join(_SPRINT, "configs", "FREEZE.json")


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def vau_tree_hash(root=None):
    """sha256 over (relative path, file sha256) of every *.py file under code/vau, sorted by path."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for dp, _, fns in sorted(os.walk(root)):
        for fn in sorted(fns):
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                h.update(os.path.relpath(p, root).encode() + b"\0" + _sha_file(p).encode() + b"\n")
    return h.hexdigest()


def freeze_ok(freeze_path=FREEZE_PATH):
    """True iff FREEZE.json exists and matches the current protocol, candidate and code/vau tree."""
    if not os.path.exists(freeze_path):
        return False
    with open(freeze_path) as f:
        fz = json.load(f)
    cfg = os.path.join(_SPRINT, "configs")
    try:
        return (fz["sha256_eval_protocol"] == _sha_file(os.path.join(cfg, "eval_protocol.json"))
                and fz["sha256_candidate"] == _sha_file(os.path.join(cfg, "candidate.json"))
                and fz["sha256_code_vau_tree"] == vau_tree_hash())
    except (KeyError, FileNotFoundError):
        return False


def check_seed(seed, freeze_path=FREEZE_PATH):
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ContractViolation(f"seed must be an int, got {seed!r}")
    seed = int(seed)
    if seed in DEV:
        return seed
    if seed in VALIDATION:
        if freeze_ok(freeze_path):
            return seed
        raise ContractViolation(f"validation seed {seed} refused: configs/FREEZE.json missing or not matching")
    raise ContractViolation(f"seed {seed} is neither a development (0..99) nor a validation (7000..7011) seed")


def streams(family, seed, freeze_path=FREEZE_PATH):
    if family not in FAMILY_CODES:
        raise ContractViolation(f"unknown family {family!r}")
    seed = check_seed(seed, freeze_path)
    ss = np.random.SeedSequence(entropy=[ROOT_ENTROPY, FAMILY_CODES[family], seed])
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, ss.spawn(8))}
