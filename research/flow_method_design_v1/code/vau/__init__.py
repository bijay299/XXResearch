"""VAU (Verified Adam-metric Update) numpy CPU prototype -- flow_method_design_v1, Agent E.

Modules: contracts (errors, op counters, dev seeds), adamw, losses, models, tasks, projections, methods.
"""
from .contracts import ContractViolation, Ops, dev_rng  # noqa: F401
