"""Minimal, documented instrumentation for the IMMA author-budget run (PREREG item 1).

WHY THIS EXISTS. The budget JSON *calculates* 500 expected ascent updates from
(epochs x samples / batch); it never *measures* how many were executed. "Expected" and "executed"
are different quantities, and the preregistration's acceptance criterion is the executed one. A
skipped step -- a non-finite loss, a gradient-accumulation boundary, an early break -- would leave
the calculation unchanged and the evidence silent.

HOW IT AVOIDS TOUCHING UPSTREAM. `upstream/IMMA` is byte-unmodified. This file is a
`sitecustomize` module: CPython imports it automatically at interpreter startup when its directory
is on PYTHONPATH, so the launcher enables it with one environment variable and nothing else. It
registers a GLOBAL optimizer post-step hook
(`torch.optim.optimizer.register_optimizer_step_post_hook`, present in torch 2.4.1), which fires
after every `Optimizer.step()` in the process regardless of which object owns it. No upstream call
site is patched and no upstream file is read or rewritten.

HOW THE TWO OPTIMIZERS ARE DISTINGUISHED. `defend_text_to_image_lora.py` runs two:

    optimizer_lora   inner, DESCENT over the LoRA parameters, lr 1e-4  (README --learning_rate_lora)
    optimizer        outer, ASCENT  via backward(-loss) over attention, lr 1e-5  (README --learning_rate)

The ascent optimizer is the immunisation itself, and its count is the one the budget claims. They
are told apart by the learning rate recorded in `param_groups` at the moment of the step, which is
read from the live object rather than assumed: the hook records the lr it actually saw, so if the
two ever collide or a scheduler moves one, the artifact shows that instead of mislabelling.
`lr_scheduler` is `constant` per the README, so no drift is expected -- but it is measured, not
trusted.

WHAT IT RECORDS. Per distinguished optimizer: executed step count, the lr(s) observed, the number of
trainable tensors in its param_groups, and the first/last wall-clock step times. Process-level: the
python executable, argv, torch version, and whether any step raised. Written atomically to
$IMMA_EVIDENCE_OUT at interpreter exit (and again every 100 steps, so a killed run still leaves
partial evidence rather than none).

WHAT IT DOES NOT DO. It does not alter any update, loss, gradient, seed or ordering; the hook is
post-step and returns None. It does not infer correctness from magnitudes -- update magnitudes are
diagnostics here, not evidence of the budget (PREREG item 3).
"""
import atexit
import json
import os
import sys
import time

# PER-PROCESS OUTPUT PATH. One shared mutable file was the wrong design and a real run proved it:
# `accelerate launch` loads this module in the LAUNCHER, which then spawns the trainer as a
# SUBPROCESS. Both imported this module and both pointed at the same file; the launcher outlives
# the worker, so its atexit dump -- recording its own zero steps -- overwrote the worker's counts.
# The run then failed with "ZERO ascent updates executed" while a 374 MB checkpoint sat on disk.
# A clobber guard could not save it either, because at launcher-import time there was nothing yet
# to protect. So: every process writes its OWN file, `<base>.<pid>.json`, and the launcher-side
# verifier aggregates across them and reports which pid contributed what. No shared mutable state,
# no ordering dependency, nothing to clobber.
_BASE = os.environ.get("IMMA_EVIDENCE_OUT")
_OUT = (_BASE[:-5] if _BASE and _BASE.endswith(".json") else _BASE) + ".%d.json" % os.getpid() \
    if _BASE else None

if _OUT:
    _state = {
        "schema": "imma_executed_updates_v1",
        "python": sys.executable,
        "argv": list(sys.argv),
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "optimizers": {},          # key -> record
        "total_steps_observed": 0,
        "hook_errors": [],
        "note": ("Counts are EXECUTED Optimizer.step() calls observed by a global post-step hook. "
                 "They are not derived from epochs x samples. 'expected' values live in the "
                 "budget JSON and are compared by the launcher, not here."),
    }

    def _key_for(opt):
        """Label an optimizer by its class and the lr it is actually using right now."""
        lrs = sorted({g.get("lr") for g in getattr(opt, "param_groups", []) if "lr" in g})
        lr = lrs[0] if len(lrs) == 1 else None
        cls = type(opt).__name__
        # The two IMMA optimizers are separated by lr: 1e-5 ascent (the immunisation), 1e-4 LoRA.
        if lr is not None and abs(lr - 1e-5) < 1e-12:
            role = "ascent_outer_attention"
        elif lr is not None and abs(lr - 1e-4) < 1e-12:
            role = "descent_inner_lora"
        else:
            role = "unclassified"
        return f"{role}|{cls}|lr={lr}", role, cls, lrs

    def _n_params(opt):
        n = 0
        for g in getattr(opt, "param_groups", []):
            for p in g.get("params", []):
                n += 1
        return n

    def _hook(opt, args, kwargs):
        try:
            key, role, cls, lrs = _key_for(opt)
            rec = _state["optimizers"].get(key)
            now = time.time()
            if rec is None:
                rec = {"role": role, "class": cls, "lrs_observed": lrs,
                       "n_param_tensors": _n_params(opt),
                       "executed_steps": 0, "first_step_epoch": now, "last_step_epoch": now,
                       "id": id(opt)}
                _state["optimizers"][key] = rec
            rec["executed_steps"] += 1
            rec["last_step_epoch"] = now
            for v in lrs:
                if v not in rec["lrs_observed"]:
                    rec["lrs_observed"].append(v)
            _state["total_steps_observed"] += 1
            if _state["total_steps_observed"] % 100 == 0:
                _dump()
        except Exception as e:                      # never let instrumentation break training
            if len(_state["hook_errors"]) < 20:
                _state["hook_errors"].append(repr(e))
        return None

    def _dump():
        _state["dumped_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        tmp = _OUT + ".tmp"
        try:
            os.makedirs(os.path.dirname(_OUT) or ".", exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(_state, f, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, _OUT)                   # atomic
        except Exception:
            pass

    try:
        from torch.optim.optimizer import register_optimizer_step_post_hook
        register_optimizer_step_post_hook(_hook)
        _state["hook_installed"] = True
    except Exception as e:
        _state["hook_installed"] = False
        _state["hook_install_error"] = repr(e)

    # ---------------------------------------------------------------- PREREG item 3
    # RECORD THE IN-MEMORY STATE AT THE SAVE POINT. The previous check compared the reloaded file
    # against the ESD initialization and asserted rel > 1e-5. That establishes "something moved";
    # it does NOT establish that the file on disk corresponds to the state the trainer actually
    # held when it saved. Those are different claims, and only the second one makes a reload
    # meaningful.
    # So wrap torch.save: fingerprint every tensor in the object being written, AT THE MOMENT it is
    # written, from the live in-memory object. The launcher then reloads the file and compares
    # against these fingerprints. Nothing about the save is altered -- the original function is
    # called with the original arguments and its return value is passed through.
    def _fingerprint_obj(obj, max_tensors=4096):
        import torch as _t
        out = {}
        def walk(o, prefix=""):
            if len(out) >= max_tensors:
                return
            if isinstance(o, _t.Tensor):
                try:
                    # CONTIGUOUS FIRST, THEN A BYTE HASH. The first version recorded a float64
                    # sum and the verifier demanded it match bit-exactly. A real run showed that is
                    # the wrong primitive: 13 of 160 tensors "mismatched" by ~2e-16 RELATIVE -- one
                    # float64 ulp -- because the in-memory tensor came from CUDA with its own
                    # strides while the reloaded one is contiguous, and float reduction ORDER
                    # differs between the two layouts. Same dtype, same shape, same values.
                    # A content hash of the canonicalised bytes is exact, order-independent and
                    # layout-independent, which is what "is this the same tensor" actually needs.
                    # The float summaries are kept as human-readable diagnostics only.
                    import hashlib as _h
                    d = o.detach().to("cpu").contiguous()
                    out[prefix] = {
                        "shape": list(d.shape), "dtype": str(d.dtype),
                        "numel": int(d.numel()),
                        "sha256": _h.sha256(d.numpy().tobytes()).hexdigest(),
                        "sum_f64": float(d.double().sum().item()),        # diagnostic only
                        "absmax_f64": float(d.double().abs().max().item()) if d.numel() else 0.0,
                        "finite": bool(_t.isfinite(d).all().item()) if d.is_floating_point() else True,
                    }
                except Exception as e:
                    out[prefix] = {"error": repr(e)}
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}")
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    walk(v, f"{prefix}[{i}]")
        walk(obj)
        return out

    try:
        import torch as _torch
        _real_save = _torch.save

        def _save_with_evidence(obj, f, *a, **kw):
            rec = {"called_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "target": str(f) if isinstance(f, (str, bytes, os.PathLike)) else repr(type(f))}
            try:
                fp = _fingerprint_obj(obj)
                rec["n_tensors_in_memory"] = len(fp)
                rec["tensors"] = fp
            except Exception as e:
                rec["fingerprint_error"] = repr(e)
            _state.setdefault("torch_saves", []).append(rec)
            _dump()
            return _real_save(obj, f, *a, **kw)

        _torch.save = _save_with_evidence
        _state["torch_save_wrapped"] = True
    except Exception as e:
        _state["torch_save_wrapped"] = False
        _state["torch_save_wrap_error"] = repr(e)

    # DO NOT CLOBBER A RICHER RECORD. Belt and braces for the plumbing bug above: if this module
    # is somehow loaded by a second process pointed at the same file, it must not overwrite a
    # record that already contains observed steps. Losing collected evidence is worse than having
    # none, because the loss looks exactly like "training did nothing".
    def _existing_steps():
        try:
            with open(_OUT) as f:
                return int(json.load(f).get("total_steps_observed", 0) or 0)
        except Exception:
            return 0

    _PRIOR = _existing_steps()
    if _PRIOR > 0 and os.environ.get("IMMA_EVIDENCE_ALLOW_CLOBBER") != "1":
        _state["refused_to_clobber"] = {
            "existing_total_steps_observed": _PRIOR,
            "this_pid": os.getpid(), "this_argv": list(sys.argv),
            "why": ("another process already recorded observed steps into this path. Overwriting "
                    "would destroy collected evidence and look like a zero-step run."),
        }
        def _dump():                                  # noqa: F811  -- deliberately neutered
            return None

    atexit.register(_dump)
    _dump()                                          # a file exists from the start, even if killed
