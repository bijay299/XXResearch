# Repair cycle 2 (post-freeze) — GPU-gate sampling-mode assertion counted the wrong thing

* Context: first `entry_gpu` phase (2026-10-19T18:40Z) ran detector qualification (GPU1) and the native GPU gate trajectory (GPU0) concurrently.
  Both exited non-zero: `qualify` exit 4 (a genuine qualification failure, see ENTRY report; NOT repairable) and `train.dev.native_gem.lr_1e-3` exit 3 (GPU gate).
* GPU-gate failure: all checks passed except `student_sampling_train_mode_no_grad`. Its recorded fields were `transformer_training=true`,
  `grad_enabled=false`, `lora_dropout_modules_training=true`, `n_lora_dropout_modules=152` against the asserted 76.
* Diagnosis (harness defect, not a native behaviour deviation): the instrument matched every module whose NAME contains `lora_dropout`, i.e. both the
  `lora_dropout` ModuleDict container and the `lora_dropout.default` nn.Dropout inside it, giving 2 x 76 layers = 152. The asserted intent (76 dropout
  layers, p=0.1, in training mode during student sampling, `no_grad`) was satisfied. A second diagnostic field, `adapter_disabled_flag`, read diffusers'
  `PeftAdapterMixin.disable_adapters` METHOD (always truthy) and was misleading; it was never part of an assertion and is replaced by a count of LoRA layers whose
  `disable_adapters` attribute is literally True (diagnostic only, recorded).
* Patch (additive commit before the retest): count only `torch.nn.Dropout` instances in `lora_dropout` containers (assertion `== 76` unchanged);
  replace the diagnostic field; make the orchestrator report qualification pass/fail from the recorded summary (it previously would have reported True whenever a summary existed);
  add the `gpu_gate` phase that reruns only the GPU gate trajectory. No objective, fixture, tolerance, category, threshold, matrix or decision rule changed.
* Failed run preserved under `artifacts/failed_attempts/gpu_gate_run1/` (train record, per-state log, captured triplets, adapters); its charged GPU time stays in the ledger.
* Retest: one rerun of the GPU gate trajectory (`train.dev.native_gem.lr_1e-3.r2`), same seeds.
