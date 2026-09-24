# Upstream provenance

Source: https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479

`gemma.py`, `pi0.py`, and `pi0_config.py` come from that commit.
Local adaptations: imports use this namespace, Pi0 accepts the existing
ObservationIncontext type, and preprocessing is an overridable method.
The backbone parameter names and numerical operations are retained.
Existing SigLIP, LoRA and sharding modules are shared with this repository.

This separate namespace preserves the original ContextFlow prompt expert and
its checkpoint compatibility. Do not replace missing pretrained backbone
parameters with random values; FullBackboneWeightLoader checks them strictly.
