# Full pretrained VLM + ContextFlow

This branch prioritizes `ContextFlow_pi05_full`. It keeps the complete pretrained
SigLIP + PaliGemma/Gemma-2B + action expert from `pi05_base` and adds the existing
ContextFlow demonstration projections and Perceiver compressors. The original
`ContextFlow` configuration and checkpoints remain available.

The question is whether context becomes more useful when the full pretrained VLM
is retained. The original small `gemma_300m_v2` prompt expert is not used here.
The new context modules need training; loading a base checkpoint alone is not a
trained context-conditioned policy.

## Architecture and initialization

- Upstream source is pinned in `src/openpi/models/full_pi/UPSTREAM.md`.
- Every backbone checkpoint leaf must match by name and shape. Missing backbone
  weights, unexpected weights and shape mismatches raise errors. Only new context
  modules and optional LoRA parameters may initialize randomly.
- Demonstration image, state and action tokens use the existing bidirectional
  ContextFlow prefix. Current image tokens are not pooled.
- π0.5 uses its native time MLP and adaptive RMSNorm, with current robot state
  discretized in the language prefix. Only the actual 8 LIBERO state dimensions
  are tokenized. The action expert does not receive the old continuous state token.
- Normalization uses quantiles for π0.5. Demonstrations use the same statistics
  as current states/actions; padding is restored to zero after normalization.
- The default is full fine-tuning: no backbone freeze and no LoRA. This increases
  memory relative to the original ContextFlow configuration.

## Matched experiment settings

The configuration preserves the original held-out task list, demonstration
sampling (8 frames / 128 actions), seed, action horizon (50), batch size (32),
20,000 training steps, optimizer and learning-rate schedule. Statistics and
checkpoints use a separate config directory. Use `pi05_base`, not a LIBERO-finetuned
checkpoint, for this comparison. This controls local fine-tuning exposure; it is
not evidence about which tasks were present in upstream pretraining.

This is not a reproduction of upstream's `pi05_libero` recipe, which uses different
training settings. Better results are an experimental hypothesis, not guaranteed.

## Environment

Run from this worktree. The existing dependency versions are retained. An existing
installation can be reused by setting `PYTHON` to its interpreter; `PYTHONPATH`
ensures the selected worktree's code is imported.

```bash
export PYTHON=/path/to/ContextFlow/.venv/bin/python
export PYTHONPATH="$PWD/src:$PWD:$PWD/packages/openpi-client/src"
export LEROBOT_HOME=/path/to/datasets
export OPENPI_DATA_HOME=/path/to/model-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=float32
```

Do not run `uv sync` against a shared environment while another experiment is
running. A separate environment is also supported by the existing project setup.

## Statistics and short validation

LIBERO actions in this configuration are used directly, so explicitly disable the
statistics script's historical delta-action default:

```bash
"$PYTHON" scripts/compute_norm_stats.py \
  --config-name ContextFlow_pi05_full --no-use-delta-joint-actions --local-files-only

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" scripts/validate_full_contextflow.py \
  --config-name ContextFlow_pi05_full \
  --params-path /path/to/pi05_base/params \
  --steps 3 --batch-size 4 --fsdp-devices 4 \
  --output logs/full_backbone_validation/pi05_smoke.json
```

Omit `--local-files-only` to allow dataset download. Omit `--params-path` to use the
registered official checkpoint URL. Short validation uses a repeated real batch,
the production AdamW training step, finite loss/gradient checks, and sampled actions
with/without demonstrations. It also compares masked-context inference against a
native reference sharing the trained backbone arrays. It saves a JSON report, not
a deployable checkpoint or optimizer state. It does not measure task success.

## Full training and evaluation

After short validation, launch a separate training experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON" scripts/train.py ContextFlow_pi05_full \
  --exp-name pi05_full_seed42 --batch-size 32 --fsdp-devices 8 \
  --num-train-steps 20000 --no-wandb-enabled

"$PYTHON" scripts/serve_policy.py --env LIBERO \
  policy:checkpoint --policy.config ContextFlow_pi05_full \
  --policy.dir checkpoints/ContextFlow_pi05_full/pi05_full_seed42/19999
```

Use the existing LIBERO evaluator in its simulator environment. For the paired
correct/no/wrong protocol, enable `--policy.context-ablation` on the server and run
`examples/libero/context_ablation.py`. Use identical task scenes, initial states,
replan settings and random seeds across models. Compare correct minus no-context
success as well as wrong-context behavior; overall success alone cannot establish
that demonstrations are being used effectively.

The separate attention-capture experiments in the uncommitted main worktree are
not part of this branch. The original paired context-ablation protocol is retained.
