#!/usr/bin/env bash
set -euo pipefail
cd /data/zjc/workspace/ContextFlow/.worktrees/pi05-full
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LEROBOT_HOME=/data/zjc/workspace/datasets
export HF_DATASETS_CACHE=/data/zjc/workspace/datasets/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/data/zjc/workspace/models/openpi
export HF_HUB_OFFLINE=1
export PYTHONPATH=/data/zjc/workspace/ContextFlow/.worktrees/pi05-full/src:/data/zjc/workspace/ContextFlow/.worktrees/pi05-full:/data/zjc/workspace/ContextFlow/.worktrees/pi05-full/packages/openpi-client/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=float32
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec /data/zjc/workspace/ContextFlow/.venv/bin/python scripts/train.py ContextFlow_pi05_full --exp-name pi05_full_seed42_20260924_214930 --batch-size 32 --fsdp-devices 8 --num-train-steps 20000 --seed 42 --data.local-files-only --no-wandb-enabled --weight-loader.params-path /data/zjc/workspace/ContextFlow/.model-cache/openpi-assets/checkpoints/pi05_base/params
