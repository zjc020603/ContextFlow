import sys
import jax
import jax.numpy as jnp
import torch
import flax.nnx as nnx
from openpi.models.contextflow import PerceiverCompressor
from openpi.training.config import get_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print('Python:', sys.version)
print('JAX:', jax.__version__, jax.devices())
assert jax.default_backend() == 'gpu'
a = jnp.ones((64, 64), dtype=jnp.float32)
assert float(jax.jit(lambda x: (x @ x).sum())(a).block_until_ready()) == 262144.0
print('JAX GPU JIT matrix multiply: PASS')
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available()
t = torch.ones((64, 64), device='cuda')
assert (t @ t).sum().item() == 262144.0
print('PyTorch GPU matrix multiply: PASS')
model = PerceiverCompressor(num_queries=4, embed_dim=64, num_heads=4, num_layers=2, rngs=nnx.Rngs(0))
y = model(jnp.ones((1, 8, 64)))
assert y.shape == (1, 4, 64) and bool(jnp.isfinite(y).all())
print('ContextFlow Perceiver forward:', y.shape, 'PASS')
config = get_config('ContextFlow')
print('ContextFlow config:', config.name, 'batch_size:', config.batch_size)
print('LeRobot dataset import: PASS')
