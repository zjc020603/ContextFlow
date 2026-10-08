"""Read-only verification of the final trained checkpoint on a real LIBERO batch."""
import dataclasses
import datetime
import hashlib
import json
import logging
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as model_lib
from openpi.training import config as configs
from openpi.training import data_loader

logging.basicConfig(level=logging.INFO)
run = pathlib.Path(__file__).resolve().parent
manifest = json.loads((run / 'manifest.json').read_text())
checkpoint = pathlib.Path(manifest['expected_final_checkpoint'])
started = time.monotonic()
for item in ['params', 'train_state', 'assets']:
    assert (checkpoint / item).is_dir(), f'Missing checkpoint item {item}'
assert not any('orbax-checkpoint-tmp' in p.name for p in checkpoint.rglob('*'))
with ocp.PyTreeCheckpointer() as ckptr:
    step = ckptr.restore(
        checkpoint / 'train_state',
        args=ocp.args.PyTreeRestore(
            item={'step': np.zeros((), dtype=np.int32)},
            restore_args={'step': ocp.RestoreArgs(restore_type=np.ndarray)},
            transforms={'step': ocp.Transform(original_key='step')},
        ),
    )
optimizer_step = int(np.asarray(step['step']))
assert optimizer_step == 20000, optimizer_step
logging.info('Saved optimizer step verified: %s', optimizer_step)
norm_path = checkpoint / 'assets/physical-intelligence/libero/norm_stats.json'
norm_hash = hashlib.sha256(norm_path.read_bytes()).hexdigest()
assert norm_hash == manifest['normalization_sha256']
params = model_lib.restore_params(checkpoint / 'params', dtype=jnp.float32)
leaves = jax.tree.leaves(params)
parameter_count = sum(int(p.size) for p in leaves)
for i, leaf in enumerate(leaves):
    assert np.isfinite(np.asarray(leaf)).all(), f'Non-finite parameter leaf {i}'
logging.info('Restored and checked %s parameters in %s leaves', parameter_count, len(leaves))
cfg = configs.get_config('ContextFlow_pi05_full')
cfg = dataclasses.replace(
    cfg, batch_size=1, num_workers=0, fsdp_devices=1, wandb_enabled=False,
    assets_repo_override=str(checkpoint / 'assets'),
    data=dataclasses.replace(cfg.data, local_files_only=True, policy_local_files_only=True),
)
model = cfg.model.load(params, remove_extra_params=False)
model.eval()
loader = data_loader.create_custom_incontext_data_loader(cfg, num_workers=0, num_batches=1, shuffle=False)
obs, actions = next(iter(loader))
logging.info('Loaded real normalized LIBERO batch; validating loss and 10-step sampled actions')
loss_fn = nnx.jit(lambda m, o, a: m.compute_loss(jax.random.key(7), o, a, train=False))
loss = np.asarray(loss_fn(model, obs, actions))
assert np.isfinite(loss).all()
sample_fn = nnx.jit(lambda m, o: m.sample_actions(jax.random.key(7), o, num_steps=10))
predictions = np.asarray(sample_fn(model, obs))
assert predictions.shape == (1, 50, 32), predictions.shape
assert np.isfinite(predictions).all()
report = {
    'status': 'passed', 'finished_at': datetime.datetime.now().astimezone().isoformat(),
    'checkpoint': str(checkpoint), 'config': cfg.name,
    'optimizer_step': optimizer_step, 'parameter_leaves': len(leaves),
    'parameter_count': parameter_count, 'all_parameters_finite': True,
    'normalization_sha256': norm_hash,
    'real_data_loss': float(loss.mean()), 'sampled_action_shape': list(predictions.shape),
    'denoising_steps': 10, 'sampled_actions_finite': True,
    'action_min': float(predictions.min()), 'action_max': float(predictions.max()),
    'behavioral_success_rate_measured': False, 'seconds': time.monotonic() - started,
}
(run / 'final_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
