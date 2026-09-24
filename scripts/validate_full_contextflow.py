"""Short real-data validation of full-backbone ContextFlow; not a success-rate evaluation."""

import dataclasses
import functools
import hashlib
import json
import logging
import pathlib
import time

import flax.nnx as nnx
import jax
import numpy as np
import tyro

from openpi.models.full_pi.pi0 import Pi0
from openpi.training import config as configs
from openpi.training import sharding
from openpi.training.full_backbone_weights import CONTEXT_ROOTS
from openpi.training.full_backbone_weights import FullBackboneWeightLoader
from scripts import train


def main(
    config_name: str = "ContextFlow_pi05_full",
    *,
    params_path: str | None = None,
    steps: int = 3,
    batch_size: int = 4,
    fsdp_devices: int = 4,
    output: pathlib.Path = pathlib.Path("logs/full_backbone_validation/smoke.json"),
):
    if steps < 1:
        raise ValueError("steps must be positive")
    logging.basicConfig(level=logging.INFO)
    cfg = configs.get_config(config_name)
    if params_path is not None:
        cfg = dataclasses.replace(cfg, weight_loader=FullBackboneWeightLoader(params_path))
    cfg = dataclasses.replace(
        cfg,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=0,
        wandb_enabled=False,
        data=dataclasses.replace(
            cfg.data, base_config=dataclasses.replace(cfg.data.base_config, local_files_only=True)
        ),
    )
    if batch_size % jax.device_count():
        raise ValueError("batch_size must be divisible by visible JAX device count")
    jax.config.update("jax_threefry_partitionable", True)  # noqa: FBT003
    mesh = sharding.make_mesh(fsdp_devices)
    ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = train._create_train_data_loader(cfg, sharding=ds, num_workers=0, shuffle=True)  # noqa: SLF001
    batch = next(iter(loader))
    rng = jax.random.key(cfg.seed)
    state, state_sharding = train.init_train_state(cfg, rng, mesh, resume=False)
    jax.block_until_ready(state)
    logging.info("Full checkpoint loaded; compiling a real-data optimizer step")
    step_fn = jax.jit(
        functools.partial(train.train_step, cfg),
        in_shardings=(replicated, state_sharding, ds),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    metrics = []
    for index in range(steps):
        start = time.monotonic()
        with sharding.set_mesh(mesh):
            state, info = step_fn(rng, state, batch)
        jax.block_until_ready(info)
        record = {k: float(v) for k, v in info.items()}
        if not all(np.isfinite(v) for v in record.values()):
            raise ValueError(f"Non-finite training metrics: {record}")
        record.update(step=index, seconds=time.monotonic() - start)
        metrics.append(record)
        logging.info("smoke step: %s", record)

    # Reuse actual trained backbone arrays in a native reference, without copying
    # or randomly reinitializing them. Verify no-context parity after training.
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    native = nnx.eval_shape(lambda: Pi0(cfg.model, rngs=nnx.Rngs(0)))
    native_state = nnx.state(native, nnx.Param)
    params = state.params.to_pure_dict()
    native_state.replace_by_pure_dict({k: v for k, v in params.items() if k not in CONTEXT_ROOTS})
    nnx.update(native, native_state)
    native.eval()
    obs, _ = batch
    masked = dataclasses.replace(
        obs,
        incontext_image_masks=jax.tree.map(np.zeros_like, obs.incontext_image_masks),
        incontext_state_masks=np.zeros_like(obs.incontext_state_masks),
        incontext_action_masks=np.zeros_like(obs.incontext_action_masks),
    )
    # Native preprocessing ignores demonstrations; the live observations are identical.
    key = jax.random.key(7)
    sample = nnx.jit(lambda m, o: m.sample_actions(key, o, num_steps=2))
    with sharding.set_mesh(mesh):
        native_actions = np.asarray(sample(native, masked))
        no_context_actions = np.asarray(sample(model, masked))
        context_actions = np.asarray(sample(model, obs))
    assert np.all(np.isfinite(context_actions))
    assert np.all(np.isfinite(no_context_actions))
    np.testing.assert_allclose(no_context_actions, native_actions, atol=0.03, rtol=0.03)
    report = {
        "config": config_name,
        "checkpoint": cfg.weight_loader.params_path,
        "steps": metrics,
        "optimizer_step": int(state.step),
        "batch_size": batch_size,
        "fsdp_devices": fsdp_devices,
        "repeated_real_batch": True,
        "sample_frames": cfg.model.sample_frames,
        "sample_actions": cfg.model.sample_actions,
        "action_horizon": cfg.model.action_horizon,
        "no_context_native_max_abs_error": float(np.max(np.abs(no_context_actions - native_actions))),
        "context_action_mean_abs_change": float(np.mean(np.abs(context_actions - no_context_actions))),
        "training_episode_count": len(loader.data_config().train_episode),
        "training_episode_sha256": hashlib.sha256(json.dumps(loader.data_config().train_episode).encode()).hexdigest(),
        "normalization_sha256": hashlib.sha256(
            (cfg.assets_dirs / loader.data_config().asset_id / "norm_stats.json").read_bytes()
        ).hexdigest(),
        "behavioral_success_rate_measured": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Validation complete: %s", output)


if __name__ == "__main__":
    tyro.cli(main)
