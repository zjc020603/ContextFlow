"""Numerical contracts for full pretrained backbones plus demonstrations."""

import dataclasses

import flax.linen as linen
import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as observation_model
from openpi.models.full_contextflow import FullContextFlowConfig
from openpi.models.full_pi import gemma
from openpi.models.full_pi.pi0 import Pi0
from openpi.models.full_pi.pi0 import make_attn_mask
from openpi.training.full_backbone_weights import is_new_parameter
from openpi.training.full_backbone_weights import merge_full_backbone


class TinyVision(linen.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @linen.compact
    def __call__(self, image, *, train=False):
        return linen.Dense(self.num_classes)(jnp.mean(image, axis=(1, 2)))[:, None, :], {}


@pytest.fixture
def tiny(monkeypatch):
    from openpi.models import siglip
    from openpi.models.full_contextflow import FullContextFlow

    monkeypatch.setattr(siglip, "Module", TinyVision)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    identity = staticmethod(lambda rng, observation, **kwargs: observation)
    monkeypatch.setattr(Pi0, "preprocess_observation", identity)
    monkeypatch.setattr(FullContextFlow, "preprocess_observation", identity)
    return {
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "dtype": "float32",
        "action_dim": 4,
        "action_horizon": 3,
        "max_token_len": 4,
        "num_image_queries": 2,
        "num_state_queries": 2,
        "num_action_queries": 2,
        "num_image_compressor_layers": 2,
        "num_state_compressor_layers": 2,
        "num_action_compressor_layers": 2,
    }


def observations(*, enabled=True):
    return observation_model.ObservationIncontext(
        images={k: jnp.ones((1, 14, 14, 3)) for k in observation_model.IMAGE_KEYS},
        image_masks={k: jnp.ones((1,), dtype=bool) for k in observation_model.IMAGE_KEYS},
        state=jnp.ones((1, 4)),
        tokenized_prompt=jnp.array([[1, 2, 3, 0]]),
        tokenized_prompt_mask=jnp.array([[True, True, True, False]]),
        incontext_images={k: jnp.ones((1, 2, 14, 14, 3)) for k in observation_model.IMAGE_KEYS},
        incontext_image_masks={k: jnp.full((1, 2), enabled) for k in observation_model.IMAGE_KEYS},
        incontext_states=jnp.ones((1, 3, 4)),
        incontext_state_masks=jnp.full((1, 3), enabled),
        incontext_actions=jnp.ones((1, 3, 4)) * 0.5,
        incontext_action_masks=jnp.full((1, 3), enabled),
    )


def activate_adaptive_gates(model):
    # Upstream initializes adaptive residual gates to zero; real pretrained
    # weights have learned gates. Activate them in this tiny synthetic fixture.
    state = nnx.state(model, nnx.Param)
    flat = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    for key, value in flat.items():
        if "norm_1/Dense_0/bias" in key:
            flat[key] = value.at[..., -value.shape[-1] // 3 :].set(1)
    state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(flat, sep="/"))
    nnx.update(model, state)


@pytest.mark.parametrize("pi05", [True, False])
def test_masked_context_matches_native_loss_and_sampling(tiny, pi05):
    config = FullContextFlowConfig(**tiny, pi05=pi05)
    model = config.create(jax.random.key(0))
    native = Pi0(config, rngs=nnx.Rngs(1))
    activate_adaptive_gates(model)
    native_state = nnx.state(native, nnx.Param)
    native_roots = native_state.to_pure_dict()
    full = nnx.state(model, nnx.Param).to_pure_dict()
    native_state.replace_by_pure_dict({k: full[k] for k in native_roots})
    nnx.update(native, native_state)
    obs = observations(enabled=False)
    actions = jnp.ones((1, 3, 4)) * 0.2
    key = jax.random.key(4)
    np.testing.assert_allclose(
        model.compute_loss(key, obs, actions), native.compute_loss(key, obs, actions), rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        model.sample_actions(key, obs, num_steps=2), native.sample_actions(key, obs, num_steps=2), rtol=2e-5, atol=2e-5
    )
    # Masked demonstration values must have no effect.
    changed = dataclasses.replace(obs, incontext_actions=obs.incontext_actions * 100)
    np.testing.assert_allclose(
        model.sample_actions(key, obs, num_steps=2),
        model.sample_actions(key, changed, num_steps=2),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("pi05", [True, False])
def test_cached_action_pass_matches_joint_forward(tiny, pi05):
    model = FullContextFlowConfig(**tiny, pi05=pi05).create(jax.random.key(0))
    activate_adaptive_gates(model)
    obs = observations()
    prefix, pmask, par = model.embed_prefix(obs)
    suffix, smask, sar, cond = model.embed_suffix(obs, jnp.ones((1, 3, 4)), jnp.array([0.4]))
    mask = jnp.concatenate([pmask, smask], axis=1)
    ar = jnp.concatenate([par, sar])
    (_, joint), _ = model.PaliGemma.llm(
        [prefix, suffix],
        positions=jnp.cumsum(mask, axis=1) - 1,
        mask=make_attn_mask(mask, ar),
        adarms_cond=[None, cond],
    )
    _, cache = model.PaliGemma.llm(
        [prefix, None],
        positions=jnp.cumsum(pmask, axis=1) - 1,
        mask=make_attn_mask(pmask, par),
    )
    cross_mask = jnp.broadcast_to(pmask[:, None, :], (1, suffix.shape[1], prefix.shape[1]))
    (_, cached), _ = model.PaliGemma.llm(
        [None, suffix],
        positions=jnp.sum(pmask, axis=1)[:, None] + jnp.cumsum(smask, axis=1) - 1,
        mask=jnp.concatenate([cross_mask, make_attn_mask(smask, sar)], axis=-1),
        adarms_cond=[None, cond],
        kv_cache=cache,
    )
    np.testing.assert_allclose(joint, cached, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("pi05", [True, False])
def test_context_gets_gradients_and_changes_prediction(tiny, pi05):
    model = FullContextFlowConfig(**tiny, pi05=pi05).create(jax.random.key(0))
    activate_adaptive_gates(model)
    obs = observations()
    loss, grads = nnx.value_and_grad(
        lambda m: jnp.mean(m.compute_loss(jax.random.key(2), obs, jnp.zeros((1, 3, 4)), train=True))
    )(model)
    assert np.isfinite(loss)
    leaves = jax.tree.leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in leaves)
    for root in ("demo_action_proj", "demo_state_proj", "image_compressor", "state_compressor", "action_compressor"):
        assert sum(float(jnp.sum(jnp.abs(g))) for g in jax.tree.leaves(grads[root])) > 0
    changed = dataclasses.replace(obs, incontext_actions=obs.incontext_actions * -5)
    before = model.sample_actions(jax.random.key(3), obs, num_steps=2)
    after = model.sample_actions(jax.random.key(3), changed, num_steps=2)
    assert not np.allclose(before, after)


def test_full_backbone_loader_rejects_missing_or_wrong_weights():
    template = {
        "PaliGemma": {"kernel": np.zeros((2, 3), np.float32)},
        "demo_action_proj": {"kernel": np.zeros((4, 3), np.float32)},
    }
    loaded = {"PaliGemma": {"kernel": np.ones((2, 3), np.float32)}}
    merged = merge_full_backbone(loaded, template)
    np.testing.assert_array_equal(merged["PaliGemma"]["kernel"], 1)
    np.testing.assert_array_equal(merged["demo_action_proj"]["kernel"], 0)
    with pytest.raises(ValueError, match="missing"):
        merge_full_backbone({}, template)
    with pytest.raises(ValueError, match="shape_mismatch"):
        merge_full_backbone({"PaliGemma": {"kernel": np.zeros((2, 4))}}, template)
    with pytest.raises(ValueError, match="unexpected"):
        merge_full_backbone({**loaded, "state_proj": np.zeros(2)}, template)


@pytest.mark.parametrize("pi05", [True, False])
def test_production_parameter_tree_preserves_every_native_weight(pi05):
    config = FullContextFlowConfig(pi05=pi05)
    full = nnx.eval_shape(config.create, jax.random.key(0))
    native = nnx.eval_shape(lambda: Pi0(config, rngs=nnx.Rngs(0)))
    ref = nnx.state(full, nnx.Param).to_pure_dict()
    pretrained = nnx.state(native, nnx.Param).to_pure_dict()
    # Shape-only loading still exercises all production parameter names/shapes.
    flat_ref = flax.traverse_util.flatten_dict(ref, sep="/")
    flat_pretrained = flax.traverse_util.flatten_dict(pretrained, sep="/")
    assert set(flat_pretrained) <= set(flat_ref)
    assert all(flat_ref[k].shape == v.shape for k, v in flat_pretrained.items())
    assert all(is_new_parameter(k) for k in flat_ref.keys() - flat_pretrained.keys())
    assert not any("prompt_expert" in k for k in flat_ref)
