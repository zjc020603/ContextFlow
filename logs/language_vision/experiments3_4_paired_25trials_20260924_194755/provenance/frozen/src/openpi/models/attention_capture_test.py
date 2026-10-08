from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models.attention_capture import call_with_attention


def small_llm():
    config = gemma.Config(width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16)
    module = bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="float32", voc_size=128), rngs=nnx.Rngs(0))
    module.lazy_init(
        [jnp.ones((1, 3, 32)), jnp.ones((1, 2, 32))], positions=jnp.arange(5)[None], mask=jnp.ones((1, 5, 5), bool)
    )
    return module


def test_capture_preserves_outputs_parameters_and_masks():
    module = small_llm()
    xs = [jnp.ones((1, 3, 32)), jnp.ones((1, 2, 32))]
    positions = jnp.arange(5)[None]
    mask = jnp.ones((1, 5, 5), bool).at[:, :, 1].set(False)
    before = nnx.state(module, nnx.Param).to_pure_dict()
    regular = module(xs, positions=positions, mask=mask)
    recorded, probabilities = call_with_attention(module, xs, positions, mask, None)
    for a, b in zip(jax.tree.leaves(regular), jax.tree.leaves(recorded), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(nnx.state(module, nnx.Param).to_pure_dict()), strict=True):
        np.testing.assert_array_equal(a, b)
    assert probabilities.shape == (2, 1, 2, 5, 5)
    np.testing.assert_array_equal(probabilities[..., 1], 0)
    np.testing.assert_allclose(probabilities.sum(-1), 1, atol=2e-7)
    assert not hasattr(module, "attention")


def test_capture_handles_cached_prefix_without_mutating_cache():
    module = small_llm()
    _, cache = module([jnp.ones((1, 3, 32)), None], positions=jnp.arange(3)[None], mask=jnp.ones((1, 3, 3), bool))
    inputs = [None, jnp.ones((1, 2, 32))]
    positions = jnp.array([[3, 4]])
    mask = jnp.ones((1, 2, 5), bool)
    regular = module(inputs, positions=positions, mask=mask, kv_cache=cache)
    recorded, probabilities = call_with_attention(module, inputs, positions, mask, cache)
    for a, b in zip(jax.tree.leaves(regular), jax.tree.leaves(recorded), strict=True):
        np.testing.assert_array_equal(a, b)
    assert probabilities.shape == (2, 1, 2, 2, 5)
    assert cache[0].shape[2] == 3


def test_full_sampling_loop_capture_is_exact(monkeypatch):
    import flax.struct

    from openpi.models import contextflow
    from openpi.models import model as model_lib
    from openpi.models.attention_capture import sample_actions_with_attention
    from openpi.shared.nnx_utils import module_jit

    @flax.struct.dataclass
    class Observation:
        state: jax.Array
        valid: jax.Array

    class ToyContext(nnx.Module):
        def __init__(self):
            self.PaliGemma = nnx.Dict(llm=small_llm())
            self.action_horizon = 2
            self.action_dim = 4
            self.action_in = nnx.Linear(4, 32, rngs=nnx.Rngs(1))
            self.action_out_proj = nnx.Linear(32, 4, rngs=nnx.Rngs(2))

        def embed_midfix(self, obs):
            return jnp.ones((1, 3, 32)), obs.valid, jnp.array([False, False, False])

        def embed_suffix(self, obs, actions, time):
            tokens = jnp.concatenate([jnp.ones((1, 1, 32)), self.action_in(actions)], axis=1)
            return tokens, jnp.ones((1, 3), bool), jnp.array([True, True, False])

        def ordinary(self, rng, obs):
            return contextflow.ContextFlow.sample_actions(self, rng, obs, num_steps=3)

        def captured(self, rng, obs):
            return sample_actions_with_attention(self, rng, obs, num_steps=3, capture_steps=(0, 2), layers=(0, 1))

    monkeypatch.setattr(model_lib, "preprocess_observation_incontext", lambda rng, obs, train: obs)
    model = ToyContext()
    ordinary = module_jit(model.ordinary)
    captured = module_jit(model.captured)
    obs = Observation(jnp.zeros((1, 4)), jnp.array([[True, False, True]]))
    rng = jax.random.key(8)
    baseline = ordinary(rng, obs)
    actual, trace = captured(rng, obs)
    np.testing.assert_array_equal(baseline, actual)
    assert trace["probabilities"].shape == (2, 2, 1, 2, 3, 6)
    assert int(trace["executed_steps"]) == 3
    np.testing.assert_array_equal(trace["probabilities"][..., 1], 0)


def test_token_layout_uses_rectangular_patch_dimensions_and_retains_masked_slots():
    from types import SimpleNamespace

    from openpi.models.attention_capture import token_layout

    model = SimpleNamespace(
        _image_patch_size=(14, 16),
        avg_current_img=False,
        use_text_prompts=True,
        use_image_prompts=True,
        use_action_state_prompts=True,
        num_image_queries=4,
        num_state_queries=3,
        num_action_queries=2,
        state_compressor=object(),
        action_compressor=object(),
        action_horizon=5,
    )
    obs = SimpleNamespace(
        images={"front": np.zeros((1, 28, 48, 3))},
        tokenized_prompt=np.zeros((1, 8)),
        incontext_images={"front": np.zeros((1, 1, 2, 28, 48, 3))},
        incontext_states=np.zeros((1, 1, 2, 4)),
        incontext_actions=np.zeros((1, 1, 2, 4)),
    )
    layout = token_layout(model, obs)
    assert [(b["start"], b["stop"]) for b in layout["blocks"]] == [
        (0, 6),
        (6, 14),
        (14, 18),
        (18, 21),
        (21, 23),
        (23, 24),
        (24, 29),
    ]
    assert layout["prefix_size"] == 23
    assert layout["key_size"] == 29
    assert layout["query_layout"]["predicted_actions"] == [1, 6]
