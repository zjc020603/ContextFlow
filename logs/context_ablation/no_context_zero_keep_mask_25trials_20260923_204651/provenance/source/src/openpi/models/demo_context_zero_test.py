import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.contextflow import make_attn_mask
from openpi.models.demo_context_zero import zero_demo_blocks


def test_all_demo_modalities_removed_without_changing_observation_or_positions():
    kinds = ["current_image", "language", "demo_image_latents", "demo_states", "demo_actions"]
    layout = {"blocks": [{"kind": k, "start": 2 * i, "stop": 2 * i + 2} for i, k in enumerate(kinds)]}
    tokens = jnp.arange(80, dtype=jnp.float32).reshape(1, 10, 8)
    mask = jnp.ones((1, 10), bool).at[:, 5].set(False)
    ar = jnp.zeros(10, bool)
    fn = jax.jit(lambda t: zero_demo_blocks((t, mask, ar), layout))
    changed, new_mask, new_ar = fn(tokens)
    np.testing.assert_array_equal(changed[:, :4], tokens[:, :4])
    np.testing.assert_array_equal(changed[:, 4:], 0)
    np.testing.assert_array_equal(fn(tokens.at[:, 4:].set(-999))[0], changed)
    np.testing.assert_array_equal(new_mask, mask)
    np.testing.assert_array_equal(jnp.cumsum(new_mask, -1), jnp.cumsum(mask, -1))
    np.testing.assert_array_equal(make_attn_mask(new_mask, new_ar), make_attn_mask(mask, ar))
    assert changed.shape == tokens.shape
    assert changed.dtype == tokens.dtype
    assert np.any(np.asarray(tokens[:, 4:]) != 0)
    with pytest.raises(ValueError, match="No demonstration"):
        zero_demo_blocks((tokens, mask, ar), {"blocks": []})
