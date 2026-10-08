import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.contextflow import make_attn_mask
from openpi.models.demo_action_ablation import zero_action_tail


def test_zero_action_preserves_other_tokens_masks_positions_and_input():
    tokens = jax.random.normal(jax.random.key(0), (2, 14, 8))
    mask = jnp.ones((2, 14), bool).at[:, 3].set(False)
    ar_mask = jnp.zeros(14, bool)
    before = np.asarray(tokens).copy()
    changed, new_mask, new_ar = jax.jit(lambda t, m, a: zero_action_tail(t, m, a, 3))(tokens, mask, ar_mask)
    np.testing.assert_array_equal(changed[:, :-3], tokens[:, :-3])
    np.testing.assert_array_equal(changed[:, -3:], 0)
    np.testing.assert_array_equal(tokens, before)
    np.testing.assert_array_equal(new_mask, mask)
    np.testing.assert_array_equal(new_ar, ar_mask)
    np.testing.assert_array_equal(jnp.cumsum(new_mask, axis=-1), jnp.cumsum(mask, axis=-1))
    np.testing.assert_array_equal(make_attn_mask(mask, ar_mask), make_attn_mask(new_mask, new_ar))
    assert changed.shape == tokens.shape
    assert changed.dtype == tokens.dtype


def test_action_input_cannot_change_zeroed_output():
    tokens = jnp.arange(56, dtype=jnp.float32).reshape(1, 7, 8)
    mask = jnp.ones((1, 7), bool)
    ar = jnp.zeros(7, bool)
    other = tokens.at[:, -2:].set(-1000)
    a = zero_action_tail(tokens, mask, ar, 2)[0]
    b = zero_action_tail(other, mask, ar, 2)[0]
    np.testing.assert_array_equal(a, b)
    for count in (0, -1, 8):
        with pytest.raises(ValueError, match="Invalid demo action block size"):
            zero_action_tail(tokens, mask, ar, count)
