import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.contextflow import make_attn_mask
from openpi.models.demo_modality_ablation import MODALITIES
from openpi.models.demo_modality_ablation import zero_modalities


@pytest.mark.parametrize("variant", ["state", "state_action"])
def test_selective_intervention_preserves_all_other_content_and_positions(variant):
    kinds = ["current_image", "language", "demo_image_latents", "demo_states", "demo_actions"]
    layout = {"blocks": [{"kind": k, "start": i * 2, "stop": i * 2 + 2} for i, k in enumerate(kinds)]}
    original = jnp.arange(80, dtype=jnp.float32).reshape(1, 10, 8)
    mask = jnp.ones((1, 10), bool).at[:, 5].set(False)
    ar = jnp.zeros(10, bool)
    zero = jax.jit(lambda t: zero_modalities((t, mask, ar), layout, MODALITIES[variant]))
    changed, new_mask, new_ar = zero(original)
    altered = original
    for block in layout["blocks"]:
        start, stop = block["start"], block["stop"]
        if block["kind"] in MODALITIES[variant]:
            np.testing.assert_array_equal(changed[:, start:stop], 0)
            altered = altered.at[:, start:stop].set(-913)
        else:
            np.testing.assert_array_equal(changed[:, start:stop], original[:, start:stop])
    np.testing.assert_array_equal(zero(altered)[0], changed)
    np.testing.assert_array_equal(new_mask, mask)
    np.testing.assert_array_equal(jnp.cumsum(new_mask, -1), jnp.cumsum(mask, -1))
    np.testing.assert_array_equal(make_attn_mask(new_mask, new_ar), make_attn_mask(mask, ar))
    np.testing.assert_array_equal(original, jnp.arange(80).reshape(1, 10, 8))
    assert changed.shape == original.shape
    assert changed.dtype == original.dtype


def test_reject_missing_or_non_demo_blocks():
    prefix = (jnp.ones((1, 2, 3)), jnp.ones((1, 2), bool), jnp.zeros(2, bool))
    with pytest.raises(ValueError, match="missing"):
        zero_modalities(prefix, {"blocks": []}, ("demo_states",))
    with pytest.raises(ValueError, match="Only explicit"):
        zero_modalities(prefix, {"blocks": []}, ("current_image",))
