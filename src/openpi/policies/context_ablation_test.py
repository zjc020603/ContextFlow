import jax
import numpy as np
import pytest

from openpi.policies.context_ablation import mask_demonstration
from openpi.policies.context_ablation import prepare_request


def test_wrong_changes_demo_selection_but_preserves_language_and_live_inputs():
    image = np.ones((4, 4, 3))
    inputs = {"task_index": 25, "prompt": "milk", "split": "test", "image": image}
    correct, _, correct_rng = prepare_request(
        inputs,
        {
            "mode": "correct",
            "rng_components": [0, 25, 1, 3],
        },
    )
    wrong, info, wrong_rng = prepare_request(
        inputs,
        {
            "mode": "wrong",
            "demo_task_index": 28,
            "rng_components": [0, 25, 1, 3],
        },
    )
    assert inputs["task_index"] == correct["task_index"] == 25
    assert wrong["task_index"] == 28
    assert wrong["prompt"] == "milk"
    assert wrong["image"] is image
    assert info["task_index"] == 25
    np.testing.assert_array_equal(jax.random.key_data(correct_rng), jax.random.key_data(wrong_rng))
    _, _, next_rng = prepare_request(inputs, {"mode": "no", "rng_components": [0, 25, 2, 3]})
    assert not np.array_equal(jax.random.key_data(correct_rng), jax.random.key_data(next_rng))


def test_no_context_masks_every_demo_modality_without_mutating_cache_or_language():
    inputs = {
        "dem_prompt_images": {"base": np.ones((2, 4, 4, 3))},
        "dem_prompt_images_mask": {"base": np.ones(2, dtype=bool)},
        "dem_prompt_all_states": np.ones((8, 7)),
        "dem_prompt_all_states_mask": np.ones(8, dtype=bool),
        "dem_prompt_all_actions": np.ones((8, 7)),
        "dem_prompt_all_actions_mask": np.ones(8, dtype=bool),
        "tokenized_prompt": np.array([1, 2]),
        "image": np.ones((4, 4, 3)),
    }
    result = mask_demonstration(inputs)
    for key, value in inputs.items():
        if key.startswith("dem_"):
            for leaf in jax.tree.leaves(result[key]):
                assert not leaf.any()
            for leaf in jax.tree.leaves(value):
                assert leaf.all()
        else:
            np.testing.assert_array_equal(value, result[key])


@pytest.mark.parametrize(("mode", "demo"), [("correct", 28), ("no", 28), ("wrong", 25)])
def test_rejects_mislabeled_intervention(mode, demo):
    with pytest.raises(ValueError, match="Only wrong context"):
        prepare_request(
            {"task_index": 25, "split": "test"},
            {
                "mode": mode,
                "demo_task_index": demo,
                "rng_components": [0, 25, 0, 0],
            },
        )
