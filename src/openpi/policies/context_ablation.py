"""Inference-only demonstration interventions; training and default inference are unchanged."""

import jax
import numpy as np


def prepare_request(inputs, experiment):
    mode = experiment["mode"]
    if mode not in ("correct", "no", "wrong"):
        raise ValueError(f"Unknown context mode: {mode}")
    if inputs.get("split") != "test":
        raise ValueError("Context interventions require the test split")
    task = int(inputs["task_index"])
    demo = int(experiment.get("demo_task_index", task))
    if (mode == "wrong") != (demo != task):
        raise ValueError("Only wrong context may select a different demo task")
    components = np.asarray(experiment["rng_components"])
    if components.shape != (4,) or not np.issubdtype(components.dtype, np.integer):
        raise ValueError("rng_components must be [seed, task_index, episode_index, replan_index]")
    if np.any(components < 0) or np.any(components > np.iinfo(np.uint32).max) or int(components[1]) != task:
        raise ValueError("Invalid paired random seed components")
    rng = jax.random.key(int(components[0]))
    for value in components[1:]:
        rng = jax.random.fold_in(rng, int(value))
    # task_index is consumed only by the demo loader; preserve the actual language,
    # live images and state. Its existing cache now keys on the selected demo task.
    inputs = {**inputs, "task_index": demo}
    return (
        inputs,
        {
            "mode": mode,
            "task_index": task,
            "demo_task_index": None if mode == "no" else demo,
            "rng_components": components.tolist(),
        },
        rng,
    )


def mask_demonstration(inputs):
    # Run after normalization and model transforms, so no later transform can
    # recreate valid masks. Replace arrays rather than mutating cached demos.
    result = dict(inputs)
    for key in (
        "dem_prompt_images",
        "dem_prompt_images_mask",
        "dem_prompt_all_states",
        "dem_prompt_all_states_mask",
        "dem_prompt_all_actions",
        "dem_prompt_all_actions_mask",
    ):
        result[key] = jax.tree.map(np.zeros_like, inputs[key])
    return result
