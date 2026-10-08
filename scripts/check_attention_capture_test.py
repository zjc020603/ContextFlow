"""Failure-path checks for the attention acceptance gate."""

import numpy as np
import pytest

from scripts.check_attention_capture import validate_attention


def example_trace():
    # Two prefix keys (one masked), then state + two action keys.
    p = np.zeros((1, 1, 1, 1, 3, 5), np.float32)
    p[..., 0] = 0.5
    p[..., 2] = 0.5
    return {"probabilities": p, "prefix_mask": np.array([[True, False]])}


def test_gate_rejects_normalized_attention_to_masked_keys():
    trace = example_trace()
    trace["probabilities"][..., 1] = 0.25
    trace["probabilities"][..., 0] = 0.25
    layout = {"key_size": 5, "prefix_size": 2, "blocks": []}
    with pytest.raises(AssertionError, match="Masked prefix"):
        validate_attention(trace, layout, trace["prefix_mask"], "correct")


def test_gate_rejects_no_context_demo_attention_and_future_action_leak():
    trace = example_trace()
    layout = {
        "key_size": 5,
        "prefix_size": 2,
        "blocks": [{"kind": "demo_image_latents", "name": "front", "start": 0, "stop": 1}],
    }
    with pytest.raises(AssertionError, match="No-context"):
        validate_attention(trace, layout, trace["prefix_mask"], "no")
    trace["probabilities"][..., 0, 0] = 0.25
    trace["probabilities"][..., 0, 3] = 0.25
    with pytest.raises(AssertionError, match="future action"):
        validate_attention(trace, layout, trace["prefix_mask"], "correct")
