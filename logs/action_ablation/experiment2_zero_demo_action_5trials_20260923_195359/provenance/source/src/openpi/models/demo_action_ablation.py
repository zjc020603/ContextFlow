"""Inference-only removal of compressed demo-action content, preserving positions.

This is an OOD intervention, not a trained null token or a zero physical action.
The default ContextFlow implementation/checkpoint is not modified.
"""

import dataclasses

from flax import nnx
import jax.numpy as jnp

from openpi.models import contextflow
from openpi.models import model as model_lib


def zero_action_tail(tokens, mask, ar_mask, count):
    """The final prefix block is demo actions; keep mask, shape and other tokens."""
    if count <= 0 or count > tokens.shape[1]:
        raise ValueError("Invalid demo action block size")
    return tokens.at[:, -count:, :].set(jnp.zeros((), tokens.dtype)), mask, ar_mask


@dataclasses.dataclass(frozen=True)
class DemoActionAblationConfig(contextflow.ContextFlowConfig):
    def create(self, rng):
        if not self.use_action_state_prompts or not self.compress_state_action_prompts:
            raise ValueError("This experiment requires the existing compressed state/action configuration")
        return DemoActionAblation(self, rngs=nnx.Rngs(rng))


class DemoActionAblation(contextflow.ContextFlow):
    def embed_midfix(self, obs, *, train=False):
        if train:
            raise ValueError("Demo action ablation is inference-only")
        return zero_action_tail(*super().embed_midfix(obs, train=False), self.num_action_queries)

    def inspect_ablation(self, obs):
        """Return before/after prefixes for the real-checkpoint acceptance gate."""
        obs = model_lib.preprocess_observation_incontext(None, obs, train=False)
        original = super().embed_midfix(obs, train=False)
        changed = zero_action_tail(*original, self.num_action_queries)
        return original, changed
