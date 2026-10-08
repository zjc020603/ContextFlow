"""Inference-only zero demo representations with full-demo masks and positions."""

import dataclasses

from flax import nnx
import jax.numpy as jnp

from openpi.models import contextflow
from openpi.models import model as model_lib
from openpi.models.attention_capture import token_layout


def zero_demo_blocks(prefix, layout):
    tokens, mask, ar_mask = prefix
    blocks = [b for b in layout["blocks"] if b["kind"].startswith("demo_")]
    if not blocks:
        raise ValueError("No demonstration blocks to zero")
    for block in blocks:
        tokens = tokens.at[:, block["start"] : block["stop"], :].set(jnp.zeros((), tokens.dtype))
    return tokens, mask, ar_mask


@dataclasses.dataclass(frozen=True)
class DemoContextZeroConfig(contextflow.ContextFlowConfig):
    def create(self, rng):
        if not (self.use_image_prompts and self.use_action_state_prompts and self.compress_state_action_prompts):
            raise ValueError("Requires all three compressed demonstration modalities")
        return DemoContextZero(self, rngs=nnx.Rngs(rng))


class DemoContextZero(contextflow.ContextFlow):
    def embed_midfix(self, obs, *, train=False):
        if train:
            raise ValueError("Demo context zeroing is inference-only")
        return zero_demo_blocks(super().embed_midfix(obs, train=False), token_layout(self, obs))

    def inspect_ablation(self, obs):
        obs = model_lib.preprocess_observation_incontext(None, obs, train=False)
        original = super().embed_midfix(obs, train=False)
        return original, zero_demo_blocks(original, token_layout(self, obs))
