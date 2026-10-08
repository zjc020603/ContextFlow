"""Zero selected encoded demonstration modalities before Gemma, retaining positions."""

import dataclasses

from flax import nnx
import jax.numpy as jnp

from openpi.models import contextflow
from openpi.models import model as model_lib
from openpi.models.attention_capture import token_layout

MODALITIES = {"state": ("demo_states",), "state_action": ("demo_states", "demo_actions")}


def zero_modalities(prefix, layout, kinds):
    if not kinds or any(kind not in ("demo_states", "demo_actions") for kind in kinds):
        raise ValueError("Only explicit demonstration state/action blocks may be zeroed")
    tokens, mask, ar = prefix
    blocks = [block for block in layout["blocks"] if block["kind"] in kinds]
    if {b["kind"] for b in blocks} != set(kinds):
        raise ValueError("Requested demonstration modality missing from token layout")
    for block in blocks:
        tokens = tokens.at[:, block["start"] : block["stop"], :].set(jnp.zeros((), tokens.dtype))
    return tokens, mask, ar


@dataclasses.dataclass(frozen=True)
class DemoModalityAblationConfig(contextflow.ContextFlowConfig):
    ablation: str = "state"

    def create(self, rng):
        if self.ablation not in MODALITIES:
            raise ValueError("Expected state or state_action ablation")
        if not (self.use_image_prompts and self.use_action_state_prompts and self.compress_state_action_prompts):
            raise ValueError("Requires the checkpoint's three compressed demonstration modalities")
        return DemoModalityAblation(self, rngs=nnx.Rngs(rng))


class DemoModalityAblation(contextflow.ContextFlow):
    def __init__(self, config, rngs):
        super().__init__(config, rngs=rngs)
        self.ablated_kinds = MODALITIES[config.ablation]

    def embed_midfix(self, obs, *, train=False):
        if train:
            raise ValueError("Modality ablation is inference-only")
        return zero_modalities(super().embed_midfix(obs, train=False), token_layout(self, obs), self.ablated_kinds)

    def inspect_ablation(self, obs):
        obs = model_lib.preprocess_observation_incontext(None, obs, train=False)
        original = super().embed_midfix(obs, train=False)
        return original, zero_modalities(original, token_layout(self, obs), self.ablated_kinds)
