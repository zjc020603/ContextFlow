"""ContextFlow demonstrations on the complete pretrained π0 / π0.5 backbone."""

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import contextflow
from openpi.models import model as _model
from openpi.models import siglip
from openpi.models.full_pi import gemma
from openpi.models.full_pi.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class FullContextFlowConfig(contextflow.ContextFlowConfig):
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    pi05: bool = True
    max_token_len: int | None = None
    discrete_state_input: bool | None = None
    # Dataset dimensions before the legacy input adapter pads to action_dim.
    state_dim: int | None = None
    native_action_dim: int | None = None

    def __post_init__(self):
        if self.paligemma_variant not in ("gemma_2b", "gemma_2b_lora", "dummy"):
            raise ValueError("FullContextFlow requires the complete pretrained Gemma-2B VLM")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pi05 and not self.use_text_prompts:
            raise ValueError("π0.5 requires the language/state prefix; do not disable use_text_prompts")
        if self.avg_current_img:
            raise ValueError("The full pretrained backbone requires unpooled current-image tokens")

        for dim in (self.state_dim, self.native_action_dim):
            if dim is not None and not 0 < dim <= self.action_dim:
                raise ValueError("Native dimensions must be in [1, action_dim]")

    def create(self, rng):
        return FullContextFlow(self, rngs=nnx.Rngs(rng))


class FullContextFlow(Pi0):
    """Retain all upstream backbone weights, adding only demonstration modules.

    ContextFlow's bidirectional prefix layout and Perceiver implementation are
    shared verbatim. Current state and action generation follow upstream Pi0.
    The inherited model type stays PI0_INCONTEXT for the existing data/policy API.
    """

    preprocess_observation = staticmethod(_model.preprocess_observation_incontext)
    _empty_image_tokens = contextflow.ContextFlow._empty_image_tokens  # noqa: SLF001
    _encode_image_tokens = contextflow.ContextFlow._encode_image_tokens  # noqa: SLF001
    embed_midfix = contextflow.ContextFlow.embed_midfix

    def __init__(self, config: FullContextFlowConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        width = gemma.get_config(config.paligemma_variant).width
        self.use_image_prompts = config.use_image_prompts
        self.use_text_prompts = config.use_text_prompts
        self.use_action_state_prompts = config.use_action_state_prompts
        self.avg_current_img = config.avg_current_img
        self.num_image_queries = config.num_image_queries
        self.num_state_queries = config.num_state_queries
        self.num_action_queries = config.num_action_queries
        self._image_embed_dim = width
        self._image_patch_size = siglip.decode_variant("So400m/14")["patch_size"]
        self._image_token_dtype = jnp.dtype(config.dtype)

        if self.use_action_state_prompts:
            self.demo_action_proj = nnx.Linear(config.action_dim, width, rngs=rngs)
            self.demo_state_proj = nnx.Linear(config.action_dim, width, rngs=rngs)
        if self.use_image_prompts:
            self.image_compressor = contextflow.PerceiverCompressor(
                num_queries=config.num_image_queries,
                embed_dim=width,
                num_heads=config.num_attn_heads,
                num_layers=config.num_image_compressor_layers,
                rngs=rngs,
            )
        if self.use_action_state_prompts and config.compress_state_action_prompts:
            self.state_compressor = contextflow.PerceiverCompressor(
                num_queries=config.num_state_queries,
                embed_dim=width,
                num_heads=config.num_attn_heads,
                num_layers=config.num_state_compressor_layers,
                rngs=rngs,
            )
            self.action_compressor = contextflow.PerceiverCompressor(
                num_queries=config.num_action_queries,
                embed_dim=width,
                num_heads=config.num_attn_heads,
                num_layers=config.num_action_compressor_layers,
                rngs=rngs,
            )

    def embed_prefix(self, obs):
        return self.embed_midfix(obs)
