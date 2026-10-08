"""Opt-in, pure attention capture for ContextFlow diagnostics (no parameter changes)."""

import einops
from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib


def call_with_attention(llm, embedded, positions, mask, kv_cache):
    """Use the same Linen module/parameters; return a separate immutable collection."""
    variables = {"params": nnx.state(llm, nnx.Param).to_pure_dict()}
    outputs, captured = llm.module.apply(
        variables, embedded, positions=positions, mask=mask, kv_cache=kv_cache, mutable=["attention"]
    )
    probabilities = captured["attention"]["layers"]["attn"]["probs"][0]
    # Linen scan stacks layers; Gemma uses grouped query heads K,G.
    probabilities = einops.rearrange(probabilities, "l b k g q s -> l b (k g) q s")
    return outputs, probabilities


def sample_actions_with_attention(model, rng, observation, *, num_steps=10, capture_steps=(0, 5, 9), layers=(0, 9, 17)):
    """Independent captured forward, with ordinary math and a separate trace carry.

    Scope: Gemma suffix-query attention to all prefix/suffix keys. Perceiver and
    SigLIP internal attention are not captured here. No heads or queries averaged.
    """
    # Local import avoids a model-definition import cycle.
    from openpi.models.contextflow import make_attn_mask

    depth = model.PaliGemma.llm.module.configs[0].depth
    if not capture_steps or tuple(sorted(set(capture_steps))) != capture_steps:
        raise ValueError("capture_steps must be sorted, unique and nonempty")
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("layers must be sorted, unique and nonempty")
    if min(capture_steps) < 0 or max(capture_steps) >= num_steps or min(layers) < 0 or max(layers) >= depth:
        raise ValueError("Capture index out of range")
    observation = model_lib.preprocess_observation_incontext(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    midfix_tokens, midfix_mask, midfix_ar_mask = model.embed_midfix(observation)
    prefix_attn_mask = make_attn_mask(midfix_mask, midfix_ar_mask)
    positions = jnp.cumsum(midfix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([midfix_tokens, None], mask=prefix_attn_mask, positions=positions)
    query_count = model.action_horizon + 1
    head_count = model.PaliGemma.llm.module.configs[0].num_heads
    dtype = midfix_tokens.dtype
    trace = jnp.zeros(
        (len(capture_steps), len(layers), batch_size, head_count, query_count, midfix_tokens.shape[1] + query_count),
        dtype=dtype,
    )

    def step(carry):
        x_t, time, index, trace = carry
        suffix_tokens, suffix_mask, suffix_ar_mask = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_mask = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        ((_, suffix_out), _), probabilities = call_with_attention(
            model.PaliGemma.llm, [None, suffix_tokens], positions, full_attn_mask, kv_cache
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        selected = probabilities[jnp.asarray(layers)]
        for slot, requested_step in enumerate(capture_steps):
            trace = jax.lax.cond(
                index == requested_step,
                lambda value, slot=slot: value.at[slot].set(selected),
                lambda value: value,
                trace,
            )
        return x_t + dt * v_t, time + dt, index + 1, trace

    def cond(carry):
        return carry[1] >= -dt / 2

    actions, _, executed_steps, trace = jax.lax.while_loop(cond, step, (noise, 1.0, 0, trace))
    return actions, {
        "probabilities": trace,
        "prefix_mask": midfix_mask,
        "noise": noise,
        "executed_steps": executed_steps,
        "layers": jnp.asarray(layers),
        "capture_steps": jnp.asarray(capture_steps),
    }


def token_layout(model, observation):
    """Key offsets matching embed_midfix, retaining invalid/padded token slots."""
    blocks = []
    offset = 0

    def append(name, kind, count):
        nonlocal offset
        blocks.append({"name": name, "kind": kind, "start": offset, "stop": offset + count})
        offset += count

    patch_h, patch_w = model._image_patch_size  # noqa: SLF001
    for camera, image in observation.images.items():
        count = 1 if model.avg_current_img else (image.shape[-3] // patch_h) * (image.shape[-2] // patch_w)
        append(camera, "current_image", count)
    if model.use_text_prompts and observation.tokenized_prompt is not None:
        append("language", "language", observation.tokenized_prompt.shape[-1])
    if model.use_image_prompts:
        for camera in observation.incontext_images:
            append(camera, "demo_image_latents", model.num_image_queries)
    if model.use_action_state_prompts:
        states = observation.incontext_states
        actions = observation.incontext_actions
        state_count = (
            model.num_state_queries
            if hasattr(model, "state_compressor")
            else states.reshape(states.shape[0], -1, states.shape[-1]).shape[1]
        )
        action_count = (
            model.num_action_queries
            if hasattr(model, "action_compressor")
            else actions.reshape(actions.shape[0], -1, actions.shape[-1]).shape[1]
        )
        append("demo_states", "demo_states", state_count)
        append("demo_actions", "demo_actions", action_count)
    prefix_size = offset
    append("current_state", "current_state", 1)
    append("predicted_actions", "predicted_actions", model.action_horizon)
    return {
        "blocks": blocks,
        "prefix_size": prefix_size,
        "key_size": offset,
        "query_layout": {"current_state": [0, 1], "predicted_actions": [1, 1 + model.action_horizon]},
    }
