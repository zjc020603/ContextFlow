"""Experiment 0: paired ordinary/captured ContextFlow inference on verified snapshots.

No environment stepping, no behavioral ablation, and no attention interpretation.
Run after examples.libero.prepare_attention_observations in the model environment.
"""
# ruff: noqa: SLF001  # Diagnostic harness intentionally inspects the existing policy transforms/model.

import argparse
import dataclasses
import hashlib
import json
import logging
import pathlib
import time

from flax import nnx
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.models.attention_capture import token_layout
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config


def array_hash(tree):
    digest = hashlib.sha256()
    paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, value in paths:
        arr = np.asarray(value)
        digest.update(str(path).encode())
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def expected_mask(model, observation):
    masks = []
    layout = token_layout(model, observation)
    for block in layout["blocks"]:
        kind, name = block["kind"], block["name"]
        count = block["stop"] - block["start"]
        if kind == "current_image":
            mask = np.repeat(np.asarray(observation.image_masks[name])[:, None], count, axis=1)
        elif kind == "language":
            mask = np.asarray(observation.tokenized_prompt_mask)
        elif kind == "demo_image_latents":
            mask = np.repeat(
                np.asarray(observation.incontext_image_masks[name]).reshape(1, -1).any(axis=1)[:, None], count, axis=1
            )
        elif kind == "demo_states":
            mask = np.repeat(
                np.asarray(observation.incontext_state_masks).reshape(1, -1).any(axis=1)[:, None], count, axis=1
            )
        elif kind == "demo_actions":
            mask = np.repeat(
                np.asarray(observation.incontext_action_masks).reshape(1, -1).any(axis=1)[:, None], count, axis=1
            )
        else:
            continue
        masks.append(mask)
    return np.concatenate(masks, axis=1)


def validate_attention(trace, layout, expected, mode, probability_dtype="float32"):
    p = np.asarray(trace["probabilities"])
    mask = np.asarray(trace["prefix_mask"])
    np.testing.assert_array_equal(mask, expected)
    if p.ndim != 6 or p.shape[2] != 1 or p.shape[-1] != layout["key_size"]:
        raise AssertionError(f"Incorrect attention axes: {p.shape}")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise AssertionError("Invalid attention probabilities")
    row_error = float(np.max(np.abs(p.astype(np.float64).sum(axis=-1) - 1)))
    # Rounding probabilities to BF16 can change their float64 sum by up to ~0.004.
    row_tolerance = 0.004 if probability_dtype == "bfloat16" else 2e-6
    if row_error > row_tolerance:
        raise AssertionError(f"Attention rows not normalized: {row_error}")
    invalid = np.flatnonzero(~mask[0])
    if invalid.size and np.any(p[..., invalid] != 0):
        raise AssertionError("Masked prefix keys received attention")
    if np.any(p[..., 0, layout["prefix_size"] + 1 :] != 0):
        raise AssertionError("State query attended to future action block")
    mass = {}
    for block in layout["blocks"]:
        values = p[..., 1:, block["start"] : block["stop"]]
        mass[block["kind"] + "/" + block["name"]] = float(values.sum(axis=-1).mean())
        if mode == "no" and block["kind"].startswith("demo_") and np.any(values != 0):
            raise AssertionError("No-context demonstration keys received attention")
    return {
        "attention_shape": list(p.shape),
        "row_sum_max_error": row_error,
        "row_sum_tolerance": row_tolerance,
        "masked_key_max": 0.0,
        "block_mass": mass,
    }


def run(args):
    observations = pathlib.Path(args.observations).resolve()
    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "pairs.jsonl"
    if records_path.exists():
        raise FileExistsError(records_path)
    manifest = json.loads((observations / "manifest.json").read_text())
    cases = manifest["cases"]
    smoke = [
        case for case in cases if case["episode"] == manifest["episodes"][0] and case["step"] == manifest["steps"][0]
    ]
    condition_count = len({(case["layout"], case["task"], case["mode"]) for case in cases})
    if len(smoke) != condition_count:
        raise ValueError("Expected one initial smoke case per condition")
    smoke_ids = {case["case_id"] for case in smoke}
    cases = smoke if args.smoke_only else smoke + [case for case in cases if case["case_id"] not in smoke_ids]
    cfg = config.get_config("ContextFlow")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, policy_local_files_only=True))
    policy = policy_config.create_trained_policy_incontext(
        cfg, args.checkpoint, inference_dtype="float32", context_ablation=True
    )
    model = policy._model_for_diagnostics
    captured_fn = nnx_utils.module_jit(model.sample_actions_with_attention)
    tokenizer = PaligemmaTokenizer(max_len=cfg.model.max_token_len)
    parameter_hash_before = array_hash(nnx.state(model, nnx.Param).to_pure_dict())
    (output / "config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
                "observation_manifest": str(observations / "manifest.json"),
                "pairs": len(cases),
                "layers": [0, 9, 17],
                "capture_steps": [0, 5, 9],
                "parameter_sha256_before": parameter_hash_before,
                "model_config": repr(cfg.model),
                "weight_load_dtype": "float32",
                "attention_probability_dtype": model.PaliGemma.llm.module.embed_dtype,
                "jax_version": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "capture_scope": "Gemma suffix queries to all prefix and suffix keys; Perceiver and SigLIP internals not captured",
            },
            indent=2,
        )
        + "\n"
    )
    records = []
    for number, case in enumerate(cases):
        started = time.monotonic()
        with np.load(observations / case["npz"]) as source:
            request = {key: source[key].copy() for key in source.files if key != "reference_executed_actions"}
            reference = source["reference_executed_actions"].copy()
        request.update(case["request"])
        request_hash = array_hash({key: value for key, value in request.items() if isinstance(value, np.ndarray)})
        inputs, info, rng = policy.prepare_inputs(request)
        if info["selected_episode"] != case["expected_selected_episode"]:
            raise AssertionError("Selected demonstration episode differs from original run")
        observation = model_lib.ObservationIncontext.from_dict(inputs)
        input_hash = array_hash(inputs)
        layout = token_layout(model, observation)
        rng_data = np.asarray(jax.random.key_data(rng))
        # Alternate order to detect persistent capture state or cache contamination.
        if number % 2 == 0:
            ordinary = np.asarray(policy._sample_actions(rng, observation))
            captured, trace = captured_fn(rng, observation)
        else:
            captured, trace = captured_fn(rng, observation)
            ordinary = np.asarray(policy._sample_actions(rng, observation))
        captured = np.asarray(captured)
        trace = jax.tree.map(np.asarray, trace)
        if array_hash(inputs) != input_hash:
            raise AssertionError("Inference mutated the shared preprocessed input")
        expected_noise = np.asarray(jax.random.normal(rng, ordinary.shape))
        np.testing.assert_array_equal(trace["noise"], expected_noise)
        if int(trace["executed_steps"]) != 10:
            raise AssertionError("Wrong number of flow sampling steps")
        raw_difference = float(np.max(np.abs(ordinary.astype(np.float64) - captured.astype(np.float64))))
        exact = bool(np.array_equal(ordinary, captured))
        finished = [
            policy._output_transform({"state": np.asarray(inputs["state"])[0].copy(), "actions": actions[0].copy()})[
                "actions"
            ]
            for actions in (ordinary, captured)
        ]
        final_difference = float(np.max(np.abs(finished[0] - finished[1])))
        reference_difference = float(np.max(np.abs(finished[0][:5] - reference)))
        checks = validate_attention(
            trace, layout, expected_mask(model, observation), case["mode"], model.PaliGemma.llm.module.embed_dtype
        )
        directory = output / case["case_id"]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "capture.npz",
            **trace,
            ordinary_actions=ordinary,
            captured_actions=captured,
            ordinary_executable_actions=finished[0],
            captured_executable_actions=finished[1],
            tokenized_prompt=np.asarray(observation.tokenized_prompt),
            language_mask=np.asarray(observation.tokenized_prompt_mask),
        )
        tokens = np.asarray(observation.tokenized_prompt)[0]
        (directory / "layout.json").write_text(
            json.dumps(
                {
                    **layout,
                    "instruction": request["prompt"],
                    "context": info,
                    "language_token_pieces": [tokenizer._tokenizer.id_to_piece(int(token)) for token in tokens],
                },
                indent=2,
            )
            + "\n"
        )
        record = {
            **case,
            "raw_input_sha256": request_hash,
            "processed_input_sha256": input_hash,
            "rng_key": rng_data.tolist(),
            "noise_sha256": array_hash(expected_noise),
            "context": info,
            "raw_action_shape": list(ordinary.shape),
            "raw_actions_exact": exact,
            "raw_action_max_abs_diff": raw_difference,
            "executable_action_max_abs_diff": final_difference,
            "historical_executed_action_max_abs_diff": reference_difference,
            **checks,
            "seconds": time.monotonic() - started,
        }
        with records_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        records.append(record)
        logging.info(
            "Pair %d/%d %s: raw exact=%s maxdiff=%g, historical=%g (%.1fs)",
            number + 1,
            len(cases),
            case["case_id"],
            exact,
            raw_difference,
            reference_difference,
            record["seconds"],
        )
        if not exact or final_difference != 0 or not np.isfinite(ordinary).all():
            raise AssertionError(
                f"Capture changed actions: {case['case_id']}, raw={raw_difference}, final={final_difference}"
            )
        if reference_difference > 1e-6:
            raise AssertionError(f"Ordinary inference differs from historical rollout: {reference_difference}")
        if number == condition_count - 1:
            (output / "smoke_passed.json").write_text(json.dumps({"pairs": condition_count, "all_exact": True}) + "\n")
    parameter_hash_after = array_hash(nnx.state(model, nnx.Param).to_pure_dict())
    if parameter_hash_after != parameter_hash_before:
        raise AssertionError("Model parameters changed")
    result = {
        "pairs": len(records),
        "all_raw_actions_exact": all(row["raw_actions_exact"] for row in records),
        "max_raw_action_difference": max(row["raw_action_max_abs_diff"] for row in records),
        "max_executable_action_difference": max(row["executable_action_max_abs_diff"] for row in records),
        "max_historical_action_difference": max(row["historical_executed_action_max_abs_diff"] for row in records),
        "max_attention_row_sum_error": max(row["row_sum_max_error"] for row in records),
        "parameter_sha256_before": parameter_hash_before,
        "parameter_sha256_after": parameter_hash_after,
        "passed": True,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999")
    parser.add_argument("--smoke-only", action="store_true")
    run(parser.parse_args())
