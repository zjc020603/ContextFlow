"""Gated no-context control: zero all encoded demo content, retain full masks."""
# ruff: noqa: SLF001

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from flax import nnx
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.models.attention_capture import token_layout
from openpi.models.demo_context_zero import DemoContextZeroConfig
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import config
from scripts.check_attention_capture import array_hash
from scripts.check_attention_capture import expected_mask
from scripts.check_attention_capture import validate_attention
from scripts.serve_demo_action_ablation import exact
from scripts.serve_demo_action_ablation import output_actions

INTERVENTION = "zero_all_encoded_demo_tokens_keep_mask"


def check_prefix(before, after, layout):
    tokens, mask, ar = map(np.asarray, before)
    changed, new_mask, new_ar = map(np.asarray, after)
    exact(mask, new_mask, "Prefix masks changed")
    exact(ar, new_ar, "Attention block masks changed")
    exact(np.cumsum(mask, -1), np.cumsum(new_mask, -1), "Positions changed")
    ranges = []
    for block in layout["blocks"]:
        start, stop = block["start"], block["stop"]
        if start >= layout["prefix_size"]:
            continue
        if block["kind"].startswith("demo_"):
            assert np.all(changed[:, start:stop] == 0)
            ranges.append([start, stop])
        else:
            exact(tokens[:, start:stop], changed[:, start:stop], "Live image/language tokens changed")
    assert np.isfinite(changed).all()
    return {
        "mask_and_positions_exact": True,
        "non_demo_tokens_exact": True,
        "demo_zero_ranges": ranges,
        "valid_prefix_tokens": int(mask.sum()),
        "first_action_position_id": int(mask.sum()) + 1,
    }


class ZeroPolicy:
    def __init__(self, args):
        self.root = Path(args.output).resolve()
        if self.root.exists():
            raise FileExistsError("Use a fresh server output directory")
        self.root.mkdir(parents=True)
        self.calls = 0
        self.capture_ids = set()
        cfg = config.get_config("ContextFlow")
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, policy_local_files_only=True))
        baseline = policy_config.create_trained_policy_incontext(
            cfg, args.checkpoint, inference_dtype="float32", context_ablation=True
        )
        zero_cfg = dataclasses.replace(cfg, model=DemoContextZeroConfig(**dataclasses.asdict(cfg.model)))
        self.policy = policy_config.create_trained_policy_incontext(
            zero_cfg, args.checkpoint, inference_dtype="float32", context_ablation=True
        )
        self.model = self.policy._model_for_diagnostics
        self.capture_fn = nnx_utils.module_jit(self.model.sample_actions_with_attention)
        self.inspect_fn = nnx_utils.module_jit(self.model.inspect_ablation)
        self.parameter_hash = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
        assert self.parameter_hash == array_hash(nnx.state(baseline._model_for_diagnostics, nnx.Param).to_pure_dict())
        logging.getLogger().setLevel(logging.INFO)
        self.gate(baseline, Path(args.source))
        self.metadata = {
            **self.policy.metadata,
            "no_context_intervention": INTERVENTION,
            "baseline_gate_passed": True,
            "attention_environment_steps": [0, 80, 160],
        }
        (self.root / "server_config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "intervention": INTERVENTION,
                    "parameter_sha256": self.parameter_hash,
                    "devices": [str(d) for d in jax.devices()],
                    "weight_dtype": "float32",
                    "gemma_dtype": self.model.PaliGemma.llm.module.embed_dtype,
                },
                indent=2,
            )
            + "\n"
        )
        (self.root / "READY.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def prepare_zero(self, request):
        # Load the normal same-task template ONLY to retain the original masks.
        # Its complete encoded content is erased by DemoContextZero before Gemma.
        request = dict(request)
        spec = dict(request["context_ablation"])
        if spec["mode"] != "no":
            raise ValueError("Only no-context requests are accepted")
        spec.update(mode="correct", demo_task_index=int(request["task_index"]))
        request["context_ablation"] = spec
        inputs, info, rng = self.policy.prepare_inputs(request)
        template_episode = info["selected_episode"]
        info.update(
            mode="no",
            demo_task_index=None,
            selected_episode=[],
            mask_template_episode=template_episode,
            no_context_intervention=INTERVENTION,
        )
        return inputs, info, rng

    def gate(self, baseline, source):
        cases = json.loads((source / "observations/manifest.json").read_text())["cases"]
        cases = [c for c in cases if c["layout"] == "original" and c["episode"] == 0 and c["step"] == 0]
        assert len(cases) == 6
        results = []
        for case in cases:
            with np.load(source / "observations" / case["npz"]) as snap:
                request = {k: snap[k].copy() for k in snap.files if k != "reference_executed_actions"}
            request.update(case["request"])
            inputs, _, rng = baseline.prepare_inputs(request)
            obs = model_lib.ObservationIncontext.from_dict(inputs)
            full = np.asarray(baseline._sample_actions(rng, obs))
            with np.load(source / "validation_attempt3" / case["case_id"] / "capture.npz") as previous:
                exact(full, previous["ordinary_actions"], "Historical model output changed")
                exact(
                    output_actions(baseline, inputs, full),
                    previous["ordinary_executable_actions"],
                    "Historical executable actions changed",
                )
            result = {"case_id": case["case_id"], "historical_baseline_exact": True}
            if case["mode"] == "correct":
                no_request = {**request, "context_ablation": {**request["context_ablation"], "mode": "no"}}
                new_inputs, info, new_rng = self.prepare_zero(no_request)
                exact(jax.random.key_data(rng), jax.random.key_data(new_rng), "Noise key changed")
                new_obs = model_lib.ObservationIncontext.from_dict(new_inputs)
                assert array_hash(obs) == array_hash(new_obs)
                assert array_hash(inputs) == array_hash(new_inputs)
                before, after = self.inspect_fn(obs)
                layout = token_layout(self.model, obs)
                result.update(check_prefix(before, after, layout))
                zero = np.asarray(self.policy._sample_actions(rng, obs))
                altered = dataclasses.replace(
                    obs,
                    incontext_images=jax.tree.map(lambda x: -x + 0.19, obs.incontext_images),
                    incontext_states=-obs.incontext_states + 0.23,
                    incontext_actions=-obs.incontext_actions + 0.37,
                )
                exact(zero, self.policy._sample_actions(rng, altered), "Demo values leaked through zeroing")
                wrong_request = {
                    **request,
                    "context_ablation": {
                        **request["context_ablation"],
                        "mode": "wrong",
                        "demo_task_index": 28 if case["task"] == "milk" else 25,
                    },
                }
                wrong_inputs, _, _ = baseline.prepare_inputs(wrong_request)
                wrong_obs = model_lib.ObservationIncontext.from_dict(wrong_inputs)
                exact(
                    expected_mask(self.model, obs),
                    expected_mask(self.model, wrong_obs),
                    "Demo templates have different masks",
                )
                exact(zero, self.policy._sample_actions(rng, wrong_obs), "Demo identity leaked through zeroing")
                recorded, trace = self.capture_fn(rng, obs)
                exact(zero, recorded, "Attention capture changed zero-context actions")
                # The legacy validator's mode=no requires demo keys to be masked.
                # Here the keys intentionally stay valid, so use generic validation.
                result["attention_checks"] = validate_attention(
                    jax.tree.map(np.asarray, trace),
                    layout,
                    expected_mask(self.model, obs),
                    INTERVENTION,
                    self.model.PaliGemma.llm.module.embed_dtype,
                )
                old_inputs, _, _ = baseline.prepare_inputs(no_request)
                old_mask = expected_mask(self.model, model_lib.ObservationIncontext.from_dict(old_inputs))
                result.update(
                    all_demo_values_invariant=True,
                    swapped_demo_identity_invariant=True,
                    capture_actions_exact=True,
                    prepared_full_inputs_exact=True,
                    old_masked_first_action_position_id=int(old_mask.sum()) + 1,
                )
            results.append(result)
            (self.root / "gate_cases.json").write_text(json.dumps(results, indent=2) + "\n")
            logging.info("No-context gate %d/6 passed: %s", len(results), case["case_id"])
        (self.root / "GATE_PASSED.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "historical_baselines": 6,
                    "zero_intervention_checks": 2,
                    "parameter_sha256": self.parameter_hash,
                },
                indent=2,
            )
            + "\n"
        )

    def infer(self, request):
        if request.get("finalize_no_context_zero") is True:
            after = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
            assert after == self.parameter_hash
            result = {
                "passed": True,
                "runtime_calls": self.calls,
                "captures": len(self.capture_ids),
                "parameter_sha256_before": self.parameter_hash,
                "parameter_sha256_after": after,
            }
            (self.root / "SERVER_FINISHED.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        request = dict(request)
        if request.pop("no_context_intervention") != INTERVENTION:
            raise ValueError("Wrong intervention protocol")
        inputs, info, rng = self.prepare_zero(request)
        episode, replan = info["rng_components"][2:]
        assert info["task_index"] in (25, 28)
        assert 0 <= episode < 25
        assert 0 <= replan < 56
        obs = model_lib.ObservationIncontext.from_dict(inputs)
        ordinary = np.asarray(self.policy._sample_actions(rng, obs))
        assert np.isfinite(ordinary).all()
        actions = output_actions(self.policy, inputs, ordinary)
        task = "milk" if info["task_index"] == 25 else "tomato_sauce"
        step = replan * 5
        capture_id = f"{task}/ep{episode:03d}_step{step:03d}"
        if step in (0, 80, 160) and capture_id not in self.capture_ids:
            captured, trace = self.capture_fn(rng, obs)
            exact(ordinary, captured, "Recording changed executed actions")
            trace = jax.tree.map(np.asarray, trace)
            exact(trace["noise"], jax.random.normal(rng, ordinary.shape), "Wrong noise")
            assert int(trace["executed_steps"]) == 10
            layout = token_layout(self.model, obs)
            checks = validate_attention(
                trace, layout, expected_mask(self.model, obs), INTERVENTION, self.model.PaliGemma.llm.module.embed_dtype
            )
            prefix = check_prefix(*self.inspect_fn(obs), layout)
            directory = self.root / "attention" / capture_id
            directory.mkdir(parents=True)
            np.savez_compressed(
                directory / "capture.npz",
                **trace,
                ordinary_actions=ordinary,
                captured_actions=np.asarray(captured),
                executable_actions=actions,
                tokenized_prompt=np.asarray(obs.tokenized_prompt),
                language_mask=np.asarray(obs.tokenized_prompt_mask),
            )
            np.savez_compressed(
                directory / "observation.npz", **{k: v for k, v in request.items() if isinstance(v, np.ndarray)}
            )
            (directory / "layout.json").write_text(
                json.dumps(
                    {
                        **layout,
                        "context": info,
                        "instruction": request["prompt"],
                        "intervention": INTERVENTION,
                        "prefix_checks": prefix,
                        "attention_checks": checks,
                        "processed_input_sha256": array_hash(inputs),
                        "rng_key": np.asarray(jax.random.key_data(rng)).tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            self.capture_ids.add(capture_id)
        self.calls += 1
        if replan == 55:
            logging.info("Completed zero-context %s episode %d", task, episode)
        return {"actions": actions, "context_ablation": info}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999")
    parser.add_argument("--source", default="logs/attention_capture/experiment0_consistency_20260923_143830")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    policy = ZeroPolicy(args)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()
