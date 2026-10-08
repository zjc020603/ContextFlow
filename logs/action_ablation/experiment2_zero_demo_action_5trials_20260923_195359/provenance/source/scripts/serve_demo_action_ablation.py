"""Gated, local-only experiment-2 server. Records attention without analyzing it."""
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
from openpi.models.demo_action_ablation import DemoActionAblationConfig
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import config
from scripts.check_attention_capture import array_hash
from scripts.check_attention_capture import expected_mask
from scripts.check_attention_capture import validate_attention

INTERVENTION = "zero_compressed_demo_action_keep_mask"


def exact(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise AssertionError(label)


def output_actions(policy, inputs, actions):
    return policy._output_transform(
        {"state": np.asarray(inputs["state"])[0].copy(), "actions": np.asarray(actions)[0].copy()}
    )["actions"]


def check_prefix(before, after, count=32):
    tokens, mask, ar = map(np.asarray, before)
    changed, new_mask, new_ar = map(np.asarray, after)
    exact(mask, new_mask, "Ablation changed prefix mask")
    exact(ar, new_ar, "Ablation changed attention block mask")
    exact(tokens[:, :-count], changed[:, :-count], "Ablation changed non-action prefix")
    assert np.all(changed[:, -count:] == 0)
    assert np.isfinite(tokens).all()
    exact(np.cumsum(mask, axis=-1), np.cumsum(new_mask, axis=-1), "Position IDs changed")
    return {
        "prefix_shape": list(tokens.shape),
        "demo_action_range": [tokens.shape[1] - count, tokens.shape[1]],
        "valid_prefix_tokens": int(mask.sum()),
        "first_action_position_id": int(mask.sum()) + 1,
        "original_action_token_l2": float(np.linalg.norm(tokens[:, -count:].astype(np.float64))),
        "zero_action_token_l2": 0.0,
        "non_action_tokens_exact": True,
        "mask_and_positions_exact": True,
    }


class ExperimentPolicy:
    def __init__(self, args):
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime_calls = 0
        self.captures = 0
        cfg = config.get_config("ContextFlow")
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, policy_local_files_only=True))
        baseline = policy_config.create_trained_policy_incontext(
            cfg, args.checkpoint, inference_dtype="float32", context_ablation=True
        )
        zero_cfg = dataclasses.replace(cfg, model=DemoActionAblationConfig(**dataclasses.asdict(cfg.model)))
        self.policy = policy_config.create_trained_policy_incontext(
            zero_cfg, args.checkpoint, inference_dtype="float32", context_ablation=True
        )
        self.model = self.policy._model_for_diagnostics
        self.capture_fn = nnx_utils.module_jit(self.model.sample_actions_with_attention)
        self.inspect_fn = nnx_utils.module_jit(self.model.inspect_ablation)
        self.parameter_hash = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
        if self.parameter_hash != array_hash(nnx.state(baseline._model_for_diagnostics, nnx.Param).to_pure_dict()):
            raise AssertionError("Baseline and zero model weights differ")
        logging.getLogger().setLevel(logging.INFO)
        self.gate(baseline, Path(args.source).resolve())
        self.metadata = {
            **self.policy.metadata,
            "demo_action_intervention": INTERVENTION,
            "baseline_gate_passed": True,
            "attention_environment_steps": [0, 80, 160],
            "position_mask_preserved": True,
            "ood_intervention": True,
        }
        (self.root / "server_config.json").write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "source": args.source,
                    "intervention": INTERVENTION,
                    "weight_dtype": "float32",
                    "gemma_dtype": self.model.PaliGemma.llm.module.embed_dtype,
                    "model_config": repr(zero_cfg.model),
                    "parameter_sha256": self.parameter_hash,
                    "jax": jax.__version__,
                    "devices": [str(d) for d in jax.devices()],
                },
                indent=2,
            )
            + "\n"
        )
        (self.root / "READY.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def gate(self, baseline, source):
        manifest = json.loads((source / "observations/manifest.json").read_text())
        cases = [c for c in manifest["cases"] if c["step"] == 0 and c["episode"] < 5]
        if len(cases) != 60:
            raise AssertionError("Expected 60 paired historical initial observations")
        results = []
        for case in cases:
            with np.load(source / "observations" / case["npz"]) as snap:
                request = {k: snap[k].copy() for k in snap.files if k != "reference_executed_actions"}
            request.update(case["request"])
            inputs, info, rng = baseline.prepare_inputs(request)
            obs = model_lib.ObservationIncontext.from_dict(inputs)
            ordinary = np.asarray(baseline._sample_actions(rng, obs))
            with np.load(source / "validation_attempt3" / case["case_id"] / "capture.npz") as previous:
                exact(
                    ordinary,
                    previous["ordinary_actions"],
                    "Current baseline differs from validated historical model: " + case["case_id"],
                )
                exact(
                    output_actions(baseline, inputs, ordinary),
                    previous["ordinary_executable_actions"],
                    "Historical output transform changed",
                )
            assert info["selected_episode"] == case["expected_selected_episode"]
            result = {"case_id": case["case_id"], "historical_baseline_raw_and_executable_exact": True}
            if case["episode"] == 0:
                before, after = self.inspect_fn(obs)
                result.update(check_prefix(before, after))
                zero = np.asarray(self.policy._sample_actions(rng, obs))
                recorded, trace = self.capture_fn(rng, obs)
                exact(zero, recorded, "Attention capture changed ablated output")
                trace = jax.tree.map(np.asarray, trace)
                validate_attention(
                    trace,
                    token_layout(self.model, obs),
                    expected_mask(self.model, obs),
                    info["mode"],
                    self.model.PaliGemma.llm.module.embed_dtype,
                )
                result["zero_capture_exact"] = True
                result["zero_vs_baseline_max_abs_diff"] = float(np.max(np.abs(zero - ordinary)))
                if info["mode"] == "no":
                    exact(zero, ordinary, "No-context should be unaffected by zeroing masked demo action tokens")
                else:
                    # Hold every input but numeric demo actions fixed: the zero path must be invariant.
                    altered = dataclasses.replace(obs, incontext_actions=-obs.incontext_actions + 0.37)
                    exact(
                        zero,
                        self.policy._sample_actions(rng, altered),
                        "Demo action values leaked past the intervention",
                    )
                    result["changed_demo_action_values_have_zero_effect"] = True
            results.append(result)
            with (self.root / "gate_cases.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")
            logging.info("Gate %d/%d passed %s", len(results), len(cases), case["case_id"])
        assert array_hash(nnx.state(self.model, nnx.Param).to_pure_dict()) == self.parameter_hash
        (self.root / "GATE_PASSED.json").write_text(
            json.dumps(
                {
                    "historical_initial_cases": 60,
                    "capture_and_prefix_cases": 12,
                    "action_value_invariance_cases": 8,
                    "no_context_unchanged_cases": 4,
                    "parameter_sha256": self.parameter_hash,
                    "passed": True,
                },
                indent=2,
            )
            + "\n"
        )

    def infer(self, request):
        if request.get("finalize_demo_action_ablation") is True:
            after = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
            if after != self.parameter_hash:
                raise AssertionError("Parameters changed during rollouts")
            result = {
                "runtime_inferences": self.runtime_calls,
                "saved_attention_records": self.captures,
                "parameter_sha256_before": self.parameter_hash,
                "parameter_sha256_after": after,
                "passed": True,
            }
            (self.root / "SERVER_FINISHED.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        request = dict(request)
        intervention = request.pop("demo_action_ablation")
        if intervention.get("intervention") != INTERVENTION or intervention.get("layout") not in (
            "original",
            "swap_milk_tomato",
        ):
            raise ValueError("Invalid action ablation protocol")
        inputs, info, rng = self.policy.prepare_inputs(request)
        if info is None or info["mode"] not in ("correct", "wrong") or info["task_index"] not in (25, 28):
            raise ValueError("Pilot accepts the paired correct/wrong unseen object conditions only")
        episode, replan = info["rng_components"][2:]
        if not 0 <= episode < 5 or not 0 <= replan < 56:
            raise ValueError("Unexpected pilot episode or replan")
        step = replan * 5
        obs = model_lib.ObservationIncontext.from_dict(inputs)
        input_hash = array_hash(inputs)
        ordinary = np.asarray(self.policy._sample_actions(rng, obs))
        if not np.isfinite(ordinary).all():
            raise FloatingPointError("Nonfinite action output")
        actions = output_actions(self.policy, inputs, ordinary)
        if step in (0, 80, 160):
            captured, trace = self.capture_fn(rng, obs)
            exact(ordinary, captured, "Capture changed rollout actions")
            trace = jax.tree.map(np.asarray, trace)
            exact(trace["noise"], jax.random.normal(rng, ordinary.shape), "Incorrect captured noise")
            if int(trace["executed_steps"]) != 10:
                raise AssertionError("Wrong flow step count")
            layout = token_layout(self.model, obs)
            checks = validate_attention(
                trace, layout, expected_mask(self.model, obs), info["mode"], self.model.PaliGemma.llm.module.embed_dtype
            )
            prefix = check_prefix(*self.inspect_fn(obs))
            if array_hash(inputs) != input_hash:
                raise AssertionError("Diagnostic changed observation inputs")
            task = "milk" if info["task_index"] == 25 else "tomato_sauce"
            directory = (
                self.root
                / "attention"
                / intervention["layout"]
                / info["mode"]
                / task
                / f"ep{episode:03d}_step{step:03d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
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
                        "instruction": request["prompt"],
                        "context": info,
                        "action_intervention": INTERVENTION,
                        "prefix_intervention_checks": prefix,
                        "attention_checks": checks,
                        "processed_input_sha256": input_hash,
                        "rng_key": np.asarray(jax.random.key_data(rng)).tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            self.captures += 1
            logging.info("Saved ablated attention %s", directory.relative_to(self.root))
        self.runtime_calls += 1
        if self.runtime_calls % 56 == 0:
            logging.info("Finished %d rollout replans", self.runtime_calls)
        return {"actions": actions, "context_ablation": {**info, "demo_action_intervention": INTERVENTION}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999")
    parser.add_argument("--source", default="logs/attention_capture/experiment0_consistency_20260923_143830")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8122)
    args = parser.parse_args()
    if (Path(args.output) / "gate_cases.jsonl").exists():
        raise FileExistsError("Use a fresh server output directory")
    policy = ExperimentPolicy(args)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()
