"""Auditable full-context benchmark server; ordinary ContextFlow math and masks."""
# ruff: noqa: SLF001

import argparse
import dataclasses
import faulthandler
import json
import logging
from pathlib import Path
import signal

from flax import nnx
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.models.attention_capture import token_layout
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import config
from scripts.check_attention_capture import array_hash

PROTOCOL = "goal_long_full_correct_context_v1"


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise AssertionError("Historical baseline actions changed")


def executable(policy, inputs, raw):
    return policy._output_transform(
        {"state": np.asarray(inputs["state"])[0].copy(), "actions": np.asarray(raw)[0].copy()}
    )["actions"]


class BenchmarkPolicy:
    def __init__(self, args):
        self.root = Path(args.output).resolve()
        if self.root.exists():
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True)
        self.tasks = {t["dataset_task_index"]: t for t in json.loads(Path(args.manifest).read_text())["tasks"]}
        logging.warning("Startup stage: loading checkpoint and demonstration dataset")
        cfg = config.get_config("ContextFlow")
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, policy_local_files_only=True))
        self.policy = policy_config.create_trained_policy_incontext(
            cfg, args.checkpoint, inference_dtype="float32", context_ablation=False
        )
        logging.warning("Startup stage: policy loaded; hashing parameters")
        self.model = self.policy._model_for_diagnostics
        self.prefix_fn = nnx_utils.module_jit(self.model.embed_midfix)
        self.parameter_hash = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
        logging.getLogger().setLevel(logging.INFO)
        logging.warning("Startup stage: verifying historical baseline")
        self.gate(Path(args.source))
        self.active = None
        self.calls = 0
        self.total = 0
        self.finished = []
        self.metadata = {
            **self.policy.metadata,
            "goal_long_protocol": PROTOCOL,
            "baseline_gate_passed": True,
            "context": "complete correct demonstration; no ablation",
        }
        (self.root / "READY.json").write_text(
            json.dumps(
                {
                    **self.metadata,
                    "parameter_sha256": self.parameter_hash,
                    "model_config": repr(cfg.model),
                    "checkpoint": args.checkpoint,
                    "devices": [str(d) for d in jax.devices()],
                    "matmul_precision": str(jax.config.jax_default_matmul_precision),
                    "inference_dtype": "float32",
                    "internal_gemma_dtype": self.model.PaliGemma.llm.module.embed_dtype,
                },
                indent=2,
            )
            + "\n"
        )

    def gate(self, source):
        cases = json.loads((source / "observations/manifest.json").read_text())["cases"]
        case = next(c for c in cases if c["case_id"] == "original/correct/milk/ep000_step000")
        with np.load(source / "observations" / case["npz"]) as snap:
            request = {k: snap[k].copy() for k in snap.files if k != "reference_executed_actions"}
        request.update({k: v for k, v in case["request"].items() if k != "context_ablation"})
        inputs, info, _ = self.policy.prepare_inputs(request)
        assert info is None
        rng = jax.random.key(0)
        for v in case["request"]["context_ablation"]["rng_components"][1:]:
            rng = jax.random.fold_in(rng, v)
        raw = self.policy._sample_actions(rng, model_lib.ObservationIncontext.from_dict(inputs))
        with np.load(source / "validation_attempt3" / case["case_id"] / "capture.npz") as old:
            exact(raw, old["ordinary_actions"])
            exact(executable(self.policy, inputs, raw), old["ordinary_executable_actions"])
        (self.root / "GATE_PASSED.json").write_text(
            json.dumps(
                {
                    "historical_case": case["case_id"],
                    "raw_and_executable_exact": True,
                    "parameter_sha256": self.parameter_hash,
                },
                indent=2,
            )
            + "\n"
        )

    def infer(self, request):
        command = request.get("benchmark_control")
        if command == "start_task":
            if self.active is not None:
                raise ValueError("Previous task not finalized")
            task = int(request["task_index"])
            if task not in self.tasks or task in self.finished:
                raise ValueError("Unknown or duplicate task")
            self.active = task
            self.calls = 0
            self.policy._rng = jax.random.key(0)
            return {"task_index": task, "policy_seed": 0, "demo_episode": self.tasks[task]["demo_episode"]}
        if command == "end_task":
            assert int(request["task_index"]) == self.active
            result = {"task_index": self.active, "inferences": self.calls, "policy_seed": 0, "complete_context": True}
            (self.root / f"task{self.active:02d}_finished.json").write_text(json.dumps(result, indent=2) + "\n")
            self.finished.append(self.active)
            self.active = None
            return result
        if command == "finalize":
            assert self.active is None
            after = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
            assert after == self.parameter_hash
            result = {
                "tasks": self.finished,
                "total_inferences": self.total,
                "parameter_sha256_before": self.parameter_hash,
                "parameter_sha256_after": after,
                "passed": True,
            }
            (self.root / "FINISHED.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        assert command is None
        assert self.active is not None
        assert request["task_index"] == self.active
        assert request["prompt"] == self.tasks[self.active]["instruction"]
        assert request["split"] == "test"
        assert "context_ablation" not in request
        inputs, info, rng = self.policy.prepare_inputs(request)
        assert info is None
        selected = np.asarray(inputs["selected_episode"]).reshape(-1).tolist()
        assert selected == [self.tasks[self.active]["demo_episode"]]
        obs = model_lib.ObservationIncontext.from_dict(inputs)
        if self.calls == 0:
            processed = model_lib.preprocess_observation_incontext(None, obs, train=False)
            tokens, mask, ar = self.prefix_fn(processed)
            layout = token_layout(self.model, processed)
            blocks = []
            for block in layout["blocks"]:
                if block["kind"].startswith("demo_"):
                    valid = np.asarray(mask[:, block["start"] : block["stop"]])
                    assert np.all(valid == ("right" not in block["name"]))
                    blocks.append(
                        {
                            **block,
                            "valid_tokens": int(valid.sum()),
                            "token_l2": float(
                                np.linalg.norm(np.asarray(tokens[:, block["start"] : block["stop"]], dtype=np.float64))
                            ),
                        }
                    )
            directory = self.root / f"task{self.active:02d}_inputs"
            directory.mkdir()
            np.savez_compressed(
                directory / "demo_and_observation.npz",
                **{
                    **{f"current_{k}": np.asarray(v) for k, v in obs.images.items()},
                    **{f"demo_{k}": np.asarray(v) for k, v in obs.incontext_images.items()},
                    "demo_states": np.asarray(obs.incontext_states),
                    "demo_actions": np.asarray(obs.incontext_actions),
                    "current_state": np.asarray(obs.state),
                    "tokens": np.asarray(obs.tokenized_prompt),
                    "language_mask": np.asarray(obs.tokenized_prompt_mask),
                    "prefix_mask": np.asarray(mask),
                },
            )
            (directory / "audit.json").write_text(
                json.dumps(
                    {
                        "task": self.tasks[self.active],
                        "selected_episode": selected,
                        "prefix_shape": list(tokens.shape),
                        "valid_prefix_tokens": int(np.asarray(mask).sum()),
                        "demo_blocks": blocks,
                        "demo_intervention": "none",
                        "input_sha256": array_hash(inputs),
                    },
                    indent=2,
                )
                + "\n"
            )
        raw = self.policy._sample_actions(rng, obs)
        actions = executable(self.policy, inputs, raw)
        if not np.isfinite(actions).all():
            raise FloatingPointError("Nonfinite action")
        result = {
            "actions": actions,
            "benchmark_audit": {
                "task_index": self.active,
                "selected_episode": selected,
                "inference_index": self.calls,
                "rng_key": np.asarray(jax.random.key_data(rng)).tolist(),
                "protocol": PROTOCOL,
            },
        }
        self.calls += 1
        self.total += 1
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--port", type=int, default=8130)
    p.add_argument("--checkpoint", default="/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999")
    p.add_argument("--source", default="logs/attention_capture/experiment0_consistency_20260923_143830")
    args = p.parse_args()
    policy = BenchmarkPolicy(args)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()
