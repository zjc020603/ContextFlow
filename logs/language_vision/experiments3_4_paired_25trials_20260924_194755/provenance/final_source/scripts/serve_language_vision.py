"""Original-checkpoint server for language / live-vision interventions (experiments 3, 4)."""

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
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import config
from scripts.check_attention_capture import array_hash, expected_mask, validate_attention

PROMPTS = {
    "milk": "pick up the milk and place it in the basket",
    "tomato_sauce": "pick up the tomato sauce and place it in the basket",
    "empty": "",
}


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


class Policy:
    def __init__(self, args):
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        cfg = config.get_config("ContextFlow")
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, policy_local_files_only=True))
        self.policy = policy_config.create_trained_policy_incontext(
            cfg, args.checkpoint, inference_dtype="float32", context_ablation=True
        )
        self.model = self.policy._model_for_diagnostics
        self.capture = nnx_utils.module_jit(self.model.sample_actions_with_attention)
        self.weight_hash = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
        self.calls = 0
        source = Path(args.source)
        manifest = json.loads((source / "observations/manifest.json").read_text())
        checks = []
        for case in manifest["cases"]:
            if not (
                case["layout"] == "original"
                and case["task"] == "milk"
                and case["episode"] == 0
                and case["step"] == 0
                and case["mode"] in ("correct", "wrong")
            ):
                continue
            with np.load(source / "observations" / case["npz"]) as z:
                req = {k: z[k] for k in z.files if k != "reference_executed_actions"}
            req.update(case["request"])
            inputs, info, rng = self.policy.prepare_inputs(req)
            obs = model_lib.ObservationIncontext.from_dict(inputs)
            ordinary = self.policy._sample_actions(rng, obs)
            with np.load(source / "validation_attempt3" / case["case_id"] / "capture.npz") as z:
                exact(ordinary, z["ordinary_actions"])
            captured, trace = self.capture(rng, obs)
            exact(ordinary, captured)
            checks.append({"case": case["case_id"], "historical_and_capture_exact": True})
        assert len(checks) == 2
        self.metadata = {**self.policy.metadata, "language_vision_protocol": 1}
        (self.root / "READY.json").write_text(json.dumps({"checks": checks, "parameter_sha256": self.weight_hash}))

    def evaluate(self, req):
        inputs, info, rng = self.policy.prepare_inputs(req)
        obs = model_lib.ObservationIncontext.from_dict(inputs)
        raw = np.asarray(self.policy._sample_actions(rng, obs))
        actions = self.policy._output_transform(
            {"state": np.asarray(inputs["state"])[0].copy(), "actions": raw[0].copy()}
        )["actions"]
        assert np.isfinite(actions).all()
        nonlanguage = {k: v for k, v in inputs.items() if k not in ("tokenized_prompt", "tokenized_prompt_mask")}
        details = {
            **info,
            "nonlanguage_hash": array_hash(nonlanguage),
            "demo_hash": array_hash({k: v for k, v in inputs.items() if k.startswith("dem_prompt")}),
            "noise_hash": array_hash(jax.random.normal(rng, raw.shape)),
            "language_tokens": np.asarray(obs.tokenized_prompt).tolist(),
            "language_mask": np.asarray(obs.tokenized_prompt_mask).tolist(),
        }
        return actions, raw, obs, rng, details

    def save_capture(self, path, req, values):
        path.mkdir(parents=True, exist_ok=False)
        actions, raw, obs, rng, details = values
        captured, trace = self.capture(rng, obs)
        exact(raw, captured)
        trace = jax.tree.map(np.asarray, trace)
        layout = token_layout(self.model, obs)
        checks = validate_attention(
            trace, layout, expected_mask(self.model, obs), details["mode"], self.model.PaliGemma.llm.module.embed_dtype
        )
        # Retain all heads/queries at initial frames and all episode-0 snapshots; aggregate every capture.
        full = (
            {"probabilities": trace["probabilities"]}
            if any(
                part.startswith("ep000_step") or (part.startswith("ep") and part.endswith("_step000"))
                for part in path.parts
            )
            else {}
        )
        np.savez_compressed(
            path / "capture.npz",
            **full,
            mean_attention=trace["probabilities"].astype(np.float64)[:, :, 0, :, 1:6].mean((2, 3)),
            actions=actions,
            raw_actions=raw,
            noise=trace["noise"],
            prefix_mask=trace["prefix_mask"],
            front=req["observation/image"],
            wrist=req["observation/wrist_image"],
        )
        (path / "metadata.json").write_text(
            json.dumps({**details, "layout": layout, "checks": checks, "instruction": req["prompt"]}, indent=2)
        )

    def infer(self, request):
        if request.get("finalize_language_vision"):
            after = array_hash(nnx.state(self.model, nnx.Param).to_pure_dict())
            assert after == self.weight_hash
            result = {"calls": self.calls, "parameter_sha256": after, "unchanged": True}
            (self.root / "FINISHED.json").write_text(json.dumps(result))
            return result
        req = dict(request)
        meta = req.pop("language_vision")
        condition = meta["condition"]
        if not condition.replace("_", "").isalnum():
            raise ValueError("Unsafe condition name")
        ep, step = meta["episode"], meta["step"]
        assert req["task_index"] == 25 and req["context_ablation"]["rng_components"] == [0, 25, ep, step // 5]
        assert req["prompt"] in PROMPTS.values()
        values = self.evaluate(req)
        actions, raw, obs, rng, details = values
        if ep < 5 and step in (0, 80, 160):
            base = self.root / "attention" / condition / f"ep{ep:03d}_step{step:03d}"
            self.save_capture(base, req, values)
            if meta.get("paired_languages"):
                paired = {meta["language"]: values}
                for label, prompt in PROMPTS.items():
                    if label not in paired:
                        changed = {**req, "prompt": prompt}
                        paired[label] = self.evaluate(changed)
                        self.save_capture(base / ("language_" + label), changed, paired[label])
                assert len({v[-1]["nonlanguage_hash"] for v in paired.values()}) == 1
                assert len({v[-1]["noise_hash"] for v in paired.values()}) == 1
                np.savez_compressed(base / "paired_language_actions.npz", **{k: v[0] for k, v in paired.items()})
        self.calls += 1
        return {"actions": actions, "context_ablation": details}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999")
    p.add_argument("--source", default="logs/attention_capture/experiment0_consistency_20260923_143830")
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    policy = Policy(args)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()
