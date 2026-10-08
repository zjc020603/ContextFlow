"""Paired state/state+action ablations using the established position trial runner."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from openpi_client import websocket_client_policy

from examples.libero import context_ablation as context

INTERVENTIONS = {name: "zero_compressed_demo_" + name + "_keep_mask" for name in ("state", "state_action")}


class DemoModalityClient:
    def __init__(self, host, port, layout, ablation):
        self.intervention = INTERVENTIONS[ablation]
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.layout = layout
        meta = self.client.get_server_metadata()
        if meta.get("demo_modality_intervention") != self.intervention or not meta.get("baseline_gate_passed"):
            raise ValueError("Connect to the gated demo-modality ablation server")

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def infer(self, request):
        if request["context_ablation"]["mode"] not in ("correct", "wrong"):
            raise ValueError("Pilot runs correct/wrong only; no-context is reused as reference")
        response = self.client.infer(
            {**request, "demo_modality_ablation": {"layout": self.layout, "intervention": self.intervention}}
        )
        if response["context_ablation"].get("demo_modality_intervention") != self.intervention:
            raise AssertionError("Server did not confirm the requested intervention")
        return response


def run(args):
    root = Path(args.output)
    if not 1 <= args.trials <= 5:
        raise ValueError("This pilot supports initial states 0..4 (at most five trials)")
    for layout in ("original", "swap_milk_tomato"):
        for mode in ("correct", "wrong"):
            trial_args = context.Args(
                host=args.host,
                port=args.port,
                mode=mode,
                num_trials_per_task=args.trials,
                output_dir=str(root / layout),
                scene_task="tomato_sauce",
                layout=layout,
                seed=7,
                policy_seed=0,
                max_steps=280,
                replan_steps=5,
                num_steps_wait=10,
            )
            # The established runner saves all objects' trajectories/predicates and 20 FPS videos.
            context.run(
                trial_args,
                client_factory=lambda host, port, layout=layout: DemoModalityClient(host, port, layout, args.ablation),
            )
            (root / layout / mode / "modality_ablation_config.json").write_text(
                json.dumps(
                    {
                        **dataclasses.asdict(trial_args),
                        "demo_modality_intervention": INTERVENTIONS[args.ablation],
                        "attention_environment_steps": [0, 80, 160],
                        "out_of_distribution_caveat": True,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--ablation", choices=tuple(INTERVENTIONS), required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
