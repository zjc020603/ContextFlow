"""Repeat only the original per-language no-context cells, retaining demo masks."""

import argparse
import logging
from pathlib import Path

from openpi_client.websocket_client_policy import WebsocketClientPolicy

from examples.libero import context_ablation as context

INTERVENTION = "zero_all_encoded_demo_tokens_keep_mask"


class ZeroClient:
    def __init__(self, host, port):
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        if meta.get("no_context_intervention") != INTERVENTION or not meta.get("baseline_gate_passed"):
            raise ValueError("Requires the gated no-context-zero server")

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def infer(self, request):
        response = self.client.infer({**request, "no_context_intervention": INTERVENTION})
        if response["context_ablation"].get("no_context_intervention") != INTERVENTION:
            raise AssertionError("Wrong no-context definition")
        return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()
    context.run(
        context.Args(
            port=args.port,
            mode="no",
            num_trials_per_task=args.trials,
            output_dir=str(Path(args.output)),
            scene_task="per_language",
            layout="original",
            seed=7,
            policy_seed=0,
            max_steps=280,
            replan_steps=5,
            num_steps_wait=10,
            verify_interventions=True,
        ),
        client_factory=ZeroClient,
    )
