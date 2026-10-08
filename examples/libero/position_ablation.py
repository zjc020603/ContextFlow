"""Run or validate the fixed-scene, paired position/context experiment."""

import dataclasses
import json
import logging
import pathlib
from typing import Literal

import imageio
from libero.libero import benchmark
import numpy as np
import tyro

from examples.libero import context_ablation as context
from examples.libero import position_intervention as intervention


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    layout: Literal["all", "original", "swap_milk_tomato"] = "all"
    mode: Literal["all", "correct", "no", "wrong"] = "all"
    num_trials_per_task: int = 25
    output_dir: str = "logs/position_ablation"
    seed: int = 7
    policy_seed: int = 0
    max_steps: int = 280
    num_steps_wait: int = 10
    verify_interventions: bool = False
    # Validate all physical initializations without connecting to a policy server.
    validate_only: bool = False


def validate(args):
    root = pathlib.Path(args.output_dir) / "initialization_validation"
    root.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_id = next(i for i in range(suite.n_tasks) if suite.get_task(i).language == context.TASKS["tomato_sauce"])
    mapping = context.baseline.get_task_to_index_mapping(context.baseline.LIBERO_TASKS_JSONL)
    task_index = mapping[context.TASKS["tomato_sauce"]]
    initial_states = suite.get_task_init_states(task_id)
    env, _ = context.baseline._get_libero_env(suite.get_task(task_id), 256, args.seed)  # noqa: SLF001
    records = []
    try:
        for episode in range(args.num_trials_per_task):
            for layout in intervention.LAYOUTS:
                seed = args.seed + task_index * 1000 + episode
                np.random.seed(seed)
                env.seed(seed)
                env.reset()
                env.set_init_state(initial_states[episode])
                obs, meta = intervention.apply_layout(env, layout)
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(context.baseline.LIBERO_DUMMY_ACTION)
                intervention.validate_settled(env, meta)
                if any(context.scene_object_status(env).values()):
                    raise ValueError("An object is already in the basket")
                imageio.imwrite(
                    root / f"ep{episode:03d}_{layout}.png", context.image_for_policy(obs, "agentview_image")
                )
                records.append({"episode": episode, **meta})
                logging.info("Validated %s episode %d", layout, episode)
    finally:
        env.close()
    for index in range(0, len(records), 2):
        if records[index]["source_state_sha256"] != records[index + 1]["source_state_sha256"]:
            raise AssertionError("Layout source states differ")
    (root / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    logging.info("Validated all %d physical initializations", len(records))


def run(args):
    if not 1 <= args.num_trials_per_task <= 50 or args.num_steps_wait < 1:
        raise ValueError("Trials must be 1..50, settling steps must be positive")
    if args.validate_only:
        validate(args)
        return
    layouts = intervention.LAYOUTS if args.layout == "all" else (args.layout,)
    # Preflight every destination before running any condition.
    modes = context.MODES if args.mode == "all" else (args.mode,)
    for layout in layouts:
        for mode in modes:
            path = pathlib.Path(args.output_dir) / layout / mode / "episodes.jsonl"
            if path.exists():
                raise FileExistsError(path)
    for layout in layouts:
        context.run(
            context.Args(
                host=args.host,
                port=args.port,
                mode=args.mode,
                num_trials_per_task=args.num_trials_per_task,
                output_dir=str(pathlib.Path(args.output_dir) / layout),
                seed=args.seed,
                policy_seed=args.policy_seed,
                max_steps=args.max_steps,
                num_steps_wait=args.num_steps_wait,
                verify_interventions=args.verify_interventions,
                scene_task="tomato_sauce",
                layout=layout,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(tyro.cli(Args))
