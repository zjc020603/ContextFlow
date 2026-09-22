"""Paired language versus demonstration experiment for the two unseen Object tasks."""
# ruff: noqa: SLF001  # Reuse baseline helpers and LIBERO's own goal predicate implementation.

import collections
import dataclasses
import hashlib
import json
import logging
import pathlib
import time
from typing import Literal

import imageio
from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

from examples.libero import main_incontext as baseline

TASKS = {
    "milk": "pick up the milk and place it in the basket",
    "tomato_sauce": "pick up the tomato sauce and place it in the basket",
}
MODES = ("correct", "no", "wrong")


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    mode: Literal["all", "correct", "no", "wrong"] = "all"
    num_trials_per_task: int = 25
    output_dir: str = "logs/context_ablation"
    seed: int = 7
    policy_seed: int = 0
    max_steps: int = 280
    num_steps_wait: int = 10
    replan_steps: int = 5
    # Validate identical observations/noise across interventions before rollouts.
    verify_interventions: bool = False


def image_for_policy(obs, name):
    image = np.ascontiguousarray(obs[name][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))


def make_request(obs, task, demo_task, mode, episode, replan, args):
    return {
        "observation/image": image_for_policy(obs, "agentview_image"),
        "observation/wrist_image": image_for_policy(obs, "robot0_eye_in_hand_image"),
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                baseline._quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": task["description"],
        "task_index": task["index"],
        "split": "test",
        "context_ablation": {
            "mode": mode,
            "demo_task_index": demo_task,
            "rng_components": [args.policy_seed, task["index"], episode, replan],
        },
    }


def object_status(env):
    inside, positions = {}, {}
    for name in TASKS:
        inside[name] = bool(env.env._eval_predicate(("in", name + "_1", "basket_1_contain_region")))
        positions[name] = env.env.object_states_dict[name + "_1"].get_geom_state()["pos"].copy()
    return inside, positions


def outcome(ever, target):
    other = next(name for name in TASKS if name != target)
    if ever[target] and ever[other]:
        return "both"
    if ever[target]:
        return "language_only"
    if ever[other]:
        return "other_only"
    return "neither"


def scene_object_status(env):
    """Include distractors so a wrong-object placement is not labeled only as failure."""
    names = [name for name in env.env.objects_dict if name != "basket_1"]
    return {name: bool(env.env._eval_predicate(("in", name, "basket_1_contain_region"))) for name in names}


def verify(client, request):
    responses = [client.infer(request), client.infer(request)]
    np.testing.assert_array_equal(responses[0]["actions"], responses[1]["actions"])
    return {"same_observation_and_rng_repeats_exactly": True}


def run(args):
    if not 1 <= args.num_trials_per_task <= 50 or args.replan_steps < 1 or args.max_steps < 1:
        raise ValueError("Trials must be 1..50; steps must be positive")
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    mapping = baseline.get_task_to_index_mapping(baseline.LIBERO_TASKS_JSONL)
    tasks = {}
    for name, description in TASKS.items():
        assert description in baseline.LIBERO_UNSEEN_TASKS
        ids = [i for i in range(suite.n_tasks) if suite.get_task(i).language == description]
        if len(ids) != 1:
            raise ValueError(f"Expected one task for {description}")
        tasks[name] = {"id": ids[0], "index": mapping[description], "description": description}
    root = pathlib.Path(args.output_dir)
    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        if (root / mode / "episodes.jsonl").exists():
            raise FileExistsError(f"Refusing to overwrite {root / mode}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    if client.get_server_metadata().get("context_ablation_protocol") != 1:
        raise ValueError("Server must enable --policy.context-ablation")
    for mode in modes:
        directory = root / mode
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(
                {
                    **dataclasses.asdict(args),
                    "mode": mode,
                    "tasks": tasks,
                    "termination": "fixed horizon; neither goal terminates early",
                    "success_measure": "LIBERO In predicate at any post-action step",
                    "video_fps": 20,
                },
                indent=2,
            )
            + "\n"
        )
        records = []
        for name, task in tasks.items():
            other = next(key for key in TASKS if key != name)
            demo_task = tasks[other]["index"] if mode == "wrong" else task["index"]
            env, _ = baseline._get_libero_env(suite.get_task(task["id"]), 256, args.seed)
            try:
                initial_states = suite.get_task_init_states(task["id"])
                for episode in range(args.num_trials_per_task):
                    started = time.monotonic()
                    env_seed = args.seed + task["index"] * 1000 + episode
                    np.random.seed(env_seed)
                    env.seed(env_seed)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode])
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step(baseline.LIBERO_DUMMY_ACTION)
                    initial_inside, _ = object_status(env)
                    if any(initial_inside.values()):
                        raise ValueError("An object is already in the basket before intervention")
                    initial_state = env.get_sim_state().copy()
                    observation_hash = hashlib.sha256()
                    for key in (
                        "agentview_image",
                        "robot0_eye_in_hand_image",
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ):
                        observation_hash.update(np.asarray(obs[key]).tobytes())
                    plan = collections.deque()
                    ever = dict.fromkeys(TASKS, False)
                    scene_ever = dict.fromkeys(scene_object_status(env), False)
                    first = dict.fromkeys(TASKS, None)
                    frames, actions, states, positions, predicates, context_records = [], [], [], [], [], []
                    checks = {}
                    for step in range(args.max_steps):
                        if not plan:
                            request = make_request(obs, task, demo_task, mode, episode, len(context_records), args)
                            if args.verify_interventions and episode == 0 and step == 0:
                                checks = verify(client, request)
                            response = client.infer(request)
                            context = response["context_ablation"]
                            if context["mode"] != mode or context["task_index"] != task["index"]:
                                raise ValueError("Server returned incorrect context intervention")
                            expected_demo = None if mode == "no" else demo_task
                            if context["demo_task_index"] != expected_demo:
                                raise ValueError("Server selected an incorrect demo task")
                            if not np.isfinite(response["actions"]).all():
                                raise FloatingPointError("Nonfinite actions")
                            plan.extend(response["actions"][: args.replan_steps])
                            context_records.append(context)
                        action = plan.popleft()
                        obs, _, _, _ = env.step(action.tolist())
                        inside, pos = object_status(env)
                        for key, value in scene_object_status(env).items():
                            scene_ever[key] = scene_ever[key] or value
                        # Cross-check custom language metric against the original environment goal.
                        if bool(env.check_success()) != inside[name]:
                            raise AssertionError("Language metric disagrees with LIBERO goal")
                        for key in TASKS:
                            if inside[key] and first[key] is None:
                                first[key] = step + 1
                            ever[key] = ever[key] or inside[key]
                        frames.append(image_for_policy(obs, "agentview_image"))
                        actions.append(action)
                        states.append(np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"])))
                        positions.append(np.stack([pos[key] for key in TASKS]))
                        predicates.append([inside[key] for key in TASKS])
                    label = outcome(ever, name)
                    stem = f"ep{episode:03d}_{label}"
                    task_dir = directory / "videos" / task["description"].replace(" ", "_")
                    task_dir.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(task_dir / (stem + ".mp4"), frames, fps=20)
                    trajectory_dir = directory / "trajectories" / name
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        trajectory_dir / (stem + ".npz"),
                        initial_sim_state=initial_state,
                        actions=actions,
                        eef_and_gripper=states,
                        object_positions=positions,
                        object_names=list(TASKS),
                        in_basket=predicates,
                    )
                    record = {
                        "task": name,
                        **task,
                        "episode_index": episode,
                        "initial_state_index": episode,
                        "environment_seed": env_seed,
                        "mode": mode,
                        "language_success": ever[name],
                        "other_success": ever[other],
                        "ever_in_basket": ever,
                        "scene_ever_in_basket": scene_ever,
                        "final_in_basket": inside,
                        "first_in_basket_step": first,
                        "outcome": label,
                        "initial_observation_sha256": observation_hash.hexdigest(),
                        "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
                        "context_requests": context_records,
                        "verification": checks,
                        "steps": args.max_steps,
                        "elapsed_seconds": time.monotonic() - started,
                        "video": str((task_dir / (stem + ".mp4")).relative_to(root)),
                        "trajectory": str((trajectory_dir / (stem + ".npz")).relative_to(root)),
                    }
                    with (directory / "episodes.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    records.append(record)
                    logging.info(
                        "%s %s %d/%d: %s (%.1fs)",
                        mode,
                        name,
                        episode + 1,
                        args.num_trials_per_task,
                        label,
                        record["elapsed_seconds"],
                    )
            finally:
                env.close()
        summary = {}
        for name in TASKS:
            rows = [row for row in records if row["task"] == name]
            summary[name] = {
                "episodes": len(rows),
                "language_successes": sum(row["language_success"] for row in rows),
                "other_successes": sum(row["other_success"] for row in rows),
                "outcomes": dict(collections.Counter(row["outcome"] for row in rows)),
            }
        (directory / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(tyro.cli(Args))
