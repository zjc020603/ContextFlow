"""Full-context Goal/Long benchmark; standard goals plus noninvasive predicate traces."""
# ruff: noqa: SLF001

import argparse
import collections
import hashlib
import json
import logging
from pathlib import Path
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import bddl_utils
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from examples.libero import main_incontext as original
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS

PROTOCOL = "goal_long_full_correct_context_v1"
LIMITS = {"libero_goal": 300, "libero_10": 520}


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def make_manifest(output):
    mapping = original.get_task_to_index_mapping(original.LIBERO_TASKS_JSONL)
    episodes = [json.loads(s) for s in (original.LIBERO_TASKS_JSONL.parent / "episodes.jsonl").read_text().splitlines()]
    tasks = []
    root = Path(output)
    for name, limit in LIMITS.items():
        suite = benchmark.get_benchmark_dict()[name]()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            parsed = bddl_utils.robosuite_parse_problem(str(bddl))
            demos = [e for e in episodes if task.language in e["tasks"]]
            retained = [e for e in demos if not any(t in LIBERO_UNSEEN_TASKS for t in e["tasks"])]
            unseen = task.language in LIBERO_UNSEEN_TASKS
            assert len(demos) > 0
            assert bool(retained) != unseen
            states = suite.get_task_init_states(i)
            record = {
                "suite": name,
                "suite_task_id": i,
                "dataset_task_index": mapping[task.language],
                "instruction": task.language,
                "split": "unseen" if unseen else "seen",
                "max_steps": limit,
                "goals": parsed["goal_state"],
                "initial_states_shape": list(states.shape),
                "initial_states_sha256": digest(states),
                "dataset_episodes": len(demos),
                "retained_training_episodes_in_released_config": len(retained),
                "demo_episode": demos[0]["episode_index"],
                "demo_length": demos[0]["length"],
                "demo_image_frame_indices": np.linspace(0, demos[0]["length"] - 1, 8, dtype=int).tolist(),
                "demo_state_action_frame_indices": np.linspace(
                    0, demos[0]["length"] - 1, min(128, demos[0]["length"]), dtype=int
                ).tolist(),
                "bddl_file": str(bddl),
                "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
            }
            record["task_key"] = f"{name}/task{mapping[task.language]:02d}_{task.language.replace(' ', '_')}"
            tasks.append(record)
    assert len(tasks) == 20
    assert {t["dataset_task_index"] for t in tasks if t["split"] == "unseen"} == {1, 5, 10, 17}
    root.mkdir(parents=True, exist_ok=True)
    (root / "task_manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "tasks": tasks}, indent=2) + "\n")


def request_from_obs(obs, task):
    from openpi_client import image_tools

    def image(name):
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(obs[name][::-1, ::-1]), 224, 224)
        )

    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], original._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": task["instruction"],
        "task_index": task["dataset_task_index"],
        "split": "test",
    }


def goal_metrics(predicates):
    p = np.asarray(predicates, dtype=bool)
    if p.ndim != 2 or not len(p) or p.shape[1] == 0:
        raise ValueError("Expected nonempty [steps, goal clauses] predicates")
    return {
        "success": bool(p.all(axis=1).any()),
        "goal_ever": p.any(axis=0).tolist(),
        "goal_final": p[-1].tolist(),
        "max_simultaneous_goals": int(p.sum(axis=1).max()),
    }


def evaluate(args):
    tasks = json.loads((Path(args.root) / "task_manifest.json").read_text())["tasks"]
    task = next(t for t in tasks if t["dataset_task_index"] == args.task)
    directory = Path(args.root) / "tasks" / task["task_key"]
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                **task,
                "trials": args.trials,
                "environment_seed": 7,
                "policy_seed": 0,
                "policy_rng": "reset per task, continuous split across episodes/replans",
                "wait_steps": 10,
                "replan_steps": 5,
                "video_fps": 20,
                "termination": "stop at full LIBERO success or standard horizon",
                "protocol": PROTOCOL,
                "port": args.port,
            },
            indent=2,
        )
        + "\n"
    )
    np.random.seed(7)
    suite = benchmark.get_benchmark_dict()[task["suite"]]()
    env, _ = original._get_libero_env(suite.get_task(task["suite_task_id"]), 256, 7)
    states = suite.get_task_init_states(task["suite_task_id"])
    assert digest(states) == task["initial_states_sha256"]
    goals = env.env.parsed_problem["goal_state"]
    assert goals == task["goals"]
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    assert client.get_server_metadata().get("goal_long_protocol") == PROTOCOL
    start = client.infer({"benchmark_control": "start_task", "task_index": args.task})
    assert start["policy_seed"] == 0
    assert start["demo_episode"] == task["demo_episode"]
    records = []
    try:
        for episode in range(args.trials):
            started = time.monotonic()
            env.reset()
            obs = env.set_init_state(states[episode])
            source_state = env.get_sim_state().copy()
            for _ in range(10):
                obs, _, _, _ = env.step(original.LIBERO_DUMMY_ACTION)
            initial = env.get_sim_state().copy()
            initial_goals = [bool(env.env._eval_predicate(g)) for g in goals]
            if all(initial_goals):
                raise ValueError("Full goal already satisfied before policy action")
            first = request_from_obs(obs, task)
            initial_hash = {k: digest(v) for k, v in first.items() if isinstance(v, np.ndarray)}
            action_plan = collections.deque()
            frames, actions, predicates, ee, sim_states, rng_records = [], [], [], [], [], []
            for _step in range(task["max_steps"]):
                request = request_from_obs(obs, task)
                frames.append(request["observation/image"])
                if not action_plan:
                    response = client.infer(request)
                    info = response["benchmark_audit"]
                    assert info["task_index"] == args.task
                    assert info["selected_episode"] == [task["demo_episode"]]
                    action_chunk = response["actions"]
                    if action_chunk.shape != (50, 7) or not np.isfinite(action_chunk).all():
                        raise ValueError("Invalid model output")
                    action_plan.extend(action_chunk[:5])
                    rng_records.append(info)
                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                flags = [bool(env.env._eval_predicate(g)) for g in goals]
                assert bool(done) == bool(env.check_success()) == all(flags)
                actions.append(action)
                predicates.append(flags)
                ee.append(np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"])))
                sim_states.append(env.get_sim_state().copy())
                if done:
                    break
            metrics = goal_metrics(predicates)
            assert metrics["success"] == bool(done)
            stem = f"ep{episode:03d}_{'success' if done else 'failure'}"
            video = directory / "videos" / task["instruction"].replace(" ", "_") / (stem + ".mp4")
            video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(video, frames, fps=20)
            trajectory = directory / "trajectories" / (stem + ".npz")
            trajectory.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                trajectory,
                source_sim_state=source_state,
                initial_sim_state=initial,
                sim_states=sim_states,
                actions=actions,
                goal_predicates=predicates,
                eef_and_gripper=ee,
                initial_image=first["observation/image"],
                initial_wrist=first["observation/wrist_image"],
                initial_robot_state=first["observation/state"],
            )
            row = {
                "dataset_task_index": args.task,
                "episode": episode,
                **metrics,
                "steps": len(actions),
                "initial_goals": initial_goals,
                "initial_sim_state_sha256": digest(initial),
                "source_sim_state_sha256": digest(source_state),
                "initial_observation_sha256": initial_hash,
                "rng_records": rng_records,
                "video": str(video.relative_to(Path(args.root))),
                "trajectory": str(trajectory.relative_to(Path(args.root))),
                "elapsed_seconds": time.monotonic() - started,
            }
            with (directory / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            records.append(row)
            logging.info(
                "Task %d %s %d/%d success=%s goals=%s (%.1fs)",
                args.task,
                task["split"],
                episode + 1,
                args.trials,
                done,
                metrics["goal_ever"],
                row["elapsed_seconds"],
            )
    finally:
        env.close()
    audit = client.infer({"benchmark_control": "end_task", "task_index": args.task})
    assert audit["inferences"] == sum(len(r["rng_records"]) for r in records)
    (directory / "results.json").write_text(
        json.dumps(
            {
                "task": task,
                "episodes": len(records),
                "successes": sum(r["success"] for r in records),
                "goal_ever_counts": np.sum([r["goal_ever"] for r in records], axis=0).tolist(),
                "goal_final_counts": np.sum([r["goal_final"] for r in records], axis=0).tolist(),
                "goal_initial_counts": np.sum([r["initial_goals"] for r in records], axis=0).tolist(),
                "server_audit": audit,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", action="store_true")
    p.add_argument("--task", type=int)
    p.add_argument("--port", type=int, default=8130)
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()
    if args.manifest:
        make_manifest(args.root)
    else:
        if not 1 <= args.trials <= 50:
            raise ValueError("Trials must be 1..50")
        evaluate(args)
