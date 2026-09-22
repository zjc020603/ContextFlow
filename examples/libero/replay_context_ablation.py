"""Replay saved actions to measure all scene objects, verifying the original trajectory."""
# ruff: noqa: SLF001  # Reuse baseline helpers and LIBERO's own goal predicate implementation.

import argparse
import collections
import hashlib
import json
import logging
import pathlib

from libero.libero import benchmark
import numpy as np

from examples.libero import main_incontext as baseline


def replay(root, *, resume=False, selected_mode=None, selected_task=None):
    root = pathlib.Path(root)
    output = root / "all_object_replay.jsonl"
    if output.exists() and not resume:
        raise FileExistsError(output)
    completed = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    completed_keys = {(row["mode"], row["task"], row["episode_index"]) for row in completed}
    if len(completed_keys) != len(completed):
        raise ValueError("Duplicate replay records")
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    summaries = {}
    for mode in ("correct", "no", "wrong"):
        if selected_mode is not None and mode != selected_mode:
            continue
        config = json.loads((root / mode / "config.json").read_text())
        rows = [json.loads(line) for line in (root / mode / "episodes.jsonl").read_text().splitlines()]
        for task in ("milk", "tomato_sauce"):
            if selected_task is not None and task != selected_task:
                continue
            task_rows = [row for row in rows if row["task"] == task]
            task_id = task_rows[0]["id"]
            env, _ = baseline._get_libero_env(suite.get_task(task_id), 256, config["seed"])
            try:
                initial_states = suite.get_task_init_states(task_id)
                counts = collections.Counter()
                max_error = 0.0
                names = [name for name in env.env.objects_dict if name != "basket_1"]
                for row in task_rows:
                    previous = next(
                        (
                            item
                            for item in completed
                            if (item["mode"], item["task"], item["episode_index"]) == (mode, task, row["episode_index"])
                        ),
                        None,
                    )
                    if previous is not None:
                        counts.update(name for name, success in previous["ever_in_basket"].items() if success)
                        max_error = max(max_error, previous["max_original_trajectory_error"])
                        continue
                    np.random.seed(row["environment_seed"])
                    env.seed(row["environment_seed"])
                    env.reset()
                    env.set_init_state(initial_states[row["initial_state_index"]])
                    for _ in range(config["num_steps_wait"]):
                        env.step(baseline.LIBERO_DUMMY_ACTION)
                    digest = hashlib.sha256(env.get_sim_state().tobytes()).hexdigest()
                    if digest != row["initial_state_sha256"]:
                        raise AssertionError("Replayed initial state differs from original")
                    ever = dict.fromkeys(names, False)
                    first = dict.fromkeys(names, None)
                    positions = {
                        name: env.env.object_states_dict[name].get_geom_state()["pos"].copy() for name in names
                    }
                    rise = dict.fromkeys(names, 0.0)
                    displacement = dict.fromkeys(names, 0.0)
                    episode_error = 0.0
                    with np.load(root / row["trajectory"]) as trace:
                        for step, action in enumerate(trace["actions"]):
                            obs, _, _, _ = env.step(action.tolist())
                            state = np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"]))
                            episode_error = max(
                                episode_error, float(np.max(np.abs(state - trace["eef_and_gripper"][step])))
                            )
                            for index, name in enumerate(trace["object_names"]):
                                pos = env.env.object_states_dict[str(name) + "_1"].get_geom_state()["pos"]
                                episode_error = max(
                                    episode_error, float(np.max(np.abs(pos - trace["object_positions"][step, index])))
                                )
                                inside = bool(
                                    env.env._eval_predicate(("in", str(name) + "_1", "basket_1_contain_region"))
                                )
                                if inside != bool(trace["in_basket"][step, index]):
                                    raise AssertionError("Replayed predicate differs from original")
                            if episode_error > 1e-7:
                                raise AssertionError(
                                    f"Replay diverged: {mode} {task} {row['episode_index']} {episode_error}"
                                )
                            for name in names:
                                inside = bool(env.env._eval_predicate(("in", name, "basket_1_contain_region")))
                                if inside and first[name] is None:
                                    first[name] = step + 1
                                ever[name] |= inside
                                pos = env.env.object_states_dict[name].get_geom_state()["pos"]
                                rise[name] = max(rise[name], float(pos[2] - positions[name][2]))
                                displacement[name] = max(
                                    displacement[name], float(np.linalg.norm(pos - positions[name]))
                                )
                    record = {
                        "mode": mode,
                        "task": task,
                        "episode_index": row["episode_index"],
                        "ever_in_basket": ever,
                        "first_in_basket_step": first,
                        "max_vertical_rise_m": rise,
                        "max_displacement_m": displacement,
                        "max_original_trajectory_error": episode_error,
                    }
                    with output.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    counts.update(name for name, success in ever.items() if success)
                    max_error = max(max_error, episode_error)
                    logging.info(
                        "Replay %s %s %d: %s", mode, task, row["episode_index"], [name for name in names if ever[name]]
                    )
                summaries[f"{mode}/{task}"] = {
                    "episodes": len(task_rows),
                    "object_in_basket_counts": dict(counts),
                    "max_original_trajectory_error": max_error,
                }
            finally:
                env.close()
    (root / "all_object_comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("correct", "no", "wrong"))
    parser.add_argument("--task", choices=("milk", "tomato_sauce"))
    args = parser.parse_args()
    replay(args.root, resume=args.resume, selected_mode=args.mode, selected_task=args.task)
