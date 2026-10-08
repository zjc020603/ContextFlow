"""Replay existing position trials and export exact pre-inference observations.

Run in the LIBERO environment. No policy inference and no new environment actions:
only recorded actions are replayed and checked against the saved trajectories.
"""
# ruff: noqa: SLF001

import argparse
import hashlib
import json
import pathlib

from libero.libero import benchmark
import numpy as np

from examples.libero import context_ablation as experiment
from examples.libero import position_intervention


def prepare(source, output, episodes=(0, 1, 2, 3, 4), steps=(0, 80, 160)):
    source, output = pathlib.Path(source).resolve(), pathlib.Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(output / "manifest.json")
    if any(step < 0 or step % 5 for step in steps):
        raise ValueError("Select pre-inference steps aligned to the 5-step replan interval")
    output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    manifest = []
    for layout in ("original", "swap_milk_tomato"):
        for mode in experiment.MODES:
            directory = source / layout / mode
            cfg = json.loads((directory / "config.json").read_text())
            rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
            rows = [row for row in rows if row["episode_index"] in episodes]
            env, _ = experiment.baseline._get_libero_env(suite.get_task(rows[0]["scene_id"]), 256, cfg["seed"])
            try:
                initial_states = suite.get_task_init_states(rows[0]["scene_id"])
                for row in rows:
                    np.random.seed(row["environment_seed"])
                    env.seed(row["environment_seed"])
                    env.reset()
                    env.set_init_state(initial_states[row["initial_state_index"]])
                    obs, meta = position_intervention.apply_layout(env, layout)
                    assert meta["source_state_sha256"] == row["position_intervention"]["source_state_sha256"]
                    for _ in range(cfg["num_steps_wait"]):
                        obs, _, _, _ = env.step(experiment.baseline.LIBERO_DUMMY_ACTION)
                    assert position_intervention.state_hash(env.get_sim_state()) == row["initial_state_sha256"]
                    digest = hashlib.sha256()
                    for key in (
                        "agentview_image",
                        "robot0_eye_in_hand_image",
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ):
                        digest.update(np.asarray(obs[key]).tobytes())
                    assert digest.hexdigest() == row["initial_observation_sha256"]
                    emitted = []
                    max_error = 0.0
                    with np.load(source / layout / row["trajectory"]) as trace:
                        if max(steps) + 5 > len(trace["actions"]):
                            raise ValueError("Snapshot step exceeds saved action horizon")
                        for step in range(max(steps) + 5):
                            if step in steps:
                                request = experiment.make_request(
                                    obs,
                                    row,
                                    row["context_requests"][step // 5]["demo_task_index"] or row["index"],
                                    mode,
                                    row["episode_index"],
                                    step // 5,
                                    experiment.Args(policy_seed=cfg["policy_seed"]),
                                )
                                case_id = f"{layout}/{mode}/{row['task']}/ep{row['episode_index']:03d}_step{step:03d}"
                                path = output / (case_id + ".npz")
                                path.parent.mkdir(parents=True, exist_ok=True)
                                data = {key: value for key, value in request.items() if isinstance(value, np.ndarray)}
                                data["reference_executed_actions"] = trace["actions"][step : step + 5]
                                np.savez_compressed(path, **data)
                                info = {
                                    key: value for key, value in request.items() if not isinstance(value, np.ndarray)
                                }
                                entry = {
                                    "case_id": case_id,
                                    "npz": str(path.relative_to(output)),
                                    "layout": layout,
                                    "mode": mode,
                                    "task": row["task"],
                                    "episode": row["episode_index"],
                                    "step": step,
                                    "request": info,
                                    "expected_selected_episode": row["context_requests"][step // 5]["selected_episode"],
                                    "source_initial_state_sha256": row["initial_state_sha256"],
                                }
                                emitted.append(entry)
                            obs, _, _, _ = env.step(trace["actions"][step].tolist())
                            actual = np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"]))
                            max_error = max(max_error, float(np.max(np.abs(actual - trace["eef_and_gripper"][step]))))
                            for i, name in enumerate(trace["scene_object_names"]):
                                pos = env.env.object_states_dict[str(name)].get_geom_state()["pos"]
                                max_error = max(
                                    max_error, float(np.max(np.abs(pos - trace["scene_object_positions"][step, i])))
                                )
                                predicate = bool(env.env._eval_predicate(("in", str(name), "basket_1_contain_region")))
                                assert predicate == bool(trace["scene_in_basket"][step, i])
                            if max_error > 1e-7:
                                raise AssertionError(
                                    f"Replay diverged: {layout}/{mode}/{row['task']}/{row['episode_index']}: {max_error}"
                                )
                    for entry in emitted:
                        entry["replay_max_error"] = max_error
                    manifest.extend(emitted)
                    print(
                        f"Replayed {layout}/{mode}/{row['task']}/ep{row['episode_index']:03d}: error={max_error}, snapshots={len(emitted)}",
                        flush=True,
                    )
            finally:
                env.close()
    expected = 12 * len(episodes) * len(steps)
    assert len(manifest) == expected
    (output / "manifest.json").write_text(
        json.dumps(
            {"source": str(source), "episodes": list(episodes), "steps": list(steps), "cases": manifest}, indent=2
        )
        + "\n"
    )
    print(f"Exported {len(manifest)} verified observation snapshots", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 80, 160])
    args = parser.parse_args()
    prepare(args.source, args.output, tuple(args.episodes), tuple(args.steps))
