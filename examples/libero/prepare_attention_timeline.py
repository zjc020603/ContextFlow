"""Recover all control frames and planning observations from verified saved rollouts.

Run in the LIBERO environment. Replays saved actions, never new policy actions.
"""
# ruff: noqa: SLF001

import argparse
import hashlib
import json
from pathlib import Path

from libero.libero import benchmark
import numpy as np

from examples.libero import context_ablation as experiment
from examples.libero import position_intervention


def prepare(source, output, episodes):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(output / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    cases, rollouts = [], []
    # Episode first: the three primary trajectories become available together.
    for episode in episodes:
        for mode in ("correct", "no", "wrong"):
            directory = source / "original" / mode
            cfg = json.loads((directory / "config.json").read_text())
            row = next(
                r
                for r in map(json.loads, (directory / "episodes.jsonl").read_text().splitlines())
                if r["episode_index"] == episode and r["task"] == "milk"
            )
            env, _ = experiment.baseline._get_libero_env(suite.get_task(row["scene_id"]), 256, cfg["seed"])
            try:
                initial_states = suite.get_task_init_states(row["scene_id"])
                np.random.seed(row["environment_seed"])
                env.seed(row["environment_seed"])
                env.reset()
                env.set_init_state(initial_states[row["initial_state_index"]])
                obs, meta = position_intervention.apply_layout(env, "original")
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
                front, wrist, states, inside = [], [], [], []
                emitted = []
                error = 0.0
                with np.load(source / "original" / row["trajectory"]) as trace:
                    horizon = len(trace["actions"])
                    assert horizon == 280
                    assert cfg["replan_steps"] == 5
                    for step in range(horizon + 1):
                        front.append(experiment.image_for_policy(obs, "agentview_image"))
                        wrist.append(experiment.image_for_policy(obs, "robot0_eye_in_hand_image"))
                        states.append(np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"])))
                        inside.append(
                            [
                                bool(env.env._eval_predicate(("in", str(name), "basket_1_contain_region")))
                                for name in trace["scene_object_names"]
                            ]
                        )
                        if step == horizon:
                            break
                        if step % 5 == 0:
                            request = experiment.make_request(
                                obs,
                                row,
                                row["context_requests"][step // 5]["demo_task_index"] or row["index"],
                                mode,
                                episode,
                                step // 5,
                                experiment.Args(policy_seed=cfg["policy_seed"]),
                            )
                            case_id = f"original/{mode}/milk/ep{episode:03d}_step{step:03d}"
                            path = output / (case_id + ".npz")
                            path.parent.mkdir(parents=True, exist_ok=True)
                            data = {key: value for key, value in request.items() if isinstance(value, np.ndarray)}
                            data["reference_executed_actions"] = trace["actions"][step : step + 5].copy()
                            np.savez_compressed(path, **data)
                            entry = {
                                "case_id": case_id,
                                "npz": str(path.relative_to(output)),
                                "layout": "original",
                                "mode": mode,
                                "task": "milk",
                                "episode": episode,
                                "step": step,
                                "request": {k: v for k, v in request.items() if not isinstance(v, np.ndarray)},
                                "expected_selected_episode": row["context_requests"][step // 5]["selected_episode"],
                                "source_initial_state_sha256": row["initial_state_sha256"],
                            }
                            emitted.append(entry)
                        obs, _, _, _ = env.step(trace["actions"][step].tolist())
                        actual = np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"]))
                        error = max(error, float(np.max(np.abs(actual - trace["eef_and_gripper"][step]))))
                        for i, name in enumerate(trace["scene_object_names"]):
                            actual = env.env.object_states_dict[str(name)].get_geom_state()["pos"]
                            error = max(error, float(np.max(np.abs(actual - trace["scene_object_positions"][step, i]))))
                            assert bool(env.env._eval_predicate(("in", str(name), "basket_1_contain_region"))) == bool(
                                trace["scene_in_basket"][step, i]
                            )
                        if error > 1e-7:
                            raise AssertionError(f"Replay divergence {episode}/{mode}/{step}: {error}")
                    frame_path = output / "frames" / f"ep{episode:03d}_{mode}.npz"
                    frame_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(
                        frame_path,
                        front=front,
                        wrist=wrist,
                        state=states,
                        in_basket=inside,
                        object_names=trace["scene_object_names"],
                        actions=trace["actions"],
                    )
                for entry in emitted:
                    entry["replay_max_error"] = error
                cases.extend(emitted)
                rollouts.append(
                    {
                        "episode": episode,
                        "mode": mode,
                        "frames": str(frame_path.relative_to(output)),
                        "control_steps": horizon,
                        "frame_count": len(front),
                        "replay_max_error": error,
                        "source_trajectory": str(source / "original" / row["trajectory"]),
                        "first_in_basket_step": row.get("first_in_basket_step"),
                        "source_video": str(source / "original" / row["video"]),
                    }
                )
                print(f"Replayed episode {episode} / {mode}: 281 frames, 56 calls, error={error}", flush=True)
            finally:
                env.close()
    assert len(cases) == len(episodes) * 3 * 56
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "episodes": episodes,
                "steps": list(range(0, 280, 5)),
                "cases": cases,
                "rollouts": rollouts,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Complete: {len(cases)} calls, {len(rollouts)} full trajectories", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    prepare(args.source, args.output, args.episodes)
