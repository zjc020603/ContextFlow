import json
from types import SimpleNamespace
from unittest import mock

import numpy as np

from examples.libero import context_ablation as experiment


def test_wrong_context_records_both_goals_and_runs_full_horizon(monkeypatch, tmp_path):
    descriptions = list(experiment.TASKS.values())
    suite = mock.Mock(n_tasks=2)
    suite.get_task.side_effect = lambda index: SimpleNamespace(language=descriptions[index])
    suite.get_task_init_states.return_value = [np.zeros(8)]
    monkeypatch.setattr(experiment.benchmark, "get_benchmark_dict", lambda: {"libero_object": lambda: suite})
    monkeypatch.setattr(
        experiment.baseline, "get_task_to_index_mapping", lambda _: {descriptions[0]: 25, descriptions[1]: 28}
    )
    observation = {
        "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    envs = []

    def create_env(*args):
        env = mock.Mock()
        env.set_init_state.return_value = observation
        env.get_sim_state.return_value = np.zeros(8)
        env.step.return_value = (observation, 1, True, {})
        env.check_success.return_value = True
        envs.append(env)
        return env, ""

    monkeypatch.setattr(experiment.baseline, "_get_libero_env", create_env)
    monkeypatch.setattr(
        experiment,
        "object_status",
        lambda env: (
            dict.fromkeys(experiment.TASKS, env.step.call_count > 0),
            {name: np.zeros(3) for name in experiment.TASKS},
        ),
    )
    monkeypatch.setattr(experiment, "scene_object_status", lambda env: {"cream_cheese_1": env.step.call_count > 0})
    client = mock.Mock()
    client.get_server_metadata.return_value = {"context_ablation_protocol": 1}

    def infer(request):
        params = request["context_ablation"]
        assert params["demo_task_index"] != request["task_index"]
        assert request["prompt"] == descriptions[[25, 28].index(request["task_index"])]
        return {
            "actions": np.zeros((5, 7)),
            "context_ablation": {
                "mode": "wrong",
                "task_index": request["task_index"],
                "demo_task_index": params["demo_task_index"],
            },
        }

    client.infer.side_effect = infer
    monkeypatch.setattr(experiment.websocket_client_policy, "WebsocketClientPolicy", lambda *args: client)
    monkeypatch.setattr(experiment.imageio, "mimwrite", mock.Mock())
    experiment.run(
        experiment.Args(mode="wrong", num_trials_per_task=1, max_steps=2, num_steps_wait=0, output_dir=str(tmp_path))
    )
    rows = [json.loads(line) for line in (tmp_path / "wrong/episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["outcome"] == "both"
        assert row["language_success"]
        assert row["other_success"]
        assert row["steps"] == 2
        assert row["scene_ever_in_basket"]["cream_cheese_1"]
    for env in envs:
        assert env.step.call_count == 2  # Original task success must not stop the experiment early.
        env.close.assert_called_once_with()
