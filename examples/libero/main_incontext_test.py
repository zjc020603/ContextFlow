import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
from openpi_client import websocket_client_policy
import pytest

from examples.libero import main_incontext as evaluation


def _prepare(monkeypatch, tmp_path):
    description = "pick up the milk and place it in the basket"
    suite = mock.Mock(n_tasks=1)
    suite.get_task.return_value = SimpleNamespace(language=description)
    suite.get_task_init_states.return_value = [None]
    monkeypatch.setattr(evaluation.benchmark, "get_benchmark_dict", lambda: {"libero_object": lambda: suite})
    monkeypatch.setattr(evaluation, "get_task_to_index_mapping", lambda path: {description: 25})
    monkeypatch.setattr(evaluation.tqdm, "tqdm", lambda items: items)
    env = mock.Mock()
    env.set_init_state.return_value = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    monkeypatch.setattr(evaluation, "_get_libero_env", lambda *args: (env, description))
    client = mock.Mock()
    client.infer.return_value = {"actions": np.zeros((5, 7))}
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", lambda *args: client)
    monkeypatch.setattr(evaluation.imageio, "mimwrite", mock.Mock())
    args = evaluation.Args(
        task_suite_name="libero_object",
        num_trials_per_task=1,
        num_steps_wait=0,
        results_out_path=str(tmp_path / "results.json"),
        video_out_path=str(tmp_path / "videos"),
    )
    return args, env, client


def test_inference_error_aborts_without_reporting_task_failure(monkeypatch, tmp_path):
    args, env, client = _prepare(monkeypatch, tmp_path)
    client.infer.side_effect = ConnectionError("policy connection lost")
    with pytest.raises(RuntimeError, match="execution error"):
        evaluation.eval_libero(args)
    env.close.assert_called_once_with()
    assert not (tmp_path / "results.json").exists()
    assert (tmp_path / "results.episodes.jsonl").read_text() == ""


def test_completed_trial_is_recorded_and_environment_closed(monkeypatch, tmp_path):
    args, env, _ = _prepare(monkeypatch, tmp_path)
    env.step.return_value = ({}, 1.0, True, {})
    evaluation.eval_libero(args)
    result = json.loads((tmp_path / "results.json").read_text())
    trial = json.loads((tmp_path / "results.episodes.jsonl").read_text())
    assert result["summary"]["total_episodes"] == 1
    assert result["summary"]["total_successes"] == 1
    assert trial["success"] is True
    assert trial["episode_index"] == 0
    assert evaluation.imageio.mimwrite.call_args.kwargs["fps"] == 20
    assert result["config"]["video_fps"] == 20
    env.close.assert_called_once_with()
