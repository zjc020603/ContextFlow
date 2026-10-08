import json

import pytest

from examples.libero.summarize_position_ablation import LAYOUTS
from examples.libero.summarize_position_ablation import MODES
from examples.libero.summarize_position_ablation import TASKS
from examples.libero.summarize_position_ablation import demo_targets
from examples.libero.summarize_position_ablation import summarize


def write_fixture(root):
    for layout in LAYOUTS:
        for mode in MODES:
            directory = root / layout / mode
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(
                json.dumps(
                    {
                        "scene_task": "tomato_sauce",
                        "layout": layout,
                        "num_trials_per_task": 1,
                    }
                )
            )
            rows = []
            for task in TASKS:
                identity, occupant = demo_targets(task, mode, layout)
                demo_id = {"milk": 25, "tomato_sauce": 28, None: None}[identity]
                # Synthetic policy follows the old position, so swapped success is other_only for correct.
                goals = {name: name == occupant for name in TASKS}
                label = "neither" if occupant is None else "language_only" if occupant == task else "other_only"
                rows.append(
                    {
                        "mode": mode,
                        "task": task,
                        "episode_index": 0,
                        "scene_task": "tomato_sauce",
                        "layout": layout,
                        "position_intervention": {"source_state_sha256": "same-source", "settled_state_sha256": layout},
                        "initial_state_sha256": layout,
                        "initial_observation_sha256": layout,
                        "environment_seed": 28007,
                        "steps": 280,
                        "ever_in_basket": goals,
                        "language_success": goals[task],
                        "other_success": any(value for name, value in goals.items() if name != task),
                        "outcome": label,
                        "video": f"{mode}/{task}.mp4",
                        "trajectory": f"{mode}/{task}.npz",
                        "context_requests": [
                            {
                                "rng_components": [0, 25 if task == "milk" else 28, 0, 0],
                                "selected_episode": [] if identity is None else [814 if identity == "milk" else 821],
                                "demo_task_index": demo_id,
                            }
                        ],
                    }
                )
                (directory / (task + ".mp4")).touch()
                (directory / (task + ".npz")).touch()
            (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_summary_distinguishes_object_from_old_position_and_has_paired_transitions(tmp_path):
    write_fixture(tmp_path)
    result = summarize(tmp_path)
    assert len(result["comparison"]) == 12
    for row in result["comparison"]:
        if row["context_mode"] == "no":
            assert row["demo_identity_successes"] is None
        elif row["layout"] == "swap_milk_tomato":
            assert row["demo_identity_successes"] == 0
            assert row["demo_position_occupant_successes"] == 1
    assert result["paired_original_to_swapped"]["milk/correct"] == {"language_only -> other_only": 1}


@pytest.mark.parametrize("tamper", ["source", "demo", "observation"])
def test_summary_rejects_unpaired_experiment(tmp_path, tamper):
    write_fixture(tmp_path)
    path = tmp_path / "swap_milk_tomato/correct/episodes.jsonl"
    rows = list(map(json.loads, path.read_text().splitlines()))
    if tamper == "source":
        rows[0]["position_intervention"]["source_state_sha256"] = "wrong-source"
    elif tamper == "demo":
        rows[0]["context_requests"][0]["selected_episode"] = [999]
    else:
        rows[0]["initial_observation_sha256"] = "wrong-observation"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="[Uu]npaired|Source state|Layout changed"):
        summarize(tmp_path)
