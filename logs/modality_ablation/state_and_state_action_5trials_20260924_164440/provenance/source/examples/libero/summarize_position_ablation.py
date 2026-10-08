"""Validate paired layouts and summarize object identity versus old-position outcomes."""

import argparse
import collections
import csv
import json
import pathlib

from examples.libero.summarize_context_ablation import summarize as summarize_context

LAYOUTS = ("original", "swap_milk_tomato")
MODES = ("correct", "no", "wrong")
TASKS = ("milk", "tomato_sauce")


def demo_targets(language, mode, layout):
    """Identity and current occupant of the demo's original region; None without demo."""
    if mode == "no":
        return None, None
    other = next(name for name in TASKS if name != language)
    identity = language if mode == "correct" else other
    occupant = identity if layout == "original" else next(name for name in TASKS if name != identity)
    return identity, occupant


def summarize(root):
    root = pathlib.Path(root)
    rows_by_key = {}
    counts = set()
    for layout in LAYOUTS:
        summarize_context(root / layout)
        for mode in MODES:
            directory = root / layout / mode
            config = json.loads((directory / "config.json").read_text())
            if config["scene_task"] != "tomato_sauce" or config["layout"] != layout:
                raise ValueError("Incorrect fixed-scene configuration")
            counts.add(config["num_trials_per_task"])
            for row in map(json.loads, (directory / "episodes.jsonl").read_text().splitlines()):
                if row["layout"] != layout or row["scene_task"] != "tomato_sauce":
                    raise ValueError("Incorrect scene/layout in episode")
                rows_by_key[layout, mode, row["task"], row["episode_index"]] = row
    if len(counts) != 1:
        raise ValueError("Unequal trial counts")
    n = counts.pop()
    for episode in range(n):
        base = rows_by_key["original", "correct", "milk", episode]
        source_hash = base["position_intervention"]["source_state_sha256"]
        for layout in LAYOUTS:
            layout_base = rows_by_key[layout, "correct", "milk", episode]
            for mode in MODES:
                for task in TASKS:
                    row = rows_by_key[layout, mode, task, episode]
                    if row["position_intervention"]["source_state_sha256"] != source_hash:
                        raise ValueError("Source state not paired across languages/layouts")
                    for field in ("initial_state_sha256", "initial_observation_sha256", "environment_seed", "steps"):
                        if row[field] != layout_base[field]:
                            raise ValueError(f"Unpaired within-layout {field}")
                    original = rows_by_key["original", mode, task, episode]
                    for field in ("environment_seed", "steps"):
                        if row[field] != original[field]:
                            raise ValueError(f"Layout changed {field}")
                    if row["position_intervention"]["settled_state_sha256"] != row["initial_state_sha256"]:
                        raise ValueError("Saved intervention state mismatch")
                    for field in ("rng_components", "selected_episode", "demo_task_index"):
                        if [r[field] for r in row["context_requests"]] != [
                            r[field] for r in original["context_requests"]
                        ]:
                            raise ValueError(f"Layout changed inference {field}")
    comparison = []
    transitions = {}
    for layout in LAYOUTS:
        for task in TASKS:
            for mode in MODES:
                rows = [rows_by_key[layout, mode, task, i] for i in range(n)]
                identity, occupant = demo_targets(task, mode, layout)
                comparison.append(
                    {
                        "layout": layout,
                        "language": task,
                        "context_mode": mode,
                        "episodes": n,
                        "demo": identity or "none",
                        "demo_original_position_occupant": occupant or "not_applicable",
                        "milk_in_basket": sum(row["ever_in_basket"]["milk"] for row in rows),
                        "tomato_sauce_in_basket": sum(row["ever_in_basket"]["tomato_sauce"] for row in rows),
                        "language_successes": sum(row["language_success"] for row in rows),
                        "demo_identity_successes": None
                        if identity is None
                        else sum(row["ever_in_basket"][identity] for row in rows),
                        "demo_position_occupant_successes": None
                        if occupant is None
                        else sum(row["ever_in_basket"][occupant] for row in rows),
                        "both": sum(row["outcome"] == "both" for row in rows),
                        "neither": sum(row["outcome"] == "neither" for row in rows),
                    }
                )
    for task in TASKS:
        for mode in MODES:
            pairs = [
                (rows_by_key["original", mode, task, i], rows_by_key["swap_milk_tomato", mode, task, i])
                for i in range(n)
            ]
            transitions[f"{task}/{mode}"] = dict(
                collections.Counter(a["outcome"] + " -> " + b["outcome"] for a, b in pairs)
            )
    result = {
        "comparison": comparison,
        "paired_original_to_swapped": transitions,
        "verification": "Same source states across all cells; identical settled states/observations within layout; identical noise and demo episodes across layouts within language/mode",
        "interpretation": "Counts describe basket outcomes, not proof of an internal object/position mechanism. Both and neither are explicit outcomes.",
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    summarize(parser.parse_args().root)
