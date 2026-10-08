"""Validate completed paired context trials and write auditable comparison tables."""

import argparse
import collections
import csv
import json
import pathlib


def summarize(root):
    root = pathlib.Path(root)
    modes = ("correct", "no", "wrong")
    tasks = ("milk", "tomato_sauce")
    all_rows = {}
    configs = {}
    comparison = []
    scene_comparison = {}
    for mode in modes:
        configs[mode] = json.loads((root / mode / "config.json").read_text())
        rows = [json.loads(line) for line in (root / mode / "episodes.jsonl").read_text().splitlines()]
        all_rows[mode] = {(row["task"], row["episode_index"]): row for row in rows}
        if len(all_rows[mode]) != len(rows):
            raise ValueError(f"Duplicate episode in {mode}")
        n = configs[mode]["num_trials_per_task"]
        expected = {(task, episode) for task in tasks for episode in range(n)}
        if set(all_rows[mode]) != expected:
            raise ValueError(f"Incomplete trials in {mode}")
        for task in tasks:
            selected = [row for row in rows if row["task"] == task]
            counts = collections.Counter(row["outcome"] for row in selected)
            if all("scene_ever_in_basket" in row for row in selected):
                objects = collections.Counter(
                    name for row in selected for name, success in row["scene_ever_in_basket"].items() if success
                )
                scene_comparison[f"{mode}/{task}"] = {"episodes": n, "object_in_basket_counts": dict(objects)}
            comparison.append(
                {
                    "language": task,
                    "context_mode": mode,
                    "demo": "none"
                    if mode == "no"
                    else (task if mode == "correct" else next(t for t in tasks if t != task)),
                    "episodes": n,
                    "language_successes": sum(row["language_success"] for row in selected),
                    "other_successes": sum(row["other_success"] for row in selected),
                    **{label: counts[label] for label in ("language_only", "other_only", "both", "neither")},
                }
            )
        for row in rows:
            if row["mode"] != mode:
                raise ValueError("Mode mismatch")
            for key in ("video", "trajectory"):
                if not (root / row[key]).is_file():
                    raise FileNotFoundError(row[key])
    paired = {}
    for key, reference in all_rows["correct"].items():
        for mode in modes[1:]:
            other = all_rows[mode][key]
            for field in ("initial_observation_sha256", "initial_state_sha256", "environment_seed", "steps"):
                if reference[field] != other[field]:
                    raise ValueError(f"Unpaired {field}: {mode} {key}")
            if [r["rng_components"] for r in reference["context_requests"]] != [
                r["rng_components"] for r in other["context_requests"]
            ]:
                raise ValueError(f"Unpaired action noise: {mode} {key}")
    for task in tasks:
        pairs = [(row, all_rows["wrong"][key]) for key, row in all_rows["correct"].items() if key[0] == task]
        paired[task] = {
            "pairs": len(pairs),
            "correct_language_only_to_wrong_other_only": sum(
                a["outcome"] == "language_only" and b["outcome"] == "other_only" for a, b in pairs
            ),
            "correct_other_only_to_wrong_language_only": sum(
                a["outcome"] == "other_only" and b["outcome"] == "language_only" for a, b in pairs
            ),
            "transition_counts": dict(collections.Counter(a["outcome"] + " -> " + b["outcome"] for a, b in pairs)),
        }
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    result = {
        "comparison": comparison,
        "paired_correct_vs_wrong": paired,
        "verified": "Matching initial simulator states, observations and per-replan noise seeds in all conditions",
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if len(scene_comparison) == 6:
        (root / "all_object_comparison.json").write_text(json.dumps(scene_comparison, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    summarize(parser.parse_args().root)
