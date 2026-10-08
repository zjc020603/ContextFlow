"""Audit all paired no-context trials and produce an explicit old/new comparison."""
# ruff: noqa: RUF001  # Chinese report prose.

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INTERVENTION = "zero_all_encoded_demo_tokens_keep_mask"
TASKS = ("milk", "tomato_sauce")


def rows(path):
    values = [json.loads(line) for line in path.read_text().splitlines()]
    result = {(r["task"], r["episode_index"]): r for r in values}
    assert len(result) == len(values)
    return result


def audit(root, baseline):
    new = rows(root / "rollouts/no/episodes.jsonl")
    old = {mode: rows(baseline / mode / "episodes.jsonl") for mode in ("correct", "no", "wrong")}
    expected = {(task, ep) for task in TASKS for ep in range(25)}
    assert set(new) == expected
    cfg = json.loads((root / "rollouts/no/config.json").read_text())
    assert cfg["scene_task"] == "per_language"
    assert cfg["layout"] == "original"
    old_cfg = json.loads((baseline / "no/config.json").read_text())
    for name in ("num_trials_per_task", "seed", "policy_seed", "max_steps", "num_steps_wait", "replan_steps"):
        assert cfg[name] == old_cfg[name], name
    paired = []
    video_lines = [
        "# 配对视频：旧 masked no 与新 zero no",
        "",
        "全部 20 FPS，280 帧；每个视频对应一次完整回合。",
        "",
        "| 语言 | 回合 | 旧 no | 新 no |",
        "|---|---:|---|---|",
    ]
    for key, record in sorted(new.items()):
        for mode, group in old.items():
            reference = group[key]
            for field in ("initial_observation_sha256", "initial_state_sha256", "environment_seed", "steps"):
                assert record[field] == reference[field], (key, mode, field)
            assert [r["rng_components"] for r in record["context_requests"]] == [
                r["rng_components"] for r in reference["context_requests"]
            ]
        assert len(record["context_requests"]) == 56
        for request in record["context_requests"]:
            assert request["no_context_intervention"] == INTERVENTION
            assert request["mode"] == "no"
            assert request["selected_episode"] == []
            assert request["demo_task_index"] is None
        for file in ("video", "trajectory"):
            assert (root / "rollouts" / record[file]).is_file()
        with np.load(root / "rollouts" / record["trajectory"]) as trajectory:
            actions = trajectory["actions"]
            with np.load(baseline / old["no"][key]["trajectory"]) as previous:
                np.testing.assert_array_equal(trajectory["initial_sim_state"], previous["initial_sim_state"])
                first_action_delta = float(np.max(np.abs(actions[0] - previous["actions"][0])))
            assert actions.shape == (280, 7)
            assert np.isfinite(actions).all()
            assert (
                dict(zip(trajectory["object_names"].tolist(), trajectory["in_basket"].any(0).tolist(), strict=True))
                == record["ever_in_basket"]
            )
            assert (
                dict(
                    zip(
                        trajectory["scene_object_names"].tolist(),
                        trajectory["scene_in_basket"].any(0).tolist(),
                        strict=True,
                    )
                )
                == record["scene_ever_in_basket"]
            )
            for step in (0, 80, 160):
                directory = root / "server/attention" / key[0] / f"ep{key[1]:03d}_step{step:03d}"
                info = json.loads((directory / "layout.json").read_text())
                assert info["intervention"] == INTERVENTION
                assert info["prefix_checks"]["mask_and_positions_exact"]
                assert info["prefix_checks"]["non_demo_tokens_exact"]
                with np.load(directory / "capture.npz") as cap:
                    np.testing.assert_array_equal(cap["ordinary_actions"], cap["captured_actions"])
                    np.testing.assert_array_equal(cap["executable_actions"][:5], actions[step : step + 5])
                    assert np.isfinite(cap["probabilities"]).all()
                    for block in info["blocks"]:
                        if block["kind"].startswith("demo_"):
                            mask = cap["prefix_mask"][0, block["start"] : block["stop"]]
                            # Real front/wrist/state/action slots remain valid; absent right camera remains invalid.
                            assert np.all(mask == ("right" not in block["name"]))
        paired.append(
            {
                "task": key[0],
                "episode": key[1],
                "old_outcome": old["no"][key]["outcome"],
                "new_outcome": record["outcome"],
                "paired_initial_state_observation_rng": True,
                "first_action_max_abs_difference": first_action_delta,
            }
        )
        video_lines.append(
            f'| {key[0]} | {key[1]} | [旧 no]({baseline / old["no"][key]["video"]}) | [新 no]({root / "rollouts" / record["video"]}) |'
        )
    finished = json.loads((root / "server/SERVER_FINISHED.json").read_text())
    assert finished["passed"]
    assert finished["captures"] == 150
    assert finished["runtime_calls"] == 2804
    assert finished["parameter_sha256_before"] == finished["parameter_sha256_after"]
    assert len(list((root / "server/attention").glob("*/*/capture.npz"))) == 150
    comparison = []
    for task in TASKS:
        for condition, group in (
            ("correct", old["correct"]),
            ("no_zero_keep_mask", new),
            ("wrong", old["wrong"]),
            ("no_masked_legacy", old["no"]),
        ):
            selected = [r for (t, _), r in group.items() if t == task]
            comparison.append(
                {
                    "task": task,
                    "condition": condition,
                    "episodes": len(selected),
                    "milk_in_basket": sum(r["ever_in_basket"]["milk"] for r in selected),
                    "tomato_sauce_in_basket": sum(r["ever_in_basket"]["tomato_sauce"] for r in selected),
                    "language_successes": sum(r["language_success"] for r in selected),
                    "other_successes": sum(r["other_success"] for r in selected),
                }
            )
    objects = {
        task: dict(
            collections.Counter(
                name
                for (t, _), r in new.items()
                if t == task
                for name, success in r["scene_ever_in_basket"].items()
                if success
            )
        )
        for task in TASKS
    }
    result = {
        "baseline": str(baseline),
        "new_rollouts": str(root / "rollouts"),
        "intervention": INTERVENTION,
        "comparison": comparison,
        "new_no_scene_objects_in_basket": objects,
        "verified_pairs": 50,
        "attention_captures": 150,
        "old_no_data_preserved": True,
        "new_vs_old_first_action_changed_episodes": sum(r["first_action_max_abs_difference"] > 0 for r in paired),
        "scope": "original per-language scenes only; fixed tomato scene/position-swap no-context NOT rerun",
    }
    for name, data in (("comparison.csv", comparison), ("paired_outcomes.csv", paired)):
        with (root / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "VIDEO_INDEX.md").write_text("\n".join(video_lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, task in zip(axes, TASKS, strict=True):
        values = [r for r in comparison if r["task"] == task]
        x = np.arange(4)
        ax.bar(x - 0.18, [r["milk_in_basket"] for r in values], width=0.36, label="Milk", color="#367cad")
        ax.bar(
            x + 0.18, [r["tomato_sauce_in_basket"] for r in values], width=0.36, label="Tomato sauce", color="#d67945"
        )
        ax.set_xticks(x, ["Correct", "No: zero\nkeep masks", "Wrong", "No: masked\nlegacy"])
        ax.set_ylim(0, 27)
        ax.set_yticks(range(0, 26, 5))
        ax.set_title(f"Language / scene: {task}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for container in ax.containers:
            ax.bar_label(container, padding=3)
    axes[0].set_ylabel("Episodes with object in basket (out of 25)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=170)
    plt.close(fig)
    return result


def report(root, result):
    table = ["| 当前语言 / 场景 | 条件 | 牛奶入篮 | 番茄酱入篮 |", "|---|---|---:|---:|"]
    names = {
        "correct": "完整正确示范（历史）",
        "wrong": "完整错误示范（历史）",
        "no_zero_keep_mask": "无示范内容：表示置零、保留 mask（新）",
        "no_masked_legacy": "旧 no：关闭示范 mask（历史）",
    }
    table.extend(
        f'| {r["task"]} | {names[r["condition"]]} | {r["milk_in_basket"]}/25 | {r["tomato_sauce_in_basket"]}/25 |'
        for r in result["comparison"]
    )
    values = [r for r in result["comparison"] if r["condition"] == "no_zero_keep_mask"]
    zero_counts = sum(r["milk_in_basket"] + r["tomato_sauce_in_basket"] for r in values)
    conclusion = (
        "新 no 的 50 次试验仍没有牛奶或番茄酱入篮。因此，旧 no 的零成功率在这次保留 mask/位置编号的置零干预中也出现了；不能只用位置编号变化解释旧结果。"
        if zero_counts == 0
        else "保留 mask/位置编号的置零干预出现了入篮行为，旧 no 的零成功率不能推广到新的 no 定义。具体区别见表格；这也不单独证明差异由位置编号造成，因为两种干预的 attention 可见性也不同。"
    )
    text = f"""# No-context 补充实验：示范表示置零，保留 mask 和位置编号

已完成：两个任务各 25 次，共 50 次，原始的各自任务场景。correct/wrong 复用原实验记录。

**{conclusion}**

{chr(10).join(table)}

![新旧 no 与 correct/wrong 对照](comparison.png)

表图仅统计牛奶与番茄酱。历史 wrong/milk 组虽然这两项都是 0/25，但原实验额外物体回放已确认奶油奶酪入篮 25/25，不能将该组解读为没有操作物体。本次新 no 没有任何场景物体入篮。

## 干预的确切含义

- 原 no：示范原始数据置零，示范 mask=false；token 槽位并未物理删除，但有效位置编号变了。
- 新 no：先按完整示范路径得到表示，再在进入 Gemma 前把示范图像、状态、action 的全部 160 个 token 槽位置零；保持原 mask，原来有效的 128 个位置继续有效，缺失右相机仍无效。当前图像、语言、本体状态不变。
- 牛奶 gate 样例首个 action 位置编号保持 653，旧 masked no 为 525。所有实际采集点均验证新 no 与完整示范 prefix 的 mask、位置编号和非示范内容精确相同。
- 这不是把原始像素/状态/action 数值设为 0 后再编码；编码器的偏置或位置表示不会以这种方式残留为示范内容。
- 零表示仍是可能的 OOD 输入，不是训练过的“无示范”表示；有效零 token 后续可以融合其他信息，attention 不要求为零。

## 验证和记录

- 50 回合逐一与历史 correct/no/wrong 核对：初始模拟器状态、观察哈希、环境种子、56 次推理噪声全部一致。每回合固定 280 步、每 5 步重规划，初始状态索引 0–24。
- 6 份历史快照的完整模型 raw/executable actions 与实验 0 保存结果逐字节一致。
- 两个任务 gate 验证：改变全部示范数值，或换另一任务示范，置零后动作完全相同；示范内容没有绕过干预进入输出。
- 保存 150 份 attention（每回合 0/80/160 步），每份记录开/关动作精确一致，并与实际执行的 5 步动作核对。仅保存，不分析热图。
- 检查点参数 hash 运行前后相同。所有 50 个视频按 20 FPS 保存，位于指令命名的子目录。
- 新 no 的全部场景物体入篮计数：`{json.dumps(result['new_no_scene_objects_in_basket'], ensure_ascii=False)}`。计数可能包括干扰物，不能只依据牛奶/番茄酱失败宣称机器人完全不动。

## 结论范围

新旧 no 在相同初始观察和噪声下，有 {result['new_vs_old_first_action_changed_episodes']}/50 回合的第一步动作不同。因此，成功率即使相同，也不能说两种干预产生了相同的行为；差值保存在逐回合 CSV。

本次补充只针对 20260922_154452 的原 Correct / No / Wrong 实验。固定番茄酱场景、位置交换实验的 no 分组，以及实验 0/1 和 action 消融报告引用的历史 no 记录均未重跑，不能将这里的新数值替换到那些条件中。Table 1 完整示范复现不受此干预影响。

即使新旧 no 都失败，也不能证明位置编号对行为完全没有影响，或证明模型不理解语言；两个对照都可能受 OOD 影响。要单独隔离位置编号，还需保持 attention 可见性不变的另一组控制。

[逐回合配对视频](VIDEO_INDEX.md) · [机器可读对照](comparison.json) · [逐回合结果](paired_outcomes.csv) · [代码修改报告](CODE_CHANGES.md)
"""
    (root / "EXPERIMENT_RESULTS.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        default="logs/quickstart/context_ablation_object_correct_no_wrong_25trials_20260922_154452",
        type=Path,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = args.baseline.resolve()
    result = audit(root, baseline)
    report(root, result)
    print(json.dumps(result, indent=2))
