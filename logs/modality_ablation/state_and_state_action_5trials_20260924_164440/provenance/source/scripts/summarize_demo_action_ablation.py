"""Audit matched behavior trials and report a position-preserving action-token pilot."""
# ruff: noqa: RUF001

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.libero.summarize_position_ablation import demo_targets

LAYOUTS = ("original", "swap_milk_tomato")
TASKS = ("milk", "tomato_sauce")


def read_rows(root, layout, mode):
    path = root / layout / mode / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keyed = {(r["task"], r["episode_index"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate episodes: {path}")
    return keyed


def summary_rows(rows, layout, task, mode, variant):
    identity, occupant = demo_targets(task, mode, layout)
    return {
        "layout": layout,
        "language": task,
        "context": mode,
        "variant": variant,
        "trials": len(rows),
        "milk_in_basket": sum(r["ever_in_basket"]["milk"] for r in rows),
        "tomato_sauce_in_basket": sum(r["ever_in_basket"]["tomato_sauce"] for r in rows),
        "language_successes": sum(r["language_success"] for r in rows),
        "other_successes": sum(r["other_success"] for r in rows),
        "demo_identity": identity,
        "demo_position_occupant": occupant,
        "demo_position_occupant_successes": None
        if occupant is None
        else sum(r["ever_in_basket"][occupant] for r in rows),
        "language_only": sum(r["outcome"] == "language_only" for r in rows),
        "other_only": sum(r["outcome"] == "other_only" for r in rows),
        "both": sum(r["outcome"] == "both" for r in rows),
        "neither": sum(r["outcome"] == "neither" for r in rows),
        "distractor_ever_in_basket": sum(
            any(v for k, v in r["scene_ever_in_basket"].items() if k not in ("milk_1", "tomato_sauce_1")) for r in rows
        ),
    }


def run(args):
    root, baseline, server = Path(args.root).resolve(), Path(args.baseline).resolve(), Path(args.server).resolve()
    gate = json.loads((server / "GATE_PASSED.json").read_text())
    finished = json.loads((server / "SERVER_FINISHED.json").read_text())
    if not gate["passed"] or not finished["passed"]:
        raise AssertionError("Server validation failed")
    n = 5
    counts = []
    pairs = []
    oldrefs = []
    attention = []
    for layout in LAYOUTS:
        for mode in ("correct", "wrong", "no"):
            before = read_rows(baseline, layout, mode)
            if mode != "no":
                after = read_rows(root / "rollouts", layout, mode)
                if len(after) != 2 * n:
                    raise AssertionError("Missing/new duplicate trials")
                oldcfg = json.loads((baseline / layout / mode / "config.json").read_text())
                newcfg = json.loads((root / "rollouts" / layout / mode / "config.json").read_text())
                for key in (
                    "seed",
                    "policy_seed",
                    "max_steps",
                    "num_steps_wait",
                    "replan_steps",
                    "scene_task",
                    "layout",
                ):
                    if oldcfg[key] != newcfg[key]:
                        raise AssertionError("Protocol differs: " + key)
            for task in TASKS:
                originals = [before[task, i] for i in range(n)]
                counts.append(
                    summary_rows(originals, layout, task, mode, "no_reference" if mode == "no" else "full_demo")
                )
                oldrefs.extend(
                    {
                        "layout": layout,
                        "mode": mode,
                        "task": task,
                        "episode": row["episode_index"],
                        "video": str(baseline / layout / row["video"]),
                        "trajectory": str(baseline / layout / row["trajectory"]),
                        "initial_state_sha256": row["initial_state_sha256"],
                    }
                    for row in originals
                )
                if mode == "no":
                    continue
                changed = [after[task, i] for i in range(n)]
                counts.append(summary_rows(changed, layout, task, mode, "zero_demo_action"))
                for a, b in zip(originals, changed, strict=True):
                    for key in (
                        "initial_state_sha256",
                        "initial_observation_sha256",
                        "environment_seed",
                        "steps",
                        "scene_task",
                        "scene_id",
                        "layout",
                        "index",
                        "description",
                    ):
                        if a[key] != b[key]:
                            raise AssertionError(f'Unpaired {key}: {layout}/{mode}/{task}/{a["episode_index"]}')
                    if (
                        a["position_intervention"]["source_state_sha256"]
                        != b["position_intervention"]["source_state_sha256"]
                    ):
                        raise AssertionError("Layout source mismatch")
                    for key in ("selected_episode", "rng_components", "demo_task_index"):
                        if [r[key] for r in a["context_requests"]] != [r[key] for r in b["context_requests"]]:
                            raise AssertionError("Demo or noise protocol mismatch")
                    assert len(b["context_requests"]) == 56
                    assert all(
                        r["demo_action_intervention"] == "zero_compressed_demo_action_keep_mask"
                        for r in b["context_requests"]
                    )
                    episode = b["episode_index"]
                    path = root / "rollouts" / layout / b["trajectory"]
                    with np.load(path) as trace:
                        assert trace["actions"].shape == (280, 7)
                        assert np.isfinite(trace["actions"]).all()
                        for obj_idx, obj in enumerate(trace["object_names"]):
                            assert bool(trace["in_basket"][:, obj_idx].any()) == b["ever_in_basket"][str(obj)]
                        assert (
                            hashlib.sha256(trace["initial_sim_state"].tobytes()).hexdigest()
                            == b["initial_state_sha256"]
                        )
                    for step in (0, 80, 160):
                        directory = server / "attention" / layout / mode / task / f"ep{episode:03d}_step{step:03d}"
                        info = json.loads((directory / "layout.json").read_text())
                        checks = info["prefix_intervention_checks"]
                        assert checks["zero_action_token_l2"] == 0
                        assert checks["mask_and_positions_exact"]
                        assert checks["non_action_tokens_exact"]
                        with np.load(directory / "capture.npz") as data:
                            assert data["ordinary_actions"].tobytes() == data["captured_actions"].tobytes()
                            np.testing.assert_array_equal(data["executable_actions"][:5], trace_actions(path, step))
                            assert data["probabilities"].shape == (3, 3, 1, 8, 51, 1027)
                            assert np.isfinite(data["probabilities"]).all()
                            prefix = np.asarray(data["prefix_mask"])
                            assert np.all(data["probabilities"][..., np.flatnonzero(~prefix[0])] == 0)
                        attention.append(
                            {
                                "layout": layout,
                                "mode": mode,
                                "task": task,
                                "episode": episode,
                                "environment_step": step,
                                "path": str(directory.relative_to(root)),
                                "sha256": hashlib.sha256((directory / "capture.npz").read_bytes()).hexdigest(),
                            }
                        )
                    pairs.append(
                        {
                            "layout": layout,
                            "context": mode,
                            "language": task,
                            "episode": episode,
                            "before": a["outcome"],
                            "after": b["outcome"],
                            "baseline_video": str(baseline / layout / a["video"]),
                            "zero_video": str((root / "rollouts" / layout / b["video"]).relative_to(root)),
                            "baseline_language_success": a["language_success"],
                            "zero_language_success": b["language_success"],
                            "rollout_seconds": b["elapsed_seconds"],
                        }
                    )
    assert len(pairs) == 40
    assert len(attention) == 120
    assert finished["runtime_inferences"] == 2240
    assert finished["saved_attention_records"] == 120
    result = {
        "pilot": True,
        "trials_per_cell": 5,
        "new_rollouts": 40,
        "new_rollout_elapsed_seconds": sum(p["rollout_seconds"] for p in pairs),
        "matched_historical_full_demo_rollouts": 40,
        "reused_historical_no_context_rollouts": 20,
        "attention_records": 120,
        "comparison": counts,
        "paired_transitions": dict(Counter(p["before"] + " -> " + p["after"] for p in pairs)),
        "all_initial_states_observations_demo_seeds_paired": True,
        "baseline": str(baseline),
        "server": str(server),
        "no_context_was_not_changed_or_rerun": True,
        "ood_caveat": True,
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in [("comparison.csv", counts), ("paired_outcomes.csv", pairs)]:
        with (root / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "baseline_references.json").write_text(json.dumps(oldrefs, indent=2) + "\n")
    (root / "attention_manifest.json").write_text(json.dumps(attention, indent=2) + "\n")
    video_lines = [
        "# 配对视频索引",
        "",
        "每行对比相同初始状态、语言、context 和布局。视频均为 20 FPS；状态是整条轨迹的入篮结果。",
        "",
        "| 布局 / 语言 / context / 初始状态 | 完整示范 | action 表示置零 |",
        "|---|---|---|",
    ]
    for pair in pairs:
        label = f"{pair['layout']} / {pair['language']} / {pair['context']} / {pair['episode']}"
        video_lines.append(
            f"| {label} | [{pair['before']}]({pair['baseline_video']}) | [{pair['after']}]({pair['zero_video']}) |"
        )
    (root / "VIDEO_INDEX.md").write_text("\n".join(video_lines) + "\n")
    plot(root, counts)
    report(root, result, pairs)
    print(json.dumps(result, indent=2))


def trace_actions(path, step):
    with np.load(path) as data:
        return data["actions"][step : step + 5]


def plot(root, counts):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    choices = [
        ("correct", "full_demo"),
        ("correct", "zero_demo_action"),
        ("wrong", "full_demo"),
        ("wrong", "zero_demo_action"),
        ("no", "no_reference"),
    ]
    labels = ["Correct\nfull", "Correct\nzero action", "Wrong\nfull", "Wrong\nzero action", "No context\nreference"]
    for row, layout in enumerate(LAYOUTS):
        for col, task in enumerate(TASKS):
            ax = axs[row, col]
            selected = [
                next(
                    c
                    for c in counts
                    if c["layout"] == layout and c["language"] == task and (c["context"], c["variant"]) == choice
                )
                for choice in choices
            ]
            x = np.arange(5)
            for offset, key, label, color in [
                (-0.18, "milk_in_basket", "Milk placed", "#287e9b"),
                (0.18, "tomato_sauce_in_basket", "Tomato sauce placed", "#c3683d"),
            ]:
                bars = ax.bar(x + offset, [c[key] for c in selected], 0.34, label=label, color=color)
                ax.bar_label(bars, fmt="%d")
            ax.set_xticks(x, labels)
            ax.set_ylim(0, 6)
            ax.set_yticks(range(6))
            ax.set_ylabel("Episodes out of 5")
            ax.set_title(layout + " | instruction: " + task)
            ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(loc="upper right")
    fig.suptitle(
        "Experiment 2 pilot: zero compressed demo-action tokens; preserve masks and position IDs\n"
        "Matched initial states 0..4; 280 control steps; full/no bars reuse historical trials. Both objects may count.",
        fontsize=12,
    )
    fig.savefig(root / "behavior_comparison.png", dpi=160)
    plt.close(fig)


def report(root, result, pairs):
    table = []
    for layout in LAYOUTS:
        for task in TASKS:
            for mode in ("correct", "wrong"):
                rows = [
                    next(
                        r
                        for r in result["comparison"]
                        if r["layout"] == layout
                        and r["language"] == task
                        and r["context"] == mode
                        and r["variant"] == v
                    )
                    for v in ("full_demo", "zero_demo_action")
                ]
                a, b = rows
                table.append(
                    f"| {layout} | {task} | {mode} | {a['language_successes']}/5 → {b['language_successes']}/5 | {a['other_successes']}/5 → {b['other_successes']}/5 | {a['neither']}/5 → {b['neither']}/5 |"
                )
    transitions = "\n".join(f"- `{k}`：{v} 对" for k, v in result["paired_transitions"].items())
    full = [r for r in result["comparison"] if r["variant"] == "full_demo"]
    zero = [r for r in result["comparison"] if r["variant"] == "zero_demo_action"]
    position_counts = [sum(r["demo_position_occupant_successes"] for r in group) for group in (full, zero)]
    original_wrong = [
        next(
            r
            for r in result["comparison"]
            if r["layout"] == "original"
            and r["context"] == "wrong"
            and r["language"] == task
            and r["variant"] == "zero_demo_action"
        )
        for task in TASKS
    ]
    wrong_other = sum(r["other_successes"] for r in original_wrong)
    wrong_language = sum(r["language_successes"] for r in original_wrong)
    wrong_neither = sum(r["neither"] for r in original_wrong)
    corrected = sum(p["before"] == "other_only" and p["after"] == "language_only" for p in pairs)
    failure_after_wrong = sum(p["before"] == "other_only" and p["after"] == "neither" for p in pairs)
    interpretation = (
        "本次仍保留了错误物体操作：显式 action 模态不是这一行为在本干预下出现的唯一来源，保留的状态/图像等信息仍可能支持它。不能据此断言原模型不使用 action。"
        if wrong_other > 0
        else "本次未观察到原始布局 wrong-context 下的另一物体入篮；需要结合语言目标成功与失败次数判断是纠正还是性能损失。"
    )
    headline = (
        f"**主要观察：显式 action 表示置零后，示范原位置上的物体仍在 {position_counts[1]}/40 次试验中被放入篮子；完整示范为 {position_counts[0]}/40。**\n\n"
        f"原始布局的 wrong-context 中，置零后另一物体入篮 {wrong_other}/10 次，语言目标入篮 {wrong_language}/10 次，两者均未入篮 {wrong_neither}/10 次。"
        f"{interpretation}\n\n"
        "交换位置后 wrong-context 的语言目标成功若增加，也可能仍是原位置占据者被更顺利地放入篮子，不应直接解释成更遵循语言。"
        f"本次 40 对中，`other_only → language_only` 为 {corrected} 对，`other_only → neither` 为 {failure_after_wrong} 对；必须区分纠正与失败。\n\n"
        f"新增轨迹累计运行约 {result['new_rollout_elapsed_seconds']/60:.1f} 分钟（不含模型加载、运行前校验和汇总）；attention 已保存但尚未判读。\n\n"
    )
    body = (
        """# 实验 2：示范 action 表示置零（低成本试验）

__HEADLINE__本报告使用每条件 5 次试验，新增 40 条完整 rollout。结果用于决定下一步实验，不能替代每条件 25 次或跨更多初始状态的确认实验。具体结果见下表；“完整示范”和 no-context 均复用历史运行中同一初始状态索引 0–4，不把历史 25 次比例与本次 5 次直接比较。

## 做了什么

- 场景固定为番茄酱场景；2 条语言 × 2 种布局 × correct/wrong = 8 个新增条件，各 5 次。
- 将示范 action 经投影和 Perceiver 压缩后的 32 个 token **在进入 Gemma 前置零**。保留 token 槽位、有效 mask、位置编号；示范图像和状态保持原样。
- 不是给机器人发送零动作，也不是把原始示范改成一段静止动作；不是训练过的 null embedding。此干预可能 OOD。
- 固定 280 个控制步，每 5 步重规划；任何物体入篮都不会提前结束。视频为原速 20 FPS。
- 基线是此前位置交换实验，不是实验 0 的一致性检查。本次也没有修改旧 no-context，未重跑其闭环轨迹。

## 结果

![物体入篮次数](behavior_comparison.png)

下表为 **完整示范 → action 表示置零**。“另一物体”指牛奶与番茄酱中未被当前语言指定的那个；“两者都未入篮”不等同于完全不动，也不排除操作其他干扰物。

| 布局 | 语言目标 | context | 语言目标入篮 | 另一物体入篮 | 两者都未入篮 |
|---|---|---|---:|---:|---:|
"""
        + "\n".join(table)
        + """

所有 40 对试验的 outcome 转换：

"""
        + transitions
        + """

`language_only`/`other_only`/`both`/`neither` 使用整条轨迹中是否曾满足 LIBERO 入篮谓词判定。完整计数含干扰物、示范物体、示范原位置占据者，见 [comparison.csv](comparison.csv)。逐次配对结果见 [paired_outcomes.csv](paired_outcomes.csv)，可点击的配对视频入口见 [VIDEO_INDEX.md](VIDEO_INDEX.md)。

## 为什么这还不能直接证明 action 的因果重要性

置零可能产生训练中未出现的表示；性能下降可能来自分布偏移。若 wrong 下错误行为减少但 correct 同样崩溃，不能称为纠正了示范误导。若 wrong 下语言目标成功增加而 correct 保持，才是更有针对性的线索，仍需要更大样本或其他干预确认。

此外，示范图像与状态还包含运动信息。本实验只移除显式 action 模态通过这些压缩 token 输入 Gemma 的路径，不是移除所有动作线索。零 token 保留有效槽位，后续层还能向这些位置写入来自其他模态的信息，因而其 attention 不要求为零。

## 验证与配对

- 启动前，当前完整示范模型在 60 份历史初始观测上的原始完整动作及输出变换后动作，与实验 0 保存结果逐字节一致。
- 12 份实际检查点样本上，置零前后的非 action prefix、mask、位置编号精确一致；32 个 action token 为零；记录 attention 不改变消融后动作。
- 8 份 correct/wrong 样本中，再改变原始示范 action 的数值，消融输出仍完全相同；4 份 no-context 样本的输出不受此置零影响。
- 40 条新增轨迹与历史对应项的初始模拟器状态、初始图像/状态 hash、环境 seed、示范 episode、各次推理随机 seed 全部匹配。
- 全部动作有限；保存轨迹的入篮谓词与汇总一致；实验前后模型参数 SHA256 一致。

## 保留的 attention

保存环境步 0、80、160，共 **120 份**原始 attention 及对应观测；每份含采样步 0/5/9、层 0/9/17、全部 8 个 heads、51 个 queries、1027 个 keys。每次采集同时验证记录关闭/开启的动作精确相同。已核对这些动作的前 5 步确实是后续执行的动作。

本次不判读热图，也没有预先平均数据。[attention_manifest.json](attention_manifest.json) 给出全部位置与 checksum。中后期画面会随行为分化，不能当成相同观测对照。

## 关于之前 no-context 的位置编码问题

问题是在实验 1 解读已有 no-context 记录时发现的：原模型用有效 prefix mask 的累积计数产生位置编号。全部示范 mask 关闭后，在牛奶示例中有效 prefix 从 652 降到 524，首个 action 位置编号从 653 变为 525。它影响“无示范为何改变注意力”的解释；没有使已经执行的行为记录或成功率成为无效数据，但 no-context 不再是只移除语义、其余结构全保持的对照。

这个定义已用于之前的 correct/no/wrong 行为实验及位置交换实验的 no 分组，也进入实验 0/1 的 no 记录。correct/wrong 分组及原 Table 1 完整示范复现不涉及这一变化。实验 0 的记录开/关一致性结论也不受影响，因为两条路径使用同样的 mask。

此前只增加了说明，**没有修复旧实现或重跑旧结果**。本次 action-only 置零保留 mask，因此不引入这项位置变化。mask=false 与表示置零并不等价。

相关文件：[代码修改报告](CODE_CHANGES.md)、[机器可读结果](comparison.json)、[复用基线索引](baseline_references.json)。
"""
    )

    report_path = root / "EXPERIMENT_RESULTS.md"
    report_path.write_text(body.replace("__HEADLINE__", headline))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--baseline", default="logs/quickstart/position_swap_object_25trials_20260922_175506")
    run(parser.parse_args())
