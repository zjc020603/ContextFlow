"""Audit four paired demonstration-modality conditions and report state ablations."""
# ruff: noqa: RUF001

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.libero.summarize_position_ablation import demo_targets
from scripts.summarize_demo_action_ablation import read_rows
from scripts.summarize_demo_action_ablation import summary_rows

LAYOUTS = ("original", "swap_milk_tomato")
TASKS = ("milk", "tomato_sauce")
VARIANTS = ("full_demo", "zero_action", "zero_state", "zero_state_action")
LABELS = {
    "full_demo": "完整示范",
    "zero_action": "仅 action 置零",
    "zero_state": "仅 state 置零",
    "zero_state_action": "state/action 同时置零",
}


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired(a, b):
    for field in (
        "initial_state_sha256",
        "initial_observation_sha256",
        "environment_seed",
        "steps",
        "scene_task",
        "scene_id",
        "layout",
        "index",
        "description",
        "episode_index",
    ):
        assert a[field] == b[field], field
    assert a["position_intervention"]["source_state_sha256"] == b["position_intervention"]["source_state_sha256"]
    for field in ("selected_episode", "rng_components", "demo_task_index"):
        assert [r[field] for r in a["context_requests"]] == [r[field] for r in b["context_requests"]], field


def audit(root, baseline, action):
    paths = {
        "full_demo": baseline,
        "zero_action": action / "rollouts",
        "zero_state": root / "zero_state/rollouts",
        "zero_state_action": root / "zero_state_action/rollouts",
    }
    old_finished = json.loads((action / "server_attempt1/SERVER_FINISHED.json").read_text())
    assert old_finished["passed"]
    distractor_events = []
    captures, counts, pair_rows, video_lines, all_rows = (
        [],
        [],
        [],
        [
            "# 四条件配对视频",
            "",
            "相同布局、语言、context、初始状态；每条视频 280 帧、20 FPS。",
            "",
            "| 布局 / 语言 / context / 回合 | 完整示范 | 去 action | 去 state | 去 state/action |",
            "|---|---|---|---|---|",
        ],
        {},
    )
    for variant in ("zero_state", "zero_state_action"):
        server = root / variant / "server"
        gate = json.loads((server / "GATE_PASSED.json").read_text())
        finished = json.loads((server / "SERVER_FINISHED.json").read_text())
        assert gate["passed"]
        assert finished["passed"]
        assert finished["runtime_inferences"] == 2240
        assert finished["saved_attention_records"] == 120
        assert (
            finished["parameter_sha256_before"]
            == finished["parameter_sha256_after"]
            == old_finished["parameter_sha256_before"]
        )
        assert len(list((server / "attention").glob("*/*/*/*/capture.npz"))) == 120
    for layout in LAYOUTS:
        for mode in ("correct", "wrong"):
            groups = {v: read_rows(path, layout, mode) for v, path in paths.items()}
            for variant in ("zero_action", "zero_state", "zero_state_action"):
                assert set(groups[variant]) == {(t, e) for t in TASKS for e in range(5)}
                reference_cfg = json.loads((baseline / layout / mode / "config.json").read_text())
                cfg = json.loads((paths[variant] / layout / mode / "config.json").read_text())
                for field in (
                    "seed",
                    "policy_seed",
                    "max_steps",
                    "num_steps_wait",
                    "replan_steps",
                    "scene_task",
                    "layout",
                ):
                    assert cfg[field] == reference_cfg[field], field
            for task in TASKS:
                for variant in VARIANTS:
                    selected = [groups[variant][task, e] for e in range(5)]
                    counts.append(summary_rows(selected, layout, task, mode, variant))
                for episode in range(5):
                    trials = {v: groups[v][task, episode] for v in VARIANTS}
                    all_rows[layout, mode, task, episode] = trials
                    entry = {"layout": layout, "context": mode, "language": task, "episode": episode}
                    video_links = []
                    for variant, record in trials.items():
                        if variant != "full_demo":
                            paired(trials["full_demo"], record)
                        entry[variant + "_outcome"] = record["outcome"]
                        video = paths[variant] / layout / record["video"]
                        assert video.exists()
                        video_links.append(f'[{record["outcome"]}]({video})')
                        if variant not in ("zero_state", "zero_state_action"):
                            continue
                        distractors = [
                            obj
                            for obj, placed in record["scene_ever_in_basket"].items()
                            if placed and obj not in ("milk_1", "tomato_sauce_1")
                        ]
                        if distractors:
                            distractor_events.append(
                                {
                                    "variant": variant,
                                    "layout": layout,
                                    "language": task,
                                    "context": mode,
                                    "episode": episode,
                                    "objects": distractors,
                                    "outcome": record["outcome"],
                                    "video": str(video),
                                }
                            )
                        expected = "zero_compressed_demo_" + variant.removeprefix("zero_") + "_keep_mask"
                        assert len(record["context_requests"]) == 56
                        assert all(r["demo_modality_intervention"] == expected for r in record["context_requests"])
                        with np.load(paths[variant] / layout / record["trajectory"]) as trace:
                            actions = trace["actions"]
                            assert actions.shape == (280, 7)
                            assert np.isfinite(actions).all()
                            assert (
                                hashlib.sha256(trace["initial_sim_state"].tobytes()).hexdigest()
                                == record["initial_state_sha256"]
                            )
                            for names, predicates, field in (
                                ("object_names", "in_basket", "ever_in_basket"),
                                ("scene_object_names", "scene_in_basket", "scene_ever_in_basket"),
                            ):
                                assert (
                                    dict(zip(trace[names].tolist(), trace[predicates].any(0).tolist(), strict=True))
                                    == record[field]
                                )
                            for step in (0, 80, 160):
                                directory = (
                                    root
                                    / variant
                                    / "server/attention"
                                    / layout
                                    / mode
                                    / task
                                    / f"ep{episode:03d}_step{step:03d}"
                                )
                                info = json.loads((directory / "layout.json").read_text())
                                assert info["modality_intervention"] == expected
                                checks = info["prefix_intervention_checks"]
                                assert checks["mask_and_positions_exact"]
                                assert checks["non_ablated_tokens_exact"]
                                kinds = ["demo_states"] if variant == "zero_state" else ["demo_states", "demo_actions"]
                                assert [b["kind"] for b in checks["zeroed_blocks"]] == kinds
                                for block in checks["zeroed_blocks"]:
                                    assert block["stop"] - block["start"] == 32
                                with np.load(directory / "capture.npz") as capture:
                                    assert (
                                        capture["ordinary_actions"].tobytes() == capture["captured_actions"].tobytes()
                                    )
                                    np.testing.assert_array_equal(
                                        capture["executable_actions"][:5], actions[step : step + 5]
                                    )
                                    probs = capture["probabilities"]
                                    assert probs.shape == (3, 3, 1, 8, 51, 1027)
                                    assert np.isfinite(probs).all()
                                    assert np.all(probs[..., np.flatnonzero(~capture["prefix_mask"][0])] == 0)
                                captures.append(
                                    {
                                        "variant": variant,
                                        "layout": layout,
                                        "context": mode,
                                        "task": task,
                                        "episode": episode,
                                        "step": step,
                                        "path": str(directory.relative_to(root)),
                                        "sha256": hashlib.sha256((directory / "capture.npz").read_bytes()).hexdigest(),
                                    }
                                )
                    pair_rows.append(entry)
                    video_lines.append(f"| {layout}/{task}/{mode}/{episode} | " + " | ".join(video_links) + " |")
    assert len(pair_rows) == 40
    assert len(captures) == 240
    totals = {}
    for variant in VARIANTS:
        selected = [r for r in counts if r["variant"] == variant]
        totals[variant] = {
            key: sum(r[key] for r in selected)
            for key in (
                "trials",
                "language_successes",
                "other_successes",
                "demo_position_occupant_successes",
                "neither",
                "both",
                "distractor_ever_in_basket",
            )
        }
        totals[variant]["original_correct_language_successes"] = sum(
            r["language_successes"] for r in selected if r["layout"] == "original" and r["context"] == "correct"
        )
        totals[variant]["original_wrong_other_successes"] = sum(
            r["other_successes"] for r in selected if r["layout"] == "original" and r["context"] == "wrong"
        )
    contrasts = {}
    for label, before, after in (
        ("remove_state_with_action", "full_demo", "zero_state"),
        ("remove_state_without_action", "zero_action", "zero_state_action"),
        ("remove_action_with_state", "full_demo", "zero_action"),
        ("remove_action_without_state", "zero_state", "zero_state_action"),
    ):
        items = []
        for (layout, mode, task, _), trials in all_rows.items():
            occupant = demo_targets(task, mode, layout)[1]
            a, b = trials[before], trials[after]
            items.append(
                {
                    "before_position": a["ever_in_basket"][occupant],
                    "after_position": b["ever_in_basket"][occupant],
                    "before_language": a["language_success"],
                    "after_language": b["language_success"],
                    "outcome": a["outcome"] + " -> " + b["outcome"],
                }
            )
        contrasts[label] = {
            "before": before,
            "after": after,
            "pairs": 40,
            "outcome_transitions": dict(Counter(r["outcome"] for r in items)),
        }
        for metric in ("position", "language"):
            contrasts[label][metric + "_lost"] = sum(r["before_" + metric] and not r["after_" + metric] for r in items)
            contrasts[label][metric + "_gained"] = sum(
                not r["before_" + metric] and r["after_" + metric] for r in items
            )
    context_switch = {}
    for variant in VARIANTS:
        transitions = []
        same = different = follows_both = 0
        for layout in LAYOUTS:
            for task in TASKS:
                for episode in range(5):
                    a = all_rows[layout, "correct", task, episode][variant]
                    b = all_rows[layout, "wrong", task, episode][variant]
                    a_set = {obj for obj, placed in a["ever_in_basket"].items() if placed}
                    b_set = {obj for obj, placed in b["ever_in_basket"].items() if placed}
                    a_label = "+".join(sorted(a_set)) or "neither"
                    b_label = "+".join(sorted(b_set)) or "neither"
                    transitions.append(a_label + " -> " + b_label)
                    same += len(a_set) == 1 and a_set == b_set
                    different += len(a_set) == len(b_set) == 1 and a_set != b_set
                    follows_both += a_set == {demo_targets(task, "correct", layout)[1]} and b_set == {
                        demo_targets(task, "wrong", layout)[1]
                    }
        context_switch[variant] = {
            "pairs": 20,
            "same_single_object_in_both": same,
            "different_single_objects": different,
            "both_follow_demo_region": follows_both,
            "other_pairs": 20 - same - different,
            "transitions": dict(Counter(transitions)),
        }
    result = {
        "context_switch_pairs": context_switch,
        "distractor_events": distractor_events,
        "trials_per_cell": 5,
        "new_rollouts": 80,
        "captures": 240,
        "paired_initial_state_observation_demo_rng": True,
        "baseline": str(baseline),
        "action_reference": str(action),
        "totals": totals,
        "comparison": counts,
        "contrasts": contrasts,
        "new_elapsed_trial_seconds": {
            v: sum(t[v]["elapsed_seconds"] for t in all_rows.values()) for v in ("zero_state", "zero_state_action")
        },
        "no_context_excluded": "The 25-trial no-zero supplement used different per-language scenes; historical fixed-scene no masked positions. Neither is pooled here.",
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(root / "comparison.csv", counts)
    write_csv(root / "paired_outcomes.csv", pair_rows)
    (root / "attention_manifest.json").write_text(json.dumps(captures, indent=2) + "\n")
    (root / "VIDEO_INDEX.md").write_text("\n".join(video_lines) + "\n")
    return result


def plot(root, result):
    labels = ["Full", "Zero A", "Zero S", "Zero S+A"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for row, layout in enumerate(LAYOUTS):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            selected = [
                next(
                    r
                    for r in result["comparison"]
                    if r["layout"] == layout and r["language"] == task and r["context"] == mode and r["variant"] == v
                )
                for mode in ("correct", "wrong")
                for v in VARIANTS
            ]
            x = np.arange(8)
            for offset, key, label, color in (
                (-0.18, "milk_in_basket", "Milk", "#287e9b"),
                (0.18, "tomato_sauce_in_basket", "Tomato sauce", "#c3683d"),
            ):
                bars = ax.bar(x + offset, [r[key] for r in selected], 0.34, label=label, color=color)
                ax.bar_label(bars, fmt="%d", fontsize=9)
            ax.set_xticks(x, labels * 2, fontsize=9)
            ax.set_ylim(0, 6)
            ax.set_yticks(range(6))
            ax.axvline(3.5, color="gray", linewidth=0.7)
            ax.text(1.5, 5.65, "Correct context", ha="center")
            ax.text(5.5, 5.65, "Wrong context", ha="center")
            ax.set_title(layout + " | instruction: " + task)
            ax.set_ylabel("Episodes out of 5")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.28), ncol=2)
    fig.suptitle(
        "Paired demonstration modality ablations: images always retained; masks and position IDs unchanged\nS = demo state; A = demo action; all conditions use initial states 0..4",
        fontsize=12,
    )
    fig.savefig(root / "behavior_comparison.png", dpi=170)
    plt.close(fig)


def plot_samples(root):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    choices = [(v, layout) for layout in LAYOUTS for v in ("zero_state", "zero_state_action")]
    for row, (variant, layout) in enumerate(choices):
        record = read_rows(root / variant / "rollouts", layout, "correct")["milk", 0]
        reader = imageio.get_reader(root / variant / "rollouts" / layout / record["video"])
        for col, step in enumerate((0, 79, 159, 279)):
            ax = axes[row, col]
            ax.imshow(reader.get_data(step))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Control step {step + 1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(variant + "\n" + layout, fontsize=9)
        reader.close()
    fig.suptitle("Video spot check: milk instruction, correct demonstration, episode 0", fontsize=12)
    fig.savefig(root / "sample_video_frames.png", dpi=110)
    plt.close(fig)


def report(root, result):
    totals = result["totals"]
    retained = {
        "full_demo": "图像 + state + action",
        "zero_action": "图像 + state",
        "zero_state": "图像 + action",
        "zero_state_action": "仅示范图像",
    }
    total_table = [
        "| 条件 | 保留的示范内容 | 示范原位置上的物体入篮 | 语言目标入篮 | 两目标均未入篮 |",
        "|---|---|---:|---:|---:|",
    ]
    for v in VARIANTS:
        t = totals[v]
        total_table.append(
            f'| {LABELS[v]} | {retained[v]} | {t["demo_position_occupant_successes"]}/40 | {t["language_successes"]}/40 | {t["neither"]}/40 |'
        )
    cells = [
        "| 布局 | 语言 | context | 完整示范 | 去 action | 去 state | 去 state/action |",
        "|---|---|---|---|---|---|---|",
    ]
    for layout in LAYOUTS:
        for task in TASKS:
            for mode in ("correct", "wrong"):
                selected = [
                    next(
                        r
                        for r in result["comparison"]
                        if r["layout"] == layout
                        and r["language"] == task
                        and r["context"] == mode
                        and r["variant"] == v
                    )
                    for v in VARIANTS
                ]
                cells.append(
                    "| "
                    + ("原始" if layout == "original" else "交换位置")
                    + " | "
                    + task
                    + " | "
                    + mode
                    + " | "
                    + " | ".join(f'奶 {r["milk_in_basket"]} / 酱 {r["tomato_sauce_in_basket"]}' for r in selected)
                    + " |"
                )
    source_examples = ["| 条件 | 原布局 correct：语言目标成功 | 原布局 wrong：另一物体入篮 |", "|---|---:|---:|"]
    source_examples.extend(
        f'| {LABELS[v]} | {totals[v]["original_correct_language_successes"]}/10 | {totals[v]["original_wrong_other_successes"]}/10 |'
        for v in VARIANTS
    )
    pair_table = ["| 配对干预 | 原位置入篮：成功 → 失败 | 失败 → 成功 |", "|---|---:|---:|"]
    pair_table.extend(
        f'| {LABELS[c["before"]]} → {LABELS[c["after"]]} | {c["position_lost"]}/40 | {c["position_gained"]}/40 |'
        for c in result["contrasts"].values()
    )
    context_table = [
        "| 条件 | 换示范后，两个回合入篮物体不同 | 其中两回合均对应示范位置 | 两回合均为同一物体 | 其余配对 |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        c = result["context_switch_pairs"][v]
        context_table.append(
            f'| {LABELS[v]} | {c["different_single_objects"]}/20 | {c["both_follow_demo_region"]}/20 | {c["same_single_object_in_both"]}/20 | {c["other_pairs"]}/20 |'
        )
    switch = result["context_switch_pairs"]
    interpretation = (
        "在本次配对试验和表示置零干预下，示范 state 对维持‘随示范切换操作目标’的行为比显式 action 更关键。"
        "去掉 action 后这一行为基本保留；去掉 state 后，即使图像和 action 都保留，也未观察到一对回合成功切换入篮物体。"
        "但执行失败也增加了，所以这不是原模型完全不使用图像/action 的证据。"
        if switch["zero_state"]["different_single_objects"] == 0
        and switch["zero_action"]["different_single_objects"] > 0
        else "需要同时比较示范切换与执行成功：行为消失可能来自目标选择变化，也可能来自取放失败或 OOD。"
    )
    state_original_tomato = sum(
        r["tomato_sauce_in_basket"]
        for r in result["comparison"]
        if r["variant"] == "zero_state" and r["layout"] == "original"
    )
    state_swapped_milk = sum(
        r["milk_in_basket"]
        for r in result["comparison"]
        if r["variant"] == "zero_state" and r["layout"] == "swap_milk_tomato"
    )
    distractor_text = (
        "；".join(
            f"{LABELS[e['variant']]} / {e['layout']} / {e['language']} / {e['context']} / episode {e['episode']:03d}：{', '.join(e['objects'])} 入篮（两目标分类为 {e['outcome']}）"
            for e in result["distractor_events"]
        )
        or "两个新实验均未出现其他场景物体入篮"
    )
    text = f"""# 示范 state / state+action 表示消融结果

完成两个新实验，各 40 回合，共 80 回合。每个条件使用同一批 5 个初始状态；所有示范图像内容均保留。与历史完整示范和仅 action 置零逐回合配对，不拿 5 次比例与 25 次总比例比较。

**示范原位置上的物体入篮：完整示范 {totals['full_demo']['demo_position_occupant_successes']}/40；去 action {totals['zero_action']['demo_position_occupant_successes']}/40；去 state {totals['zero_state']['demo_position_occupant_successes']}/40；同时去 state/action {totals['zero_state_action']['demo_position_occupant_successes']}/40。**

## 主要判读

**{interpretation}**

更直接对应“固定语言，只换示范，行为是否跟着变”的指标是：20 对 correct/wrong 回合中，成功入篮物体随示范切换的配对数为完整示范 {switch['full_demo']['different_single_objects']}/20、去 action {switch['zero_action']['different_single_objects']}/20、去 state {switch['zero_state']['different_single_objects']}/20、去 state/action {switch['zero_state_action']['different_single_objects']}/20。完整示范和去 action 的这些切换都对应示范的原位置。

去 state 后，原布局有 {state_original_tomato}/20 回合将番茄酱放入篮子；交换后有 {state_swapped_milk}/20 回合将占据原番茄酱区域的牛奶放入篮子。分组表显示了其对语言/示范的变化响应。这个结果与偏向某个固定区域相容，但成功回合数有限，不能把其余未成功回合的目标选择自动补成同一种。

## 四种条件的对照

{chr(10).join(total_table)}

“示范原位置上的物体”用于检查按示范位置操作的倾向：例如牛奶示范在交换后，其原区域由番茄酱占据，番茄酱入篮就计入该列。它不是语言成功率，也不表示找到了示范中的同类物体。

所有条件保留当前图像、语言和机器人本体状态。表里的“仅示范图像”只针对示范部分，不是把模型所有其他输入移除。

## 分组行为

每个单元格为 **牛奶入篮次数 / 番茄酱入篮次数**，均以 5 次为分母；例如“奶 4 / 酱 0”表示牛奶 4/5、番茄酱 0/5。

{chr(10).join(cells)}

![四条件行为对照](behavior_comparison.png)

为区分纠正 wrong context 与整体执行失败，单独看原布局：

{chr(10).join(source_examples)}

交换布局中 wrong 的语言目标恰好处在示范原位置上，不能把这些成功自动解释成更服从语言。

额外物体：{distractor_text}。`chocolate_pudding_1` 是巧克力布丁。其视频见对应配对索引；不能将 `neither` 理解成没有操作或放入任何物体。

## 固定语言，只换示范，入篮物体是否跟着变

每种条件有 20 对 correct/wrong 回合（两布局 × 两语言 × 5 初始状态）；这里的比较固定当前语言、初始状态和噪声，只切换示范任务。“其余配对”包括至少一个回合未完成两种目标，或两种物体都曾入篮，不能算作明确的目标切换。

{chr(10).join(context_table)}

“其中两回合均对应示范位置”是“入篮物体不同”的子集；它更贴近此前观察到的按示范区域操作。这里统计完成入篮的行为，不将未成功入篮的回合说成没有抓取。

## 配对变化

{chr(10).join(pair_table)}

这些是同初始状态的变化，不是仅用总数相减。逐回合四条件结果见 [paired_outcomes.csv](paired_outcomes.csv)，视频见 [VIDEO_INDEX.md](VIDEO_INDEX.md)。

## 实验控制与验收

- 固定番茄酱任务场景，原始 / 牛奶番茄酱 XY 交换两种布局，牛奶 / 番茄酱两条语言，correct / wrong 两种示范，每格初始状态索引 0–4；每回合 280 步，每 5 步重规划，10 个 flow 步。
- 仅 state 置零：Gemma 前的 32 个示范 state token（当前布局 912:944）置零，保留示范图像/action；state/action 同时置零：这 32 个 state 与 32 个 action token（944:976）置零，保留示范图像。不是对物理动作或原始状态输入做零值替代。
- 三种消融均保留原 token 槽位、mask、位置编号。每个新实验 12 份真实样本及全部 120 份 rollout capture 检查未消融 token 精确不变。
- 每个新服务核对 60 份历史快照的普通模型 raw/executable actions 逐字节相同；8 份 correct/wrong 样本验证被消融模态的数值单独/联合改变后输出精确不变；4 份旧 masked no 快照输出保持不变（仅用于诊断，不计入行为表）。
- 80 条新轨迹与历史完整示范、仅 action 置零的初始模拟器状态、观察、布局干预来源、示范 episode、每次推理噪声全部配对；逐步入篮谓词与 JSON 一致。
- 每个新实验保留 120 份 attention，共 240 份（环境步 0/80/160，flow 步 0/5/9，层 0/9/17，全 8 个头）；每份普通/记录动作完全相同，并与实际执行动作核对。暂不分析热图。
- 模型参数 hash 与上一轮相同，运行前后不变。视频 20 FPS，按指令分文件夹。

## 视频抽查

[原始 / 交换布局的示例画面](sample_video_frames.png)来自两种新干预的牛奶语言、correct context、episode 0，分别显示第 1/80/160/280 步。完整回放见[配对视频索引](VIDEO_INDEX.md)。成功表只统计 LIBERO 的入篮谓词，不把未成功入篮解释为没有选中、抓取或搬运物体。

## 解释边界

这是表示置零干预，可能产生 OOD；不能仅凭失败认定原模型不使用保留模态，也不能把“去 state”直接等同于去掉所有位置信息。示范图像与 action 仍可能含空间/运动信息，示范 state 也可能含运动过程。有效零 token 在后续层仍可融合其他信息。

四格对照允许分别看有/无 action 时移除 state 的影响，以及有/无 state 时移除 action 的影响。每格仅 5 次，这是与上一轮对齐的探索性试验，不做显著性声明，也不能据此断言模型唯一依赖某个模态。

本报告不混入 no-context：昨天的全示范置零补跑使用各自任务场景，而本次使用固定番茄酱场景；旧位置交换 no 又采用关闭 mask 的定义，都不是这里四格对照的一部分。

[代码修改报告](CODE_CHANGES.md) · [机器可读结果](comparison.json) · [attention 索引](attention_manifest.json)
"""
    (root / "EXPERIMENT_RESULTS.md").write_text(text)
    for variant in ("zero_state", "zero_state_action"):
        selected = [r for r in result["comparison"] if r["variant"] in ("full_demo", "zero_action", variant)]
        (root / variant / "comparison.json").write_text(
            json.dumps({"variant": variant, "comparison": selected, "totals": totals[variant]}, indent=2) + "\n"
        )
        (root / variant / "EXPERIMENT_RESULTS.md").write_text(
            f'# {LABELS[variant]}\n\n本实验新增 40 回合，示范原位置上的物体入篮 {totals[variant]["demo_position_occupant_successes"]}/40，语言目标成功 {totals[variant]["language_successes"]}/40。\n\n完整分组对照、配对转移、验收与解释边界见[总报告](../EXPERIMENT_RESULTS.md)，代码见[修改报告](../CODE_CHANGES.md)。\n'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--baseline", type=Path, default="logs/quickstart/position_swap_object_25trials_20260922_175506"
    )
    parser.add_argument(
        "--action-reference",
        type=Path,
        default="logs/action_ablation/experiment2_zero_demo_action_5trials_20260923_195359",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, args.baseline.resolve(), args.action_reference.resolve())
    plot(root, result)
    plot_samples(root)
    report(root, result)
    print(json.dumps(result["totals"], indent=2))
