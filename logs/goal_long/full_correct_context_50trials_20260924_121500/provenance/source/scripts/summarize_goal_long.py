"""Audit all Goal/Long episodes, report task rates and predicate diagnostics."""
# ruff: noqa: RUF001

import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import imageio.v2 as imageio
import jax
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = "https://arxiv.org/html/2609.06852v1"


def wilson(k, n):
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (k / n + z * z / (2 * n)) / denominator
    spread = z * np.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def noise_keys(count):
    def step(key, _):
        key, sample = jax.random.split(key)
        return key, jax.random.key_data(sample)

    return np.asarray(jax.lax.scan(step, jax.random.key(0), None, length=count)[1])


def audit(root):
    tasks = json.loads((root / "task_manifest.json").read_text())["tasks"]
    rows = []
    clauses = []
    videos = []
    manifest = []
    server_finish = list((root / "servers").glob("gpu*/FINISHED.json"))
    assert len(server_finish) == 8
    hashes = set()
    finished_ids = []
    server_calls = 0
    for p in server_finish:
        value = json.loads(p.read_text())
        assert value["passed"]
        assert value["parameter_sha256_before"] == value["parameter_sha256_after"]
        hashes.add(value["parameter_sha256_before"])
        finished_ids.extend(value["tasks"])
        server_calls += value["total_inferences"]
    assert len(hashes) == 1
    assert sorted(finished_ids) == list(range(20))
    client_calls = 0
    for task in tasks:
        directory = root / "tasks" / task["task_key"]
        records = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
        assert [r["episode"] for r in records] == list(range(50))
        result = json.loads((directory / "results.json").read_text())
        assert result["episodes"] == 50
        assert result["task"] == task
        success = 0
        subgoals = []
        final = []
        initial = []
        every_seen_without_complete = 0
        all_keys = []
        indices = []
        for r in records:
            assert r["dataset_task_index"] == task["dataset_task_index"]
            with np.load(root / r["trajectory"]) as trace:
                pred = trace["goal_predicates"]
                act = trace["actions"]
                initial.append(r["initial_goals"])
                assert act.shape == (r["steps"], 7)
                assert np.isfinite(act).all()
                assert pred.shape == (r["steps"], len(task["goals"]))
                full = bool(pred.all(1).any())
                assert full == r["success"]
                assert pred.any(0).tolist() == r["goal_ever"]
                assert pred[-1].tolist() == r["goal_final"]
                assert np.isfinite(trace["sim_states"]).all()
                assert hashlib.sha256(trace["source_sim_state"].tobytes()).hexdigest() == r["source_sim_state_sha256"]
                assert hashlib.sha256(trace["initial_sim_state"].tobytes()).hexdigest() == r["initial_sim_state_sha256"]
                if full:
                    assert pred[-1].all()
                    assert not pred[:-1].all(1).any()
                else:
                    assert r["steps"] == task["max_steps"]
                for key, name in [
                    ("observation/image", "initial_image"),
                    ("observation/wrist_image", "initial_wrist"),
                    ("observation/state", "initial_robot_state"),
                ]:
                    assert hashlib.sha256(trace[name].tobytes()).hexdigest() == r["initial_observation_sha256"][key]
            assert (root / r["video"]).is_file()
            assert len(r["rng_records"]) == (r["steps"] + 4) // 5
            for record in r["rng_records"]:
                assert record["selected_episode"] == [task["demo_episode"]]
                assert record["task_index"] == task["dataset_task_index"]
                all_keys.append(record["rng_key"])
                indices.append(record["inference_index"])
            success += full
            subgoals.append(r["goal_ever"])
            final.append(r["goal_final"])
            every_seen_without_complete += int(all(r["goal_ever"]) and not full)
            manifest.append(
                {
                    "task_index": task["dataset_task_index"],
                    "episode": r["episode"],
                    "trajectory": r["trajectory"],
                    "video": r["video"],
                    "success": full,
                }
            )
        assert success == result["successes"]
        assert indices == list(range(len(indices)))
        np.testing.assert_array_equal(np.asarray(all_keys, dtype=np.uint32), noise_keys(len(indices)))
        assert result["server_audit"]["inferences"] == len(indices)
        client_calls += len(indices)
        ci = wilson(success, 50)
        row = {
            "suite": task["suite"],
            "split": task["split"],
            "dataset_task_index": task["dataset_task_index"],
            "suite_task_id": task["suite_task_id"],
            "instruction": task["instruction"],
            "successes": success,
            "trials": 50,
            "success_rate_percent": 2 * success,
            "wilson95_low_percent": 100 * ci[0],
            "wilson95_high_percent": 100 * ci[1],
            "demo_episode": task["demo_episode"],
            "demo_length": task["demo_length"],
            "training_episodes_in_released_config": task["retained_training_episodes_in_released_config"],
            "all_clauses_ever_true_without_full_success": every_seen_without_complete,
        }
        rows.append(row)
        for i, goal in enumerate(task["goals"]):
            clauses.append(
                {
                    "dataset_task_index": task["dataset_task_index"],
                    "suite": task["suite"],
                    "split": task["split"],
                    "goal_clause": " ".join(goal),
                    "initially_true": int(np.asarray(initial)[:, i].sum()),
                    "ever_true": int(np.asarray(subgoals)[:, i].sum()),
                    "finally_true": int(np.asarray(final)[:, i].sum()),
                    "newly_true_from_initially_false": int(
                        (np.asarray(subgoals)[:, i] & ~np.asarray(initial, dtype=bool)[:, i]).sum()
                    ),
                    "trials": 50,
                }
            )
        input_audits = list((root / "servers").glob(f"gpu*/task{task['dataset_task_index']:02d}_inputs/audit.json"))
        assert len(input_audits) == 1
        inp = json.loads(input_audits[0].read_text())
        assert inp["selected_episode"] == [task["demo_episode"]]
        assert inp["demo_intervention"] == "none"
        assert sum(b["valid_tokens"] for b in inp["demo_blocks"]) == 128
        for block in inp["demo_blocks"]:
            if block["valid_tokens"]:
                assert block["token_l2"] > 0
        example = [next((r for r in records if r["success"] == flag), None) for flag in (True, False)]
        videos.append({"task": task, "examples": [r for r in example if r is not None]})
    assert client_calls == server_calls
    groups = []
    for suite in ("libero_goal", "libero_10"):
        for split in ("unseen", "seen"):
            selected = [r for r in rows if r["suite"] == suite and r["split"] == split]
            k = sum(r["successes"] for r in selected)
            n = sum(r["trials"] for r in selected)
            groups.append(
                {
                    "suite": suite,
                    "split": split,
                    "tasks": len(selected),
                    "successes": k,
                    "trials": n,
                    "success_rate_percent": 100 * k / n,
                    "paper_rate_percent": 0 if split == "unseen" else (90 if suite == "libero_goal" else 82),
                }
            )
    out = {
        "rows": rows,
        "suite_summaries": groups,
        "goal_clauses": clauses,
        "total_episodes": 1000,
        "validated_model_inferences": client_calls,
        "parameter_sha256": next(iter(hashes)),
        "protocol": "full correct context; per-task seed 0, continuous within task",
        "paper_reference": PAPER,
    }
    for filename, values in [
        ("success_rates.csv", rows),
        ("goal_clause_rates.csv", clauses),
        ("suite_summary.csv", groups),
    ]:
        with (root / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (root / "comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    (root / "episode_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# 回放索引",
        "",
        "每个任务完整 50 个视频；下表各列一个成功/失败示例（如存在）。全部为 20 FPS。",
        "",
        "| ID | 任务 | 示例 |",
        "|---:|---|---|",
    ]
    for item in videos:
        links = " · ".join(
            f"[ep{r['episode']:03d} {'成功' if r['success'] else '失败'}]({r['video']})" for r in item["examples"]
        )
        lines.append(f"| {item['task']['dataset_task_index']} | {item['task']['instruction']} | {links} |")
    (root / "VIDEO_INDEX.md").write_text("\n".join(lines) + "\n")
    return out


def plot(root, result):
    fig, axes = plt.subplots(1, 2, figsize=(19, 10))
    for ax, suite in zip(axes, ("libero_goal", "libero_10"), strict=True):
        rows = sorted(
            [r for r in result["rows"] if r["suite"] == suite],
            key=lambda r: (r["split"] != "unseen", r["dataset_task_index"]),
        )
        y = np.arange(len(rows))
        ax.barh(
            y,
            [r["success_rate_percent"] for r in rows],
            color=["#d77932" if r["split"] == "unseen" else "#397bad" for r in rows],
        )
        ax.set_yticks(
            y,
            [f"ID {r['dataset_task_index']} [{r['split']}]\n" + textwrap.fill(r["instruction"], 43) for r in rows],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 112)
        ax.set_xlabel("Success rate (%)")
        ax.set_title(suite + " — full correct context")
        for i, r in enumerate(rows):
            ax.text(r["success_rate_percent"] + 1, i, f"{r['successes']}/50", va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(root / "success_rates.png", dpi=160)
    fig.savefig(root / "success_rates.svg")
    plt.close(fig)


def render_examples(root):
    tasks = json.loads((root / "task_manifest.json").read_text())["tasks"]
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    for row, task_id in enumerate((10, 17, 1, 5)):
        task = next(t for t in tasks if t["dataset_task_index"] == task_id)
        record = json.loads((root / "tasks" / task["task_key"] / "episodes.jsonl").read_text().splitlines()[0])
        reader = imageio.get_reader(root / record["video"])
        for ax, step in zip(axes[row], np.linspace(0, record["steps"] - 1, 5, dtype=int), strict=True):
            ax.imshow(reader.get_data(int(step)))
            ax.set_title(f"Task {task_id}, ep0, step {step}")
            ax.axis("off")
        reader.close()
    fig.tight_layout()
    fig.savefig(root / "unseen_episode0_frames.png", dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    for row, task_id in enumerate((10, 17)):
        task = next(t for t in tasks if t["dataset_task_index"] == task_id)
        path = next((root / "servers").glob(f"gpu*/task{task_id:02d}_inputs/demo_and_observation.npz"))
        with np.load(path) as data:
            frames = data["demo_base_0_rgb"][0]
            for i, ax in enumerate(axes[row]):
                ax.imshow(np.clip((frames[i] + 1) * 127.5, 0, 255).astype(np.uint8))
                ax.set_title(f"Task {task_id}, demo frame {task['demo_image_frame_indices'][i]}", fontsize=9)
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(root / "goal_unseen_actual_demo_frames.png", dpi=125)
    plt.close(fig)


def report(root, result):
    text = [
        "# ContextFlow：Goal / Long 全任务评估",
        "",
        "已完成 20 个任务 × 50 次，共 1,000 回合。先评估论文指定的 4 个 unseen 任务，再评估其余 16 个 seen 任务。所有回合使用完整正确示范。",
        "",
        "## 首先修正训练范围的理解",
        "",
        f"论文 §4.1 表明训练使用四个 suite；Table S4 每个 suite 划出两个 unseen 任务。发布配置 `remove_task_list=LIBERO_UNSEEN_TASKS` 和本地数据元信息与之相符。因此 Goal/Long 其余任务属于 seen，不是额外的 unseen 泛化任务。[论文]({PAPER})",
        "",
        "这是按论文和发布配置判定的划分；检查点没有附带可独立审计的完整训练数据日志，不能仅从权重反推出实际训练样本。",
        "",
        "## 与论文 Table S3 对照",
        "",
        "| Suite | 划分 | 本次成功数 | 本次成功率 | 论文成功率 |",
        "|---|---|---:|---:|---:|",
    ]
    text.extend(
        f"| {r['suite']} | {r['split']} ({r['tasks']} tasks) | {r['successes']}/{r['trials']} | {r['success_rate_percent']:.2f}% | {r['paper_rate_percent']:.1f}% |"
        for r in result["suite_summaries"]
    )
    heldout = [r for r in result["rows"] if r["split"] == "unseen"]
    if sum(r["successes"] for r in heldout) == 0:
        text += [
            "",
            "**四个论文 unseen 任务均为 0/50，复现了该检查点在这些任务上的零观测成功率。** 0/50 不是对真实成功概率严格为零的证明；逐任务的 Wilson 95% 区间见 CSV。",
        ]
    else:
        text += [
            "",
            "**本次 unseen 结果没有逐项复现全部 0%；如实保留观测到的成功，不通过调整种子或判据追求零成功率。**",
        ]
    text += [
        "",
        "本次仅评估 run1/19999 这一个检查点、每任务一条固定正确示范；50 回合覆盖官方初始状态，不代表对不同示范和训练运行取平均。论文汇总用于参照，不要求 seen 成功率逐点相等。",
        "",
        "## 全部任务成功率",
        "",
        "| Suite | 数据集 ID | 划分 | 原始任务指令 | 成功数 | 成功率 |",
        "|---|---:|---|---|---:|---:|",
    ]
    for r in sorted(result["rows"], key=lambda r: (r["suite"], r["split"] != "unseen", r["dataset_task_index"])):
        text.append(
            f"| {r['suite']} | {r['dataset_task_index']} | {r['split']} | {r['instruction']} | {r['successes']}/50 | {r['success_rate_percent']:.0f}% |"
        )
    text += [
        "",
        "![全任务成功率](success_rates.png)",
        "",
        "## Long 的部分目标完成情况",
        "",
        "这些是 BDDL 目标谓词，不是自动识别的动作基元。整任务成功要求同一时刻全部成立；不同时间分别成立不会累计成整任务成功。初始已满足的条件不代表模型执行了相应操作。",
        "",
        "| Task ID | 目标谓词 | 初始满足 | 曾满足 | 结束时满足 | 初始不满足而后来满足 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in result["goal_clauses"]:
        if r["suite"] == "libero_10":
            text.append(
                f"| {r['dataset_task_index']} | `{r['goal_clause']}` | {r['initially_true']}/50 | {r['ever_true']}/50 | {r['finally_true']}/50 | {r['newly_true_from_initially_false']}/50 |"
            )
    text += ["", "论文两个 Long unseen 的目标条件分别为：", ""]
    for task_id in (1, 5):
        selected = [r for r in result["goal_clauses"] if r["dataset_task_index"] == task_id]
        details = "；".join(
            f"`{r['goal_clause']}` 曾满足 {r['ever_true']}/50，最后满足 {r['finally_true']}/50" for r in selected
        )
        text.append(f"- Task {task_id}：{details}。")
    text += [
        "",
        "这些统计区分各个目标从未完成与仅完成部分目标，但不能把目标条件直接等同于时序动作基元。要严格研究组合泛化，还需在同布局下分别验证每个单项操作及其组合；本轮不额外改变指令或示范来做这项干预。",
        "",
        "## 对研究切入点的解释",
        "",
        "Goal 的两个 unseen 都是放碗任务，可与 seen 的“碗放到柜顶”等任务对照；这一比较比“整个 Goal 都未训练”更准确。Goal suite 同时包含开抽屉、开炉子、推盘子，Long 也包含单目标放书和开关器具的组合，不能仅依据 suite 名称把所有任务分成单基元/双基元。",
        "",
        "已直接比较 Task 10（碗放盘子）、17（碗放炉子）、18（碗放柜顶）的 BDDL：物体、fixtures、区域、场景属性和初始关系完全一致，目标与语言不同；官方具体初始状态数组不相同，见 goal_scene_comparison.json。因此这里可以重点研究同场景分布下改变放置目标的泛化，但当前不是逐初始状态匹配的因果对照。",
        "",
        "本实验能定位哪个目标/布局会失败，并通过 Long 的目标谓词区分部分完成与完全失败。成功率差异本身不能证明失败来自组合推理；还可能涉及目标位置、操作高度、示范压缩、轨迹分布等。这里没有做位置、语言或示范模态干预，也没有用 attention 作因果解释。",
        "",
        "## 实际示范与回合 0 画面",
        "",
        "下面展示的是服务实际收到的两条 Goal 示范，而非另选的示意轨迹。两条示范分别把碗送向盘子和炉子。四个 unseen 的 rollout 均固定选 episode 0，避免按成败挑选展示案例；画面只能说明这些样例，不作为全部回合的行为分类统计。",
        "",
        "![Goal 实际示范采样帧](goal_unseen_actual_demo_frames.png)",
        "",
        "![四个 unseen 的第 0 回合](unseen_episode0_frames.png)",
        "",
        "## 协议与实现细节",
        "",
        "- 检查点：`ContextFlow_run1/19999`；普通 ContextFlow 模型，float32 权重与 JAX matmul precision，Gemma 内部仍按原配置 bfloat16。没有零化、删除或关闭示范 mask。",
        "- 每任务取数据集同任务的第一条演示，固定供 50 次推理使用。沿用全轨迹均匀采样 8 帧图像、最多 128 个状态/action 点及原 Perceiver 压缩。Long 使用一条完整组合任务示范，不拼接两条单任务示范，也不提供阶段切换信号。具体 episode/采样帧索引见 task_manifest.json。短于 128 帧时原加载器重复末尾 state/action 补齐；本次 Goal 两条 unseen 示范分别补 25、16 个重复位置，未修改该策略。",
        "- 每个任务重置环境随机种子 7 和策略 key(0)，随后沿用原连续 split 随机流，使用官方初始状态 0–49。任务间独立可并行；这与此前每 suite 连续策略随机流的部署安排不同，因此不是承诺逐帧复刻作者未公开的随机流。",
        "- 等待 10 个稳定步，Goal 最多执行 300 步，Long 最多 520 步。每 5 步重新规划，50 步动作窗口、10 步 flow 采样。完整目标首次成功就停止，失败跑满时限，不使用先前消融实验的固定 280 步规则。",
        "- 错误会终止评估，不计为策略失败；不会筛选初始状态、补种子到指定成功率或给 Long 的单个子目标记完整成功。",
        "- 视频按原速 20 FPS 保存在每任务的指令目录；保留逐步动作、仿真状态、末端/夹爪状态、目标谓词、初始图像和每次推理的 RNG key。没有为本实验采集 attention。",
        "",
        "## 验证与材料",
        "",
        f'- 八个服务各自通过历史完整示范样本的 raw/executable action 逐字节一致性检查；运行前后模型参数 hash 一致：`{result["parameter_sha256"]}`。',
        f'- 1,000 条轨迹的完整目标与逐步谓词、JSON 成功计数一致；{result["validated_model_inferences"]} 次推理的 RNG 序列、示范 episode 均已核对。20 个任务首次输入保留了所有有效示范 token，审计快照见 servers/。',
        "- 逻辑 worker 5 两次在启动参数读取检查时停滞，未开始回合；迁移到物理 GPU 0 后通过同一数值检查，完成原定任务。其余回合不重抽。资源映射与清理记录见 provenance/startup_recovery.json。",
        "- [逐任务 CSV](success_rates.csv)、[seen/unseen 汇总](suite_summary.csv)、[全部目标谓词 CSV](goal_clause_rates.csv)、[完整 JSON](comparison.json)、[视频入口](VIDEO_INDEX.md)、[代码修改报告](CODE_CHANGES.md)。",
        "",
    ]
    (root / "EXPERIMENT_RESULTS.md").write_text("\n".join(text))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    result = audit(args.root)
    plot(args.root, result)
    render_examples(args.root)
    report(args.root, result)
    print(json.dumps(result["suite_summaries"], indent=2))
