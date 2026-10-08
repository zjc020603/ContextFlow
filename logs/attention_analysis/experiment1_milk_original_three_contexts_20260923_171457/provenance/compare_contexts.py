# ruff: noqa: RUF001
"""Paired, offline comparison: original layout / milk / environment step zero."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.attention.make_tutorial import CAMERAS
from scripts.attention.make_tutorial import CHINESE
from scripts.attention.make_tutorial import LABELS
from scripts.attention.make_tutorial import image_uri

MODES = ("correct", "no", "wrong")
TITLES = ("Correct: milk demo", "No demonstration", "Wrong: tomato-sauce demo")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditional(values):
    mass = values.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("Cannot normalize zero camera attention")
    return values / mass


def load_episode(run, manifest, records, episode, hashes):
    common = None
    arrays, metadata = {}, {}
    for mode in MODES:
        case = f"original/{mode}/milk/ep{episode:03d}_step000"
        row = next(r for r in manifest["cases"] if r["case_id"] == case)
        directory = run / "validation_attempt3" / case
        obs_path = run / "observations" / row["npz"]
        with np.load(obs_path) as saved:
            inputs = {
                key: saved[key].copy() for key in ("observation/image", "observation/wrist_image", "observation/state")
            }
        with np.load(directory / "capture.npz") as capture:
            for key in ("noise", "tokenized_prompt", "language_mask"):
                inputs[key] = capture[key].copy()
            if capture["ordinary_actions"].tobytes() != capture["captured_actions"].tobytes():
                raise AssertionError(f"Capture equivalence failed: {case}")
            raw = capture["probabilities"].astype(np.float64)
            assert raw.shape == (3, 3, 1, 8, 51, 1027)
            assert np.isfinite(raw).all()
            assert np.all(raw[..., np.flatnonzero(~capture["prefix_mask"][0])] == 0)
            assert capture["layers"].tolist() == [0, 9, 17]
            assert capture["capture_steps"].tolist() == [0, 5, 9]
            arrays[mode] = raw[:, :, 0, :, 1:6, :].mean(axis=(2, 3))
            valid_prefix_count = int(capture["prefix_mask"].sum())
            assert valid_prefix_count == (524 if mode == "no" else 652)
        layout = json.loads((directory / "layout.json").read_text())
        inputs["prompt"] = row["request"]["prompt"]
        inputs["initial_state_hash"] = row["source_initial_state_sha256"]
        inputs["rng_key"] = records[case]["rng_key"]
        inputs["language_token_pieces"] = layout["language_token_pieces"]
        if common is None:
            common = inputs
        else:
            for key in common:
                np.testing.assert_array_equal(common[key], inputs[key], err_msg=f"Unpaired {key}: {case}")
        expected_demo = {"correct": [814], "no": [], "wrong": [821]}[mode]
        assert layout["context"]["selected_episode"] == expected_demo
        assert layout["context"]["task_index"] == 25
        if mode == "no":
            assert np.all(arrays[mode][..., 816:976] == 0)
        metadata[mode] = {
            "case_id": case,
            "context": layout["context"],
            "source_rng_key": inputs["rng_key"],
            "valid_prefix_count": valid_prefix_count,
            "first_action_position_id": valid_prefix_count + 1,
        }
        for path in (obs_path, directory / "capture.npz", directory / "layout.json"):
            hashes[str(path)] = digest(path)
    return common, arrays, metadata, layout["blocks"]


def plot_episode(out, episode, common, arrays, blocks):
    out.mkdir(parents=True, exist_ok=True)
    p = {m: arrays[m][2, 2] for m in MODES}
    fig, axs = plt.subplots(2, 4, figsize=(15, 8.2), constrained_layout=True)
    for camera, (_, obskey, label) in enumerate(CAMERAS):
        image = common[obskey]
        axs[camera, 0].imshow(image)
        axs[camera, 0].set_title(f"{label}\nIdentical input across contexts")
        for column, mode in enumerate(MODES, 1):
            values = p[mode][camera * 256 : (camera + 1) * 256]
            heat = values.reshape(16, 16) * 100
            ax = axs[camera, column]
            ax.imshow(image, extent=(0, 16, 16, 0))
            im = ax.imshow(
                heat, cmap="magma", vmin=0, vmax=heat.max(), alpha=0.52, interpolation="nearest", extent=(0, 16, 16, 0)
            )
            ax.set_title(f"{TITLES[column-1]}\nCamera total: {heat.sum():.3f}%")
            fig.colorbar(im, ax=ax, shrink=0.72, label="% of ALL keys / patch")
        for ax in axs[camera]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"Milk instruction / original layout / initial state {episode} / environment step 0\n"
        "Flow 9, layer 17; mean of 8 heads and first 5 action queries\n"
        "Each panel is contrast-expanded separately: compare numbers, not brightness",
        fontsize=13,
    )
    fig.savefig(out / "01_context_heatmaps.png", dpi=155)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 8.8), constrained_layout=True)
    metrics = []
    for camera, (_, obskey, label) in enumerate(CAMERAS):
        section = slice(camera * 256, (camera + 1) * 256)
        baseline = conditional(p["correct"][section])
        differences = [(conditional(p[m][section]) - baseline) * 100 for m in MODES[1:]]
        vmax = max(np.abs(d).max() for d in differences)
        for col, (mode, difference) in enumerate(zip(MODES[1:], differences, strict=True)):
            tv = np.abs(difference).sum() / 200
            metrics.append(
                {"episode": episode, "camera": label, "comparison": mode + " - correct", "spatial_tv": float(tv)}
            )
            ax = axs[camera, col]
            ax.imshow(common[obskey], extent=(0, 16, 16, 0))
            im = ax.imshow(
                difference.reshape(16, 16),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                alpha=0.65,
                extent=(0, 16, 16, 0),
            )
            ax.set_title(f"{label}: {mode} - correct\nSpatial redistribution (TV): {tv:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axs[camera], shrink=0.75, label="Camera-conditional percentage-point change")
    fig.suptitle(
        "Same input; compare WHERE attention is distributed inside each camera\n"
        "Red = higher relative share; blue = lower. Same scale across each row.\n"
        "Normalize each camera to 100% first; absolute camera totals are in Figure 1.",
        fontsize=12,
    )
    fig.savefig(out / "02_spatial_differences.png", dpi=155)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    labels = LABELS.copy()
    labels[-1] = "Current noisy-action sequence"
    tokens = [
        s.replace("▁", " ").replace("\n", "<newline>").strip()
        for s, valid in zip(common["language_token_pieces"], common["language_mask"][0], strict=True)
        if valid
    ]
    indices = np.flatnonzero(common["language_mask"][0]) + 768
    colors = ("#247c87", "#9a9a9a", "#bf633d")
    for j, mode in enumerate(MODES):
        values = np.array([p[mode][b["start"] : b["stop"]].sum() * 100 for b in blocks])
        axs[0].barh(np.arange(len(blocks)) + (j - 1) * 0.24, values, height=0.23, label=mode, color=colors[j])
        axs[1].barh(
            np.arange(len(tokens)) + (j - 1) * 0.24, p[mode][indices] * 100, height=0.23, label=mode, color=colors[j]
        )
    axs[0].set_yticks(np.arange(len(blocks)), labels=labels)
    axs[1].set_yticks(np.arange(len(tokens)), labels=tokens)
    for ax in axs:
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlabel("% of attention over ALL keys")
    axs[0].set_title("Block totals (blocks contain different numbers of tokens)")
    axs[1].set_title("Language tokens (same instruction)")
    fig.suptitle(
        "Same layer / flow step / heads / queries as Figure 1\n"
        "No context masks demo keys: a changed denominator also redistributes mass",
        fontsize=13,
    )
    fig.savefig(out / "03_attention_budget.png", dpi=155)
    plt.close(fig)
    return metrics


def build(args):
    run, out = args.run.resolve(), args.output.resolve()
    if (out / "summary.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "observations/manifest.json").read_text())
    records = {
        r["case_id"]: r for r in map(json.loads, (run / "validation_attempt3/pairs.jsonl").read_text().splitlines())
    }
    hashes, episodes, block_rows, metrics, all_arrays = {}, [], [], [], {}
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    for episode in range(5):
        common, arrays, metadata, blocks = load_episode(run, manifest, records, episode, hashes)
        metrics.extend(plot_episode(out / f"ep{episode:03d}", episode, common, arrays, blocks))
        for mode in MODES:
            all_arrays[f"ep{episode:03d}_{mode}"] = arrays[mode]
            for b, label in zip(blocks, CHINESE, strict=True):
                block_rows.append(
                    {
                        "episode": episode,
                        "context": mode,
                        "block": label,
                        "mass_percent": float(100 * arrays[mode][2, 2, b["start"] : b["stop"]].sum()),
                    }
                )
        episodes.append(
            {
                "episode": episode,
                "images": {name: image_uri(common[key]) for name, key, _ in CAMERAS},
                "views": {mode: arrays[mode].tolist() for mode in MODES},
                "context": metadata,
                "language_indices": np.flatnonzero(common["language_mask"][0]).tolist(),
                "tokens": [
                    s.replace("▁", " ").replace("\n", "<newline>").strip()
                    for s, v in zip(common["language_token_pieces"], common["language_mask"][0], strict=True)
                    if v
                ],
            }
        )
    summary = {
        "source_run": str(run),
        "task": "milk",
        "layout": "original",
        "environment_step": 0,
        "episodes": list(range(5)),
        "primary_episode": 0,
        "modes": MODES,
        "default_layer": 17,
        "default_flow_step": 9,
        "heads": "mean of 8",
        "queries": "first 5 action positions; state query excluded",
        "paired_checks": "images, state, language tokens/mask, instruction, initial simulator hash, noise, PRNG key: exact across all three contexts in each episode",
        "default_block_masses": block_rows,
        "default_spatial_differences": metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for filename, rows in [("block_masses.csv", block_rows), ("spatial_differences.csv", metrics)]:
        with (out / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(out / "averaged_attention.npz", **all_arrays)
    payload = {
        "episodes": episodes,
        "blocks": blocks,
        "labels": CHINESE[:-1] + ["当前采样步的动作序列"],
        "modes": MODES,
        "steps": [0, 5, 9],
        "layers": [0, 9, 17],
        "lut": (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist(),
        "diffLut": (plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist(),
    }
    template = Path(__file__).with_name("context_comparison.html")
    (out / "index.html").write_text(
        template.read_text().replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    )
    for path in [
        Path(__file__).resolve(),
        template.resolve(),
        run / "observations/manifest.json",
        run / "validation_attempt3/pairs.jsonl",
    ]:
        hashes[str(path)] = digest(path)
    (out / "provenance.json").write_text(json.dumps(hashes, indent=2) + "\n")
    write_report(out, summary)
    print(out)


def write_report(out, summary):
    masses = summary["default_block_masses"]
    rows = []
    for mode in MODES:
        selected = [r for r in masses if r["episode"] == 0 and r["context"] == mode]
        rows.append(
            "| " + mode + " | " + " | ".join(f"{selected[i]['mass_percent']:.3f}%" for i in (0, 1, 3, 10)) + " |"
        )
    metrics = summary["default_spatial_differences"]
    tvrows = []
    for episode in range(5):
        selected = [
            next(
                r["spatial_tv"]
                for r in metrics
                if r["episode"] == episode and r["camera"] == camera and r["comparison"] == comp
            )
            for camera in ("Front camera", "Wrist camera")
            for comp in ("no - correct", "wrong - correct")
        ]
        tvrows.append("| " + str(episode) + " | " + " | ".join(f"{v:.3f}" for v in selected) + " |")
    (out / "REPORT.md").write_text(
        """# 牛奶任务：correct / no / wrong context 的初始注意力对照

先看 [交互页面](index.html) 或下面的并排图。主例沿用教程的 episode 0；另外 4 个初始状态只用来检查这个现象是否重复出现，没有挑选更好看的热图。

## 比较口径

- **同一牛奶指令、原始布局、环境第 0 步**；correct 提供牛奶示范 episode 814，wrong 提供番茄酱示范 episode 821，no 关闭全部示范模态。
- 每个初始状态内，三种 context 的当前图像、机器人状态、语言 tokens/mask、初始模拟器状态 hash、初始动作噪声、PRNG key 均核对为完全相同。只在 context 处理上不同。
- 默认沿用教程：Gemma 层 17、采样步 9、8 个 heads 平均、前 5 个 action queries 平均，不包含 state query。
- 本次只读取实验 0 的 15 份记录，不新增推理或 rollout，不分析第 80/160 步。
- 注意：相同初始噪声不代表后面每个采样步的动作候选相同。更换 context 会改变前面更新得到的动作，最终层/步的差异包含这一累积过程。页面可切换采样步 0，查看尚未经历前面采样更新的情形；同一前向内部的早期层仍会受 context 影响。

## 1. 三种 context 并排看

![初始状态 0 的热图](ep000/01_context_heatmaps.png)

每行是一个相机，第一列是三组共用的原图；后面依次是 correct、no、wrong。**每张热图分别放大颜色**以看清位置，不能只凭亮度比较组间大小；相机总量写在图标题里，每张图也有自己的绝对色条。交互页面还提供“同相机、跨 context 共用刻度”，只在当前选择的三组中确定范围，不再被其他层或另一台相机的极值压暗。

默认初始状态 0 的总量：

| context | 主视角 | 手腕视角 | 语言 | 当前采样步的动作序列 |
|---|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + """

## 2. 分清总量变化与空间重分配

![相机内部归一化后的空间差值](ep000/02_spatial_differences.png)

差值图先把**每个相机内部**的 256 个权重归一化到总和 1，再计算 `no - correct`、`wrong - correct`。红色表示该位置在本相机内部的份额增加，蓝色表示减少。数值单位是百分点，与主图“对全部 keys 的百分比”不同；它刻意去掉相机总量变化，专门比较空间分布。同一相机的两个差值图使用相同的正负颜色范围。

使用 TV = `0.5 × sum(abs(归一化分布A - 归一化分布B))` 描述空间重分配，范围 0–1；0 表示分布完全相同，1 表示完全不重叠。它不是显著性检验、正确率，也不是“多少物体改变了注意力”。

| 初始状态 | 主视角 no−correct | 主视角 wrong−correct | 手腕 no−correct | 手腕 wrong−correct |
|---|---:|---:|---:|---:|
"""
        + "\n".join(tvrows)
        + """

在这 5 个固定初始状态、默认层/步及平均方式下，**关闭示范引起的空间分布变化，明显大于把牛奶示范换成番茄酱示范引起的变化**。correct 与 wrong 的分布并非相同，但变化较小。这一比较描述当前记录，不外推为所有层、所有时刻或所有任务的结论。

## 3. 语言与全部 keys 的权重

![各部分和语言 token 的权重](ep000/03_attention_budget.png)

no-context 中示范 keys 被屏蔽，softmax 的可用 keys 集合变了。因此其他部分总量改变既可能包含重新归一化的影响，也可能包含隐藏表征和动作候选改变的影响，不能全部解释为“模型更依赖某模态”。上面的相机内归一化差值可以排除简单的相机总量缩放，但仍不是因果贡献分解。

还有一个具体结构变化：correct/wrong 的有效 prefix token 数为 **652**，no 为 **524**。原模型根据有效 prefix 数计算 suffix 的位置编号，所以第一个 action token 的位置编号从 **653** 变成 **525**（减少 128）；图像的有效位置编号不随之平移。这意味着 no 干预同时改变了示范信息、可用 keys 和 action 相对图像的位置编码。热图位置的大幅变化可能包含这些因素，不能单独归为示范语义的作用；本次没有为拆开这些因素新增对照。

## 当前可以下到哪一步结论

1. 三种 context 的输入配对成立，可以把此次推理中出现的差异归于 context 干预及其下游计算过程，而非初始画面不同。
2. 默认视图中，no-context 对空间 attention 分布的改变很明显；correct/wrong 的差别较小，且这个相对关系在其余四个初始状态重复出现。
3. **这些平均热图目前不足以证明“错误示范把注意力明确从牛奶转移到了番茄酱”。** 没有加入物体分割/投影区域，不能把背景或夹爪的热点自动算到某个物体。也不能因为 correct/wrong 热图接近，就断言示范不影响行为：动作 token、value 向量、其他层/head 和残差仍可能不同。
4. 先据此判读图像，再决定是否增加更细的 head/query 分析或干预；本次没有自动扩大到其他任务、布局、环境步或遮挡实验。

## 数据与复现

- [汇总数值](summary.json)、[模态总量 CSV](block_masses.csv)、[空间差异 CSV](spatial_differences.csv)。
- [平均前向记录](averaged_attention.npz)：每个初始状态/context 保存 3 个采样步 × 3 层 × 1027 keys 的 head/query 平均结果；原始全 heads/queries 仍在实验 0 目录。
- [来源 SHA256](provenance.json)。
- `ep001` 到 `ep004` 各有与主例相同格式的三张图，也可在交互页面切换。

```bash
source scripts/activate_env.sh train
python -m scripts.attention.compare_contexts \\
  --run logs/attention_capture/experiment0_consistency_20260923_143830 \\
  --output logs/attention_analysis/milk_original_three_contexts_new
```
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
