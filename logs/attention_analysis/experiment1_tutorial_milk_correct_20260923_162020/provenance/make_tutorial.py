# ruff: noqa: RUF001  # Chinese tutorial prose intentionally uses Chinese punctuation.
"""Build a self-contained attention tutorial from one verified experiment-0 case.

No model loading/inference. Requires numpy, matplotlib and Pillow only.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CAMERAS = [
    ("base_0_rgb", "observation/image", "Front camera"),
    ("left_wrist_0_rgb", "observation/wrist_image", "Wrist camera"),
]
LABELS = [
    "Current front image",
    "Current wrist image",
    "Current right image (masked)",
    "Language",
    "Demo front image latents",
    "Demo wrist image latents",
    "Demo right image (masked)",
    "Demo state latents",
    "Demo action latents",
    "Current state",
    "Predicted action tokens",
]
CHINESE = [
    "当前主视角图像",
    "当前手腕图像",
    "当前右手图像（屏蔽）",
    "语言",
    "示范主视角压缩 token",
    "示范手腕压缩 token",
    "示范右手图像（屏蔽）",
    "示范状态压缩 token",
    "示范动作压缩 token",
    "当前状态",
    "预测动作 token",
]
SCOPES = {
    "first5": (1, 6, "前 5 个动作位置，平均"),
    "first": (1, 2, "第 1 个动作位置"),
    "all50": (1, 51, "全部 50 个动作位置，平均"),
}


def image_uri(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def build(args):
    run = args.run.resolve()
    directory = run / "validation_attempt3" / args.case
    manifest = json.loads((run / "observations/manifest.json").read_text())
    case = next(row for row in manifest["cases"] if row["case_id"] == args.case)
    observation_path = run / "observations" / case["npz"]
    layout = json.loads((directory / "layout.json").read_text())
    with np.load(directory / "capture.npz") as saved:
        raw = saved["probabilities"].astype(np.float64)
        layers, steps = saved["layers"].tolist(), saved["capture_steps"].tolist()
        valid_language = saved["language_mask"][0].copy()
        prefix_mask = saved["prefix_mask"][0].copy()
        assert saved["ordinary_actions"].tobytes() == saved["captured_actions"].tobytes()
    assert raw.shape == (3, 3, 1, 8, 51, 1027)
    assert np.isfinite(raw).all()
    assert np.all(raw[..., np.flatnonzero(~prefix_mask)] == 0)
    with np.load(observation_path) as saved:
        images = {name: saved[key].copy() for name, key, _ in CAMERAS}
    assert all(im.shape == (224, 224, 3) and im.dtype == np.uint8 for im in images.values())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(out / "summary.json")
    blocks = layout["blocks"]
    assert len(blocks) == len(LABELS)
    for name, _, _ in CAMERAS:
        block = next(b for b in blocks if b["kind"] == "current_image" and b["name"] == name)
        assert block["stop"] - block["start"] == 256
        Image.fromarray(images[name]).save(out / f"{name}.png")
    # [step, layer, batch, head, query, key] -> [step, layer, key].
    averages = {name: raw[:, :, 0, :, start:stop, :].mean(axis=(2, 3)) for name, (start, stop, _) in SCOPES.items()}
    p = averages["first5"][2, 2]
    assert np.allclose(averages["first5"].sum(axis=-1), 1, atol=0.004, rtol=0)
    mass = np.array([p[b["start"] : b["stop"]].sum() for b in blocks])
    np.testing.assert_allclose(mass.sum(), p.sum(), atol=1e-14, rtol=0)
    # A fixed absolute color range shared by both cameras, every saved layer/step/query scope.
    global_vmax = max(a[..., :512].max() for a in averages.values()) * 100
    language_block = next(b for b in blocks if b["kind"] == "language")
    token_indices = np.flatnonzero(valid_language)
    token_labels = [
        layout["language_token_pieces"][i].replace("▁", " ").replace("\n", "<newline>").strip() for i in token_indices
    ]
    token_values = p[language_block["start"] + token_indices] * 100
    token_labels = [s or "<space>" for s in token_labels]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 8), constrained_layout=True)
    for row, (camera, _, title) in enumerate(CAMERAS):
        b = next(b for b in blocks if b["name"] == camera and b["kind"] == "current_image")
        heat = p[b["start"] : b["stop"]].reshape(16, 16) * 100
        axs[row, 0].imshow(images[camera])
        axs[row, 0].set_title(f"{title}: actual 224 x 224 input")
        m = axs[row, 1].imshow(heat, cmap="magma", vmin=0, vmax=heat.max(), interpolation="nearest")
        axs[row, 1].set_title(f"16 x 16 patch weights | camera total {heat.sum():.3f}%")
        fig.colorbar(m, ax=axs[row, 1], shrink=0.8, label="% of attention across ALL keys / patch")
        axs[row, 2].imshow(images[camera], extent=(0, 16, 16, 0))
        axs[row, 2].imshow(
            heat, cmap="magma", vmin=0, vmax=heat.max(), alpha=0.48, extent=(0, 16, 16, 0), interpolation="nearest"
        )
        y, x = np.unravel_index(heat.argmax(), heat.shape)
        axs[row, 2].add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor="#00e5ff", linewidth=2))
        axs[row, 2].set_title(f"Overlay | outlined patch: {heat.max():.4f}%")
        for ax in axs[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "One observation: milk / correct context / environment step 0\n"
        "Flow step 9, layer 17 (zero-based); mean of 8 heads and first 5 action queries\n"
        "Each camera has its OWN color range here: compare numbers, not brightness",
        fontsize=13,
    )
    fig.savefig(out / "01_read_the_heatmap.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 6.2), constrained_layout=True)
    bars = axs[0].barh(LABELS[::-1], (mass * 100)[::-1], color="#31678a")
    axs[0].bar_label(bars, fmt="%.3f%%", padding=3, fontsize=9)
    axs[0].set_xlim(0, max(mass * 100) * 1.23)
    axs[0].set_xlabel("Summed weight of ALL tokens in the block (%)")
    axs[0].set_title(f"Attention budget across all keys | sum {100 * p.sum():.4f}%")
    bars = axs[1].barh(np.arange(len(token_labels)), token_values[::-1], color="#b26726")
    axs[1].set_yticks(np.arange(len(token_labels)), labels=token_labels[::-1])
    axs[1].bar_label(bars, fmt="%.4f%%", padding=3, fontsize=9)
    axs[1].set_xlim(0, max(token_values) * 1.4)
    axs[1].set_title(f"Language tokens | total {token_values.sum():.4f}%")
    axs[1].set_xlabel("Weight per token as % of ALL keys (NOT language-only %)")
    fig.suptitle(
        "Same layer, flow step, heads and queries as Figure 1\n"
        "These are attention weights, not causal importance scores; block sizes differ",
        fontsize=13,
    )
    fig.savefig(out / "02_attention_budget.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(11.8, 7.6), constrained_layout=True)
    front_vmax = averages["first5"][..., :256].max() * 100
    for row in range(2):
        for col in range(3):
            si, li = (2, col) if row == 0 else (col, 2)
            heat = averages["first5"][si, li, :256].reshape(16, 16) * 100
            ax = axs[row, col]
            ax.imshow(images["base_0_rgb"], extent=(0, 16, 16, 0))
            m = ax.imshow(
                heat, cmap="magma", vmin=0, vmax=front_vmax, alpha=0.5, extent=(0, 16, 16, 0), interpolation="nearest"
            )
            ax.set_title(f"Flow {steps[si]}, layer {layers[li]} | front total {heat.sum():.3f}%")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(m, ax=axs, shrink=0.7, label="% per patch; SAME scale in all six panels")
    fig.suptitle(
        "Same observation; change ONE internal coordinate at a time\n"
        "Top: hold flow step 9, vary layer. Bottom: hold layer 17, vary flow step.\n"
        "The robot has not moved between these panels; mean of 8 heads / first 5 queries",
        fontsize=12,
    )
    fig.savefig(out / "03_layers_and_flow_steps.png", dpi=160)
    plt.close(fig)
    peak_info = []
    for i, (camera, _, _) in enumerate(CAMERAS):
        heat = p[i * 256 : (i + 1) * 256]
        j = int(heat.argmax())
        row, col = divmod(j, 16)
        peak_info.append(
            {
                "camera": camera,
                "camera_mass_percent": float(100 * heat.sum()),
                "patch_index": j,
                "row": row,
                "col": col,
                "pixel_x_half_open": [col * 14, (col + 1) * 14],
                "pixel_y_half_open": [row * 14, (row + 1) * 14],
                "patch_global_percent": float(100 * heat[j]),
                "patch_within_camera_percent": float(100 * heat[j] / heat.sum()),
            }
        )
    summary = {
        "case_id": args.case,
        "source_run": str(run),
        "instruction": layout["instruction"],
        "default_layer": 17,
        "default_flow_step": 9,
        "heads": "mean of 8",
        "action_queries": "first 5; suffix indices [1,6)",
        "raw_probability_shape": list(raw.shape),
        "total_percent": float(100 * p.sum()),
        "blocks": [
            {
                **b,
                "label": lab,
                "mass_percent": float(v * 100),
                "valid_tokens": int(prefix_mask[b["start"] : b["stop"]].sum())
                if b["stop"] <= len(prefix_mask)
                else b["stop"] - b["start"],
            }
            for b, lab, v in zip(blocks, CHINESE, mass, strict=True)
        ],
        "language": [{"token": t, "global_percent": float(v)} for t, v in zip(token_labels, token_values, strict=True)],
        "image_peaks": peak_info,
        "shared_color_vmax_percent": float(global_vmax),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(out / "averaged_attention.npz", **averages, layers=layers, capture_steps=steps)
    with (out / "blocks.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["label", "tokens", "valid_tokens", "mass_percent", "mean_per_valid_token_percent"]
        )
        writer.writeheader()
        for b in summary["blocks"]:
            n = b["valid_tokens"]
            writer.writerow(
                {
                    "label": b["label"],
                    "tokens": b["stop"] - b["start"],
                    "valid_tokens": n,
                    "mass_percent": b["mass_percent"],
                    "mean_per_valid_token_percent": b["mass_percent"] / n if n else 0,
                }
            )
    payload = {
        "summary": summary,
        "blocks": blocks,
        "blockLabels": CHINESE,
        "layers": layers,
        "steps": steps,
        "scopeLabels": {k: v[2] for k, v in SCOPES.items()},
        "views": {k: v.tolist() for k, v in averages.items()},
        "images": {k: image_uri(v) for k, v in images.items()},
        "tokens": token_labels,
        "tokenIndices": token_indices.tolist(),
        "lut": (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist(),
    }
    template = Path(__file__).with_name("tutorial.html").read_text()
    (out / "index.html").write_text(
        template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    )
    sources = [
        directory / "capture.npz",
        directory / "layout.json",
        observation_path,
        Path(__file__),
        Path(__file__).with_name("tutorial.html"),
    ]
    (out / "provenance.json").write_text(
        json.dumps({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2) + "\n"
    )
    write_report(out, summary)
    print(out)


def write_report(out, summary):
    front, wrist = summary["image_peaks"]
    block_table = "\n".join(
        f"| {b['label']} | {b['valid_tokens']} | {b['mass_percent']:.4f}% |" for b in summary["blocks"]
    )
    (out / "READ_ME_FIRST.md").write_text(f"""# 用一个例子读懂 attention

先打开 [交互式讲解 index.html](index.html)（下载后用浏览器直接打开即可，不需要启动服务），或按下面三张图顺序阅读。本次只处理实验 0 保存的数据，没有重新运行模型或机器人。

## 1. 固定我们正在看的那一次计算

来源：`{summary['case_id']}`。原始布局，语言“把牛奶放入篮子”，正确牛奶示范，episode 0，环境第 0 步，机器人尚未执行本轮预测动作。该案例按固定条件选取，没有根据热图效果筛选。

主图固定：**采样步 9、层 17；8 个 heads 求平均；前 5 个 action queries 求平均。** 层和采样步从零计数，因此是第 10 次采样更新、第 18 层。前 5 个位置对应本轮实际执行的动作范围；这里记录的是它们在最后一次更新内部的 attention，不是最终动作的重要性分解。每个 action token 对应一个动作时间位置，不是一个 x/y 坐标分量。

三个层次：

- **环境步**：机器人实际执行到哪里。本例为 0。
- **采样步**：在当前观测下，从噪声逐步得到动作序列的内部计算。一次预测有 10 次更新；图里的 0/5/9 不是机器人运动的时间点。
- **Transformer 层**：每次采样计算依次经过多层网络，图里的 0/9/17 是不同深度。一个 head 是同一层中的一组注意力计算；本图对 8 组取平均，可能掩盖某个 head 的特殊分布。

## 2. 从原图到热图

![原图、patch 注意力与叠加图](01_read_the_heatmap.png)

左列是实际送入模型的 224×224 图像（已经使用原评估的方向与尺寸，不再翻转）。中列是 16×16 个 patch 的注意力。每格对应输入图像的 14×14 像素；右列把中列叠在原图上，青色框标出该相机权重最大的 patch，**不是物体检测框**。没有平滑插值，以免让粗粒度格子看起来像精确分割。

主视角最亮格为第 **{front['row']} 行、第 {front['col']} 列**（从零计数），对应 x∈[{front['pixel_x_half_open'][0]}, {front['pixel_x_half_open'][1]})、y∈[{front['pixel_y_half_open'][0]}, {front['pixel_y_half_open'][1]}) 像素。它占全部 keys 的 **{front['patch_global_percent']:.4f}%**，占主视角图像自身注意力的 **{front['patch_within_camera_percent']:.2f}%**。这是同一个数用两个不同分母表达。

本图为了看清每个相机内部分布，**两个相机各用自己的颜色上限**，上限标在色条上。不能用两行的颜色亮度比较哪台相机权重更大。交互页面默认使用所有视图共享的绝对刻度，并提供“各相机放大颜色”的切换，可亲自观察同一份数据怎样因为配色不同而显得更亮。

特别注意：图像 token 在进入这些层前已经经过 SigLIP 等上下文计算，能混合其他位置的信息。因此它有空间位置索引，但不等于“只来自这块像素的信息”。本图是 action query → 当前图像位置 token 的直接 attention，**不是像素级因果归因，也不是跨层 attention rollout**。

## 3. 先看分配总量，再看局部热点

![各部分总量和具体语言 token](02_attention_budget.png)

| keys 部分 | 有效 token 数 | 对全部 keys 的权重 |
|---|---:|---:|
{block_table}

主视角总量 **{front['camera_mass_percent']:.3f}%**，手腕总量 **{wrist['camera_mass_percent']:.3f}%**。各项总和为 **{summary['total_percent']:.4f}%**，与 100% 的微小差异来自原 BF16 概率舍入；这里没有重新归一化。图中语言每个 token 的百分比也以全部 keys 为分母，没有把语言单独放大到 100%。`<bos>` 和换行符保留，padding 不展示且权重为零。

预测动作 token 的权重很大，是因为 action queries 也允许关注动作序列内部的 keys。不能据此断言它“主要依赖自身、不看图像”：这里还没有考虑 value 向量、残差、其他层和已有的信息融合。同样，语言的直接权重较低，也不能据此判断语言没有用。

各部分 token 数不同，求和权重并不是天然公平的模态重要性评分；[blocks.csv](blocks.csv) 同时保存 token 数和每个有效 token 的平均权重，供检查口径。

## 4. 层和采样步究竟改变了什么

![固定观测，分别改变层与采样步](03_layers_and_flow_steps.png)

上排固定采样步 9，只换层；下排固定层 17，只换采样步。六张图中的相机画面完全相同，机器人没有在它们之间移动。六张图共用一个颜色刻度，便于比较。它们展示同一动作生成过程的不同计算位置，不是机器人从左到右移动的轨迹；也不能预先假设越深就越聚焦目标。

## 5. 这些图的计算口径

原始概率维度为 `[采样步, 层, batch, head, query, key]`。主图计算：

```python
# 2 对应保存列表中的 step 9 / layer 17；query 0 是 state，因此排除。
a = probabilities[2, 2, 0, :, 1:6, :].astype(float).mean(axis=(0, 1))
front_heatmap = a[0:256].reshape(16, 16)
front_total = a[0:256].sum()
language_total = a[768:816].sum()
```

选定 head/query 的注意力行在全部合法 keys 上约等于 1。平均后依然如此，再取指定 keys 求和或按 patch 排列。没有对单张热图除以其最大值来重新定义概率，也没有跨层相乘。

## 本例能回答和不能回答的问题

能回答：在这一层、这一采样步、这一组 queries 和 heads 上，哪些 token 获得了多少直接注意力，以及它们在当前图像的位置。

不能回答：模型是否识别了牛奶、是否按位置而非物体行动、语言和示范谁控制了行为、遮住亮区是否一定改变动作。这些需要跨条件分析及后续干预，不能凭这一个示例下结论。本例也没有生成示范图像内部的空间热图。

数据：[summary.json](summary.json)、[averaged_attention.npz](averaged_attention.npz)、[provenance.json](provenance.json)。本次新增生成脚本 `scripts/attention/make_tutorial.py` 和网页模板 `scripts/attention/tutorial.html`，没有修改模型或原始实验数据。
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", default="original/correct/milk/ep000_step000")
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
