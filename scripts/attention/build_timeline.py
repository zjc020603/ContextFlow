# ruff: noqa: RUF001
"""Offline full-control-trajectory player with attention at real planning calls."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODES = ("correct", "no", "wrong")


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def jpeg(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="JPEG", quality=85)
    return base64.b64encode(stream.getvalue()).decode()


def wait_for(predicate, watch, message):
    deadline = time.monotonic() + 5400
    while not predicate():
        if not watch:
            raise FileNotFoundError(message)
        if time.monotonic() > deadline:
            raise TimeoutError(message)
        time.sleep(5)


def build_episode(root, out, manifest, episode):
    payload = {
        "episode": episode,
        "modes": MODES,
        "steps": [0, 5, 9],
        "layers": [0, 9, 17],
        "horizon": 280,
        "stride": 5,
        "fps": 20,
        "rollouts": {},
        "lut": (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist(),
    }
    summaries = []
    source_hashes = {}
    for mode in MODES:
        item = next(r for r in manifest["rollouts"] if r["episode"] == episode and r["mode"] == mode)
        frames_path = root / "observations" / item["frames"]
        with np.load(frames_path) as f:
            front, wrist = f["front"], f["wrist"]
            assert front.shape == wrist.shape == (281, 224, 224, 3)
            state = f["state"].copy()
            inside = f["in_basket"].copy()
            names = f["object_names"].tolist()
        arrays = []
        layouts = []
        for step in range(0, 280, 5):
            directory = root / "capture" / f"original/{mode}/milk/ep{episode:03d}_step{step:03d}"
            capture_path = directory / "capture.npz"
            with np.load(capture_path) as c:
                assert c["ordinary_actions"].tobytes() == c["captured_actions"].tobytes()
                assert c["ordinary_executable_actions"].tobytes() == c["captured_executable_actions"].tobytes()
                raw = c["probabilities"].astype(np.float64)
                assert np.isfinite(raw).all()
                assert raw.shape == (3, 3, 1, 8, 51, 1027)
                assert np.all(raw[..., np.flatnonzero(~c["prefix_mask"][0])] == 0)
                avg = raw[:, :, 0, :, 1:6].mean((2, 3))
                if mode == "no":
                    assert np.all(avg[..., 816:976] == 0)
                arrays.append(avg)
            layout = json.loads((directory / "layout.json").read_text())
            assert layout["context"]["selected_episode"] == {"correct": [814], "no": [], "wrong": [821]}[mode]
            layouts.append(layout)
            source_hashes[str(capture_path.relative_to(root))] = hashlib.sha256(capture_path.read_bytes()).hexdigest()
        averages = np.stack(arrays)
        assert averages.shape == (56, 3, 3, 1027)
        assert np.allclose(averages.sum(-1), 1, rtol=0, atol=0.004)
        # Keep float64 analysis data; browser uses float32 for transport only.
        np.savez_compressed(out / f"ep{episode:03d}_{mode}_averages.npz", attention=averages)
        browser = averages.astype(np.float32)
        np.testing.assert_allclose(browser, averages, atol=3e-8, rtol=1e-7)
        events = {
            name: (int(np.flatnonzero(inside[:, i])[0]) if inside[:, i].any() else None) for i, name in enumerate(names)
        }
        tokens = [
            s.replace("▁", " ").replace("\n", "<newline>").strip() or "<space>"
            for s in layouts[0]["language_token_pieces"]
        ]
        payload["rollouts"][mode] = {
            "images": {"front": [jpeg(im) for im in front], "wrist": [jpeg(im) for im in wrist]},
            "attention_b64": b64(browser),
            "shape": list(browser.shape),
            "state": state.tolist(),
            "inside": inside.tolist(),
            "object_names": names,
            "first_in_basket": events,
            "blocks": layouts[0]["blocks"],
            "tokens": tokens,
            "source_state_error": item["replay_max_error"],
        }
        totals = {
            name: (averages[:, :, :, a:b].sum(-1) * 100).tolist()
            for name, a, b in [
                ("front", 0, 256),
                ("wrist", 256, 512),
                ("language", 768, 816),
                ("demo_actions", 944, 976),
                ("action_sequence", 977, 1027),
            ]
        }
        summaries.append(
            {
                "episode": episode,
                "mode": mode,
                "first_in_basket": events,
                "mass_percent": totals,
                "max_browser_rounding_error": float(np.max(np.abs(browser - averages))),
            }
        )
        if episode == 0:
            for camera, images in [("front", front), ("wrist", wrist)]:
                # Used below for a stable contact sheet, not inference.
                for step in (0, 80, 160, 275):
                    Image.fromarray(images[step]).save(out / f"preview_{mode}_{camera}_{step:03d}.png")
        print(f"Encoded episode {episode}/{mode}: 281 frames + 56 attention calls", flush=True)
    template = Path(__file__).with_name("trajectory_timeline.html")
    filename = "index.html" if episode == 0 else f"episode_{episode:03d}.html"
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (out / filename).write_text(template.read_text().replace("__PAYLOAD__", text))
    (out / f"episode_{episode:03d}_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (out / f"episode_{episode:03d}_sources.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    if episode == 0:
        contact_sheet(out, payload)
    return summaries


def contact_sheet(out, payload):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    for camera, start in [("front", 0), ("wrist", 256)]:
        fig, axs = plt.subplots(3, 4, figsize=(12, 9.5), constrained_layout=True)
        for i, mode in enumerate(MODES):
            with np.load(out / f"ep000_{mode}_averages.npz") as f:
                a = f["attention"]
            for j, step in enumerate((0, 80, 160, 275)):
                heat = a[step // 5, 2, 2, start : start + 256].reshape(16, 16) * 100
                image = np.asarray(Image.open(out / f"preview_{mode}_{camera}_{step:03d}.png"))
                ax = axs[i, j]
                ax.imshow(image, extent=(0, 16, 16, 0))
                m = ax.imshow(
                    heat,
                    extent=(0, 16, 16, 0),
                    vmin=0,
                    vmax=heat.max(),
                    cmap="magma",
                    alpha=0.5,
                    interpolation="nearest",
                )
                ax.set_title(f"{mode} / env {step}\nCamera total {heat.sum():.2f}%")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(m, ax=ax, shrink=0.7, label="% / patch")
        fig.suptitle(
            f"Episode 0 / {camera} | layer 17, flow 9 | mean of 8 heads / first 5 actions\n"
            "Different contexts follow different actual trajectories. Individual panel color ranges.",
            fontsize=12,
        )
        fig.savefig(out / f"keyframes_{camera}.png", dpi=145)
        plt.close(fig)


def report(root, out, summaries):
    rows = []
    for ep in range(5):
        grouped = [next(s for s in summaries if s["episode"] == ep and s["mode"] == m) for m in MODES]
        rows.append(
            "| "
            + str(ep)
            + " | "
            + " | ".join(
                str(s["first_in_basket"]["milk_1"]) if s["first_in_basket"]["milk_1"] is not None else "未入篮"
                for s in grouped
            )
            + " | "
            + str(grouped[2]["first_in_basket"]["tomato_sauce_1"])
            + " |"
        )
    results = json.loads((root / "capture/results.json").read_text())
    text = """# 实验 1：牛奶任务的整条轨迹 attention 播放器

打开 [交互页面](index.html)，拖动控制步 0–280。页面自包含图片和数据，下载单个 HTML 后即可离线使用；切换其他初始状态需要同目录的其余 HTML 文件。

## 原来保存了什么，这次补了什么

原实验保存了全部 280 个控制动作、逐步机器人/物体状态和主视角视频。此前 attention 只采集了环境步 0、80、160，并不代表其他轨迹丢失。

本次对牛奶语言、原始布局、correct/no/wrong、初始状态 0–4 共 **15 条原轨迹**进行完整确定性回放。每条导出 **281 个观测状态**（初始 0，加执行 280 次动作后的状态），两台相机都可逐帧查看。每 5 步模型才重新规划，所以真实模型调用时刻为 **0、5、10……275，共 56 次**；共补采 **840 次** attention。不是在每个控制步强行增加一次模型调用。

模型每次生成 50 个动作，原实验只执行前 5 个再规划。因此页面继续平均前 5 个 action queries 和 8 个 heads，可切换保存的层 0/9/17、采样步 0/5/9。

## 拖动条怎样与 attention 对齐

每个 context、每台相机并排显示两张图：

- **当前状态图**：严格对应拖动条控制步 t。
- **最近规划的热图**：对应 u = 5 × floor(t/5)，最高为 275；背景也是 u 时刻的图像，绝不把旧热图叠到 t 时刻的新画面上。

例如拖到 83，左图是环境步 83，右图是环境步 80 那次规划的 attention；81–84 没有新的模型调用。到 280 时轨迹结束，右图明确标为最后一次规划 275。按钮可逐控制帧或逐规划点移动，也可按原速 20Hz 播放。机器性能不足时显示可能跳帧；拖动仍可定位所有已保存状态。

页面同时显示机器人末端状态、两种目标物体当前/曾经入篮状态，并可跳到首次入篮事件。曲线展示每次真实规划时，两台相机分别分到的 attention 总量。

## 配对与复现检查

- 每条回放核对原始初始模拟器状态、初始画面 hash、全部 280 步的机器人/全部物体位置及入篮判据。
- 每次补采以完全相同的预处理输入和噪声分别运行普通与记录推理，比较完整动作，并比较当时实际执行的后续 5 个历史动作。
- 原始 raw attention 及完整动作保存在 `../capture/`；本页面使用其 head/query 平均结果。
- 页面以 float32 二进制打包权重，分析用 float64 平均文件仍保留。JPEG 图像仅用于显示，不用于补采推理；推理观测使用无损 NPZ。
- 使用本次开始时冻结的 `openpi` 代码副本，并保留 SHA256，避免另一个 agent 的实验 2 改动混入。

实际验收结果：
"""
    text += f"\n- {results['pairs']}/840 次检查完成；原始动作最大差异 {results['max_raw_action_difference']}，可执行动作差异 {results['max_executable_action_difference']}，历史动作差异 {results['max_historical_action_difference']}。\n- 参数前后 SHA256 相同：`{results['parameter_sha256_after']}`。\n"
    text += (
        """
## 必须结合实际轨迹看图

初始状态 0 仍作为默认，与此前教程一致；但它的 correct 轨迹最终没有牛奶入篮，不能把它当作所有 correct 的代表。各条首次入篮控制步如下（步骤从初始状态 0 开始，执行一个动作后为 1）：

| 初始状态 | correct 牛奶入篮 | no 牛奶入篮 | wrong 牛奶入篮 | wrong 番茄酱入篮 |
|---|---|---|---|---|
"""
        + "\n".join(rows)
        + """

想看成功的牛奶执行过程，可切换初始状态 1–4；保留初始状态 0 是为了避免只展示成功样例。入篮判据说明的是原 rollout 的实际行为，不由热图推断。

## 怎样判读后续热图

![主视角关键帧](keyframes_front.png)

![手腕关键帧](keyframes_wrist.png)

这里展示 0、80、160、275 四个规划时刻，完整 56 个时刻见页面。每格分别放大颜色，绝对大小要读相机总量及色条；相同颜色不代表相同权重。

环境步增加后，correct/no/wrong 的机器人位置、物体状态和画面都会不同。因此后续热图回答“沿各自真实行为过程如何分配注意力”，不能把跨 context 的同坐标差值当作固定输入下的示范效应，也不计算这种容易误导的像素差值图。

no-context 仍会屏蔽示范 keys，并改变 suffix 的位置编码；不同 context 的后续采样动作候选也不同。注意力变化不是孤立的示范语义贡献。热点落在物体附近也不等于已证明抓取由该热点决定。

本阶段提供完整的时间观察窗口，保留同样的平均口径；没有物体区域分割、head 筛选或新的 mask/action 消融。实验 2 由其他 agent 执行，本次没有修改其代码。

## 文件

- `index.html`：初始状态 0；`episode_001.html` 至 `episode_004.html`：其他状态。
- `ep###_<context>_averages.npz`：56 个规划点 × 3 采样步 × 3 层 × 1027 keys。
- `episode_###_summary.json`：每次规划的模态总量和入篮事件；`episode_###_sources.json`：原始 capture 哈希。
- `../observations/frames/`：完整双视角无损帧、状态、入篮判据；`../observations/manifest.json`：观测来源。
- `../capture/results.json`、`../capture/pairs.jsonl`：全部推理验收结果。
"""
    )
    spatial_rows = []
    for episode in range(5):
        for mode in MODES:
            with np.load(out / f"ep{episode:03d}_{mode}_averages.npz") as saved:
                wrist = saved["attention"][:, 2, 2, 256:512].reshape(56, 16, 16)
            share = wrist[:, 12:, :].sum(axis=(1, 2)) / wrist.sum(axis=(1, 2))
            spatial_rows.append(
                {
                    "episode": episode,
                    "mode": mode,
                    "bottom_quarter_median_percent": float(np.median(share) * 100),
                    "minimum_percent": float(share.min() * 100),
                    "maximum_percent": float(share.max() * 100),
                }
            )
    (out / "temporal_spatial_summary.json").write_text(json.dumps(spatial_rows, indent=2) + "\n")
    table = []
    for episode in range(5):
        values = [
            next(
                r["bottom_quarter_median_percent"]
                for r in spatial_rows
                if r["episode"] == episode and r["mode"] == mode
            )
            for mode in MODES
        ]
        table.append("| " + str(episode) + " | " + " | ".join(f"{v:.1f}%" for v in values) + " |")
    text += (
        """
## 把时间拉长后，当前能看到什么

默认层 17、采样步 9、8 heads / 前 5 个动作位置平均的主例关键帧中，correct 与 wrong 的手腕画面和所操作物体已不同，但较强的下沿带状热点仍反复出现。它并没有给出清晰、可直接标注为“从牛奶转向番茄酱”的目标转移证据。

为描述这个可见现象，固定统计手腕图像最后 4 行 patch（16×16 网格中的第 12–15 行，即画面下 1/4）获得的**相机内部**注意力份额。下面是每条轨迹 56 个规划时刻的中位数：

| 初始状态 | correct | no | wrong |
|---|---:|---:|---:|
"""
        + "\n".join(table)
        + """

这项区域统计是在观察图后做的描述性检查，不是预先注册的假设检验。区域只由像素坐标定义，不代表物体或夹爪分割；中位数也不意味着每个时刻都相同。具体范围见 `temporal_spatial_summary.json`。

因此，拉长观察确实揭示了真实行为分化以及 attention 随时间变化，但当前这种平均热图仍不足以直接解释“为什么 wrong 示范改变了抓取对象”。这不是“context 不起作用”的证据：它既没有计算 value/残差贡献，也会把单个 head 的变化平均掉。后续干预实验不必等到热图出现符合直觉的目标转移才成立。
"""
    )
    (out / "REPORT.md").write_text(text)


def main(args):
    root = args.run.resolve()
    out = root / "viewer"
    out.mkdir(exist_ok=True)
    wait_for(lambda: (root / "observations/manifest.json").exists(), args.watch, "Waiting for replay")
    manifest = json.loads((root / "observations/manifest.json").read_text())
    summaries = []
    for ep in range(5):

        def ready(ep=ep):
            p = root / "capture/pairs.jsonl"
            if not p.exists():
                return False
            lines = p.read_text().splitlines()
            try:
                records = [json.loads(s) for s in lines]
            except json.JSONDecodeError:
                return False
            cases = [r for r in records if r["episode"] == ep]
            if len(cases) != 168:
                return False
            assert all(r["raw_actions_exact"] and r["historical_executed_action_max_abs_diff"] <= 1e-6 for r in cases)
            return True

        wait_for(ready, args.watch, f"Waiting for episode {ep} captures")
        summaries.extend(build_episode(root, out, manifest, ep))
        print(f"PAGE READY: episode {ep}", flush=True)
    wait_for(lambda: (root / "capture/results.json").exists(), args.watch, "Waiting for final parameter check")
    report(root, out, summaries)
    (out / "build_complete.json").write_text(
        json.dumps({"episodes": 5, "trajectories": 15, "frames": 4215, "attention_calls": 840}) + "\n"
    )
    print("ALL PAGES COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    main(parser.parse_args())
