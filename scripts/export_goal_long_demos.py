"""Export the exact demonstrations selected by a completed Goal/Long run."""
# ruff: noqa: RUF001

import argparse
import hashlib
import html
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from openpi_client import image_tools
from PIL import Image
from PIL import ImageDraw
import pyarrow.parquet as pq


def export(root, dataset):
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = info["fps"]
    tasks = json.loads((root / "task_manifest.json").read_text())["tasks"]
    order = [1, 5, 4]
    tasks.sort(
        key=lambda t: (
            order.index(t["dataset_task_index"]) if t["dataset_task_index"] in order else 3,
            t["dataset_task_index"],
        )
    )
    out = root / "demonstrations"
    out.mkdir(exist_ok=True)
    records = []
    for task in tasks:
        tid, ep = task["dataset_task_index"], task["demo_episode"]
        source = dataset / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        table = pq.read_table(source)
        n = len(table)
        assert n == task["demo_length"]
        assert set(table["episode_index"].to_pylist()) == {ep}
        assert set(table["task_index"].to_pylist()) == {tid}
        assert table["frame_index"].to_pylist() == list(range(n))
        np.testing.assert_allclose(np.diff(table["timestamp"].to_numpy()), 1 / fps, atol=1e-5)
        snapshots = list((root / "servers").glob(f"gpu*/task{tid:02d}_inputs/demo_and_observation.npz"))
        assert len(snapshots) == 1
        directory = out / f"task{tid:02d}_{task['instruction'].replace(' ', '_')}"
        directory.mkdir(exist_ok=True)
        cameras = {}
        for column in ("image", "wrist_image"):
            cameras[column] = np.stack(
                [np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB")) for row in table[column].to_pylist()]
            )
        with np.load(snapshots[0]) as snap:
            sheet = Image.new("RGB", (8 * 224, 480), "white")
            draw = ImageDraw.Draw(sheet)
            for row, (column, key) in enumerate(
                (("image", "demo_base_0_rgb"), ("wrist_image", "demo_left_wrist_0_rgb"))
            ):
                resized = image_tools.resize_with_pad(cameras[column][task["demo_image_frame_indices"]], 224, 224)
                actual = snap[key][0]
                np.testing.assert_allclose(resized.astype(np.float32) / 255 * 2 - 1, actual, atol=1e-6)
                pixels = np.rint((actual + 1) * 127.5).clip(0, 255).astype(np.uint8)
                for i, frame in enumerate(pixels):
                    x, y = i * 224, row * 240
                    sheet.paste(Image.fromarray(frame), (x, y + 16))
                    draw.text((x + 3, y + 2), f"{column} frame {task['demo_image_frame_indices'][i]}", fill="black")
            sheet.save(directory / "model_input_8frames.png")
        joined = np.concatenate((cameras["image"], cameras["wrist_image"]), axis=2)
        video_info = []
        for rate, label in ((fps, "dataset_timestamps"), (20, "control_frame_comparison")):
            name = f"demo_episode{ep:06d}_main_wrist_{rate}fps_{label}.mp4"
            path = directory / name
            imageio.mimwrite(path, joined, fps=rate)
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
            decoder = imageio_ffmpeg.read_frames(str(path))
            meta = next(decoder)
            decoder.close()
            assert frames == n
            assert meta["fps"] == rate
            video_info.append({"path": str(path.relative_to(out)), "fps": rate, "frames": frames, "seconds": seconds})
        rec = {
            "task_index": tid,
            "instruction": task["instruction"],
            "split": task["split"],
            "demo_episode": ep,
            "frames": n,
            "source_parquet": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sampled_images_match_captured_model_input": True,
            "sample_indices": task["demo_image_frame_indices"],
            "model_input_png": str((directory / "model_input_8frames.png").relative_to(out)),
            "videos": video_info,
            "rollout_episode0": "../"
            + json.loads((root / "tasks" / task["task_key"] / "episodes.jsonl").read_text().splitlines()[0])["video"],
        }
        records.append(rec)
        print(f"Exported task {tid}, demo {ep}, {n} frames; captured inputs verified", flush=True)
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = [
        "# 实际示范视频",
        "",
        "这些是本轮评估实际选择的 episode，不是另选的成功示例。完整视频供人对照；模型的示范图像只有均匀采样的 8 帧，另有 128 个 state/action 槽位。采样图已与服务保存的实际输入逐像素核对（归一化浮点误差容差 1e-6）。",
        "",
        "左为主相机，右为手腕相机。10 FPS 对应数据集元信息及时间戳；20 FPS 版本按评估视频的帧播放速度展示，不代表已证明原始采集时间尺度相同。两个版本帧内容完全相同，没有删帧或插帧。",
        "",
        "[打开交互对照页](index.html)：默认播放 20 FPS 版本，可切换数据集 10 FPS；右侧为同任务评估 ep000。两条轨迹独立播放，不做阶段对齐。",
        "",
        "| ID | 任务 | 划分 | 示范 | 完整视频 | 模型采样图 |",
        "|---:|---|---|---:|---|---|",
    ]
    cards = []
    for rec in records:
        links = " / ".join(f"[{v['fps']} FPS]({v['path']})" for v in rec["videos"])
        lines.append(
            f"| {rec['task_index']} | {rec['instruction']} | {rec['split']} | ep{rec['demo_episode']} | {links} | [8 帧]({rec['model_input_png']}) |"
        )
        esc = html.escape
        choices = "".join(
            f'<option value="{esc(v["path"])}" {"selected" if v["fps"] == 20 else ""}>{v["fps"]} FPS</option>'
            for v in rec["videos"]
        )
        cards.append(f"""<section id="task{rec['task_index']}"><h2>Task {rec['task_index']} · {esc(rec['split'])} · {esc(rec['instruction'])}</h2>
<p>实际示范 episode {rec['demo_episode']} · {rec['frames']} 帧 · 采样帧与模型输入核对通过</p><div class="pair"><div><h3>完整示范（左主相机／右手腕）</h3><select onchange="let v=this.parentElement.querySelector('video');v.src=this.value;v.load()">{choices}</select><video controls preload="none" src="{esc(rec['videos'][1]['path'])}"></video></div><div><h3>评估 ep000 · 20 FPS · 主相机</h3><video controls preload="none" src="{esc(rec['rollout_episode0'])}"></video></div></div>
<details><summary>模型实际看到的 8 帧（上主相机／下手腕）</summary><a href="{esc(rec['model_input_png'])}"><img loading="lazy" src="{esc(rec['model_input_png'])}"></a></details></section>""")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    nav = " · ".join(f'<a href="#task{r["task_index"]}">Task {r["task_index"]}</a>' for r in records)
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Goal / Long 实际示范与评估对照</title><style>body{font:16px system-ui;margin:24px auto;max-width:1450px;padding:0 20px;color:#202734;background:#f6f7fa}section{background:white;padding:20px;margin:24px 0;border-radius:12px}h2{font-size:20px}.pair{display:grid;grid-template-columns:2fr 1fr;gap:20px;align-items:end}video,img{width:100%;display:block}select{margin:8px 0}summary{cursor:pointer;margin:18px 0}a{color:#225dab}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>实际示范与评估对照</h1><p>这是本轮实际使用的示范。完整视频供人观看；模型的示范图像只取 8 帧，另有 128 个 state/action 槽位。</p><p>默认 20 FPS 便于按评估视频的帧速度观看；可切换为数据集时间戳对应的 10 FPS。两个版本不删帧、不插帧；不把数据集与模拟器的时间尺度假定为相同。左右两条轨迹独立播放，没有阶段对齐。Task 4 是含 alphabet soup 的 seen 对照。</p>"""
    (out / "index.html").write_text(page + "<nav>" + nav + "</nav>" + "".join(cards) + "</html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("/data/zjc/workspace/datasets/physical-intelligence/libero")
    )
    args = parser.parse_args()
    export(args.root, args.dataset)
