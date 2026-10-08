"""Pilot attention contact sheets using actual intervention inputs, not clean-image substitutes."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build(root):
    root = Path(root)
    dest = root / "attention_figures"
    dest.mkdir(exist_ok=True)
    sources = {}
    for path in root.glob("servers/*/attention/*/ep*_step*/capture.npz"):
        key = (path.parent.parent.name, path.parent.name)
        assert key not in sources, key
        sources[key] = path
    pages = []
    for ep in range(5):
        for demo in ("milk", "tomato_sauce"):
            for experiment in ("language", "vision"):
                conditions = (
                    [("language_" + l + "__normal", l) for l in ("milk", "tomato_sauce", "empty")]
                    if experiment == "language"
                    else [
                        ("language_milk__" + v, v)
                        for v in ("normal", "mask_target", "mask_other", "mask_background", "freeze_near")
                    ]
                )
                for camera, offset in (("front", 0), ("wrist", 256)):
                    fig, axes = plt.subplots(len(conditions), 3, figsize=(9, 3 * len(conditions)), squeeze=False)
                    masses = []
                    for row, (suffix, label) in enumerate(conditions):
                        condition = f"demo_{demo}__{suffix}"
                        for col, step in enumerate((0, 80, 160)):
                            path = sources[(condition, f"ep{ep:03d}_step{step:03d}")]
                            with np.load(path) as z:
                                weights = z["mean_attention"][-1, -1, offset : offset + 256].reshape(16, 16)
                                rgb = z[camera]
                            ax = axes[row, col]
                            ax.imshow(rgb)
                            # Smooth display only; exact patch weights remain in NPZ.
                            ax.imshow(
                                weights,
                                extent=(-0.5, 223.5, 223.5, -0.5),
                                alpha=0.48,
                                cmap="turbo",
                                vmin=0,
                                vmax=float(weights.max()),
                                interpolation="bilinear",
                            )
                            ax.set_title(f"{label} / t={step}\ncamera mass={weights.sum()*100:.2f}%", fontsize=9)
                            ax.axis("off")
                            masses.append({"condition": condition, "step": step, "mass": float(weights.sum())})
                    fig.suptitle(
                        f"{experiment} | demo={demo} | initial state {ep} | {camera}\nflow step 9, layer 17; 8 heads x first 5 action queries; color stretched separately per view",
                        fontsize=10,
                    )
                    fig.tight_layout(rect=(0, 0, 1, 0.96))
                    filename = f"{experiment}_{demo}_ep{ep:03d}_{camera}.png"
                    fig.savefig(dest / filename, dpi=110)
                    plt.close(fig)
                    pages.append(
                        {
                            "experiment": experiment,
                            "demo": demo,
                            "episode": ep,
                            "camera": camera,
                            "file": filename,
                            "masses": masses,
                        }
                    )
    (dest / "manifest.json").write_text(json.dumps(pages, indent=2))
    html = """<!doctype html><meta charset="utf-8"><title>实验3/4 注意力辅助图</title><style>body{font:16px system-ui;margin:24px}select{font:inherit;margin:8px}img{max-width:100%}</style><h1>注意力辅助图</h1><p>背景是模型实际接收的图像，包含遮挡或冻结。每格单独放大颜色；跨格绝对权重看 camera mass。不是物体识别概率，也不是因果贡献。</p><select id="experiment"><option value="language">实验3：语言</option><option value="vision">实验4：当前视觉</option></select><select id="demo"><option value="milk">牛奶示范</option><option value="tomato_sauce">番茄酱示范</option></select><select id="episode"></select><select id="camera"><option value="front">前视</option><option value="wrist">腕部</option></select><br><img id="figure"><script>const data=DATA;const keys=['experiment','demo','episode','camera'];const elements=keys.map(x=>document.getElementById(x));for(let i=0;i<5;i++)elements[2].add(new Option('初始状态 '+i,i));function update(){const row=data.find(r=>keys.every((k,i)=>String(r[k])===elements[i].value));document.getElementById('figure').src=row.file;}elements.forEach(e=>e.onchange=update);update();</script>""".replace(
        "DATA", json.dumps(pages)
    )
    (dest / "index.html").write_text(html)
    print("Wrote", len(pages), "attention contact sheets")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    build(p.parse_args().run)
