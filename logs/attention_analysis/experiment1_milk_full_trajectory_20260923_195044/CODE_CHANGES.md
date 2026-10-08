# 实验 1：完整轨迹 attention 的代码改动

本次只增加实验 1 的回放、补采和离线可视化功能，不修改模型计算或实验 2 的示范 action 消融。

| 文件 | 改动 |
|---|---|
| `examples/libero/prepare_attention_timeline.py` | 新增完整回放入口。固定牛奶/原始布局/三种 context/初始状态 0–4，导出 281 帧双视角状态和 56 个真实规划观测，并核对每步机器人/物体状态和入篮判据。 |
| `scripts/check_attention_capture.py` | 将初检固定的 12 个条件改为从观测清单计算条件数。这次为 3，原实验 0 的清单仍为 12。普通/记录推理、噪声、历史动作、参数哈希的验收要求未放宽。 |
| `scripts/attention/build_timeline.py` | 新增播放器数据生成器。每个初始状态生成独立页面，保留 float64 平均数据；HTML 传输 float32 数据和 JPEG 显示帧。可边补采边生成已完成的页面。 |
| `scripts/attention/trajectory_timeline.html` | 新增全轨迹拖动条、原速播放、逐控制帧/逐规划点移动、入篮事件跳转、相机总量曲线和 keys 分配表。当前帧与最近规划热图分别显示，杜绝错帧叠加。 |
| `scripts/attention/README.md` | 补充完整回放、补采和页面生成命令，说明依赖与时间对齐。 |

模型推理从本次开始时复制的 `provenance/frozen/src/openpi` 导入，并验证实际 import 路径。原有检查点不变，使用 GPU 7。这样其他 agent 在工作区修改实验 2 时不会改变本次推理代码。

完整轨迹源于此前的位置交换实验的 original 分支。这里 replay 旧动作、重建当时观测并验证原预测，没有重新执行一套新的行为实验。

页面只在 0、5……275 的真实模型调用时刻有 attention。其他控制帧显示当前画面，但热图连同其背景留在最近规划帧，界面同时标明这两个时间。最终控制步 280 仍显示最后一次规划 275 的热图。

数据与页面位于本地 logs 目录，没有提交或 push。详细验收和结果见 `viewer/REPORT.md`、`viewer/VALIDATION.md`。
