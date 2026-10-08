# ContextFlow 发布检查点评估

日期：2026-09-21。状态：已完成，四个任务各 50 次，共 200 次，运行异常为 0。

## 启动故障及修复

原服务在完成检查点恢复和数据加载后调用 `socket.gethostbyname(socket.gethostname())`，
用于打印日志。本机名称 `10-118-231-160` 无法通过 DNS / hosts 解析，因此抛出 `socket.gaierror`。
服务实际监听 `0.0.0.0`，并不需要解析主机名；现改为直接记录监听地址和端口。
未修改系统 DNS、hosts、VPN 或模型计算代码。

ROCm / TPU 的初始化提示不是这次退出原因；本次使用 NVIDIA CUDA。
此前的基础环境和轻量服务检查未覆盖真实服务的这个日志步骤，现已增加回归测试，
并用真实检查点完成服务启动和仿真交互。

评估脚本同时增加逐回合 JSONL 记录，运行异常会中止评估并报告错误，
避免把服务断开等程序错误当成模型任务失败；每个任务结束后关闭仿真环境。

## 对照设置

对照 [论文第 4.1 节、Table 1 和 Table S4](https://arxiv.org/html/2609.06852v1)：
Spatial 和 Object 各两个未见任务，每任务 50 次，共 200 次。
论文 Table 1 的四任务平均成功率为 73.5%。

- 检查点：`/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999`。
- 数据：`physical-intelligence/libero` v2.0，提交 `9dfa69510ea9e1613fc54112bc706444b686a231`。
- 模型配置：`ContextFlow`；权重与 JAX 矩阵乘法采用 float32。
- 仿真随机种子：仓库默认 `7`；初始状态索引为每任务的 `0..49`。
- 每个任务固定采用同任务的第一条演示，由现有 `InjectDemoFromCustomDataset` 选择并缓存。
  这是带演示的 in-context 评估，没有更新模型权重。
- 每条演示采样 8 帧图像、128 个状态和动作；模型动作窗口为 50，flow matching 为 10 步，
  每执行 5 个动作重新规划；图像输入为 224×224。
- Spatial 使用 GPU 0 / 端口 8765；Object 使用 GPU 1 / 端口 8766。
  两套服务独立初始化，各自的策略随机数种子从仓库默认 `0` 开始。

## 结果

| 任务 | 论文成功率 | 本次成功数 / 50 | 本次成功率 |
| --- | ---: | ---: | ---: |
| Spatial：将饼干盒上的黑碗放到盘子上 | 86% | 43 / 50 | 86% |
| Spatial：将盘子旁的黑碗放到盘子上 | 42% | 18 / 50 | 36% |
| Object：将牛奶放入篮子 | 76% | 43 / 50 | 86% |
| Object：将番茄酱放入篮子 | 90% | 47 / 50 | 94% |
| 四任务平均 | 73.5% | 151 / 200 | 75.5% |

Spatial 平均为 **61%**（论文 64%），Object 平均为 **90%**（论文 83%）。
四任务平均比论文高 2 个百分点，但“盘子旁的黑碗”低 6 个百分点，牛奶任务高 10 个百分点。
因此本轮复现支持发布检查点达到论文所报告的整体性能水平，各任务数值并非完全一致。

本轮检验发布检查点在指定任务上的表现；没有重新训练模型，也没有重跑基线模型。
论文未提供可核对的完全相同的推理随机状态，单次评估的数值差异需要结合这一限制解释。
本轮结果不能单独验证论文关于训练方法、基线优劣或多随机种子稳定性的全部结论。

已核对每个任务的 50 条初始状态索引完整且不重复、JSONL 与最终 JSON 的成功数一致，
并保存 200 个回放视频；两套服务日志均无推理或环境异常。完成后端口 8765 / 8766 已关闭，
GPU 0 / 1 显存占用均恢复为 0 MiB。启动故障回归测试 1 项、评估错误处理与记录测试 2 项通过。

## 记录位置

- 配置、源码补丁和控制日志：`logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/provenance/`。
- 汇总对照：`logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/provenance/comparison.json` 和 `comparison.csv`。
- Spatial：`logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_spatial/`。
- Object：`logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_object/`。
- 各组目录内：`server.log`、`libero_<suite>.log`、最终 `.json`、逐回合 `.episodes.jsonl`，
  以及 `videos/libero_<suite>/<任务指令>/` 中的回放视频，每个指令文件夹 50 个。

使用 `bash quickstart.sh` 可再次启动；依次输入 **6 → 5（Spatial + Object）→ 50**，
然后在检查点、GPU、端口提示处各按一次回车使用默认设置。菜单会自动为两个任务集分别启动新服务，
每个任务集的策略随机数从 0 开始，与本轮设置一致。新运行会保存到单独的带时间戳目录，不覆盖本轮结果。

## 2026-09-22：视频与目录整理

原脚本以 10 FPS 保存每个控制步的图像，而 LIBERO 的控制频率为 20 Hz，因此回放为半速。
已将本轮 200 个视频的 MP4 时间尺度调整为 **20 FPS**，没有重新编码；逐个校验编码画面数据哈希
和帧数均保持一致，视频时长减半。后续评估也会直接按 20 FPS 导出，表示仿真原速。
成功率和评估 JSON 未修改。原始 10 FPS 视频保存在运行目录的
`provenance/original_10fps_videos.tar.gz`，逐文件校验见 `provenance/video_timing_update.json`。

原来两组含随机后缀的目录已整理到
`logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/`，
内部的 `libero_spatial`、`libero_object` 与各自结果 JSON 名称对应；报告链接已同步更新。
视频目录再按任务指令分层（英文指令的空格替换为下划线），共 4 个指令文件夹，每个 50 个视频。

## 回放示例

下表各取一个成功和失败回合；全部视频在各任务集的 `videos/` 目录。

| 任务 | 成功回合 | 失败回合 |
| --- | --- | --- |
| pick up the black bowl on the cookie box and place it on the plate | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_spatial/videos/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate/rollout_pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_ep000_success.mp4) | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_spatial/videos/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate/rollout_pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_ep014_failure.mp4) |
| pick up the black bowl next to the plate and place it on the plate | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_spatial/videos/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate/rollout_pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_ep003_success.mp4) | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_spatial/videos/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate/rollout_pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_ep000_failure.mp4) |
| pick up the milk and place it in the basket | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_object/videos/libero_object/pick_up_the_milk_and_place_it_in_the_basket/rollout_pick_up_the_milk_and_place_it_in_the_basket_ep000_success.mp4) | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_object/videos/libero_object/pick_up_the_milk_and_place_it_in_the_basket/rollout_pick_up_the_milk_and_place_it_in_the_basket_ep001_failure.mp4) |
| pick up the tomato sauce and place it in the basket | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_object/videos/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket/rollout_pick_up_the_tomato_sauce_and_place_it_in_the_basket_ep000_success.mp4) | [视频](logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/libero_object/videos/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket/rollout_pick_up_the_tomato_sauce_and_place_it_in_the_basket_ep026_failure.mp4) |

## 2026-09-22：牛奶 / 番茄酱示范干预实验

已完成 correct / no / wrong 六组各 25 回合，共 150 回合。
示范明显改变行为，但双向物体类别跟随不成立；其中一个方向稳定转向了示范位置上的奶油奶酪。
详见 [实验报告](logs/quickstart/context_ablation_object_correct_no_wrong_25trials_20260922_154452/REPORT.md)。

## 2026-09-22：固定场景的位置交换控制实验

已完成 12 组各 25 回合，共 300 回合。固定番茄酱场景和原始示范，仅交换牛奶与番茄酱位置后，结果更支持示范的空间／动作信息驱动；四个有示范条件在交换后，示范原物体入篮均为 0/25。
详见 [实验报告](logs/quickstart/position_swap_object_25trials_20260922_175506/REPORT.md)。

## 2026-09-23：no-context 保留位置编号的补充对照

**补充结果：新 no 的两个任务仍均为牛奶 0/25、番茄酱 0/25。零成功率在保留 mask 和位置编号的置零干预中也成立，不能只用旧版位置编号变化解释。**

原 Correct / No / Wrong 实验的 no 条件在各自原场景各补跑 25 次。correct/wrong 沿用原记录，旧 masked no 保留作对照。详见[补充报告](logs/context_ablation/no_context_zero_keep_mask_25trials_20260923_204651/EXPERIMENT_RESULTS.md)。固定场景的位置交换 no 条件没有在本次重跑，不能混用这些数值。
