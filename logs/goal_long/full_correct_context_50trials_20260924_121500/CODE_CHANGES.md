# Goal / Long 评估：代码修改报告

本分支独立评估完整正确示范，不延续 action 消融、no-context 或 attention 实验。原模型、训练配置、mask、采样和归一化不改；新增评估与审计入口。原工作区修改保存在 `provenance/preexisting.patch`。

## 文件与职责

| 文件 | 新增部分与目的 |
|---|---|
| [goal_long_eval.py](/data/zjc/workspace/ContextFlow/examples/libero/goal_long_eval.py) | `make_manifest` 从 LIBERO 和本地数据构建 20 任务清单、官方初始状态 hash、BDDL 目标和默认示范，核对 seen/unseen 排除列表；`evaluate` 复用原环境和图像/动作处理，记录真实仿真与谓词，不改环境；`goal_metrics` 严格区分同时满足所有条件与各条件分别曾满足。 |
| [goal_long_eval_test.py](/data/zjc/workspace/ContextFlow/examples/libero/goal_long_eval_test.py) | 两项测试覆盖单目标成功、并列条件同时成立、不同时间分别成立不能计整任务成功、空记录拒绝。 |
| [serve_goal_long.py](/data/zjc/workspace/ContextFlow/scripts/serve_goal_long.py) | 用普通配置恢复同一 checkpoint；`gate` 在保存的完整正确示范样本上逐字节核对 raw/executable actions；任务开始时重置随机 key，推理仍调用原 `prepare_inputs`、`sample_actions`、输出变换；附带实际示范和 RNG 元信息；初次输入检查全部有效示范 token 未置零/未屏蔽；结束检查模型参数 hash。 |
| [run_goal_long.py](/data/zjc/workspace/ContextFlow/scripts/run_goal_long.py) | 首先并行运行 4 个 unseen 任务，全部完成后才运行其余 16 个 seen 任务。每任务单独客户端、随机流、结果目录；默认至多 8 个逻辑服务，通常每个 GPU 对应独立服务和仿真进程，限制 CPU 亲和性避免加载时线程互相挤占。只停止本次自己启动或明确附接的服务。 |
| [summarize_goal_long.py](/data/zjc/workspace/ContextFlow/scripts/summarize_goal_long.py) | 核对 1,000 回合动作和逐步目标谓词、首个成功即停止/失败跑满时限、真实示范 episode 与连续 RNG、所有服务参数 hash；生成每任务 CSV、seen/unseen 汇总、子目标表、成功率图、视频入口和结果报告。 |
| [GOAL_LONG_EVALUATION.md](/data/zjc/workspace/ContextFlow/scripts/GOAL_LONG_EVALUATION.md) | 本地再运行命令、资源分配、依赖的历史一致性检查数据和评估定义。 |

## 明确的设计决定

1. **完整正确 context**：三种示范内容均保留，未使用旧 no-context mask，也未调用 zero-demo 模型。标准模型有 160 个示范槽位，其中缺失右相机 32 个原本无效，其余 128 个有效。
2. **训练范围按真实配置核对**：调用原 `get_kept_episode_indices` 得到 1,350 个训练 episode、32 个任务。Goal/Long 各 8 个 seen 的本地 episode 数与该函数结果吻合，四个 held-out 任务训练 episode 数均为 0。见 `training_split_audit.json`。这不是从 checkpoint 权重反推作者训练日志。
3. **沿用默认示范**：Goal unseen task 10 / 17 对应 episode 379 / 392，Long unseen task 1 / 5 对应 episode 1 / 8。采样覆盖完整轨迹，8 帧图像、128 个 state/action 槽位。103/112 帧的 Goal 演示按原代码重复最后数据补 25/16 个位置；原 mask 不改，详见 `sampling_audit.json`。
4. **每任务独立随机流**：环境种子 7，策略 key(0)，随后连续 split。这样可并行而不随任务调度次序改变结果。不同于此前 per-suite 连续策略流，不声称与作者未公开的推理随机状态逐帧一致。
5. **完整成功才提前停止**：原 300/520 步上限；部分目标状态只作为诊断。谓词数量不等于物理动作基元数量，初始满足不算模型新完成操作。保留 initial/ever/final 统计。
6. **原速和审计数据**：20 FPS，指令目录。每步保留动作、仿真状态、末端/夹爪和原任务谓词；每次重规划记录选中示范与 RNG。额外数据采集没有更改动作。此次没有采集或分析 attention。

## 验证

- 目标统计测试：2 passed，见 `metric_tests.log`。
- 每个服务必须先通过历史完整模型输出的逐字节检查才监听；服务 `GATE_PASSED.json`、首任务输入与最终 `FINISHED.json` 可复核。
- 完整 rollout 审核与统计见结果报告、`comparison.json`；源文件快照和哈希保存在 `provenance/source/`。
- 代码相对于原标准评估器保留同一图像旋转/缩放、状态构造、动作窗口与执行频率；新增严格拒绝异常动作、检查完整目标与环境 done 一致。运行错误不会悄悄计为失败。

## 边界

代码以独立文件实现，未改 `contextflow.py`、`gemma.py`、`transforms.py`、`custom_dataset.py`、原 serve_policy、quickstart 或旧报告。此次结果和代码报告放在独立 `logs/goal_long/` 目录，未提交或推送 GitHub。

## 启动故障与恢复

逻辑 worker 5 在物理 GPU 5 上两次卡在 JAX 参数转 CPU 的 hash 检查，均未开始任何评估回合。第二次加入 `faulthandler`、SIGUSR1 和启动阶段日志，堆栈确认停在数组 materialization；这不足以诊断底层硬件原因。没有跳过参数检查或一致性 gate。迁移到物理 GPU 0 后，两项检查正常通过，仍执行原定种子和任务。

`goal_long_eval.evaluate` 新增可选 `resource_overrides.json`（端口到物理 GPU 映射），在建环境前设置 CUDA/EGL 设备，并记录到任务 config。本次 `8135 → 0`，其余客户端保持原映射。迁移期间仅暂停调度器，已运行的仿真回合继续；无重试回合、无丢弃失败。启动前两个尝试的日志和目录单独保留。

原调度器内存保存了被替换的旧 PID，因此本次恢复助手负责在所有服务完成最终 hash 后清理实际进程。恢复过程与实际 PID 见 `provenance/startup_recovery.json`，助手源码随 provenance 保存。这是进程恢复措施，不是模型或试验协议更改。最初七个服务和后续替换服务的源文件版本分别保留，后者仅增加启动诊断。

另外直接解析原 BDDL 比较 Goal 的 10/17/18 任务：物体、区域和初始关系一致，但官方具体初始状态 hash 不同；记录于 `goal_scene_comparison.json`，没有修改场景或交换物体。

## 最终验证结果

20 个任务、1,000 回合全部完成。逐步动作/目标判据、52,972 次推理的连续随机数与示范 ID 均通过审核；全部 1,000 个视频为 20 FPS 且帧数等于执行步数，见 `video_audit.json`。8 个服务运行前后参数哈希完全一致。新增 Python 文件通过 Ruff 检查，指标测试 2 passed。

原调度器在最后清理时确因保留旧 PID 而退出码 1（`controller.log`），发生在全部回合和 8 份最终参数检查完成之后；恢复助手验证完成状态后关闭剩余实际服务。`recovery.log` 与恢复记录确认清理完成，最终 GPU 占用为 0。没有把这次清理异常当作策略失败，也没有重跑任何回合。

汇总时 imageio 启动 ffmpeg 打印了 JAX 多线程与 fork 的提示；汇总正常结束，图片、全部视频审核均成功，无挂起。源文件初版与最终版本分别在 `provenance/source_at_start/`、`provenance/source/`，新增文件整体 diff 为 `provenance/new_files.patch`。最终 tracked diff 与任务开始时一致，本次未改已有跟踪文件。

## 后续补充：实际示范视频导出

新增 `scripts/export_goal_long_demos.py`：按本次 task_manifest 中记录的 episode 读取本地原始 Parquet，核对 task/episode ID、帧数和时间戳；将两路相机完整拼接导出为视频。数据集为 10 FPS（timestamp 间隔 0.1 秒），另提供 20 FPS 版本便于按评估视频的帧速度观看，两者明确标识，不假定采集物理时间相同。

从服务保存的 `demo_and_observation.npz` 导出实际 8 帧采样图，并把原 Parquet 对应帧按模型同一 resize/归一化过程与实际输入核对，20 个任务全部通过，浮点容差 1e-6。40 个视频都核对了完整帧数和 FPS。`demonstrations/manifest.json` 保存来源 hash、采样索引和视频信息；HTML 对照页把完整示范与本轮 ep000 评估放在一起，独立播放，不做阶段对齐。

这是现有数据的离线导出，没有重新运行策略或改变本次成功率。导出脚本通过 Ruff 检查，页面所有本地文件引用已核对存在。复跑：`source scripts/activate_env.sh train` 后运行 `python -m scripts.export_goal_long_demos --root logs/goal_long/full_correct_context_50trials_20260924_121500`。

## 后续补充：VS Code 视频预览入口

新增 `scripts/serve_goal_long_viewer.py`：使用现有 aiohttp 提供仅监听 127.0.0.1 的静态 HTTP 服务，复用 FileResponse 的视频 Range/HEAD、MIME 与条件请求；禁止目录列表和跟随越界符号链接。`--ensure` 通过带实验目录身份的健康检查复用服务，否则后台启动并记录 PID/日志。未改动或重新编码已有视频。

新增本地 `.vscode/tasks.json`、`launch.json` 和 `settings.json`（项目已忽略 `.vscode`）：使用 editor-browser 打开对照页，预启动任务确保服务存在，并提供远程浏览器代理设置。多根工作区设置作用域的处理见 demonstrations/VSCODE_VIEWING.md。

已验证页面及 80 个资源可访问；60 个 MP4 支持正确 MIME、头部和尾部 206 分段响应；Task 1、5 的示范通过 HTTP 实际解码成功。重复 --ensure 未启动重复服务。脚本 Ruff 检查通过。未能直接验证用户端 VS Code GUI；不将服务端检查写成客户端播放成功。服务保留运行供用户观看。
