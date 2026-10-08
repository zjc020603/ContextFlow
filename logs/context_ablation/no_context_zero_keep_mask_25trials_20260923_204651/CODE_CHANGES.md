# No-context 置零对照：代码修改报告

本次新增独立的模型、服务、评估和汇总入口，用于重新评估原 Correct / No / Wrong 实验的两个 no 分组。没有修改普通模型、checkpoint、训练配置、历史 no 实现或 correct/wrong 路径。当前工作区中已有的其他修改属于先前实验，见 `provenance/preexisting.patch`。

## 实验定义

`zero_all_encoded_demo_tokens_keep_mask`：在 Gemma 前，把示范图像、状态、action 的编码表示全部置零，保留 token 槽位、mask、attention block mask 和位置编号。按布局信息定位各块，当前配置为 160 个示范槽位（其中 128 个有效，缺失右相机的 32 个仍无效）。这与“将原始物理动作设为 0”以及“关闭示范 mask”均不同。

## 文件和关键位置

| 文件 | 本次新增内容 |
|---|---|
| [demo_context_zero.py](/data/zjc/workspace/ContextFlow/src/openpi/models/demo_context_zero.py:13) | `zero_demo_blocks` 按 `token_layout` 将所有 `demo_*` 块置零，返回原 mask 和 block mask；不原地改输入。 |
| 同文件，第 24 / 31 行 | 独立 config 和 `DemoContextZero` 子类，复用原 `sample_actions`。仅覆盖 `embed_midfix`；拒绝训练，要求三种示范模态和压缩配置。`inspect_ablation` 提供真实 prefix 前后对照。 |
| [demo_context_zero_test.py](/data/zjc/workspace/ContextFlow/src/openpi/models/demo_context_zero_test.py) | 验证三种示范全部归零、非示范内容/形状/类型不变、mask 与位置和 attention 可见性不变、原输入未被改写、改变示范值后相同。 |
| [serve_no_context_zero.py](/data/zjc/workspace/ContextFlow/scripts/serve_no_context_zero.py:102) | `prepare_zero` 把外部 no 请求转为完整同任务示范的预处理，以获得完整 mask 模板；独立模型在 Gemma 前抹去全部示范内容。对外报告 `mode=no`、无选中示范，同时明确保留 `mask_template_episode` 供审计。 |
| 同文件，第 122 行 | `gate` 核对 6 个历史快照的基线动作；两个任务分别验证 prefix/位置保持、任意修改所有示范数值与交换示范任务后输出完全一致、记录 attention 不改变输出。全部通过后才监听端口。 |
| 同文件，第 212 行 | 专用协议拒绝非 no 请求。第 0/80/160 步记录 attention、输入、动作、布局和逐项检查；普通动作与记录动作严格一致才写入。重复推理验证不会重复保存同一 capture。结束接口核对参数 hash。 |
| [no_context_zero.py](/data/zjc/workspace/ContextFlow/examples/libero/no_context_zero.py) | 客户端校验专用服务器元数据与每次响应。复用现有 `context_ablation.run`，固定 `scene_task=per_language`、原布局、25 次、相同种子和 280 步，不改变仿真与成功判据。 |
| [summarize_no_context_zero.py](/data/zjc/workspace/ContextFlow/scripts/summarize_no_context_zero.py:27) | 审核全部 50 回合与历史三组的初始状态/观察/推理噪声；核对逐步谓词、场景物体、150 个 capture 与实际执行动作；生成新旧 no 对照 CSV/JSON、图表、配对视频入口和结果报告。 |
| [NO_CONTEXT_ZERO.md](/data/zjc/workspace/ContextFlow/scripts/NO_CONTEXT_ZERO.md) | 完整启动、finalize、汇总命令及定义说明。旧 quickstart 的 no 仍是历史 mask 版本，新对照用这里的专用入口。 |

本次没有改动 `contextflow.py`、`gemma.py`、`transforms.py`、`policy_incontext.py`、`context_ablation.py`；它们在本轮开始前已有修改的部分保留原样。

## 验证

- 模型/干预相关测试 **8 passed**：`model_tests.log`。
- 仿真和位置干预测试 **9 passed**：`simulator_tests.log`。
- 新文件 Ruff 检查和格式检查通过。
- 真实 checkpoint gate 全部通过，见 `server/GATE_PASSED.json` 和 `server/gate_cases.json`。参数 SHA256 为 `19e92cff3ca1839d800da29b2f8362e8058522556bd316164d6bd00243d6b4d4`。
- 完整 rollout 和采集验收见实验完成后的 `comparison.json`、`server/SERVER_FINISHED.json`、`EXPERIMENT_RESULTS.md`。

## 遇到的问题及修复

第一次 gate 的预处理一致性断言失败：`ObservationIncontext.from_dict` 会将嵌套字典内的 uint8 图像转换为 float32；检查时一侧已经转换，另一侧尚未转换。修复检查脚本，使两侧都完成同一转换后再比较整个 observation 和输入字典，第二次 gate 通过。未修改原预处理、未放宽数值容差，失败发生在任何仿真 rollout 之前。失败日志和目录保存在 `server_failed_gate_attempt1.log` / `server_failed_gate_attempt1/`。

原 attention validator 的 `mode=no` 专门检查所有示范 key 权重为零，只适用于旧 mask=false 版本。本脚本使用明确的 `zero_all_encoded_demo_tokens_keep_mask` 标签走通用检查，仍严格检查实际 mask、概率范围/归一化与 action 一致性；不错误要求新零 token 的注意力为零。

## 可复核材料与边界

- `provenance/NO_CONTEXT_ZERO.patch` 仅包含本次新增的六个代码/说明文件。
- `provenance/source/`、`source_sha256.json` 保存本次相关代码快照与哈希。
- 原 no 的视频、轨迹、CSV/JSON 均保留。原实验报告会指向新补充报告与显式区分新旧 no 的汇总文件。
- 固定番茄酱场景和位置交换实验的 no 条件未重跑，不能把本次结果混用到它们；action-only 置零和 Table 1 复现不受本次修改影响。
- 所有新增数据和报告位于本地 `logs/`。本轮不提交或推送 GitHub。
