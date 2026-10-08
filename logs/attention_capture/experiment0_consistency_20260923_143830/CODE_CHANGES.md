# 实验 0：代码修改报告

本次增加可关闭的 attention 记录入口，并验证它不改变 ContextFlow 的推理结果。没有训练或修改检查点，也没有开始新的行为消融。普通策略服务仍使用原来的 `sample_actions`。

## 文件与具体位置

| 文件 | 修改位置 | 用途 |
|---|---|---|
| [gemma.py](/data/zjc/workspace/ContextFlow/src/openpi/models/gemma.py:319) | `Attention.__call__` 的 softmax 后；`Module.setup` 的 scan | 仅显式开启 mutable `attention` collection 时记录实际使用的概率；按层堆叠。初始化和普通推理不产生该 collection，不新增参数。 |
| [contextflow.py](/data/zjc/workspace/ContextFlow/src/openpi/models/contextflow.py:630) | 新增 `sample_actions_with_attention` | 独立诊断入口；原 `sample_actions` 的计算保持不变。 |
| [attention_capture.py](/data/zjc/workspace/ContextFlow/src/openpi/models/attention_capture.py:11) | 新文件：`call_with_attention`、采样循环、`token_layout` | 用同一个 Linen 模块和参数执行采样，将记录作为独立返回值，不写回模型。保存指定层/采样步的完整 heads、queries 和 keys；提供各模态的 token 偏移。 |
| [policy_incontext.py](/data/zjc/workspace/ContextFlow/src/openpi/policies/policy_incontext.py:47) | 抽出 `prepare_inputs`；保留模型引用供诊断使用 | 普通 `infer` 与对照脚本复用原有示范选择、归一化、mask 和随机种子处理，防止复制预处理逻辑产生偏差。 |
| [prepare_attention_observations.py](/data/zjc/workspace/ContextFlow/examples/libero/prepare_attention_observations.py:20) | 新文件：`prepare` | 在 LIBERO 环境回放原实验动作，核对初始状态/图像以及逐步机器人、物体位置和入篮判据，导出动作执行前的观测。 |
| [check_attention_capture.py](/data/zjc/workspace/ContextFlow/scripts/check_attention_capture.py:101) | 新文件：`run`、`validate_attention` | 在训练环境加载检查点，对比记录关闭/开启，交替执行顺序；核对噪声、输入、示范、完整动作、历史动作、概率与 mask；保存逐例原始数据和汇总。 |
| [attention_capture_test.py](/data/zjc/workspace/ContextFlow/src/openpi/models/attention_capture_test.py:20) | 4 个测试 | 覆盖输出/参数、prefix KV cache、完整 JIT 采样循环、二维 patch 尺寸和 token 偏移。 |
| [check_attention_capture_test.py](/data/zjc/workspace/ContextFlow/scripts/check_attention_capture_test.py:17) | 2 个测试 | 验证验收程序能拒绝 masked key 泄漏、no-context 示范注意力和 state query 偷看后续 action。 |
| [README.md](/data/zjc/workspace/ContextFlow/README.md:139) | 新增实验 0 小节 | 两套环境的执行命令、默认矩阵、保存格式、验收标准与解释范围。 |

## 记录的内容与范围

- Gemma 第 0、9、17 层；flow 采样第 0、5、9 步，均从零计数。动作仍执行完整 10 步采样。
- 每个采样点保留 8 个 heads、51 个 suffix queries（1 个 state + 50 个 action），不事先平均。
- keys 对应当前图像、语言、压缩后的示范图像/状态/action、当前 state 和预测 action；保存无效相机和 padding 对应的位置及 mask。
- 本次不记录 SigLIP/Perceiver 内部 attention。示范图像的 32 个压缩 token 不能直接当成 32 个空间 patch 来画物体热图。
- Pelican 的 PyTorch/Qwen 采集器未直接移植；这里实现适合 JAX/Flax 的采集入口。后续可以使用其绘图思路处理导出的数组。

## 遇到的问题与修正

1. **工具沙箱启动失败**：默认工具报 bubblewrap mountinfo 错误。改用经过权限检查的命令执行方式，在本仓库内完成修改与验证。
2. **patch 尺寸接口不匹配**：真实模型的 `_image_patch_size` 为 `(patch_h, patch_w)`，初版位置映射误当作标量。改为分别计算高、宽方向的 patch 数，并加入非正方形 patch 测试。第一次修复命令在激活虚拟环境前调用了不存在的 `python`，第二次启动仍使用旧代码；随后用 `python3` 完成修复并重新启动。失败日志 `validation_attempt1.log`、`validation_attempt2.log` 保留。
3. **精度说明需准确**：现有服务是 float32 加载权重、Gemma 内部 bfloat16 计算。保持原配置；attention 概率 float64 求和允许 0.004 的 BF16 舍入误差，动作差异仍要求零。没有通过放宽动作误差阈值让实验通过。

## 测试与代码边界

新增 6 个测试及现有 context-ablation/serve-policy 相关测试合计 **12 passed**，见 [tests.log](tests.log)。新增 Python 文件 Ruff 检查通过，`git diff --check` 通过；真实检查点实验结果见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。

本次未提交或 push。工作区此前已有位置交换实验的未提交文件；这些改动仍保留，未计作本次新增功能。README 中已有的位置交换说明也保留。本目录下的报告、观测、attention 数组和日志均为本地实验产物。`provenance/source/` 保存相关源码快照，`experiment_files.json` 标明本次涉及的九个文件；`workspace.patch` 包含整个工作区差异，因此也包含此前的位置交换工作。
