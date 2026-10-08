# 实验 0：attention 记录一致性结果

**通过：180/180 组，开启记录前后的完整动作逐元素一致；落盘后的动作数组进一步核对为逐字节一致。** 原始动作、反归一化动作的最大差异均为 **0**。这说明在本次覆盖的输入、检查点和数值配置下，记录 attention 没有改变推理结果，可以继续做下一步 attention 分析。

## 实验设计

- 检查点：`/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999`，配置 `ContextFlow`。
- GPU：NVIDIA A100 80GB，物理 GPU 1（进程内 `cuda:0`）；JAX `0.5.0`。
- 维持原服务的混合精度：float32 加载权重，Gemma 内部 bfloat16；未为通过检查而改变计算精度。
- 使用此前位置交换实验的番茄酱场景，2 种布局 × 2 条语言 × correct/no/wrong = **12 个条件**。
- 每个条件取初始状态索引 0–4，在控制步 0、80、160 的**动作执行前**各取一个观测，即每条件 15 组，总计 **180 组配对推理**。它们不是 180 条新的完整机器人 rollout。
- 首先运行每条件 episode 0 / step 0 的 12 组，全部通过后扩展剩余 168 组。
- 每一对只改变“是否记录 attention”。当前图像、机器人状态、语言 tokens/mask、示范图像/状态/action/mask、checkpoint、PRNG key 和初始 action noise 完全相同。交替执行两种路径的先后顺序，检查潜在的记录状态残留。
- correct：milk 语言配 milk 示范，tomato_sauce 语言配 tomato_sauce 示范；wrong 反过来；no：示范值和 mask 全部关闭。实际示范 episode：milk **814**，tomato_sauce **821**；no 的 selected_episode 为空。

“相同输入”指**每组记录关闭/开启之间相同**。不同 context 条件的中后期观测来自各自原有 rollout，可以不同。本实验不据此直接比较不同 context 的注意力大小或因果效应。

## 各条件结果

| 布局 | 语言目标 | context | 配对数 | 动作完全一致 | 原始动作最大差异 |
|---|---|---|---:|---:|---:|
| original | milk | correct | 15 | 15 | 0.0 |
| original | milk | no | 15 | 15 | 0.0 |
| original | milk | wrong | 15 | 15 | 0.0 |
| original | tomato_sauce | correct | 15 | 15 | 0.0 |
| original | tomato_sauce | no | 15 | 15 | 0.0 |
| original | tomato_sauce | wrong | 15 | 15 | 0.0 |
| swap_milk_tomato | milk | correct | 15 | 15 | 0.0 |
| swap_milk_tomato | milk | no | 15 | 15 | 0.0 |
| swap_milk_tomato | milk | wrong | 15 | 15 | 0.0 |
| swap_milk_tomato | tomato_sauce | correct | 15 | 15 | 0.0 |
| swap_milk_tomato | tomato_sauce | no | 15 | 15 | 0.0 |
| swap_milk_tomato | tomato_sauce | wrong | 15 | 15 | 0.0 |

## 验收结果

| 检查 | 结果 |
|---|---|
| 原始完整动作 `(1, 50, 32)` | 180/180 逐元素与逐字节一致，最大差异 0 |
| 反归一化完整动作 `(50, 7)` | 180/180 逐字节一致，最大差异 0 |
| 与原实验实际执行的后续 5 个动作比较 | 最大差异 0.0 |
| 初始 action noise | 每组与同一个 PRNG key 生成的 noise 完全一致 |
| 输入是否被修改 | 180 组推理前后 processed input SHA256 一致 |
| 参数是否被修改 | 实验前后完整参数 SHA256 一致 |
| attention 是否有效 | 所有概率有限且位于 [0, 1] |
| 无效相机、padding 等 masked keys | 注意力为 0 |
| no-context 的示范 keys | 60/60 组，所有记录的 queries/heads/layers/steps 上注意力为 0 |
| state query 到后续 action keys | 注意力为 0，符合原 attention mask |
| attention 行和误差 | 最大 0.0030389501，小于 BF16 舍入检查阈值 0.004 |
| 原轨迹回放 | 60 条来源轨迹的初始状态/观测核验通过，回放至控制步 164；机器人与全部物体位置最大误差 0，入篮判据相同 |
| 自动测试 | 12 passed；新增文件 Ruff、git diff --check 通过 |

参数 SHA256：`19e92cff3ca1839d800da29b2f8362e8058522556bd316164d6bd00243d6b4d4`。

动作的零差异要求与 attention 行和的舍入检查是两回事。记录的是模型实际计算、已舍入到 BF16 的概率，存储数组为 float32；这只是无损扩大存储类型，不表示 attention 用 float32 计算。

## 记录格式

每组 `capture.npz` 的 attention 形状为 **`(3, 3, 1, 8, 51, 1027)`**，依次为：采样步、层、batch、head、query、key。

- 采样步：0、5、9；层：0、9、17，均从零计数。
- 51 个 queries：当前 state 1 个 + 预测 action 50 个。
- 1027 个 keys：当前图像 3×256（包含被 mask 的右手相机占位）+ 语言 48 + 示范图像压缩 token 3×32 + 示范 state 32 + 示范 action 32 + 当前 state 1 + action 50。
- `layout.json` 保存上述区间、逐 token 语言片段、真实示范选择；没有提前对 heads 或 queries 做平均。

## 如何读取本次文件

- [代码修改报告](CODE_CHANGES.md)：逐文件说明、修复过程与测试。
- [总体结果](validation_attempt3/results.json)、[条件汇总 CSV](condition_summary.csv)。
- [逐组比较](validation_attempt3/pairs.jsonl)：input/noise hashes、实际示范、误差和 attention 检查。
- [落盘独立核验](artifact_audit.json)：逐文件 checksum、动作逐字节一致检查。
- [观测清单](observations/manifest.json)：180 个快照及来源与回放校验。
- [示例 attention](validation_attempt3/original/correct/milk/ep000_step000/capture.npz)、[示例 token 映射](validation_attempt3/original/correct/milk/ep000_step000/layout.json)。
- `provenance/source/`：源码快照；`provenance/source_sha256.json`：文件哈希；`tests.log`：测试输出。

首次和第二次启动因 patch 尺寸接口问题失败，没有产生通过结论；修复后 `validation_attempt3` 完整完成全部 180 组。具体修复见代码报告。日志中的 Hugging Face 本地缓存提示未影响已有本地示范读取。

## 结论边界

本实验验证的是记录工具的一致性，不增加新的行为机制结论。不能从本报告推断模型更看重语言、示范位置或物体身份；也不能把压缩后的示范图像 token 直接当作空间 patch。SigLIP/Perceiver 内部 attention 尚未采集。

下一步可以在这些已核验的输入上分析当前图像和语言 token 的注意力分布；是否具有因果影响仍需后续 mask 或模态干预实验。这次没有自动启动下一阶段。
