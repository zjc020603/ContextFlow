# 实验 2：代码修改报告

## 改动范围

本次增加独立的示范 action 消融模型类、服务入口、评估客户端和汇总脚本。普通模型 `ContextFlow`、训练配置、checkpoint、原 no-context 实现均未修改。已有评估器只增加可选的 client factory，用于复用原先的物理初始化、视频、轨迹和成功判据。

实验定义为 `zero_compressed_demo_action_keep_mask`：在完整示范的 action 投影与 Perceiver 压缩之后、进入 Gemma 之前，把最后 32 个 action token 置零；所有其他 prefix token、有效 mask、attention block mask、序列长度、位置编号不变。

## 逐文件修改

| 文件与位置 | 改动及理由 |
|---|---|
| [demo_action_ablation.py](/data/zjc/workspace/ContextFlow/src/openpi/models/demo_action_ablation.py:16) | 新增 `zero_action_tail`，只对最后指定数量的 token 做非原地置零，返回原 mask/ar_mask；拒绝非法范围。 |
| 同文件 `DemoActionAblationConfig`、`DemoActionAblation` | 通过独立 config 加载同一检查点，继承原模型的采样路径。仅覆盖 `embed_midfix`：调用原实现后置零 action 尾块。要求已有的 action/state 压缩配置，拒绝训练模式，不添加参数。`inspect_ablation` 返回前后 prefix 供验收。 |
| [demo_action_ablation_test.py](/data/zjc/workspace/ContextFlow/src/openpi/models/demo_action_ablation_test.py:10) | 新增两个测试，覆盖 JIT 后仅指定 token 变零、输入不被原地修改、mask/attention mask/位置编号保持、action 尾块数值变化无法泄漏到输出，以及非法范围。 |
| [serve_demo_action_ablation.py](/data/zjc/workspace/ContextFlow/scripts/serve_demo_action_ablation.py:61) | 专用本地服务。加载完整模型及置零模型，比较参数 hash，先运行历史动作和消融一致性检查，通过后才开始监听。推理请求必须明确携带消融协议，避免误连普通服务。 |
| 同文件 `gate` | 比较 60 份历史初始观测的完整 raw/executable actions；12 份验证 prefix/记录一致性；8 份验证改变 action 输入后置零输出仍一致；4 份 no-context 验证此额外置零不改变其输出。 |
| 同文件 `infer` | 每次返回实际干预标记；在环境步 0/80/160 保存未经平均的 attention、观测、动作、token 范围和 prefix 检查。每次采集比较正常置零推理与带记录置零推理。完成接口复核参数 hash。 |
| [context_ablation.py](/data/zjc/workspace/ContextFlow/examples/libero/context_ablation.py:106) | 本次仅修改 `run(args, *, client_factory=None)` 和创建客户端的两行。未指定 factory 时仍使用原 websocket 客户端；原初始化、模拟步进和统计逻辑不变。该文件中其他未提交差异来自之前的位置交换工作。 |
| [demo_action_ablation.py](/data/zjc/workspace/ContextFlow/examples/libero/demo_action_ablation.py:16) | 新增专用客户端与 40 条 rollout 入口；检查服务元数据、附加 layout/干预标签并核对响应。复用原评估器运行两个布局、两条语言和 correct/wrong。每条件最多 5 次，固定原有 280 步和种子协议。 |
| [summarize_demo_action_ablation.py](/data/zjc/workspace/ContextFlow/scripts/summarize_demo_action_ablation.py:59) | 新增配对审计与汇总：取历史对应初始状态 0–4，核对初始化、配置、示范、噪声种子；重算入篮结果；验证 120 份记录中的动作确实执行；输出 CSV、JSON、图和中文结果报告。 |
| [DEMO_ACTION_ABLATION.md](/data/zjc/workspace/ContextFlow/scripts/DEMO_ACTION_ABLATION.md) | 启动、完成参数复核、停止服务和生成报告的命令，以及 OOD 与 no-context 的解释范围。 |

## 为什么没有直接把 action mask 关掉

旧 no-context **本来就没有删除张量中的 token 槽位**，而是将全部示范值及 mask 置零。问题在于原模型用 `cumsum(mask)`/有效 prefix 数计算位置编号，因此即使张量长度不变，mask=false 仍会改变 suffix 相对图像的位置。

本次使用表示置零、mask 保持 true（原先有效的位置仍有效），让原 mask 和位置编号精确不变。这不是修复或替换旧 no-context 的定义，而是为新的 action-only 实验避免同一项混杂。旧 no-context 记录仍保留原定义。

32 个零 token 仍是可读取的 key，并且在后续层可以融合其他模态的信息。因此不能用“这些位置的 attention 是否为零”判断消融是否生效；验收检查的是进入 Gemma 前确实没有原始 demo action 内容，以及改变 demo action 数值后输出是否保持不变。

## OOD 边界

零向量不是训练过的缺失模态标记。不能保证它在训练分布内，也不能把性能下降直接等价为“模型必须依赖 action”。同时，保留的示范图像和状态仍含运动信息。本次先做每条件 5 次的有限试验，不把结果当作最终机制定论。

## 测试与运行问题

- 模型侧相关测试：13 passed，见 [model_tests.log](model_tests.log)。
- 原评估与位置干预测试：9 passed，见 [simulator_tests.log](simulator_tests.log)。
- 新文件通过 Ruff；最终改动检查见验证记录。真实检查点门槛见 [GATE_PASSED.json](server_attempt1/GATE_PASSED.json) 与 [逐样本记录](server_attempt1/gate_cases.jsonl)。
- 最初在模拟器环境直接执行 `pytest` 时，入口没有把仓库根目录加入 import path，测试收集报 `No module named examples`。改用 `python -m pytest` 后 9 项通过；未为此改动环境依赖。失败输出保存在 `simulator_tests_initial_failure.log`。
- 未修改模型精度：float32 加载参数，Gemma 内部沿用 bfloat16。没有放宽动作一致性标准。

## 复核与代码边界

[EXPERIMENT2.patch](provenance/EXPERIMENT2.patch) 只包含本次新增文件及评估器的三行逻辑变化，便于从此前工作区继续审查；[源码快照](provenance/source) 和 [SHA256](provenance/source_sha256.json) 保存相关实现。`preexisting_worktree.patch`/`preexisting_status.txt` 保存开始前的未提交状态，避免把此前或其他分支的工作混为本次修改。

本次没有提交或 push。报告、视频、轨迹和 attention 均保存在本地 logs 目录。实验结果见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。
