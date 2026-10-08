# 示范 state / state+action 消融：代码修改报告

本轮只新增六个独立文件，普通模型、checkpoint、训练配置、旧 action/no 消融入口和现有仿真逻辑均未修改。开始时已有的工作区变更完整保存于 `provenance/preexisting.patch` / `preexisting_status.txt`，不将它们算作本轮改动。

## 改动位置

| 文件 | 本轮新增内容 |
|---|---|
| [demo_modality_ablation.py](/data/zjc/workspace/ContextFlow/src/openpi/models/demo_modality_ablation.py:15) | `zero_modalities` 按现有 `token_layout` 找到示范 state/action 块，拒绝未知或缺失块，只置零指定表示，返回原 mask/ar_mask。避免按尾部猜测 state 位置。 |
| 同文件第 28 / 39 行 | `DemoModalityAblationConfig` 用 `ablation=state/state_action` 选择干预；独立子类只覆盖 `embed_midfix`，继承原采样算法，禁止训练。`inspect_ablation` 返回完整/置零 prefix 供实际检查点核对。 |
| [demo_modality_ablation_test.py](/data/zjc/workspace/ContextFlow/src/openpi/models/demo_modality_ablation_test.py) | 两种干预的 JIT 测试：指定块归零、其他块精确保留、mask/位置/attention 可见性不变、输入不被改写、被消融数值不影响置零结果；另测非法模态和缺失块。 |
| [serve_demo_modality_ablation.py](/data/zjc/workspace/ContextFlow/scripts/serve_demo_modality_ablation.py:41) | 从上一轮 action 实验服务适配的独立入口。`check_prefix` 按模态检查零块，并逐块确认所有非消融内容原样保留，包括图像、语言和 state-only 条件的 action。 |
| 同文件第 122 行 | 每个进程的启动 gate：60 份历史完整模型动作逐字节复核；12 份 prefix/attention 一致性；8 份 correct/wrong 输入的被消融模态逐项和联合数值扰动不变性；4 份旧 masked no 输出不变性。参数 hash 相同才继续。 |
| 同文件第 197 行 | 独立请求协议与干预标识，拒绝错误服务/非法条件。0/80/160 步保存 attention、原始观测、动作、token 布局和 prefix 检查。记录与普通动作严格相同，输入 hash 不变；结束接口复核模型参数。 |
| [demo_modality_ablation.py](/data/zjc/workspace/ContextFlow/examples/libero/demo_modality_ablation.py:39) | 使用已有 `context_ablation.run` 的 client factory 注入带干预检查的客户端。保持固定番茄酱场景、两布局、两语言、correct/wrong、初始状态 0–4、280 步、每 5 步重规划。服务器元数据和每次响应必须匹配指定干预。 |
| [summarize_demo_modality_ablation.py](/data/zjc/workspace/ContextFlow/scripts/summarize_demo_modality_ablation.py:58) | 核对 80 条新轨迹与历史 full/action 的配对；检查 240 个 capture 与实际执行动作；生成四格对照、逐回合转移、固定语言换示范的物体切换计数、CSV/JSON、图、视频索引和总/分实验报告。 |
| [DEMO_MODALITY_ABLATION.md](/data/zjc/workspace/ContextFlow/scripts/DEMO_MODALITY_ABLATION.md) | 两种干预的命令、GPU/端口分离方式、最终参数核对和报告生成步骤。 |

## 精确定义

- `state`：把 Gemma 前 32 个压缩示范 state token 置零（当前 912:944），保留示范图像和 action。
- `state_action`：把上述 state 和 32 个 action token 都置零（当前 944:976），仅示范部分保留图像。
- 两种条件都保留当前摄像头、自然语言、当前机器人本体状态；序列长度、有效 mask、attention block mask、位置编号不变。
- 置零发生在各模态投影/Perceiver 压缩之后，不是给输入状态填零，也不是给机器人发送零动作。零 token 保持有效，后续层仍能融合其他信息。
- 两服务独立进程/显卡运行，避免静态干预配置互相切换；使用相同 checkpoint，未添加可训练参数。

## 为什么复用历史完整示范和 action 条件

每格使用历史同一批初始状态 0–4，共 40 个完整示范与 40 个 action 置零回合。每个新服务先确认普通模型输出与实验 0 记录逐字节一致；汇总再逐回合检查初始模拟器状态、观测哈希、布局来源、示范 episode 和全部 56 次噪声种子。因此这是四条件配对，不拿不同回合数的历史总成功率混比。

新的 no-context 25 次补跑用的是各自任务场景，本次固定番茄酱场景；旧位置交换 no 又关闭了 mask。这两种 no 不加入本次四格表。

## 验证与复核材料

- 新旧模态消融相关测试：**11 passed**，见 `model_tests.log`。
- 仿真/位置干预测试：**9 passed**，见 `simulator_tests.log`。
- 两个真实检查点 gate 均通过：`zero_state/server/GATE_PASSED.json`、`zero_state_action/server/GATE_PASSED.json`；详细样例在各自 `gate_cases.jsonl`。
- 最终轨迹/attention 验收见 `comparison.json`、`VALIDATION.md`，每个服务的参数终检在 `SERVER_FINISHED.json`。
- `provenance/source/` 与 `source_sha256.json` 保存本次相关代码快照；`DEMO_MODALITY_ABLATION.patch` 仅包含上述六个新增文件。
- 原实验报告与原始记录保留，新实验结果单独保存。本轮没有提交或推送 GitHub。

## 解释上的限制

表示置零可能产生 OOD。观察到失败或偏向另一物体，能说明该输入干预改变了行为，但不能直接证明原模型只使用 state、完全不使用图像，或某个模态已经学到语义。需要结合 correct/wrong 和布局交换判断：错误行为减少可能是纠正，也可能是执行失败；在交换布局中语言成功也可能是恰好操作了示范位置上的物体。
