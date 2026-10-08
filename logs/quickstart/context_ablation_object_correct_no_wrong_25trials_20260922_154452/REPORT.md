# 牛奶 / 番茄酱 correct / no / wrong context 对照

原实验：2026-09-22，6 组各 25 次，共 150 次。2026-09-23 补跑保留 mask/位置编号的 no-context：两个任务各 25 次，共新增 50 次。以下主矩阵已采用新 no，correct/wrong 复用原记录；旧 no 数据保留。所有回放均为 20 FPS。

## 结论

**补充结果：新 no 的两个任务仍均为牛奶 0/25、番茄酱 0/25。零成功率在保留 mask 和位置编号的置零干预中也成立，不能只用旧版位置编号变化解释。**

详细定义、验收和新旧对照见[补充实验报告](../../context_ablation/no_context_zero_keep_mask_25trials_20260923_204651/EXPERIMENT_RESULTS.md)。

示范对行为有明显因果影响，但“两个方向都改为操作示范中的物体类别”不成立。
语言为番茄酱、示范改成牛奶时，22/25 回合把牛奶放入篮子；同初始状态的配对中，
19/25 从 correct 组的番茄酱入篮变为 wrong 组的牛奶入篮。
反方向语言为牛奶、示范改成番茄酱时，两个指定目标均为 0/25，进一步核对发现
**25/25 实际把奶油奶酪放入篮子**。这是行为改变，而不是简单不动或全部执行失败。

## 六组矩阵（no 更新为表示置零、保留 mask）

| 固定语言 / 当前任务场景 | 示范 | 牛奶入篮 | 番茄酱入篮 |
| --- | --- | ---: | ---: |
| 牛奶 | 牛奶（correct） | 21/25 = 84% | 0/25 |
| 牛奶 | 无示范内容（新 no：保留 mask） | 0/25 | 0/25 |
| 牛奶 | 番茄酱（wrong） | 0/25 | 0/25 |
| 番茄酱 | 番茄酱（correct） | 0/25 | 21/25 = 84% |
| 番茄酱 | 无示范内容（新 no：保留 mask） | 0/25 | 0/25 |
| 番茄酱 | 牛奶（wrong） | 22/25 = 88% | 0/25 |

wrong 牛奶语言组：奶油奶酪入篮 **25/25 = 100%**，其余物体均未入篮。
主表只报告牛奶和番茄酱。新 no 另记录了全部场景物体，见补充报告；历史 correct/wrong 的其他物体结论仍按原回放核对范围解释。

![正确示范、新 no、错误示范及旧 no 对照](comparison_no_context_zero.png)

旧 no（关闭 mask）在两个任务中均为牛奶 0/25、番茄酱 0/25；原图 `comparison.png` 和原 `comparison.csv/json` 保留为历史记录。新版汇总见 [comparison_no_context_zero.json](comparison_no_context_zero.json) 和 [CSV](comparison_no_context_zero.csv)。

## 配对证据与空间位置因素

- 牛奶语言：correct 组有 21 个成功回合；同一批初始状态换番茄酱示范后，
  这 21 个回合全部改为奶油奶酪入篮。
- 番茄酱语言：19 个回合从番茄酱入篮转为牛奶入篮；2 个从成功变为两个指定目标均失败；
  原本失败的 4 个中，3 个换示范后牛奶入篮，1 个仍未完成两个指定目标。
- 检查 BDDL 场景定义：番茄酱任务的目标初始区域与牛奶任务中奶油奶酪的初始区域相同，
  x 范围 0.025–0.075 m、y 范围 −0.125–−0.075 m。牛奶在两个任务中的初始区域也相同。
  这解释了为什么两个方向的“物体类别跟随”不对称：空间位置线索也是混杂因素。
- 结果支持模型强烈利用示范中的空间／动作信息；不能仅凭本实验判定它完全忽略语言，
  或已理解示范物体的语义类别。区分“按类别”与“按位置”，下一步应交换物体初始位置做控制。
- 新 no 在 Gemma 前将三种示范模态的全部编码表示置零，保留原 mask、token 槽位和位置编号，正确语言、实时图像和本体状态不变。旧 no 则将示范数据置零并关闭 mask，没有物理删除槽位，但有效位置编号改变。两者均可能受到未训练过的输入分布影响；失败不能简单等同于“模型不理解语言”。

## 原实验同一初始状态的回放（episode 001；此处 no 为旧 masked 版本）

| 固定语言 | correct | no | wrong |
| --- | --- | --- | --- |
| 牛奶 | [牛奶入篮](correct/videos/pick_up_the_milk_and_place_it_in_the_basket/ep001_language_only.mp4) | [两个目标均未完成](no/videos/pick_up_the_milk_and_place_it_in_the_basket/ep001_neither.mp4) | [奶油奶酪入篮](wrong/videos/pick_up_the_milk_and_place_it_in_the_basket/ep001_neither.mp4) |
| 番茄酱 | [番茄酱入篮](correct/videos/pick_up_the_tomato_sauce_and_place_it_in_the_basket/ep001_language_only.mp4) | [两个目标均未完成](no/videos/pick_up_the_tomato_sauce_and_place_it_in_the_basket/ep001_neither.mp4) | [牛奶入篮](wrong/videos/pick_up_the_tomato_sauce_and_place_it_in_the_basket/ep001_other_only.mp4) |

新旧 no 的 50 对视频见[配对视频入口](../../context_ablation/no_context_zero_keep_mask_25trials_20260923_204651/VIDEO_INDEX.md)。

文件名中的 `neither` 仅表示牛奶和番茄酱均未入篮，不表示没有操作其他物体。

## 实验协议

- 两个任务均属于 `LIBERO_UNSEEN_TASKS`；测试阶段使用示范，无参数更新。
- 检查点：`/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999`；float32。
- 固定演示：牛奶 dataset task 25 / episode 814；番茄酱 task 28 / episode 821。
- 每个任务使用初始状态索引 0–24；新 no 与历史 correct/no/wrong 逐一核对初始状态、观察哈希和每次推理噪声一致。两个任务各自使用原任务场景，未混用固定番茄酱场景或交换位置的结果。
- 环境种子：`7 + dataset_task_index * 1000 + episode_index`。
- 推理噪声：`fold_in(key(0), task_index, episode_index, replan_index)`，各组相同，
  不受先前任务或回合的结束时间影响；每 5 个动作重新规划。
- 每回合先等待 10 步，再固定执行 280 步，两个目标都不会触发提前终止。
- 使用 LIBERO 的 `In` 判定，每一步同时检查两种物体，报告曾经入篮与最终入篮状态。
  同时逐步验证语言目标的判定与原环境 `check_success()` 一致。
- 与 Table 1 的提前成功终止、连续策略随机流协议不同，correct 在本实验内重新评估；
  不把旧 Table 1 结果直接作为这次干预的配对基线。

## 原始 150 回合的验证与记录（补跑验收见补充报告）

- 全部 150 回合完成；每组 50 个视频、50 个轨迹文件；所有视频 280 帧、20 FPS。
- JSONL 与保存的逐步谓词一致；动作均为有限数；同回合的三组初始模拟器状态、
  初始观察哈希及 56 次推理种子全部一致。
- 六组各一次短回合检查通过；重复相同观察和种子得到完全相同动作。
- 9 项相关测试通过，新文件 lint、shell 语法和差异空白检查通过。
- 奶油奶酪结果来自原动作回放，严格核对原机器人末端／夹爪状态及牛奶、番茄酱的位置和谓词，
  wrong/milk 25 回合的最大轨迹误差均为 0；未修改模型输出重新生成轨迹。
- 额外的真实检查点“任意改变被屏蔽示范内容也不影响动作”检查没有完成：GPU 自动审批两次超时；
  CPU 替代方案停在 Orbax 元数据读取，已终止／超时退出。因此不声称该项验证通过。

主要文件：[汇总 CSV](comparison.csv)、[配对转移 JSON](comparison.json)、
[额外物体核对](all_object_comparison.json)、[逐回合物体回放记录](all_object_replay.jsonl)、
[运行配置和执行备注](run.json)。各条件下还有 `episodes.jsonl`、`results.json`、`videos/`、`trajectories/`。

## 遇到的问题及处理

1. no 组启动时 Hugging Face SSL 失败，尚无回合执行；切换本地数据模式后正常完成。
2. 临时并行控制脚本在运行期间被编辑，correct/wrong 两组的 shell 在评估全部结束后报 EOF。
   两组评估程序均已写完 50 回合和最终结果；退出清理已运行，主评估 GPU 显存已核对释放。
   发布用 quickstart 是另一个脚本，语法及菜单预览检查通过。
3. 补充回放尝试无渲染模式时出现原生库退出问题或严格状态／轨迹核对不通过；
   不采纳这些尝试的结果，最终 wrong/milk 全部使用原渲染配置回放并通过零误差核对。

## 再运行

本次新 no 使用[独立启动说明](/data/zjc/workspace/ContextFlow/scripts/NO_CONTEXT_ZERO.md)。

`bash quickstart.sh` → **9** → 每组次数（默认 25）→ 检查点、GPU、端口，仍对应旧 masked no 定义，不能用它冒充本次保留 mask 的对照。
新版本会直接记录所有场景物体的入篮情况，不必事后回放补算。
报告和运行数据保留在本地 `logs/`；此次实验代码尚未提交或推送到 GitHub。
