# 代码修改报告：实验 3 / 4

本次新增独立的实验入口，沿用原始 ContextFlow 检查点和实验 0 的 attention 采集工具。没有修改模型结构、训练代码、实验 2 的消融实现或 quickstart 菜单。

## 新增文件及职责

| 文件 | 具体部分 | 作用 |
|---|---|---|
| [scripts/serve_language_vision.py](/data/zjc/workspace/ContextFlow/scripts/serve_language_vision.py) | `Policy.__init__ / evaluate / save_capture / infer` | 加载原始 float32 权重；与实验 0 的 correct/wrong 初始动作逐位核对；普通推理与记录 attention 输出核对；固定采样锚点 `[0,25,episode,replan]`；检查同观测语言对照的非语言输入及噪声完全相同；保留示范、噪声和 token 记录。 |
| [examples/libero/language_vision_experiment.py](/data/zjc/workspace/ContextFlow/examples/libero/language_vision_experiment.py) | `run` | 相同模拟器初始化、固定 280 步完整回合；独立设置语言/示范/视觉条件；记录六个物体的接近、接触、双侧夹持、升高和入篮事件；保存实际模型输入及双相机视频。 |
| [examples/libero/live_vision_intervention.py](/data/zjc/workspace/ContextFlow/examples/libero/live_vision_intervention.py) | `segmentation / box_mask / matched_object_mask / background_mask / intervene` | MuJoCo 可见几何体定位；当前双相机像素遮挡；严格等面积背景预算；记录部分遮挡、形状改变和夹爪遮挡；检查渲染不改变物理状态、未遮挡像素保持不变。 |
| [examples/libero/live_vision_intervention_test.py](/data/zjc/workspace/ContextFlow/examples/libero/live_vision_intervention_test.py) | 8 个针对性测试 | 物体框定位、背景等面积与避开物体、不可见物体、可用背景不足、碎片背景及共同面积上限。 |
| [scripts/experiments/launch_language_vision.py](/data/zjc/workspace/ContextFlow/scripts/experiments/launch_language_vision.py) | `launch` | 指定 GPU 启动独立服务；冻结模型源码并存 hash、Git 状态、PID/端口；路径通过参数/环境变量传递。 |
| [scripts/experiments/run_language_vision.py](/data/zjc/workspace/ContextFlow/scripts/experiments/run_language_vision.py) | `main` | 每个服务串行运行一组任务，不同 GPU 并行；先 0–4，再 5–24；禁止覆盖既有回合，子任务失败则该阶段不标记成功。 |
| [scripts/experiments/audit_language_vision.py](/data/zjc/workspace/ContextFlow/scripts/experiments/audit_language_vision.py) | `audit` | 独立核对初始状态、示范、噪声、语言对照；正常图像与历史整条轨迹相同；冻结前动作相同、冻结后图像不变；遮挡像素/面积与事件统计一致。 |
| [scripts/experiments/summarize_language_vision.py](/data/zjc/workspace/ContextFlow/scripts/experiments/summarize_language_vision.py) | `summarize` | 行为汇总、同观测动作差异、配对成功/失败转换、结果报告、视频浏览器。 |
| [scripts/experiments/plot_language_vision_attention.py](/data/zjc/workspace/ContextFlow/scripts/experiments/plot_language_vision_attention.py) | `build` | 使用实际遮挡/冻结输入制作 attention 辅助图；标出相机绝对权重，颜色在各图内单独放大。 |
| [scripts/experiments/LANGUAGE_VISION.md](/data/zjc/workspace/ContextFlow/scripts/experiments/LANGUAGE_VISION.md) | 方法与命令 | 定义干预、行为指标、限制和复现步骤。 |

## 如何与其他实验隔离

- 服务使用 `provenance/frozen/src/openpi` 的模型源码。各服务启动时都核对历史输出，结束时再次核对参数 hash。
- 请求中的 task_index 固定为 25，仅作采样/示范协议锚点；自然语言由 prompt 单独控制。协议的 correct/wrong 字段相对锚点定义，分析使用明确的 demo / language 字段。
- 实验 3 的三条指令都保持同一示范图像、state、action 和噪声；空字符串仍经过原 tokenizer，未人为清零 embedding。
- 实验 4 不修改场景物体、示范或 state。冻结只作用于当前双相机图像。每个条件的原始动作和物理状态都保留。
- 报告、视频、模型副本、数据均放在本次 logs 目录；没有执行 Git 提交或推送。

## 遇到的问题与处理

1. 原始牛奶基线的初始状态 0 接触牛奶但未夹持。保留失败回合；以首次接触物体确定遮挡对象，并标记 contact_without_grasp，而非虚构成功抓取。
2. 腕部近景有时放不下与物体框同尺寸的纯背景矩形。优先使用矩形；无可行矩形时，选择相同像素数的纯背景区域，记录形状差异。
3. 更近时，全部背景面积可能小于物体框。目标与背景遮挡共用可用背景面积上限；优先遮挡目标真实像素，明确记录部分遮挡。统一规则后重跑视觉试验；失败和旧规则尝试保留在 failed_attempts，不混入最终汇总。

4. 番茄酱示范的初始状态 12 基线实际抓起了巧克力布丁。原先只允许牛奶/番茄酱对象的保护检查停止了扩展。修正为允许六个实际场景物体，并对这个回合使用布丁作为遮挡/冻结目标；添加回归测试，单独报告该例外及其余 24 个回合的敏感性结果。

## 验证

具体通过数量以 `audit_full_25.json`、各服务的 `READY.json / FINISHED.json` 和最终结果报告为准。新工具的 8 个针对性测试及 Python 编译检查已通过。已通过 390 种页面选择组合与链接检查、播放/暂停/跳转逻辑的 Node DOM 模拟，以及 28 个代表视频的 14 秒时长和 20/4 fps 检查。没有进行真实浏览器交互测试。40 张 attention 图中已人工检查代表性图像。


最终验证：350 个回合、50 条历史正常图像轨迹逐动作/状态完全一致、50 条冻结轨迹触发前动作一致及触发后图像恒定、270 份 attention 记录通过概率与 mask 检查。模型参数在所有服务开始和结束时 hash 一致。
