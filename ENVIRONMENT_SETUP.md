# 本机环境配置记录

配置日期：2026-09-21。依据根目录 README 和 `examples/libero/LIBERO_README.md`。

## 使用环境

在项目根目录执行交互式菜单：

```bash
bash quickstart.sh
```

输入编号并回车即可选择：环境检查、进入两个环境的终端、启动策略服务、连接已有服务评估、
一键启动服务并评估、计算训练归一化统计或开始新训练。进入环境终端后输入 `exit` 返回菜单。

首次运行建议选择 **1** 检查环境，再选择 **6** 一键评估。先选择任务集和次数，再确认检查点、
GPU 和端口；默认使用 GPU 0、已下载的 ContextFlow 发布权重、8000 端口、Spatial 未见任务，
每个任务评估 1 次。若已设置 `CUDA_VISIBLE_DEVICES`，菜单优先使用该设置。
README 的正式评估次数为每任务 50 次。

复现 Table 1 的四个未见任务：运行 `bash quickstart.sh` 后依次输入 **6 → 5 → 50**，
然后在检查点、GPU、端口提示处各按一次回车（或输入自己的设置），共评估 200 次。

一键评估为每个任务集启动独立的新策略服务，确认 WebSocket 可用后启动模拟器，最多等待 30 分钟。
一个任务集完成后关闭服务，再启动下一个，保持各任务集策略随机状态与已完成的复现一致。
评估结束、失败或按 Ctrl+C 中断时，会关闭本次启动的服务及其子进程。
选择 **5** 连接已有服务时，菜单不会关闭那个服务。
Table 1 复现目录形如：

```text
logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_YYYYMMDD_HHMMSS/
  run.json
  libero_spatial/libero_spatial.json
  libero_spatial/server.log
  libero_spatial/videos/libero_spatial/<任务指令>/
  libero_object/libero_object.json
  libero_object/server.log
  libero_object/videos/libero_object/<任务指令>/
```

其他评估目录也包含任务集和次数；同秒重复启动时使用 `_run02` 等序号，避免覆盖。
`run.json` 记录检查点、GPU、种子、次数和结果位置。视频按仿真控制频率 **20 FPS** 原速导出。
视频按任务指令分别存放，文件夹名使用英文指令并将空格替换为下划线；选择 50 次时，每个指令文件夹内有 50 个视频。

选择 **8** 前需先选择 **7** 计算训练用归一化统计；训练使用新实验名，已有实验目录会被拒绝。
训练默认使用 W&B 离线记录，可提前设置 `WANDB_MODE=online` 使用在线记录。
菜单使用已经安装的依赖，不会每次启动都重新安装。
如果只想预览操作和生成的启动命令：

```bash
bash quickstart.sh --dry-run
```

也可以继续手动激活环境。从项目根目录，在 Bash 中执行：

```bash
# 训练、统计脚本、策略服务
source scripts/activate_env.sh train

# 在另一个终端运行 LIBERO 模拟器
source scripts/activate_env.sh libero
```

主环境位于 `.venv`（Python 3.11.16）；模拟器环境位于
`examples/libero/.venv`（Python 3.8.20）。不要把模拟器的旧依赖安装进主环境。
激活后 `uv` 和 `python` 都可以直接使用。模拟器环境建议直接运行 `python`，
或者使用 README 中带 `--no-project --python examples/libero/.venv/bin/python` 的命令。

激活脚本设置 EGL 渲染、float32 JAX 矩阵乘法精度，以及客户端需要的 `PYTHONPATH`。
JAX 默认关闭显存预分配，避免环境验证时占用整张 GPU；未限制可见 GPU 数量。

工具缓存放在项目内，数据和权重使用指定的共享目录（已经设置对应变量时保留用户设置）：

| 内容 | 路径 |
| --- | --- |
| uv 工具 | `.cache/uv-bootstrap/bin/uv` |
| uv 包缓存、Python 解释器 | `.cache/uv`、`.cache/python` |
| Hugging Face 缓存 | `.cache/huggingface` |
| LeRobot 数据集 (`LEROBOT_HOME`) | `/data/zjc/workspace/datasets` |
| 数据处理缓存 (`HF_DATASETS_CACHE`) | `/data/zjc/workspace/datasets/.cache/huggingface/datasets` |
| OpenPI 权重缓存 (`OPENPI_DATA_HOME`) | `/data/zjc/workspace/models/openpi` |
| LIBERO 路径配置 | `.cache/libero/config.yaml` |
| 检查日志 | `logs/environment_setup/` |

激活脚本继承终端已有的代理配置，不会修改 Windows VPN、SSH 配置或全局 Git 配置。

## 问题与处理

1. **没有 uv，默认 Python 不符合 README。** 在项目内安装 uv 0.12.17，
   使用 uv 下载 Python 3.11.16 和 3.8.20，分别创建两个环境。
2. **官方 PyPI 经代理下载 CUDA 大包时多次重试。** 降低并发为 4，
   将 `UV_HTTP_TIMEOUT` 设置为 1800 秒。从原始 `uv.lock` 导出精确版本与哈希，
   经阿里云镜像安装，再运行 `uv sync --locked` 核对并完成主项目可编辑安装。
   根目录的 `pyproject.toml` 和 `uv.lock` 未修改。
   参数含义见 [uv 官方文档](https://docs.astral.sh/uv/reference/environment/)。
3. **镜像缺少旧版 pyav 14.0.1。** 使用 `uv.lock` 保存的官方 Python 3.11 / Linux
   wheel URL 及其对应 SHA256，随后由 `uv sync --locked` 恢复为锁文件要求的包来源。
4. **LIBERO 的 PyTorch 1.11.0+cu113 wheel 约 1.5 GiB，单流下载不稳定。**
   使用官方 PyTorch URL 分块下载，每块 32 MiB、6 个并发，可复用已完成的块；
   实际捕获到 `IncompleteRead` 连接中断，分块重试后恢复。49 块下载完成后，
   合并文件通过官方索引提供的 SHA256 校验：
   `b6a799bdb6ee3d914e5e62bddb4276d4a10248c1af4f2d217738e5f9ee27485b`。
   脚本及下载文件保存在 `.cache/libero-torch-download`
   和 `.cache/download_libero_torch.py`，日志见 `torch-download.log`。
5. **LIBERO 首次导入会交互式询问路径。** 预先生成项目内的配置，明确指向子模块的
   BDDL、初始状态、模拟器资源路径，通过 `LIBERO_CONFIG_PATH` 使用。
6. **沙箱无法访问 NVIDIA 驱动。** 在宿主机完成硬件与 GPU 验证：8 张 A100-SXM4-80GB，
   驱动 575.57.08，NVIDIA EGL 库已存在，无需安装系统驱动。
7. **真实策略服务加载完成后因主机名解析失败退出。** 本机主机名没有 DNS / hosts 记录，
   `serve_policy.py` 原来仅为打印 IP 调用了 `socket.gethostbyname`。改为直接记录实际监听地址
   `0.0.0.0:<port>`，服务启动不再依赖主机名解析；已增加回归测试并完成真实模型评估。

## 验证记录

- 主环境：233 个包，`uv pip check` 通过。
- 交互式菜单：Bash 语法、两个环境终端切换、数字输入校验和命令预览通过。
  使用轻量 WebSocket 测试服务验证就绪检测、端口冲突、服务提前退出、客户端失败和
  SIGINT / SIGTERM 清理；已有独立进程不受影响。记录见 `quickstart-check.json`。
- JAX 0.5.0 GPU JIT 矩阵乘法通过。
- PyTorch 2.6.0+cu124 GPU 矩阵乘法通过。
- ContextFlow Perceiver 小规模前向计算通过，训练配置可加载。
- LeRobot 数据集模块导入通过。
- 已运行的配置、任务划分、归一化和客户端测试：29 passed、1 manual test deselected。
- `scripts/train.py`、`scripts/serve_policy.py`、`scripts/compute_norm_stats.py` 的帮助命令通过。
- LIBERO 环境：127 个包，`uv pip check` 通过；两份原始 requirements 的固定版本未改动。
- 在 Python 3.8 模拟器环境中另外运行客户端测试：21 passed。
- LIBERO：PyTorch 1.11.0+cu113 可识别 CUDA，robosuite 1.4.1、MuJoCo 3.2.3、NumPy 1.22.4。
- `examples/libero/main_incontext.py --help` 通过。
- 实际运行 `libero_spatial` 的一个任务：reset、初始状态加载、5 步动作均通过。
- EGL 离屏渲染通过：两路相机均输出 128×128×3 图像，像素标准差约 49.9 / 62.5，图像内容正常。
- FFmpeg 视频导出通过，见 `logs/environment_setup/libero_smoke.mp4`；
  相机图像为同目录下 `agentview_image.png` 和 `robot0_eye_in_hand_image.png`。

保留了旧版 Gym 的维护提醒、Numba 的 NumPy 弃用提醒，以及 robosuite 可选
`macros_private.py` 配置提醒。它们没有影响本次验证；未为消除提示而更换 README 固定的依赖。

## 安装与验证的复用命令

主环境已经安装完毕，日常使用不需要重新下载依赖：

```bash
source scripts/activate_env.sh train
uv sync --locked
uv pip check
CUDA_VISIBLE_DEVICES=0 python .cache/check_main_environment.py
```

本机已保留全部 LIBERO 下载缓存，如需重新执行依赖安装：

```bash
source scripts/activate_env.sh train
uv pip install --offline --python examples/libero/.venv/bin/python \
  .cache/libero-torch-download/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl \
  -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install --offline --python examples/libero/.venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  -e packages/openpi-client -e third_party/libero

source scripts/activate_env.sh libero
uv pip check --python examples/libero/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 python .cache/check_libero_environment.py
```

这些复用命令使用本机已下载缓存；迁移到其他机器时需要重新安装 uv / Python、创建虚拟环境，
重新生成 LIBERO 绝对路径配置，并下载依赖。详细包版本保存在
`logs/environment_setup/main-packages.txt` 和 `libero-packages.txt`。

## 数据和权重

按 README 的 LIBERO 训练和评估流程准备以下资源：

| 资源 | 本地路径 | 原始文件大小 |
| --- | --- | --- |
| LIBERO 数据集 | `/data/zjc/workspace/datasets/physical-intelligence/libero` | 32.54 GiB，1,699 个文件 |
| π₀ base 初始权重 | `/data/zjc/workspace/models/openpi/openpi-assets/checkpoints/pi0_base` | 11.19 GiB，33 个文件 |
| ContextFlow 发布权重 | `/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999` | 6.87 GiB，20 个文件 |
| PaliGemma tokenizer | `/data/zjc/workspace/models/openpi/big_vision/paligemma_tokenizer.model` | 4.07 MiB |

项目内的 `checkpoints/ContextFlow` 是指向上述共享模型目录的软链接，README 中的
`checkpoints/ContextFlow/ContextFlow_run1/19999` 路径可以直接使用。
重新执行 `source scripts/activate_env.sh train` 或 `source scripts/activate_env.sh libero`
即可启用这些路径。原有 `models/pi05_base` 和其他数据集保持原样。

### 下载来源与问题处理

- **数据格式版本：** 使用 `physical-intelligence/libero` 的 `v2.0`，固定提交
  `9dfa69510ea9e1613fc54112bc706444b686a231`，与当前锁定的 LeRobot 版本匹配。
  包含 1,693 个 episode、273,465 帧和 40 个任务；图像嵌在 Parquet 内，没有额外视频文件。
  经 Hugging Face 镜像下载，每个 LFS 文件核对原始 SHA256，其他文件核对 Git blob SHA1。
  已写入 Hugging Face 本地下载元数据，避免重复传输。
- **S3 / Google 下载较慢且连接中断：** 采用分块、断点续传和自动重试，保持 TLS 校验。
  π₀ 使用内容相同的 `oldTOM/pi0_base` 镜像（提交
  `88e794fb312e073e9f09d2d7e10737a92c4aadcb`）加速；33 个文件的路径和大小均与官方
  `s3://openpi-assets/checkpoints/pi0_base` 清单一致，合并后全部通过官方 S3 ETag 校验，
  包括 multipart ETag。
- **Drive 连接器单文件大小限制：** 转为读取 README 公开 Google Drive 文件的下载链接，
  16 MiB 分块续传，20 个最终文件均通过 Google 响应头提供的 CRC32C 校验。
  下载约 97% 时曾持续超时并耗尽重试；确认代理重新可用后刷新下载链接，
  复用已完成分块与原始校验信息，最终补齐缺失内容。
- **Tokenizer：** 从官方 Google Storage 下载，通过官方 MD5 校验；SentencePiece 实际编码通过，
  词表大小 257,152。

以上资源均已下载完成，原始文件合计约 50.60 GiB。
下载清单、续传进度及校验日志位于 `logs/environment_setup/`。
续传脚本为 `.cache/download_assets.py` 和 `.cache/download_contextflow.py`。
实际读取 LIBERO 的 episode 0 已通过：214 帧，双路图像均为 `[3, 256, 256]`，
状态为 `[8]`，动作为 `[7]`，8 个 held-out 任务均存在。
π₀ 使用项目的 `restore_params` / Orbax 在宿主机完整恢复成功：50 个参数数组、
3,238,048,528 个参数，抽样数值均有限；报告见 `pi0-weight-verification.json`。
ContextFlow 同样完整恢复成功：179 个参数数组、2,170,151,696 个参数，抽样数值均有限，
且通过当前 `ContextFlow` 配置的参数形状检查；报告见 `contextflow-weight-verification.json`。
两份权重的临时下载分块已清理。完整资源清单见 `download-completion.json`。

环境准备完成后，已于 2026-09-21 实际运行发布 ContextFlow 检查点，对 Spatial 和 Object
的四个未见任务各评估 50 次：共 151 / 200 成功，成功率 75.5%，无运行异常。
逐任务结果、论文对照及限制见 [EVALUATION_REPORT.md](EVALUATION_REPORT.md)。没有启动模型训练。
重新训练前仍需按 README 运行
`python scripts/compute_norm_stats.py --config-name ContextFlow`；发布权重自带评估用的
`assets/physical-intelligence/libero/norm_stats.json`。
