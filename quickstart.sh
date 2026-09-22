#!/usr/bin/env bash
# Interactive launcher for the environments described in README.md.

QS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QS_DRY_RUN=0

qs_ask() {
    local qs_input_value
    read -r -p "$2 [$3]: " qs_input_value || return 1
    printf -v "$1" '%s' "${qs_input_value:-$3}"
}

qs_number() {
    local answer
    while qs_ask answer "$2" "$3"; do
        if [[ "$answer" =~ ^[0-9]{1,5}$ ]] && ((10#$answer >= $4 && 10#$answer <= $5)); then
            printf -v "$1" '%d' "$((10#$answer))"
            return 0
        fi
        printf '请输入 %s 到 %s 之间的整数。\n' "$4" "$5"
    done
    return 1
}

qs_activate() {
    # shellcheck source=scripts/activate_env.sh
    source "$QS_ROOT/scripts/activate_env.sh" "$1"
}

qs_require() {
    local path
    for path in "$@"; do
        if [[ ! -e "$path" ]]; then
            printf '缺少文件或目录：%s\n请先检查 README.md 中的安装和资源路径。\n' "$path" >&2
            return 1
        fi
    done
}

qs_dataset() {
    qs_require "$LEROBOT_HOME/physical-intelligence/libero/meta/info.json" \
        "$LEROBOT_HOME/physical-intelligence/libero/meta/tasks.jsonl" \
        "$LEROBOT_HOME/physical-intelligence/libero/meta/episodes.jsonl" \
        "$LEROBOT_HOME/physical-intelligence/libero/data"
}

qs_gpu() {
    local gpu
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    while qs_ask gpu '使用的 GPU 编号（多卡用逗号分隔，如 0,1）' "${CUDA_VISIBLE_DEVICES:-0}"; do
        if [[ "$gpu" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            export CUDA_VISIBLE_DEVICES="$gpu"
            return 0
        fi
        printf '请输入数字编号，例如 0 或 0,1。\n'
    done
    return 1
}

qs_checkpoint() {
    qs_ask QS_CHECKPOINT 'ContextFlow 检查点目录' "$QS_ROOT/checkpoints/ContextFlow/ContextFlow_run1/19999" || return
    if [[ "$QS_CHECKPOINT" == '~/'* ]]; then QS_CHECKPOINT="$HOME/${QS_CHECKPOINT:2}"; fi
    qs_require "$QS_CHECKPOINT/params/manifest.ocdbt" "$QS_CHECKPOINT/params/_METADATA" \
        "$QS_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
}

qs_eval_options() {
    local suite
    printf '\n任务集：1 Spatial  2 Object  3 Goal  4 LIBERO-10  5 Spatial + Object\n'
    qs_number suite '选择任务集' 1 1 5 || return
    case "$suite" in
        1) QS_SUITES=(libero_spatial) ;; 2) QS_SUITES=(libero_object) ;;
        3) QS_SUITES=(libero_goal) ;; 4) QS_SUITES=(libero_10) ;;
        5) QS_SUITES=(libero_spatial libero_object) ;;
    esac
    qs_number QS_TRIALS '每个未见任务的评估次数（快速检查用 1，README 用 50）' 1 1 50 || return
    local prefix=evaluation_contextflow names
    if [[ "$QS_TRIALS" == 50 && "$suite" =~ ^[125]$ ]]; then prefix=reproduction_table1_contextflow; fi
    names=$(IFS=_; printf '%s' "${QS_SUITES[*]//libero_/}")
    QS_RUN_NAME="${prefix}_${names}_${QS_TRIALS}trials"
}

qs_print_command() {
    printf '\n运行：'
    printf '%q ' "$@"
    printf '\n'
}

qs_logs() {
    if [[ -z "${QS_LOG_DIR:-}" ]]; then
        local base candidate sequence=1
        mkdir -p "$QS_ROOT/logs/quickstart" || return
        base="$QS_ROOT/logs/quickstart/${QS_RUN_NAME:-contextflow}_$(date +%Y%m%d_%H%M%S)"
        candidate="$base"
        while ! mkdir -- "$candidate" 2>/dev/null; do
            [[ -d "$candidate" ]] || { printf '无法创建运行目录：%s\n' "$candidate" >&2; return 1; }
            ((sequence += 1))
            printf -v candidate '%s_run%02d' "$base" "$sequence"
        done
        QS_LOG_DIR="$candidate"
        printf '本次日志、评估结果和视频：%s\n' "$QS_LOG_DIR"
    fi
}

# Each launched command has its own process group. Cleanup includes its workers,
# and only touches groups created by this invocation.
qs_stop() {
    local pid="$1" attempt i
    if kill -0 -- "-$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        for attempt in {1..20}; do
            kill -0 -- "-$pid" 2>/dev/null || break
            sleep 0.25
        done
        if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
    fi
    wait "$pid" 2>/dev/null || true
    for i in "${!QS_PIDS[@]}"; do
        if [[ "${QS_PIDS[$i]}" == "$pid" ]]; then unset 'QS_PIDS[i]'; fi
    done
}

qs_cleanup() {
    local pid
    for pid in "${QS_PIDS[@]}"; do qs_stop "$pid"; done
}

qs_start() {
    local label="$1"
    shift
    qs_logs || return
    qs_print_command "$@"
    setsid "$@" < /dev/null > "$QS_LOG_DIR/$label.log" 2>&1 &
    QS_PID=$!
    QS_PIDS+=("$QS_PID")
    printf 'PID=%s，日志：%s/%s.log\n' "$QS_PID" "$QS_LOG_DIR" "$label"
}

qs_run() {
    local label="$1" pid follower status
    shift
    if ((QS_DRY_RUN)); then qs_print_command "$@"; return 0; fi
    qs_start "$label" "$@" || return
    pid="$QS_PID"
    tail -n +1 --pid="$pid" -f "$QS_LOG_DIR/$label.log" &
    follower=$!
    wait "$pid"
    status=$?
    wait "$follower" 2>/dev/null || true
    qs_stop "$pid"
    return "$status"
}

qs_server_command() {
    QS_SERVER=("$QS_ROOT/.venv/bin/python" -u "$QS_ROOT/scripts/serve_policy.py"
        --port "$QS_PORT" policy:checkpoint --policy.inference_dtype=float32
        --policy.config=ContextFlow --policy.dir="$QS_CHECKPOINT")
}

qs_port_free() {
    "$QS_ROOT/.venv/bin/python" - "$QS_PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", int(sys.argv[1])))
    except OSError as exc:
        print(f"端口 {sys.argv[1]} 不可用：{exc}。请选择其他端口。", file=sys.stderr)
        sys.exit(1)
PY
}

qs_probe() {
    "$QS_ROOT/.venv/bin/python" - "$QS_HOST" "$QS_PORT" <<'PY'
import sys
from websockets.sync.client import connect
from openpi_client import msgpack_numpy
try:
    with connect(f"ws://{sys.argv[1]}:{sys.argv[2]}", open_timeout=2, close_timeout=1,
                 compression=None, max_size=None) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=2))
        if not isinstance(metadata, dict):
            sys.exit(1)
except Exception:
    sys.exit(1)
PY
}

qs_wait_server() {
    local pid="$1" started=$SECONDS deadline=$((SECONDS + 1800))
    printf '等待策略服务就绪（模型和数据加载可能需要几分钟）。\n日志：%s/server.log\n' "$QS_LOG_DIR"
    while ((SECONDS < deadline)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            if [[ -t 1 ]]; then printf '\n'; fi
            printf '策略服务提前退出，日志末尾如下：\n' >&2
            tail -n 30 "$QS_LOG_DIR/server.log" >&2
            return 1
        fi
        if qs_probe && kill -0 "$pid" 2>/dev/null; then
            if [[ -t 1 ]]; then printf '\n'; fi
            printf '策略服务已就绪：%s:%s\n' "$QS_HOST" "$QS_PORT"
            return 0
        fi
        if [[ -t 1 ]]; then
            printf '\r等待策略服务就绪… 已等待 %d 秒' "$((SECONDS - started))"
        fi
        sleep 3
    done
    if [[ -t 1 ]]; then printf '\n'; fi
    printf '等待服务超过 30 分钟，请检查 %s/server.log\n' "$QS_LOG_DIR" >&2
    return 1
}

qs_evaluate() {
    local suite output
    qs_activate libero || return
    if ((QS_DRY_RUN)); then output="${QS_LOG_DIR:-$QS_ROOT/logs/quickstart/${QS_RUN_NAME}_<时间戳>}";
    else qs_logs || return; output="$QS_LOG_DIR"; fi
    for suite in "${QS_SUITES[@]}"; do
        qs_run "$suite" "$QS_ROOT/examples/libero/.venv/bin/python" -u \
            "$QS_ROOT/examples/libero/main_incontext.py" --host "$QS_HOST" --port "$QS_PORT" \
            --task-suite-name "$suite" --num-trials-per-task "$QS_TRIALS" \
            --results-out-path "$output/$suite.json" --video-out-path "$output/videos/$suite" || return
    done
}

qs_managed_evaluate() {
    local root suite server_pid
    local -a suites=("${QS_SUITES[@]}") QS_SUITES=()
    if ((QS_DRY_RUN)); then
        root="$QS_ROOT/logs/quickstart/${QS_RUN_NAME}_<时间戳>"
    else
        qs_logs || return
        root="$QS_LOG_DIR"
        "$QS_ROOT/.venv/bin/python" - "$root" "$QS_CHECKPOINT" "$QS_PORT" "$QS_TRIALS" "${suites[@]}" <<'PY'
import datetime
import json
import os
from pathlib import Path
import sys
root, checkpoint, port, trials, *suites = sys.argv[1:]
Path(root, "run.json").write_text(json.dumps({
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "config": "ContextFlow", "checkpoint": str(Path(checkpoint).resolve()),
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "port": int(port),
    "suites": suites, "num_trials_per_task": int(trials), "client_seed": 7,
    "policy_rng_initial_seed_per_suite": 0, "fresh_server_per_suite": True,
    "inference_dtype": "float32", "video_fps": 20,
    "results": {suite: f"{suite}/{suite}.json" for suite in suites},
}, indent=2))
PY
        [[ $? == 0 ]] || return 1
    fi
    for suite in "${suites[@]}"; do
        QS_LOG_DIR="$root/$suite"
        QS_SUITES=("$suite")
        printf '\n任务集：%s；使用独立的新策略服务。\n结果：%s/%s.json\n' "$suite" "$QS_LOG_DIR" "$suite"
        qs_activate train || return
        if ((QS_DRY_RUN)); then
            qs_print_command "${QS_SERVER[@]}"
            qs_evaluate || return
            continue
        fi
        mkdir -p -- "$QS_LOG_DIR" || return
        qs_port_free || return
        qs_start server "${QS_SERVER[@]}" || return
        server_pid="$QS_PID"
        qs_wait_server "$server_pid" || return
        qs_evaluate || return
        qs_stop "$server_pid"
    done
    QS_LOG_DIR="$root"
    if ((QS_DRY_RUN)); then printf '\n预览完成，计划结果目录：%s\n' "$root";
    else printf '\n全部任务集已完成，结果目录：%s\n' "$root"; fi
}

qs_check() {
    local failed=0 mode
    for mode in train libero; do
        (
            qs_activate "$mode" || exit
            printf '\n[%s] Python: %s\n' "$mode" "$(command -v python)"
            python --version
            if [[ "$mode" == train ]]; then
                python -c 'import jax, flax, orbax.checkpoint, lerobot; print("JAX / Flax / Orbax / LeRobot 导入通过")'
            else
                python -c 'import torch, mujoco, libero, openpi_client; print("PyTorch / MuJoCo / LIBERO / 客户端导入通过")'
            fi
        ) || failed=1
    done
    qs_activate train || return
    printf '\n数据目录：%s\n模型缓存：%s\n' "$LEROBOT_HOME" "$OPENPI_DATA_HOME"
    qs_dataset || failed=1
    qs_require "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params/manifest.ocdbt" \
        "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" \
        "$QS_ROOT/checkpoints/ContextFlow/ContextFlow_run1/19999/params/manifest.ocdbt" || failed=1
    if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || failed=1;
    else printf '未找到 nvidia-smi。\n'; failed=1; fi
    if ((failed == 0)); then printf '\n基础检查通过。\n'; fi
    return "$failed"
}

qs_action() (
    local action="$1" mode exp batch
    local QS_LOG_DIR='' QS_PID='' QS_HOST=127.0.0.1 QS_PORT=8000 QS_CHECKPOINT='' QS_TRIALS=1 QS_RUN_NAME=contextflow
    local -a QS_PIDS=() QS_SERVER=() QS_SUITES=()
    trap qs_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    case "$action" in
        1) qs_check ;;
        2|3)
            mode=train; [[ "$action" == 3 ]] && mode=libero
            qs_activate "$mode" || return
            printf '进入 %s 环境终端；输入 exit 返回菜单。\n' "$mode"
            if ((QS_DRY_RUN)); then return 0; fi
            bash --noprofile --rcfile <(printf 'source %q %q\n' "$QS_ROOT/scripts/activate_env.sh" "$mode") -i
            ;;
        4|6)
            QS_RUN_NAME=serve_contextflow
            if [[ "$action" == 6 ]]; then qs_eval_options || return; fi
            qs_activate train && qs_dataset && qs_checkpoint && qs_gpu || return
            qs_number QS_PORT '策略服务端口' 8000 1 65535 || return
            qs_require "$QS_ROOT/examples/libero/.venv/bin/python" || return
            qs_server_command
            if [[ "$action" == 6 ]]; then qs_managed_evaluate; return; fi
            if ((QS_DRY_RUN)); then
                qs_print_command "${QS_SERVER[@]}"
                return
            fi
            qs_port_free || return
            qs_run server "${QS_SERVER[@]}"
            ;;
        5)
            qs_activate libero && qs_dataset && qs_gpu || return
            qs_ask QS_HOST '已有策略服务的地址' 127.0.0.1 || return
            qs_number QS_PORT '策略服务端口' 8000 1 65535 || return
            qs_eval_options || return
            if (( ! QS_DRY_RUN )) && ! qs_probe; then
                printf '未能连接策略服务 %s:%s，请先启动服务。\n' "$QS_HOST" "$QS_PORT" >&2
                return 1
            fi
            qs_evaluate
            ;;
        7)
            QS_RUN_NAME=norm_stats_contextflow
            qs_activate train && qs_dataset || return
            qs_run norm_stats "$QS_ROOT/.venv/bin/python" -u "$QS_ROOT/scripts/compute_norm_stats.py" --config-name ContextFlow
            ;;
        8)
            qs_activate train && qs_dataset || return
            qs_require "$QS_ROOT/assets/ContextFlow/physical-intelligence/libero/norm_stats.json" || {
                printf '训练前请先选择 7，计算训练集的归一化统计。\n'; return 1;
            }
            qs_require "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params/manifest.ocdbt" || return
            qs_gpu || return
            qs_ask exp '新训练实验名' "run_$(date +%Y%m%d_%H%M%S)" || return
            if [[ ! "$exp" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
                printf '实验名只支持英文字母、数字、下划线和横线，且以字母或数字开头。\n'; return 1;
            fi
            if [[ -e "$QS_ROOT/checkpoints/ContextFlow/$exp" ]]; then
                printf '实验目录已存在，请换一个实验名：checkpoints/ContextFlow/%s\n' "$exp"; return 1;
            fi
            qs_number batch '训练 batch size' 32 1 65535 || return
            export WANDB_MODE="${WANDB_MODE:-offline}"
            QS_RUN_NAME="training_contextflow_${exp}"
            printf '开始 ContextFlow 训练，共 20,000 步；W&B 模式：%s。\n' "$WANDB_MODE"
            qs_run train "$QS_ROOT/.venv/bin/python" -u "$QS_ROOT/scripts/train.py" ContextFlow --exp-name "$exp" --batch-size "$batch"
            ;;
        *) printf '请输入菜单中的编号。\n'; return 1 ;;
    esac
)

qs_main() {
    local choice status
    case "${1:-}" in
        --dry-run) QS_DRY_RUN=1 ;;
        --help|-h) printf '用法：bash quickstart.sh [--dry-run]\n--dry-run：预览所选启动命令，不启动训练、策略服务或模拟器。\n'; return 0 ;;
        '') ;;
        *) printf '未知参数：%s\n' "$1" >&2; return 2 ;;
    esac
    cd -- "$QS_ROOT" || return
    command -v setsid >/dev/null || { printf '缺少 setsid（通常由 util-linux 提供）。\n' >&2; return 1; }
    trap ':' INT
    while true; do
        printf '\n========== ContextFlow 启动菜单 ==========\n'
        ((QS_DRY_RUN == 0)) || printf '当前为预览模式，不启动计算任务。\n'
        printf '%s\n' '1  检查环境、资源路径和 GPU' '2  进入训练 / 策略服务环境终端' \
            '3  进入 LIBERO 模拟器环境终端' '4  启动 ContextFlow 策略服务' \
            '5  连接已有策略服务，运行 LIBERO 评估' '6  一键评估：启动服务 → 等待就绪 → 评估 → 关闭服务' \
            '7  计算训练集归一化统计' '8  启动新的 ContextFlow 训练' '0  退出'
        read -r -p '输入编号并回车：' choice || break
        [[ "$choice" != 0 ]] || break
        qs_action "$choice"
        status=$?
        if ((status != 0)); then printf '\n本次操作结束（退出码 %s），已返回菜单。\n' "$status"; fi
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -o pipefail
    qs_main "$@"
fi
