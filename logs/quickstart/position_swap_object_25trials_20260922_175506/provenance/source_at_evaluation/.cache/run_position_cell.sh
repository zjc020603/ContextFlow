#!/usr/bin/env bash
set -euo pipefail
source scripts/activate_env.sh train
source quickstart.sh
position_root="$1" position_layout="$2" position_mode="$3"
export CUDA_VISIBLE_DEVICES="$4"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
QS_LOG_DIR="$position_root/processes/$position_layout/$position_mode"
mkdir -p "$QS_LOG_DIR"
QS_HOST=127.0.0.1 QS_PORT="$5"
QS_PIDS=()
trap qs_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
qs_port_free
qs_start server "$QS_ROOT/.venv/bin/python" -u scripts/serve_policy.py --port "$QS_PORT" policy:checkpoint --policy.config ContextFlow --policy.dir /data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999 --policy.inference-dtype float32 --policy.context-ablation --policy.local-files-only
qs_wait_server "$QS_PID"
source scripts/activate_env.sh libero
"$QS_ROOT/examples/libero/.venv/bin/python" -u -m examples.libero.position_ablation --port "$QS_PORT" --layout "$position_layout" --mode "$position_mode" --num-trials-per-task 1 --output-dir "$position_root/smoke" --verify-interventions > "$QS_LOG_DIR/smoke.log" 2>&1
touch "$QS_LOG_DIR/smoke_complete"
position_deadline=$((SECONDS + 1800))
while [[ ! -f "$position_root/start_full_evaluation" ]]; do
    ((SECONDS < position_deadline)) || exit 2
    kill -0 "$QS_PID" || exit 3
    sleep 2
done
"$QS_ROOT/examples/libero/.venv/bin/python" -u -m examples.libero.position_ablation --port "$QS_PORT" --layout "$position_layout" --mode "$position_mode" --num-trials-per-task 25 --output-dir "$position_root" --verify-interventions > "$QS_LOG_DIR/evaluation.log" 2>&1
touch "$QS_LOG_DIR/evaluation_complete"
