#!/usr/bin/env bash
set -e
source scripts/activate_env.sh train
source quickstart.sh
root="$1" mode="$2"
export CUDA_VISIBLE_DEVICES="$3"
QS_LOG_DIR="$root/processes/$mode"
mkdir -p "$QS_LOG_DIR"
QS_HOST=127.0.0.1 QS_PORT="$4"
QS_PIDS=()
trap qs_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
qs_port_free
qs_start server "$QS_ROOT/.venv/bin/python" -u scripts/serve_policy.py --port "$QS_PORT" policy:checkpoint --policy.config ContextFlow --policy.dir /data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999 --policy.inference-dtype float32 --policy.context-ablation --policy.local-files-only
qs_wait_server "$QS_PID"
source scripts/activate_env.sh libero
if [[ "$mode" == correct ]]; then
  "$QS_ROOT/examples/libero/.venv/bin/python" -u -m examples.libero.context_ablation --port "$QS_PORT" --num-trials-per-task 1 --max-steps 5 --output-dir "$root/smoke" --verify-interventions > "$QS_LOG_DIR/smoke.log" 2>&1
fi
"$QS_ROOT/examples/libero/.venv/bin/python" -u -m examples.libero.context_ablation --port "$QS_PORT" --mode "$mode" --num-trials-per-task 25 --output-dir "$root" --verify-interventions > "$QS_LOG_DIR/evaluation.log" 2>&1
