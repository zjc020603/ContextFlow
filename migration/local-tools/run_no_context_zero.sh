#!/usr/bin/env bash
set -euo pipefail
cd /data/zjc/workspace/ContextFlow
source scripts/activate_env.sh libero
export CUDA_VISIBLE_DEVICES=3
RUN=logs/context_ablation/no_context_zero_keep_mask_25trials_20260923_204651
python - <<'PY'
from pathlib import Path
import os
import time
root=Path('logs/context_ablation/no_context_zero_keep_mask_25trials_20260923_204651')
for attempt in range(180):
    if (root/'server/READY.json').exists():
        print('Gated no-context server ready', flush=True)
        break
    os.kill(853170, 0) if False else None
    time.sleep(5)
else:
    raise TimeoutError('Server gate did not finish within 15 minutes')
PY
python -m examples.libero.no_context_zero --port 8123 --trials 25 --output "$RUN/rollouts"
python - <<'PY'
from openpi_client.websocket_client_policy import WebsocketClientPolicy
print(WebsocketClientPolicy('127.0.0.1',8123).infer({'finalize_no_context_zero':True}))
PY
