#!/usr/bin/env bash
set -euo pipefail
cd /data/zjc/workspace/ContextFlow
ABLATION="$1"
SIM_GPU="$2"
SERVER_PORT="$3"
RUN="logs/modality_ablation/state_and_state_action_5trials_20260924_164440/zero_${ABLATION}"
source scripts/activate_env.sh libero
export CUDA_VISIBLE_DEVICES="$SIM_GPU"
python - "$RUN" <<'PY'
from pathlib import Path
import os,sys,time
root=Path(sys.argv[1]);pid=int((root/'server.pid').read_text())
for attempt in range(180):
    if (root/'server/READY.json').exists():
        print('Gate passed; starting paired rollouts',flush=True)
        break
    os.kill(pid,0)
    time.sleep(5)
else:
    raise TimeoutError('Server not ready after 15 minutes')
PY
python -m examples.libero.demo_modality_ablation --ablation "$ABLATION" --port "$SERVER_PORT" --trials 5 --output "$RUN/rollouts"
python - "$SERVER_PORT" <<'PY'
import sys
from openpi_client.websocket_client_policy import WebsocketClientPolicy
print(WebsocketClientPolicy('127.0.0.1',int(sys.argv[1])).infer({'finalize_demo_modality_ablation':True}))
PY
