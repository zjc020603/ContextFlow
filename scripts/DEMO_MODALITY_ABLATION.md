# Paired demonstration state and state/action ablations

Two additional conditions extend the previous action-only pilot to a 2×2
factorial comparison with demo images always retained:

| | Demo action present | Demo action zeroed |
|---|---|---|
| Demo state present | Historical full demonstration | Historical action-only ablation |
| Demo state zeroed | New `state` ablation | New `state_action` ablation |

Each new condition runs 2 layouts × 2 languages × correct/wrong × 5 initial
states = 40 episodes in the fixed tomato-sauce scene. Use the matched first
five episodes of the historical position and action-only experiments, not the
25-trial aggregate. Raw videos and full object trajectories are retained.

The intervention zeros the compressed state tokens, or state and action tokens,
immediately before Gemma. Demo images, current observation, language, robot
state, token slots, masks and position IDs stay unchanged. This is potentially
OOD, not a trained missing-modality embedding. The normal model and prior
experiment entry points are unchanged.

Start a server in the model environment (fresh output directory):

```bash
source scripts/activate_env.sh train
export CUDA_VISIBLE_DEVICES=2
RUN=logs/modality_ablation/my_run
python -m scripts.serve_demo_modality_ablation \
  --ablation state --port 8124 --output "$RUN/zero_state/server"
```

Wait for `server/READY.json`, then start the paired simulator:

```bash
source scripts/activate_env.sh libero
export CUDA_VISIBLE_DEVICES=3
RUN=logs/modality_ablation/my_run
python -m examples.libero.demo_modality_ablation \
  --ablation state --port 8124 --trials 5 --output "$RUN/zero_state/rollouts"
python - <<'PY'
from openpi_client.websocket_client_policy import WebsocketClientPolicy
print(WebsocketClientPolicy('127.0.0.1',8124).infer({'finalize_demo_modality_ablation':True}))
PY
```

Repeat with `--ablation state_action`, output `zero_state_action`, and a distinct
port (e.g. 8125). If running simultaneously, use separate GPU pairs (e.g. 4/5)
and separate server processes. Stop each dedicated server after finalization.
The evaluator rejects a server with the wrong intervention metadata.

Generate the joint result/individual result reports after both finish:

```bash
source scripts/activate_env.sh train
RUN=logs/modality_ablation/my_run
python -m scripts.summarize_demo_modality_ablation --root "$RUN"
```

Each server verifies 60 historical baseline snapshots, 12 prefix/capture cases,
eight cases of removed-modality value invariance, and four unchanged masked-no
snapshots before listening. The last four are diagnostics only. No no-context
behavioral rows enter the main four-condition comparison.

Attention is saved at environment steps 0/80/160 (flow steps 0/5/9, layers
0/9/17, all eight heads). Every recorded action must equal ordinary inference
and the executed action trace. Summaries require 80 new episodes and 240
captures. Video is 20 FPS, organized under instruction-named folders.
