# Demo action ablation: position-preserving pilot

This experiment zeros the 32 compressed demonstration-action tokens immediately
before Gemma. It retains their masks and slots, so other prefix tokens and
position IDs remain unchanged. It is an inference-only, potentially OOD
intervention, not a trained null embedding. Demo images and states still carry
motion information. The ordinary policy launcher is unaffected.

Prerequisites: the local checkpoint/dataset, the completed experiment-0 capture
run, and the historical position experiment. This runner is an experiment with
a mandatory historical-baseline gate, not a general policy deployment option.
The pilot intentionally uses five initial states and eight new cells (two
layouts × two languages × correct/wrong). No-context references are reused.

Run the server in a model-environment terminal with a fresh output directory:

```bash
source scripts/activate_env.sh train
export CUDA_VISIBLE_DEVICES=2
RUN=logs/action_ablation/my_action_pilot
python -m scripts.serve_demo_action_ablation \
  --output "$RUN/server" --port 8122 \
  --source logs/attention_capture/experiment0_consistency_20260923_143830 \
  --checkpoint /data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999
```

It refuses to listen until baseline checks pass: 60 historical initial
observations, 12 intervention/capture checks, eight action-value invariance
checks and four unchanged no-context checks. `READY.json` and
`GATE_PASSED.json` document readiness.

In another terminal, use the same run path:

```bash
source scripts/activate_env.sh libero
export CUDA_VISIBLE_DEVICES=3
RUN=logs/action_ablation/my_action_pilot
python -m examples.libero.demo_action_ablation \
  --port 8122 --trials 5 --output "$RUN/rollouts"
python - <<'PY'
from openpi_client.websocket_client_policy import WebsocketClientPolicy
client = WebsocketClientPolicy('127.0.0.1', 8122)
print(client.infer({'finalize_demo_action_ablation': True}))
PY
```

Finalization checks the model parameter hash again. After finalization, stop
this dedicated server using Ctrl-C. It never modifies the checkpoint.

Generate the report in the model environment:

```bash
source scripts/activate_env.sh train
RUN=logs/action_ablation/my_action_pilot
python -m scripts.summarize_demo_action_ablation \
  --root "$RUN" --server "$RUN/server" \
  --baseline logs/quickstart/position_swap_object_25trials_20260922_175506
```

The summary expects the full five-trial pilot (40 trajectories), 2240 replans,
and 120 captured observations. Smaller `--trials` values are for smoke runs
only and will not pass this full-pilot summary.

Videos use 20 FPS and instruction-named subdirectories. The evaluator checks
all scene objects for the full 280-step horizon. Every fifth control step it
replans using the existing paired seed protocol. At environment steps 0, 80,
160 the server saves raw attention, observations, complete predicted actions,
and token layouts, checking recorded versus ordinary ablated actions exactly.
Those artifacts are preserved for later analysis, without generating or
interpreting heatmaps in this experiment.

The no-context implementation has not been changed. Its mask-based removal
changes suffix position IDs; this pilot's action-only representation zeroing
does not. Performance loss alone cannot distinguish action dependence from
distribution shift. Compare correct and wrong effects jointly, and retain the
matched historical no-context results as a separate reference.
