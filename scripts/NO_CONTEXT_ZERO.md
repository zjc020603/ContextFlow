# No-context control with unchanged prefix positions

This supplements the original correct/no/wrong experiment on the milk and
 tomato-sauce tasks in their own original scenes (25 trials each).
It does not overwrite historical runs or change the regular policy server.

`DemoContextZero` zeros every compressed demonstration image/state/action token
immediately before Gemma, retaining the original masks, token slots and position
IDs. It loads a normal same-task demonstration only to obtain the original mask
structure, and erases all encoded content. Missing-camera slots stay invalid.
This is an inference-time intervention, not a learned null context. Valid zero
slots can subsequently absorb information from other tokens.

The gate checks the historical full-model outputs, unchanged live-image/language
prefix and positions, invariance to changing all demo values or the template task,
and equality of ordinary and attention-recording inference.

Start the gated server in a fresh directory (model environment):

```bash
source scripts/activate_env.sh train
export CUDA_VISIBLE_DEVICES=2
RUN=logs/context_ablation/my_no_context_zero_run
python -m scripts.serve_no_context_zero --port 8123 --output "$RUN/server"
```

After `server/READY.json` exists, run the simulator in another terminal:

```bash
source scripts/activate_env.sh libero
export CUDA_VISIBLE_DEVICES=3
RUN=logs/context_ablation/my_no_context_zero_run
python -m examples.libero.no_context_zero --port 8123 --trials 25 --output "$RUN/rollouts"
python - <<'PY'
from openpi_client.websocket_client_policy import WebsocketClientPolicy
print(WebsocketClientPolicy('127.0.0.1',8123).infer({'finalize_no_context_zero':True}))
PY
```

Stop that dedicated server after finalization, then validate and summarize:

```bash
source scripts/activate_env.sh train
RUN=logs/context_ablation/my_no_context_zero_run
python -m scripts.summarize_no_context_zero --root "$RUN"
```

The summary intentionally requires all 50 paired episodes and 150 captures. All
videos use 20 FPS and instruction-named directories. Attention is saved at steps
0/80/160 with all heads and selected flow steps/layers, without interpreting it.

The existing quickstart context-ablation menu still uses the historical masked
no-context definition. Use this dedicated entry point for this new control.
The fixed tomato-scene position-swap conditions require a separate rerun and
must not be relabeled using results from these per-language scenes.
