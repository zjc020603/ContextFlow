# Attention tutorial from one saved observation

Use an existing experiment-0 run, in the model environment:

```bash
source scripts/activate_env.sh train
python scripts/attention/make_tutorial.py \
  --run logs/attention_capture/experiment0_consistency_20260923_143830 \
  --case original/correct/milk/ep000_step000 \
  --output logs/attention_analysis/my_first_example
```

This script reads `observations/manifest.json` and `validation_attempt3/<case>`.
It currently expects the verified ContextFlow capture layout (3 sampling steps,
3 layers, 8 heads, 51 queries, 1027 keys); it does not load a model or run inference.
Use a fresh output directory to preserve previous reports.

Start with `READ_ME_FIRST.md`, or open `index.html` in a browser. The interactive
page embeds the data and images and works offline without a server. Download the
whole output directory if you also want its links to static figures and reports.
The PNG figures use English labels; the page and reading guide are in Chinese.

Default: flow step 9, layer 17, mean of 8 heads and the first 5 action queries
(suffix indices 1:6; excludes state query 0). The HTML can change the captured
layer, sampling step and query grouping. The same image is used throughout.

Colors show direct attention weights, not object probabilities or causal
importance. Absolute color scales and per-camera display scaling are explicitly
separated. Language weights retain their denominator over all keys; repeated
words are separate tokens. See `summary.json`, `blocks.csv` and
`averaged_attention.npz` for the exact values and aggregation definitions.

## Compare milk / original layout / three context modes

```bash
python -m scripts.attention.compare_contexts \
  --run logs/attention_capture/experiment0_consistency_20260923_143830 \
  --output logs/attention_analysis/milk_original_three_contexts_new
```

This comparison fixes environment step 0 and checks identical images, robot
state, language, noise, initial simulator state hashes and PRNG keys across
correct/no/wrong. Initial state 0 is the main example; indices 1–4 are retained
for consistency checks. The interactive page lets you select a saved layer and
sampling step, with the first 5 action queries and all 8 heads averaged.

Per-camera contrast expansion is the default. A second option shares a color
scale across the three contexts of the same camera at the selected layer/step.
Spatial difference maps normalize each camera to unit mass before subtraction;
these are different quantities from the all-key attention percentages.

The no-context intervention also changes the number of valid prefix tokens and
thus suffix position IDs in the existing model. The report makes this explicit;
no new intervention is introduced to isolate this effect.

## Full trajectory with a control-step slider

The original rollout saves all actions/states, while experiment 0 only exported
three observation times. Recover all 281 observations (0 through 280) per
rollout, and all 56 real planning calls (0, 5, ..., 275), in the simulator env:

```bash
source scripts/activate_env.sh libero
python -m examples.libero.prepare_attention_timeline \
  --source logs/quickstart/position_swap_object_25trials_20260922_175506 \
  --output "$RUN/observations"

source scripts/activate_env.sh train
python scripts/check_attention_capture.py \
  --observations "$RUN/observations" --output "$RUN/capture"
python -m scripts.attention.build_timeline --run "$RUN"
```

Set `RUN` to a fresh directory first. This produces 840 captured and verified
calls for milk / original layout / three contexts / initial-state indices 0–4.
The replay verifies all saved robot/object states and basket predicates.
`build_timeline --watch` can run while capture is in progress and builds each
initial-state page as its three trajectories finish.

During concurrent model development, use a frozen copy of `src/openpi` at the
front of `PYTHONPATH` for capture, preserving the source hashes. The completed
full-trajectory run includes such a copy under `provenance/frozen/src`.

Open `viewer/index.html` for initial state 0; the other states are separate
`episode_###.html` pages. Each page embeds its data and images, works offline,
and plays at a target 20 Hz. A control-step slider visits all 281 states. At a
non-planning time (e.g. 83), the current observation is displayed next to the
last planning heatmap (80), whose own background stays at 80. Old attention is
never overlaid on a new image. At the final state 280 the last plan is 275.

The browser displays JPEG copies; capture uses lossless observations. Browser
transport uses float32 averages; float64 analysis arrays are kept as NPZ files.
Actual object-in-basket state and first-occurrence events are shown independently
from the heatmap. Later cross-context images differ, so this view does not imply
a fixed-input causal comparison at every control step.
