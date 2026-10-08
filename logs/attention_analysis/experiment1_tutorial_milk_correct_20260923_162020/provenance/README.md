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
