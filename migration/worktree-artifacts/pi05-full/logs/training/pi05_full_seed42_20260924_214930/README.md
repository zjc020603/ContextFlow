# pi05 full ContextFlow training run

- Code branch: `experiment/pi05-full-contextflow`, commit `597539deeb868bea72c8e2ee34f054a71e32bcce`.
- Initialization: official `gs://openpi-assets/checkpoints/pi05_base/params` (complete pretrained backbone; new context parameters initialized randomly).
- Dataset: `physical-intelligence/libero`, 32 seen task instructions, 1,350 episodes, 223,060 frames; eight held-out tasks excluded from both targets and demonstration candidates.
- Full fine-tuning, seed 42, global batch 32, FSDP 8, 20,000 updates, 8 A100 80GB GPUs.
- Normalization is computed only from retained training episodes; pi05 uses quantiles.
- Checkpoints scheduled at 5000, 10000, 15000, 19999. The final checkpoint is the prespecified primary evaluation checkpoint.
- `manifest.json` records the exact command, environment, source weights, split, paths and PID. `train.log` contains training progress. `launch.sh` records the launch command; do not launch a duplicate process while the recorded run is active.

## Evaluation after training

Follow [Table 1](https://arxiv.org/html/2609.06852v1): Spatial's two held-out bowl tasks and Object's held-out milk/tomato-sauce tasks, 50 rollouts per task, 200 total. Provide held-out demonstrations only at inference; do not update weights on those tasks. Preserve rollout settings, initial states, demonstration choices and random seeds in model comparisons. Report per-task, per-suite and overall success.

Then run paired correct/no/wrong context interventions using the same trained weights. Report correct-minus-no-context success separately from absolute performance. No-context inference is an intervention, not a separately trained pi05 baseline. A matched full pi0 + ContextFlow experiment is needed to assess whether changing the pretrained backbone increases the context benefit. A separately trained no-context model would answer the additional question of context training's value.

This extends Table 1's evaluation protocol with a new architecture. Complete pretrained VLM retention and full fine-tuning intentionally differ from the original ContextFlow recipe; it is not an exact reproduction of that model. The 20k training budget is a matched starting point, not a guaranteed performance ceiling.
