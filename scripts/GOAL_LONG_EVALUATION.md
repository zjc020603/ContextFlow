# Goal / Long full-context evaluation

The released training configuration excludes **two tasks per suite**. The other
eight Goal tasks and eight Long (`libero_10`) tasks are seen according to the
paper and configuration. This runner first evaluates the four held-out tasks,
then the remaining sixteen tasks, with 50 official initial states per task.

The ordinary ContextFlow model and checkpoint are unchanged. Each task uses the
first same-task demonstration, 8 uniformly sampled image frames and 128
state/action slots. The existing loader repeats the final state/action when a
demonstration has fewer than 128 frames. It retains the original masks. Long tasks receive one complete demonstration
of that combined task, not two concatenated single-task demonstrations; there
is no phase signal or prescribed order beyond the original goal conjunction.

Each task resets the policy key to 0 and environment seed to 7; the policy then
uses its ordinary continuous random stream across that task's episodes. This
makes task scheduling independent without introducing per-episode seed folding.
Trials stop on full LIBERO success or at the original limits: Goal 300 control
steps, Long 520, following 10 settling steps. Replan every 5 steps. Separate
predicate measurements do not replace the full conjunction success criterion.

Create a fresh run manifest in the simulator environment:

```bash
source scripts/activate_env.sh libero
RUN=logs/goal_long/my_full_context_evaluation
python -m examples.libero.goal_long_eval --manifest --root "$RUN"
```

The following local orchestration uses GPUs 0–3 for held-out tasks, then 0–7 for
remaining tasks, ports 8130–8137, and CPU groups of 24 cores per worker. Ensure
these resources are available before running. It starts gated policy servers,
records per-task data, finalizes parameter hashes, and stops its own servers.

```bash
source scripts/activate_env.sh train
RUN=logs/goal_long/my_full_context_evaluation
python -m scripts.run_goal_long --root "$RUN"
JAX_PLATFORMS=cpu python -m scripts.summarize_goal_long --root "$RUN"
```

Each server must reproduce a saved full-context experiment-0 action exactly
before opening its port. This gate depends on the local experiment-0 archive
at the default `--source` of `scripts.serve_goal_long`; it is a local audit,
not a claim that the checkpoint's full training provenance is available.
`--gpu0-pid` allows attaching the independently started GPU 0 server during
initial development; normal reruns should omit it.

For a single task or another GPU, start a server directly with
`scripts.serve_goal_long --manifest <run>/task_manifest.json --output <fresh-server-dir>
--port <port>` in the model environment, then run
`examples.libero.goal_long_eval --root <run> --task <dataset-task-id> --port <port>`
in the simulator environment. Its default is 50 trials; refuse existing task
output directories rather than overwriting or merging partial trials. Finalize
that server by sending `{'benchmark_control': 'finalize'}` after tasks finish.
The full summary expects all 20 tasks and the eight default workers.

Artifacts include instruction-grouped 20 FPS videos, every executed action,
post-action simulator states, end-effector/gripper states, initial observations,
per-step original goal predicates and inference RNG keys. No attention capture,
context masking, position swaps, or changes to demonstrations occur here.

If a worker must use another physical GPU, set the model process's
`CUDA_VISIBLE_DEVICES` accordingly. An optional `<run>/resource_overrides.json`
maps client ports to physical EGL GPU IDs, for example `{"8135": 0}`. The
client records the selected GPU in its task configuration. This changes only
resource placement, not task seeds, demonstrations, or model inputs. It does
not automatically restart a running controller or adopt a replacement PID;
manual recovery must finalize and clean up the actual owned processes.
