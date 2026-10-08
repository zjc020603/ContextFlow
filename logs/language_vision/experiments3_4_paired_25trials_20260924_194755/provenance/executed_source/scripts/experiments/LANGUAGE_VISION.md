# Language and current-vision interventions (experiments 3 and 4)

Uses the original ContextFlow checkpoint and existing context-selection protocol.
No experiment-2 modules, model changes, retraining, or altered demonstration modalities.

Scope: fixed tomato-sauce LIBERO Object scene, original layout, initial-state
indices 0–24; both demonstration episodes (milk 814, tomato sauce 821).

- Experiment 3: two demonstrations × three literal prompts (milk, tomato sauce,
  empty string). The request task index remains 25 as a **sampling anchor**, not
  a record of the instruction. All cases fold `[0, 25, episode, replan]` into the
  JAX seed. The supplied prompt is recorded separately. The existing correct/wrong
  protocol names are relative to this anchor; use `demo` and `language` for analysis.
- Experiment 4: fixed milk instruction, two demonstrations × normal images,
  target box occlusion, other-object box occlusion, equal-area background box,
  freeze after approach. Normal-image trials are shared with experiment 3.

Every condition first runs initial indices 0–4, passes the audit, then extends
with indices 5–24. Total: 14 distinct conditions × 25 = 350 physical rollouts.
The initial simulator state, observations, demonstration contents, and noise
hashes are checked. Later observations naturally diverge across policies.

At planning steps 0, 80, 160 of the first five episodes, store attention averaged
across 8 heads and the first 5 action queries for layers 0/9/17 and flow steps
0/5/9. All heads/queries are retained for episode 0. At these same observations
of normal milk-language rollouts, perform paired inference with all three prompts:
no action from these counterfactual requests is executed in the simulator.

## Vision definition

MuJoCo visible geom-ID renderings identify each object's current visible pixel
bounding box in **both** front and wrist cameras. Rotate and resize consistently
with the model input; replace the rectangle with RGB `(127,127,127)`.
The other-object mask covers the other member of the milk/tomato pair.
The background rectangle normally has the same height and width as the target rectangle
in that condition's current observation and covers no visible robot/object geom.
Select its valid location deterministically, farthest from the target rectangle.
If no rectangle fits (possible in the wrist camera), use a compact region of exactly
the same number of background pixels, preserving the exclusion of robot/objects.
Record whether it remains rectangular; shape then differs from target masking.
If the full object box is larger than the available background, cap both object and
background masks to that shared area budget. Prefer masking actual object pixels
inside its box, then other box pixels. Record the cap and the fraction of visible
object pixels actually removed; such views are partial occlusions. Invisible objects have zero masked pixels.

A box can also cover robot/background pixels; record robot pixel overlap.
Object masks reveal position and sometimes approximate size. This is an occlusion
intervention, not object removal or a proof of object identity understanding.

Mask target: first bilateral-finger grasp observed in the corresponding normal
milk-language rollout; if absent, the first actual gripper contact, explicitly
flagged `contact_without_grasp`. Do not discard baseline failures. Abort if no
contact target exists, rather than silently substituting a demonstration label.

Freeze: at the first planning time where EEF-to-target-center Euclidean distance
is <= 0.12 m, retain both input images for the rest of the rollout. Robot state
and simulator physics continue normally. Audit that all pre-trigger actions
match baseline and all post-trigger image arrays remain identical. This tests
visual feedback after approach, including grasp and transport, not only grasp.

## Behavior outputs

280 control actions, 20 Hz, 5 actions per model call; never terminate on first
success. Record all six objects' positions, gripper contact, bilateral grasp,
height change, and LIBERO In predicates every control step.

- `first_region`: first initial object XY region within 8 cm, with EEF no more
  than 15 cm above the initial object center; a geometric approach proxy.
- contact: any gripper contact geom contacts object contact geoms.
- grasp: robosuite bilateral fingerpad contact test; not a guarantee of lifting.
- lift: center rises >= 3 cm relative to the settled initial height; consult
  grasp/contact alongside this since another collision can also move an object.
- placement: LIBERO In predicate true at any post-action step, with first time.

Each episode stores environment video (both cameras, 20 fps), exact policy input
video (one frame per replan, 4 fps), NPZ trajectory/masks/input images and JSON
metadata. Both videos play in real simulated time (14 seconds).

## Execution

Use independently gated servers on free GPUs (this run used GPUs 0–7), and save
`servers/processes.json` entries `{gpu,port,pid}`. A server command:

```bash
source scripts/activate_env.sh train
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$PYTHONPATH" \
 python scripts/serve_language_vision.py --port 8230 --output "$RUN/servers/worker0"
```

For concurrent development, put a copied `src/openpi` directory first in
`PYTHONPATH`; preserve its SHA-256 hashes. This run freezes the model code.
A reusable launcher creates a fresh run directory, freezes source, and starts only
the GPUs explicitly listed:

```bash
python3 scripts/experiments/launch_language_vision.py --run "$RUN" --gpus 0 1 2 3 4 5 6 7
```

Then in the simulator environment, with project root also on `PYTHONPATH`:

```bash
python scripts/experiments/run_language_vision.py --run "$RUN" --phase language --stop 5
# Run audits using the main .venv Python (NumPy installed).
.venv/bin/python scripts/experiments/audit_language_vision.py --run "$RUN" --n 5
python scripts/experiments/run_language_vision.py --run "$RUN" --phase vision --stop 5
.venv/bin/python scripts/experiments/audit_language_vision.py --run "$RUN" --n 5 --vision
python scripts/experiments/run_language_vision.py --run "$RUN" --phase language --start 5 --stop 25
python scripts/experiments/run_language_vision.py --run "$RUN" --phase vision --start 5 --stop 25
.venv/bin/python scripts/experiments/audit_language_vision.py --run "$RUN" --n 25 --vision
```

The controller never runs two jobs on the same server concurrently. A condition
can extend with a disjoint episode interval, but existing episodes are never
overwritten. Fix and archive any failed partial episode before retrying it.
