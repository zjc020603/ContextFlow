import dataclasses
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from openpi.training import config
from openpi.policies import policy_config, context_ablation
from openpi.models import model

root = Path('logs/quickstart/context_ablation_object_correct_no_wrong_25trials_20260922_154452')
checkpoint = '/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999'
cfg = config.get_config('ContextFlow')
cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, policy_local_files_only=True))
p = policy_config.create_trained_policy_incontext(cfg, checkpoint, inference_dtype='float32', context_ablation=True)
raw = {'observation/image': np.zeros((224,224,3),dtype=np.uint8), 'observation/wrist_image': np.zeros((224,224,3),dtype=np.uint8), 'observation/state': np.zeros(8,dtype=np.float32), 'prompt': 'pick up the milk and place it in the basket', 'task_index':25, 'split':'test'}
experiment = {'mode':'correct', 'demo_task_index':25, 'rng_components':[0,25,0,0]}
_, _, rng = context_ablation.prepare_request(raw, experiment)
inputs = p._input_transform(jax.tree.map(lambda x:x, raw))

def forward(data):
    batch = jax.tree.map(lambda x:jnp.asarray(x)[None,...], data)
    output = {'state':batch['state'], 'actions':p._sample_actions(rng, model.ObservationIncontext.from_dict(batch))}
    return p._output_transform(jax.tree.map(lambda x:np.asarray(x[0]), output))['actions']

baseline = forward(inputs)
correct = p.infer({**raw, 'context_ablation':experiment})['actions']
np.testing.assert_array_equal(baseline, correct)
masked = context_ablation.mask_demonstration(inputs)
no = forward(masked)
altered = dict(masked)
for field in ['dem_prompt_images','dem_prompt_all_states','dem_prompt_all_actions']:
    altered[field] = jax.tree.map(lambda x:np.full_like(x,127 if x.dtype==np.uint8 else 0.75), masked[field])
altered['selected_episode'] = np.full_like(masked['selected_episode'],999)
no_altered = forward(altered)
np.testing.assert_array_equal(no, no_altered)
assert np.isfinite(no).all()
request_no = p.infer({**raw, 'context_ablation':{**experiment,'mode':'no'}})['actions']
np.testing.assert_array_equal(no, request_no)
result = {'checkpoint':checkpoint, 'correct_matches_original_forward_with_identical_rng':True, 'masked_demo_content_and_episode_id_do_not_affect_actions':True, 'no_context_actions_finite':True, 'no_context_request_matches_direct_masked_forward':True, 'no_vs_correct_action_rms':float(np.sqrt(np.mean((no-correct)**2))), 'fixture':'fixed synthetic live observation; real checkpoint and real milk demonstration'}
(root/'provenance').mkdir(exist_ok=True)
(root/'provenance/model_intervention_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
