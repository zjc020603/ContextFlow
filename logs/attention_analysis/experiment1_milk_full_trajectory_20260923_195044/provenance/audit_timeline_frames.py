from pathlib import Path
import json,numpy as np
r=Path(Path('.cache/attention_timeline_root').read_text().strip())
m=json.loads((r/'observations/manifest.json').read_text())
checked=0
for rollout in m['rollouts']:
 with np.load(r/'observations'/rollout['frames']) as f,np.load(rollout['source_trajectory']) as old:
  front,wrist=f['front'],f['wrist']
  assert len(front)==len(wrist)==281
  np.testing.assert_array_equal(f['state'][1:],old['eef_and_gripper'])
  np.testing.assert_array_equal(f['in_basket'][1:],old['scene_in_basket'])
  np.testing.assert_array_equal(f['actions'],old['actions'])
  for case in m['cases']:
   if case['episode']!=rollout['episode'] or case['mode']!=rollout['mode']:continue
   t=case['step']
   with np.load(r/'observations'/case['npz']) as snapshot:
    np.testing.assert_array_equal(snapshot['observation/image'],front[t])
    np.testing.assert_array_equal(snapshot['observation/wrist_image'],wrist[t])
    np.testing.assert_array_equal(snapshot['reference_executed_actions'],old['actions'][t:t+5])
   checked+=1
 print('Verified all frame/attention time mappings:',rollout['episode'],rollout['mode'],flush=True)
assert checked==840
(r/'frame_audit.json').write_text(json.dumps({'trajectories':15,'frames':4215,'planning_snapshots_matched_to_exact_display_frame':840,'state_and_basket_trace_error':0},indent=2)+'\n')
print('ALL FRAME ALIGNMENT CHECKS PASSED',flush=True)
