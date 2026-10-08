from pathlib import Path
import json, hashlib, collections, datetime
import numpy as np
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.libero.summarize_position_ablation import summarize, LAYOUTS, MODES, TASKS, demo_targets

root=Path(Path('.cache/position_run_root').read_text())
result=summarize(root)
records={}; scene_summary={}; movement=[]; total=0
for layout in LAYOUTS:
 for mode in MODES:
  directory=root/layout/mode
  rows=list(map(json.loads,(directory/'episodes.jsonl').read_text().splitlines()))
  assert len(rows)==50
  for row in rows:
   task=row['task'];episode=row['episode_index'];records[layout,mode,task,episode]=row
   assert row['steps']==280 and len(row['context_requests'])==56
   identity,_=demo_targets(task,mode,layout)
   expected_episode=[] if identity is None else [814 if identity=='milk' else 821]
   expected_index=None if identity is None else (25 if identity=='milk' else 28)
   for i,request in enumerate(row['context_requests']):
    assert request['selected_episode']==expected_episode
    assert request['demo_task_index']==expected_index
    assert request['task_index']==(25 if task=='milk' else 28)
    assert request['rng_components']==[0,25 if task=='milk' else 28,episode,i]
    assert request['mode']==mode
   intervention=row['position_intervention']
   for name,before in intervention['before_poses'].items():
    expected=np.array(before)
    if layout!='original' and name in ('milk_1','tomato_sauce_1'):
     other='tomato_sauce_1' if name=='milk_1' else 'milk_1'
     expected[:2]=intervention['before_poses'][other][:2]
    np.testing.assert_array_equal(expected,intervention['intervened_poses'][name])
   assert (root/layout/row['initial_view']).is_file()
   with np.load(root/layout/row['trajectory']) as z:
    assert z['actions'].shape==(280,7) and np.isfinite(z['actions']).all()
    assert hashlib.sha256(z['initial_sim_state'].tobytes()).hexdigest()==row['initial_state_sha256']
    assert np.isfinite(z['object_positions']).all() and np.isfinite(z['scene_object_positions']).all()
    for i,name in enumerate(TASKS):
     assert bool(z['in_basket'][:,i].any())==row['ever_in_basket'][name]
     assert bool(z['in_basket'][-1,i])==row['final_in_basket'][name]
    for i,name in enumerate(z['scene_object_names']):
     assert bool(z['scene_in_basket'][:,i].any())==row['scene_ever_in_basket'][str(name)]
     p=z['scene_object_positions'][:,i,:]
     # First post-action frame is the reference for descriptive motion only.
     movement.append({'layout':layout,'mode':mode,'language':task,'episode':episode,'object':str(name),
       'max_rise_from_first_frame_m':float(max(0,(p[:,2]-p[0,2]).max())),
       'max_displacement_from_first_frame_m':float(np.linalg.norm(p-p[0],axis=1).max()),
       'ever_in_basket':row['scene_ever_in_basket'][str(name)]})
   with av.open(str(root/layout/row['video'])) as video:
    assert float(video.streams.video[0].average_rate)==20
    assert video.streams.video[0].frames==280
   total+=1
  for task in TASKS:
   selected=[row for row in rows if row['task']==task]
   names=selected[0]['scene_ever_in_basket']
   scene_summary[f'{layout}/{mode}/{task}']={name:sum(row['scene_ever_in_basket'][name] for row in selected) for name in names}
assert total==300
for layout in LAYOUTS:
 for mode in MODES:
  smoke=list(map(json.loads,(root/'smoke'/layout/mode/'episodes.jsonl').read_text().splitlines()))
  for row in smoke:
   formal=records[layout,mode,row['task'],row['episode_index']]
   assert row['initial_observation_sha256']==formal['initial_observation_sha256']
   with np.load(root/'smoke'/layout/row['trajectory']) as a,np.load(root/layout/formal['trajectory']) as b:
    for field in ('actions','eef_and_gripper','object_positions','scene_object_positions','in_basket','scene_in_basket'):
     np.testing.assert_array_equal(a[field],b[field])
(root/'all_object_comparison.json').write_text(json.dumps(scene_summary,indent=2)+'\n')
(root/'object_motion.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in movement))
# Explicit paired physical-object switches under unchanged language/demo/noise.
paired={}
for task in TASKS:
 for mode in MODES:
  identity,_=demo_targets(task,mode,'original')
  other=None if identity is None else next(t for t in TASKS if t!=identity)
  pairs=[(records['original',mode,task,e],records['swap_milk_tomato',mode,task,e]) for e in range(25)]
  exclusive=lambda row,obj: row['ever_in_basket'][obj] and not row['ever_in_basket'][next(t for t in TASKS if t!=obj)]
  paired[f'{task}/{mode}']={'pairs':25,'demo_object':identity,
    'original_demo_only_to_swapped_old_position_occupant_only':None if identity is None else sum(exclusive(a,identity) and exclusive(b,other) for a,b in pairs),
    'original_demo_only_to_swapped_same_demo_only':None if identity is None else sum(exclusive(a,identity) and exclusive(b,identity) for a,b in pairs)}
(root/'paired_object_switches.json').write_text(json.dumps(paired,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
labels=[f'{task} / {mode}' for task in TASKS for mode in MODES]
for ax,layout,title in zip(axes,LAYOUTS,['Original layout','Milk and tomato sauce exchanged']):
 values=[[sum(records[layout,mode,task,e]['ever_in_basket'][obj] for e in range(25)) for obj in TASKS] for task in TASKS for mode in MODES]
 ax.imshow(values,vmin=0,vmax=25,cmap='Blues',aspect='auto')
 ax.set_xticks([0,1],['Milk in basket','Tomato sauce in basket'])
 ax.set_yticks(range(6),labels)
 ax.set_title(title)
 for i,row in enumerate(values):
  for j,count in enumerate(row):ax.text(j,i,f'{count}/25 ({count*4}%)',ha='center',va='center',color='white' if count>12 else 'black')
fig.suptitle('Fixed tomato-sauce scene | Language / context mode\n300 paired rollouts, unchanged demonstrations')
fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
meta=json.loads((root/'run.json').read_text());meta['verified']={'episodes':300,'videos':300,'trajectories':300,'video_fps':20,'video_frames':280,'finite_actions':True,'exact_xy_only_intervention':True,'paired_states_observations_rng_demo_episodes':True,'smoke_full_exact_repeats':12};meta['audited_at']=datetime.datetime.now().astimezone().isoformat()
(root/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Verified 300 episodes, 300 videos at 20 FPS, all trajectories/goals, XY-only interventions, and paired inference inputs.')
