import json,time,hashlib
from pathlib import Path
import numpy as np
root=Path(Path('.cache/attention_timeline_root').read_text().strip())
old=Path('logs/attention_capture/experiment0_consistency_20260923_143830')
seen=set();overlap=0;max_error=0.;rows=[]
while len(seen)<840:
 p=root/'capture/pairs.jsonl'
 try:records=[json.loads(s) for s in p.read_text().splitlines()]
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(3);continue
 fresh=[r for r in records if r['case_id'] not in seen]
 for row in fresh:
  cid=row['case_id'];file=root/'capture'/cid/'capture.npz'
  with np.load(file) as c:
   for a,b in [('ordinary_actions','captured_actions'),('ordinary_executable_actions','captured_executable_actions')]:
    assert c[a].shape==c[b].shape and c[a].dtype==c[b].dtype
    assert c[a].tobytes()==c[b].tobytes(),cid
   assert int(c['executed_steps'])==10
   p=c['probabilities']
   assert np.isfinite(p).all()
   assert np.all(p[...,np.flatnonzero(~c['prefix_mask'][0])]==0)
   if row['mode']=='no':assert np.all(p[...,816:976]==0)
   oldfile=old/'validation_attempt3'/cid/'capture.npz'
   if oldfile.exists():
    with np.load(oldfile) as o:
     for name in ['probabilities','noise','ordinary_actions','captured_actions','prefix_mask']:
      np.testing.assert_array_equal(c[name],o[name],err_msg=cid+'/'+name)
    with np.load(root/'observations'/row['npz']) as current,np.load(old/'observations'/row['npz']) as previous:
     for key in previous.files:np.testing.assert_array_equal(current[key],previous[key],err_msg=cid+'/'+key)
    overlap+=1
   with np.load(root/'observations'/row['npz']) as obs:
    history=float(np.max(np.abs(c['ordinary_executable_actions'][:5]-obs['reference_executed_actions'])))
   assert history==0.,(cid,history)
  seen.add(cid);rows.append({'case_id':cid,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
  if len(seen)%56==0:print('Audited',len(seen),'old captures exactly reproduced',overlap,flush=True)
 if not fresh:time.sleep(5)
assert overlap==45,overlap
(root/'artifact_audit.json').write_text(json.dumps({'pairs':840,'raw_and_executable_actions_bytewise_equal':True,
  'historical_action_max_difference':0.0,'previous_experiment0_exact_reproductions':overlap,'files':rows},indent=2)+'\n')
print('ALL 840 ARTIFACTS PASSED',flush=True)
