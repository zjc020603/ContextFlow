import json,os,subprocess,time
from pathlib import Path
root=Path(Path('.cache/goal_long_root').read_text().strip())
while len(list((root/'servers').glob('gpu*/FINISHED.json')))<8:time.sleep(5)
print('All worker parameter checks complete; auditing',flush=True)
env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run(['taskset','-c','192-215','.venv/bin/python','-u','-m','scripts.summarize_goal_long','--root',str(root)],env=env,check=True)
while not json.loads((root/'video_audit.json').read_text())['passed']:time.sleep(5)
print('Statistical and video audits complete',flush=True)
