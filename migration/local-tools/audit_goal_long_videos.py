from concurrent.futures import ThreadPoolExecutor
import json,time
from pathlib import Path
import imageio_ffmpeg as ff
root=Path(Path('.cache/goal_long_root').read_text().strip())
checked={}
def check(r):
 path=str(root/r['video'])
 reader=ff.read_frames(path); meta=next(reader);reader.close()
 frames,seconds=ff.count_frames_and_secs(path)
 assert meta['fps']==20 and frames==r['steps'], (path,meta,frames,r['steps'])
 return r['video'],dict(frames=frames,fps=meta['fps'],seconds=seconds)
with ThreadPoolExecutor(max_workers=4) as pool:
 while len(checked)<1000:
  records=[]
  for p in (root/'tasks').glob('*/*/episodes.jsonl'):
   for line in p.read_text().splitlines():
    r=json.loads(line)
    if r['video'] not in checked: records.append(r)
  for name,info in pool.map(check,records): checked[name]=info
  (root/'video_audit.json').write_text(json.dumps(dict(checked=len(checked),expected=1000,passed=len(checked)==1000,videos=checked),indent=2)+'\n')
  print('validated videos',len(checked),flush=True)
  if len(checked)<1000:time.sleep(30)
