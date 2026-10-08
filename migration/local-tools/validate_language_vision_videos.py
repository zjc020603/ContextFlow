import json
from pathlib import Path
import imageio_ffmpeg
r=Path(Path('.cache/language_vision_root').read_text().strip())
checks=[]
for directory in sorted((r/'rollouts').iterdir()):
 for name,fps in [('environment',20),('policy_input',4)]:
  path=directory/'ep000'/(name+'.mp4')
  frames=imageio_ffmpeg.read_frames(str(path));meta=next(frames);frames.close()
  assert meta['fps']==fps,meta
  assert abs(meta['duration']-14)<.05,meta
  assert meta['size']==(448,224),meta
  checks.append({'path':str(path.relative_to(r)),**meta})
(r/'video_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
print('28 videos checked, 14 sec each at 20/4 fps')
