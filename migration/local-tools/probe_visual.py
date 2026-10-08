from libero.libero import benchmark
from examples.libero import context_ablation as c, position_intervention as p
from examples.libero.live_vision_intervention import intervene, object_geom_ids
import numpy as np
from PIL import Image
s=benchmark.get_benchmark_dict()['libero_object']()
e,_=c.baseline._get_libero_env(s.get_task(5),256,7)
e.reset();e.set_init_state(s.get_task_init_states(5)[0]);o,_=p.apply_layout(e,'original')
for _ in range(10):o,*_=e.step(c.baseline.LIBERO_DUMMY_ACTION)
req=c.make_request(o,{'description':c.TASKS['milk'],'index':25},25,'correct',0,0,c.Args())
print('geoms', object_geom_ids(e,'milk_1'))
for mode in ('mask_target','mask_other','mask_background'):
 r,m,counts=intervene(e,req,'milk_1','tomato_sauce_1',mode)
 Image.fromarray(np.concatenate([req['observation/image'],r['observation/image'],req['observation/wrist_image'],r['observation/wrist_image']],axis=1)).save('.cache/'+mode+'.png')
 print(mode,counts)
print('grasp',e.env._check_grasp(e.env.robots[0].gripper,e.env.objects_dict['milk_1']))
e.close()
