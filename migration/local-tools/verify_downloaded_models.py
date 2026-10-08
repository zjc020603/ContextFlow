import argparse
import gc
import faulthandler
import json
from pathlib import Path
import numpy as np
faulthandler.dump_traceback_later(60, repeat=True)
from flax.traverse_util import flatten_dict
from openpi.models.model import restore_params
from openpi.training.config import get_config

parser=argparse.ArgumentParser();parser.add_argument('model',choices=['pi0','contextflow']);args=parser.parse_args()
root=Path('/data/zjc/workspace/models')
checkpoint=root/('openpi/openpi-assets/checkpoints/pi0_base' if args.model=='pi0' else 'ContextFlow/ContextFlow_run1/19999')
print('Restoring checkpoint:',checkpoint,flush=True)
params=restore_params(checkpoint/'params',restore_type=np.ndarray)
flat=flatten_dict(params)
count=sum(x.size for x in flat.values())
size=sum(x.nbytes for x in flat.values())
print('Restored',len(flat),'arrays;',count,'parameters;',round(size/2**30,3),'GiB',flush=True)
assert count>0
for key,x in flat.items():
    if x.size:
        samples=x.reshape(-1)[::max(1,x.size//32)]
        assert np.isfinite(samples).all(),f'Non-finite weights in {key}'
report={'checkpoint':str(checkpoint),'arrays':len(flat),'parameters':count,'bytes':size,'orbax_restore':'passed','finite_samples':'passed'}
if args.model=='contextflow':
    config=get_config('ContextFlow')
    model=config.model.load(params)
    print('ContextFlow parameter shapes match current config: PASS',flush=True)
    report['config_shape_check']='passed'
    del model
Path(f'logs/environment_setup/{args.model}-weight-verification.json').write_text(json.dumps(report,indent=2))
print('CHECKPOINT VERIFIED',flush=True)
