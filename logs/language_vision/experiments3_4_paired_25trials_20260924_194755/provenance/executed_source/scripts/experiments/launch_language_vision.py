"""Freeze original-model source and start explicitly selected GPU servers for experiments 3/4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def launch(args):
    project=Path(__file__).resolve().parents[2]
    run=Path(args.run).resolve()
    run.mkdir(parents=True,exist_ok=False)
    assert args.gpus and len(set(args.gpus))==len(args.gpus)
    if any(g<0 for g in args.gpus):
        raise ValueError('GPU indices must be nonnegative')
    frozen=run/'provenance/frozen/src/openpi'
    shutil.copytree(project/'src/openpi',frozen,ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*.py')}
    (run/'provenance/frozen_sha256.json').write_text(json.dumps(hashes,indent=2))
    (run/'provenance/git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=project,text=True))
    processes=[]
    for gpu in args.gpus:
        output=run/'servers'/f'worker{gpu}';output.mkdir(parents=True)
        port=args.base_port+gpu
        # Pass paths through argv/env, never interpolate user text as shell code.
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 LV_FROZEN_SRC=str(frozen.parent),LV_PORT=str(port),LV_OUTPUT=str(output))
        command='source scripts/activate_env.sh train\nexport PYTHONPATH="$LV_FROZEN_SRC:$PWD:$PYTHONPATH"\nexec python scripts/serve_language_vision.py --port "$LV_PORT" --output "$LV_OUTPUT"'
        with (output/'server.log').open('w') as f:
            proc=subprocess.Popen(['bash','-c',command],cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append({'gpu':gpu,'port':port,'pid':proc.pid})
    (run/'servers/processes.json').write_text(json.dumps(processes,indent=2))
    print(run)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--gpus',type=int,nargs='+',required=True)
    p.add_argument('--base-port',type=int,default=8230);launch(p.parse_args())
