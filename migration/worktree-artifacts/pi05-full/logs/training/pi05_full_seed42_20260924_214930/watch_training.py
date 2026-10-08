"""Watch the existing authorized run and verify its final checkpoint after exit."""
import datetime
import json
import os
import pathlib
import re
import subprocess
import time

run = pathlib.Path(__file__).resolve().parent
manifest = json.loads((run / 'manifest.json').read_text())
pid = manifest['pid']
proc = pathlib.Path(f'/proc/{pid}/stat')
identity = proc.read_text().split()[21]
final = pathlib.Path(manifest['expected_final_checkpoint'])
while True:
    stamp = datetime.datetime.now().astimezone().isoformat()
    text = (run / 'train.log').read_text(errors='replace')
    progress = re.findall(r'[^\r\n]*Progress on:[^\r\n]+', text)
    metrics = re.findall(r'Step \d+: [^\r\n]+', text)
    try:
        fields = proc.read_text().split()
        active = fields[21] == identity and fields[2] != 'Z'
    except FileNotFoundError:
        active = False
    checkpoints = sorted(int(p.name) for p in final.parent.iterdir() if p.is_dir() and p.name.isdigit())
    status = {'time': stamp, 'training_active': active, 'latest_progress': progress[-1] if progress else None, 'latest_metrics': metrics[-1] if metrics else None, 'saved_checkpoints': checkpoints}
    (run / 'watch_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
    if not active:
        if not final.is_dir():
            raise RuntimeError('Training exited without final checkpoint; inspect train.log and resume safely.')
        if 'Waiting for checkpoint manager to finish' not in text:
            raise RuntimeError('Final checkpoint exists but normal training completion was not logged.')
        env = {**os.environ, **manifest['environment_overrides'], 'CUDA_VISIBLE_DEVICES': '0'}
        with (run / 'verify_final.log').open('w') as out:
            result = subprocess.run([manifest['command'][0], str(run / 'verify_final.py')], cwd=manifest['cwd'], env=env, stdout=out, stderr=subprocess.STDOUT)
        status.update(training_active=False, verification_exit_code=result.returncode, status='complete' if result.returncode == 0 else 'verification_failed')
        (run / 'watch_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status), flush=True)
        raise SystemExit(result.returncode)
    time.sleep(45)
