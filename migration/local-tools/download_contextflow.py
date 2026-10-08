"""Download the verified public Drive checkpoint; CRC32C-check each original file."""
import base64
import concurrent.futures as cf
import json
import os
from pathlib import Path
import time
from importlib import import_module
import google_crc32c
import requests
from download_assets import transfer

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/environment_setup'
DEST=Path('/data/zjc/workspace/models/ContextFlow/ContextFlow_run1/19999')
entries=json.loads((LOG/'contextflow-manifest.json').read_text())
confirm=import_module('gdown.download').get_url_from_gdrive_confirmation
jobs=[];checks=[];chunk=16*1024*1024
saved_checks={e['path']:e for e in json.loads((LOG/'contextflow-crc32c.json').read_text())} if (LOG/'contextflow-crc32c.json').exists() else {}
for e in entries:
    dest=DEST/e['path']
    saved=saved_checks.get(e['path'])
    if saved and saved['size']==e['size']:
        complete=dest.is_file() and dest.stat().st_size==e['size']
        if not complete and e['size']>chunk:
            parts=dest.with_name(dest.name+'.parts')
            complete=all((parts/f'{start//chunk:05d}').is_file() and (parts/f'{start//chunk:05d}').stat().st_size==min(chunk,e['size']-start) for start in range(0,e['size'],chunk))
        if complete:
            checks.append(saved)
            print('REUSING COMPLETE',e['path'],flush=True)
            continue
    session=requests.Session()
    url=f"https://drive.google.com/uc?id={e['id']}"
    for attempt in range(6):
        try:
            for _ in range(4):
                with session.get(url,stream=True,timeout=(15,60)) as response:
                    response.raise_for_status()
                    if 'Content-Disposition' in response.headers:
                        assert int(response.headers['Content-Length'])==e['size']
                        crc=next((s.strip().split('=',1)[1] for s in response.headers.get('X-Goog-Hash','').split(',') if s.strip().startswith('crc32c=')),None)
                        assert crc,'Missing Google CRC32C'
                        break
                    url=confirm(response.text)
            else:raise RuntimeError('No download response')
            break
        except Exception:
            if attempt==5:raise
            time.sleep(3)
    print('SOURCE VERIFIED',e['path'],e['size'],'CRC32C',crc,flush=True)
    checks.append({'path':e['path'],'size':e['size'],'crc32c':crc})
    if dest.exists() and dest.stat().st_size==e['size']:continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    if e['size']<=chunk:
        jobs.append((dest,e['size'],[(url,True)],None,'sha256',0,None))
    else:
        parts=dest.with_name(dest.name+'.parts');parts.mkdir(parents=True,exist_ok=True)
        for start in range(0,e['size'],chunk):
            jobs.append((parts/f'{start//chunk:05d}',min(chunk,e['size']-start),[(url,True)],None,'sha256',start,e['size']))
(LOG/'contextflow-crc32c.json').write_text(json.dumps(checks,indent=2))
with cf.ThreadPoolExecutor(max_workers=int(os.environ.get('CONTEXTFLOW_DOWNLOAD_WORKERS', '4'))) as pool:
    futures=[pool.submit(transfer,*job) for job in jobs]
    for done,future in enumerate(cf.as_completed(futures),1):
        future.result()
        if done%10==0 or done==len(jobs):
            status={'chunks':done,'total_chunks':len(jobs)}
            (LOG/'contextflow-progress.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
for e in checks:
    dest=DEST/e['path']
    if not dest.exists():
        with dest.with_name(dest.name+'.assembling').open('wb') as out:
            for i in range((e['size']+chunk-1)//chunk):
                with (dest.with_name(dest.name+'.parts')/f'{i:05d}').open('rb') as src:
                    for b in iter(lambda:src.read(chunk),b''):out.write(b)
        os.replace(dest.with_name(dest.name+'.assembling'),dest)
    crc=google_crc32c.Checksum()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(chunk),b''):crc.update(block)
    assert base64.b64encode(crc.digest()).decode()==e['crc32c'],f'CRC32C mismatch: {dest}'
    assert dest.stat().st_size==e['size']
    print('VERIFIED',e['path'],flush=True)
print('CONTEXTFLOW COMPLETE: all 20 files passed original Google CRC32C',flush=True)
