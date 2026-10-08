"""Download manifest-pinned public assets with resumable, verified transfers."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'logs/environment_setup'
DATA = Path('/data/zjc/workspace/datasets/physical-intelligence/libero')
MODELS = Path('/data/zjc/workspace/models')
_state = threading.local()

def session(proxy):
    key = 'proxy' if proxy else 'direct'
    if not hasattr(_state, key):
        s = requests.Session(); s.trust_env = proxy
        setattr(_state, key, s)
    return getattr(_state, key)

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def valid(path, size, checksum=None, algorithm='sha256'):
    return path.is_file() and path.stat().st_size == size and (not checksum or digest(path, algorithm) == checksum)

def transfer(dest, size, urls, checksum=None, algorithm='sha256', start=0, total=None):
    if valid(dest,size,checksum,algorithm): return
    if dest.exists(): raise RuntimeError(f'Existing file does not match manifest: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    part = dest.with_name(dest.name + '.partial')
    for attempt in range(16):
        url, proxy = urls[attempt % len(urls)]
        offset = part.stat().st_size if part.exists() else 0
        if offset > size: raise RuntimeError(f'Oversized partial: {part}')
        try:
            if offset < size:
                headers = {'Accept-Encoding':'identity'}
                if start or offset or total is not None:
                    headers['Range'] = f'bytes={start+offset}-{start+size-1}'
                with session(proxy).get(url,headers=headers,stream=True,timeout=(15,60)) as r:
                    r.raise_for_status()
                    if 'Range' in headers:
                        expected=f'bytes {start+offset}-{start+size-1}/{total if total is not None else size}'
                        if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                            raise RuntimeError('Server did not honor the requested byte range')
                    elif r.status_code != 200:
                        raise RuntimeError(f'Unexpected response status {r.status_code}')
                    with part.open('ab' if offset else 'wb') as f:
                        for block in r.iter_content(1024*1024): f.write(block)
            if not valid(part,size,checksum,algorithm):
                # Do not discard a partial on connection errors. A complete invalid file is retained for inspection.
                raise RuntimeError(f'Size or checksum mismatch: {part.name}')
            os.replace(part,dest)
            return
        except Exception as exc:
            print(f'RETRY {dest.name} {attempt+1}/16 {type(exc).__name__}',flush=True)
            if attempt==15: raise RuntimeError(f'Failed transfer: {dest}') from None
            time.sleep(min(2+attempt,15))

def dataset(workers):
    m=json.loads((LOG/'libero-manifest.json').read_text())
    entries=sorted(m['files'],key=lambda x:(not x['path'].startswith('meta/'), x['path']))
    started=time.monotonic();done=0;size_done=0
    def one(e):
        path=e['path'];q=quote(path,safe='/')
        urls=[(f"https://hf-mirror.com/datasets/{m['repo_id']}/resolve/{m['sha']}/{q}",False),
              (f"https://huggingface.co/datasets/{m['repo_id']}/resolve/{m['sha']}/{q}?download=true",True)]
        checksum=e['lfs']['sha256'] if e['lfs'] else None
        transfer(DATA/path,e['size'],urls,checksum)
        if not e['lfs']:
            b=(DATA/path).read_bytes()
            h=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
            if h != e['blob_id']: raise RuntimeError(f'Git blob hash mismatch: {path}')
        return e['size']
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,e) for e in entries]
        for f in cf.as_completed(futures):
            size_done+=f.result();done+=1
            if done%10==0 or done==len(entries):
                status={'kind':'dataset','files':done,'total_files':len(entries),'verified_bytes':size_done,'total_bytes':sum(x['size'] for x in entries),'elapsed_seconds':round(time.monotonic()-started,1)}
                (LOG/'dataset-progress.json').write_text(json.dumps(status))
                print(json.dumps(status),flush=True)
    (DATA/'.contextflow-revision.json').write_text(json.dumps({'repo':m['repo_id'],'revision':m['revision'],'commit':m['sha'],'files':len(entries),'verified':True},indent=2))
    print('DATASET COMPLETE',flush=True)

def s3(workers):
    entries=json.loads((LOG/'pi0-manifest.json').read_text())
    root=MODELS/'openpi/openpi-assets/checkpoints/pi0_base'
    chunk=8*1024*1024  # Matches the original multipart S3 ETag block size.
    jobs=[]; assembled=[]
    for e in entries:
        if not e['size']: continue
        dest=root/e['key'].removeprefix('checkpoints/pi0_base/')
        etag=e['etag'].strip('"')
        if dest.exists() and dest.stat().st_size==e['size']:
            assembled.append((dest,e));continue
        url='https://openpi-assets.storage.googleapis.com/'+quote(e['key'],safe='/')
        fallback='https://openpi-assets.s3.amazonaws.com/'+quote(e['key'],safe='/')
        mirror='https://hf-mirror.com/oldTOM/pi0_base/resolve/88e794fb312e073e9f09d2d7e10737a92c4aadcb/pi0_base/'+quote(e['key'].removeprefix('checkpoints/pi0_base/'),safe='/')
        if e['size']<=chunk:
            jobs.append((dest,e['size'],[(mirror,False),(url,False),(fallback,True)],etag,'md5',0,None))
        else:
            parts=dest.with_name(dest.name+'.parts');parts.mkdir(parents=True,exist_ok=True)
            for start in range(0,e['size'],chunk):
                jobs.append((parts/f'{start//chunk:05d}',min(chunk,e['size']-start),[(mirror,False),(url,False),(fallback,True)],None,'sha256',start,e['size']))
        assembled.append((dest,e))
    started=time.monotonic();done=0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(transfer,*job) for job in jobs]
        for f in cf.as_completed(futures):
            f.result();done+=1
            if done%20==0 or done==len(jobs):
                print(json.dumps({'kind':'pi0','chunks':done,'total_chunks':len(jobs),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
                (LOG/'pi0-progress.json').write_text(json.dumps({'chunks':done,'total_chunks':len(jobs)}))
    for dest,e in assembled:
        etag=e['etag'].strip('"')
        if not dest.exists():
            tmp=dest.with_name(dest.name+'.assembling')
            with tmp.open('wb') as out:
                for i in range((e['size']+chunk-1)//chunk):
                    with (dest.with_name(dest.name+'.parts')/f'{i:05d}').open('rb') as src:
                        for b in iter(lambda:src.read(chunk),b''): out.write(b)
            os.replace(tmp,dest)
        if '-' in etag:
            hashes=[]
            with dest.open('rb') as f:
                for b in iter(lambda:f.read(chunk),b''): hashes.append(hashlib.md5(b).digest())
            actual=hashlib.md5(b''.join(hashes)).hexdigest()+f'-{len(hashes)}'
        else:actual=digest(dest,'md5')
        if actual!=etag:raise RuntimeError(f'S3 ETag verification failed: {dest}')
        print('VERIFIED',str(dest.relative_to(root)),flush=True)
    print('PI0 COMPLETE: all files verified against S3 ETags',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['dataset','pi0']);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    (dataset if args.kind=='dataset' else s3)(args.workers)
