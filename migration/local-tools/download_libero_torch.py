"""Resume a large official wheel in bounded, verified HTTP range requests."""
import concurrent.futures
import hashlib
import html
from pathlib import Path
import re
import time
from urllib.parse import unquote, urljoin, urlsplit
import requests

filename = 'torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl'
index = 'https://download.pytorch.org/whl/cu113/torch/'
response = requests.get(index, timeout=(15, 60))
response.raise_for_status()
links = [urljoin(index, html.unescape(x)) for x in re.findall(r'href="([^"]+)"', response.text)]
link = next(x for x in links if unquote(urlsplit(x).path).endswith('/' + filename))
expected = urlsplit(link).fragment.removeprefix('sha256=')
assert re.fullmatch('[a-f0-9]{64}', expected), 'Official wheel SHA256 missing'
url = link.split('#')[0]
with requests.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(15, 60)) as r:
    r.raise_for_status()
    assert r.status_code == 206, 'Server must support HTTP ranges'
    total = int(r.headers['Content-Range'].split('/')[-1])
chunk_size = 32 * 1024 * 1024
root = Path('.cache/libero-torch-download')
root.mkdir(parents=True, exist_ok=True)
output = root / filename
count = (total + chunk_size - 1) // chunk_size
print(f'Official wheel: {total / 2**20:.1f} MiB; {count} resumable chunks; sha256={expected}', flush=True)

def download(i):
    start, end = i * chunk_size, min((i + 1) * chunk_size, total) - 1
    part = root / f'{i:04d}.part'
    if part.exists() and part.stat().st_size == end - start + 1:
        return i
    for attempt in range(5):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                assert r.status_code == 206
                assert r.headers['Content-Range'] == f'bytes {start}-{end}/{total}', r.headers.get('Content-Range')
                with part.open('wb') as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
            assert part.stat().st_size == end - start + 1
            return i
        except Exception as exc:
            print(f'Chunk {i} retry {attempt + 1}: {exc}', flush=True)
            if attempt == 4:
                raise
            time.sleep(attempt + 1)

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for done, future in enumerate(concurrent.futures.as_completed([pool.submit(download, i) for i in range(count)]), 1):
        future.result()
        print(f'Downloaded {done}/{count} chunks', flush=True)
hash_ = hashlib.sha256()
with output.open('wb') as dest:
    for i in range(count):
        with (root / f'{i:04d}.part').open('rb') as src:
            while block := src.read(1024 * 1024):
                hash_.update(block)
                dest.write(block)
assert hash_.hexdigest() == expected, 'Official SHA256 verification FAILED'
print(f'SHA256 verified: {output}', flush=True)
