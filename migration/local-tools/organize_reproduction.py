"""Organize the completed Table 1 run and correct video timestamps without re-encoding."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tarfile

import av

root = Path(__file__).resolve().parents[1]
run = root / 'logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409'
mapping = {
    root / 'logs/quickstart/20260921_210409_Oz66aI': run / 'libero_spatial',
    root / 'logs/quickstart/20260921_210415_r29Qv3': run / 'libero_object',
    root / 'logs/reproduction/20260921_contextflow': run / 'provenance',
}
run.mkdir(exist_ok=True)
for old, new in mapping.items():
    if old.exists():
        assert not new.exists(), new
        shutil.move(str(old), str(new))
    assert new.is_dir(), new

# Update navigation and manifests; historical raw logs / source hashes stay intact.
paths = [root / 'EVALUATION_REPORT.md', root / 'ENVIRONMENT_SETUP.md',
         run / 'provenance/manifest.json',
         root / '.cache/monitor_reproduction.py', root / '.cache/finalize_reproduction.py']
for path in paths:
    text = path.read_text()
    for old, new in mapping.items():
        text = text.replace(str(old.relative_to(root)), str(new.relative_to(root)))
    path.write_text(text)

videos = sorted(run.glob('libero_*/videos/libero_*/**/*.mp4'))
assert len(videos) == 200
backup = run / 'provenance/original_10fps_videos.tar.gz'
if not backup.exists():
    with tarfile.open(backup, 'x:gz') as archive:
        for video in videos:
            archive.add(video, arcname=str(video.relative_to(run)))

def inspect(path):
    with av.open(str(path)) as container:
        assert len(container.streams) == 1 and len(container.streams.video) == 1
        stream = container.streams.video[0]
        digest = hashlib.sha256()
        packets = 0
        for packet in container.demux(stream):
            if packet.size:
                digest.update(bytes(packet))
                packets += 1
        return {'fps': float(stream.average_rate), 'frames': stream.frames,
                'duration': float(stream.duration * stream.time_base),
                'packets': packets, 'encoded_frame_sha256': digest.hexdigest()}

def retime(path):
    before = inspect(path)
    if before['fps'] == 20:
        return {'path': str(path.relative_to(run)), 'already_20fps': True, 'after': before}
    assert before['fps'] == 10, (path, before)
    data = bytearray(path.read_bytes())
    changed = []

    def visit(start, end):
        position = start
        while position < end:
            size, kind = struct.unpack_from('>I4s', data, position)
            header = 8
            if size == 1:
                size = struct.unpack_from('>Q', data, position + 8)[0]
                header = 16
            if size == 0:
                size = end - position
            assert size >= header and position + size <= end
            payload = position + header
            if kind in (b'mvhd', b'mdhd'):
                version = data[payload]
                assert version in (0, 1)
                offset = payload + (20 if version == 1 else 12)
                scale = struct.unpack_from('>I', data, offset)[0]
                assert 0 < scale < 2**31
                struct.pack_into('>I', data, offset, scale * 2)
                changed.append(kind)
            elif kind in (b'moov', b'trak', b'mdia'):
                visit(payload, position + size)
            position += size

    visit(0, len(data))
    assert sorted(changed) == [b'mdhd', b'mvhd']
    temporary = path.with_name(path.stem + '.retiming.mp4')
    temporary.write_bytes(data)
    after = inspect(temporary)
    assert after['fps'] == 20
    assert after['frames'] == before['frames']
    assert after['packets'] == before['packets']
    assert after['encoded_frame_sha256'] == before['encoded_frame_sha256']
    assert abs(after['duration'] * 2 - before['duration']) < 1e-6
    os.replace(temporary, path)
    return {'path': str(path.relative_to(run)), 'before': before, 'after': after}

results = []
for number, video in enumerate(videos, 1):
    results.append(retime(video))
    if number % 50 == 0:
        print(f'Corrected and verified {number}/200 videos', flush=True)
record = {'updated_at': datetime.datetime.now().astimezone().isoformat(),
          'renamed_paths': {str(k): str(v) for k, v in mapping.items()},
          'video_fps': 20, 'method': 'MP4 movie/media timescale doubled; encoded frames unchanged',
          'original_videos_archive': str(backup.relative_to(run)), 'videos': results}
(run / 'provenance/video_timing_update.json').write_text(json.dumps(record, indent=2))
manifest = json.loads((run / 'provenance/manifest.json').read_text())
manifest['video_postprocessing'] = {'original_fps': 10, 'current_fps': 20, 'record': 'video_timing_update.json'}
(run / 'provenance/manifest.json').write_text(json.dumps(manifest, indent=2))
(run / 'run.json').write_text(json.dumps({
    'config': 'ContextFlow', 'checkpoint': manifest['checkpoint'], 'num_trials_per_task': 50,
    'client_seed': 7, 'policy_rng_initial_seed_per_suite': 0, 'fresh_server_per_suite': True,
    'video_fps': 20, 'suites': list(manifest['runs']),
    'results': {suite: f'{suite}/{suite}.json' for suite in manifest['runs']},
    'provenance': 'provenance/manifest.json',
}, indent=2))
print('Completed:', run)
