import collections
import json
from pathlib import Path

base = Path('logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/provenance')
manifest = json.loads((base / 'manifest.json').read_text())
for suite, location in manifest['runs'].items():
    root = Path(location)
    records_path = root / f'{suite}.episodes.jsonl'
    rows = []
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    by_task = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        stats = by_task[row['task_description']]
        stats[0] += int(row['success'])
        stats[1] += 1
    print(json.dumps({'suite': suite, 'completed': len(rows), 'successes': sum(r['success'] for r in rows),
                      'by_task_success_total': by_task, 'final_results': (root / f'{suite}.json').exists()}, ensure_ascii=False))
    for name in ['server', suite]:
        path = root / f'{name}.log'
        if path.exists():
            lines = path.read_text(errors='replace').splitlines()
            print(name, ':', '\n'.join(lines[-3:]))
