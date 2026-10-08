import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

base = Path('logs/quickstart/reproduction_table1_contextflow_spatial_object_50trials_20260921_210409/provenance')
manifest = json.loads((base / 'manifest.json').read_text())
tasks = {}
videos_total = 0
for suite, location in manifest['runs'].items():
    root = Path(location)
    results = json.loads((root / f'{suite}.json').read_text())
    trials = [json.loads(line) for line in (root / f'{suite}.episodes.jsonl').read_text().splitlines()]
    assert len(trials) == 100
    assert results['summary']['total_episodes'] == 100
    assert results['summary']['num_unseen_tasks'] == 2
    assert results['summary']['total_successes'] == sum(row['success'] for row in trials)
    for entry in results['per_task_results']:
        rows = [row for row in trials if row['task_description'] == entry['task_description']]
        assert len(rows) == entry['episodes'] == 50
        assert sorted(row['episode_index'] for row in rows) == list(range(50))
        assert sum(row['success'] for row in rows) == entry['successes']
        assert entry['category'] == 'unseen'
        tasks[entry['task_description']] = dict(entry, suite=suite)
    for name in ['server', suite]:
        text = (root / f'{name}.log').read_text(errors='replace')
        assert 'Traceback (most recent call last)' not in text
        assert 'Evaluation error' not in text
    videos = list((root / 'videos' / suite).rglob('*.mp4'))
    assert len(videos) == 100 and all(video.stat().st_size > 1000 for video in videos)
    videos_total += len(videos)
for name, expected in manifest['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name

paper = [
    ('pick up the black bowl on the cookie box and place it on the plate', 'Spatial：将饼干盒上的黑碗放到盘子上', 86),
    ('pick up the black bowl next to the plate and place it on the plate', 'Spatial：将盘子旁的黑碗放到盘子上', 42),
    ('pick up the milk and place it in the basket', 'Object：将牛奶放入篮子', 76),
    ('pick up the tomato sauce and place it in the basket', 'Object：将番茄酱放入篮子', 90),
]
comparison = []
report_path = Path('EVALUATION_REPORT.md')
report = report_path.read_text()
report = report.replace('状态：正在运行，最终结果完成后补齐。', '状态：已完成，四个任务各 50 次，共 200 次，运行异常为 0。')
for description, label, reference in paper:
    item = tasks[description]
    rate = item['successes'] * 2
    comparison.append(dict(item, paper_rate_percent=reference, actual_rate_percent=rate, delta_percentage_points=rate-reference))
    report = report.replace(f'| {label} | {reference}% | 待完成 | 待完成 |',
                            f"| {label} | {reference}% | {item['successes']} / 50 | {rate}% |")
successes = sum(row['successes'] for row in comparison)
rate = successes / 2
report = report.replace('| 四任务平均 | 73.5% | 待完成 / 200 | 待完成 |',
                        f'| 四任务平均 | 73.5% | {successes} / 200 | {rate:g}% |')
report_path.write_text(report)
summary = {
    'completed_at': datetime.datetime.now().astimezone().isoformat(),
    'total_episodes': 200, 'successes': successes, 'success_rate_percent': rate,
    'paper_success_rate_percent': 73.5, 'delta_percentage_points': rate-73.5,
    'execution_errors': 0, 'videos': videos_total, 'tasks': comparison,
    'suite_rates_percent': {s: sum(t['successes'] for t in comparison if t['suite'] == s) for s in manifest['runs']},
}
(base / 'comparison.json').write_text(json.dumps(summary, indent=2))
with (base / 'comparison.csv').open('w', newline='') as f:
    fields = ['suite', 'task_description', 'successes', 'episodes', 'actual_rate_percent', 'paper_rate_percent', 'delta_percentage_points']
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(comparison)
print(json.dumps(summary, indent=2))
