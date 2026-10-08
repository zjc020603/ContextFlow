from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess
import numpy as np
root=Path.cwd();run=Path((root/'.cache/language_vision_root').read_text().strip())
audit=json.loads((run/'audit_full_25.json').read_text());assert audit['passed'] and audit['episodes']==350
finished=[json.loads(p.read_text()) for p in run.glob('servers/*/FINISHED.json')]
assert len(finished)==8 and all(x['unchanged'] for x in finished)
assert len({x['parameter_sha256'] for x in finished})==1
assert json.loads((run/'html_validation.json').read_text())['passed']
assert json.loads((run/'video_validation.json').read_text())['passed']
mean_checks=0
for path in run.glob('servers/*/attention/**/capture.npz'):
 with np.load(path) as z:
  if 'probabilities' in z.files:
   expected=z['probabilities'].astype(np.float64)[:,:,0,:,1:6].mean((2,3))
   np.testing.assert_array_equal(expected,z['mean_attention']);mean_checks+=1
assert mean_checks==126
files=[Path('scripts/serve_language_vision.py'),Path('examples/libero/language_vision_experiment.py'),Path('examples/libero/live_vision_intervention.py'),Path('examples/libero/live_vision_intervention_test.py')]+list(Path('scripts/experiments').glob('*.py'))+[Path('scripts/experiments/LANGUAGE_VISION.md')]
manifest={}
for file in files:
 dst=run/'provenance/final_source'/file;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dst);manifest[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
(run/'provenance/final_source_sha256.json').write_text(json.dumps(manifest,indent=2))
frozen=run/'provenance/frozen/src/openpi'
(run/'provenance/frozen_sha256.json').write_text(json.dumps({str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*.py')},indent=2))
for file in ('.cache/validate_language_vision.cjs','.cache/validate_language_vision_videos.py','.cache/finalize_language_vision.py'):
 shutil.copy2(file,run/'provenance'/Path(file).name)
code=run/'CODE_CHANGES.md';s=code.read_text();s=re.sub(r'`((?:scripts|examples)/[^`]+\.(?:py|md))`',lambda m:f'[{m.group(1)}]({root/m.group(1)})',s);code.write_text(s)
for f in (run/'EXPERIMENT_RESULTS.md',code):
 for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if '://' not in target:assert (f.parent/target).exists(),target
status={'status':'complete','completed_at':datetime.datetime.now().astimezone().isoformat(),'physical_rollouts':350,
        'conditions':14,'trials_per_condition':25,'language_rollouts':150,'vision_rollouts_including_shared_baselines':250,
        'shared_baseline_rollouts':50,'attention_records':270,'full_attention_records':126,'mean_attention_exact_checks':126,
        'historical_action_and_state_traces_exact':50,'parameter_sha256':finished[0]['parameter_sha256'],
        'servers_stopped':True,'unit_tests':8,'language_vision_does_not_depend_on_experiment2':True,
        'reports':['EXPERIMENT_RESULTS.md','CODE_CHANGES.md'],'failed_attempts_excluded':True}
(run/'run.json').write_text(json.dumps(status,indent=2))
proto=json.loads((run/'protocol.json').read_text());proto['status']='complete';(run/'protocol.json').write_text(json.dumps(proto,indent=2))
(run/'VALIDATION.md').write_text('''# 验证记录

- 350 条最终轨迹、14 个条件，每条件 25 次；失败开发尝试不在 rollouts 统计内。
- 初始观测/状态、示范内容和每回合 56 个噪声 hash 配对通过。
- 50 条正常图像轨迹的全部动作及机器人状态与历史记录精确一致。
- 50 条冻结轨迹在触发前动作与基线一致，触发后两相机输入数组恒定。
- 270 份 attention 概率/mask 检查通过；126 份保留全部头/query，独立重算其均值全部精确一致。
- 8 个 GPU 服务启动/结束参数 hash 一致，服务已停止。
- 8 个针对性单元测试通过；新增 Python 静态检查、模拟器 Python 3.8 编译和入口 --help 检查通过。
- 视频浏览器 350 种选择、attention 图 40 种选择与文件路径检查通过；播放/暂停/跳转逻辑经 Node DOM 模拟检查。没有实际浏览器交互/性能测试。
- 28 个代表视频检查：环境 20 fps、模型输入 4 fps，均 14 秒、448×224 双相机画面。其余视频存在性已检查。
- 已人工查看物体遮挡预览及 attention 图。报告链接检查通过。
''')
print(json.dumps(status,indent=2))
