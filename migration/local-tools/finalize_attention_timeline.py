from pathlib import Path
import datetime,hashlib,json,shutil,time,zipfile
from scripts.attention.build_timeline import report,contact_sheet
root=Path(Path('.cache/attention_timeline_root').read_text().strip());out=root/'viewer'
for _ in range(720):
 if (out/'build_complete.json').exists() and (root/'artifact_audit.json').exists():break
 time.sleep(5)
else:raise TimeoutError('Final timeline checks did not finish')
results=json.loads((root/'capture/results.json').read_text());audit=json.loads((root/'artifact_audit.json').read_text())
assert results['pairs']==840 and results['passed']
assert results['max_raw_action_difference']==results['max_executable_action_difference']==results['max_historical_action_difference']==0
assert audit['previous_experiment0_exact_reproductions']==45
summaries=[]
for ep in range(5):summaries.extend(json.loads((out/f'episode_{ep:03d}_summary.json').read_text()))
report(root,out,summaries)
contact_sheet(out,{})
(out/'VALIDATION.md').write_text('''# 完整轨迹验收

- 15 条完整回放全部 280 步的机器人、物体位置误差为 0，入篮判据完全一致。
- 4215 个状态帧；840 个规划观测与该时刻显示的无损帧完全一致，无前后一步错位。
- 840 次普通/记录推理完整动作逐字节一致；与原实验实际执行动作的最大差异为 0。
- 模型参数前后 SHA256 一致。与原实验 0 重叠的 45 个时刻，其完整 attention、噪声、动作、mask 和观测均逐元素一致。
- 已对每个初始状态的页面检查全部 281 个控制步与最近规划帧的映射；另检查 36 种层/采样步/颜色/显示组合、事件元数据、终点 280、页面切换及 20Hz 播放逻辑。工具为 Node 的 DOM/canvas 模拟，不代表真实浏览器性能测量。
- 静态关键帧 PNG 已实际查看；新增脚本 Ruff 和 git diff --check 通过；attention 相关 6 项自动测试通过。
- 本次 inference 从冻结代码副本导入。网页 JPEG 和 float32 传输格式仅用于显示，不用于模型推理；float64 汇总数组保留。

完整原始验收记录位于上一级的 `capture/results.json`、`artifact_audit.json`、`frame_audit.json` 和各 `timeline_ui_check*.log`。
''')
files=['examples/libero/prepare_attention_timeline.py','scripts/check_attention_capture.py','scripts/attention/build_timeline.py',
       'scripts/attention/trajectory_timeline.html','scripts/attention/README.md']
provenance={}
for file in files:
 p=Path(file);dest=root/'provenance/final_source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 provenance[file]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'provenance/final_source_sha256.json').write_text(json.dumps(provenance,indent=2)+'\n')
for name in ['check_attention_timeline.cjs','audit_attention_timeline.py','audit_timeline_frames.py','finalize_attention_timeline.py']:
 shutil.copyfile(Path('.cache')/name,root/'provenance'/name)
run=json.loads((root/'run.json').read_text());run.update(status='passed',completed_at=datetime.datetime.now().astimezone().isoformat(),
 result=results,exact_repeated_experiment0_captures=45,viewer='viewer/index.html')
(root/'run.json').write_text(json.dumps(run,indent=2)+'\n')
with zipfile.ZipFile(root/'viewer_offline.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
 for pattern in ['*.html','*.md','keyframes_*.png','episode_*_summary.json','temporal_spatial_summary.json']:
  for p in sorted(out.glob(pattern)):archive.write(p,p.name)
print('Finalized all reports, provenance and offline viewer archive',flush=True)
