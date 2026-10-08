from pathlib import Path
import json, shutil, os

root=Path(Path('.cache/no_context_zero_root').read_text().strip())
base=Path('logs/quickstart/context_ablation_object_correct_no_wrong_25trials_20260922_154452').resolve()
result=json.loads((root/'comparison.json').read_text())
assert result['verified_pairs']==50
assert json.loads((root/'server/SERVER_FINISHED.json').read_text())['passed']
new={r['task']:r for r in result['comparison'] if r['condition']=='no_zero_keep_mask'}
new_rel=Path(os.path.relpath(root,base))
backup=root/'provenance/reports_before_update';backup.mkdir(parents=True,exist_ok=True)
for p in [base/'REPORT.md',base/'run.json',Path('EVALUATION_REPORT.md').resolve()]:
    dest=backup/(p.name if p.parent==base else 'global_'+p.name)
    if not dest.exists():shutil.copy2(p,dest)
s=(backup/'REPORT.md').read_text()
s=s.replace('2026-09-22。已完成 6 组，每组 25 回合，共 150 回合；均有 20 FPS 视频和动作轨迹。',
    '原实验：2026-09-22，6 组各 25 次，共 150 次。2026-09-23 补跑保留 mask/位置编号的 no-context：两个任务各 25 次，共新增 50 次。以下主矩阵已采用新 no，correct/wrong 复用原记录；旧 no 数据保留。所有回放均为 20 FPS。')
countsum=sum(r['milk_in_basket']+r['tomato_sauce_in_basket'] for r in new.values())
conclusion=('**补充结果：新 no 的两个任务仍均为牛奶 0/25、番茄酱 0/25。零成功率在保留 mask 和位置编号的置零干预中也成立，不能只用旧版位置编号变化解释。**'
            if countsum==0 else '**补充结果：新 no 出现了入篮行为，因此旧 masked no 的零成功率不能推广到所有无示范干预。请以更新后的主表为准。**')
s=s.replace('## 结论\n','## 结论\n\n'+conclusion+'\n\n详细定义、验收和新旧对照见[补充实验报告]('+str(new_rel/'EXPERIMENT_RESULTS.md')+')。\n',1)
s=s.replace('## 六组矩阵','## 六组矩阵（no 更新为表示置零、保留 mask）')
for task,cn in [('milk','牛奶'),('tomato_sauce','番茄酱')]:
    r=new[task]
    s=s.replace(f'| {cn} | 无（no） | 0/25 | 0/25 |',f'| {cn} | 无示范内容（新 no：保留 mask） | {r["milk_in_basket"]}/25 | {r["tomato_sauce_in_basket"]}/25 |')
s=s.replace('其他五组的主表只报告牛奶和番茄酱目标，没有将未全面复核的其他物体结果写成零。',
    '主表只报告牛奶和番茄酱。新 no 另记录了全部场景物体，见补充报告；历史 correct/wrong 的其他物体结论仍按原回放核对范围解释。')
s=s.replace('![六组结果](comparison.png)', '![正确示范、新 no、错误示范及旧 no 对照](comparison_no_context_zero.png)\n\n旧 no（关闭 mask）在两个任务中均为牛奶 0/25、番茄酱 0/25；原图 `comparison.png` 和原 `comparison.csv/json` 保留为历史记录。新版汇总见 [comparison_no_context_zero.json](comparison_no_context_zero.json) 和 [CSV](comparison_no_context_zero.csv)。')
s=s.replace('- no 组删除的是三种示范模态，仍保留正确语言、实时图像和本体状态。其失败还可能包含\n  未训练过的无示范输入分布影响，不能简单等同于“模型不理解语言”。',
    '- 新 no 在 Gemma 前将三种示范模态的全部编码表示置零，保留原 mask、token 槽位和位置编号，正确语言、实时图像和本体状态不变。旧 no 则将示范数据置零并关闭 mask，没有物理删除槽位，但有效位置编号改变。两者均可能受到未训练过的输入分布影响；失败不能简单等同于“模型不理解语言”。')
s=s.replace('## 同一初始状态的回放（episode 001）','## 原实验同一初始状态的回放（episode 001；此处 no 为旧 masked 版本）')
s=s.replace('文件名中的 `neither` 仅表示牛奶和番茄酱均未入篮，不表示没有操作其他物体。',
    '新旧 no 的 50 对视频见[配对视频入口]('+str(new_rel/'VIDEO_INDEX.md')+')。\n\n文件名中的 `neither` 仅表示牛奶和番茄酱均未入篮，不表示没有操作其他物体。')
s=s.replace('- 每个任务使用初始状态索引 0–24；三种条件保留相同环境和语言，只干预示范。',
    '- 每个任务使用初始状态索引 0–24；新 no 与历史 correct/no/wrong 逐一核对初始状态、观察哈希和每次推理噪声一致。两个任务各自使用原任务场景，未混用固定番茄酱场景或交换位置的结果。')
s=s.replace('## 验证与记录','## 原始 150 回合的验证与记录（补跑验收见补充报告）')
s=s.replace('## 再运行\n\n`bash quickstart.sh` → **9** → 每组次数（默认 25）→ 检查点、GPU、端口。',
    '## 再运行\n\n本次新 no 使用[独立启动说明](/data/zjc/workspace/ContextFlow/scripts/NO_CONTEXT_ZERO.md)。\n\n`bash quickstart.sh` → **9** → 每组次数（默认 25）→ 检查点、GPU、端口，仍对应旧 masked no 定义，不能用它冒充本次保留 mask 的对照。')
(base/'REPORT.md').write_text(s)
for name in ['comparison.json','comparison.csv','comparison.png']:
    dest=base/('comparison_no_context_zero'+Path(name).suffix)
    shutil.copy2(root/name,dest)
run=json.loads((backup/'run.json').read_text())
run['no_context_zero_supplement']={'root':str(root),'trials_per_task':25,'verified_pairs':50,'intervention':result['intervention'],'legacy_no_raw_data_preserved':True,'updated_comparison':'comparison_no_context_zero.json'}
(base/'run.json').write_text(json.dumps(run,indent=2)+'\n')
global_path=Path('EVALUATION_REPORT.md')
global_text=(backup/'global_EVALUATION_REPORT.md').read_text()
global_text+='\n## 2026-09-23：no-context 保留位置编号的补充对照\n\n'+conclusion+'\n\n原 Correct / No / Wrong 实验的 no 条件在各自原场景各补跑 25 次。correct/wrong 沿用原记录，旧 masked no 保留作对照。详见[补充报告]('+str(Path(os.path.relpath(root,global_path.resolve().parent))/'EXPERIMENT_RESULTS.md')+')。固定场景的位置交换 no 条件没有在本次重跑，不能混用这些数值。\n'
global_path.write_text(global_text)
print('Updated original report, explicit new comparison files, run metadata, and EVALUATION_REPORT.md; historical raw data preserved.')
