"""Aggregate paired language/vision rollouts and create reports and a video browser."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import numpy as np

DEMO_NAMES={'milk':'牛奶','tomato_sauce':'番茄酱'}
LANG_NAMES={**DEMO_NAMES,'empty':'空指令'}
VISION_NAMES={'normal':'正常图像','mask_target':'遮挡基线操作物体','mask_other':'遮挡另一物体',
              'mask_background':'等面积背景遮挡','freeze_near':'接近后冻结双相机'}


def paired_change(a,b):
    lost=sum(x and not y for x,y in zip(a,b));gained=sum(y and not x for x,y in zip(a,b))
    n=lost+gained
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(lost,gained)+1))/(2**n)) if n else 1.
    return {'lost':lost,'gained':gained,'unchanged':len(a)-n,'exact_mcnemar_p_unadjusted':p}


def summarize(root,n):
    root=Path(root)
    groups={}
    for path in sorted(root.glob('rollouts/*/episodes.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows=sorted([r for r in rows if r['episode']<n],key=lambda r:r['episode'])
        assert len(rows)==n,path
        groups[path.parent.name]=rows
    assert len(groups)==14
    summaries=[]
    for name,rows in groups.items():
        r=rows[0]
        record={'condition':name,'demo':r['demo'],'language':r['language'],'vision':r['vision'],'n':n,
                'milk_in_basket':sum(x['ever_in_basket']['milk_1'] for x in rows),
                'tomato_in_basket':sum(x['ever_in_basket']['tomato_sauce_1'] for x in rows),
                'first_grasp_counts':dict(collections.Counter(x['actual_first_grasp'] or 'none' for x in rows)),
                'first_region_counts':dict(collections.Counter(x['first_region'] or 'none' for x in rows))}
        target=r['demo']+'_1'
        for event in ('approach','contact','grasp','lift','basket'):
            record['demo_object_'+event]=sum(x['events'][target][event] is not None for x in rows)
        base=groups[f"demo_{r['demo']}__language_milk__normal"]
        record['paired_demo_basket_vs_milk_normal']=paired_change(
            [x['ever_in_basket'][target] for x in base],[x['ever_in_basket'][target] for x in rows])
        trajectory_diff=[]
        for row,ref in zip(rows,base):
            with np.load(root/'rollouts'/name/f"ep{row['episode']:03d}"/'trajectory.npz') as a, np.load(
                    root/'rollouts'/ref['condition']/f"ep{row['episode']:03d}"/'trajectory.npz') as b:
                trajectory_diff.append(float(np.linalg.norm(a['eef_and_gripper'][:,:3]-b['eef_and_gripper'][:,:3],axis=1).mean()))
        record['mean_paired_eef_path_difference_m']=float(np.mean(trajectory_diff))
        if r['vision']=='freeze_near':
            record['freeze_steps']=[x['freeze_step'] for x in rows]
            record['baseline_grasp_steps']=[x['events'][target]['grasp'] for x in base]
        if r['vision'].startswith('mask_'):
            infos=[q['intervention'][key] for x in rows for q in x['requests'] for key in ('observation/image','observation/wrist_image')]
            record['background_nonrectangle_views']=sum(not x.get('mask_is_rectangle',True) for x in infos)
            record['area_capped_views']=sum(x.get('area_cap_active',False) for x in infos)
            record['views_with_robot_occluded']=sum(x['robot_pixels_occluded']>0 for x in infos)
            record['mean_robot_pixels_occluded']=float(np.mean([x['robot_pixels_occluded'] for x in infos]))
            record['mean_masked_pixels']=float(np.mean([x['masked_pixels'] for x in infos]))
        summaries.append(record)
    paired=[]
    for path in root.glob('servers/*/attention/**/paired_language_actions.npz'):
        condition=path.parent.parent.name
        with np.load(path) as z:
            for other in ('tomato_sauce','empty'):
                delta=z[other][:5]-z['milk'][:5]
                paired.append({'condition':condition,'observation':path.parent.name,'comparison':other+'-milk',
                               'first5_rmse_all7':float(np.sqrt(np.mean(delta**2))),
                               'first5_max_abs':float(np.abs(delta).max()),
                               'first5_translation_l2_mean':float(np.linalg.norm(delta[:,:3],axis=1).mean())})
    result={'n_per_condition':n,'physical_rollouts':14*n,'conditions':summaries,'same_observation_language_actions':paired}
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    cols=['condition','n','milk_in_basket','tomato_in_basket','demo_object_contact','demo_object_grasp','demo_object_lift','demo_object_basket','mean_paired_eef_path_difference_m']
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    text=['# 实验 3、4：语言与当前视觉干预','',f'共 {14*n} 条完整轨迹，每条件 {n} 个配对初始状态。两个实验均使用原始检查点，不依赖实验 2。',
          '', '固定番茄酱场景、原始布局；两条示范为牛奶 episode 814、番茄酱 episode 821。每回合 280 个控制步，20 Hz，每 5 步重新规划。正常图像回合在两个实验之间共用，不重复计数。',
          '', '## 实验 3：只改变语言','', '| 示范 | 语言 | 牛奶入篮 | 番茄酱入篮 | 示范物体夹持 / 抬起 |', '|---|---|---:|---:|---:|']
    for r in summaries:
        if r['vision']=='normal':
            text.append(f"| {DEMO_NAMES[r['demo']]} | {LANG_NAMES[r['language']]} | {r['milk_in_basket']}/{n} | {r['tomato_in_basket']}/{n} | {r['demo_object_grasp']}/{n} · {r['demo_object_lift']}/{n} |")
    text+=['','不同指令保持同一个噪声序列、示范及初始状态；完整轨迹中后续观测会自然分岔。另在正常牛奶指令轨迹的第 0/80/160 步（初始状态 0–4）固定同一观测，分别推理三条指令，保存独立的动作差异。这些额外动作不在环境中执行。',
           '', '“空指令”是空字符串，保留 tokenizer 自身的特殊 token；不是手动置零语言 embedding。有效 token 数和后续位置编号可能随指令长度变化，因此不能将结果解释为词义本身的纯效应。',
           '', '## 实验 4：当前图像干预','', '语言始终为牛奶。遮挡对象由同一初始状态、同一示范的正常图像基线决定：优先首次双侧夹持的物体；若未夹持，使用首次接触物体并单独记录。遮挡两路当前相机，不改变示范、机器人 state 或模拟器物体。',
           '', '| 示范 | 视觉条件 | 牛奶入篮 | 番茄酱入篮 | 示范物体接触 / 夹持 / 抬起 | 基线入篮 → 未入篮 / 反向 |', '|---|---|---:|---:|---:|---:|']
    for demo in DEMO_NAMES:
        for vision in VISION_NAMES:
            r=next(x for x in summaries if x['demo']==demo and x['language']=='milk' and x['vision']==vision)
            pair=r['paired_demo_basket_vs_milk_normal']
            text.append(f"| {DEMO_NAMES[demo]} | {VISION_NAMES[vision]} | {r['milk_in_basket']}/{n} | {r['tomato_in_basket']}/{n} | {r['demo_object_contact']} / {r['demo_object_grasp']} / {r['demo_object_lift']} | {pair['lost']} / {pair['gained']} |")
    text+=['','冻结条件在夹爪中心与基线操作物体中心距离首次不超过 12 cm 的规划时刻触发，随后两路图像一直保持该帧，机器人 state 仍更新。它干预接近后的整个过程，包括抓取和搬运，不能单独定位为某个抓取微调环节。',
           '', '物体遮挡使用可见物体包围矩形、填充 RGB 127，会保留位置线索，也可能同时挡住夹爪。背景对照优先同尺寸矩形；若腕部视角放不下，使用严格等面积且避开物体/机器人的背景像素区域。后者形状不同；若背景总面积不足，目标与背景遮挡都使用同一面积上限，此时物体可能仅被部分遮挡。形状变化、面积上限和遮挡到夹爪的次数记录在 comparison.json。',
           '', '## 行为指标与解释范围','', '- 接触：任意夹爪接触几何体与物体接触。夹持：robosuite 双侧指垫接触判据。抬起：物体中心较初始位置升高至少 3 cm，需结合接触记录解释。入篮：任一步 LIBERO In 判据成立。',
           '- 六个物体都记录了事件，不能把操作干扰物计为单纯无动作。第一接近区域采用初始位置的几何阈值，是接近行为的代理指标。',
           '- 配对变化使用相同初始状态索引逐回合比较，保留失败回合。comparison.json 还提供未做多重比较校正的描述性精确 McNemar p 值，不据此挑选结论。',
           '- 只覆盖一个场景、原始布局及两条固定示范；不能泛化为所有任务或所有视觉条件。',
           '', '## 验证与可复查文件','', '- [配对与干预审计](audit_full_25.json)：初始观测、示范及 56 次噪声逐项核对；正常图像基线与历史 280 步动作/状态核对；冻结前动作相同、冻结后图像不变。',
           '- [完整统计](comparison.json)、[CSV 表格](comparison.csv)。',
           '- [视频浏览器](index.html)：环境双相机与模型实际输入同时查看；均按真实模拟时间播放。',
           '- [代码修改报告](CODE_CHANGES.md)。原始轨迹在 rollouts；首批 attention 在 servers/*/attention。',
           '- attention 仅作辅助：平均 8 heads 和前 5 个 action queries，保存 flow steps 0/5/9、layers 0/9/17；初始状态 0 保留完整 heads/queries。注意力权重本身不构成因果贡献。', '']
    (root/'EXPERIMENT_RESULTS.md').write_text('\n'.join(text))
    browser=[]
    for rows in groups.values():
        for r in rows:
            browser.append({k:r[k] for k in ('condition','episode','demo','language','vision','events','first_region','actual_first_grasp','freeze_step')})
    html='''<!doctype html><meta charset="utf-8"><title>实验3/4 配对视频</title><style>body{font:16px system-ui;margin:24px;background:#f6f7fa;color:#172238}video{width:48%;max-width:900px}select,button{font:inherit;margin:8px}pre{white-space:pre-wrap}small{display:block}</style>
<h1>语言与当前视觉干预</h1><p>左：模拟器当前画面。右：模型实际接收的画面。两者均为前视＋腕部相机，14 秒原速。右侧每 5 个控制步更新一次。</p><select id="condition"></select><select id="episode"></select><button id="play">同步播放 / 暂停</button><br><video id="environment" controls preload="metadata"></video><video id="input" controls preload="metadata"></video><small>拖动左侧进度条会同步右侧；冻结条件中右侧画面应停止变化，而左侧继续运动。注意：视频为压缩显示，精确输入保存在 NPZ。</small><pre id="info"></pre><script>
const data=DATA;
const conditions=[...new Set(data.map(x=>x.condition))];const c=document.getElementById('condition'),e=document.getElementById('episode'),v=document.getElementById('environment'),p=document.getElementById('input');
conditions.forEach(x=>c.add(new Option(x,x)));for(let i=0;i<COUNT;i++)e.add(new Option('初始状态 '+i,i));
function update(){const r=data.find(x=>x.condition===c.value&&x.episode===+e.value);const base='rollouts/'+r.condition+'/ep'+String(r.episode).padStart(3,'0')+'/';v.src=base+'environment.mp4';p.src=base+'policy_input.mp4';document.getElementById('info').textContent=JSON.stringify(r,null,2);}
c.onchange=e.onchange=update;v.onseeked=()=>{p.currentTime=v.currentTime};document.getElementById('play').onclick=()=>{if(v.paused){p.currentTime=v.currentTime;v.play();p.play()}else{v.pause();p.pause()}};update();
</script>'''.replace('DATA',json.dumps(browser,ensure_ascii=False)).replace('COUNT',str(n))
    (root/'index.html').write_text(html)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--n',type=int,default=25)
    a=p.parse_args();summarize(a.run,a.n)
