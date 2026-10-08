"""Independent paired-input, historical action, and visual intervention audit."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root, n, include_vision, only_demo=None):
    root = Path(root)
    rows = {}
    for path in root.glob('rollouts/*/episodes.jsonl'):
        parsed = [json.loads(line) for line in path.read_text().splitlines()]
        rows[path.parent.name] = {r['episode']: r for r in parsed if r['episode'] < n}
    expected = [(d, l, 'normal') for d in ('milk', 'tomato_sauce') for l in ('milk', 'tomato_sauce', 'empty')]
    if include_vision:
        expected += [(d, 'milk', v) for d in ('milk', 'tomato_sauce') for v in ('mask_target', 'mask_other', 'mask_background', 'freeze_near')]
    if only_demo:
        expected = [entry for entry in expected if entry[0] == only_demo]
    selected = []
    for d, l, v in expected:
        key = f'demo_{d}__language_{l}__{v}'
        assert set(rows[key]) == set(range(n)), key
        selected.extend(rows[key].values())
    for ep in range(n):
        group = [r for r in selected if r['episode'] == ep]
        assert len({r['initial_state_hash'] for r in group}) == 1
        assert len({r['initial_observation_hash'] for r in group}) == 1
        for step in range(56):
            assert len({r['requests'][step]['noise_hash'] for r in group}) == 1
        for demo in sorted({entry[0] for entry in expected}):
            drows = [r for r in group if r['demo'] == demo]
            assert len({q['demo_hash'] for r in drows for q in r['requests']}) == 1
            langs = [r for r in drows if r['vision'] == 'normal']
            assert len({r['requests'][0]['nonlanguage_hash'] for r in langs}) == 1
            assert len({json.dumps(r['requests'][0]['language_tokens']) for r in langs}) == 3
    historical = Path('logs/quickstart/position_swap_object_25trials_20260922_175506/original')
    historical_exact = []
    frozen_exact = []
    for r in selected:
        ep = r['episode']; condition = r['condition']
        with np.load(root/'rollouts'/condition/f'ep{ep:03d}'/'trajectory.npz') as archive:
            z = {key: archive[key] for key in archive.files}
            assert z['actions'].shape == (280, 7) and np.isfinite(z['actions']).all()
            assert z['policy_images'].shape == (56, 224, 448, 3)
            for i, name in enumerate(z['object_names']):
                assert bool(z['in_basket'][:, i].any()) == r['ever_in_basket'][str(name)]
                for event, key in [('grasp','grasp'), ('contact','contact'), ('basket','in_basket')]:
                    found = np.flatnonzero(z[key][:, i])
                    assert r['events'][str(name)][event] == (int(found[0])+1 if len(found) else None)
            if r['vision'] == 'normal' and r['language'] == 'milk':
                mode = 'correct' if r['demo'] == 'milk' else 'wrong'
                old = next(x for x in map(json.loads,(historical/mode/'episodes.jsonl').read_text().splitlines())
                           if x['task']=='milk' and x['episode_index']==ep)
                with np.load(historical/old['trajectory']) as previous:
                    np.testing.assert_array_equal(z['actions'], previous['actions'])
                    np.testing.assert_array_equal(z['eef_and_gripper'], previous['eef_and_gripper'])
                historical_exact.append([condition,ep])
            if r['vision'].startswith('mask_'):
                for t, q in enumerate(r['requests']):
                    for camera, key in enumerate(('observation/image','observation/wrist_image')):
                        info=q['intervention'][key]
                        assert int(z['pixel_masks'][t,camera].sum()) == info['masked_pixels']
                        image=z['policy_images'][t,:,camera*224:(camera+1)*224]
                        assert (image[z['pixel_masks'][t,camera]]==127).all()
                        if r['vision']=='mask_background':
                            assert info['masked_pixels']==info['target_box_pixels']
                            assert info['robot_pixels_occluded']==0
            if r['vision']=='freeze_near':
                step=r['freeze_step']
                basekey=f"demo_{r['demo']}__language_milk__normal"
                with np.load(root/'rollouts'/basekey/f'ep{ep:03d}'/'trajectory.npz') as baseline:
                    end = 280 if step is None else step
                    np.testing.assert_array_equal(z['actions'][:end],baseline['actions'][:end])
                if step is not None:
                    for frame in z['policy_images'][step//5:]:
                        np.testing.assert_array_equal(frame,z['policy_images'][step//5])
                frozen_exact.append([condition,ep,step])
    captures = list(root.glob('servers/*/attention/**/metadata.json'))
    for path in captures:
        metadata=json.loads(path.read_text())
        assert metadata['checks']['masked_key_max']==0
    result={'passed':True,'conditions':len(expected),'episodes':len(selected),
            'paired_initial_states_and_observations_exact':True,'all_56_noise_hashes_paired':True,
            'demo_inputs_constant':True,'language_only_initial_nonlanguage_inputs_exact':True,
            'historical_280_action_and_state_traces_exact':len(historical_exact),
            'freeze_prefix_and_stale_images_checked':frozen_exact,'attention_checks':len(captures)}
    out=root/f"audit_{'full' if include_vision else 'language'}_{n}{'_' + only_demo if only_demo else ''}.json"
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='freeze_prefix_and_stale_images_checked'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--n',type=int,default=5)
    p.add_argument('--vision',action='store_true');p.add_argument('--demo',choices=('milk','tomato_sauce'));a=p.parse_args();audit(a.run,a.n,a.vision,a.demo)
