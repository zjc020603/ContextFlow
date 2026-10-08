"""Paired original-scene rollouts for language and live-camera interventions."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import imageio
import numpy as np
from libero.libero import benchmark
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from examples.libero import context_ablation as base, position_intervention
from examples.libero.live_vision_intervention import CAMERAS, intervene

PROMPTS = {**base.TASKS, 'empty': ''}
VISIONS = ('normal', 'mask_target', 'mask_other', 'mask_background', 'freeze_near')


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def condition_name(demo, language, vision):
    return f'demo_{demo}__language_{language}__{vision}'


def run(args):
    root = Path(args.output)
    condition = condition_name(args.demo, args.language, args.vision)
    directory = root / 'rollouts' / condition
    directory.mkdir(parents=True, exist_ok=True)
    record_file = directory / 'episodes.jsonl'
    existing = [json.loads(x) for x in record_file.read_text().splitlines()] if record_file.exists() else []
    assert not any(args.start <= r['episode'] < args.stop for r in existing), 'Refuse duplicate episodes'
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    assert client.get_server_metadata()['language_vision_protocol'] == 1
    suite = benchmark.get_benchmark_dict()['libero_object']()
    scene = suite.get_task(5)
    assert scene.language == base.TASKS['tomato_sauce']
    env, _ = base.baseline._get_libero_env(scene, 256, 7)
    init_states = suite.get_task_init_states(5)
    (directory / 'config.json').write_text(json.dumps(vars(args), indent=2))
    try:
        for ep in range(args.start, args.stop):
            started = time.monotonic()
            np.random.seed(28007 + ep)
            env.seed(28007 + ep)
            env.reset()
            env.set_init_state(init_states[ep])
            obs, layout = position_intervention.apply_layout(env, 'original')
            for _ in range(10):
                obs, _, _, _ = env.step(base.baseline.LIBERO_DUMMY_ACTION)
            position_intervention.validate_settled(env, layout)
            initial_state = env.get_sim_state().copy()
            names = list(base.scene_object_status(env))
            positions0 = np.array([env.env.object_states_dict[n].get_geom_state()['pos'].copy() for n in names])
            assert not any(base.scene_object_status(env).values())
            task = {'description': PROMPTS[args.language], 'index': 25}
            demo_index = 25 if args.demo == 'milk' else 28
            mode = 'correct' if demo_index == 25 else 'wrong'  # protocol anchor, independent of actual language
            baseline = None
            if args.vision != 'normal':
                path = root / 'rollouts' / condition_name(args.demo, 'milk', 'normal') / 'episodes.jsonl'
                baseline = next(r for r in map(json.loads, path.read_text().splitlines()) if r['episode'] == ep)
                assert baseline['initial_state_hash'] == digest(initial_state)
                target = baseline['actual_first_grasp'] or baseline['actual_first_contact']
                target_basis = 'bilateral_grasp' if baseline['actual_first_grasp'] else 'contact_without_grasp'
                if target not in ('milk_1', 'tomato_sauce_1'):
                    raise ValueError(f'Baseline ep{ep} has no observed target contact: {target}')
                other = 'tomato_sauce_1' if target == 'milk_1' else 'milk_1'
            else:
                target, other = args.demo + '_1', None
                target_basis = 'baseline_not_masked'
            freeze_step = None
            frozen = None
            records, actions, poses, eefs, grasps, touches, baskets, heights = [], [], [], [], [], [], [], []
            frames, policy_frames, wrist_frames, masks_saved = [], [], [], []
            first_grasp = first_contact = first_region = None
            events = {name: {'approach': None, 'contact': None, 'grasp': None, 'lift': None, 'basket': None} for name in names}
            initial_observation_hash = None
            for step in range(280):
                if step % 5 == 0:
                    request = base.make_request(obs, task, demo_index, mode, ep, step // 5, base.Args())
                    if step == 0:
                        initial_observation_hash = digest(request['observation/image'], request['observation/wrist_image'], request['observation/state'])
                    clean = {key: request[key].copy() for _, key in CAMERAS}
                    mask_info = {}
                    masks = {key: np.zeros((224, 224), bool) for _, key in CAMERAS}
                    if args.vision.startswith('mask_'):
                        request, masks, mask_info = intervene(env, request, target, other, args.vision)
                    if args.vision == 'freeze_near':
                        pos = env.env.object_states_dict[target].get_geom_state()['pos']
                        distance = float(np.linalg.norm(obs['robot0_eef_pos'] - pos))
                        if frozen is None and distance <= 0.12:
                            frozen = {key: request[key].copy() for _, key in CAMERAS}
                            freeze_step = step
                        if frozen is not None:
                            request.update({key: value.copy() for key, value in frozen.items()})
                        mask_info = {'distance_to_target_m': distance, 'frozen': frozen is not None}
                    request['language_vision'] = {'condition': condition, 'episode': ep, 'step': step,
                                                  'language': args.language,
                                                  'paired_languages': args.vision == 'normal' and args.language == 'milk'}
                    response = client.infer(request)
                    info = response['context_ablation']
                    assert info['demo_task_index'] == demo_index
                    assert info['selected_episode'] == ([814] if demo_index == 25 else [821]) or info['selected_episode'] == (814 if demo_index == 25 else 821)
                    plan = response['actions'][:5].copy()
                    assert np.isfinite(plan).all()
                    records.append({**info, 'step': step, 'intervention': mask_info,
                                    'clean_image_hash': digest(*clean.values()),
                                    'policy_image_hash': digest(request['observation/image'], request['observation/wrist_image'])})
                    policy_frames.append(np.concatenate([request[key] for _, key in CAMERAS], axis=1))
                    masks_saved.append(np.stack([masks[key] for _, key in CAMERAS]))
                action = plan[step % 5]
                obs, _, _, _ = env.step(action.tolist())
                pos = np.array([env.env.object_states_dict[n].get_geom_state()['pos'].copy() for n in names])
                grasp = [bool(env.env._check_grasp(env.env.robots[0].gripper, env.env.objects_dict[n])) for n in names]
                touch = [bool(env.env.check_contact(env.env.robots[0].gripper.contact_geoms,
                                                    env.env.objects_dict[n].contact_geoms)) for n in names]
                inside = list(base.scene_object_status(env).values())
                dz = pos[:, 2] - positions0[:, 2]
                distance = np.linalg.norm(pos - obs['robot0_eef_pos'], axis=1)
                for i, name in enumerate(names):
                    indicators = {'approach': distance[i] <= 0.12, 'contact': touch[i], 'grasp': grasp[i],
                                  'lift': dz[i] >= 0.03, 'basket': inside[i]}
                    for event, flag in indicators.items():
                        if flag and events[name][event] is None:
                            events[name][event] = step + 1
                if first_grasp is None and any(grasp):
                    first_grasp = names[int(np.argmax(grasp))]
                if first_contact is None and any(touch):
                    first_contact = names[int(np.argmax(touch))]
                # Region is tied to initial XY, independent of objects moving later.
                near = np.linalg.norm(positions0[:, :2] - obs['robot0_eef_pos'][:2], axis=1)
                region = int(np.argmin(near))
                if first_region is None and near[region] <= 0.08 and obs['robot0_eef_pos'][2] <= positions0[region, 2] + 0.15:
                    first_region = names[region]
                actions.append(action); poses.append(pos); grasps.append(grasp); touches.append(touch)
                baskets.append(inside); heights.append(dz)
                eefs.append(np.concatenate([obs['robot0_eef_pos'], obs['robot0_gripper_qpos']]))
                frames.append(base.image_for_policy(obs, 'agentview_image'))
                wrist_frames.append(base.image_for_policy(obs, 'robot0_eye_in_hand_image'))
            epdir = directory / f'ep{ep:03d}'
            epdir.mkdir(exist_ok=False)
            np.savez_compressed(epdir / 'trajectory.npz', initial_sim_state=initial_state, initial_positions=positions0,
                                actions=actions, object_names=names, object_positions=poses, eef_and_gripper=eefs,
                                grasp=grasps, contact=touches, in_basket=baskets, height_change=heights,
                                policy_images=policy_frames, pixel_masks=masks_saved)
            imageio.mimwrite(epdir / 'environment.mp4', [np.concatenate([f,w],1) for f,w in zip(frames,wrist_frames)], fps=20)
            # Each pair is the exact input for one 5-control-step action chunk; 4Hz = real elapsed time.
            imageio.mimwrite(epdir / 'policy_input.mp4', policy_frames, fps=4)
            row = {'condition': condition, 'episode': ep, 'demo': args.demo, 'language': args.language,
                   'vision': args.vision, 'initial_state_hash': digest(initial_state),
                   'initial_observation_hash': initial_observation_hash, 'target_for_mask': target,
                   'other_for_mask': other, 'target_basis': target_basis, 'freeze_step': freeze_step, 'first_region': first_region,
                   'actual_first_contact': first_contact, 'actual_first_grasp': first_grasp, 'events': events,
                   'ever_in_basket': dict(zip(names, np.any(baskets, axis=0).tolist())),
                   'max_height_change': dict(zip(names, np.max(heights, axis=0).tolist())),
                   'requests': records, 'seconds': time.monotonic()-started}
            with record_file.open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(condition, ep, row['ever_in_basket'], f"{row['seconds']:.1f}s", flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--demo', choices=tuple(base.TASKS), required=True)
    p.add_argument('--language', choices=tuple(PROMPTS), default='milk')
    p.add_argument('--vision', choices=VISIONS, default='normal')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=5)
    run(p.parse_args())
