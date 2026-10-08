import sys
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS
from openpi_client import websocket_client_policy

print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
print('LIBERO tasks:', suite.n_tasks, 'held-out tasks:', len(LIBERO_UNSEEN_TASKS))
print('Task:', task.language)
env = OffScreenRenderEnv(
    bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
    camera_heights=128, camera_widths=128,
)
try:
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    frames = []
    for _ in range(5):
        obs, reward, done, info = env.step([0.] * 6 + [-1.])
        frames.append(obs['agentview_image'])
    for key in ('agentview_image', 'robot0_eye_in_hand_image'):
        img = obs[key]
        assert img.shape == (128, 128, 3) and np.std(img) > 1
        print(key, img.shape, 'pixel std:', float(np.std(img)), 'PASS')
        imageio.imwrite('logs/environment_setup/' + key + '.png', img)
    imageio.mimwrite('logs/environment_setup/libero_smoke.mp4', frames, fps=10)
    print('LIBERO reset, init state, step, EGL render and MP4 export: PASS')
finally:
    env.close()
