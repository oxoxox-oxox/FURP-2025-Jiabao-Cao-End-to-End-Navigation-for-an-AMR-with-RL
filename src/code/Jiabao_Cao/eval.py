"""
Evaluation / visualization script for trained CNNTD3 navigation model.

Usage:
    python eval.py [model_path] [--episodes N]

Default model: ./td3_models/best_model
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# ===== Set matplotlib backend BEFORE any library that imports pyplot =====
import matplotlib
try:
    matplotlib.use('TkAgg')   # cross-platform, ships with tkinter
except ImportError:
    try:
        matplotlib.use('QtAgg')   # fallback: requires PyQt/PySide
    except ImportError:
        print("[WARN] No GUI backend available (TkAgg, QtAgg). "
              "Rendering will be disabled.")
# ========================================================================

from env import IrSimNavEnv
from CNNTD3 import CNNTD3


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained CNNTD3 navigation policy'
    )
    parser.add_argument(
        'path', nargs='?', default='./td3_models/best_model',
        help='Path to model checkpoint directory (default: ./td3_models/best_model)'
    )
    parser.add_argument(
        '--episodes', type=int, default=3,
        help='Number of evaluation episodes (default: 3)'
    )
    args = parser.parse_args()

    state_dim = 89     # 84 rays + 5
    action_dim = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = Path(args.path).parent
    model_name = Path(args.path).name

    print(f"Loading CNNTD3 model from: {args.path}")
    agent = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        device=torch.device(device),
        save_every=0,
        load_model=True,
        model_name=model_name,
        load_directory=model_dir,
    )

    env = IrSimNavEnv(
        yaml_file='./env/env.yaml',
        render_mode='human',
    )

    print(f"\n=== CNNTD3 Evaluation: {args.episodes} episodes (deterministic) ===\n")
    episode_count = 0
    obs, _ = env.reset()
    step_count = 0

    while episode_count < args.episodes:
        action_td3 = agent.get_action(obs, add_noise=False)

        # Map from TD3 space [-1,1] to env space [v:0~1, w:-1~1]
        v_env = (float(action_td3[0]) + 1.0) / 2.0
        w_env = float(action_td3[1])
        env_action = np.array([v_env, w_env], dtype=np.float32)

        obs, reward, done, truncated, info = env.step(env_action)
        env.render()
        step_count += 1

        if done or truncated:
            episode_count += 1
            result = info.get('result', 'timeout')
            print(f"Episode {episode_count}: result={result}, "
                  f"steps={step_count}, reward={reward:.2f}")
            obs, _ = env.reset()
            step_count = 0

    env.close()
    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
