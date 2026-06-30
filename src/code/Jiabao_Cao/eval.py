"""
RCPG (Recurrent + CNN + TD3) 导航模型的评估 / 可视化脚本。

用法:
    python eval.py [model_path] [--episodes N]

默认模型: ./rcpg_models/best_model
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# ===== 在任何导入 pyplot 的库之前设置 matplotlib backend =====
import matplotlib
try:
    matplotlib.use('TkAgg')   # 跨平台，自带 tkinter
except ImportError:
    try:
        matplotlib.use('QtAgg')   # 备选: 需要 PyQt/PySide
    except ImportError:
        print("[警告] 无可用的 GUI backend (TkAgg, QtAgg)。"
              "渲染将被禁用。")
# ========================================================================

from env import IrSimNavEnv
from CNNTD3 import CNNRC


def main():
    parser = argparse.ArgumentParser(
        description='评估训练好的 CNNRC 导航策略'
    )
    parser.add_argument(
        'path', nargs='?', default='./rcpg_models/best_model',
        help='模型检查点目录路径 (默认: ./rcpg_models/best_model)'
    )
    parser.add_argument(
        '--episodes', type=int, default=3,
        help='评估 episode 数 (默认: 3)'
    )
    args = parser.parse_args()

    HIST_N = 3
    state_dim = 89 * HIST_N     # 堆叠帧: (84 个射线 + 5) × hist_n
    action_dim = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = Path(args.path).parent
    model_name = Path(args.path).name

    print(f"加载 RCPG 模型: {args.path}")
    agent = CNNRC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        device=torch.device(device),
        hist_n=HIST_N,
        save_every=0,
        load_model=True,
        model_name=model_name,
        load_directory=model_dir,
    )

    env = IrSimNavEnv(
        yaml_file='./env/env_corridor_2.yaml',
        render_mode='human',
        hist_n=HIST_N,
    )

    print(f"\n=== RCPG 评估: {args.episodes} episodes (确定性) ===\n")
    episode_count = 0
    obs, _ = env.reset()
    step_count = 0

    while episode_count < args.episodes:
        action = agent.get_action(obs, add_noise=False)

        # 从 RCPG 空间 [-1,1] 映射到 env 空间 [v:0~1, w:-1~1]
        v_env = (float(action[0]) + 1.0) / 2.0
        w_env = float(action[1])
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
    print("\n评估完成。")


if __name__ == '__main__':
    main()
