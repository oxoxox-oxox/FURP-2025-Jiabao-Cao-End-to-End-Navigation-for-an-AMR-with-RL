#!/usr/bin/env python3
"""
NeuPAN 公平对比评估脚本
用法：cd ~/NeuPAN && python eval_neupan.py
输出：各场景 SR、平均到达时间，与 CNNTD3 对比表
"""

import subprocess
import time
import yaml
import numpy as np
from pathlib import Path
import sys
import os

# ── 场景配置 ──────────────────────────────────────────────────
SCENARIOS = {
    'standard_eval': {
        'env':     'example/standard_eval/diff/env.yaml',
        'planner': 'example/standard_eval/diff/planner.yaml',
        'max_steps': 200,
    },
    'u_trap': {
        'env':     'example/u_trap/diff/env.yaml',
        'planner': 'example/u_trap/diff/planner.yaml',
        'max_steps': 200,
    },
    'double_u': {
        'env':     'example/double_u/diff/env.yaml',
        'planner': 'example/double_u/diff/planner.yaml',
        'max_steps': 200,
    },
    'symmetric_corridor': {
        'env':     'example/symmetric_corridor/diff/env.yaml',
        'planner': 'example/symmetric_corridor/diff/planner.yaml',
        'max_steps': 200,
    },
}

# CNNTD3 已有结果（用于对比输出）
CNNTD3_RESULTS = {
    'standard_eval':      {'sr': 0.92, 'time': 16.9},
    'u_trap':             {'sr': 0.00, 'time': None},
    'double_u':           {'sr': 0.33, 'time': None},
    'symmetric_corridor': {'sr': 0.83, 'time': None},
}

CNNTD3_IMPROVED_RESULTS = {
    'standard_eval':      {'sr': 0.78, 'time': None},
    'u_trap':             {'sr': 1.00, 'time': None},
    'double_u':           {'sr': 0.33, 'time': None},
    'symmetric_corridor': {'sr': 1.00, 'time': None},
}

N_TRIALS = 10  # 每个场景跑几次


def run_neupan_once(env_path: str, planner_path: str, max_steps: int):
    """
    单次运行 NeuPAN，返回 (success, elapsed_time)
    依赖 NeuPAN 的 Python API
    """
    try:
        import irsim
        from neupan import NeuPAN

        env = irsim.make(env_path, display=False, save_ani=False)
        planner = NeuPAN(planner_path)

        start_time = time.time()
        success = False

        for step in range(max_steps):
            obs = env.get_observation()
            lidar = obs[0]['lidar']          # (N,) 距离数组
            robot_state = env.get_robot_state(id=0)  # [x, y, theta, v]
            goal = env.get_goal(id=0)        # [gx, gy, gtheta]

            vel = planner.plan(lidar, robot_state, goal)
            done, info = env.step(vel)

            if info.get('arrive', False):
                success = True
                break
            if done:
                break

        elapsed = time.time() - start_time
        env.end()
        return success, elapsed if success else None

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False, None


def evaluate_scenario(name: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"场景: {name}  ({N_TRIALS} 次)")
    print(f"{'='*50}")

    successes = 0
    times = []

    for i in range(N_TRIALS):
        ok, t = run_neupan_once(cfg['env'], cfg['planner'], cfg['max_steps'])
        status = 'OK' if ok else 'FAIL'
        t_str = f"{t:.1f}s" if t else "---"
        print(f"  Trial {i+1:2d}: {status}  {t_str}")
        if ok:
            successes += 1
            times.append(t)

    sr = successes / N_TRIALS
    avg_t = np.mean(times) if times else None
    print(f"  SR = {sr:.0%}  |  avg_time = {f'{avg_t:.1f}s' if avg_t else 'N/A'}")
    return sr, avg_t


def print_comparison_table(results: dict):
    print("\n" + "="*72)
    print(f"{'场景':<20} {'CNNTD3':>10} {'Improved':>10} {'NeuPAN(修复)':>14}")
    print("-"*72)
    for name, (sr, _) in results.items():
        cn = CNNTD3_RESULTS[name]['sr']
        im = CNNTD3_IMPROVED_RESULTS[name]['sr']
        winner = '← RL wins' if sr < max(cn, im) else '← NeuPAN wins'
        print(f"{name:<20} {cn:>9.0%}  {im:>9.0%}  {sr:>12.0%}  {winner}")
    print("="*72)

    print("\n到达时间对比（仅标准环境）:")
    std = results.get('standard_eval')
    if std:
        n_t = f"{std[1]:.1f}s" if std[1] else "N/A"
        print(f"  CNNTD3:  {CNNTD3_RESULTS['standard_eval']['time']}s")
        print(f"  NeuPAN:  {n_t}")


def main():
    neupan_dir = Path(__file__).parent
    os.chdir(neupan_dir)
    print(f"工作目录: {neupan_dir}")

    # 先检查 API 可用性
    try:
        import irsim
        from neupan import NeuPAN
        print("NeuPAN + irsim 导入成功")
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        print("请确认在 neupan conda 环境中运行：conda activate neupan")
        sys.exit(1)

    results = {}
    for name, cfg in SCENARIOS.items():
        sr, avg_t = evaluate_scenario(name, cfg)
        results[name] = (sr, avg_t)

    print_comparison_table(results)

    # 保存结果
    out = {
        name: {'sr': sr, 'avg_time': t}
        for name, (sr, t) in results.items()
    }
    with open('neupan_eval_results.yaml', 'w') as f:
        yaml.dump(out, f, allow_unicode=True)
    print("\n结果已保存到 neupan_eval_results.yaml")


if __name__ == '__main__':
    main()
