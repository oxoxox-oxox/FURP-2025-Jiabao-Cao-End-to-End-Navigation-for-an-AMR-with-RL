#!/usr/bin/env python3
"""
NeuPAN 公平对比评估脚本
用法: cd ~/NeuPAN/example && python ../eval_neupan_fair.py
"""
from neupan import neupan
import irsim
import numpy as np
import time

SCENARIOS = {
    'standard_eval': {
        'env':     'standard_eval/diff/env.yaml',
        'planner': 'standard_eval/diff/planner.yaml',
    },
    'u_trap': {
        'env':     'u_trap/diff/env.yaml',
        'planner': 'u_trap/diff/planner.yaml',
    },
    'double_u': {
        'env':     'double_u/diff/env.yaml',
        'planner': 'double_u/diff/planner.yaml',
    },
    'symmetric_corridor': {
        'env':     'symmetric_corridor/diff/env.yaml',
        'planner': 'symmetric_corridor/diff/planner.yaml',
    },
}

CNNTD3     = {'standard_eval': 0.92, 'u_trap': 0.00, 'double_u': 0.33, 'symmetric_corridor': 0.83}
IMPROVED   = {'standard_eval': 0.78, 'u_trap': 1.00, 'double_u': 0.33, 'symmetric_corridor': 1.00}
N_TRIALS   = 10
MAX_STEPS  = 200

def run_once(env_path, planner_path):
    env = irsim.make(env_path, display=False, save_ani=False)
    planner = neupan.init_from_yaml(planner_path)
    success = False
    t0 = time.time()
    for i in range(MAX_STEPS):
        robot_state = env.get_robot_state()
        lidar_scan  = env.get_lidar_scan()
        points      = planner.scan_to_point(robot_state, lidar_scan)
        action, info = planner(robot_state, points, None)
        if info['arrive']:
            success = True
            break
        env.step(action)
        if env.done():
            break
    elapsed = time.time() - t0
    env.end()
    return success, elapsed if success else None

def main():
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), 'example'))

    results = {}
    for name, cfg in SCENARIOS.items():
        print(f"\n{'='*52}\n场景: {name}  ({N_TRIALS} 次)\n{'='*52}")
        successes, times = 0, []
        for i in range(N_TRIALS):
            try:
                ok, t = run_once(cfg['env'], cfg['planner'])
            except Exception as e:
                print(f"  Trial {i+1:2d}: ERROR — {e}")
                ok, t = False, None
            print(f"  Trial {i+1:2d}: {'OK ' if ok else 'FAIL'}  {f'{t:.1f}s' if t else '---'}")
            if ok:
                successes += 1
                times.append(t)
        sr = successes / N_TRIALS
        avg_t = float(np.mean(times)) if times else None
        results[name] = (sr, avg_t)
        print(f"  → SR={sr:.0%}  avg_time={f'{avg_t:.1f}s' if avg_t else 'N/A'}")

    print(f"\n{'='*70}")
    print(f"{'场景':<22} {'CNNTD3':>8} {'Improved':>10} {'NeuPAN':>8}")
    print(f"{'-'*70}")
    for name, (sr, _) in results.items():
        print(f"{name:<22} {CNNTD3[name]:>8.0%} {IMPROVED[name]:>10.0%} {sr:>8.0%}")
    print(f"{'='*70}")

    # 保存
    import yaml
    out = {n: {'sr': sr, 'avg_time_s': t} for n, (sr, t) in results.items()}
    with open('../neupan_eval_results.yaml', 'w') as f:
        yaml.dump(out, f)
    print("\n结果已保存到 ~/NeuPAN/neupan_eval_results.yaml")

if __name__ == '__main__':
    main()
