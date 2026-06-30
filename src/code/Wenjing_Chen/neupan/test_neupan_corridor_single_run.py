"""
NeuPAN 走廊场景单次可视化测试

测试目的：
在 20m 走廊 + 2个挡板场景中，运行一次完整的 NeuPAN 导航测试，
实时打印机器人位置、速度、规划器状态，用于直观观察导航行为。

使用方法：
    conda activate neupan
    cd ~/DRL-robot-navigation-IR-SIM
    python test_neupan_corridor_single_run.py

依赖：
    - NeuPAN 库 (~/NeuPAN)
    - 重新训练的小车DUNE模型: ~/NeuPAN/example/dune_train/model/diff_robot_default_2/model_5000.pth
    - 场景文件: robot_nav/worlds/neupan_corridor_train.yaml
    - Planner配置: ~/NeuPAN/example/standard_eval_small/diff/planner.yaml

注意：
    若需要图形界面可视化，将 disable_plotting 改为 False，
    并确保 $DISPLAY 环境变量正确设置、X11 client 数量未达上限
    （可用 `xlsclients | wc -l` 检查）。
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, '/home/ubuntu22/NeuPAN')
import os
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan.neupan import neupan
from robot_nav.SIM_ENV.sim import SIM
sys.path.insert(0, '/home/ubuntu22/DRL-robot-navigation-IR-SIM/robot_nav')

WORLD = '/home/ubuntu22/DRL-robot-navigation-IR-SIM/robot_nav/worlds/neupan_corridor_train.yaml'
PLANNER = '/home/ubuntu22/NeuPAN/example/standard_eval_small/diff/planner.yaml'

START = [[0], [20], [0]]
GOAL = [[60], [20], [0]]
MAX_STEPS = 1000
PRINT_EVERY = 30
ENABLE_VISUALIZATION = False  # set True if X11 display is available


def main():
    sim = SIM(world_file=WORLD, disable_plotting=not ENABLE_VISUALIZATION)
    planner = neupan.init_from_yaml(PLANNER)

    scan, dist, cos, sin, col, goal, a, r = sim.reset(
        robot_state=START, robot_goal=GOAL, random_obstacles=False)
    state = sim.env.get_robot_state()
    start = np.array([[state[0, 0]], [state[1, 0]], [state[2, 0]]])
    goal_arr = np.array(GOAL, dtype=float)
    planner.reset()
    planner.update_initial_path_from_goal(start, goal_arr)

    print(f"NeuPAN corridor test | start={START} goal={GOAL}")
    print(f"Robot: TurtleBot3 Burger size, retrained DUNE model")
    print("-" * 60)

    for step in range(MAX_STEPS):
        rs = sim.env.get_robot_state()
        cur = np.array([[rs[0, 0]], [rs[1, 0]], [rs[2, 0]]])
        scan_dict = {
            'ranges': scan.tolist(),
            'angle_min': -np.pi, 'angle_max': np.pi,
            'range_max': 7.0, 'range_min': 0.0,
        }
        pts = planner.scan_to_point(cur, scan_dict)
        action, info = planner.forward(cur, pts)
        v, w = action[0, 0], action[1, 0]

        if step % PRINT_EVERY == 0:
            print(f"Step {step:4d}: pos=({rs[0,0]:.2f},{rs[1,0]:.2f}) "
                  f"v={v:.3f} min_dist={planner.min_distance:.3f} "
                  f"stop={info.get('stop')} arrive={info.get('arrive')}")

        if info.get('arrive') or goal:
            print(f"\nSUCCESS: arrived at step={step+1}, "
                  f"time={(step+1)*0.1:.1f}s")
            break
        if info.get('stop'):
            print(f"\nSTOPPED: NeuPAN stop triggered at step={step}, "
                  f"min_dist={planner.min_distance:.3f}")
            break
        if col:
            print(f"\nCOLLISION at step={step}")
            break

        scan, dist, cos, sin, col, goal, a, r = sim.step(v, w)
    else:
        print(f"\nTIMEOUT after {MAX_STEPS} steps, "
              f"final position=({rs[0,0]:.2f},{rs[1,0]:.2f})")

    if ENABLE_VISUALIZATION:
        time.sleep(5)
    sim.env.end()


if __name__ == '__main__':
    main()
