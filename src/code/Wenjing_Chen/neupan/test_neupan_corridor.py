"""
NeuPAN 走廊场景测试
用法：cd ~/DRL-robot-navigation-IR-SIM && conda activate neupan && python test_neupan_corridor.py
"""
import sys, numpy as np, time
sys.path.insert(0, '.')
sys.path.insert(0, '/home/ubuntu22/NeuPAN')
import os
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan.neupan import neupan
from robot_nav.SIM_ENV.sim import SIM
sys.path.insert(0, '/home/ubuntu22/DRL-robot-navigation-IR-SIM/robot_nav')

WORLD   = '/home/ubuntu22/DRL-robot-navigation-IR-SIM/robot_nav/worlds/neupan_corridor_train.yaml'
PLANNER = '/home/ubuntu22/NeuPAN/example/corridor_fair/diff/planner.yaml'

CASES = [
    ([[0],[20],[0]],   [[60],[20],[0]], "正向"),
    ([[0],[20],[1.57]],[[60],[20],[0]], "朝上"),
    ([[0],[20],[-1.57]],[[60],[20],[0]],"朝下"),
    ([[0],[18],[0]],   [[60],[20],[0]], "偏下"),
    ([[0],[22],[0]],   [[60],[20],[0]], "偏上"),
]
MAX_STEPS = 1000

print("NeuPAN 走廊测试")
print("="*50)
total, success, times = 0, 0, []

for robot_state, robot_goal, label in CASES:
    for rep in range(3):
        sim = SIM(world_file=WORLD, disable_plotting=True)
        planner = neupan.init_from_yaml(PLANNER)

        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state,
            robot_goal=robot_goal,
            random_obstacles=False)

        state = sim.env.get_robot_state()
        start = np.array([[state[0,0]],[state[1,0]],[state[2,0]]])
        goal_arr = np.array(robot_goal, dtype=float)
        planner.reset()
        planner.update_initial_path_from_goal(start, goal_arr)

        t0 = time.time()
        for step in range(MAX_STEPS):
            rs = sim.env.get_robot_state()
            cur = np.array([[rs[0,0]],[rs[1,0]],[rs[2,0]]])
            scan_dict = {
                'ranges': scan.tolist(),
                'angle_min': -np.pi, 'angle_max': np.pi,
                'range_max': 10.0, 'range_min': 0.0,
            }
            pts = planner.scan_to_point(cur, scan_dict)
            action, info = planner.forward(cur, pts)
            v, w = action[0,0], action[1,0]
            scan,dist,cos,sin,col,goal,a,r = sim.step(
                lin_velocity=v, ang_velocity=w)
            if info.get('arrive') or goal:
                elapsed = time.time() - t0
                success += 1
                times.append(elapsed)
                print(f"  {label} rep{rep+1}: ✅ {step+1}步 {elapsed:.1f}s")
                break
            if col or info.get('stop'):
                print(f"  {label} rep{rep+1}: ❌ step={step+1} stop={info.get('stop')}")
                break
        else:
            print(f"  {label} rep{rep+1}: ⏰ 超时")
        total += 1
        sim.env.end()

sr = success/total
avg_t = np.mean(times) if times else None
print(f"\nNeuPAN 走廊 SR={sr:.0%}  avg_time={f'{avg_t:.1f}s' if avg_t else 'N/A'}  ({success}/{total})")
