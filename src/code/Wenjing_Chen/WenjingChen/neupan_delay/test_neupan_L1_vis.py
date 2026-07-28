"""
NeuPAN L1延迟补偿 - 可视化单次运行

用于肉眼观察动态场景下延迟补偿的效果，带窗口显示 + 保存动画。

用法:
  cd ~/NeuPAN/example && conda activate neupan

  # 看baseline（无延迟）
  python test_neupan_L1_vis.py --mode A_baseline

  # 看有延迟无补偿（应该会撞）
  python test_neupan_L1_vis.py --mode B_naive

  # 看仅自身状态补偿
  python test_neupan_L1_vis.py --mode C_self_only

  # 看L1完整补偿（自身+障碍物外推）
  python test_neupan_L1_vis.py --mode D_L1_full

  # 指定DUNE模型（默认原版，可切换随机训练版）
  python test_neupan_L1_vis.py --mode D_L1_full --planner planner_delay_rand_trained.yaml
"""
import sys, os, argparse
import numpy as np
from collections import deque

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="D_L1_full",
                    choices=["A_baseline", "B_naive", "C_self_only", "D_L1_full"])
parser.add_argument("--planner", type=str, default="planner_turn_simple.yaml")
parser.add_argument("--env", type=str, default="env_turn_dynamic.yaml")
parser.add_argument("--min-delay-ms", type=int, default=100)
parser.add_argument("--max-delay-ms", type=int, default=1000)
parser.add_argument("--save", action="store_true", help="保存动画gif")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42, help="固定随机种子便于对比不同mode")
args = parser.parse_args()

MODES = {
    "A_baseline": {"add_delay": False, "compensate_self": False, "compensate_obs": False,
                   "desc": "无延迟 baseline"},
    "B_naive": {"add_delay": True, "compensate_self": False, "compensate_obs": False,
                "desc": "有延迟 无补偿"},
    "C_self_only": {"add_delay": True, "compensate_self": True, "compensate_obs": False,
                     "desc": "有延迟 + 仅自身状态补偿"},
    "D_L1_full": {"add_delay": True, "compensate_self": True, "compensate_obs": True,
                   "desc": "有延迟 + 自身状态 + 障碍物外推(L1)"},
}

mode_cfg = MODES[args.mode]
STEP_TIME_MS = 100
dt = 0.1

np.random.seed(args.seed)

print(f"\n{'='*60}")
print(f"可视化: {args.mode} - {mode_cfg['desc']}")
print(f"Planner: {args.planner}")
print(f"延迟范围: {args.min_delay_ms}-{args.max_delay_ms}ms")
print(f"{'='*60}\n")


def predict_robot_state(delayed_state, action_history, dt_val):
    state = delayed_state.copy().flatten()
    for v, w in action_history:
        state[0] += v * np.cos(state[2]) * dt_val
        state[1] += v * np.sin(state[2]) * dt_val
        state[2] += w * dt_val
    return state.reshape(3, 1)


def predict_obstacle_points(delayed_points, point_velocities, delay_seconds):
    if point_velocities is None or delayed_points is None:
        return delayed_points
    if delayed_points.shape != point_velocities.shape:
        return delayed_points
    return delayed_points + point_velocities * delay_seconds


env = irsim.make(args.env, save_ani=args.save, display=True)
planner = neupan.init_from_yaml(args.planner)

import math
max_history = int(args.max_delay_ms / STEP_TIME_MS) + 5
state_history = deque(maxlen=max_history)
lidar_history = deque(maxlen=max_history)
action_history = deque(maxlen=max_history)

outcome = "timeout"
add_delay = mode_cfg["add_delay"]
compensate_self = mode_cfg["compensate_self"]
compensate_obs = mode_cfg["compensate_obs"]

for i in range(args.max_steps):
    robot_state_now = env.get_robot_state()
    lidar_scan_now = env.get_lidar_scan()

    state_history.append(robot_state_now.copy())
    lidar_history.append(
        lidar_scan_now.copy() if hasattr(lidar_scan_now, 'copy') else lidar_scan_now
    )

    if add_delay and args.min_delay_ms > 0:
        delay_ms = np.random.uniform(args.min_delay_ms, args.max_delay_ms)
    else:
        delay_ms = 0

    delay_steps_int = int(math.floor(delay_ms / STEP_TIME_MS))
    delay_seconds = delay_ms / 1000.0

    if delay_steps_int == 0 or not add_delay or len(state_history) <= delay_steps_int:
        state_for_planner = robot_state_now
        lidar_for_planner = lidar_scan_now
        points = planner.scan_to_point(state_for_planner, lidar_for_planner)
        action, info = planner(state_for_planner, points, None)
    else:
        idx = max(0, len(state_history) - 1 - delay_steps_int)
        delayed_state = state_history[idx]
        delayed_lidar = lidar_history[idx]

        if compensate_self and len(action_history) >= delay_steps_int:
            recent_actions = list(action_history)[-delay_steps_int:]
            predicted_state = predict_robot_state(delayed_state, recent_actions, dt)
        else:
            predicted_state = delayed_state

        if compensate_obs:
            try:
                delayed_points, point_vels = planner.scan_to_point_velocity(
                    delayed_state, delayed_lidar
                )
                predicted_points = predict_obstacle_points(
                    delayed_points, point_vels, delay_seconds
                )
                action, info = planner(predicted_state, predicted_points, None)
            except Exception:
                points = planner.scan_to_point(predicted_state, delayed_lidar)
                action, info = planner(predicted_state, points, None)
        else:
            points = planner.scan_to_point(predicted_state, delayed_lidar)
            action, info = planner(predicted_state, points, None)

    if info.get("arrive", False):
        print(f"Step {i}: 到达目标！")
        outcome = "arrive"

    v = float(action[0, 0])
    w = float(action[1, 0])
    action_history.append((v, w))
    action_exec = np.array([[v], [w]])

    try:
        env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(planner.opt_trajectory, "r", refresh=True)
        env.draw_trajectory(planner.ref_trajectory, "b", refresh=True)
    except Exception:
        pass

    env.step(action_exec)
    env.render()

    if env.done():
        print(f"Step {i}: 碰撞！")
        outcome = "collision"
        break
    if outcome == "arrive":
        break

    if i == 0:
        try:
            env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=False)
            env.render()
        except Exception:
            pass

ani_name = f"L1_vis_{args.mode}" if args.save else ""
env.end(2, ani_name=ani_name)

print(f"\n结果: {outcome}, 总步数: {i+1}")
