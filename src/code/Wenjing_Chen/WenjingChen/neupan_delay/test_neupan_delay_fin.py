"""
NeuPAN 延迟可视化实验 - 多种延迟模式（action / observation / inference）

用法:
  cd ~/NeuPAN/example && conda activate neupan

  # 无延迟基线
  python test_neupan_delay_fin.py --delay 0 --speed-scale 1.0 --save

  # observation延迟（传感器→planner）
  python test_neupan_delay_fin.py --delay 5 --delay-mode observation --speed-scale 1.0 --save

  # inference延迟（拉长决策周期，最贴近真实场景）
  python test_neupan_delay_fin.py --delay 5 --delay-mode inference --speed-scale 1.0 --save
"""
import sys, os, argparse, time
import numpy as np
from collections import deque

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

# ====== 参数 ======
parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=int, default=0,
                    help="延迟步数上限 (0=无延迟, 5=500ms, 10=1000ms)")
parser.add_argument("--delay-mode", type=str, choices=["action", "observation", "inference"],
                    default="action",
                    help="延迟类型：action（planner->actuator），observation（sensor->planner），inference（拉长决策周期）")
parser.add_argument("--random-delay", action="store_true",
                    help="启用随机延迟（每步在 [0, delay] 之间采样）")
parser.add_argument("--speed-scale", type=float, default=1.0,
                    help="执行时线速度放大因子（默认1.0）")
parser.add_argument("--no-display", action="store_true", help="不显示窗口")
parser.add_argument("--save", action="store_true", help="保存动画")
parser.add_argument("--max-steps", type=int, default=1000, help="最大步数")
parser.add_argument("--run-id", type=str, default="", help="运行ID，区分输出文件")
args = parser.parse_args()

max_delay = max(0, args.delay)
use_random_delay = args.random_delay
speed_scale = float(args.speed_scale)
delay_mode = args.delay_mode
step_time = 0.1  # 每步100ms
delay_ms = max_delay * int(step_time * 1000)

ENV_FILE = "env_turn_simple.yaml"
PLANNER_FILE = "planner_turn_simple.yaml"

print(f"{'='*60}")
print(f"NeuPAN 延迟可视化实验")
print(f"延迟模式 = {delay_mode}")
print(f"最大延迟 = {max_delay} 步 (~{delay_ms}ms)")
print(f"随机延迟 = {use_random_delay}")
print(f"线速度放大因子 = {speed_scale}")
print(f"{'='*60}")

# ====== 初始化环境 ======
env = irsim.make(ENV_FILE, save_ani=args.save, display=not args.no_display)
planner = neupan.init_from_yaml(PLANNER_FILE)

# ====== 缓冲区初始化 ======
buffer_len = max_delay + 1

# action buffer (for action mode)
action_buffer = deque()
for _ in range(buffer_len):
    action_buffer.append(np.array([[0.0], [0.0]]))

# observation buffers (for observation mode)
lidar_buffer = deque()
robot_state_buffer = deque()
for _ in range(buffer_len):
    lidar_buffer.append(None)
    robot_state_buffer.append(None)

# inference mode variables
last_action = np.array([[0.0], [0.0]])
last_info = {"stop": False, "arrive": False}
current_decision_interval = 1
decision_interval_counter = 0

# ====== 记录数据 ======
trajectory_x, trajectory_y = [], []
angular_velocities_planned, angular_velocities_executed = [], []
linear_velocities_planned, linear_velocities_executed = [], []
sampled_delays = []
outcome = "timeout"
total_steps = 0

# ====== 主循环 ======
for i in range(args.max_steps):
    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    # ===== OBSERVATION MODE =====
    if delay_mode == "observation":
        lidar_buffer.append(lidar_scan.copy())
        robot_state_buffer.append(robot_state.copy())
        while len(lidar_buffer) > buffer_len:
            lidar_buffer.popleft()
        while len(robot_state_buffer) > buffer_len:
            robot_state_buffer.popleft()

        if max_delay == 0:
            sensor_delay = 0
        else:
            sensor_delay = int(np.random.randint(0, max_delay + 1)) if use_random_delay else max_delay

        lidar_for_planner = lidar_buffer[-(sensor_delay + 1)]
        robot_state_for_planner = robot_state_buffer[-(sensor_delay + 1)]
        if lidar_for_planner is None:
            lidar_for_planner = lidar_scan
        if robot_state_for_planner is None:
            robot_state_for_planner = robot_state

        sampled_delays.append(int(sensor_delay))
        points = planner.scan_to_point(robot_state_for_planner, lidar_for_planner)
        action, info = planner(robot_state_for_planner, points, None)

    # ===== INFERENCE MODE =====
    elif delay_mode == "inference":
        if decision_interval_counter <= 0:
            if max_delay == 0:
                sampled_delay_here = 0
            else:
                sampled_delay_here = int(np.random.randint(0, max_delay + 1)) if use_random_delay else max_delay
            current_decision_interval = sampled_delay_here + 1
            decision_interval_counter = current_decision_interval

            points = planner.scan_to_point(robot_state, lidar_scan)
            action, info = planner(robot_state, points, None)
            last_action = action.copy()
            last_info = info.copy() if isinstance(info, dict) else info
            sampled_delays.append(int(sampled_delay_here))
        else:
            action = last_action.copy()
            info = last_info
            sampled_delays.append(int(current_decision_interval - 1))

        decision_interval_counter -= 1

    # ===== ACTION MODE (default) =====
    else:
        points = planner.scan_to_point(robot_state, lidar_scan)
        action, info = planner(robot_state, points, None)

        action_buffer.append(action.copy())
        while len(action_buffer) > buffer_len:
            action_buffer.popleft()

        if max_delay == 0:
            sampled_delay_here = 0
        else:
            sampled_delay_here = int(np.random.randint(0, max_delay + 1)) if use_random_delay else max_delay

        action = action_buffer[-(sampled_delay_here + 1)].copy()
        sampled_delays.append(int(sampled_delay_here))

    # Check planner status
    if isinstance(info, dict):
        if info.get("stop", False):
            print(f"Step {i}: NeuPAN stopped (minimum distance)")
        if info.get("arrive", False):
            print(f"Step {i}: NeuPAN arrived at target!")
            outcome = "arrive"
            total_steps = i + 1

    # Record planned action
    v_planned = float(action[0, 0])
    w_planned = float(action[1, 0])
    linear_velocities_planned.append(v_planned)
    angular_velocities_planned.append(w_planned)

    # Apply speed scaling (only to linear velocity)
    action_exec = action.copy()
    action_exec[0, 0] = float(action_exec[0, 0]) * speed_scale

    v_exec = float(action_exec[0, 0])
    w_exec = float(action_exec[1, 0])
    linear_velocities_executed.append(v_exec)
    angular_velocities_executed.append(w_exec)

    # Record trajectory
    trajectory_x.append(float(robot_state[0, 0]))
    trajectory_y.append(float(robot_state[1, 0]))

    # Visualization
    try:
        env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(planner.opt_trajectory, "r", refresh=True)
        env.draw_trajectory(planner.ref_trajectory, "b", refresh=True)
    except Exception:
        pass

    # Execute action
    env.step(action_exec)
    env.render()

    if env.done():
        print(f"Step {i}: Environment done (collision)")
        if outcome != "arrive":
            outcome = "collision"
        total_steps = i + 1
        break

    if outcome == "arrive":
        break

    # Draw initial path on first step
    if i == 0:
        try:
            env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=False)
            env.render()
        except Exception:
            pass

if total_steps == 0:
    total_steps = len(trajectory_x)

ani_name = f"neupan_{delay_mode}_delay{max_delay}"
if args.run_id:
    ani_name += f"_{args.run_id}"
env.end(3, ani_name=ani_name)

# ====== 统计 ======
w_exec_arr = np.array(angular_velocities_executed) if angular_velocities_executed else np.array([])
y_arr = np.array(trajectory_y) if trajectory_y else np.array([])

sign_changes = int(np.sum(np.diff(np.sign(w_exec_arr)) != 0)) if len(w_exec_arr) > 1 else 0
lat_dev = float(np.std(y_arr[len(y_arr)//2:])) if len(y_arr) > 10 else 0.0

print(f"\n{'='*60}")
print(f"结果统计")
print(f"  模式:            {delay_mode}")
print(f"  最大延迟:         {delay_ms}ms ({max_delay}步)")
print(f"  Outcome:         {outcome}")
print(f"  总步数:          {total_steps}")
print(f"  角速度反转:       {sign_changes} 次")
print(f"  后半段横向偏移std: {lat_dev:.4f}")
if trajectory_x:
    print(f"  最终位置:         ({trajectory_x[-1]:.2f}, {trajectory_y[-1]:.2f})")
print(f"{'='*60}")

# ====== 保存 ======
import json
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_{delay_mode}_delay{max_delay}_{ts}.json"

data = {
    "delay_mode": delay_mode,
    "max_delay_steps": max_delay,
    "delay_ms": delay_ms,
    "random_delay": use_random_delay,
    "speed_scale": speed_scale,
    "outcome": outcome,
    "total_steps": total_steps,
    "sign_changes": sign_changes,
    "lateral_deviation": lat_dev,
    "trajectory_x": trajectory_x,
    "trajectory_y": trajectory_y,
    "w_executed": angular_velocities_executed,
    "v_executed": linear_velocities_executed,
    "sampled_delays": sampled_delays,
}
with open(out_path, "w") as f:
    json.dump(data, f)
print(f"数据已保存: {out_path}")
