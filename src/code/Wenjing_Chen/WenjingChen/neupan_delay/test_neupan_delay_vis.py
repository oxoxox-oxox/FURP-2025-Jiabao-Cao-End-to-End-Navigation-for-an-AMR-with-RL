"""
NeuPAN 延迟可视化实验 - 走廊转弯场景
基于原版 corridor 场景（验证可用），注入不同延迟观察转弯后的震荡

用法：cd ~/NeuPAN/example && conda activate neupan && python test_neupan_delay_vis.py
可选参数：
  --delay 10        # 延迟步数（默认0）
  --no-display      # 不显示窗口
  --save            # 保存动画
"""
import sys, os, argparse
import numpy as np
from collections import deque

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

# ====== 参数 ======
parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=int, default=0, help="延迟步数 (0=无延迟, 5=250ms, 10=500ms, 20=1000ms)")
parser.add_argument("--no-display", action="store_true", help="不显示窗口")
parser.add_argument("--save", action="store_true", help="保存动画")
parser.add_argument("--max-steps", type=int, default=1000, help="最大步数")
args = parser.parse_args()

delay_steps = args.delay
delay_ms = delay_steps * 100  # step_time=0.1s, 所以每步100ms

ENV_FILE = "corridor/diff/env.yaml"
PLANNER_FILE = "corridor/diff/planner.yaml"

print(f"{'='*60}")
print(f"NeuPAN 延迟可视化实验")
print(f"延迟 = {delay_steps} 步 (~{delay_ms}ms)")
print(f"{'='*60}")

# ====== 初始化环境 ======
env = irsim.make(
    ENV_FILE,
    save_ani=args.save,
    display=not args.no_display,
)
planner = neupan.init_from_yaml(PLANNER_FILE)

# ====== 延迟Buffer初始化 ======
action_buffer = deque()
for _ in range(delay_steps):
    action_buffer.append(np.array([[0.0], [0.0]]))  # 零动作填充

# ====== 记录数据 ======
trajectory_x = []
trajectory_y = []
angular_velocities_planned = []   # NeuPAN计划的角速度
angular_velocities_executed = []  # 实际执行的角速度
linear_velocities_planned = []
linear_velocities_executed = []

# ====== 主循环 ======
for i in range(args.max_steps):
    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    points = planner.scan_to_point(robot_state, lidar_scan)
    action, info = planner(robot_state, points, None)

    if info["stop"]:
        print(f"Step {i}: NeuPAN stopped (minimum distance)")

    if info["arrive"]:
        print(f"Step {i}: NeuPAN arrived at target!")
        break

    # 记录NeuPAN计划的动作
    v_planned = float(action[0, 0])
    w_planned = float(action[1, 0])
    linear_velocities_planned.append(v_planned)
    angular_velocities_planned.append(w_planned)

    # ====== 延迟注入 ======
    if delay_steps == 0:
        action_exec = action
    else:
        action_buffer.append(action.copy())
        action_exec = action_buffer.popleft()

    v_exec = float(action_exec[0, 0])
    w_exec = float(action_exec[1, 0])
    linear_velocities_executed.append(v_exec)
    angular_velocities_executed.append(w_exec)

    # 记录轨迹
    trajectory_x.append(float(robot_state[0, 0]))
    trajectory_y.append(float(robot_state[1, 0]))

    # 可视化
    env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
    env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
    env.draw_trajectory(planner.opt_trajectory, "r", refresh=True)
    env.draw_trajectory(planner.ref_trajectory, "b", refresh=True)

    # 执行（可能延迟的）动作
    env.step(action_exec)
    env.render()

    if env.done():
        print(f"Step {i}: Environment done (collision)")
        break

    # 第一步画初始路径
    if i == 0:
        env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=False)
        env.render()

ani_name = f"neupan_delay_{delay_ms}ms"
env.end(3, ani_name=ani_name)

# ====== 统计 ======
w_planned = np.array(angular_velocities_planned)
w_executed = np.array(angular_velocities_executed)
y_arr = np.array(trajectory_y)

# 震荡指标
sign_changes_planned = int(np.sum(np.diff(np.sign(w_planned)) != 0)) if len(w_planned) > 1 else 0
sign_changes_executed = int(np.sum(np.diff(np.sign(w_executed)) != 0)) if len(w_executed) > 1 else 0
lat_dev = float(np.std(y_arr[len(y_arr)//2:])) if len(y_arr) > 10 else 0.0

print(f"\n{'='*60}")
print(f"结果统计 (延迟={delay_ms}ms)")
print(f"  总步数:          {len(trajectory_x)}")
print(f"  角速度反转(计划): {sign_changes_planned} 次")
print(f"  角速度反转(执行): {sign_changes_executed} 次")
print(f"  后半段横向偏移std: {lat_dev:.4f}")
print(f"  最终位置:         ({trajectory_x[-1]:.2f}, {trajectory_y[-1]:.2f})")
print(f"{'='*60}")

# ====== 保存轨迹数据用于后续画图 ======
import json
data = {
    "delay_steps": delay_steps,
    "delay_ms": delay_ms,
    "trajectory_x": trajectory_x,
    "trajectory_y": trajectory_y,
    "w_planned": angular_velocities_planned,
    "w_executed": angular_velocities_executed,
    "v_planned": linear_velocities_planned,
    "v_executed": linear_velocities_executed,
    "sign_changes_planned": sign_changes_planned,
    "sign_changes_executed": sign_changes_executed,
    "lateral_deviation": lat_dev,
}
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_delay_{delay_ms}ms_trace.json"
with open(out_path, "w") as f:
    json.dump(data, f)
print(f"轨迹数据已保存: {out_path}")
