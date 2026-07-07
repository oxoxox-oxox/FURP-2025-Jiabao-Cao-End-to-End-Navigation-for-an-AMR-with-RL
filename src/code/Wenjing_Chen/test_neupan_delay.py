"""
NeuPAN 延迟可视化实验 - 走廊转弯场景
基于原版 corridor 场景（验证可用），注入不同延迟观察转弯后的震荡

用法：cd ~/NeuPAN/example && conda activate neupan && python test_neupan_delay_vis.py
可选参数：
  --delay 10           # 延迟步数上限（对于随机延迟为最大值；默认0）
  --random-delay       # 启用每步随机延迟（在 [0, delay] 之间采样），否则为固定延迟
  --speed-scale 1.2    # 执行时线速度放大因子（默认1.2）
  --no-display         # 不显示窗口
  --save               # 保存动画
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
parser.add_argument("--delay", type=int, default=0, help="延迟步数上限 (0=无延迟, 5=250ms, 10=500ms, 20=1000ms)")
parser.add_argument("--random-delay", action="store_true", help="启用随机延迟（每步在 [0, delay] 之间采样），否则为固定延迟")
parser.add_argument("--speed-scale", type=float, default=1.2, help="执行时线速度放大因子（默认1.2）")
parser.add_argument("--no-display", action="store_true", help="不显示窗口")
parser.add_argument("--save", action="store_true", help="保存动画")
parser.add_argument("--max-steps", type=int, default=1000, help="最大步数")
args = parser.parse_args()

max_delay = max(0, args.delay)
use_random_delay = args.random_delay
speed_scale = float(args.speed_scale)

delay_ms = max_delay * 100  # step_time=0.1s, 所以每步100ms

ENV_FILE = "env_turn_simple.yaml"
PLANNER_FILE = "planner_turn_simple.yaml"

print(f"{'='*60}")
print(f"NeuPAN 延迟可视化实验")
print(f"最大延迟 = {max_delay} 步 (~{delay_ms}ms)")
print(f"随机延迟 = {use_random_delay}")
print(f"线速度放大因子 = {speed_scale}")
print(f"{'='*60}")

# ====== 初始化环境 ======
env = irsim.make(
    ENV_FILE,
    save_ani=args.save,
    display=not args.no_display,
)
planner = neupan.init_from_yaml(PLANNER_FILE)

# ====== 延迟Buffer初始化 ======
# 为支持随机延迟，我们初始化一个长度为 max_delay+1 的缓冲区，右端为最新动作
buffer_len = max_delay + 1
action_buffer = deque()
for _ in range(buffer_len):
    action_buffer.append(np.array([[0.0], [0.0]]))  # 零动作填充

# ====== 记录数据 ======
trajectory_x = []
trajectory_y = []
angular_velocities_planned = []   # NeuPAN计划的角速度
angular_velocities_executed = []  # 实际执行的角速度
linear_velocities_planned = []
linear_velocities_executed = []
sampled_delays = []  # 每一步实际使用的延迟（步数）

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

    # 记录NeuPAN计划的动作（未放大）
    v_planned = float(action[0, 0])
    w_planned = float(action[1, 0])
    linear_velocities_planned.append(v_planned)
    angular_velocities_planned.append(w_planned)

    # ====== 延迟注入（支持固定或随机延迟） ======
    # 将当前动作加入缓冲区右端
    action_buffer.append(action.copy())
    # 保持缓冲区长度不超过 buffer_len
    while len(action_buffer) > buffer_len:
        action_buffer.popleft()

    if max_delay == 0:
        sampled_delay = 0
    else:
        if use_random_delay:
            sampled_delay = int(np.random.randint(0, max_delay + 1))
        else:
            sampled_delay = max_delay

    # 取出延迟后的动作（-1 表示最新动作，-(d+1) 表示延迟 d 步）
    action_exec = action_buffer[-(sampled_delay + 1)].copy()
    sampled_delays.append(int(sampled_delay))

    # 在执行前放大线速度（仅放大线速度，不改变角速度）
    action_exec_scaled = action_exec.copy()
    action_exec_scaled[0, 0] = float(action_exec_scaled[0, 0]) * speed_scale

    v_exec = float(action_exec_scaled[0, 0])
    w_exec = float(action_exec_scaled[1, 0])
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

    # 执行（可能延迟且放大线速度的）动作
    env.step(action_exec_scaled)
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
print(f"结果统计 (最大延迟={delay_ms}ms, 随机延迟={use_random_delay}, 线速度放大={speed_scale})")
print(f"  总步数:          {len(trajectory_x)}")
print(f"  角速度反转(计划): {sign_changes_planned} 次")
print(f"  角速度反转(执行): {sign_changes_executed} 次")
print(f"  后半段横向偏移std: {lat_dev:.4f}")
if len(trajectory_x) > 0:
    print(f"  最终位置:         ({trajectory_x[-1]:.2f}, {trajectory_y[-1]:.2f})")
else:
    print(f"  最终位置:         (N/A)")
print(f"  延迟样本统计 (前20步): {sampled_delays[:20]}")
print(f"{'='*60}")

# ====== 保存轨迹数据用于后续画图 ======
import json
data = {
    "max_delay_steps": max_delay,
    "random_delay": use_random_delay,
    "speed_scale": speed_scale,
    "delay_ms_max": delay_ms,
    "sampled_delays": sampled_delays,
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