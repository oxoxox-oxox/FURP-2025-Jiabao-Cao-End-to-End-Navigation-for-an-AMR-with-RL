"""
NeuPAN Action Chunking 延迟实验

核心思想：NeuPAN每隔chunk_size步才重新规划一次，
中间执行已规划的轨迹动作。模拟推理延迟被chunk吸收的效果。

用法:
  cd ~/NeuPAN/example && conda activate neupan

  # 无chunk（原始，每步都规划）
  python test_neupan_chunk.py --chunk 1 --save

  # chunk=5（每5步规划一次，~500ms决策周期）
  python test_neupan_chunk.py --chunk 5 --save

  # chunk=10（每10步规划一次，~1000ms决策周期）
  python test_neupan_chunk.py --chunk 10 --save
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, default=1,
                    help="Action chunk大小：每隔chunk步才重新规划 (1=原始, 5=500ms, 10=1000ms)")
parser.add_argument("--speed-scale", type=float, default=1.0, help="线速度放大因子")
parser.add_argument("--no-display", action="store_true", help="不显示窗口")
parser.add_argument("--save", action="store_true", help="保存动画")
parser.add_argument("--max-steps", type=int, default=1000, help="最大步数")
args = parser.parse_args()

chunk_size = max(1, args.chunk)
dt = 0.1  # step_time

ENV_FILE = "env_turn_simple.yaml"
PLANNER_FILE = "planner_turn_simple.yaml"

print(f"{'='*60}")
print(f"NeuPAN Action Chunking 实验")
print(f"Chunk大小 = {chunk_size} 步 (~{chunk_size * 100}ms决策周期)")
print(f"{'='*60}")

env = irsim.make(ENV_FILE, save_ani=args.save, display=not args.no_display)
planner = neupan.init_from_yaml(PLANNER_FILE)

# 存储规划出的动作序列
planned_actions = []  # list of (v, w) tuples
plan_index = 0  # 当前在chunk中的位置

# 记录数据
trajectory_x, trajectory_y = [], []
angular_velocities = []
linear_velocities = []
replan_steps = []  # 记录哪些步重新规划了
outcome = "timeout"


def extract_actions_from_trajectory(opt_traj, dt_val):
    """从opt_trajectory的状态序列提取动作序列 [v, w]。

    opt_traj: list of (3,1) arrays [x, y, theta]
    returns: list of (v, w) tuples
    """
    actions = []
    for i in range(len(opt_traj) - 1):
        s0 = np.array(opt_traj[i]).flatten()   # [x, y, theta]
        s1 = np.array(opt_traj[i + 1]).flatten()

        dx = s1[0] - s0[0]
        dy = s1[1] - s0[1]
        dtheta = s1[2] - s0[2]

        # 归一化角度差到 [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

        # 线速度 = 位移 / dt（沿heading方向的投影）
        dist = np.sqrt(dx**2 + dy**2)
        # 判断前进还是后退
        heading = s0[2]
        forward_component = dx * np.cos(heading) + dy * np.sin(heading)
        v = dist / dt_val if forward_component >= 0 else -dist / dt_val

        # 角速度
        w = dtheta / dt_val

        actions.append((v, w))

    return actions


for i in range(args.max_steps):
    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    # 是否需要重新规划？
    need_replan = (i % chunk_size == 0) or len(planned_actions) == 0 or plan_index >= len(planned_actions)

    if need_replan:
        # 调用NeuPAN规划
        points = planner.scan_to_point(robot_state, lidar_scan)
        action_now, info = planner(robot_state, points, None)

        # 从opt_trajectory提取未来动作序列
        try:
            planned_actions = extract_actions_from_trajectory(planner.opt_trajectory, dt)
        except Exception:
            planned_actions = [(float(action_now[0, 0]), float(action_now[1, 0]))]

        # chunk_size内的动作取前chunk_size个（如果轨迹够长）
        plan_index = 0
        replan_steps.append(i)

        if info.get("arrive", False):
            print(f"Step {i}: NeuPAN arrived!")
            outcome = "arrive"
        if info.get("stop", False):
            print(f"Step {i}: NeuPAN stopped")
    else:
        # 使用已规划的动作，不调用planner
        info = {"arrive": False, "stop": False}

    # 取当前chunk中的动作
    if plan_index < len(planned_actions):
        v, w = planned_actions[plan_index]
    else:
        # chunk用完了但还没到重规划时间，用最后一个动作
        v, w = planned_actions[-1] if planned_actions else (0.0, 0.0)
    plan_index += 1

    # 放大线速度
    v *= args.speed_scale

    # 构造action
    action_exec = np.array([[v], [w]])

    # 记录
    trajectory_x.append(float(robot_state[0, 0]))
    trajectory_y.append(float(robot_state[1, 0]))
    linear_velocities.append(v)
    angular_velocities.append(w)

    # 可视化
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
        print(f"Step {i}: Collision!")
        if outcome != "arrive":
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

ani_name = f"neupan_chunk{chunk_size}"
env.end(3, ani_name=ani_name)

# ====== 统计 ======
w_arr = np.array(angular_velocities)
y_arr = np.array(trajectory_y)
sign_changes = int(np.sum(np.diff(np.sign(w_arr)) != 0)) if len(w_arr) > 1 else 0
lat_dev = float(np.std(y_arr[len(y_arr)//2:])) if len(y_arr) > 10 else 0.0

print(f"\n{'='*60}")
print(f"结果统计 (chunk_size={chunk_size})")
print(f"  Outcome:         {outcome}")
print(f"  总步数:          {len(trajectory_x)}")
print(f"  重规划次数:       {len(replan_steps)}")
print(f"  角速度反转:       {sign_changes} 次")
print(f"  后半段横向偏移std: {lat_dev:.4f}")
if trajectory_x:
    print(f"  最终位置:         ({trajectory_x[-1]:.2f}, {trajectory_y[-1]:.2f})")
print(f"{'='*60}")

# 保存
import json
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_chunk{chunk_size}_{ts}.json"
data = {
    "chunk_size": chunk_size,
    "outcome": outcome,
    "total_steps": len(trajectory_x),
    "replan_count": len(replan_steps),
    "sign_changes": sign_changes,
    "lateral_deviation": lat_dev,
    "trajectory_x": trajectory_x,
    "trajectory_y": trajectory_y,
    "w_executed": angular_velocities,
    "v_executed": linear_velocities,
    "replan_steps": replan_steps,
}
with open(out_path, "w") as f:
    json.dump(data, f)
print(f"数据已保存: {out_path}")
