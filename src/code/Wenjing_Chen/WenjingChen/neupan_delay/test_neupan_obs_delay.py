"""
NeuPAN 观测延迟 × Action Chunking 统一对比实验

观测延迟：planner每步都规划，但拿到的robot_state和lidar_scan是d步前的旧数据。
机器人已经移动了，但planner不知道。

4组实验：
  A. 无延迟 + 无chunk（baseline）
  B. 观测延迟d步 + 无chunk（展示问题）
  C. 无延迟 + 有chunk（chunk副作用检查）
  D. 观测延迟d步 + 有chunk（chunk能否补偿观测延迟）

用法:
  cd ~/NeuPAN/example && conda activate neupan
  python test_neupan_obs_delay.py --delay 5 --save
  python test_neupan_obs_delay.py --delay 5 --save --no-display
  python test_neupan_obs_delay.py --delay 10 --save  # 加大延迟
"""
import sys, os, argparse, json
import numpy as np
from collections import deque
from datetime import datetime

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=int, default=5, help="观测延迟步数 (5=500ms, 10=1000ms)")
parser.add_argument("--chunk-size", type=int, default=5, help="Action chunk大小")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--reps", type=int, default=3, help="每组实验重复次数")
args = parser.parse_args()

ENV_FILE = "env_turn_simple.yaml"
PLANNER_FILE = "planner_turn_simple.yaml"
dt = 0.1


def extract_actions_from_trajectory(opt_traj, dt_val):
    """从状态序列提取动作 [v, w]"""
    actions = []
    for i in range(len(opt_traj) - 1):
        s0 = np.array(opt_traj[i]).flatten()
        s1 = np.array(opt_traj[i + 1]).flatten()
        dx, dy = s1[0] - s0[0], s1[1] - s0[1]
        dtheta = (s1[2] - s0[2] + np.pi) % (2 * np.pi) - np.pi
        dist = np.sqrt(dx**2 + dy**2)
        forward = dx * np.cos(s0[2]) + dy * np.sin(s0[2])
        v = dist / dt_val if forward >= 0 else -dist / dt_val
        w = dtheta / dt_val
        actions.append((float(v), float(w)))
    return actions


def run_experiment(obs_delay, use_chunk, chunk_size, display, save_ani, max_steps, label=""):
    """
    obs_delay: 观测延迟步数。0=无延迟，5=planner看到5步前的数据
    use_chunk: True=每chunk_size步规划一次，用预规划轨迹填充；False=每步规划
    """
    env = irsim.make(ENV_FILE, save_ani=save_ani, display=display)
    planner = neupan.init_from_yaml(PLANNER_FILE)

    # 观测历史缓冲区
    buffer_len = obs_delay + 1
    state_buffer = deque(maxlen=buffer_len)
    lidar_buffer = deque(maxlen=buffer_len)

    # chunk相关
    planned_actions = []
    plan_index = 0
    steps_since_plan = chunk_size  # 强制第一步规划
    last_action = np.array([[0.0], [0.0]])

    # 记录
    trajectory_x, trajectory_y = [], []
    angular_velocities = []
    outcome = "timeout"

    for i in range(max_steps):
        # 获取当前真实状态和LiDAR
        robot_state_now = env.get_robot_state()
        lidar_scan_now = env.get_lidar_scan()

        # 存入缓冲区
        state_buffer.append(robot_state_now.copy())
        lidar_buffer.append(lidar_scan_now.copy() if hasattr(lidar_scan_now, 'copy') else lidar_scan_now)

        # 选择planner的输入：延迟的观测
        if obs_delay == 0 or len(state_buffer) < buffer_len:
            # 无延迟或缓冲区还没满：用当前观测
            state_for_planner = robot_state_now
            lidar_for_planner = lidar_scan_now
        else:
            # 用d步前的旧观测
            state_for_planner = state_buffer[0]
            lidar_for_planner = lidar_buffer[0]

        # 是否需要规划？
        if use_chunk:
            need_plan = (steps_since_plan >= chunk_size) or len(planned_actions) == 0
        else:
            need_plan = True  # 无chunk时每步都规划

        if need_plan:
            # planner用（可能延迟的）观测来规划
            points = planner.scan_to_point(state_for_planner, lidar_for_planner)
            action_now, info = planner(state_for_planner, points, None)
            last_action = action_now.copy()

            if use_chunk:
                try:
                    planned_actions = extract_actions_from_trajectory(
                        planner.opt_trajectory, dt
                    )
                except Exception:
                    planned_actions = [(float(action_now[0, 0]), float(action_now[1, 0]))]
                plan_index = 0

            steps_since_plan = 0

            if info.get("arrive", False):
                outcome = "arrive"
            if info.get("stop", False):
                pass
        else:
            info = {"arrive": False, "stop": False}

        # 选择执行的动作
        if use_chunk:
            if plan_index < len(planned_actions):
                v, w = planned_actions[plan_index]
                plan_index += 1
            else:
                v = float(last_action[0, 0])
                w = float(last_action[1, 0])
        else:
            v = float(last_action[0, 0])
            w = float(last_action[1, 0])

        steps_since_plan += 1

        action_exec = np.array([[v], [w]])

        # 记录（用真实位置）
        trajectory_x.append(float(robot_state_now[0, 0]))
        trajectory_y.append(float(robot_state_now[1, 0]))
        angular_velocities.append(float(w))

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

    ani_name = f"neupan_{label}" if save_ani else ""
    env.end(1, ani_name=ani_name)

    # 统计
    w_arr = np.array(angular_velocities)
    y_arr = np.array(trajectory_y)
    sign_changes = int(np.sum(np.diff(np.sign(w_arr)) != 0)) if len(w_arr) > 1 else 0
    lat_dev = float(np.std(y_arr[len(y_arr)//2:])) if len(y_arr) > 10 else 0.0

    return {
        "outcome": outcome,
        "steps": len(trajectory_x),
        "sign_changes": sign_changes,
        "lateral_deviation": round(lat_dev, 4),
        "final_x": round(float(trajectory_x[-1]), 2) if trajectory_x else 0,
        "final_y": round(float(trajectory_y[-1]), 2) if trajectory_y else 0,
    }


# ====== 实验配置 ======
d = args.delay
cs = args.chunk_size

experiments = [
    {"name": "A_baseline",
     "obs_delay": 0, "use_chunk": False, "chunk_size": 1,
     "desc": f"无延迟 + 无chunk（baseline）"},
    {"name": f"B_obs_delay{d}",
     "obs_delay": d, "use_chunk": False, "chunk_size": 1,
     "desc": f"观测延迟{d}步({d*100}ms) + 无chunk"},
    {"name": f"C_chunk{cs}",
     "obs_delay": 0, "use_chunk": True, "chunk_size": cs,
     "desc": f"无延迟 + chunk={cs}"},
    {"name": f"D_obs_delay{d}_chunk{cs}",
     "obs_delay": d, "use_chunk": True, "chunk_size": cs,
     "desc": f"观测延迟{d}步({d*100}ms) + chunk={cs}"},
]

all_results = {}

print(f"\n{'='*70}")
print(f"NeuPAN 观测延迟 × Action Chunking 统一对比")
print(f"观测延迟 = {d}步 ({d*100}ms), chunk_size = {cs}")
print(f"{'='*70}")

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"实验 {exp['name']}: {exp['desc']}")
    print(f"{'='*60}")

    exp_results = []
    for rep in range(args.reps):
        save_this = args.save and (rep == 0)
        result = run_experiment(
            obs_delay=exp["obs_delay"],
            use_chunk=exp["use_chunk"],
            chunk_size=exp["chunk_size"],
            display=not args.no_display,
            save_ani=save_this,
            max_steps=args.max_steps,
            label=f"{exp['name']}_rep{rep}",
        )
        icon = {"arrive": "✅", "collision": "💥", "timeout": "⏰"}.get(result["outcome"], "?")
        print(f"  rep{rep+1}: {icon} {result['outcome']} | "
              f"steps={result['steps']} | 震荡={result['sign_changes']} | "
              f"偏移={result['lateral_deviation']}")
        exp_results.append(result)

    sr = sum(1 for r in exp_results if r["outcome"] == "arrive") / len(exp_results) * 100
    avg_osc = np.mean([r["sign_changes"] for r in exp_results])
    avg_lat = np.mean([r["lateral_deviation"] for r in exp_results])
    avg_steps = np.mean([r["steps"] for r in exp_results])

    all_results[exp["name"]] = {
        "desc": exp["desc"],
        "obs_delay": exp["obs_delay"],
        "use_chunk": exp["use_chunk"],
        "success_rate": round(sr, 1),
        "avg_oscillation": round(float(avg_osc), 1),
        "avg_lateral_dev": round(float(avg_lat), 4),
        "avg_steps": round(float(avg_steps), 1),
        "details": exp_results,
    }

# ====== 最终汇总 ======
print(f"\n{'='*70}")
print(f"NeuPAN 观测延迟 × Action Chunking 对比结果")
print(f"{'='*70}")
print(f"{'实验':<35} {'SR%':<6} {'步数':<8} {'震荡':<8} {'横向偏移':<10}")
print("-" * 67)
for name, r in all_results.items():
    print(f"{r['desc']:<35} {r['success_rate']:<6} {r['avg_steps']:<8} "
          f"{r['avg_oscillation']:<8} {r['avg_lateral_dev']:<10}")

# 保存
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_obs_delay_unified_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out_path}")
