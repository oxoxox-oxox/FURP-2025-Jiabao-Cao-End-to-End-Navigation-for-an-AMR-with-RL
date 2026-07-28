"""
NeuPAN 统一对比实验：延迟 × Action Chunking

4组实验：
  A. 无延迟 + 无chunk（baseline）
  B. 随机inference延迟 + 无chunk（展示问题）
  C. 无延迟 + 有chunk（chunk副作用检查）
  D. 随机inference延迟 + 有chunk（chunk能否解决延迟）

延迟位置：拉长决策周期（inference delay）
  - 每隔 delay_interval 步才允许重新规划
  - 中间步骤：无chunk时重复旧动作，有chunk时执行预规划轨迹

用法:
  cd ~/NeuPAN/example && conda activate neupan
  python test_neupan_unified.py --save
  python test_neupan_unified.py --save --no-display  # 无窗口批量跑
"""
import sys, os, argparse, json
import numpy as np
from datetime import datetime

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

parser = argparse.ArgumentParser()
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


def run_experiment(delay_interval, use_chunk, display, save_ani, max_steps, label=""):
    """
    delay_interval: 决策间隔步数。1=每步决策（无延迟），5=每5步决策一次
    use_chunk: True=用预规划轨迹填充间隔，False=重复旧动作
    """
    env = irsim.make(ENV_FILE, save_ani=save_ani, display=display)
    planner = neupan.init_from_yaml(PLANNER_FILE)

    # 状态变量
    last_action = np.array([[0.0], [0.0]])
    planned_actions = []
    plan_index = 0
    steps_since_plan = delay_interval  # 强制第一步规划

    trajectory_x, trajectory_y = [], []
    angular_velocities = []
    outcome = "timeout"

    for i in range(max_steps):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()

        # 是否该重新规划？
        need_replan = (steps_since_plan >= delay_interval)

        if need_replan:
            points = planner.scan_to_point(robot_state, lidar_scan)
            action_now, info = planner(robot_state, points, None)
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

        # 选择要执行的动作
        if need_replan:
            if use_chunk and len(planned_actions) > 0:
                v, w = planned_actions[0]
                plan_index = 1
            else:
                v = float(last_action[0, 0])
                w = float(last_action[1, 0])
        else:
            if use_chunk and plan_index < len(planned_actions):
                v, w = planned_actions[plan_index]
                plan_index += 1
            else:
                # 无chunk或chunk用完：重复上次动作
                v = float(last_action[0, 0])
                w = float(last_action[1, 0])

        steps_since_plan += 1

        action_exec = np.array([[v], [w]])

        trajectory_x.append(float(robot_state[0, 0]))
        trajectory_y.append(float(robot_state[1, 0]))
        angular_velocities.append(float(w))

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
experiments = [
    {"name": "A_no_delay_no_chunk",   "delay_interval": 1, "use_chunk": False,
     "desc": "无延迟 + 无chunk（baseline）"},
    {"name": "B_delay5_no_chunk",     "delay_interval": 5, "use_chunk": False,
     "desc": "5步inference延迟 + 无chunk（重复旧动作）"},
    {"name": "C_no_delay_chunk",      "delay_interval": 1, "use_chunk": True,
     "desc": "无延迟 + 有chunk（chunk副作用检查）"},
    {"name": "D_delay5_chunk",        "delay_interval": 5, "use_chunk": True,
     "desc": "5步inference延迟 + 有chunk（chunk解决延迟）"},
]

all_results = {}

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"实验 {exp['name']}: {exp['desc']}")
    print(f"  决策间隔={exp['delay_interval']}步, chunk={exp['use_chunk']}")
    print(f"{'='*60}")

    exp_results = []
    for rep in range(args.reps):
        save_this = args.save and (rep == 0)  # 只保存第一次的动画
        result = run_experiment(
            delay_interval=exp["delay_interval"],
            use_chunk=exp["use_chunk"],
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

    # 汇总
    sr = sum(1 for r in exp_results if r["outcome"] == "arrive") / len(exp_results) * 100
    avg_osc = np.mean([r["sign_changes"] for r in exp_results])
    avg_lat = np.mean([r["lateral_deviation"] for r in exp_results])
    avg_steps = np.mean([r["steps"] for r in exp_results])

    all_results[exp["name"]] = {
        "desc": exp["desc"],
        "delay_interval": exp["delay_interval"],
        "use_chunk": exp["use_chunk"],
        "success_rate": round(sr, 1),
        "avg_oscillation": round(float(avg_osc), 1),
        "avg_lateral_dev": round(float(avg_lat), 4),
        "avg_steps": round(float(avg_steps), 1),
        "details": exp_results,
    }

# ====== 最终汇总 ======
print(f"\n{'='*70}")
print(f"NeuPAN 延迟 × Action Chunking 统一对比")
print(f"{'='*70}")
print(f"{'实验':<30} {'SR%':<6} {'平均步数':<10} {'平均震荡':<10} {'横向偏移':<10}")
print("-" * 66)
for name, r in all_results.items():
    print(f"{r['desc']:<30} {r['success_rate']:<6} {r['avg_steps']:<10} "
          f"{r['avg_oscillation']:<10} {r['avg_lateral_dev']:<10}")

# 保存
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_unified_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out_path}")
