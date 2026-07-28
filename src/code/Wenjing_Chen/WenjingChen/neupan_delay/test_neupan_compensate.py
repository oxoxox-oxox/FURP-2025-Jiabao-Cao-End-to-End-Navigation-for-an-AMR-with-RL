"""
NeuPAN 观测延迟 + 状态预测补偿 实验

5组对比：
  A. 无延迟（baseline）
  B. 观测延迟 + 无补偿（展示问题）
  C. 观测延迟 + 运动学状态预测补偿（方案1）
  D. 无延迟 + chunk
  E. 观测延迟 + chunk + 状态预测补偿（组合方案）

用法:
  cd ~/NeuPAN/example && conda activate neupan
  python test_neupan_compensate.py --delay 5 --save
  python test_neupan_compensate.py --delay 10 --save
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
parser.add_argument("--delay", type=int, default=5, help="观测延迟步数")
parser.add_argument("--chunk-size", type=int, default=5, help="Action chunk大小")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--reps", type=int, default=3)
args = parser.parse_args()

ENV_FILE = "env_turn_simple.yaml"
PLANNER_FILE = "planner_turn_simple.yaml"
dt = 0.1


def predict_current_state(delayed_state, action_history, dt_val):
    """运动学前向推演：从延迟状态+历史动作预测当前状态。
    
    delayed_state: (3,1) [x, y, theta] — d步前的状态
    action_history: list of (v, w) — 从延迟时刻到现在执行过的动作
    """
    state = delayed_state.copy().flatten()  # [x, y, theta]
    
    for v, w in action_history:
        x, y, theta = state[0], state[1], state[2]
        state[0] = x + v * np.cos(theta) * dt_val
        state[1] = y + v * np.sin(theta) * dt_val
        state[2] = theta + w * dt_val
    
    return state.reshape(3, 1)


def extract_actions_from_trajectory(opt_traj, dt_val):
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


def run_experiment(obs_delay, use_chunk, chunk_size, use_compensation,
                   display, save_ani, max_steps, label=""):
    """
    obs_delay: 观测延迟步数
    use_chunk: 是否用action chunking
    use_compensation: 是否用运动学状态预测补偿
    """
    env = irsim.make(ENV_FILE, save_ani=save_ani, display=display)
    planner = neupan.init_from_yaml(PLANNER_FILE)

    # 观测缓冲区
    buffer_len = obs_delay + 1
    state_buffer = deque(maxlen=buffer_len)
    lidar_buffer = deque(maxlen=buffer_len)

    # 已执行动作的历史（用于状态预测）
    executed_action_history = deque(maxlen=obs_delay + 5)

    # chunk相关
    planned_actions = []
    plan_index = 0
    steps_since_plan = chunk_size
    last_action_vw = (0.0, 0.0)

    # 记录
    trajectory_x, trajectory_y = [], []
    angular_velocities = []
    outcome = "timeout"

    for i in range(max_steps):
        robot_state_now = env.get_robot_state()
        lidar_scan_now = env.get_lidar_scan()

        state_buffer.append(robot_state_now.copy())
        lidar_buffer.append(
            lidar_scan_now.copy() if hasattr(lidar_scan_now, 'copy') else lidar_scan_now
        )

        # ====== 选择planner输入 ======
        if obs_delay == 0 or len(state_buffer) < buffer_len:
            state_for_planner = robot_state_now
            lidar_for_planner = lidar_scan_now
        else:
            delayed_state = state_buffer[0]
            lidar_for_planner = lidar_buffer[0]

            if use_compensation and len(executed_action_history) >= obs_delay:
                # 运动学前向推演：从延迟状态预测当前状态
                recent_actions = list(executed_action_history)[-obs_delay:]
                state_for_planner = predict_current_state(
                    delayed_state, recent_actions, dt
                )
            else:
                state_for_planner = delayed_state

        # ====== 是否规划 ======
        if use_chunk:
            need_plan = (steps_since_plan >= chunk_size) or len(planned_actions) == 0
        else:
            need_plan = True

        if need_plan:
            points = planner.scan_to_point(state_for_planner, lidar_for_planner)
            action_now, info = planner(state_for_planner, points, None)

            last_action_vw = (float(action_now[0, 0]), float(action_now[1, 0]))

            if use_chunk:
                try:
                    planned_actions = extract_actions_from_trajectory(
                        planner.opt_trajectory, dt
                    )
                except Exception:
                    planned_actions = [last_action_vw]
                plan_index = 0

            steps_since_plan = 0

            if info.get("arrive", False):
                outcome = "arrive"
        else:
            info = {"arrive": False, "stop": False}

        # ====== 选择执行动作 ======
        if use_chunk:
            if plan_index < len(planned_actions):
                v, w = planned_actions[plan_index]
                plan_index += 1
            else:
                v, w = last_action_vw
        else:
            v, w = last_action_vw

        steps_since_plan += 1

        # 记录执行的动作（用于状态预测）
        executed_action_history.append((v, w))

        action_exec = np.array([[v], [w]])

        trajectory_x.append(float(robot_state_now[0, 0]))
        trajectory_y.append(float(robot_state_now[1, 0]))
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
     "obs_delay": 0, "use_chunk": False, "chunk_size": 1, "compensate": False,
     "desc": "无延迟（baseline）"},
    {"name": "B_delay_naive",
     "obs_delay": d, "use_chunk": False, "chunk_size": 1, "compensate": False,
     "desc": f"延迟{d}步 无补偿"},
    {"name": "C_delay_compensated",
     "obs_delay": d, "use_chunk": False, "chunk_size": 1, "compensate": True,
     "desc": f"延迟{d}步 + 状态预测补偿"},
    {"name": "D_delay_chunk",
     "obs_delay": d, "use_chunk": True, "chunk_size": cs, "compensate": False,
     "desc": f"延迟{d}步 + chunk={cs}"},
    {"name": "E_delay_compensated_chunk",
     "obs_delay": d, "use_chunk": True, "chunk_size": cs, "compensate": True,
     "desc": f"延迟{d}步 + 补偿 + chunk={cs}"},
]

all_results = {}

print(f"\n{'='*70}")
print(f"NeuPAN 观测延迟 + 状态预测补偿 对比实验")
print(f"观测延迟 = {d}步 ({d*100}ms)")
print(f"{'='*70}")

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"{exp['name']}: {exp['desc']}")
    print(f"{'='*60}")

    exp_results = []
    for rep in range(args.reps):
        save_this = args.save and (rep == 0)
        result = run_experiment(
            obs_delay=exp["obs_delay"],
            use_chunk=exp["use_chunk"],
            chunk_size=exp["chunk_size"],
            use_compensation=exp["compensate"],
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
        "success_rate": round(sr, 1),
        "avg_oscillation": round(float(avg_osc), 1),
        "avg_lateral_dev": round(float(avg_lat), 4),
        "avg_steps": round(float(avg_steps), 1),
        "details": exp_results,
    }

# ====== 汇总 ======
print(f"\n{'='*70}")
print(f"结果汇总 (观测延迟={d}步/{d*100}ms)")
print(f"{'='*70}")
print(f"{'实验':<30} {'SR%':<6} {'步数':<8} {'震荡':<8} {'横向偏移':<10}")
print("-" * 62)
for name, r in all_results.items():
    print(f"{r['desc']:<30} {r['success_rate']:<6} {r['avg_steps']:<8} "
          f"{r['avg_oscillation']:<8} {r['avg_lateral_dev']:<10}")

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_compensate_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out_path}")
