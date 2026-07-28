"""
NeuPAN 观测延迟 + 状态预测补偿（连续毫秒级随机延迟）

改进：
  1. 延迟单位为毫秒，每步随机采样 delay_ms ~ Uniform(min_ms, max_ms)
  2. 根据采样的delay_ms找到对应时间点的历史观测
  3. 补偿时根据实际经过的步数前向推演（步数不固定）
  4. 补偿器知道观测的时间戳，但不知道"延迟是多少"——只知道观测是哪一步的

用法:
  cd ~/NeuPAN/example && conda activate neupan

  # 随机延迟 200-800ms
  python test_neupan_ms_delay.py --min-delay-ms 200 --max-delay-ms 800 --save

  # 随机延迟 0-500ms
  python test_neupan_ms_delay.py --min-delay-ms 0 --max-delay-ms 500 --save

  # 固定500ms（退化为原来的实验，用于对照）
  python test_neupan_ms_delay.py --min-delay-ms 500 --max-delay-ms 500 --save
"""
import sys, os, argparse, json, math
import numpy as np
from collections import deque
from datetime import datetime

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

parser = argparse.ArgumentParser()
parser.add_argument("--min-delay-ms", type=int, default=200, help="最小延迟(ms)")
parser.add_argument("--max-delay-ms", type=int, default=800, help="最大延迟(ms)")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--reps", type=int, default=3)
args = parser.parse_args()

ENV_FILE = "env_turn_simple.yaml"
PLANNER_FILE = "planner_turn_simple.yaml"
STEP_TIME_MS = 100  # 每步100ms（step_time=0.1s）
dt = 0.1


def predict_current_state(delayed_state, action_history, dt_val, fractional_first_step=1.0):
    """运动学前向推演。

    delayed_state: (3,1) [x, y, theta]
    action_history: list of (v, w) — 从延迟观测之后执行的所有动作
    fractional_first_step: 第一步只推演部分时间（处理非整数步延迟）
    """
    state = delayed_state.copy().flatten()

    for idx, (v, w) in enumerate(action_history):
        # 第一步可能只需要推演部分时间
        step_dt = dt_val * (fractional_first_step if idx == 0 else 1.0)
        state[0] += v * np.cos(state[2]) * step_dt
        state[1] += v * np.sin(state[2]) * step_dt
        state[2] += w * step_dt

    return state.reshape(3, 1)


def run_experiment(min_delay_ms, max_delay_ms, use_compensation,
                   display, save_ani, max_steps, label=""):
    """
    min_delay_ms, max_delay_ms: 延迟范围(ms)，每步随机采样
    use_compensation: 是否用运动学状态预测补偿
    """
    env = irsim.make(ENV_FILE, save_ani=save_ani, display=display)
    planner = neupan.init_from_yaml(PLANNER_FILE)

    # 历史观测缓冲区：存储 (step_index, robot_state, lidar_scan)
    max_history = int(max_delay_ms / STEP_TIME_MS) + 5  # 多留几步余量
    state_history = deque(maxlen=max_history)
    lidar_history = deque(maxlen=max_history)

    # 已执行动作历史
    action_history = deque(maxlen=max_history)

    trajectory_x, trajectory_y = [], []
    angular_velocities = []
    sampled_delays_ms = []
    actual_delay_steps = []
    outcome = "timeout"
    last_action = np.array([[0.0], [0.0]])

    for i in range(max_steps):
        robot_state_now = env.get_robot_state()
        lidar_scan_now = env.get_lidar_scan()

        # 存入历史
        state_history.append(robot_state_now.copy())
        lidar_history.append(
            lidar_scan_now.copy() if hasattr(lidar_scan_now, 'copy') else lidar_scan_now
        )

        # ====== 随机采样延迟（毫秒） ======
        if min_delay_ms == 0 and max_delay_ms == 0:
            delay_ms = 0
        else:
            delay_ms = np.random.uniform(min_delay_ms, max_delay_ms)

        sampled_delays_ms.append(float(delay_ms))

        # 延迟对应多少步（可以是小数）
        delay_steps_float = delay_ms / STEP_TIME_MS
        delay_steps_int = int(math.floor(delay_steps_float))
        # 分数部分：延迟观测发生在某步的中间
        fractional_part = delay_steps_float - delay_steps_int

        actual_delay_steps.append(float(delay_steps_float))

        # ====== 选择延迟的观测 ======
        if delay_steps_int == 0 or len(state_history) <= delay_steps_int:
            # 延迟不够一步，或者历史不够长：用当前观测
            state_for_planner = robot_state_now
            lidar_for_planner = lidar_scan_now
            steps_to_predict = 0
        else:
            # 用 delay_steps_int 步前的观测
            idx = len(state_history) - 1 - delay_steps_int
            if idx < 0:
                idx = 0
            state_delayed = state_history[idx]
            lidar_for_planner = lidar_history[idx]

            if use_compensation and len(action_history) >= delay_steps_int:
                # 取从延迟时刻到现在执行过的动作
                recent_actions = list(action_history)[-delay_steps_int:]
                # 前向推演，第一步考虑分数部分
                first_step_fraction = 1.0 - fractional_part  # 延迟观测后剩余的时间
                state_for_planner = predict_current_state(
                    state_delayed, recent_actions, dt,
                    fractional_first_step=first_step_fraction,
                )
                steps_to_predict = delay_steps_int
            else:
                state_for_planner = state_delayed
                steps_to_predict = 0

        # ====== 规划 ======
        points = planner.scan_to_point(state_for_planner, lidar_for_planner)
        action, info = planner(state_for_planner, points, None)
        last_action = action.copy()

        if info.get("arrive", False):
            outcome = "arrive"

        v = float(action[0, 0])
        w = float(action[1, 0])

        # 记录执行的动作
        action_history.append((v, w))

        action_exec = np.array([[v], [w]])

        trajectory_x.append(float(robot_state_now[0, 0]))
        trajectory_y.append(float(robot_state_now[1, 0]))
        angular_velocities.append(w)

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
    avg_delay = float(np.mean(sampled_delays_ms)) if sampled_delays_ms else 0

    return {
        "outcome": outcome,
        "steps": len(trajectory_x),
        "sign_changes": sign_changes,
        "lateral_deviation": round(lat_dev, 4),
        "avg_delay_ms": round(avg_delay, 1),
        "final_x": round(float(trajectory_x[-1]), 2) if trajectory_x else 0,
        "final_y": round(float(trajectory_y[-1]), 2) if trajectory_y else 0,
    }


# ====== 实验配置 ======
experiments = [
    {"name": "A_baseline",
     "min_ms": 0, "max_ms": 0, "compensate": False,
     "desc": "无延迟 baseline"},
    {"name": "B_random_delay_naive",
     "min_ms": args.min_delay_ms, "max_ms": args.max_delay_ms, "compensate": False,
     "desc": f"随机延迟{args.min_delay_ms}-{args.max_delay_ms}ms 无补偿"},
    {"name": "C_random_delay_compensated",
     "min_ms": args.min_delay_ms, "max_ms": args.max_delay_ms, "compensate": True,
     "desc": f"随机延迟{args.min_delay_ms}-{args.max_delay_ms}ms + 补偿"},
]

all_results = {}

print(f"\n{'='*70}")
print(f"NeuPAN 连续随机延迟 + 状态预测补偿 实验")
print(f"延迟范围 = {args.min_delay_ms}-{args.max_delay_ms}ms")
print(f"{'='*70}")

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"{exp['name']}: {exp['desc']}")
    print(f"{'='*60}")

    exp_results = []
    for rep in range(args.reps):
        save_this = args.save and (rep == 0)
        result = run_experiment(
            min_delay_ms=exp["min_ms"],
            max_delay_ms=exp["max_ms"],
            use_compensation=exp["compensate"],
            display=not args.no_display,
            save_ani=save_this,
            max_steps=args.max_steps,
            label=f"{exp['name']}_rep{rep}",
        )
        icon = {"arrive": "✅", "collision": "💥", "timeout": "⏰"}.get(result["outcome"], "?")
        print(f"  rep{rep+1}: {icon} {result['outcome']} | "
              f"steps={result['steps']} | 震荡={result['sign_changes']} | "
              f"偏移={result['lateral_deviation']} | "
              f"平均延迟={result['avg_delay_ms']}ms")
        exp_results.append(result)

    sr = sum(1 for r in exp_results if r["outcome"] == "arrive") / len(exp_results) * 100
    avg_osc = np.mean([r["sign_changes"] for r in exp_results])
    avg_lat = np.mean([r["lateral_deviation"] for r in exp_results])
    avg_steps = np.mean([r["steps"] for r in exp_results])
    avg_delay = np.mean([r["avg_delay_ms"] for r in exp_results])

    all_results[exp["name"]] = {
        "desc": exp["desc"],
        "success_rate": round(sr, 1),
        "avg_oscillation": round(float(avg_osc), 1),
        "avg_lateral_dev": round(float(avg_lat), 4),
        "avg_steps": round(float(avg_steps), 1),
        "avg_delay_ms": round(float(avg_delay), 1),
        "details": exp_results,
    }

# ====== 汇总 ======
print(f"\n{'='*70}")
print(f"结果汇总")
print(f"{'='*70}")
print(f"{'实验':<35} {'SR%':<6} {'步数':<8} {'震荡':<8} {'横向偏移':<10} {'平均延迟ms':<10}")
print("-" * 77)
for name, r in all_results.items():
    print(f"{r['desc']:<35} {r['success_rate']:<6} {r['avg_steps']:<8} "
          f"{r['avg_oscillation']:<8} {r['avg_lateral_dev']:<10} {r['avg_delay_ms']:<10}")

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_ms_delay_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out_path}")
