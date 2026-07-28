"""
NeuPAN L1延迟补偿：自身状态运动学预测 + 障碍物线性外推

补偿两部分：
  1. 自身状态：用运动学+历史动作预测当前位置（已验证有效）
  2. 障碍物位置：用scan_to_point_velocity估计点速度，线性外推到当前时刻

对比：
  A. 无延迟 baseline
  B. 有延迟 无补偿
  C. 有延迟 + 仅自身状态补偿（之前的方案）
  D. 有延迟 + 自身状态 + 障碍物外推（L1完整方案）

用法:
  cd ~/NeuPAN/example && conda activate neupan
  python test_neupan_L1.py --min-delay-ms 100 --max-delay-ms 1000 --reps 5 --no-display
  python test_neupan_L1.py --min-delay-ms 200 --max-delay-ms 800 --reps 5
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
parser.add_argument("--min-delay-ms", type=int, default=100)
parser.add_argument("--max-delay-ms", type=int, default=1000)
parser.add_argument("--env", type=str, default="env_turn_dynamic.yaml")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--reps", type=int, default=5)
args = parser.parse_args()

PLANNER_FILE = "planner_turn_simple.yaml"
STEP_TIME_MS = 100
dt = 0.1


def predict_robot_state(delayed_state, action_history, dt_val):
    """运动学前向推演预测机器人当前状态"""
    state = delayed_state.copy().flatten()
    for v, w in action_history:
        state[0] += v * np.cos(state[2]) * dt_val
        state[1] += v * np.sin(state[2]) * dt_val
        state[2] += w * dt_val
    return state.reshape(3, 1)


def predict_obstacle_points(delayed_points, point_velocities, delay_seconds):
    """线性外推障碍物点云到当前时刻
    
    delayed_points: (2, N) 延迟时刻的障碍物点坐标
    point_velocities: (2, N) 每个点的估计速度
    delay_seconds: 延迟时间（秒）
    """
    if point_velocities is None or delayed_points is None:
        return delayed_points
    
    # 确保shape匹配
    if delayed_points.shape != point_velocities.shape:
        return delayed_points
    
    predicted = delayed_points + point_velocities * delay_seconds
    return predicted


# 补偿模式定义
MODES = {
    "A_baseline": {
        "desc": "无延迟 baseline",
        "add_delay": False, "compensate_self": False, "compensate_obs": False,
    },
    "B_naive": {
        "desc": "有延迟 无补偿",
        "add_delay": True, "compensate_self": False, "compensate_obs": False,
    },
    "C_self_only": {
        "desc": "有延迟 + 仅自身状态补偿",
        "add_delay": True, "compensate_self": True, "compensate_obs": False,
    },
    "D_L1_full": {
        "desc": "有延迟 + 自身状态 + 障碍物外推(L1)",
        "add_delay": True, "compensate_self": True, "compensate_obs": True,
    },
}


def run_experiment(mode_cfg, min_delay_ms, max_delay_ms,
                   display, save_ani, max_steps, label=""):

    env = irsim.make(args.env, save_ani=False, display=display)
    planner = neupan.init_from_yaml(PLANNER_FILE)

    add_delay = mode_cfg["add_delay"]
    compensate_self = mode_cfg["compensate_self"]
    compensate_obs = mode_cfg["compensate_obs"]

    max_history = int(max_delay_ms / STEP_TIME_MS) + 5
    state_history = deque(maxlen=max_history)
    lidar_history = deque(maxlen=max_history)
    action_history = deque(maxlen=max_history)

    trajectory_x, trajectory_y = [], []
    angular_velocities = []
    sampled_delays_ms = []
    outcome = "timeout"

    for i in range(max_steps):
        robot_state_now = env.get_robot_state()
        lidar_scan_now = env.get_lidar_scan()

        state_history.append(robot_state_now.copy())
        lidar_history.append(
            lidar_scan_now.copy() if hasattr(lidar_scan_now, 'copy') else lidar_scan_now
        )

        # ====== 采样延迟 ======
        if add_delay and min_delay_ms > 0:
            delay_ms = np.random.uniform(min_delay_ms, max_delay_ms)
        else:
            delay_ms = 0
        sampled_delays_ms.append(float(delay_ms))

        delay_steps_int = int(math.floor(delay_ms / STEP_TIME_MS))
        delay_seconds = delay_ms / 1000.0

        # ====== 选择观测 ======
        if delay_steps_int == 0 or not add_delay or len(state_history) <= delay_steps_int:
            # 无延迟或延迟不够：用当前观测
            state_for_planner = robot_state_now
            lidar_for_planner = lidar_scan_now

            # 正常调用
            points = planner.scan_to_point(state_for_planner, lidar_for_planner)
            action, info = planner(state_for_planner, points, None)

        else:
            # 取延迟的观测
            idx = len(state_history) - 1 - delay_steps_int
            if idx < 0:
                idx = 0
            delayed_state = state_history[idx]
            delayed_lidar = lidar_history[idx]

            # === 自身状态补偿 ===
            if compensate_self and len(action_history) >= delay_steps_int:
                recent_actions = list(action_history)[-delay_steps_int:]
                predicted_state = predict_robot_state(delayed_state, recent_actions, dt)
            else:
                predicted_state = delayed_state

            # === 障碍物点云补偿 ===
            if compensate_obs:
                # 用scan_to_point_velocity估计点速度
                try:
                    delayed_points, point_vels = planner.scan_to_point_velocity(
                        delayed_state, delayed_lidar
                    )
                    # 线性外推到当前时刻
                    predicted_points = predict_obstacle_points(
                        delayed_points, point_vels, delay_seconds
                    )
                    # 直接用预测的点和状态调用planner
                    action, info = planner(predicted_state, predicted_points, None)
                except Exception as e:
                    # fallback: 不补偿障碍物
                    points = planner.scan_to_point(predicted_state, delayed_lidar)
                    action, info = planner(predicted_state, points, None)
            else:
                # 不补偿障碍物，用延迟lidar + 补偿/未补偿的state
                points = planner.scan_to_point(predicted_state, delayed_lidar)
                action, info = planner(predicted_state, points, None)

        if info.get("arrive", False):
            outcome = "arrive"

        v = float(action[0, 0])
        w = float(action[1, 0])
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

    env.end(0)

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
    }


# ====== 运行实验 ======
all_results = {}

print(f"\n{'='*70}")
print(f"NeuPAN L1延迟补偿实验（自身状态 + 障碍物外推）")
print(f"延迟范围 = {args.min_delay_ms}-{args.max_delay_ms}ms, 场景 = {args.env}")
print(f"{'='*70}")

for mode_name, mode_cfg in MODES.items():
    print(f"\n{'='*60}")
    print(f"{mode_name}: {mode_cfg['desc']}")
    print(f"{'='*60}")

    exp_results = []
    for rep in range(args.reps):
        result = run_experiment(
            mode_cfg=mode_cfg,
            min_delay_ms=args.min_delay_ms,
            max_delay_ms=args.max_delay_ms,
            display=not args.no_display,
            save_ani=False,
            max_steps=args.max_steps,
            label=f"{mode_name}_rep{rep}",
        )
        icon = {"arrive": "✅", "collision": "💥", "timeout": "⏰"}.get(result["outcome"], "?")
        print(f"  rep{rep+1}: {icon} {result['outcome']} | "
              f"steps={result['steps']} | 震荡={result['sign_changes']} | "
              f"偏移={result['lateral_deviation']} | 延迟={result['avg_delay_ms']}ms")
        exp_results.append(result)

    sr = sum(1 for r in exp_results if r["outcome"] == "arrive") / len(exp_results) * 100
    avg_osc = np.mean([r["sign_changes"] for r in exp_results])
    avg_lat = np.mean([r["lateral_deviation"] for r in exp_results])
    avg_steps = np.mean([r["steps"] for r in exp_results])
    avg_delay = np.mean([r["avg_delay_ms"] for r in exp_results])
    collision_rate = sum(1 for r in exp_results if r["outcome"] == "collision") / len(exp_results) * 100

    all_results[mode_name] = {
        "desc": mode_cfg["desc"],
        "success_rate": round(sr, 1),
        "collision_rate": round(collision_rate, 1),
        "avg_oscillation": round(float(avg_osc), 1),
        "avg_lateral_dev": round(float(avg_lat), 4),
        "avg_steps": round(float(avg_steps), 1),
        "avg_delay_ms": round(float(avg_delay), 1),
        "details": exp_results,
    }

# ====== 汇总 ======
print(f"\n{'='*75}")
print(f"L1延迟补偿结果汇总 (延迟={args.min_delay_ms}-{args.max_delay_ms}ms)")
print(f"{'='*75}")
print(f"{'实验':<35} {'SR%':<6} {'碰撞%':<7} {'步数':<8} {'震荡':<8} {'横向偏移':<10}")
print("-" * 74)
for name, r in all_results.items():
    print(f"{r['desc']:<35} {r['success_rate']:<6} {r['collision_rate']:<7} "
          f"{r['avg_steps']:<8} {r['avg_oscillation']:<8} {r['avg_lateral_dev']:<10}")

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_L1_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out_path}")
