"""
NeuPAN 延迟注入实验 - 走廊转弯场景
验证：在不同延迟下，NeuPAN转弯后是否出现左右摇摆（震荡）

用法：cd ~/DRL-robot-navigation-IR-SIM && conda activate neupan && python test_neupan_delay.py
"""
import sys, numpy as np, time, json, os
from collections import deque

sys.path.insert(0, '.')
sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan.neupan import neupan
sys.path.insert(0, '/home/ubuntu22/DRL-robot-navigation-IR-SIM/robot_nav')
from robot_nav.SIM_ENV.sim import SIM

WORLD   = '/home/ubuntu22/DRL-robot-navigation-IR-SIM/robot_nav/worlds/neupan_corridor_train.yaml'
PLANNER = '/home/ubuntu22/NeuPAN/example/corridor_fair/diff/planner.yaml'

# 测试不同延迟级别（单位：步数）
# 假设仿真步长 ~50ms，则：
#   0步 = 0ms（无延迟基线）
#   2步 = 100ms
#   5步 = 250ms
#   10步 = 500ms（NeuPAN真机延迟）
#   20步 = 1000ms（大模型推理延迟）
DELAY_STEPS_LIST = [0, 2, 5, 10, 20]

# 测试场景：走廊中各种朝向
CASES = [
    ([[0], [20], [0]],    [[60], [20], [0]], "正向"),
    ([[0], [20], [1.57]], [[60], [20], [0]], "朝上"),
    ([[0], [20], [-1.57]],[[60], [20], [0]], "朝下"),
    ([[0], [18], [0]],    [[60], [20], [0]], "偏下"),
    ([[0], [22], [0]],    [[60], [20], [0]], "偏上"),
]

MAX_STEPS = 1000
NUM_REPS = 3  # 每个场景重复次数

results = {}

for delay_steps in DELAY_STEPS_LIST:
    delay_ms = delay_steps * 50  # 近似延迟毫秒数
    print(f"\n{'='*60}")
    print(f"延迟 = {delay_steps} 步 (~{delay_ms}ms)")
    print(f"{'='*60}")

    delay_results = {
        "delay_steps": delay_steps,
        "delay_ms": delay_ms,
        "total": 0,
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "oscillation_counts": [],  # 角速度符号反转次数
        "lateral_deviations": [],  # 横向偏移（衡量摇摆幅度）
        "times": [],
        "cases": [],
    }

    for robot_state, robot_goal, label in CASES:
        for rep in range(NUM_REPS):
            sim = SIM(world_file=WORLD, disable_plotting=True)
            planner = neupan.init_from_yaml(PLANNER)

            scan, dist, cos_val, sin_val, col, goal, a, r = sim.reset(
                robot_state=robot_state,
                robot_goal=robot_goal,
                random_obstacles=False,
            )

            state = sim.env.get_robot_state()
            start = np.array([[state[0, 0]], [state[1, 0]], [state[2, 0]]])
            goal_arr = np.array(robot_goal, dtype=float)
            planner.reset()
            planner.update_initial_path_from_goal(start, goal_arr)

            # === 延迟注入：初始化action buffer ===
            zero_action_v = 0.0
            zero_action_w = 0.0
            action_buffer_v = deque(maxlen=max(delay_steps, 1))
            action_buffer_w = deque(maxlen=max(delay_steps, 1))
            # 预填充零动作
            for _ in range(delay_steps):
                action_buffer_v.append(zero_action_v)
                action_buffer_w.append(zero_action_w)

            # 记录轨迹数据
            trajectory_y = []      # y坐标（衡量横向偏移）
            angular_velocities = []  # 角速度序列

            t0 = time.time()
            outcome = "timeout"

            for step in range(MAX_STEPS):
                rs = sim.env.get_robot_state()
                cur = np.array([[rs[0, 0]], [rs[1, 0]], [rs[2, 0]]])
                scan_dict = {
                    "ranges": scan.tolist(),
                    "angle_min": -np.pi,
                    "angle_max": np.pi,
                    "range_max": 10.0,
                    "range_min": 0.0,
                }
                pts = planner.scan_to_point(cur, scan_dict)
                action, info = planner.forward(cur, pts)

                # NeuPAN输出的当前最优动作
                v_new = action[0, 0]
                w_new = action[1, 0]

                # === 延迟注入逻辑 ===
                if delay_steps == 0:
                    # 无延迟：直接执行
                    v_exec, w_exec = v_new, w_new
                else:
                    # 有延迟：存入新动作，执行旧动作
                    action_buffer_v.append(v_new)
                    action_buffer_w.append(w_new)
                    v_exec = action_buffer_v[0]
                    w_exec = action_buffer_w[0]

                # 记录数据
                trajectory_y.append(float(cur[1, 0]))
                angular_velocities.append(float(w_exec))

                # 执行（可能延迟的）动作
                scan, dist, cos_val, sin_val, col, goal, a, r = sim.step(
                    lin_velocity=v_exec, ang_velocity=w_exec
                )

                if info.get("arrive") or goal:
                    elapsed = time.time() - t0
                    outcome = "success"
                    delay_results["success"] += 1
                    delay_results["times"].append(elapsed)
                    break
                if col:
                    outcome = "collision"
                    delay_results["collision"] += 1
                    break
                if info.get("stop"):
                    outcome = "stopped"
                    break
            else:
                delay_results["timeout"] += 1

            delay_results["total"] += 1

            # 计算震荡指标
            # 1. 角速度符号反转次数（越多 = 越震荡）
            w_arr = np.array(angular_velocities)
            if len(w_arr) > 1:
                sign_changes = np.sum(np.diff(np.sign(w_arr)) != 0)
            else:
                sign_changes = 0
            delay_results["oscillation_counts"].append(int(sign_changes))

            # 2. 横向偏移标准差（越大 = 摇摆越严重）
            y_arr = np.array(trajectory_y)
            if len(y_arr) > 10:
                # 取后半段轨迹（转弯后的直线部分）
                y_second_half = y_arr[len(y_arr) // 2:]
                lat_dev = float(np.std(y_second_half))
            else:
                lat_dev = 0.0
            delay_results["lateral_deviations"].append(lat_dev)

            delay_results["cases"].append({
                "label": label,
                "rep": rep + 1,
                "outcome": outcome,
                "oscillation_count": int(sign_changes),
                "lateral_deviation": round(lat_dev, 4),
            })

            status_icon = {"success": "✅", "collision": "💥", "stopped": "🛑", "timeout": "⏰"}.get(outcome, "?")
            print(f"  {label} rep{rep+1}: {status_icon} {outcome} | 震荡={sign_changes}次 | 横向偏移std={lat_dev:.4f}")

            sim.env.end()

    sr = delay_results["success"] / delay_results["total"] * 100 if delay_results["total"] > 0 else 0
    avg_osc = np.mean(delay_results["oscillation_counts"]) if delay_results["oscillation_counts"] else 0
    avg_lat = np.mean(delay_results["lateral_deviations"]) if delay_results["lateral_deviations"] else 0

    print(f"\n  汇总: SR={sr:.0f}% | 平均震荡={avg_osc:.1f}次 | 平均横向偏移={avg_lat:.4f}")
    results[f"delay_{delay_steps}"] = {
        "delay_steps": delay_steps,
        "delay_ms": delay_ms,
        "success_rate": round(sr, 1),
        "avg_oscillation": round(float(avg_osc), 1),
        "avg_lateral_deviation": round(float(avg_lat), 4),
        "collision_count": delay_results["collision"],
        "timeout_count": delay_results["timeout"],
    }

# 最终汇总
print(f"\n{'='*60}")
print("NeuPAN 延迟实验汇总")
print(f"{'='*60}")
print(f"{'延迟(ms)':<10} {'SR%':<8} {'平均震荡':<10} {'横向偏移':<12} {'碰撞':<6} {'超时':<6}")
print("-" * 52)
for key in sorted(results.keys(), key=lambda x: results[x]["delay_steps"]):
    r = results[key]
    print(f"{r['delay_ms']:<10} {r['success_rate']:<8} {r['avg_oscillation']:<10} {r['avg_lateral_deviation']:<12} {r['collision_count']:<6} {r['timeout_count']:<6}")

# 保存结果
output_path = "/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_delay_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {output_path}")
