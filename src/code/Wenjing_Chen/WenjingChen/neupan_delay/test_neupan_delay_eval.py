"""
NeuPAN 延迟评估（重写版 v2）

修正点：
  1. 延迟位置：只支持 observation（传感器→planner）和 inference（拉长决策周期），
     没有 action→执行 的延迟
  2. 偏移指标：只在"自由直行段"统计——前方扇区(±45°)最近障碍物>5m 且 已过静态障碍(x>15)
     时才计入 |y-20| 偏差；绕障碍物的段落自动排除，不再污染指标
  3. L1障碍物外推：加速度过滤（只外推 0.15~2.0 m/s 的点），静态墙不再被错误移动
  4. 固定种子：numpy + random 双重固定，延迟采样用独立RNG，保证A/B/C/D四组
     障碍物轨迹完全一致、结果可复现

用法:
  cd ~/NeuPAN/example && conda activate neupan

  # observation延迟模式（默认）
  python test_neupan_delay_eval.py --min-delay-ms 100 --max-delay-ms 1000 --reps 5 --no-display

  # inference延迟模式（拉长决策周期）
  python test_neupan_delay_eval.py --delay-mode inference --min-delay-ms 100 --max-delay-ms 1000 --reps 5 --no-display

  # 换DUNE模型
  python test_neupan_delay_eval.py --planner planner_delay_rand_trained.yaml --no-display
"""
import sys, os, argparse, json, math, random
import numpy as np
from collections import deque
from datetime import datetime

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example')

from neupan import neupan
import irsim

parser = argparse.ArgumentParser()
parser.add_argument("--delay-mode", type=str, default="observation",
                    choices=["observation", "inference"],
                    help="observation=旧传感器数据; inference=拉长决策周期")
parser.add_argument("--min-delay-ms", type=int, default=100)
parser.add_argument("--max-delay-ms", type=int, default=1000)
parser.add_argument("--planner", type=str, default="planner_turn_simple.yaml")
parser.add_argument("--env", type=str, default="env_turn_dynamic.yaml")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--reps", type=int, default=5)
args = parser.parse_args()

STEP_TIME_MS = 100
dt = 0.1
SEEDS = [42, 1, 7, 100, 2026, 555, 888, 1234, 3407, 9099]  # 支持最多10次rep
REF_Y = 20.0          # 走廊中心线
FREE_FRONT_DIST = 5.0  # 前方扇区最近障碍>5m 才算"自由直行"
PAST_OBSTACLE_X = 15.0 # 过了静态障碍(x=10)之后才开始统计


def predict_robot_state(delayed_state, action_history, dt_val):
    """运动学前向推演预测机器人当前状态"""
    state = delayed_state.copy().flatten()
    for v, w in action_history:
        state[0] += v * np.cos(state[2]) * dt_val
        state[1] += v * np.sin(state[2]) * dt_val
        state[2] += w * dt_val
    return state.reshape(3, 1)


def predict_obstacle_points(delayed_points, point_velocities, delay_seconds):
    """障碍物点云外推——只外推确实在动的点（修复静态墙漂移bug）"""
    if point_velocities is None or delayed_points is None:
        return delayed_points
    if delayed_points.shape != point_velocities.shape:
        return delayed_points
    speeds = np.linalg.norm(point_velocities, axis=0)
    mask = (speeds > 0.15) & (speeds < 2.0)  # 静态墙噪声速度<0.15，异常值>2.0都不外推
    predicted = delayed_points.copy()
    predicted[:, mask] = delayed_points[:, mask] + point_velocities[:, mask] * delay_seconds
    return predicted


def front_min_range(lidar_scan):
    """前方±45°扇区的最近距离（用于判断是否处于自由直行段）"""
    try:
        if isinstance(lidar_scan, dict):
            ranges = np.array(lidar_scan["ranges"])
            amin = lidar_scan.get("angle_min", -np.pi)
            amax = lidar_scan.get("angle_max", np.pi)
        else:
            ranges = np.array(lidar_scan)
            amin, amax = -np.pi, np.pi
        n = len(ranges)
        angles = np.linspace(amin, amax, n)
        front_mask = np.abs(angles) < (np.pi / 4)
        if front_mask.sum() == 0:
            return np.inf
        vals = ranges[front_mask]
        vals = vals[np.isfinite(vals) & (vals > 0.01)]
        return float(np.min(vals)) if len(vals) else np.inf
    except Exception:
        return np.inf


MODES = {
    "A_baseline":  {"add_delay": False, "comp_self": False, "comp_obs": False,
                    "desc": "无延迟 baseline"},
    "B_naive":     {"add_delay": True,  "comp_self": False, "comp_obs": False,
                    "desc": "有延迟 无补偿"},
    "C_self_only": {"add_delay": True,  "comp_self": True,  "comp_obs": False,
                    "desc": "有延迟 + 仅自身状态补偿"},
    "D_L1_full":   {"add_delay": True,  "comp_self": True,  "comp_obs": True,
                    "desc": "有延迟 + L1(自身+动态点外推)"},
}


def run_once(mode_cfg, seed, display):
    # ==== 固定所有随机源，保证四组障碍物轨迹一致 ====
    random.seed(seed)
    np.random.seed(seed)
    delay_rng = np.random.default_rng(seed + 77777)  # 延迟采样独立RNG，不干扰场景随机流

    env = irsim.make(args.env, save_ani=False, display=display)
    planner = neupan.init_from_yaml(args.planner)

    add_delay = mode_cfg["add_delay"]
    comp_self = mode_cfg["comp_self"]
    comp_obs = mode_cfg["comp_obs"]

    max_hist = int(args.max_delay_ms / STEP_TIME_MS) + 5
    state_hist = deque(maxlen=max_hist)
    lidar_hist = deque(maxlen=max_hist)
    action_hist = deque(maxlen=max_hist)

    # inference模式的状态
    last_action = np.array([[0.0], [0.0]])
    inf_counter = 0

    free_dev_samples = []   # 自由直行段的|y-REF_Y|
    w_series = []
    outcome, steps_used = "timeout", 0

    for i in range(args.max_steps):
        rs_now = env.get_robot_state()
        ls_now = env.get_lidar_scan()
        state_hist.append(rs_now.copy())
        lidar_hist.append(ls_now.copy() if hasattr(ls_now, "copy") else ls_now)

        # ==== 延迟采样 ====
        if add_delay:
            delay_ms = float(delay_rng.uniform(args.min_delay_ms, args.max_delay_ms))
        else:
            delay_ms = 0.0
        d_steps = int(math.floor(delay_ms / STEP_TIME_MS))
        d_sec = delay_ms / 1000.0

        info = {"arrive": False, "stop": False}

        if args.delay_mode == "inference" and add_delay:
            # ==== inference延迟：每 d_steps+1 步才规划一次，中间重复上次动作 ====
            if inf_counter <= 0:
                pts = planner.scan_to_point(rs_now, ls_now)
                action, info = planner(rs_now, pts, None)
                last_action = action.copy()
                inf_counter = d_steps  # 接下来 d_steps 步不再规划
            else:
                action = last_action.copy()
                inf_counter -= 1
        else:
            # ==== observation延迟：每步规划，但输入是旧观测 ====
            if (not add_delay) or d_steps == 0 or len(state_hist) <= d_steps:
                s_in, l_in = rs_now, ls_now
                pts = planner.scan_to_point(s_in, l_in)
                action, info = planner(s_in, pts, None)
            else:
                idx = max(0, len(state_hist) - 1 - d_steps)
                s_delay, l_delay = state_hist[idx], lidar_hist[idx]

                if comp_self and len(action_hist) >= d_steps:
                    s_in = predict_robot_state(
                        s_delay, list(action_hist)[-d_steps:], dt)
                else:
                    s_in = s_delay

                if comp_obs:
                    try:
                        p_delay, p_vel = planner.scan_to_point_velocity(s_delay, l_delay)
                        p_pred = predict_obstacle_points(p_delay, p_vel, d_sec)
                        action, info = planner(s_in, p_pred, None)
                    except Exception:
                        pts = planner.scan_to_point(s_in, l_delay)
                        action, info = planner(s_in, pts, None)
                else:
                    pts = planner.scan_to_point(s_in, l_delay)
                    action, info = planner(s_in, pts, None)

        if info.get("arrive", False):
            outcome = "arrive"

        v, w = float(action[0, 0]), float(action[1, 0])
        action_hist.append((v, w))
        w_series.append(w)

        # ==== 自由直行段偏移统计（排除避障段）====
        x_now, y_now = float(rs_now[0, 0]), float(rs_now[1, 0])
        if x_now > PAST_OBSTACLE_X and front_min_range(ls_now) > FREE_FRONT_DIST:
            free_dev_samples.append(abs(y_now - REF_Y))

        env.step(np.array([[v], [w]]))
        env.render()

        if env.done():
            if outcome != "arrive":
                outcome = "collision"
            steps_used = i + 1
            break
        if outcome == "arrive":
            steps_used = i + 1
            break
    else:
        steps_used = args.max_steps

    env.end(0)

    w_arr = np.array(w_series)
    osc = int(np.sum(np.diff(np.sign(w_arr)) != 0)) if len(w_arr) > 1 else 0
    osc_per_100 = osc / max(steps_used, 1) * 100
    free_dev = float(np.mean(free_dev_samples)) if free_dev_samples else float("nan")

    return {
        "outcome": outcome,
        "steps": steps_used,
        "osc_per_100steps": round(osc_per_100, 1),
        "free_straight_dev": round(free_dev, 4) if not math.isnan(free_dev) else None,
        "n_free_samples": len(free_dev_samples),
    }


# ================= 主流程 =================
all_results = {}
print(f"\n{'='*72}")
print(f"NeuPAN 延迟评估 v2 | 模式={args.delay_mode} | "
      f"延迟={args.min_delay_ms}-{args.max_delay_ms}ms | planner={args.planner}")
print(f"偏移指标 = 自由直行段(前方>{FREE_FRONT_DIST}m, x>{PAST_OBSTACLE_X})的|y-{REF_Y}|均值")
print(f"{'='*72}")

for name, cfg in MODES.items():
    print(f"\n--- {name}: {cfg['desc']} ---")
    reps_out = []
    for rep in range(args.reps):
        seed = SEEDS[rep % len(SEEDS)]
        r = run_once(cfg, seed, display=not args.no_display)
        icon = {"arrive": "✅", "collision": "💥", "timeout": "⏰"}.get(r["outcome"], "?")
        print(f"  seed={seed}: {icon} {r['outcome']} | steps={r['steps']} | "
              f"震荡/100步={r['osc_per_100steps']} | 直行段偏移={r['free_straight_dev']} "
              f"(n={r['n_free_samples']})")
        reps_out.append(r)

    n = len(reps_out)
    sr = sum(1 for r in reps_out if r["outcome"] == "arrive") / n * 100
    col = sum(1 for r in reps_out if r["outcome"] == "collision") / n * 100
    devs = [r["free_straight_dev"] for r in reps_out if r["free_straight_dev"] is not None]
    all_results[name] = {
        "desc": cfg["desc"],
        "SR%": round(sr, 1),
        "collision%": round(col, 1),
        "avg_steps": round(float(np.mean([r["steps"] for r in reps_out])), 1),
        "avg_osc_per_100": round(float(np.mean([r["osc_per_100steps"] for r in reps_out])), 1),
        "avg_free_dev": round(float(np.mean(devs)), 4) if devs else None,
        "details": reps_out,
    }

print(f"\n{'='*78}")
print(f"汇总 (delay-mode={args.delay_mode}, {args.min_delay_ms}-{args.max_delay_ms}ms, "
      f"planner={args.planner})")
print(f"{'='*78}")
print(f"{'实验':<32} {'SR%':<6} {'碰撞%':<7} {'步数':<8} {'震荡/100步':<10} {'直行段偏移':<10}")
print("-" * 76)
for name, r in all_results.items():
    print(f"{r['desc']:<32} {r['SR%']:<6} {r['collision%']:<7} {r['avg_steps']:<8} "
          f"{r['avg_osc_per_100']:<10} {str(r['avg_free_dev']):<10}")

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_eval_v2_{args.delay_mode}_{ts}.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out}")
