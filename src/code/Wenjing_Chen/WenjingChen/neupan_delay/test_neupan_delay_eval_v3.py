"""
NeuPAN 延迟评估 v3

相比v2的修复：
  1. 【关键】到达判定改用"到目标距离<1.5m"，不再依赖planner的info["arrive"]
     （v2中env.done()在到达和碰撞时都返回True，且planner的arrive判定滞后，
      导致正常到达被误记为collision——之前动态场景的碰撞率100%多为误判）
  2. 偏移指标只在静态场景报告（动态场景绕障碍物属正常行为，偏移无意义），
     动态场景只报SR/碰撞率/超时率
  3. 保留v2的所有修复：固定种子、独立延迟RNG、静态点速度过滤、直行段偏移

用法:
  cd ~/NeuPAN/example && conda activate neupan

  # 实验1：静态场景（偏移/震荡主实验）
  python test_neupan_delay_eval_v3.py --env env_turn_simple.yaml --reps 5 --no-display

  # 实验2：动态场景（避障SR主实验）
  python test_neupan_delay_eval_v3.py --env env_turn_dynamic.yaml --reps 5 --no-display

  # inference延迟模式 / 换DUNE模型
  python test_neupan_delay_eval_v3.py --delay-mode inference --no-display
  python test_neupan_delay_eval_v3.py --planner planner_delay_rand_trained.yaml --no-display
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
                    choices=["observation", "inference"])
parser.add_argument("--min-delay-ms", type=int, default=100)
parser.add_argument("--max-delay-ms", type=int, default=1000)
parser.add_argument("--planner", type=str, default="planner_turn_simple.yaml")
parser.add_argument("--env", type=str, default="env_turn_simple.yaml")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--reps", type=int, default=5)
args = parser.parse_args()

STEP_TIME_MS = 100
dt = 0.1
SEEDS = [42, 1, 7, 100, 2026, 555, 888, 1234, 3407, 9099]
GOAL_XY = np.array([55.0, 20.0])
GOAL_TOL = 1.5          # 到达判定：距目标<1.5m
REF_Y = 20.0
FREE_FRONT_DIST = 5.0
PAST_OBSTACLE_X = 15.0
IS_STATIC_SCENE = "simple" in args.env  # 静态场景才报偏移


def predict_robot_state(delayed_state, action_history, dt_val):
    state = delayed_state.copy().flatten()
    for v, w in action_history:
        state[0] += v * np.cos(state[2]) * dt_val
        state[1] += v * np.sin(state[2]) * dt_val
        state[2] += w * dt_val
    return state.reshape(3, 1)


def predict_obstacle_points(delayed_points, point_velocities, delay_seconds):
    if point_velocities is None or delayed_points is None:
        return delayed_points
    if delayed_points.shape != point_velocities.shape:
        return delayed_points
    speeds = np.linalg.norm(point_velocities, axis=0)
    mask = (speeds > 0.15) & (speeds < 2.0)
    predicted = delayed_points.copy()
    predicted[:, mask] = delayed_points[:, mask] + point_velocities[:, mask] * delay_seconds
    return predicted


def front_min_range(lidar_scan):
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
        fm = np.abs(angles) < (np.pi / 4)
        vals = ranges[fm]
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
    random.seed(seed)
    np.random.seed(seed)
    delay_rng = np.random.default_rng(seed + 77777)

    env = irsim.make(args.env, save_ani=False, display=display)
    planner = neupan.init_from_yaml(args.planner)

    add_delay = mode_cfg["add_delay"]
    comp_self = mode_cfg["comp_self"]
    comp_obs = mode_cfg["comp_obs"]

    max_hist = int(args.max_delay_ms / STEP_TIME_MS) + 5
    state_hist = deque(maxlen=max_hist)
    lidar_hist = deque(maxlen=max_hist)
    action_hist = deque(maxlen=max_hist)

    last_action = np.array([[0.0], [0.0]])
    inf_counter = 0

    free_dev_samples = []
    w_series = []
    outcome, steps_used = "timeout", args.max_steps

    for i in range(args.max_steps):
        rs_now = env.get_robot_state()
        ls_now = env.get_lidar_scan()
        state_hist.append(rs_now.copy())
        lidar_hist.append(ls_now.copy() if hasattr(ls_now, "copy") else ls_now)

        # ==== 【v3修复】到达判定：直接看到目标的距离 ====
        robot_xy = np.array([float(rs_now[0, 0]), float(rs_now[1, 0])])
        if np.linalg.norm(robot_xy - GOAL_XY) < GOAL_TOL:
            outcome = "arrive"
            steps_used = i
            break

        # ==== 延迟采样 ====
        delay_ms = float(delay_rng.uniform(args.min_delay_ms, args.max_delay_ms)) if add_delay else 0.0
        d_steps = int(math.floor(delay_ms / STEP_TIME_MS))
        d_sec = delay_ms / 1000.0

        if args.delay_mode == "inference" and add_delay:
            if inf_counter <= 0:
                pts = planner.scan_to_point(rs_now, ls_now)
                action, info = planner(rs_now, pts, None)
                last_action = action.copy()
                inf_counter = d_steps
            else:
                action = last_action.copy()
                inf_counter -= 1
        else:
            if (not add_delay) or d_steps == 0 or len(state_hist) <= d_steps:
                pts = planner.scan_to_point(rs_now, ls_now)
                action, info = planner(rs_now, pts, None)
            else:
                idx = max(0, len(state_hist) - 1 - d_steps)
                s_delay, l_delay = state_hist[idx], lidar_hist[idx]
                s_in = (predict_robot_state(s_delay, list(action_hist)[-d_steps:], dt)
                        if comp_self and len(action_hist) >= d_steps else s_delay)
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

        v, w = float(action[0, 0]), float(action[1, 0])
        action_hist.append((v, w))
        w_series.append(w)

        # 直行段偏移（仅静态场景有意义）
        if IS_STATIC_SCENE:
            x_now, y_now = robot_xy[0], robot_xy[1]
            if x_now > PAST_OBSTACLE_X and front_min_range(ls_now) > FREE_FRONT_DIST:
                free_dev_samples.append(abs(y_now - REF_Y))

        env.step(np.array([[v], [w]]))
        env.render()

        if env.done():
            # 到达已在循环开头判定过；此处done只可能是碰撞
            outcome = "collision"
            steps_used = i + 1
            break

    env.end(0)

    w_arr = np.array(w_series)
    osc = int(np.sum(np.diff(np.sign(w_arr)) != 0)) if len(w_arr) > 1 else 0
    osc_per_100 = osc / max(steps_used, 1) * 100
    free_dev = float(np.mean(free_dev_samples)) if free_dev_samples else None

    return {
        "outcome": outcome,
        "steps": steps_used,
        "osc_per_100steps": round(osc_per_100, 1),
        "free_straight_dev": round(free_dev, 4) if free_dev is not None else None,
        "n_free_samples": len(free_dev_samples),
    }


# ================= 主流程 =================
all_results = {}
scene_type = "静态" if IS_STATIC_SCENE else "动态"
print(f"\n{'='*74}")
print(f"NeuPAN 延迟评估 v3 | {scene_type}场景({args.env}) | 模式={args.delay_mode} | "
      f"延迟={args.min_delay_ms}-{args.max_delay_ms}ms")
print(f"planner={args.planner}")
if IS_STATIC_SCENE:
    print(f"偏移指标 = 直行段(x>{PAST_OBSTACLE_X}, 前方>{FREE_FRONT_DIST}m)的|y-{REF_Y}|均值")
else:
    print(f"动态场景：只报SR/碰撞率/超时率（偏移指标不适用）")
print(f"{'='*74}")

for name, cfg in MODES.items():
    print(f"\n--- {name}: {cfg['desc']} ---")
    reps_out = []
    for rep in range(args.reps):
        seed = SEEDS[rep % len(SEEDS)]
        r = run_once(cfg, seed, display=not args.no_display)
        icon = {"arrive": "✅", "collision": "💥", "timeout": "⏰"}.get(r["outcome"], "?")
        line = (f"  seed={seed}: {icon} {r['outcome']} | steps={r['steps']} | "
                f"震荡/100步={r['osc_per_100steps']}")
        if IS_STATIC_SCENE:
            line += f" | 直行段偏移={r['free_straight_dev']} (n={r['n_free_samples']})"
        print(line)
        reps_out.append(r)

    n = len(reps_out)
    sr = sum(1 for r in reps_out if r["outcome"] == "arrive") / n * 100
    col = sum(1 for r in reps_out if r["outcome"] == "collision") / n * 100
    to = sum(1 for r in reps_out if r["outcome"] == "timeout") / n * 100
    devs = [r["free_straight_dev"] for r in reps_out if r["free_straight_dev"] is not None]
    all_results[name] = {
        "desc": cfg["desc"],
        "SR%": round(sr, 1),
        "collision%": round(col, 1),
        "timeout%": round(to, 1),
        "avg_steps": round(float(np.mean([r["steps"] for r in reps_out])), 1),
        "avg_osc_per_100": round(float(np.mean([r["osc_per_100steps"] for r in reps_out])), 1),
        "avg_free_dev": round(float(np.mean(devs)), 4) if devs else None,
        "details": reps_out,
    }

print(f"\n{'='*84}")
print(f"汇总 | {scene_type}场景 | delay-mode={args.delay_mode} | "
      f"{args.min_delay_ms}-{args.max_delay_ms}ms | {args.planner}")
print(f"{'='*84}")
hdr = f"{'实验':<30} {'SR%':<6} {'碰撞%':<7} {'超时%':<7} {'步数':<8} {'震荡/100步':<10}"
if IS_STATIC_SCENE:
    hdr += f" {'直行段偏移':<10}"
print(hdr)
print("-" * 82)
for name, r in all_results.items():
    line = (f"{r['desc']:<30} {r['SR%']:<6} {r['collision%']:<7} {r['timeout%']:<7} "
            f"{r['avg_steps']:<8} {r['avg_osc_per_100']:<10}")
    if IS_STATIC_SCENE:
        line += f" {str(r['avg_free_dev']):<10}"
    print(line)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
tag = "static" if IS_STATIC_SCENE else "dynamic"
out = (f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/"
       f"neupan_eval_v3_{tag}_{args.delay_mode}_{ts}.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out}")
