"""
NeuPAN 延迟评估 v4（卡尔曼修复版）

相比v3卡尔曼版的关键修复：
  1. 【核心】追踪器活在"延迟时间线"上：每步用 t-d 时刻的延迟点云更新追踪器，
     需要时 predict_future(d_steps) 推到当前时刻。
     （v3的追踪器只在 d_steps==0 时更新，延迟下永远没更新过，卡尔曼从未生效）
  2. 【核心】先聚类再追踪：延迟点云先做欧氏聚类，只追踪"动态簇"的中心
     （簇中心帧间位移>阈值才算动态），不再对100个原始LiDAR点逐点建追踪器
  3. 【核心】返回合并点云：静态点原样保留 + 动态簇的点整体平移到预测位置——不丢墙
  4. slip_factor 恢复 1.0（仿真无打滑，0.95会引入系统性低估）
  5. comp_success 只在真正执行了预测平移时计数（不再把回退算成成功）

用法:
  cd ~/NeuPAN/example && conda activate neupan
  python test_neupan_delay_eval_v4.py --env env_turn_fast_dynamic.yaml --reps 10 --no-display
  python test_neupan_delay_eval_v4.py --env env_turn_simple.yaml --reps 5 --no-display
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
parser.add_argument("--reps", type=int, default=10)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

STEP_TIME_MS = 100
dt = 0.1
SEEDS = [42, 1, 7, 100, 2026, 555, 888, 1234, 3407, 9099]
GOAL_XY = np.array([55.0, 20.0])
GOAL_TOL = 1.5
REF_Y = 20.0
FREE_FRONT_DIST = 5.0
PAST_OBSTACLE_X = 15.0
IS_STATIC_SCENE = "simple" in args.env

CLUSTER_EPS = 0.8       # 欧氏聚类：点间距<0.8m 归为同簇
DYNAMIC_SPEED_MIN = 0.3 # 簇中心速度>0.3m/s 才算动态簇
DYNAMIC_SPEED_MAX = 3.0 # 速度上限（滤异常）
TRACK_ASSOC_DIST = 1.5  # 追踪器-簇关联最大距离
TRACK_MIN_AGE = 3       # 追踪器至少更新3次才可信


# ============ 卡尔曼追踪器（追踪聚类中心，活在延迟时间线上） ============
class ClusterTracker:
    """恒速模型卡尔曼滤波器，状态 [x, y, vx, vy]"""
    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.diag([0.5, 0.5, 2.0, 2.0])
        q_pos, q_vel = 0.02, 0.5
        self.Q = np.diag([q_pos**2, q_pos**2, q_vel**2, q_vel**2])
        self.R = np.diag([0.15**2, 0.15**2])
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.age = 0
        self.missed = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, cx, cy):
        z = np.array([cx, cy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.age += 1
        self.missed = 0

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()

    def future_offset(self, future_steps):
        """未来future_steps步后的位移（相对当前追踪位置）"""
        return self.vel * dt * future_steps


def euclidean_cluster(points_2xn, eps=CLUSTER_EPS):
    """简单欧氏聚类。points: (2, N) → list of (member_indices, center)"""
    if points_2xn is None or points_2xn.shape[1] == 0:
        return []
    pts = points_2xn.T  # (N, 2)
    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        members = []
        visited[i] = True
        while stack:
            j = stack.pop()
            members.append(j)
            d = np.linalg.norm(pts - pts[j], axis=1)
            neigh = np.where((d < eps) & (~visited))[0]
            for k in neigh:
                visited[k] = True
                stack.append(int(k))
        members = np.array(members)
        center = pts[members].mean(axis=0)
        clusters.append((members, center))
    return clusters


def kalman_compensate(delayed_points, trackers, d_steps):
    """
    输入: delayed_points (2,N) — t-d时刻的点云（追踪器也活在t-d时间线）
    流程: 聚类 → 关联/更新追踪器 → 动态簇的点整体平移 future_offset → 与静态点合并
    返回: (compensated_points (2,N), trackers, did_predict:bool)
    """
    if delayed_points is None or delayed_points.shape[1] == 0:
        return delayed_points, trackers, False

    clusters = euclidean_cluster(delayed_points)

    # 1) 所有追踪器先predict一步（时间线前进dt）
    for t in trackers:
        t.predict()

    # 2) 簇中心与追踪器最近邻关联
    used_clusters = set()
    for t in trackers:
        best, best_d = None, TRACK_ASSOC_DIST
        for ci, (_, center) in enumerate(clusters):
            if ci in used_clusters:
                continue
            d = np.linalg.norm(center - t.pos)
            if d < best_d:
                best, best_d = ci, d
        if best is not None:
            _, c = clusters[best]
            t.update(c[0], c[1])
            used_clusters.add(best)
        else:
            t.missed += 1

    # 3) 未匹配的簇 → 新建追踪器
    for ci, (_, center) in enumerate(clusters):
        if ci not in used_clusters:
            trackers.append(ClusterTracker(center[0], center[1]))

    # 4) 清理长期丢失的追踪器
    trackers = [t for t in trackers if t.missed < 10]

    # 5) 构建补偿点云：动态簇整体平移，静态点原样
    comp = delayed_points.copy()
    did_predict = False
    for t in trackers:
        if t.age < TRACK_MIN_AGE:
            continue
        speed = np.linalg.norm(t.vel)
        if not (DYNAMIC_SPEED_MIN < speed < DYNAMIC_SPEED_MAX):
            continue
        # 找到该追踪器对应的簇成员点，整体平移
        for members, center in clusters:
            if np.linalg.norm(center - t.pos) < TRACK_ASSOC_DIST:
                offset = t.future_offset(d_steps)  # (2,)
                comp[:, members] = delayed_points[:, members] + offset.reshape(2, 1)
                did_predict = True
                break

    return comp, trackers, did_predict


def predict_robot_state(delayed_state, action_history, dt_val):
    """自身状态运动学推演（slip_factor=1.0，仿真无打滑）"""
    state = delayed_state.copy().flatten()
    for v, w in action_history:
        state[0] += v * np.cos(state[2]) * dt_val
        state[1] += v * np.sin(state[2]) * dt_val
        state[2] += w * dt_val
    return state.reshape(3, 1)


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
                    "desc": "有延迟 + 自身补偿"},
    "D_kalman":    {"add_delay": True,  "comp_self": True,  "comp_obs": True,
                    "desc": "有延迟 + 自身补偿 + 卡尔曼(v4)"},
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

    trackers = []          # 卡尔曼追踪器（活在延迟时间线）
    last_tracked_step = -1 # 防止同一延迟帧重复更新追踪器

    last_action = np.array([[0.0], [0.0]])
    inf_counter = 0

    free_dev_samples = []
    w_series = []
    outcome, steps_used = "timeout", args.max_steps
    n_predicted, n_fallback = 0, 0

    for i in range(args.max_steps):
        rs_now = env.get_robot_state()
        ls_now = env.get_lidar_scan()
        state_hist.append(rs_now.copy())
        lidar_hist.append(ls_now.copy() if hasattr(ls_now, "copy") else ls_now)

        robot_xy = np.array([float(rs_now[0, 0]), float(rs_now[1, 0])])
        if np.linalg.norm(robot_xy - GOAL_XY) < GOAL_TOL:
            outcome = "arrive"
            steps_used = i
            break

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
                        # 延迟点云（世界坐标，用延迟状态解算——延迟时间线内自洽）
                        p_delay = planner.scan_to_point(s_delay, l_delay)

                        # 追踪器更新：每个仿真步只更新一次（延迟时间线前进一格）
                        if i != last_tracked_step:
                            p_comp, trackers, did = kalman_compensate(
                                p_delay, trackers, d_steps)
                            last_tracked_step = i
                        else:
                            p_comp, did = p_delay, False

                        if did:
                            n_predicted += 1
                        else:
                            n_fallback += 1

                        action, info = planner(s_in, p_comp, None)
                    except Exception as e:
                        n_fallback += 1
                        pts = planner.scan_to_point(s_in, l_delay)
                        action, info = planner(s_in, pts, None)
                        if args.debug:
                            print(f"[DEBUG] step {i} kalman error: {e}")
                else:
                    pts = planner.scan_to_point(s_in, l_delay)
                    action, info = planner(s_in, pts, None)

        v, w = float(action[0, 0]), float(action[1, 0])
        action_hist.append((v, w))
        w_series.append(w)

        if IS_STATIC_SCENE:
            x_now, y_now = robot_xy[0], robot_xy[1]
            if x_now > PAST_OBSTACLE_X and front_min_range(ls_now) > FREE_FRONT_DIST:
                free_dev_samples.append(abs(y_now - REF_Y))

        env.step(np.array([[v], [w]]))
        env.render()

        if env.done():
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
        "kalman_predicted": n_predicted,
        "kalman_fallback": n_fallback,
    }


# ================= 主流程 =================
all_results = {}
scene_type = "静态" if IS_STATIC_SCENE else "动态"
print(f"\n{'='*80}")
print(f"NeuPAN 延迟评估 v4（卡尔曼修复版）| {scene_type}场景({args.env})")
print(f"模式={args.delay_mode} | 延迟={args.min_delay_ms}-{args.max_delay_ms}ms | planner={args.planner}")
print(f"{'='*80}")

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
            line += f" | 直行段偏移={r['free_straight_dev']}"
        if cfg["comp_obs"]:
            tot = r["kalman_predicted"] + r["kalman_fallback"]
            line += f" | 卡尔曼真实预测={r['kalman_predicted']}/{tot}"
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

print(f"\n{'='*88}")
print(f"汇总 | {scene_type}场景 | delay-mode={args.delay_mode} | "
      f"{args.min_delay_ms}-{args.max_delay_ms}ms | {args.planner}")
print(f"{'='*88}")
hdr = f"{'实验':<32} {'SR%':<6} {'碰撞%':<7} {'超时%':<7} {'步数':<8} {'震荡/100步':<10}"
if IS_STATIC_SCENE:
    hdr += f" {'直行段偏移':<10}"
print(hdr)
print("-" * 86)
for name, r in all_results.items():
    line = (f"{r['desc']:<32} {r['SR%']:<6} {r['collision%']:<7} {r['timeout%']:<7} "
            f"{r['avg_steps']:<8} {r['avg_osc_per_100']:<10}")
    if IS_STATIC_SCENE:
        line += f" {str(r['avg_free_dev']):<10}"
    print(line)

# 逐seed配对比较（C vs D）
print(f"\n--- 逐seed配对: C(自身补偿) vs D(卡尔曼) ---")
c_det = all_results["C_self_only"]["details"]
d_det = all_results["D_kalman"]["details"]
for rep in range(min(len(c_det), len(d_det))):
    seed = SEEDS[rep % len(SEEDS)]
    co, do_ = c_det[rep]["outcome"], d_det[rep]["outcome"]
    mark = "→ D赢" if (do_ == "arrive" and co != "arrive") else (
           "→ C赢" if (co == "arrive" and do_ != "arrive") else "")
    print(f"  seed={seed}: C={co:<10} D={do_:<10} {mark}")

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
tag = "static" if IS_STATIC_SCENE else "dynamic"
out = (f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/"
       f"neupan_eval_v4_{tag}_{args.delay_mode}_{ts}.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {out}")
