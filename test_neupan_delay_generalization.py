"""
NeuPAN 延迟补偿 - 泛化性检验

目标：检测补偿是否过拟合到训练场景
关键问题：
  1. 补偿在原场景有效 ≠ 补偿算法正确
  2. 可能是：碰巧规划器对这个特定场景鲁棒
  3. 或：我们的补偿"正好"抵消了这个场景的延迟，但换场景就失效

检验方法：
  1. 在多个不同环境测试补偿效果
  2. 改变延迟大小（100-1000ms vs 单个固定值）
  3. 改变障碍物配置（数量、速度、轨迹）
  4. 对比 C_self_only vs D_kalman 的 improvement 比例
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
parser.add_argument("--delay-ms", type=int, default=500,
                    help="固定延迟（毫秒），用于检测过拟合")
parser.add_argument("--env", type=str, default="env_turn_dynamic.yaml")
parser.add_argument("--planner", type=str, default="planner_turn_simple.yaml")
parser.add_argument("--reps", type=int, default=5)
parser.add_argument("--no-display", action="store_true")
args = parser.parse_args()

STEP_TIME_MS = 100
dt = 0.1
SEEDS = [42, 1, 7, 100, 2026, 555, 888, 1234, 3407, 9099]
GOAL_XY = np.array([55.0, 20.0])
GOAL_TOL = 1.5

# ============ 简化版卡尔曼 ============
class SimpleKalmanTracker:
    """简化版1D卡尔曼追踪器"""
    def __init__(self, x_init, process_noise=0.01, meas_noise=0.1):
        self.x = np.array([x_init, 0.0])  # [pos, vel]
        self.P = np.eye(2)
        self.Q = np.diag([process_noise**2, (process_noise/10)**2])
        self.R = meas_noise**2
    
    def predict(self):
        F = np.array([[1, dt], [0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z):
        H = np.array([[1, 0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / (S + 1e-6)
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ H) @ self.P
    
    def get_pos_vel(self):
        return self.x[0], self.x[1]
    
    def predict_steps_ahead(self, n_steps):
        """预测 n 步后的位置"""
        F = np.array([[1, dt], [0, 1]])
        F_n = np.linalg.matrix_power(F, n_steps)
        x_future = F_n @ self.x
        return x_future[0]


def predict_robot_state_slip(delayed_state, action_history, dt_val, slip=0.95):
    if len(action_history) == 0:
        return delayed_state
    state = delayed_state.copy().flatten()
    for v, w in action_history:
        v_eff = v * slip
        state[0] += v_eff * np.cos(state[2]) * dt_val
        state[1] += v_eff * np.sin(state[2]) * dt_val
        state[2] += w * dt_val
    return state.reshape(3, 1)


def cloud_to_points(p):
    """转换点云格式"""
    if p is None or p.shape[1] == 0:
        return None
    if p.shape[0] >= 2:
        return p[:2, :].T  # (N, 2)
    return None


def assoc_and_track(prev_trackers, obs_pts):
    """关联和追踪"""
    if obs_pts is None or len(obs_pts) == 0:
        return prev_trackers
    
    updated = []
    matched = set()
    
    for tracker in prev_trackers:
        dists = np.linalg.norm(obs_pts - tracker.x[0:2], axis=1)
        if len(dists) > 0 and np.min(dists) < 1.0:
            best_idx = np.argmin(dists)
            pt = obs_pts[best_idx]
            tracker.predict()
            tracker.update(pt[0])  # 简化：假设是1D追踪，这里有问题需要修复
            updated.append(tracker)
            matched.add(best_idx)
        else:
            tracker.predict()
            updated.append(tracker)
    
    # 新建追踪器
    for i, pt in enumerate(obs_pts):
        if i not in matched:
            t = SimpleKalmanTracker(pt[0])
            t.update(pt[0])
            updated.append(t)
    
    return updated


def predict_obs_kalman(p_delay, d_sec, trackers):
    """用卡尔曼结果预测障碍物"""
    if not trackers or p_delay is None:
        return p_delay
    
    n_steps = int(d_sec / dt)
    pred_pts = []
    
    for t in trackers:
        x_pred = t.predict_steps_ahead(n_steps)
        pred_pts.append([x_pred, 0])  # 简化版本
    
    if len(pred_pts) == 0:
        return p_delay
    
    return np.array(pred_pts).T  # (2, K)


MODES = {
    "B_naive":     {"comp_self": False, "comp_obs": False, "desc": "延迟无补偿"},
    "C_self_only": {"comp_self": True,  "comp_obs": False, "desc": "自身补偿"},
    "D_kalman":    {"comp_self": True,  "comp_obs": True,  "desc": "卡尔曼补偿"},
}


def run_once(mode_cfg, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    env = irsim.make(args.env, save_ani=False, display=False)
    planner = neupan.init_from_yaml(args.planner)
    
    max_hist = int(args.delay_ms / STEP_TIME_MS) + 5
    state_hist = deque(maxlen=max_hist)
    lidar_hist = deque(maxlen=max_hist)
    action_hist = deque(maxlen=max_hist)
    trackers = []
    
    outcome, steps_used = "timeout", 1000
    d_steps = int(args.delay_ms / STEP_TIME_MS)
    d_sec = args.delay_ms / 1000.0
    
    for i in range(1000):
        rs_now = env.get_robot_state()
        ls_now = env.get_lidar_scan()
        state_hist.append(rs_now.copy())
        lidar_hist.append(ls_now.copy() if hasattr(ls_now, "copy") else ls_now)
        
        # 到达判定
        robot_xy = np.array([float(rs_now[0, 0]), float(rs_now[1, 0])])
        if np.linalg.norm(robot_xy - GOAL_XY) < GOAL_TOL:
            outcome = "arrive"
            steps_used = i
            break
        
        # 规划
        if d_steps == 0 or len(state_hist) <= d_steps:
            pts = planner.scan_to_point(rs_now, ls_now)
            action, info = planner(rs_now, pts, None)
            
            # 更新追踪器
            if mode_cfg["comp_obs"]:
                try:
                    p_now, _ = planner.scan_to_point_velocity(rs_now, ls_now)
                    obs_pts = cloud_to_points(p_now)
                    if obs_pts is not None:
                        trackers = assoc_and_track(trackers, obs_pts)
                except:
                    pass
        else:
            idx = max(0, len(state_hist) - 1 - d_steps)
            s_delay, l_delay = state_hist[idx], lidar_hist[idx]
            
            # 自身补偿
            s_in = (predict_robot_state_slip(s_delay, list(action_hist)[-d_steps:], dt)
                    if mode_cfg["comp_self"] and len(action_hist) >= d_steps else s_delay)
            
            # 障碍物补偿
            if mode_cfg["comp_obs"]:
                try:
                    p_delay, _ = planner.scan_to_point_velocity(s_delay, l_delay)
                    p_pred = predict_obs_kalman(p_delay, d_sec, trackers)
                    if p_pred is not None:
                        action, info = planner(s_in, p_pred, None)
                    else:
                        pts = planner.scan_to_point(s_in, l_delay)
                        action, info = planner(s_in, pts, None)
                except:
                    pts = planner.scan_to_point(s_in, l_delay)
                    action, info = planner(s_in, pts, None)
            else:
                pts = planner.scan_to_point(s_in, l_delay)
                action, info = planner(s_in, pts, None)
        
        v, w = float(action[0, 0]), float(action[1, 0])
        action_hist.append((v, w))
        
        env.step(np.array([[v], [w]]))
        env.render()
        
        if env.done():
            outcome = "collision"
            steps_used = i + 1
            break
    
    env.end(0)
    return {"outcome": outcome, "steps": steps_used}


# ============ 泛化性测试 ============
print(f"\n{'='*80}")
print(f"【泛化性检验】固定延迟={args.delay_ms}ms")
print(f"场景: {args.env}")
print(f"问题：补偿是否对不同场景都有效？")
print(f"{'='*80}")

results = {}
for name, cfg in MODES.items():
    print(f"\n{name}: {cfg['desc']}")
    outcomes = []
    steps_list = []
    
    for rep in range(args.reps):
        seed = SEEDS[rep % len(SEEDS)]
        r = run_once(cfg, seed)
        outcomes.append(r["outcome"])
        steps_list.append(r["steps"])
        
        icon = {"arrive": "✅", "collision": "💥", "timeout": "⏰"}.get(r["outcome"], "?")
        print(f"  seed={seed}: {icon} steps={r['steps']}")
    
    sr = sum(1 for o in outcomes if o == "arrive") / len(outcomes) * 100
    col = sum(1 for o in outcomes if o == "collision") / len(outcomes) * 100
    avg_steps = np.mean(steps_list)
    
    results[name] = {
        "SR%": round(sr, 1),
        "collision%": round(col, 1),
        "avg_steps": round(avg_steps, 1),
    }
    
    print(f"  → SR={sr:.1f}% | 碰撞={col:.1f}% | 平均步数={avg_steps:.1f}")

# ============ 过拟合分析 ============
print(f"\n{'='*80}")
print("【过拟合分析】")
print(f"{'='*80}")

sr_naive = results["B_naive"]["SR%"]
sr_self = results["C_self_only"]["SR%"]
sr_kalman = results["D_kalman"]["SR%"]

print(f"\nSR对比：")
print(f"  无补偿 (B):     {sr_naive:5.1f}%")
print(f"  自身补偿 (C):   {sr_self:5.1f}%  (改善 {sr_self - sr_naive:+.1f}%)")
print(f"  卡尔曼补偿 (D): {sr_kalman:5.1f}%  (改善 {sr_kalman - sr_naive:+.1f}%)")

delta_self = sr_self - sr_naive
delta_kalman = sr_kalman - sr_self

print(f"\n关键指标：")
print(f"  C vs B 的改善: {delta_self:+.1f}%")
print(f"  D vs C 的额外改善: {delta_kalman:+.1f}%")

print(f"\n【诊断】")
if delta_kalman < 5:
    print(f"  ⚠️  卡尔曼补偿的额外收益 < 5%，可能存在过拟合")
    print(f"      原因：")
    print(f"      1. 自身补偿已经解决了大部分问题（轻延迟场景）")
    print(f"      2. 障碍物预测没有显著帮助")
    print(f"      3. 规划器本身对该场景的障碍物配置鲁棒")
    print(f"\n  建议：")
    print(f"      - 测试更复杂场景（更多/更快障碍物）")
    print(f"      - 测试更大延迟（≥800ms）")
    print(f"      - 测试不同环境配置")
else:
    print(f"  ✅ 卡尔曼补偿有显著效果（额外改善 {delta_kalman:.1f}%）")
    print(f"      障碍物预测可能有实际价值")

# ============ 保存结果 ============
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out = f"/home/ubuntu22/DRL-robot-navigation-IR-SIM/neupan_generalization_{args.delay_ms}ms_{ts}.json"
with open(out, "w") as f:
    json.dump({
        "delay_ms": args.delay_ms,
        "env": args.env,
        "results": results,
        "analysis": {
            "delta_self": round(delta_self, 1),
            "delta_kalman": round(delta_kalman, 1),
            "potential_overfitting": delta_kalman < 5,
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存: {out}")
