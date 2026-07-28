"""
STPS U-trap 诊断：
1. improved单独跑12 configs确认上限
2. ESCAPE_STEPS × PROGRESS_DIST 3×3 网格扫描
3. 最佳组合的verbose输出（看切换时机）
"""
import sys, os, numpy as np, torch, json
from collections import deque
sys.path.insert(0, 'robot_nav')
os.chdir('/root/DRL-robot-navigation-IR-SIM')

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')
rng = np.random.default_rng(42)

STALL_WINDOW = 20
STALL_DIST = 0.15

UTRAP_WORLD = 'robot_nav/worlds/u_trap_world.yaml'
UTRAP_GOAL = [[9.0],[5.0],[0]]
MAX_STEPS = 500

def make_configs():
    cfgs = []
    base_thetas = [0.0, 1.57, 3.14, -1.57]
    for i in range(12):
        th = base_thetas[i % 4] + rng.uniform(-0.4, 0.4)
        x = 7.5 + rng.uniform(-0.3, 0.3)
        y = 5.0 + rng.uniform(-0.3, 0.3)
        cfgs.append(([[x],[y],[th]], i))
    return cfgs

CONFIGS = make_configs()


def run_stps(m_main, m_esc, cfg, escape_steps, progress_dist, verbose=False):
    robot_state = cfg
    sim = SIM(world_file=UTRAP_WORLD, disable_plotting=True)
    scan,dist,cos,sin,col,goal,a,r = sim.reset(
        robot_state=robot_state, robot_goal=UTRAP_GOAL, random_obstacles=False)
    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=STALL_WINDOW)
    mode = 'main'
    esc_cnt = 0
    esc_start = None
    switches = 0
    switch_log = []

    for step in range(MAX_STEPS):
        rs = sim.env.get_robot_state()
        curr_pos = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(curr_pos)

        if mode == 'main':
            if len(pos_hist) == STALL_WINDOW:
                moved = np.linalg.norm(pos_hist[-1] - pos_hist[0])
                if moved < STALL_DIST:
                    mode = 'escape'
                    esc_cnt = 0
                    esc_start = curr_pos.copy()
                    switches += 1
                    pos_hist.clear()
                    if verbose:
                        switch_log.append(f"  step {step}: STALL→escape at ({curr_pos[0]:.1f},{curr_pos[1]:.1f})")
        else:
            esc_cnt += 1
            dist_from_start = np.linalg.norm(curr_pos - esc_start)
            if esc_cnt >= escape_steps and dist_from_start > progress_dist:
                if verbose:
                    switch_log.append(f"  step {step}: escaped (moved {dist_from_start:.2f}m in {esc_cnt} steps)→main")
                mode = 'main'
                pos_hist.clear()
            elif esc_cnt >= escape_steps * 3:
                # 超长逃脱仍未脱困，强制切回避免死循环
                if verbose:
                    switch_log.append(f"  step {step}: escape TIMEOUT (moved {dist_from_start:.2f}m)→main")
                mode = 'main'
                pos_hist.clear()

        model = m_main if mode == 'main' else m_esc
        state,_ = model.prepare_state(scan,dist,cos,sin,col,goal,prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,dist,cos,sin,col,goal,a,r = sim.step(lin, ang)

        if goal:
            sim.env.end()
            return 'goal', switches, switch_log
        if col:
            sim.env.end()
            return 'collision', switches, switch_log
    sim.env.end()
    return 'timeout', switches, switch_log


def run_single(model, cfg):
    sim = SIM(world_file=UTRAP_WORLD, disable_plotting=True)
    scan,dist,cos,sin,col,goal,a,r = sim.reset(
        robot_state=cfg, robot_goal=UTRAP_GOAL, random_obstacles=False)
    prev = [0.0, 0.0]
    for step in range(MAX_STEPS):
        state,_ = model.prepare_state(scan,dist,cos,sin,col,goal,prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,dist,cos,sin,col,goal,a,r = sim.step(lin, ang)
        if goal: sim.env.end(); return 'goal'
        if col: sim.env.end(); return 'collision'
    sim.env.end()
    return 'timeout'


# 加载模型
ckpt = 'models/CNNTD3/checkpoint'
m_v7 = CNNTD3(state_dim=185, action_dim=2, max_action=1,
              device=device, load_model=False, model_name="d_v7")
m_v7.load("CNNTD3_v7_finetune_best", ckpt); m_v7.actor.eval()
m_imp = CNNTD3(state_dim=185, action_dim=2, max_action=1,
               device=device, load_model=False, model_name="d_imp")
m_imp.load("CNNTD3_improved", ckpt); m_imp.actor.eval()
print("✅ Models loaded\n")

# === Part 1: improved 单独跑（确认上限）===
print("="*50)
print("improved 单独跑 U-trap (上限参考)")
print("="*50)
imp_succ = 0
for cfg, idx in CONFIGS:
    out = run_single(m_imp, cfg)
    status = "✅" if out == 'goal' else "❌"
    print(f"  config {idx}: {out} {status}")
    if out == 'goal': imp_succ += 1
print(f"  improved U-trap SR = {imp_succ}/12 = {imp_succ/12:.0%}\n")

# === Part 2: 参数网格扫描 ===
ESC_STEPS = [60, 120, 180]
PROG_DISTS = [0.5, 1.0, 1.5]

print("="*50)
print("STPS 参数网格扫描 (U-trap only)")
print("="*50)
grid = {}
for es in ESC_STEPS:
    for pd in PROG_DISTS:
        key = f"es{es}_pd{pd}"
        succ = 0
        for cfg, idx in CONFIGS:
            out, ns, _ = run_stps(m_v7, m_imp, cfg, es, pd)
            if out == 'goal': succ += 1
        sr = succ / 12
        grid[key] = sr
        print(f"  ESCAPE_STEPS={es:>3}, PROGRESS_DIST={pd:.1f}: {succ}/12 = {sr:.0%}")

print(f"\n{'':>15}", end="")
for pd in PROG_DISTS:
    print(f"  pd={pd:.1f}", end="")
print()
for es in ESC_STEPS:
    print(f"  es={es:<6}", end="")
    for pd in PROG_DISTS:
        sr = grid[f"es{es}_pd{pd}"]
        print(f"  {sr:.0%}   ", end="")
    print()

# === Part 3: 最佳组合verbose ===
best_key = max(grid, key=grid.get)
best_es = int(best_key.split('_')[0][2:])
best_pd = float(best_key.split('_')[1][2:])
print(f"\n{'='*50}")
print(f"最佳组合 verbose: ESCAPE_STEPS={best_es}, PROGRESS_DIST={best_pd}")
print(f"{'='*50}")
for cfg, idx in CONFIGS:
    out, ns, log = run_stps(m_v7, m_imp, cfg, best_es, best_pd, verbose=True)
    status = "✅" if out == 'goal' else "❌"
    print(f"config {idx}: {out} {status} (switches={ns})")
    for l in log:
        print(l)

with open('stps_utrap_diagnosis.json', 'w') as f:
    json.dump({'improved_baseline': imp_succ/12, 'grid': grid,
               'best': {'escape_steps': best_es, 'progress_dist': best_pd}}, f, indent=2)
print("\n💾 stps_utrap_diagnosis.json")
