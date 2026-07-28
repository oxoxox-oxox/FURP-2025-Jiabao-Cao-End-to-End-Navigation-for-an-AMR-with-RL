"""
STPS 参数敏感性分析
STALL_WINDOW × STALL_DIST 的 3×3 网格
每组只跑 U-trap + Narrow_door（最敏感的两个场景），各4 configs × 3 reps
"""
import sys, os, numpy as np, torch, json, itertools, time
from collections import deque
sys.path.insert(0, 'robot_nav')
os.chdir('/root/DRL-robot-navigation-IR-SIM')

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')

# 固定参数
ESCAPE_STEPS = 60
PROGRESS_DIST = 0.5

# 扫描网格
WINDOWS = [15, 20, 30]
DISTS = [0.10, 0.15, 0.20]

SCENARIOS = {
    'S1_U_trap': {
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'cases': [
            ([[7.5],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
        ], 'max_steps': 500,
    },
    'S3_Narrow_door': {
        'world': 'robot_nav/worlds/narrow_door_world.yaml',
        'cases': [
            ([[2.0],[5.0],[0.0]],  [[8.0],[5.0],[0]]),
            ([[2.0],[5.0],[0.3]],  [[8.0],[5.0],[0]]),
            ([[2.0],[5.0],[-0.3]], [[8.0],[5.0],[0]]),
            ([[2.0],[6.0],[0.0]],  [[8.0],[5.0],[0]]),
        ], 'max_steps': 500,
    },
}


def run_episode(model_main, model_escape, world, robot_state, robot_goal,
                max_steps, stall_window, stall_dist):
    sim = SIM(world_file=world, disable_plotting=True)
    scan,dist,cos,sin,col,goal,a,r = sim.reset(
        robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)

    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=stall_window)
    mode = 'main'
    escape_counter = 0
    escape_start_pos = None
    n_switches = 0

    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        curr_pos = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(curr_pos)

        if mode == 'main':
            if len(pos_hist) == stall_window:
                moved = np.linalg.norm(pos_hist[-1] - pos_hist[0])
                if moved < stall_dist:
                    mode = 'escape'
                    escape_counter = 0
                    escape_start_pos = curr_pos.copy()
                    n_switches += 1
                    pos_hist.clear()
        else:
            escape_counter += 1
            escaped = np.linalg.norm(curr_pos - escape_start_pos) > PROGRESS_DIST
            if escape_counter >= ESCAPE_STEPS and escaped:
                mode = 'main'
                pos_hist.clear()

        model = model_main if mode == 'main' else model_escape
        state,_ = model.prepare_state(scan,dist,cos,sin,col,goal,prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,dist,cos,sin,col,goal,a,r = sim.step(lin, ang)

        if goal:
            sim.env.end()
            return 'goal', n_switches
        if col:
            sim.env.end()
            return 'collision', n_switches
    sim.env.end()
    return 'timeout', n_switches


# 加载模型
ckpt_dir = 'models/CNNTD3/checkpoint'
model_main = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                    device=device, load_model=False, model_name="sens_main")
model_main.load("CNNTD3_v7_finetune_best", ckpt_dir)
model_main.actor.eval()
model_escape = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                      device=device, load_model=False, model_name="sens_escape")
model_escape.load("CNNTD3_improved", ckpt_dir)
model_escape.actor.eval()
print("✅ Models loaded\n")

results = {}
t_start = time.time()

for w, d in itertools.product(WINDOWS, DISTS):
    key = f"w{w}_d{d}"
    results[key] = {}
    print(f"--- STALL_WINDOW={w}, STALL_DIST={d} ---")
    for scene_name, cfg in SCENARIOS.items():
        total, success = 0, 0
        for robot_state, robot_goal in cfg['cases']:
            for rep in range(3):
                outcome, ns = run_episode(model_main, model_escape,
                                          cfg['world'], robot_state, robot_goal,
                                          cfg['max_steps'], w, d)
                if outcome == 'goal': success += 1
                total += 1
        sr = success / total
        results[key][scene_name] = sr
        print(f"  {scene_name}: {success}/{total} = {sr:.0%}")
    elapsed = (time.time() - t_start) / 60
    print(f"  (elapsed {elapsed:.0f} min)\n")

# 汇总表
print(f"\n{'='*70}")
print(f"敏感性分析结果 (U-trap SR / Narrow-door SR)")
print(f"{'='*70}")
print(f"{'':>12}", end="")
for d in DISTS:
    print(f"  dist={d:<8}", end="")
print()
for w in WINDOWS:
    print(f"window={w:<5}", end="")
    for d in DISTS:
        key = f"w{w}_d{d}"
        u = results[key]['S1_U_trap']
        n = results[key]['S3_Narrow_door']
        print(f"  {u:.0%}/{n:.0%}    ", end="")
    print()
print(f"{'='*70}")

with open('stps_sensitivity_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n💾 保存到 stps_sensitivity_results.json")
