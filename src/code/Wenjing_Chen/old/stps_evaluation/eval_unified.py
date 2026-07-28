"""
统一公平对比：CNNTD3 baseline vs NeuPAN vs STPS v2
相同场景、相同起点、3个seed、报告均值±标准差
"""
import sys, os, numpy as np, torch, json, time
from collections import deque

BASE_DIR = os.path.expanduser('~/DRL-robot-navigation-IR-SIM')
sys.path.insert(0, os.path.join(BASE_DIR, 'robot_nav'))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)
os.environ.pop("DISPLAY", None)

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from neupan.neupan import neupan

device = torch.device('cpu')

# STPS v2 参数
STALL_WINDOW = 20; STALL_DIST = 0.15
BASE_ESCAPE_STEPS = 120; PROGRESS_DIST = 0.5
OSC_WINDOW = 12; OSC_REVERSAL_THRESH = 5; OSC_MIN_STEPS = 8

SEEDS = [42, 123, 2026]

SCENARIOS = {
    'S1_U_trap': {
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'base_xy': [7.5, 5.0], 'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
    },
    'S2_Double_U': {
        'world': 'robot_nav/worlds/double_u_world.yaml',
        'base_xy': [5.0, 5.0], 'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
    },
    'S3_Narrow_door': {
        'world': 'robot_nav/worlds/narrow_door_world.yaml',
        'base_xy': [2.0, 5.0], 'goal': [[8.0],[5.0],[0]], 'max_steps': 500,
    },
    'S5_Corridor': {
        'world': 'robot_nav/worlds/symmetric_corridor_world.yaml',
        'base_xy': [1.0, 5.0], 'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
    },
}

PLANNER_YAML = os.path.expanduser("~/NeuPAN/example/standard_eval/diff/planner.yaml")


def make_configs(base_xy, seed, n=12):
    rng = np.random.default_rng(seed)
    cfgs = []
    thetas = [0.0, 1.57, 3.14, -1.57]
    for i in range(n):
        th = thetas[i % 4] + rng.uniform(-0.4, 0.4)
        x = base_xy[0] + rng.uniform(-0.3, 0.3)
        y = base_xy[1] + rng.uniform(-0.3, 0.3)
        cfgs.append([[x],[y],[th]])
    return cfgs


def detect_oscillation(pos_history):
    if len(pos_history) < OSC_WINDOW:
        return False
    recent = list(pos_history)[-OSC_WINDOW:]
    reversals = 0
    pdx, pdy = None, None
    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i-1][0]
        dy = recent[i][1] - recent[i-1][1]
        if pdx is not None and dx*pdx + dy*pdy < 0:
            reversals += 1
        pdx, pdy = dx, dy
    return reversals >= OSC_REVERSAL_THRESH


# ==========================================
# Runner: CNNTD3 单策略
# ==========================================
def run_cnntd3(model, world, robot_state, robot_goal, max_steps):
    sim = SIM(world_file=world, disable_plotting=True)
    scan,dist,cos,sin,col,goal,a,r = sim.reset(
        robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    prev = [0.0, 0.0]
    for step in range(max_steps):
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


# ==========================================
# Runner: STPS v2
# ==========================================
def run_stps_v2(m_main, m_esc, world, robot_state, robot_goal, max_steps):
    sim = SIM(world_file=world, disable_plotting=True)
    scan,dist,cos,sin,col,goal,a,r = sim.reset(
        robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=max(STALL_WINDOW, OSC_WINDOW+2))
    mode = 'main'; esc_cnt = 0; esc_start = None
    switches = 0; steps_main = 0; esc_steps = BASE_ESCAPE_STEPS

    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        cp = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(cp)
        if mode == 'main':
            steps_main += 1
            trig = False
            if len(pos_hist) >= STALL_WINDOW:
                if np.linalg.norm(pos_hist[-1]-pos_hist[-STALL_WINDOW]) < STALL_DIST:
                    trig = True
            if not trig and steps_main > OSC_MIN_STEPS:
                if detect_oscillation(pos_hist):
                    trig = True
            if trig:
                mode='escape'; esc_cnt=0; esc_start=cp.copy()
                switches+=1; steps_main=0; pos_hist.clear()
                if switches > 1: esc_steps = min(BASE_ESCAPE_STEPS*2, 240)
        else:
            esc_cnt += 1
            d = np.linalg.norm(cp - esc_start)
            if esc_cnt >= esc_steps and d > PROGRESS_DIST:
                mode='main'; steps_main=0; pos_hist.clear()
            elif esc_cnt >= esc_steps*3:
                mode='main'; steps_main=0; pos_hist.clear()

        model = m_main if mode=='main' else m_esc
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


# ==========================================
# Runner: NeuPAN
# ==========================================
def run_neupan(world, robot_state, robot_goal, max_steps):
    # 每次新建planner（因为需要reset）
    try:
        planner = neupan.init_from_yaml(PLANNER_YAML)
    except Exception as e:
        print(f"    NeuPAN init failed: {e}")
        return 'error'

    sim = SIM(world_file=world, disable_plotting=True)
    scan,dist,cos,sin,col,goal,a,r = sim.reset(
        robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)

    # NeuPAN初始化路径
    st = sim.env.robot.state
    start = np.array([[st[0,0]], [st[1,0]], [st[2,0]]])
    goal_arr = np.array(robot_goal, dtype=float)
    planner.reset()
    planner.update_initial_path_from_goal(start, goal_arr)

    for step in range(max_steps):
        st = sim.env.robot.state
        cur = np.array([[st[0,0]], [st[1,0]], [st[2,0]]])

        # 构造scan_dict
        scan_list = scan.tolist() if hasattr(scan, 'tolist') else list(scan)
        scan_dict = {
            'ranges': scan_list,
            'angle_min': -np.pi/2,   # 180° FOV: -90° to +90°
            'angle_max': np.pi/2,
            'range_max': 7.0,
            'range_min': 0.0,
        }

        try:
            points = planner.scan_to_point(cur, scan_dict)
            action, info = planner.forward(cur, points)
        except Exception as e:
            # NeuPAN求解失败（可能是数值问题）
            sim.env.end()
            return 'collision'

        v, w = float(action[0,0]), float(action[1,0])

        # NeuPAN自己的到达/停止判断
        if info.get('arrive', False):
            sim.env.end()
            return 'goal'

        scan,dist,cos,sin,col,goal,a,r = sim.step(lin_velocity=v, ang_velocity=w)

        if col:
            sim.env.end()
            return 'collision'
        if goal:
            sim.env.end()
            return 'goal'

    sim.env.end()
    return 'timeout'


# ===== 加载模型 =====
ckpt_robot = 'robot_nav/models/CNNTD3/checkpoint'
ckpt_model = 'models/CNNTD3/checkpoint'

# CNNTD3 原始baseline
m_baseline = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                    device=device, load_model=False, model_name="cmp_base")
try:
    m_baseline.load("CNNTD3", ckpt_model)
except:
    m_baseline.load("CNNTD3", ckpt_robot)
m_baseline.actor.eval()
print("✅ CNNTD3 baseline")

# v7 precision (for STPS)
m_v7 = CNNTD3(state_dim=185, action_dim=2, max_action=1,
              device=device, load_model=False, model_name="cmp_v7")
m_v7.load("CNNTD3_v7_finetune_best", ckpt_model)
m_v7.actor.eval()
print("✅ v7 precision")

# improved exploration (for STPS)
m_imp = CNNTD3(state_dim=185, action_dim=2, max_action=1,
               device=device, load_model=False, model_name="cmp_imp")
try:
    m_imp.load("CNNTD3_improved", ckpt_model)
except:
    m_imp.load("CNNTD3_improved", ckpt_robot)
m_imp.actor.eval()
print("✅ improved exploration")

# 验证NeuPAN
try:
    test_p = neupan.init_from_yaml(PLANNER_YAML)
    print(f"✅ NeuPAN (step_time={test_p.step_time}s)")
    del test_p
except Exception as e:
    print(f"❌ NeuPAN failed: {e}")
    sys.exit(1)

METHODS = {
    'CNNTD3_baseline': lambda w,rs,rg,ms: run_cnntd3(m_baseline, w, rs, rg, ms),
    'NeuPAN':          lambda w,rs,rg,ms: run_neupan(w, rs, rg, ms),
    'STPS_v2':         lambda w,rs,rg,ms: run_stps_v2(m_v7, m_imp, w, rs, rg, ms),
}

# ===== 评测 =====
print(f"\n{'='*80}")
print(f"公平对比: {list(METHODS.keys())}")
print(f"Seeds: {SEEDS}, 每场景12个扰动起点")
print(f"{'='*80}\n")

all_results = {}
t0 = time.time()

for mname, mfn in METHODS.items():
    all_results[mname] = {}
    print(f"\n----- {mname} -----")

    for scene, cfg in SCENARIOS.items():
        seed_srs = []
        for seed in SEEDS:
            configs = make_configs(cfg['base_xy'], seed)
            succ = 0
            for rs in configs:
                out = mfn(cfg['world'], rs, cfg['goal'], cfg['max_steps'])
                if out == 'goal': succ += 1
            seed_srs.append(succ / len(configs))
        m, s = np.mean(seed_srs), np.std(seed_srs)
        all_results[mname][scene] = {'mean': round(m,3), 'std': round(s,3), 'seeds': seed_srs}
        print(f"  {scene:<16}: {m:.0%} ± {s:.0%}  {[f'{x:.0%}' for x in seed_srs]}")

    elapsed = (time.time()-t0)/60
    print(f"  (elapsed {elapsed:.0f} min)")

# ===== 汇总 =====
print(f"\n{'='*90}")
print(f"{'Method':<20} {'U-trap':>14} {'Double-U':>14} {'Narrow':>14} {'Corridor':>14} {'Avg':>8}")
print(f"{'-'*90}")
for mname in all_results:
    r = all_results[mname]
    vals = []
    line = f"{mname:<20}"
    for s in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']:
        m, sd = r[s]['mean'], r[s]['std']
        line += f" {m*100:>5.0f}±{sd*100:>2.0f}%     "
        vals.append(m)
    avg = np.mean(vals)
    line += f" {avg:.0%}"
    print(line)
print(f"{'='*90}")

with open('unified_comparison.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\n💾 unified_comparison.json")
print(f"总耗时: {(time.time()-t0)/60:.0f} 分钟")
