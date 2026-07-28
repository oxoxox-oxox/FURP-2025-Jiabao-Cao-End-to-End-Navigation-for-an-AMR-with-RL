"""
公平对比：CNNTD3 baseline vs STPS v2 
在3个不同seed × 12个起点下评测，报告均值±标准差

NeuPAN需要单独跑（不同的仿真接口），这里先跑CNNTD3系列
"""
import sys, os, numpy as np, torch, json
from collections import deque
sys.path.insert(0, 'robot_nav')
os.chdir(os.path.expanduser('~/DRL-robot-navigation-IR-SIM'))

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')

# STPS v2 参数
STALL_WINDOW = 20
STALL_DIST = 0.15
BASE_ESCAPE_STEPS = 120
PROGRESS_DIST = 0.5
OSC_WINDOW = 12
OSC_REVERSAL_THRESH = 5
OSC_MIN_STEPS = 8

SEEDS = [42, 123, 2026]  # 3个不同seed

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

def make_configs(base_xy, seed, n=12):
    rng = np.random.default_rng(seed)
    cfgs = []
    base_thetas = [0.0, 1.57, 3.14, -1.57]
    for i in range(n):
        th = base_thetas[i % 4] + rng.uniform(-0.4, 0.4)
        x = base_xy[0] + rng.uniform(-0.3, 0.3)
        y = base_xy[1] + rng.uniform(-0.3, 0.3)
        cfgs.append([[x],[y],[th]])
    return cfgs


def detect_oscillation(pos_history):
    if len(pos_history) < OSC_WINDOW:
        return False
    recent = list(pos_history)[-OSC_WINDOW:]
    reversals = 0
    prev_dx, prev_dy = None, None
    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i-1][0]
        dy = recent[i][1] - recent[i-1][1]
        if prev_dx is not None:
            if dx * prev_dx + dy * prev_dy < 0:
                reversals += 1
        prev_dx, prev_dy = dx, dy
    return reversals >= OSC_REVERSAL_THRESH


def run_single(model, world, robot_state, robot_goal, max_steps, random_obs=False):
    """单策略运行"""
    sim = SIM(world_file=world, disable_plotting=True)
    if robot_state is not None:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    else:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(random_obstacles=random_obs)
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


def run_stps_v2(m_main, m_esc, world, robot_state, robot_goal, max_steps,
                random_obs=False):
    """STPS v2"""
    sim = SIM(world_file=world, disable_plotting=True)
    if robot_state is not None:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    else:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(random_obstacles=random_obs)
    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=max(STALL_WINDOW, OSC_WINDOW + 2))
    mode = 'main'; esc_cnt = 0; esc_start = None
    switches = 0; steps_in_main = 0
    esc_steps_cur = BASE_ESCAPE_STEPS

    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        curr_pos = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(curr_pos)

        if mode == 'main':
            steps_in_main += 1
            trigger = False
            if len(pos_hist) >= STALL_WINDOW:
                if np.linalg.norm(pos_hist[-1] - pos_hist[-STALL_WINDOW]) < STALL_DIST:
                    trigger = True
            if not trigger and steps_in_main > OSC_MIN_STEPS:
                if detect_oscillation(pos_hist):
                    trigger = True
            if trigger:
                mode = 'escape'; esc_cnt = 0
                esc_start = curr_pos.copy(); switches += 1
                steps_in_main = 0; pos_hist.clear()
                if switches > 1:
                    esc_steps_cur = min(BASE_ESCAPE_STEPS * 2, 240)
        else:
            esc_cnt += 1
            d = np.linalg.norm(curr_pos - esc_start)
            if esc_cnt >= esc_steps_cur and d > PROGRESS_DIST:
                mode = 'main'; steps_in_main = 0; pos_hist.clear()
            elif esc_cnt >= esc_steps_cur * 3:
                mode = 'main'; steps_in_main = 0; pos_hist.clear()

        model = m_main if mode == 'main' else m_esc
        state,_ = model.prepare_state(scan,dist,cos,sin,col,goal,prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,dist,cos,sin,col,goal,a,r = sim.step(lin, ang)
        if goal: sim.env.end(); return 'goal', switches
        if col: sim.env.end(); return 'collision', switches
    sim.env.end()
    return 'timeout', switches


# ===== 加载所有模型 =====
ckpt = 'models/CNNTD3/checkpoint'

# 找所有可用的CNNTD3 checkpoint
import glob
available = set()
for f in glob.glob(f'{ckpt}/*_actor.pth'):
    name = os.path.basename(f).replace('_actor.pth', '')
    available.add(name)
print(f"可用模型: {sorted(available)}")

# 加载baseline（找原始CNNTD3）
baseline_names = ['CNNTD3', 'CNNTD3_v2']
models = {}
for bn in baseline_names:
    if bn in available:
        m = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                   device=device, load_model=False, model_name=f"cmp_{bn}")
        m.load(bn, ckpt); m.actor.eval()
        models[bn] = m
        print(f"✅ Loaded {bn}")

# v7 (precision)
m_v7 = CNNTD3(state_dim=185, action_dim=2, max_action=1,
              device=device, load_model=False, model_name="cmp_v7")
m_v7.load("CNNTD3_v7_finetune_best", ckpt); m_v7.actor.eval()
models['v7_precision'] = m_v7
print("✅ Loaded v7_precision")

# improved (exploration)
m_imp = CNNTD3(state_dim=185, action_dim=2, max_action=1,
               device=device, load_model=False, model_name="cmp_imp")
m_imp.load("CNNTD3_improved", ckpt); m_imp.actor.eval()
models['improved_explore'] = m_imp
print("✅ Loaded improved_explore")

print(f"\n共 {len(models)} 个单策略 + STPS v2\n")

# ===== 多seed评测 =====
all_results = {}

for method_name in list(models.keys()) + ['STPS_v2']:
    all_results[method_name] = {}
    print(f"\n===== {method_name} =====")

    for scene, cfg in SCENARIOS.items():
        seed_srs = []
        for seed in SEEDS:
            configs = make_configs(cfg['base_xy'], seed)
            succ = 0
            for rs in configs:
                if method_name == 'STPS_v2':
                    out, _ = run_stps_v2(m_v7, m_imp, cfg['world'], rs,
                                         cfg['goal'], cfg['max_steps'])
                else:
                    out = run_single(models[method_name], cfg['world'], rs,
                                     cfg['goal'], cfg['max_steps'])
                if out == 'goal': succ += 1
            seed_srs.append(succ / len(configs))
        mean_sr = np.mean(seed_srs)
        std_sr = np.std(seed_srs)
        all_results[method_name][scene] = {
            'mean': mean_sr, 'std': std_sr,
            'per_seed': seed_srs
        }
        print(f"  {scene:<16}: {mean_sr:.0%} ± {std_sr:.0%}  {[f'{s:.0%}' for s in seed_srs]}")

    # 标准环境（3×50=150 episodes）
    seed_srs = []
    for seed in SEEDS:
        np.random.seed(seed)  # 影响随机障碍布局
        succ = 0
        for i in range(50):
            if method_name == 'STPS_v2':
                out, _ = run_stps_v2(m_v7, m_imp,
                                     'robot_nav/worlds/robot_world.yaml',
                                     None, None, 500, random_obs=True)
            else:
                out = run_single(models[method_name],
                                 'robot_nav/worlds/robot_world.yaml',
                                 None, None, 500, random_obs=True)
            if out == 'goal': succ += 1
        seed_srs.append(succ / 50)
    mean_sr = np.mean(seed_srs)
    std_sr = np.std(seed_srs)
    all_results[method_name]['standard'] = {
        'mean': mean_sr, 'std': std_sr,
        'per_seed': seed_srs
    }
    print(f"  {'standard':<16}: {mean_sr:.0%} ± {std_sr:.0%}  {[f'{s:.0%}' for s in seed_srs]}")

# ===== 汇总表 =====
print(f"\n{'='*90}")
print(f"{'Method':<20} {'Standard':>12} {'U-trap':>12} {'Double-U':>12} {'Narrow':>12} {'Corridor':>12} {'Avg_S':>8}")
print(f"{'-'*90}")
for method_name in all_results:
    r = all_results[method_name]
    std = r['standard']
    vals = []
    line = f"{method_name:<20} {std['mean']:>5.0%}±{std['std']:.0%}    "
    for s in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']:
        m, sd = r[s]['mean'], r[s]['std']
        line += f" {m:>5.0%}±{sd:.0%}    "
        vals.append(m)
    avg = np.mean(vals)
    line += f" {avg:.0%}"
    print(line)
print(f"{'='*90}")

with open('fair_comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print("\n💾 fair_comparison_results.json")
