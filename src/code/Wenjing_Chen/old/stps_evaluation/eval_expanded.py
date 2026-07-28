"""
扩展起点评测：每场景12个扰动起点 × 1 rep（TD3推理确定性，rep无意义）
同时评测 STPS + 各单一策略，一次出全套论文数据
同时记录标准环境的切换统计（检查项4）
"""
import sys, os, numpy as np, torch, json
from collections import deque
sys.path.insert(0, 'robot_nav')
os.chdir('/root/DRL-robot-navigation-IR-SIM')

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')
rng = np.random.default_rng(42)

STALL_WINDOW, STALL_DIST = 20, 0.15
ESCAPE_STEPS, PROGRESS_DIST = 60, 0.5

BASE_SCENARIOS = {
    'S1_U_trap':      ('robot_nav/worlds/u_trap_world.yaml',
                       [7.5, 5.0], [[9.0],[5.0],[0]], 500),
    'S2_Double_U':    ('robot_nav/worlds/double_u_world.yaml',
                       [5.0, 5.0], [[9.0],[5.0],[0]], 500),
    'S3_Narrow_door': ('robot_nav/worlds/narrow_door_world.yaml',
                       [2.0, 5.0], [[8.0],[5.0],[0]], 500),
    'S5_Corridor':    ('robot_nav/worlds/symmetric_corridor_world.yaml',
                       [1.0, 5.0], [[9.0],[5.0],[0]], 500),
}

def make_configs(base_xy, n=12):
    """原点 + 扰动：位置±0.3m，朝向覆盖四个基本方向±0.4rad"""
    cfgs = []
    base_thetas = [0.0, 1.57, 3.14, -1.57]
    for i in range(n):
        th = base_thetas[i % 4] + rng.uniform(-0.4, 0.4)
        x = base_xy[0] + rng.uniform(-0.3, 0.3)
        y = base_xy[1] + rng.uniform(-0.3, 0.3)
        cfgs.append([[x],[y],[th]])
    return cfgs

CONFIGS = {k: make_configs(v[1]) for k, v in BASE_SCENARIOS.items()}


def run_ep(policy_fn, world, robot_state, robot_goal, max_steps, random_obs=False):
    sim = SIM(world_file=world, disable_plotting=True)
    if robot_state is not None:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    else:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(random_obstacles=random_obs)
    prev = [0.0, 0.0]
    ctx = {'pos_hist': deque(maxlen=STALL_WINDOW), 'mode': 'main',
           'esc_cnt': 0, 'esc_start': None, 'switches': 0}
    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        ctx['pos'] = np.array([rs[0].item(), rs[1].item()])
        model = policy_fn(ctx)
        state,_ = model.prepare_state(scan,dist,cos,sin,col,goal,prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,dist,cos,sin,col,goal,a,r = sim.step(lin, ang)
        if goal:
            sim.env.end(); return 'goal', ctx['switches']
        if col:
            sim.env.end(); return 'collision', ctx['switches']
    sim.env.end()
    return 'timeout', ctx['switches']


def single(model):
    return lambda ctx: model

def stps(m_main, m_esc):
    def fn(ctx):
        pos = ctx['pos']
        ctx['pos_hist'].append(pos)
        if ctx['mode'] == 'main':
            if len(ctx['pos_hist']) == STALL_WINDOW:
                if np.linalg.norm(ctx['pos_hist'][-1] - ctx['pos_hist'][0]) < STALL_DIST:
                    ctx['mode'] = 'escape'; ctx['esc_cnt'] = 0
                    ctx['esc_start'] = pos.copy(); ctx['switches'] += 1
                    ctx['pos_hist'].clear()
        else:
            ctx['esc_cnt'] += 1
            if (ctx['esc_cnt'] >= ESCAPE_STEPS and
                    np.linalg.norm(pos - ctx['esc_start']) > PROGRESS_DIST):
                ctx['mode'] = 'main'; ctx['pos_hist'].clear()
        return m_main if ctx['mode'] == 'main' else m_esc
    return fn


ckpt = 'models/CNNTD3/checkpoint'
def load(name):
    m = CNNTD3(state_dim=185, action_dim=2, max_action=1,
               device=device, load_model=False, model_name=f"e_{name}")
    m.load(name, ckpt); m.actor.eval(); return m

m_v7 = load("CNNTD3_v7_finetune_best")
m_imp = load("CNNTD3_improved")
print("✅ models loaded")

POLICIES = {
    'v7_precise':  single(m_v7),
    'improved_explore': single(m_imp),
    'STPS': stps(m_v7, m_imp),
}

results = {}
for pname, pfn in POLICIES.items():
    results[pname] = {}
    print(f"\n===== {pname} =====")
    for scene, (world, _, goal, max_steps) in BASE_SCENARIOS.items():
        succ, tot = 0, 0
        for cfg in CONFIGS[scene]:
            out, _ = run_ep(pfn, world, cfg, goal, max_steps)
            if out == 'goal': succ += 1
            tot += 1
        results[pname][scene] = succ / tot
        print(f"  {scene:<16}: {succ}/{tot} = {succ/tot:.0%}")

# 检查项3+4：STPS标准环境100 episodes + 切换统计
print(f"\n===== STPS standard env (100 eps) + switch stats =====")
succ = 0; sw_hist = []
for i in range(100):
    out, ns = run_ep(POLICIES['STPS'], 'robot_nav/worlds/robot_world.yaml',
                     None, None, 500, random_obs=True)
    if out == 'goal': succ += 1
    sw_hist.append((out, ns))
    if (i+1) % 20 == 0:
        print(f"  {i+1}/100  SR={succ/(i+1):.0%}")
results['STPS']['standard_100'] = succ / 100
eps_with_switch = [x for x in sw_hist if x[1] > 0]
saved = sum(1 for o, n in eps_with_switch if o == 'goal')
print(f"  标准SR(100eps) = {succ}%")
print(f"  发生切换的episode: {len(eps_with_switch)}/100, 其中成功 {saved} 个")
results['switch_stats'] = {'episodes_with_switch': len(eps_with_switch),
                           'switched_and_succeeded': saved}

with open('expanded_eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n💾 expanded_eval_results.json")
