"""
STPS v3: 更强的切换逻辑
改进点：
1. 三级检测：停滞 → 震荡 → 目标距离停滞（距离目标不减少）
2. 渐进式逃脱：第一次卡住120步，第二次180步，第三次240步
3. 逃脱后冷却期：切回main后前30步不检测（防止刚切回就又触发）
4. 可选：用utrap_specialist替代improved作为逃脱策略（如果训练了的话）
"""
import sys, os, numpy as np, torch, json
from collections import deque

BASE_DIR = os.path.expanduser('~/DRL-robot-navigation-IR-SIM')
sys.path.insert(0, os.path.join(BASE_DIR, 'robot_nav'))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)
os.environ.pop("DISPLAY", None)

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')

# === 切换参数 ===
STALL_WINDOW = 20
STALL_DIST = 0.15
PROGRESS_DIST = 0.5
OSC_WINDOW = 12
OSC_REVERSAL_THRESH = 5
COOLDOWN_STEPS = 30       # 切回main后冷却期
GOAL_STALL_WINDOW = 30    # 目标距离停滞检测窗口
GOAL_STALL_THRESH = 0.05  # 30步内目标距离变化<0.05m → 卡住

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
    thetas = [0.0, 1.57, 3.14, -1.57]
    for i in range(n):
        th = thetas[i%4] + rng.uniform(-0.4, 0.4)
        x = base_xy[0] + rng.uniform(-0.3, 0.3)
        y = base_xy[1] + rng.uniform(-0.3, 0.3)
        cfgs.append([[x],[y],[th]])
    return cfgs


def detect_oscillation(pos_history):
    if len(pos_history) < OSC_WINDOW: return False
    recent = list(pos_history)[-OSC_WINDOW:]
    rev=0; pdx,pdy=None,None
    for i in range(1,len(recent)):
        dx=recent[i][0]-recent[i-1][0]; dy=recent[i][1]-recent[i-1][1]
        if pdx is not None and dx*pdx+dy*pdy<0: rev+=1
        pdx,pdy=dx,dy
    return rev >= OSC_REVERSAL_THRESH


def run_stps_v3(m_main, m_esc, world, robot_state, robot_goal, max_steps,
                verbose=False):
    sim = SIM(world_file=world, disable_plotting=True)
    if robot_state is not None:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    else:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(random_obstacles=True)

    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=max(STALL_WINDOW, OSC_WINDOW+2))
    dist_hist = deque(maxlen=GOAL_STALL_WINDOW)
    mode = 'main'
    esc_cnt = 0
    esc_start = None
    switches = 0
    steps_main = 0
    cooldown = 0
    # 渐进式逃脱时间
    escape_schedule = [120, 180, 240]

    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        cp = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(cp)
        dist_hist.append(dist)

        if mode == 'main':
            steps_main += 1
            cooldown = max(0, cooldown - 1)

            if cooldown == 0:
                trigger = False
                reason = ""

                # 检测1：位置停滞
                if len(pos_hist) >= STALL_WINDOW:
                    moved = np.linalg.norm(pos_hist[-1] - pos_hist[-STALL_WINDOW])
                    if moved < STALL_DIST:
                        trigger = True; reason = f"stall({moved:.3f}m)"

                # 检测2：震荡
                if not trigger and steps_main > 8:
                    if detect_oscillation(pos_hist):
                        trigger = True; reason = "oscillation"

                # 检测3：目标距离停滞（新增）
                if not trigger and len(dist_hist) >= GOAL_STALL_WINDOW:
                    dist_change = abs(dist_hist[-1] - dist_hist[-GOAL_STALL_WINDOW])
                    if dist_change < GOAL_STALL_THRESH and dist > 0.5:
                        trigger = True; reason = f"goal_stall(Δd={dist_change:.3f})"

                if trigger:
                    mode = 'escape'; esc_cnt = 0; esc_start = cp.copy()
                    # 渐进式逃脱时间
                    esc_dur = escape_schedule[min(switches, len(escape_schedule)-1)]
                    switches += 1; steps_main = 0; pos_hist.clear(); dist_hist.clear()
                    if verbose:
                        print(f"    step {step}: →escape ({reason}, dur={esc_dur})")
        else:
            esc_cnt += 1
            esc_dur = escape_schedule[min(switches-1, len(escape_schedule)-1)]
            d_from_start = np.linalg.norm(cp - esc_start)
            if esc_cnt >= esc_dur and d_from_start > PROGRESS_DIST:
                mode = 'main'; steps_main = 0; cooldown = COOLDOWN_STEPS
                pos_hist.clear(); dist_hist.clear()
                if verbose:
                    print(f"    step {step}: →main (escaped {d_from_start:.2f}m)")
            elif esc_cnt >= esc_dur * 3:
                mode = 'main'; steps_main = 0; cooldown = COOLDOWN_STEPS
                pos_hist.clear(); dist_hist.clear()
                if verbose:
                    print(f"    step {step}: →main (escape timeout)")

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


# ===== 加载 =====
ckpt = 'models/CNNTD3/checkpoint'
m_v7 = CNNTD3(state_dim=185,action_dim=2,max_action=1,device=device,load_model=False,model_name="v3m")
m_v7.load("CNNTD3_v7_finetune_best", ckpt); m_v7.actor.eval()

m_imp = CNNTD3(state_dim=185,action_dim=2,max_action=1,device=device,load_model=False,model_name="v3e")
try: m_imp.load("CNNTD3_improved", ckpt)
except: m_imp.load("CNNTD3_improved", 'robot_nav/models/CNNTD3/checkpoint')
m_imp.actor.eval()

# 如果有utrap_specialist，用它替代improved
specialist_path = f"{ckpt}/CNNTD3_utrap_specialist_best_actor.pth"
if os.path.exists(specialist_path):
    m_spec = CNNTD3(state_dim=185,action_dim=2,max_action=1,device=device,load_model=False,model_name="v3s")
    m_spec.load("CNNTD3_utrap_specialist_best", ckpt); m_spec.actor.eval()
    m_escape = m_spec
    print("✅ Using utrap_specialist as escape policy")
else:
    m_escape = m_imp
    print("✅ Using improved as escape policy (no specialist found)")

print("✅ Models loaded\n")

# ===== 评测 =====
SEEDS = [42, 123, 2026]
results = {}

for scene, cfg in SCENARIOS.items():
    seed_srs = []
    for seed in SEEDS:
        configs = make_configs(cfg['base_xy'], seed)
        succ = 0
        for rs in configs:
            out, ns = run_stps_v3(m_v7, m_escape, cfg['world'], rs,
                                   cfg['goal'], cfg['max_steps'],
                                   verbose=(scene=='S1_U_trap' and seed==42))
            if out == 'goal': succ += 1
        seed_srs.append(succ/len(configs))
    m, s = np.mean(seed_srs), np.std(seed_srs)
    results[scene] = {'mean': m, 'std': s, 'seeds': seed_srs}
    print(f"{scene:<16}: {m:.0%} ± {s:.0%}  {[f'{x:.0%}' for x in seed_srs]}")

# 标准环境
print("\n标准环境 (100 episodes)...")
succ = 0; sw_total = 0
for i in range(100):
    out, ns = run_stps_v3(m_v7, m_escape, 'robot_nav/worlds/robot_world.yaml',
                          None, None, 300)
    if out == 'goal': succ += 1
    sw_total += ns
    if (i+1) % 25 == 0:
        print(f"  {i+1}/100, SR={succ/(i+1):.0%}")
results['standard'] = succ / 100
print(f"  标准SR = {succ}%, 总切换次数 = {sw_total}")

# 汇总
print(f"\n{'='*60}")
print(f"STPS v3 结果")
print(f"{'='*60}")
print(f"  标准:     {results['standard']:.0%}")
for s in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']:
    r = results[s]
    print(f"  {s:<16}: {r['mean']:.0%} ± {r['std']:.0%}")
avg = np.mean([results[s]['mean'] for s in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']])
print(f"  场景平均:  {avg:.0%}")
print(f"{'='*60}")

with open('stps_v3_results.json','w') as f:
    json.dump(results,f,indent=2,default=float)
print("\n💾 stps_v3_results.json")
