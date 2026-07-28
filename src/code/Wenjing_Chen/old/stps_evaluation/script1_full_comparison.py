"""
完整对比：CNNTD3 baseline vs NeuPAN vs STPS v2
包含标准场景（泛化性）+ 4个特殊场景
NeuPAN同时在自己的标准场景上测试（排除配置问题）
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

STALL_WINDOW = 20; STALL_DIST = 0.15
BASE_ESCAPE_STEPS = 120; PROGRESS_DIST = 0.5
OSC_WINDOW = 12; OSC_REVERSAL_THRESH = 5; OSC_MIN_STEPS = 8

SCENARIOS = {
    'S1_U_trap': {
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'base_xy': [7.5, 5.0], 'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
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

NEUPAN_PLANNER = os.path.expanduser("~/NeuPAN/example/standard_eval/diff/planner.yaml")
# NeuPAN自己的标准场景
NEUPAN_STANDARD_WORLD = os.path.expanduser("~/NeuPAN/example/standard_eval/diff/env.yaml")


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
    if len(pos_history) < OSC_WINDOW: return False
    recent = list(pos_history)[-OSC_WINDOW:]
    rev = 0; pdx, pdy = None, None
    for i in range(1, len(recent)):
        dx = recent[i][0]-recent[i-1][0]; dy = recent[i][1]-recent[i-1][1]
        if pdx is not None and dx*pdx+dy*pdy < 0: rev += 1
        pdx, pdy = dx, dy
    return rev >= OSC_REVERSAL_THRESH


def run_cnntd3(model, world, rs, rg, ms, random_obs=False):
    sim = SIM(world_file=world, disable_plotting=True)
    if rs is not None:
        scan,d,c,s,col,g,a,r = sim.reset(robot_state=rs, robot_goal=rg, random_obstacles=False)
    else:
        scan,d,c,s,col,g,a,r = sim.reset(random_obstacles=random_obs)
    prev = [0.0, 0.0]
    for step in range(ms):
        st,_ = model.prepare_state(scan,d,c,s,col,g,prev)
        act = model.get_action(np.array(st), False)
        prev = list(act)
        scan,d,c,s,col,g,a,r = sim.step(float(np.clip((act[0]+1)/4,0,0.5)), float(np.clip(act[1],-1,1)))
        if g: sim.env.end(); return 'goal'
        if col: sim.env.end(); return 'collision'
    sim.env.end(); return 'timeout'


def run_stps(m_main, m_esc, world, rs, rg, ms, random_obs=False):
    sim = SIM(world_file=world, disable_plotting=True)
    if rs is not None:
        scan,d,c,s,col,g,a,r = sim.reset(robot_state=rs, robot_goal=rg, random_obstacles=False)
    else:
        scan,d,c,s,col,g,a,r = sim.reset(random_obstacles=random_obs)
    prev=[0.0,0.0]; ph=deque(maxlen=max(STALL_WINDOW,OSC_WINDOW+2))
    mode='main'; ec=0; es=None; sw=0; sm=0; esc_s=BASE_ESCAPE_STEPS
    for step in range(ms):
        rst=sim.env.get_robot_state(); cp=np.array([rst[0].item(),rst[1].item()]); ph.append(cp)
        if mode=='main':
            sm+=1; trig=False
            if len(ph)>=STALL_WINDOW and np.linalg.norm(ph[-1]-ph[-STALL_WINDOW])<STALL_DIST: trig=True
            if not trig and sm>OSC_MIN_STEPS and detect_oscillation(ph): trig=True
            if trig: mode='escape';ec=0;es=cp.copy();sw+=1;sm=0;ph.clear()
            if sw>1: esc_s=min(BASE_ESCAPE_STEPS*2,240)
        else:
            ec+=1; dd=np.linalg.norm(cp-es)
            if ec>=esc_s and dd>PROGRESS_DIST: mode='main';sm=0;ph.clear()
            elif ec>=esc_s*3: mode='main';sm=0;ph.clear()
        mdl=m_main if mode=='main' else m_esc
        st,_=mdl.prepare_state(scan,d,c,s,col,g,prev)
        act=mdl.get_action(np.array(st),False); prev=list(act)
        scan,d,c,s,col,g,a,r=sim.step(float(np.clip((act[0]+1)/4,0,0.5)),float(np.clip(act[1],-1,1)))
        if g: sim.env.end(); return 'goal'
        if col: sim.env.end(); return 'collision'
    sim.env.end(); return 'timeout'


def run_neupan(world, rs, rg, ms, random_obs=False):
    try:
        planner = neupan.init_from_yaml(NEUPAN_PLANNER)
    except: return 'error'
    sim = SIM(world_file=world, disable_plotting=True)
    if rs is not None:
        scan,d,c,s,col,g,a,r = sim.reset(robot_state=rs, robot_goal=rg, random_obstacles=False)
    else:
        scan,d,c,s,col,g,a,r = sim.reset(random_obstacles=random_obs)
    st=sim.env.robot.state
    start=np.array([[st[0,0]],[st[1,0]],[st[2,0]]])
    goal_arr=np.array(rg,dtype=float) if rg is not None else np.array([[9.0],[9.0],[0.0]])
    planner.reset(); planner.update_initial_path_from_goal(start, goal_arr)
    for step in range(ms):
        st=sim.env.robot.state; cur=np.array([[st[0,0]],[st[1,0]],[st[2,0]]])
        scan_list=scan.tolist() if hasattr(scan,'tolist') else list(scan)
        sd={'ranges':scan_list,'angle_min':-np.pi/2,'angle_max':np.pi/2,'range_max':7.0,'range_min':0.0}
        try:
            pts=planner.scan_to_point(cur,sd); act,info=planner.forward(cur,pts)
        except: sim.env.end(); return 'collision'
        if info.get('arrive',False): sim.env.end(); return 'goal'
        v,w=float(act[0,0]),float(act[1,0])
        scan,d,c,s,col,g,a,r=sim.step(lin_velocity=v,ang_velocity=w)
        if col: sim.env.end(); return 'collision'
        if g: sim.env.end(); return 'goal'
    sim.env.end(); return 'timeout'


# 加载模型
ckpt_r='robot_nav/models/CNNTD3/checkpoint'; ckpt_m='models/CNNTD3/checkpoint'
m_base=CNNTD3(state_dim=185,action_dim=2,max_action=1,device=device,load_model=False,model_name="b")
try: m_base.load("CNNTD3",ckpt_m)
except: m_base.load("CNNTD3",ckpt_r)
m_base.actor.eval(); print("✅ CNNTD3 baseline")

m_v7=CNNTD3(state_dim=185,action_dim=2,max_action=1,device=device,load_model=False,model_name="v7")
m_v7.load("CNNTD3_v7_finetune_best",ckpt_m); m_v7.actor.eval(); print("✅ v7")

m_imp=CNNTD3(state_dim=185,action_dim=2,max_action=1,device=device,load_model=False,model_name="imp")
try: m_imp.load("CNNTD3_improved",ckpt_m)
except: m_imp.load("CNNTD3_improved",ckpt_r)
m_imp.actor.eval(); print("✅ improved")

# ====== Part 0: NeuPAN自身标准场景诊断 ======
print("\n===== NeuPAN 自身标准场景诊断 =====")
if os.path.exists(NEUPAN_STANDARD_WORLD):
    np_std_succ = 0
    for i in range(10):
        out = run_neupan(NEUPAN_STANDARD_WORLD, None, [[9.0],[9.0],[0.0]], 300, random_obs=False)
        if out == 'goal': np_std_succ += 1
        print(f"  trial {i+1}: {out}")
    print(f"  NeuPAN自身标准场景 SR = {np_std_succ}/10 = {np_std_succ*10}%")
else:
    print(f"  ❌ NeuPAN标准场景文件不存在: {NEUPAN_STANDARD_WORLD}")

# ====== Part 1: 标准环境泛化测试（100 episodes each）======
print("\n===== 标准环境泛化测试 (100 episodes) =====")
std_world = 'robot_nav/worlds/robot_world.yaml'
for name, fn in [('CNNTD3_baseline', lambda: run_cnntd3(m_base, std_world, None, None, 300, True)),
                 ('NeuPAN', lambda: run_neupan(std_world, None, [[9.0],[9.0],[0.0]], 300, True)),
                 ('STPS_v2', lambda: run_stps(m_v7, m_imp, std_world, None, None, 300, True))]:
    succ = 0
    for i in range(100):
        out = fn()
        if out == 'goal': succ += 1
        if (i+1) % 25 == 0:
            print(f"  {name}: {i+1}/100, SR={succ/(i+1):.0%}")
    print(f"  {name} 标准环境 SR = {succ}%\n")

# ====== Part 2: 特殊场景（1 seed × 12起点，快速版）======
print("\n===== 特殊场景对比 (seed=42, 12起点) =====")
METHODS = {
    'CNNTD3_baseline': lambda w,rs,rg,ms: run_cnntd3(m_base,w,rs,rg,ms),
    'NeuPAN': lambda w,rs,rg,ms: run_neupan(w,rs,rg,ms),
    'STPS_v2': lambda w,rs,rg,ms: run_stps(m_v7,m_imp,w,rs,rg,ms),
}
for mname, mfn in METHODS.items():
    print(f"\n--- {mname} ---")
    for scene, cfg in SCENARIOS.items():
        configs = make_configs(cfg['base_xy'], 42)
        succ = sum(1 for rs in configs if mfn(cfg['world'],rs,cfg['goal'],cfg['max_steps'])=='goal')
        print(f"  {scene:<16}: {succ}/12 = {succ/12:.0%}")

print("\n✅ 完成")
