"""
STPS v2: 改进切换逻辑
1. 保留原有停滞检测（位移<阈值）
2. 新增震荡检测：位置来回跳（方向反转次数过多）→ 更早触发切换
3. 新增反复卡住升级：同一episode第二次卡住时 escape_steps 翻倍

震荡检测原理：
  正常导航时机器人朝目标方向持续移动，位移方向稳定。
  在trap中机器人会"接近墙→弹回→再接近→弹回"，位移方向频繁反转。
  检测方式：计算连续步之间位移向量的夹角，如果短窗口内方向反转次数
  超过阈值，判定为震荡。
  这在窄门场景不会误触发，因为穿窄门时方向是稳定的。
"""
import sys, os, numpy as np, torch, json
from collections import deque
sys.path.insert(0, 'robot_nav')
os.chdir(os.path.expanduser('~/DRL-robot-navigation-IR-SIM'))

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')
rng = np.random.default_rng(42)

# === 切换参数 ===
STALL_WINDOW = 20
STALL_DIST = 0.15
BASE_ESCAPE_STEPS = 120
PROGRESS_DIST = 0.5

# === 震荡检测参数 ===
OSC_WINDOW = 12         # 检测窗口：最近12步
OSC_REVERSAL_THRESH = 5 # 12步内方向反转>=5次 → 震荡
OSC_MIN_STEPS = 8       # 至少走8步后才开始检测（避免初始噪声）

SCENARIOS = {
    'S1_U_trap': {
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'base_xy': [7.5, 5.0],
        'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
    },
    'S2_Double_U': {
        'world': 'robot_nav/worlds/double_u_world.yaml',
        'base_xy': [5.0, 5.0],
        'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
    },
    'S3_Narrow_door': {
        'world': 'robot_nav/worlds/narrow_door_world.yaml',
        'base_xy': [2.0, 5.0],
        'goal': [[8.0],[5.0],[0]], 'max_steps': 500,
    },
    'S5_Corridor': {
        'world': 'robot_nav/worlds/symmetric_corridor_world.yaml',
        'base_xy': [1.0, 5.0],
        'goal': [[9.0],[5.0],[0]], 'max_steps': 500,
    },
}

def make_configs(base_xy, n=12):
    cfgs = []
    base_thetas = [0.0, 1.57, 3.14, -1.57]
    for i in range(n):
        th = base_thetas[i % 4] + rng.uniform(-0.4, 0.4)
        x = base_xy[0] + rng.uniform(-0.3, 0.3)
        y = base_xy[1] + rng.uniform(-0.3, 0.3)
        cfgs.append([[x],[y],[th]])
    return cfgs


def detect_oscillation(pos_history):
    """检测位移方向是否频繁反转"""
    if len(pos_history) < OSC_WINDOW:
        return False

    recent = list(pos_history)[-OSC_WINDOW:]
    reversals = 0
    prev_dx, prev_dy = None, None

    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i-1][0]
        dy = recent[i][1] - recent[i-1][1]
        if prev_dx is not None:
            # 计算方向是否反转：内积<0表示方向翻转
            dot = dx * prev_dx + dy * prev_dy
            if dot < 0:
                reversals += 1
        prev_dx, prev_dy = dx, dy

    return reversals >= OSC_REVERSAL_THRESH


def run_stps_v2(m_main, m_esc, world, robot_state, robot_goal, max_steps,
                verbose=False):
    sim = SIM(world_file=world, disable_plotting=True)
    if robot_state is not None:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    else:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(random_obstacles=True)

    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=max(STALL_WINDOW, OSC_WINDOW + 2))
    mode = 'main'
    esc_cnt = 0
    esc_start = None
    switches = 0
    total_steps_in_main = 0
    escape_steps_current = BASE_ESCAPE_STEPS  # 可升级

    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        curr_pos = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(curr_pos)

        if mode == 'main':
            total_steps_in_main += 1
            trigger = False
            trigger_reason = ""

            # 检测1：停滞（原有）
            if len(pos_hist) >= STALL_WINDOW:
                moved = np.linalg.norm(pos_hist[-1] - pos_hist[-STALL_WINDOW])
                if moved < STALL_DIST:
                    trigger = True
                    trigger_reason = f"stall (moved {moved:.3f}m in {STALL_WINDOW} steps)"

            # 检测2：震荡（新增）
            if not trigger and total_steps_in_main > OSC_MIN_STEPS:
                if detect_oscillation(pos_hist):
                    trigger = True
                    trigger_reason = "oscillation detected"

            if trigger:
                mode = 'escape'
                esc_cnt = 0
                esc_start = curr_pos.copy()
                switches += 1
                total_steps_in_main = 0
                pos_hist.clear()
                if verbose:
                    print(f"    step {step}: → escape ({trigger_reason})")
                # 反复卡住升级：第二次及之后翻倍逃脱时间
                if switches > 1:
                    escape_steps_current = min(BASE_ESCAPE_STEPS * 2, 240)
        else:
            esc_cnt += 1
            dist_from_start = np.linalg.norm(curr_pos - esc_start)
            if esc_cnt >= escape_steps_current and dist_from_start > PROGRESS_DIST:
                mode = 'main'
                total_steps_in_main = 0
                pos_hist.clear()
                if verbose:
                    print(f"    step {step}: → main (escaped {dist_from_start:.2f}m)")
            elif esc_cnt >= escape_steps_current * 3:
                mode = 'main'
                total_steps_in_main = 0
                pos_hist.clear()
                if verbose:
                    print(f"    step {step}: → main (escape timeout)")

        model = m_main if mode == 'main' else m_esc
        state,_ = model.prepare_state(scan,dist,cos,sin,col,goal,prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,dist,cos,sin,col,goal,a,r = sim.step(lin, ang)

        if goal:
            sim.env.end(); return 'goal', switches
        if col:
            sim.env.end(); return 'collision', switches
    sim.env.end()
    return 'timeout', switches


# 加载模型
ckpt = 'models/CNNTD3/checkpoint'
m_v7 = CNNTD3(state_dim=185, action_dim=2, max_action=1,
              device=device, load_model=False, model_name="v2_main")
m_v7.load("CNNTD3_v7_finetune_best", ckpt); m_v7.actor.eval()
m_imp = CNNTD3(state_dim=185, action_dim=2, max_action=1,
               device=device, load_model=False, model_name="v2_esc")
m_imp.load("CNNTD3_improved", ckpt); m_imp.actor.eval()
print("✅ Models loaded\n")

CONFIGS = {k: make_configs(v['base_xy']) for k, v in SCENARIOS.items()}

results = {}

# 场景评测
for scene, cfg in SCENARIOS.items():
    succ, tot = 0, 0
    for rs in CONFIGS[scene]:
        out, ns = run_stps_v2(m_v7, m_imp, cfg['world'], rs, cfg['goal'],
                              cfg['max_steps'], verbose=(scene == 'S1_U_trap'))
        if out == 'goal': succ += 1
        tot += 1
    sr = succ / tot
    results[scene] = sr
    print(f"{scene:<18}: {succ}/{tot} = {sr:.0%}")

# 标准环境 100 episodes
print("\n标准环境 100 episodes...")
std_succ, sw_total, sw_saved = 0, 0, 0
for i in range(100):
    out, ns = run_stps_v2(m_v7, m_imp, 'robot_nav/worlds/robot_world.yaml',
                          None, None, 500)
    if out == 'goal': std_succ += 1
    if ns > 0:
        sw_total += 1
        if out == 'goal': sw_saved += 1
    if (i+1) % 20 == 0:
        print(f"  {i+1}/100, SR={std_succ/(i+1):.0%}")
results['standard'] = std_succ / 100

# 汇总
print(f"\n{'='*60}")
print(f"STPS v2 (震荡检测 + 反复升级)")
print(f"{'='*60}")
print(f"  标准(100ep):  {results['standard']:.0%}")
for s in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']:
    print(f"  {s:<16}: {results[s]:.0%}")
avg = np.mean([results[s] for s in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']])
print(f"  场景平均:      {avg:.0%}")
print(f"  切换统计: {sw_total}/100 触发, {sw_saved} 个成功救回")
print(f"{'='*60}")

with open('stps_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("💾 stps_v2_results.json")
