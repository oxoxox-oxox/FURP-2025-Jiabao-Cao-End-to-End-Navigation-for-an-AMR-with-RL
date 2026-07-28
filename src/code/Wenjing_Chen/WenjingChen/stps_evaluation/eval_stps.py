"""
Stall-Triggered Policy Switching (STPS)
两个互补模型 + 运行时切换：
  - 主策略: v7_finetune_best (精确导航，会窄门)
  - 逃脱策略: CNNTD3_improved (探索模式，会逃U-trap)
  - 切换逻辑: 位置停滞N步 → 切换到逃脱策略K步 → 切回主策略

不需要训练，纯推理逻辑。
"""
import sys, os, numpy as np, torch, json
from collections import deque
sys.path.insert(0, 'robot_nav')
os.chdir('/root/DRL-robot-navigation-IR-SIM')

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')

# ===== 切换参数 =====
STALL_WINDOW = 20      # 检测窗口：过去20步
STALL_DIST = 0.15      # 20步内移动距离小于0.15m → 判定卡住
ESCAPE_STEPS = 60      # 切换到逃脱策略后至少执行60步
PROGRESS_DIST = 0.5    # 逃脱策略执行中若移动超过0.5m → 认为已脱困

SPECIAL_SCENARIOS = {
    'S1_U_trap': {
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'cases': [
            ([[7.5],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
        ], 'max_steps': 500,
    },
    'S2_Double_U': {
        'world': 'robot_nav/worlds/double_u_world.yaml',
        'cases': [
            ([[5.0],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
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
    'S5_Corridor': {
        'world': 'robot_nav/worlds/symmetric_corridor_world.yaml',
        'cases': [
            ([[1.0],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
        ], 'max_steps': 500,
    },
}


def run_episode_stps(model_main, model_escape, world, robot_state, robot_goal,
                     max_steps, random_obs, verbose=False):
    """带切换逻辑的episode"""
    sim = SIM(world_file=world, disable_plotting=True)
    if robot_state is not None:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(
            robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    else:
        scan,dist,cos,sin,col,goal,a,r = sim.reset(random_obstacles=random_obs)

    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=STALL_WINDOW)
    mode = 'main'          # 'main' or 'escape'
    escape_counter = 0
    escape_start_pos = None
    n_switches = 0

    for step in range(max_steps):
        # 当前位置
        rs = sim.env.get_robot_state()
        curr_pos = np.array([rs[0].item(), rs[1].item()])
        pos_hist.append(curr_pos)

        # === 切换逻辑 ===
        if mode == 'main':
            if len(pos_hist) == STALL_WINDOW:
                moved = np.linalg.norm(pos_hist[-1] - pos_hist[0])
                if moved < STALL_DIST:
                    mode = 'escape'
                    escape_counter = 0
                    escape_start_pos = curr_pos.copy()
                    n_switches += 1
                    pos_hist.clear()
                    if verbose:
                        print(f"    step {step}: STALL detected → escape mode")
        else:  # escape mode
            escape_counter += 1
            escaped = np.linalg.norm(curr_pos - escape_start_pos) > PROGRESS_DIST
            if escape_counter >= ESCAPE_STEPS and escaped:
                mode = 'main'
                pos_hist.clear()
                if verbose:
                    print(f"    step {step}: escaped → main mode")

        # === 用当前模式的模型 ===
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


# ===== 加载两个模型 =====
ckpt_dir = 'models/CNNTD3/checkpoint'

model_main = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                    device=device, load_model=False, model_name="stps_main")
model_main.load("CNNTD3_v7_finetune_best", ckpt_dir)
model_main.actor.eval()
print("✅ Main policy: v7_finetune_best (precise)")

model_escape = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                      device=device, load_model=False, model_name="stps_escape")
model_escape.load("CNNTD3_improved", ckpt_dir)
model_escape.actor.eval()
print("✅ Escape policy: CNNTD3_improved (exploratory)")

results = {}

# 泛化评测
print("\n📊 泛化评测 (50 episodes)...")
gen_success, total_switches = 0, 0
for i in range(50):
    outcome, ns = run_episode_stps(model_main, model_escape,
                                    'robot_nav/worlds/robot_world.yaml',
                                    None, None, 500, True)
    if outcome == 'goal': gen_success += 1
    total_switches += ns
    if (i+1) % 10 == 0:
        print(f"  {i+1}/50, SR={gen_success/(i+1):.0%}, switches so far={total_switches}")
results['generalization'] = gen_success / 50
print(f"  泛化 SR = {results['generalization']:.0%}")

# 特殊场景
for scene_name, cfg in SPECIAL_SCENARIOS.items():
    total, success, switches = 0, 0, 0
    for robot_state, robot_goal in cfg['cases']:
        for rep in range(3):
            outcome, ns = run_episode_stps(model_main, model_escape,
                                            cfg['world'], robot_state, robot_goal,
                                            cfg['max_steps'], False,
                                            verbose=(rep==0))
            if outcome == 'goal': success += 1
            switches += ns
            total += 1
    sr = success / total
    results[scene_name] = sr
    print(f"  {scene_name:<18}: {success}/{total} = {sr:.0%} (avg switches={switches/total:.1f})")

# 汇总
print(f"\n{'='*60}")
print(f"STPS (Stall-Triggered Policy Switching) 结果")
print(f"{'='*60}")
print(f"  泛化(标准):  {results['generalization']:.0%}")
print(f"  S1 U-trap:   {results['S1_U_trap']:.0%}")
print(f"  S2 Double-U: {results['S2_Double_U']:.0%}")
print(f"  S3 窄门:     {results['S3_Narrow_door']:.0%}")
print(f"  S5 走廊:     {results['S5_Corridor']:.0%}")
avg = np.mean([results[k] for k in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']])
print(f"  场景平均:    {avg:.0%}")
print(f"{'='*60}")

with open('stps_eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n💾 保存到 stps_eval_results.json")
