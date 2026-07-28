"""评测 v7_finetune_best 在泛化+4个特殊场景上的表现"""
import sys, os, numpy as np, torch, json
sys.path.insert(0, 'robot_nav')
os.chdir('/root/DRL-robot-navigation-IR-SIM')

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

device = torch.device('cpu')

SPECIAL_SCENARIOS = {
    'S1_U_trap': {
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'cases': [
            ([[7.5],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
        ], 'max_steps': 300,
    },
    'S2_Double_U': {
        'world': 'robot_nav/worlds/double_u_world.yaml',
        'cases': [
            ([[5.0],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
        ], 'max_steps': 300,
    },
    'S3_Narrow_door': {
        'world': 'robot_nav/worlds/narrow_door_world.yaml',
        'cases': [
            ([[2.0],[5.0],[0.0]],  [[8.0],[5.0],[0]]),
            ([[2.0],[5.0],[0.3]],  [[8.0],[5.0],[0]]),
            ([[2.0],[5.0],[-0.3]], [[8.0],[5.0],[0]]),
            ([[2.0],[6.0],[0.0]],  [[8.0],[5.0],[0]]),
        ], 'max_steps': 300,
    },
    'S5_Corridor': {
        'world': 'robot_nav/worlds/symmetric_corridor_world.yaml',
        'cases': [
            ([[1.0],[5.0],[0.0]],  [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[1.57]], [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[3.14]], [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[-1.57]],[[9.0],[5.0],[0]]),
        ], 'max_steps': 300,
    },
}

def run_episode(model, world, robot_state, robot_goal, max_steps, random_obs):
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
        if goal:
            sim.env.end()
            return 'goal'
        if col:
            sim.env.end()
            return 'collision'
    sim.env.end()
    return 'timeout'

# 加载 v7 best
ckpt_dir = 'models/CNNTD3/checkpoint'
model = CNNTD3(state_dim=185, action_dim=2, max_action=1,
               device=device, load_model=False, model_name="v7_eval")
model.load("CNNTD3_v7_finetune_best", ckpt_dir)
model.actor.eval()
print("✅ Loaded CNNTD3_v7_finetune_best")

results = {}

# 泛化评测（50次，跟你之前的generalization_enhanced一致）
print("\n📊 泛化评测 (50 episodes)...")
gen_success = 0
for i in range(50):
    outcome = run_episode(model, 'robot_nav/worlds/robot_world.yaml', None, None, 300, True)
    if outcome == 'goal': gen_success += 1
    if (i+1) % 10 == 0:
        print(f"  {i+1}/50 done, current SR={gen_success/(i+1):.0%}")
gen_sr = gen_success / 50
results['generalization'] = gen_sr
print(f"  泛化 SR = {gen_sr:.0%} ({gen_success}/50)")

# 特殊场景评测
for scene_name, cfg in SPECIAL_SCENARIOS.items():
    total, success, collisions, timeouts = 0, 0, 0, 0
    for robot_state, robot_goal in cfg['cases']:
        for rep in range(3):
            outcome = run_episode(model, cfg['world'], robot_state, robot_goal, cfg['max_steps'], False)
            if outcome == 'goal': success += 1
            elif outcome == 'collision': collisions += 1
            else: timeouts += 1
            total += 1
    sr = success / total
    results[scene_name] = sr
    print(f"  {scene_name:<18}: {success}/{total} = {sr:.0%} (col={collisions}, timeout={timeouts})")

# 汇总
print(f"\n{'='*60}")
print(f"v7_finetune_best 完整评测结果")
print(f"{'='*60}")
print(f"  泛化(标准):  {results['generalization']:.0%}")
print(f"  S1 U-trap:   {results['S1_U_trap']:.0%}")
print(f"  S2 Double-U: {results['S2_Double_U']:.0%}")
print(f"  S3 窄门:     {results['S3_Narrow_door']:.0%}")
print(f"  S5 走廊:     {results['S5_Corridor']:.0%}")

avg_scenario = np.mean([results[k] for k in ['S1_U_trap','S2_Double_U','S3_Narrow_door','S5_Corridor']])
print(f"  场景平均:    {avg_scenario:.0%}")
print(f"{'='*60}")

with open('v7_finetune_eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n💾 结果已保存到 v7_finetune_eval_results.json")
