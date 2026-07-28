"""
增强版泛化测试：50次随机试验，固定随机种子，所有模型用完全相同的场景序列对比
用法: cd ~/DRL-robot-navigation-IR-SIM && python final_evaluate_generalization.py
"""
import sys, numpy as np, torch, random
from pathlib import Path
sys.path.insert(0, 'robot_nav')
from SIM_ENV.sim import SIM
from models.CNNTD3.CNNTD3 import CNNTD3
from models.RCPG.RCPG import RCPG

device = torch.device('cpu')

CNNTD3_MODELS = [
    'CNNTD3', 'CNNTD3_v2', 'CNNTD3_improved', 'CNNTD3_v3',
    'CNNTD3_v4_improved_best', 'CNNTD3_curriculum_only',
]
CNNTD3_DIR = Path('robot_nav/models/CNNTD3/checkpoint')

N_TRIALS = 50          # 从20增加到50
RANDOM_SEED = 42        # 固定种子，保证所有模型面对相同的随机场景序列


def run_episode_cnntd3(model, world, seed, max_steps=300):
    random.seed(seed)
    np.random.seed(seed)
    sim = SIM(world_file=world, disable_plotting=True)
    scan, dist, cos, sin, col, goal, a, r = sim.reset(random_obstacles=True)
    prev = [0.0, 0.0]
    success = False
    for step in range(max_steps):
        state, _ = model.prepare_state(scan, dist, cos, sin, col, goal, prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0] + 1) / 4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan, dist, cos, sin, col, goal, a, r = sim.step(lin, ang)
        if goal:
            success = True
            break
        if col:
            break
    sim.env.end()
    return success


def run_episode_rcpg(model, world, seed, max_steps=300):
    random.seed(seed)
    np.random.seed(seed)
    sim = SIM(world_file=world, disable_plotting=True)
    scan, dist, cos, sin, col, goal, a, r = sim.reset(random_obstacles=True)
    success = False
    history = []
    for step in range(max_steps):
        state, _ = model.prepare_state(scan, dist, cos, sin, col, goal, [0.0, 0.0])
        history.append(state)
        if len(history) > 10:
            history.pop(0)
        seq = history + [history[-1]] * (10 - len(history))
        action = model.get_action(np.array(seq), False)
        lin = float(np.clip((action[0] + 1) / 4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan, dist, cos, sin, col, goal, a, r = sim.step(lin, ang)
        if goal:
            success = True
            break
        if col:
            break
    sim.env.end()
    return success


def evaluate_generalization(model_name, run_fn, model, n_trials, base_seed):
    successes = []
    for trial in range(n_trials):
        seed = base_seed + trial  # 每个trial用不同但确定的种子
        success = run_fn(model, 'robot_nav/worlds/robot_world.yaml', seed)
        successes.append(success)
    sr = sum(successes) / n_trials
    # 计算95%置信区间（二项分布的Wilson score interval近似）
    se = np.sqrt(sr * (1 - sr) / n_trials)
    ci_low = max(0, sr - 1.96 * se)
    ci_high = min(1, sr + 1.96 * se)
    return sr, ci_low, ci_high, successes


print(f"增强版泛化测试: N={N_TRIALS} trials, fixed seed={RANDOM_SEED}")
print(f"所有模型面对完全相同的随机场景序列")
print("=" * 70)

all_results = {}

for model_name in CNNTD3_MODELS:
    try:
        model = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                       device=device, load_model=True,
                       model_name=model_name, load_directory=CNNTD3_DIR)
        model.actor.eval()
        sr, ci_low, ci_high, raw = evaluate_generalization(
            model_name, run_episode_cnntd3, model, N_TRIALS, RANDOM_SEED)
        all_results[model_name] = {'sr': sr, 'ci_low': ci_low, 'ci_high': ci_high, 'raw': raw}
        print(f"{model_name:<28}: SR={sr:.0%}  95% CI=[{ci_low:.0%}, {ci_high:.0%}]")
    except Exception as e:
        print(f"跳过 {model_name}（不存在）: {e}")

try:
    rcpg_model = RCPG(state_dim=185, action_dim=2, max_action=1,
                      device=device, load_model=True, rnn='gru')
    sr, ci_low, ci_high, raw = evaluate_generalization(
        'RCPG', run_episode_rcpg, rcpg_model, N_TRIALS, RANDOM_SEED)
    all_results['RCPG'] = {'sr': sr, 'ci_low': ci_low, 'ci_high': ci_high, 'raw': raw}
    print(f"{'RCPG':<28}: SR={sr:.0%}  95% CI=[{ci_low:.0%}, {ci_high:.0%}]")
except Exception as e:
    print(f"跳过 RCPG: {e}")

print("\n" + "=" * 70)
print("最终排名（按SR降序）")
print("=" * 70)
ranked = sorted(all_results.items(), key=lambda x: -x[1]['sr'])
for i, (name, r) in enumerate(ranked, 1):
    print(f"{i}. {name:<26}: SR={r['sr']:.0%}  95% CI=[{r['ci_low']:.0%}, {r['ci_high']:.0%}]")

import json
output = {k: {'sr': v['sr'], 'ci_low': v['ci_low'], 'ci_high': v['ci_high']}
          for k, v in all_results.items()}
with open('generalization_enhanced_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\n结果已保存到 generalization_enhanced_results.json")
