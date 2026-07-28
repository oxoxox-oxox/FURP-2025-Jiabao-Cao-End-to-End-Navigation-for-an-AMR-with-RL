"""
U形走廊陷阱场景测试脚本
用法（在项目根目录 ~/DRL-robot-navigation-IR-SIM 下运行）：

  conda activate neupan
  cd ~/DRL-robot-navigation-IR-SIM
  python test_u_trap_cnntd3.py

结果会打印 SR / 碰撞率，并把每个 episode 的结果写到 u_trap_results.csv
"""

import sys
import os
import numpy as np
import torch
import csv

# 把 robot_nav 加入路径（和 rl_test.py 保持一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_nav.models.TD3.TD3 import TD3
from robot_nav.SIM_ENV.sim import SIM

# ─── 配置 ────────────────────────────────────────────────────────────────────

WORLD_FILE  = "robot_nav/worlds/u_trap_world.yaml"  # world 文件路径
MAX_STEPS   = 300   # 每个 episode 最大步数
MAX_EPISODES = 20   # 测试 episode 数量（可以改大）

# U形陷阱：agent 在凹槽内，goal 在开口外
# 可以多设几个起点变体，覆盖不同初始朝向
TEST_CASES = [
    # (robot_state,               robot_goal)
    # state 格式: [[x],[y],[theta],[speed]]   goal: [[x],[y],[0]]
    ([[2.5],[5.0],[0.0],[0]],    [[8.0],[5.0],[0]]),   # 朝右，正对开口
    ([[2.5],[5.0],[1.57],[0]],   [[8.0],[5.0],[0]]),   # 朝上，需要转向
    ([[2.5],[5.0],[3.14],[0]],   [[8.0],[5.0],[0]]),   # 朝左（背对开口，最难）
    ([[2.5],[5.0],[-1.57],[0]],  [[8.0],[5.0],[0]]),   # 朝下
    ([[3.0],[6.0],[0.0],[0]],    [[8.0],[5.0],[0]]),   # 偏上起点
    ([[3.0],[4.0],[0.0],[0]],    [[8.0],[5.0],[0]]),   # 偏下起点
    ([[3.5],[5.0],[3.14],[0]],   [[8.0],[5.0],[0]]),   # 更靠里，背对开口
    ([[2.5],[5.5],[1.57],[0]],   [[8.0],[4.0],[0]]),   # 目标偏下
    ([[2.5],[4.5],[-1.57],[0]],  [[8.0],[6.0],[0]]),   # 目标偏上
    ([[3.0],[5.0],[2.0],[0]],    [[8.0],[5.0],[0]]),   # 斜朝向
]

# ─── 初始化 ──────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = TD3(
    state_dim=25,
    action_dim=2,
    max_action=1,
    device=device,
    load_model=True,
    model_name="TD3",
)

sim = SIM(world_file=WORLD_FILE)

# ─── 测试循环 ────────────────────────────────────────────────────────────────

results = []
total_goal = 0
total_col  = 0

print("=" * 50)
print(f"测试 {len(TEST_CASES)} 个场景，每场景重复 {MAX_EPISODES // len(TEST_CASES) + 1} 次")
print("=" * 50)

episode_id = 0
for case_idx, (robot_state, robot_goal) in enumerate(TEST_CASES):
    repeats = max(1, MAX_EPISODES // len(TEST_CASES))
    for rep in range(repeats):
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
            robot_state=robot_state,
            robot_goal=robot_goal,
            random_obstacles=False,   # 障碍物固定，只测 U 形墙
        )

        done  = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < MAX_STEPS:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]

            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            ep_reward += reward
            steps += 1
            done = collision or goal

        # 判断结果
        outcome = "goal" if goal else ("collision" if collision else "timeout")
        total_goal += int(goal)
        total_col  += int(collision)

        theta = robot_state[2][0]
        print(f"Episode {episode_id+1:>3} | case={case_idx} rep={rep} | "
              f"theta={theta:.2f} | steps={steps:>3} | {outcome}")

        results.append({
            "episode": episode_id + 1,
            "case_idx": case_idx,
            "init_theta": theta,
            "steps": steps,
            "outcome": outcome,
            "reward": round(ep_reward, 2),
        })
        episode_id += 1

# ─── 汇总 ────────────────────────────────────────────────────────────────────

n = len(results)
sr  = total_goal / n * 100
cr  = total_col  / n * 100
tor = (n - total_goal - total_col) / n * 100

print("\n" + "=" * 50)
print(f"共测试 {n} 个 episode")
print(f"  成功率  (SR) : {sr:.1f}%  ({total_goal}/{n})")
print(f"  碰撞率  (CR) : {cr:.1f}%  ({total_col}/{n})")
print(f"  超时率  (TR) : {tor:.1f}%  ({n-total_goal-total_col}/{n})")
print("=" * 50)

# 保存 CSV
csv_path = "u_trap_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\n详细结果已保存到 {csv_path}")
