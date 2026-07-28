"""
训练专攻U-trap逃脱的模型（CNNTD3_utrap_specialist）
改进：
1. 更高的探索奖励（0.25 vs 0.15）
2. U-trap场景占比更高（60% vs 35%）
3. 更大的position history窗口（100 vs 50）
4. 更长训练（80 epochs）
5. 保存U-trap评测最好的checkpoint

用法：在AutoDL上跑（本地也行，只是慢）
  screen -S utrap
  cd ~/DRL-robot-navigation-IR-SIM
  python rl_train_utrap_specialist.py
"""
import sys, os
sys.path.insert(0, 'robot_nav')
os.chdir(os.path.expanduser('~/DRL-robot-navigation-IR-SIM'))

from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch, numpy as np
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
from collections import deque
import random


def get_exploration_bonus(curr_pos, position_history, threshold=0.5):
    if len(position_history) < 5:
        return 0.0
    min_dist = min(np.linalg.norm(np.array(curr_pos)-np.array(p)) for p in position_history)
    if min_dist > threshold:
        return 0.25    # 更高奖励鼓励探索
    elif min_dist < 0.1:
        return -0.15   # 更强惩罚防止原地打转
    return 0.0


def get_stall_penalty(stall_count):
    if stall_count > 12:    # 更快触发惩罚
        return -0.4
    elif stall_count > 6:
        return -0.15
    return 0.0


def safe_reset(sim, world_file):
    for attempt in range(3):
        try:
            return sim.reset(), sim
        except Exception as e:
            sim = SIM(world_file=world_file, disable_plotting=True)
    raise RuntimeError("reset failed 3x")


def evaluate_utrap(model, eval_episodes=12):
    """专门评测U-trap，4个朝向×3个位置扰动"""
    configs = []
    thetas = [0.0, 1.57, 3.14, -1.57]
    rng = np.random.default_rng(42)
    for i in range(eval_episodes):
        th = thetas[i%4] + rng.uniform(-0.4, 0.4)
        x = 7.5 + rng.uniform(-0.3, 0.3)
        y = 5.0 + rng.uniform(-0.3, 0.3)
        configs.append([[x],[y],[th]])

    succ = 0
    for rs in configs:
        sim = SIM(world_file="robot_nav/worlds/u_trap_world.yaml", disable_plotting=True)
        scan,d,c,s,col,g,a,r = sim.reset(robot_state=rs, robot_goal=[[9.0],[5.0],[0]], random_obstacles=False)
        prev = [0.0, 0.0]
        for step in range(500):
            state,_ = model.prepare_state(scan,d,c,s,col,g,prev)
            action = model.get_action(np.array(state), False)
            prev = list(action)
            lin = float(np.clip((action[0]+1)/4, 0, 0.5))
            ang = float(np.clip(action[1], -1, 1))
            scan,d,c,s,col,g,a,r = sim.step(lin, ang)
            if g: succ += 1; break
            if col: break
        sim.env.end()
    return succ / eval_episodes


def evaluate_standard(model, n=20):
    """标准环境评测"""
    succ = 0
    for _ in range(n):
        sim = SIM(world_file="robot_nav/worlds/robot_world.yaml", disable_plotting=True)
        scan,d,c,s,col,g,a,r = sim.reset(random_obstacles=True)
        prev = [0.0, 0.0]
        for step in range(300):
            state,_ = model.prepare_state(scan,d,c,s,col,g,prev)
            action = model.get_action(np.array(state), False)
            prev = list(action)
            lin = float(np.clip((action[0]+1)/4, 0, 0.5))
            ang = float(np.clip(action[1], -1, 1))
            scan,d,c,s,col,g,a,r = sim.step(lin, ang)
            if g: succ += 1; break
            if col: break
        sim.env.end()
    return succ / n


def main():
    state_dim, action_dim, max_action = 185, 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 80
    episodes_per_epoch = 70
    train_every_n = 2
    training_iterations = 80
    batch_size = 64
    max_steps = 400     # U-trap需要更多步
    eval_every = 5

    STANDARD_WORLD = "robot_nav/worlds/robot_world.yaml"
    HARD_WORLDS = [
        "robot_nav/worlds/u_trap_world.yaml",
        "robot_nav/worlds/u_trap_world.yaml",      # U-trap权重×2
        "robot_nav/worlds/u_shape_world.yaml",
        "robot_nav/worlds/u_shape_hard_world.yaml",
    ]

    model = CNNTD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                   device=device, save_every=10, load_model=False,
                   model_name="CNNTD3_utrap_specialist")

    current_world = STANDARD_WORLD
    sim = SIM(world_file=current_world, disable_plotting=True)
    replay_buffer = get_buffer(model, sim, False, False, 10, training_iterations, batch_size)
    scan,d,c,s,col,g,a,r = sim.step(0.0, 0.0)

    position_history = deque(maxlen=100)  # 更大窗口
    stall_count = 0
    prev_pos = None
    epoch, episode, steps = 0, 0, 0
    best_utrap_sr = 0.0

    print(f"U-trap specialist training | device={device}")

    while epoch < max_epochs:
        try:
            state, terminal = model.prepare_state(scan,d,c,s,col,g,a)
            action = model.get_action(np.array(state), True)
            a_in = [(action[0]+1)/4, action[1]]
            scan,d,c,s,col,g,a,r = sim.step(a_in[0], a_in[1])
        except Exception as e:
            sim = SIM(world_file=current_world, disable_plotting=True)
            scan,d,c,s,col,g,a,r = sim.reset()
            position_history.clear(); stall_count=0; prev_pos=None; steps=0
            continue

        robot_state = sim.env.get_robot_state()
        curr_pos = (robot_state[0].item(), robot_state[1].item())
        exploration_bonus = get_exploration_bonus(curr_pos, position_history)
        stall_penalty = get_stall_penalty(stall_count)
        improved_reward = r + exploration_bonus + stall_penalty
        position_history.append(curr_pos)
        if prev_pos is not None:
            moved = np.linalg.norm(np.array(curr_pos)-np.array(prev_pos))
            stall_count = stall_count+1 if moved < 0.02 else 0
        prev_pos = curr_pos

        next_state, terminal = model.prepare_state(scan,d,c,s,col,g,a)
        replay_buffer.add(state, action, improved_reward, terminal, next_state)

        if terminal or steps == max_steps:
            last_goal, last_col = g, col
            position_history.clear(); stall_count=0; prev_pos=None

            # U-trap占60%的训练比例
            hard_prob = min(0.6, 0.3 + epoch/max_epochs * 0.3)
            if epoch >= 5 and random.random() < hard_prob:
                new_world = random.choice(HARD_WORLDS)
            else:
                new_world = STANDARD_WORLD
            if new_world != current_world:
                current_world = new_world
                sim = SIM(world_file=current_world, disable_plotting=True)

            try:
                scan,d,c,s,col,g,a,r = sim.reset()
            except:
                sim = SIM(world_file=current_world, disable_plotting=True)
                scan,d,c,s,col,g,a,r = sim.reset()

            outcome = "GOAL" if last_goal else ("COL" if last_col else "timeout")
            if (episode+1) % 10 == 0:
                print(f"Ep {epoch+1}/{max_epochs} | {episode+1}/{episodes_per_epoch} "
                      f"| {outcome} | {current_world.split('/')[-1]}", flush=True)
            episode += 1; steps = 0
            if episode % train_every_n == 0:
                model.train(replay_buffer=replay_buffer, iterations=training_iterations, batch_size=batch_size)
        else:
            steps += 1

        if episode >= episodes_per_epoch:
            episode = 0; epoch += 1
            if epoch % eval_every == 0:
                utrap_sr = evaluate_utrap(model)
                std_sr = evaluate_standard(model)
                print(f"\n  [Eval epoch {epoch}] U-trap={utrap_sr:.0%} Standard={std_sr:.0%}")
                model.writer.add_scalar("eval/utrap_sr", utrap_sr, epoch)
                model.writer.add_scalar("eval/standard_sr", std_sr, epoch)
                if utrap_sr > best_utrap_sr:
                    best_utrap_sr = utrap_sr
                    model.save("CNNTD3_utrap_specialist_best", "models/CNNTD3/checkpoint")
                    print(f"  ★ Best U-trap SR={utrap_sr:.0%} (std={std_sr:.0%})")
            if epoch % 10 == 0:
                model.save(f"CNNTD3_utrap_specialist_epoch_{epoch}", "models/CNNTD3/checkpoint")

    # 最终评测
    print(f"\n=== Final ===")
    print(f"U-trap: {evaluate_utrap(model):.0%}")
    print(f"Standard: {evaluate_standard(model, 50):.0%}")
    print(f"Best U-trap SR during training: {best_utrap_sr:.0%}")


if __name__ == '__main__':
    main()
