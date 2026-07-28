"""
v6_anneal: 探索奖励退火版
- 前 anneal_start epoch: 探索奖励全开（学逃脱）
- anneal_start ~ max_epochs: 线性衰减到0（恢复标准性能）
- 其余同 v5_combined_v2
"""
import sys
sys.path.insert(0, 'robot_nav')
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
from collections import deque
import random


def get_exploration_bonus(curr_pos, position_history, threshold=0.5):
    if len(position_history) < 5:
        return 0.0
    min_dist = min(np.linalg.norm(np.array(curr_pos) - np.array(p)) for p in position_history)
    if min_dist > threshold:
        return 0.15
    elif min_dist < 0.1:
        return -0.1
    return 0.0


def get_stall_penalty(stall_count):
    if stall_count > 15:
        return -0.3
    elif stall_count > 8:
        return -0.1
    return 0.0


def safe_reset(sim, world_file):
    for attempt in range(3):
        try:
            return sim.reset(), sim
        except (IndexError, Exception) as e:
            print(f"  [WARN] reset失败(尝试{attempt+1}/3): {e}")
            sim = SIM(world_file=world_file, disable_plotting=True)
    raise RuntimeError("reset连续失败3次")


def evaluate(model, epoch, eval_sim, eval_world, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward, col, goals = 0.0, 0, 0
    for _ in range(eval_episodes):
        count = 0
        try:
            (latest_scan, distance, cos, sin, collision, goal, a, reward), eval_sim = \
                safe_reset(eval_sim, eval_world)
        except RuntimeError:
            continue
        done = False
        while not done and count < 501:
            state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            action = model.get_action(np.array(state), False)
            a_in = [(action[0]+1)/4, action[1]]
            try:
                latest_scan, distance, cos, sin, collision, goal, a, reward = eval_sim.step(a_in[0], a_in[1])
            except Exception:
                break
            avg_reward += reward
            count += 1
            if collision: col += 1
            if goal: goals += 1
            done = collision or goal
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Collision rate: {avg_col:.2f}")
    print(f"Average Goal rate: {avg_goal:.2f}")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    return avg_goal, eval_sim


def main():
    state_dim, action_dim, max_action = 185, 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 120          # 更长训练，给退火足够时间
    episodes_per_epoch = 70
    train_every_n = 2
    training_iterations = 80
    batch_size = 64
    max_steps = 300
    save_every = 10

    # === 退火参数 ===
    anneal_start = 50         # 前50epoch探索奖励全开
    anneal_end = max_epochs   # 到120epoch衰减到0

    STANDARD_WORLD = "robot_nav/worlds/robot_world.yaml"
    HARD_WORLDS = ["robot_nav/worlds/u_trap_world.yaml",
                   "robot_nav/worlds/u_shape_world.yaml",
                   "robot_nav/worlds/u_shape_hard_world.yaml"]

    model = CNNTD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                   device=device, save_every=save_every, load_model=False,
                   model_name="CNNTD3_v6_anneal")

    current_world = STANDARD_WORLD
    sim = SIM(world_file=current_world, disable_plotting=True)
    eval_sim = SIM(world_file=STANDARD_WORLD, disable_plotting=True)
    replay_buffer = get_buffer(model, sim, False, False, 10, training_iterations, batch_size)

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(0.0, 0.0)

    position_history = deque(maxlen=50)
    stall_count = 0
    prev_pos = None
    epoch, episode, steps = 0, 0, 0
    best_sr = 0.0

    print(f"v6_anneal training | device={device}")
    print(f"Anneal schedule: full reward epoch 0-{anneal_start}, "
          f"decay to 0 by epoch {anneal_end}")

    while epoch < max_epochs:
        try:
            state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            action = model.get_action(np.array(state), True)
            a_in = [(action[0]+1)/4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(a_in[0], a_in[1])
        except Exception as e:
            print(f"  [WARN] step失败，重建sim: {e}")
            sim = SIM(world_file=current_world, disable_plotting=True)
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            position_history.clear()
            stall_count = 0
            prev_pos = None
            steps = 0
            continue

        robot_state = sim.env.get_robot_state()
        curr_pos = (robot_state[0].item(), robot_state[1].item())

        # === 退火核心：探索奖励随epoch衰减 ===
        if epoch < anneal_start:
            anneal_coeff = 1.0
        elif epoch < anneal_end:
            anneal_coeff = 1.0 - (epoch - anneal_start) / (anneal_end - anneal_start)
        else:
            anneal_coeff = 0.0

        exploration_bonus = anneal_coeff * get_exploration_bonus(curr_pos, position_history)
        stall_penalty = anneal_coeff * get_stall_penalty(stall_count)  # stall penalty也退火
        improved_reward = reward + exploration_bonus + stall_penalty

        position_history.append(curr_pos)
        if prev_pos is not None:
            moved = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            stall_count = stall_count + 1 if moved < 0.02 else 0
        prev_pos = curr_pos

        next_state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        replay_buffer.add(state, action, improved_reward, terminal, next_state)

        if terminal or steps == max_steps:
            last_goal, last_col = goal, collision
            position_history.clear()
            stall_count = 0
            prev_pos = None

            hard_prob = min(0.35, epoch / max_epochs)
            if epoch >= 10 and random.random() < hard_prob:
                new_world = random.choice(HARD_WORLDS)
            else:
                new_world = STANDARD_WORLD
            if new_world != current_world:
                current_world = new_world
                sim = SIM(world_file=current_world, disable_plotting=True)

            try:
                latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            except Exception as e:
                print(f"  [WARN] reset失败，重建sim: {e}")
                sim = SIM(world_file=current_world, disable_plotting=True)
                latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()

            outcome = "GOAL" if last_goal else ("COL" if last_col else "timeout")
            print(f"Epoch {epoch+1}/{max_epochs} | Ep {episode+1}/{episodes_per_epoch} "
                  f"| {outcome} | anneal={anneal_coeff:.2f} | {current_world.split('/')[-1]}",
                  flush=True)
            episode += 1
            steps = 0
            if episode % train_every_n == 0:
                model.train(replay_buffer=replay_buffer, iterations=training_iterations, batch_size=batch_size)
        else:
            steps += 1

        if episode >= episodes_per_epoch:
            episode = 0
            epoch += 1
            try:
                sr, eval_sim = evaluate(model, epoch, eval_sim, STANDARD_WORLD, 10)
            except Exception as e:
                print(f"  [WARN] evaluate失败，跳过本次: {e}")
                sr = best_sr
                eval_sim = SIM(world_file=STANDARD_WORLD, disable_plotting=True)

            # 记录退火系数
            model.writer.add_scalar("train/anneal_coeff", anneal_coeff, epoch)

            if epoch % save_every == 0:
                model.save(f"CNNTD3_v6_anneal_epoch_{epoch}", "models/CNNTD3/checkpoint")
            if sr > best_sr:
                best_sr = sr
                model.save("CNNTD3_v6_anneal_best", "models/CNNTD3/checkpoint")
                print(f"★ Best saved SR={sr:.0%}")

    print(f"Training finished. Best SR={best_sr:.0%}")


if __name__ == '__main__':
    main()
