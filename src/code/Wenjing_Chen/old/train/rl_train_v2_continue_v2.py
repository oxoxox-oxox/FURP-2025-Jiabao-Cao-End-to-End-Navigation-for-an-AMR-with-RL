"""
v2_continue_v2: 从已有 CNNTD3_v2 checkpoint 继续训练
- 不覆盖原始 CNNTD3_v2 (保存为 CNNTD3_v2_continue_v2_*)
- 加入IR-SIM崩溃容错（重建sim重试）
- 每个epoch后自动生成可视化图表
"""
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def plot_progress(goals, cols, rewards, baseline_sr=0.86):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(goals, 'g-o', markersize=4)
    axes[0].set_title('Success Rate (v2_continue_v2)')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=baseline_sr, color='r', linestyle='--', label=f'original v2 baseline ({baseline_sr:.0%})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cols, 'r-o', markersize=4)
    axes[1].set_title('Collision Rate')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rewards, 'b-o', markersize=4)
    axes[2].set_title('Average Reward')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v2_continue_v2_progress.png', dpi=100)
    plt.close(fig)


def evaluate(model, epoch, eval_world, eval_episodes=10):
    """容错版evaluate：reset/step失败时重建sim重试"""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward, col, goals = 0.0, 0, 0
    eval_sim = SIM(world_file=eval_world, disable_plotting=True)
    for _ in range(eval_episodes):
        count = 0
        try:
            latest_scan, distance, cos, sin, collision, goal, a, reward = eval_sim.reset()
        except Exception as e:
            print(f"  [WARN] eval reset failed, rebuilding sim: {e}")
            eval_sim = SIM(world_file=eval_world, disable_plotting=True)
            try:
                latest_scan, distance, cos, sin, collision, goal, a, reward = eval_sim.reset()
            except Exception:
                continue
        done = False
        while not done and count < 501:
            state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            action = model.get_action(np.array(state), False)
            a_in = [(action[0]+1)/4, action[1]]
            try:
                latest_scan, distance, cos, sin, collision, goal, a, reward = eval_sim.step(a_in[0], a_in[1])
            except Exception as e:
                print(f"  [WARN] eval step failed, rebuilding sim: {e}")
                eval_sim = SIM(world_file=eval_world, disable_plotting=True)
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
    return avg_goal, avg_col, avg_reward


def main():
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nr_eval_episodes = 10
    extra_epochs = 40
    episodes_per_epoch = 70
    train_every_n = 2
    training_iterations = 80
    batch_size = 64
    max_steps = 300
    save_every = 5

    STANDARD_WORLD = "worlds/robot_world.yaml"
    HARD_WORLDS = ["worlds/u_trap_world.yaml", "worlds/u_shape_world.yaml", "worlds/u_shape_hard_world.yaml"]

    # 从原始CNNTD3_v2加载，但保存到全新文件名，不覆盖原始checkpoint
    model = CNNTD3(
        state_dim=state_dim, action_dim=action_dim, max_action=max_action,
        device=device, save_every=save_every,
        load_model=True,
        model_name="CNNTD3_v2",
        load_directory="models/CNNTD3/checkpoint",
    )
    print("已从原始 CNNTD3_v2 checkpoint 加载（只读，不会被覆盖）")
    print("本次训练结果将保存为 CNNTD3_v2_continue_v2_* （全新文件名）")

    current_world = STANDARD_WORLD
    sim = SIM(world_file=current_world, disable_plotting=True)
    replay_buffer = get_buffer(model, sim, False, False, 10, training_iterations, batch_size)

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(0.0, 0.0)

    position_history = deque(maxlen=50)
    stall_count = 0
    prev_pos = None
    epoch, episode, steps = 0, 0, 0
    best_sr = 0.0

    goal_history, col_history, reward_history = [], [], []

    print(f"v2_continue_v2 training start | device={device} | extra_epochs={extra_epochs}")

    while epoch < extra_epochs:
        try:
            state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            action = model.get_action(np.array(state), True)
            a_in = [(action[0]+1)/4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(a_in[0], a_in[1])
        except Exception as e:
            print(f"  [WARN] step failed, rebuilding sim: {e}")
            sim = SIM(world_file=current_world, disable_plotting=True)
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            position_history.clear()
            stall_count = 0
            prev_pos = None
            steps = 0
            continue

        robot_state = sim.env.get_robot_state()
        curr_pos = (robot_state[0].item(), robot_state[1].item())
        exploration_bonus = get_exploration_bonus(curr_pos, position_history)
        stall_penalty = get_stall_penalty(stall_count)
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

            hard_prob = 0.3
            if random.random() < hard_prob:
                new_world = random.choice(HARD_WORLDS)
            else:
                new_world = STANDARD_WORLD
            if new_world != current_world:
                current_world = new_world
                sim = SIM(world_file=current_world, disable_plotting=True)

            try:
                latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            except Exception as e:
                print(f"  [WARN] reset failed, rebuilding sim: {e}")
                sim = SIM(world_file=current_world, disable_plotting=True)
                latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()

            outcome = "GOAL" if last_goal else ("COL" if last_col else "timeout")
            print(f"Epoch {epoch+1}/{extra_epochs} | Ep {episode+1}/{episodes_per_epoch} | {outcome} | {current_world.split('/')[-1]}", flush=True)
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
                sr, cr, rw = evaluate(model, epoch, STANDARD_WORLD, nr_eval_episodes)
            except Exception as e:
                print(f"  [WARN] evaluate failed entirely, skip: {e}")
                sr, cr, rw = best_sr, 0, 0

            goal_history.append(sr)
            col_history.append(cr)
            reward_history.append(rw)
            plot_progress(goal_history, col_history, reward_history)
            print(f"  [图表已更新] v2_continue_v2_progress.png ({len(goal_history)} epochs)")

            if epoch % save_every == 0:
                model.save(f"CNNTD3_v2_continue_v2_epoch_{epoch}", "models/CNNTD3/checkpoint")
            if sr > best_sr:
                best_sr = sr
                model.save("CNNTD3_v2_continue_v2_best", "models/CNNTD3/checkpoint")
                print(f"Best saved SR={sr:.0%}")

    print("Training finished")
    print(f"原始 CNNTD3_v2 checkpoint 未被修改，仍可独立使用")


if __name__ == '__main__':
    main()
