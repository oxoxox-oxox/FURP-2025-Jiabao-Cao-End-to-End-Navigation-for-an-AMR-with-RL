"""
改进版 CNNTD3 训练脚本
改进1：探索奖励（惩罚重复访问同一区域）
改进2：课程学习（逐步加入U形陷阱场景）
"""
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
from collections import deque
import random


def get_exploration_bonus(curr_pos, position_history, threshold=0.5):
    """如果当前位置距离历史轨迹超过threshold，给正奖励；否则给惩罚"""
    if len(position_history) < 5:
        return 0.0
    min_dist = min(
        np.linalg.norm(np.array(curr_pos) - np.array(p))
        for p in position_history
    )
    if min_dist > threshold:
        return 0.3   # 探索新区域奖励
    elif min_dist < 0.1:
        return -0.2  # 原地打转惩罚
    return 0.0


def get_stall_penalty(stall_count):
    """连续多步几乎不动，给惩罚"""
    if stall_count > 15:
        return -0.5
    elif stall_count > 8:
        return -0.2
    return 0.0


def main(args=None):
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nr_eval_episodes = 10
    max_epochs = 60
    epoch = 0
    episodes_per_epoch = 70
    episode = 0
    train_every_n = 2
    training_iterations = 80
    batch_size = 64
    max_steps = 300
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10
    save_every = 5

    # ── 课程学习：定义训练世界 ──
    STANDARD_WORLD = "worlds/robot_world.yaml"
    HARD_WORLDS = [
        "worlds/u_trap_world.yaml",
        "worlds/u_shape_world.yaml",
        "worlds/u_shape_hard_world.yaml",
    ]

    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="CNNTD3_improved",  # 新模型名，不覆盖原始checkpoint
    )

    # 初始用标准环境
    current_world = STANDARD_WORLD
    sim = SIM(world_file=current_world, disable_plotting=True)

    replay_buffer = get_buffer(
        model, sim, load_saved_buffer, pretrain,
        pretraining_iterations, training_iterations, batch_size,
    )

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        lin_velocity=0.0, ang_velocity=0.0
    )

    # ── 探索奖励相关变量 ──
    position_history = deque(maxlen=50)
    stall_count = 0
    prev_pos = None

    print(f"开始训练 CNNTD3_improved (课程学习 + 探索奖励)")
    print(f"设备: {device}")

    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        action = model.get_action(np.array(state), True)
        a_in = [(action[0] + 1) / 4, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )

        # ── 获取当前位置，计算探索奖励 ──
        robot_state = sim.env.get_robot_state()
        curr_pos = (robot_state[0].item(), robot_state[1].item())

        exploration_bonus = get_exploration_bonus(curr_pos, position_history)
        stall_penalty = get_stall_penalty(stall_count)
        improved_reward = reward + exploration_bonus + stall_penalty

        # 更新位置历史和停滞计数
        position_history.append(curr_pos)
        if prev_pos is not None:
            moved = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            stall_count = stall_count + 1 if moved < 0.02 else 0
        prev_pos = curr_pos

        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        # 用改进后的奖励存入 replay buffer
        replay_buffer.add(state, action, improved_reward, terminal, next_state)

        if terminal or steps == max_steps:
            # 重置探索历史
            position_history.clear()
            stall_count = 0
            prev_pos = None

            # ── 课程学习：根据 epoch 决定用哪个世界 ──
            hard_prob = min(0.5, epoch / max_epochs)  # 从0逐步增加到0.5
            if epoch >= 10 and random.random() < hard_prob:
                # 用困难场景
                new_world = random.choice(HARD_WORLDS)
            else:
                new_world = STANDARD_WORLD

            # 如果世界变了，重新初始化 SIM
            if new_world != current_world:
                current_world = new_world
                sim = SIM(world_file=current_world, disable_plotting=True)

            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            episode += 1

            # 进度打印
            outcome = "GOAL" if goal else ("COLLISION" if collision else "timeout")
            print(f"Epoch {epoch+1}/{max_epochs} | Ep {episode}/{episodes_per_epoch} | "
                  f"Steps {steps} | {outcome} | World: {current_world.split('/')[-1]}",
                  flush=True)

            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )
            steps = 0
        else:
            steps += 1

        if (episode + 1) % episodes_per_epoch == 0:
            episode = 0
            epoch += 1
            # 评估始终用标准环境（公平对比）
            eval_sim = SIM(world_file=STANDARD_WORLD, disable_plotting=True)
            evaluate(model, epoch, eval_sim, eval_episodes=nr_eval_episodes)


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
