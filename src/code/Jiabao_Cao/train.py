"""
RCPG (Recurrent + CNN + TD3) 在 ir-sim 上的训练脚本。

通过 1D CNN 处理每帧 LiDAR 距离数据, LSTM 对 hist_n 帧进行时序建模。
使用 TD3 双 Q 网络 (Q1, Q2), min-target, 目标策略噪声, 延迟策略更新。

用法:
    python train.py                          # 从头训练
    python train.py --resume 500037          # 从 checkpoint 续训
"""

import argparse
import os
import re
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from env import IrSimNavEnv
from CNNTD3 import CNNRC
from replay_buffer_td3 import ReplayBuffer


# ===================== 配置 =====================

# 分阶段训练: (YAML, 步数)
TRAIN_PHASES = [
    ('./env/env.yaml',   500_000)
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 环境
MAX_STEPS_PER_EPISODE = 1500

# RCPG (Recurrent + CNN + TD3) 超参数
SINGLE_OBS_DIM = 89        # 84 LiDAR + 5 (dist, cos, sin, v_enc, w_enc)
HIST_N = 3                 # 堆叠帧数 (temporal sequence length)
STATE_DIM = SINGLE_OBS_DIM * HIST_N  # = 267
ACTION_DIM = 2
MAX_ACTION = 1.0            # Tanh 输出范围
LR = 1e-4
GAMMA = 0.99
TAU = 0.005
POLICY_FREQ = 2             # 延迟策略更新
POLICY_NOISE = 0.2          # 目标策略噪声标准差
NOISE_CLIP = 0.5            # 噪声裁剪范围
BATCH_SIZE = 256
BUFFER_CAPACITY = 200_000
WARMUP_STEPS = 10_000        # 训练前的随机探索步数
COLLECT_STEPS = 1000         # 每次训练前收集的步数
TRAIN_ITERATIONS = 500       # 每次训练的梯度步数
TOTAL_TIMESTEPS = sum(steps for _, steps in TRAIN_PHASES)

# 日志与保存
SAVE_EVERY = 50_000          # 检查点间隔步数
MODEL_DIR = './rcpg_models/'
MODEL_NAME = 'RCPG'


# ===================== 自动递增实验目录 ====================

def next_experiment_dir(base_dir='./tb_logs'):
    """在 base_dir 内找到下一个可用的 experiment_N/ 目录。"""
    os.makedirs(base_dir, exist_ok=True)
    existing = []
    pattern = re.compile(r'^experiment_(\d+)$')
    for name in os.listdir(base_dir):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(base_dir, name)):
            existing.append(int(m.group(1)))
    n = max(existing) + 1 if existing else 1
    exp_dir = os.path.join(base_dir, f'experiment_{n}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def main():
    # ===================== 参数解析 =====================
    parser = argparse.ArgumentParser(description='RCPG 训练脚本')
    parser.add_argument('--resume', type=int, default=0,
                        help='从指定 step 的 checkpoint 续训 (例: --resume 500037)')
    args = parser.parse_args()

    tb_dir = next_experiment_dir()
    print(f"[实验] TensorBoard 日志目录: {tb_dir}")

    # ===================== 训练 ========================

    # ---- 确定起始 phase ----
    if args.resume > 0:
        total_steps = args.resume
        # 计算当前应处的 phase
        current_phase = 0
        accumulated = 0
        for i, (_, phase_steps) in enumerate(TRAIN_PHASES):
            if total_steps >= accumulated + phase_steps:
                accumulated += phase_steps
                current_phase = i + 1
            else:
                break
        if current_phase >= len(TRAIN_PHASES):
            print(f"[RCPG] resume step {total_steps} 已超出总训练步数 {TOTAL_TIMESTEPS}")
            return
        # phase_step_target = 当前 phase 的结束步数
        phase_step_target = sum(s for _, s in TRAIN_PHASES[:current_phase + 1])
        yaml_file = TRAIN_PHASES[current_phase][0]
        print(f"[RCPG] 续训模式: 从 step {total_steps} 恢复, 阶段 {current_phase}: {yaml_file}")
    else:
        total_steps = 0
        current_phase = 0
        phase_step_target = TRAIN_PHASES[0][1]
        yaml_file = TRAIN_PHASES[0][0]
        print(f"[RCPG] 阶段 0: {yaml_file} ({TRAIN_PHASES[0][1]} 步)")

    env = IrSimNavEnv(
        yaml_file=yaml_file,
        render_mode=None,
        hist_n=HIST_N,
    )

    # 创建 RCPG 智能体 (Recurrent + CNN + TD3)
    agent = CNNRC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        device=torch.device(DEVICE),
        lr=LR,
        hist_n=HIST_N,
        save_every=0,             # 手动保存检查点
        model_name=MODEL_NAME,
        save_directory=Path(MODEL_DIR),
    )

    # ---- 续训: 加载模型权重 ----
    if args.resume > 0:
        load_name = f'{MODEL_NAME}_step{args.resume}'
        agent.load(filename=load_name, directory=Path(MODEL_DIR))
        print(f"[RCPG] 已从 {MODEL_DIR}/{load_name}_*.pth 加载权重")

    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(buffer_size=BUFFER_CAPACITY)

    # TensorBoard writer 用于 episode 级别的指标
    writer = SummaryWriter(tb_dir)

    # 创建模型目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 训练状态
    episode = 0
    episode_rewards = []
    success_history = []
    best_mean_reward = -float('inf')

    warmup_effective = 0 if args.resume > 0 else WARMUP_STEPS
    print(f"[RCPG] 开始训练: {TOTAL_TIMESTEPS} 步, "
          f"device={DEVICE}, buffer={BUFFER_CAPACITY}, warmup={warmup_effective}")
    print(f"[RCPG] State dim={STATE_DIM} (hist_n={HIST_N} × {SINGLE_OBS_DIM}), "
          f"Action dim={ACTION_DIM}, LiDAR 射线数={SINGLE_OBS_DIM - 5}")
    if args.resume > 0:
        print(f"[RCPG] 续训起点: step={total_steps}, 跳过 warmup, buffer 重新收集中...")
    t_start = time.time()

    obs, _ = env.reset()

    while total_steps < TOTAL_TIMESTEPS:
        # ---- 阶段切换 ----
        if (current_phase + 1 < len(TRAIN_PHASES)
                and total_steps >= phase_step_target):
            current_phase += 1
            yaml_file, steps = TRAIN_PHASES[current_phase]
            phase_step_target += steps
            env.close()
            env = IrSimNavEnv(yaml_file=yaml_file, render_mode=None, hist_n=HIST_N)
            obs, _ = env.reset()   # 新环境必须 reset 以初始化帧缓冲区
            print(f"\n[RCPG] >>> 切换到阶段 {current_phase}: "
                  f"{yaml_file} ({steps} 步) at total_steps={total_steps}\n")

        # ========== 收集阶段 (CPU-bound) ==========
        collect_start = total_steps
        while (total_steps - collect_start < COLLECT_STEPS
               and total_steps < TOTAL_TIMESTEPS):
            episode_reward = 0.0
            episode_steps = 0

            for _ in range(MAX_STEPS_PER_EPISODE):
                # ---- 选择动作 ----
                if total_steps < WARMUP_STEPS:
                    action = np.random.uniform(-1.0, 1.0,
                                               size=ACTION_DIM).astype(np.float32)
                else:
                    action = agent.get_action(obs, add_noise=True)

                # ---- 映射动作: RCPG [-1,1] → env [v:0~1, w:-1~1] ----
                v_env = (float(action[0]) + 1.0) / 2.0
                w_env = float(action[1])
                env_action = np.array([v_env, w_env], dtype=np.float32)

                # ---- 执行环境步 ----
                next_obs, reward, done, truncated, info = env.step(env_action)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # ---- 存储转换 ----
                terminal = float(done or truncated)
                replay_buffer.add(obs, action, reward, terminal, next_obs)

                obs = next_obs

                if done or truncated:
                    break

            # ---- Episode 结束 ----
            episode += 1
            episode_rewards.append(episode_reward)
            success_history.append(1 if info.get('result') == 'success' else 0)

            # 日志记录
            if episode % 10 == 0:
                recent_10 = episode_rewards[-10:]
                mean_reward = np.mean(recent_10)
                recent_successes = success_history[-50:] if len(success_history) >= 50 else success_history
                success_rate = np.mean(recent_successes) if recent_successes else 0.0
                elapsed = time.time() - t_start
                fps = total_steps / max(elapsed, 0.001)

                result = info.get('result', 'timeout')
                print(f"[RCPG] ep={episode:5d} | steps={total_steps:8d} | "
                      f"rew={episode_reward:8.2f} | mean10={mean_reward:8.2f} | "
                      f"succ%={success_rate:6.1%} | "
                      f"fps={fps:6.0f} | result={result} | "
                      f"buf={replay_buffer.size():6d}")

                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/mean_reward_10', mean_reward, episode)
                writer.add_scalar('episode/success_rate_50', success_rate, episode)
                writer.add_scalar('episode/steps', episode_steps, episode)
                writer.add_scalar('train/total_steps', total_steps, episode)
                writer.add_scalar('train/buffer_size', replay_buffer.size(), episode)
                writer.add_scalar('train/fps', fps, episode)

            # ---- 定期保存检查点 ----
            if total_steps % SAVE_EVERY < episode_steps:
                agent.save(
                    filename=f'{MODEL_NAME}_step{total_steps}',
                    directory=Path(MODEL_DIR)
                )
                recent_10 = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                mean_reward = np.mean(recent_10)
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    agent.save(filename='best_model', directory=Path(MODEL_DIR))
                    print(f"[RCPG] 新最佳模型已保存 (mean10={best_mean_reward:.2f})")

            # 重置以进行下一个 episode
            obs, _ = env.reset()

        # ========== 训练阶段 (GPU burst) ==========
        if (total_steps >= WARMUP_STEPS
                and replay_buffer.size() >= BATCH_SIZE):
            for _ in range(TRAIN_ITERATIONS):
                agent.train(
                    replay_buffer=replay_buffer,
                    iterations=1,
                    batch_size=BATCH_SIZE,
                    discount=GAMMA,
                    tau=TAU,
                    policy_noise=POLICY_NOISE,
                    noise_clip=NOISE_CLIP,
                    policy_freq=POLICY_FREQ,
                )

    # ===================== 训练完成 =====================

    elapsed = time.time() - t_start
    print(f"\n[RCPG] 训练完成: {total_steps} 步, 耗时 "
          f"{elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ---- 最终总结 ----
    if success_history:
        total_episodes = len(success_history)
        total_successes = sum(success_history)
        overall_rate = total_successes / total_episodes
        recent_50_rate = (sum(success_history[-50:]) / min(50, total_episodes)
                          if total_episodes >= 10 else overall_rate)
        avg_reward = np.mean(episode_rewards)
        print(f"\n=== 训练总结 ===")
        print(f"总 episode 数:     {total_episodes}")
        print(f"总步数:            {total_steps}")
        print(f"总体成功率:        {overall_rate:.2%} ({total_successes}/{total_episodes})")
        print(f"最近 50-episode:   {recent_50_rate:.2%}")
        print(f"平均奖励:          {avg_reward:.2f}")
        print(f"最佳 mean10 奖励:  {best_mean_reward:.2f}")

    # 保存最终模型
    agent.save(filename='final_model', directory=Path(MODEL_DIR))
    print(f"\n最终模型已保存至 {MODEL_DIR}/final_model_*.pth")

    writer.close()
    agent.writer.close()
    env.close()

    print("\n运行 'python eval.py' 来可视化训练好的策略。")


if __name__ == '__main__':
    main()
