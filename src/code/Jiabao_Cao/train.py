"""
Training script for CNNTD3 (TD3 with CNN for LiDAR) on ir-sim.

CNNTD3 processes raw LiDAR range data through 1D CNN layers.
Uses TD3 algorithm: off-policy, twin critics, delayed policy updates.

Usage:
    python train.py
"""

import os
import re
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from env import IrSimNavEnv
from CNNTD3 import CNNTD3
from replay_buffer_td3 import ReplayBuffer


# ===================== Config =====================

# 分阶段训练: (YAML, 步数)
TRAIN_PHASES = [
    ('./env/env.yaml',              500_000),
    ('./env/env_convex_td3.yaml',   100_000),
    ('./env/env_corridor_td3.yaml', 200_000),
    ('./env/env_convex_td3.yaml',   200_000),
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Environment
MAX_STEPS_PER_EPISODE = 1500

# TD3 hyperparameters
STATE_DIM = 89            # 84 ranges + 5 (dist, cos, sin, v_enc, w_enc)
ACTION_DIM = 2
MAX_ACTION = 1.0          # Tanh output range
LR = 1e-4
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2           # delayed policy updates
BATCH_SIZE = 256
BUFFER_CAPACITY = 200_000
WARMUP_STEPS = 10_000      # random exploration before training
COLLECT_STEPS = 1000       # steps to collect before each training burst
TRAIN_ITERATIONS = 500     # gradient steps per training burst
TOTAL_TIMESTEPS = sum(steps for _, steps in TRAIN_PHASES)

# Logging & saving
SAVE_EVERY = 50_000        # timesteps between checkpoints
MODEL_DIR = './td3_models/'
MODEL_NAME = 'CNNTD3'


# ===================== Auto-increment experiment folder ====================

def next_experiment_dir(base_dir='./tb_logs'):
    """Find the next available experiment_N/ directory inside base_dir."""
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
    tb_dir = next_experiment_dir()
    print(f"[Experiment] TensorBoard log dir: {tb_dir}")

    # ===================== Training ========================

    # Create initial environment (phase 0)
    current_phase = 0
    phase_step_target = TRAIN_PHASES[0][1]
    env = IrSimNavEnv(
        yaml_file=TRAIN_PHASES[0][0],
        render_mode=None,
    )
    print(f"[TD3] Phase 0: {TRAIN_PHASES[0][0]} ({TRAIN_PHASES[0][1]} steps)")

    # Create CNNTD3 agent
    agent = CNNTD3(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        device=torch.device(DEVICE),
        lr=LR,
        save_every=0,             # manual checkpointing
        model_name=MODEL_NAME,
        save_directory=Path(MODEL_DIR),
        use_max_bound=False,      # optional feature, disabled
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size=BUFFER_CAPACITY)

    # TensorBoard writer for episode-level metrics
    writer = SummaryWriter(tb_dir)

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Training state
    total_steps = 0
    episode = 0
    episode_rewards = []
    success_history = []
    best_mean_reward = -float('inf')

    print(f"[TD3] Starting training: {TOTAL_TIMESTEPS} timesteps, "
          f"device={DEVICE}, buffer={BUFFER_CAPACITY}, warmup={WARMUP_STEPS}")
    print(f"[TD3] State dim={STATE_DIM}, Action dim={ACTION_DIM}, "
          f"LiDAR rays={STATE_DIM - 5}")
    t_start = time.time()

    obs, _ = env.reset()

    while total_steps < TOTAL_TIMESTEPS:
        # ---- Phase switch ----
        if (current_phase + 1 < len(TRAIN_PHASES)
                and total_steps >= phase_step_target):
            current_phase += 1
            yaml_file, steps = TRAIN_PHASES[current_phase]
            phase_step_target += steps
            env.close()
            env = IrSimNavEnv(yaml_file=yaml_file, render_mode=None)
            print(f"\n[TD3] >>> Switching to phase {current_phase}: "
                  f"{yaml_file} ({steps} steps) at total_steps={total_steps}\n")

        # ========== Collect phase (CPU-bound) ==========
        collect_start = total_steps
        while (total_steps - collect_start < COLLECT_STEPS
               and total_steps < TOTAL_TIMESTEPS):
            episode_reward = 0.0
            episode_steps = 0

            for _ in range(MAX_STEPS_PER_EPISODE):
                # ---- Select action ----
                if total_steps < WARMUP_STEPS:
                    action_td3 = np.random.uniform(-1.0, 1.0,
                                                   size=ACTION_DIM).astype(np.float32)
                else:
                    action_td3 = agent.get_action(obs, add_noise=True)

                # ---- Map action: TD3 [-1,1] → env [v:0~1, w:-1~1] ----
                v_env = (float(action_td3[0]) + 1.0) / 2.0
                w_env = float(action_td3[1])
                env_action = np.array([v_env, w_env], dtype=np.float32)

                # ---- Step environment ----
                next_obs, reward, done, truncated, info = env.step(env_action)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # ---- Store transition ----
                terminal = float(done or truncated)
                replay_buffer.add(obs, action_td3, reward, terminal, next_obs)

                obs = next_obs

                if done or truncated:
                    break

            # ---- End of episode ----
            episode += 1
            episode_rewards.append(episode_reward)
            success_history.append(1 if info.get('result') == 'success' else 0)

            # Logging
            if episode % 10 == 0:
                recent_10 = episode_rewards[-10:]
                mean_reward = np.mean(recent_10)
                recent_successes = success_history[-50:] if len(success_history) >= 50 else success_history
                success_rate = np.mean(recent_successes) if recent_successes else 0.0
                elapsed = time.time() - t_start
                fps = total_steps / max(elapsed, 0.001)

                result = info.get('result', 'timeout')
                print(f"[TD3] ep={episode:5d} | steps={total_steps:8d} | "
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

            # ---- Periodic checkpoint ----
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
                    print(f"[TD3] New best model saved (mean10={best_mean_reward:.2f})")

            # Reset for next episode
            obs, _ = env.reset()

        # ========== Train phase (GPU burst) ==========
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

    # ===================== Training complete =====================

    elapsed = time.time() - t_start
    print(f"\n[TD3] Training complete: {total_steps} steps in "
          f"{elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ---- Final summary ----
    if success_history:
        total_episodes = len(success_history)
        total_successes = sum(success_history)
        overall_rate = total_successes / total_episodes
        recent_50_rate = (sum(success_history[-50:]) / min(50, total_episodes)
                          if total_episodes >= 10 else overall_rate)
        avg_reward = np.mean(episode_rewards)
        print(f"\n=== Training Summary ===")
        print(f"Total episodes:     {total_episodes}")
        print(f"Total timesteps:    {total_steps}")
        print(f"Overall success:    {overall_rate:.2%} ({total_successes}/{total_episodes})")
        print(f"Recent 50-episode:  {recent_50_rate:.2%}")
        print(f"Average reward:     {avg_reward:.2f}")
        print(f"Best mean10 reward: {best_mean_reward:.2f}")

    # Save final model
    agent.save(filename='final_model', directory=Path(MODEL_DIR))
    print(f"\nFinal model saved to {MODEL_DIR}/final_model_*.pth")

    writer.close()
    agent.writer.close()
    env.close()

    print("\nRun 'python eval.py' to visualize the trained policy.")


if __name__ == '__main__':
    main()
