"""
对比实验: RCPG (with LSTM) vs Standard TD3 (no LSTM).

验证 LSTM 时序建模的真实增益 — 在修复 Critic 中 .data 导致的梯度断流 bug 后,
重新评估 RCPG 相对标准 TD3 的提升幅度。

用法:
    python compare_rcpg_td3.py [--steps 50000] [--seed 42]
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from env import IrSimNavEnv
from CNNTD3 import CNNRC          # RCPG (with LSTM)
from replay_buffer_td3 import ReplayBuffer


# ==================== Standard TD3 (no LSTM) ====================

class TD3Actor(nn.Module):
    """
    Standard TD3 Actor: CNN per-frame → FC head (no LSTM temporal modeling).

    Processes a SINGLE frame (hist_n=1). The observation includes:
      - 84 LiDAR rays normalized
      - 3 goal features (dist/10, cos, sin)
      - 2 velocity encodings (v*2, (w+1)/2)
    """

    def __init__(self, action_dim, max_action, single_obs_dim=89, n_lidar=84):
        super(TD3Actor, self).__init__()
        self.max_action = max_action
        self.n_lidar = n_lidar

        # ---- CNN on LiDAR (same as RCPG) ----
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)       # (B, 1, 84) → (B, 4, 20)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)       # (B, 4, 20) → (B, 8, 4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)       # (B, 8, 4)  → (B, 4, 1)

        # ---- Per-frame embeddings (same as RCPG) ----
        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        # ---- FC head (same as RCPG, input=CNN(4)+goal(10)+action(10)=24) ----
        self.layer_1 = nn.Linear(24, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Standard TD3 Actor forward: single frame → CNN → FC → action.

        Parameters
        ----------
        s : (B, single_obs_dim)  where single_obs_dim = n_lidar + 5

        Returns
        -------
        action : (B, action_dim) in [-max_action, max_action]
        """
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        batch_size = s.shape[0]

        # Decompose single frame
        laser = s[:, :self.n_lidar]                              # (B, 84)
        goal  = s[:, self.n_lidar:self.n_lidar+3]               # (B, 3)
        act   = s[:, self.n_lidar+3:self.n_lidar+5]             # (B, 2)

        # CNN
        l = laser.unsqueeze(1)                                   # (B, 1, 84)
        l = F.leaky_relu(self.cnn1(l))                           # (B, 4, 20)
        l = F.leaky_relu(self.cnn2(l))                           # (B, 8, 4)
        l = F.leaky_relu(self.cnn3(l))                           # (B, 4, 1)
        l = l.flatten(start_dim=1)                               # (B, 4)

        # Embeddings
        g = F.leaky_relu(self.goal_embed(goal))                  # (B, 10)
        a = F.leaky_relu(self.action_embed(act))                 # (B, 10)

        # Concat → FC
        feat = torch.concat((l, g, a), dim=-1)                   # (B, 24)
        h = F.leaky_relu(self.layer_1(feat))
        h = F.leaky_relu(self.layer_2(h))
        action = self.max_action * self.tanh(self.layer_3(h))
        return action


class TD3Critic(nn.Module):
    """
    Standard TD3 Critic: CNN per-frame → Q heads (no LSTM).

    Twin Q-networks (Q1, Q2) with shared CNN+embedding encoder.
    """

    def __init__(self, action_dim, single_obs_dim=89, n_lidar=84):
        super(TD3Critic, self).__init__()
        self.n_lidar = n_lidar

        # ---- CNN (same as RCPG) ----
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        # ---- Embeddings (same as RCPG) ----
        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        # ---- Q1 head ----
        self.q1_layer_1 = nn.Linear(24, 400)
        torch.nn.init.kaiming_uniform_(self.q1_layer_1.weight, nonlinearity="leaky_relu")
        self.q1_layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.q1_layer_2_s.weight, nonlinearity="leaky_relu")
        self.q1_layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.q1_layer_2_a.weight, nonlinearity="leaky_relu")
        self.q1_layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.q1_layer_3.weight, nonlinearity="leaky_relu")

        # ---- Q2 head ----
        self.q2_layer_1 = nn.Linear(24, 400)
        torch.nn.init.kaiming_uniform_(self.q2_layer_1.weight, nonlinearity="leaky_relu")
        self.q2_layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.q2_layer_2_s.weight, nonlinearity="leaky_relu")
        self.q2_layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.q2_layer_2_a.weight, nonlinearity="leaky_relu")
        self.q2_layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.q2_layer_3.weight, nonlinearity="leaky_relu")

    def _encode(self, s):
        """
        Encode single frame: CNN + embeddings → 24-dim feature.
        """
        batch_size = s.shape[0]
        laser = s[:, :self.n_lidar]
        goal  = s[:, self.n_lidar:self.n_lidar+3]
        act   = s[:, self.n_lidar+3:self.n_lidar+5]

        l = laser.unsqueeze(1)
        l = F.leaky_relu(self.cnn1(l))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))
        a = F.leaky_relu(self.action_embed(act))

        return torch.concat((l, g, a), dim=-1)                    # (B, 24)

    def forward(self, s, action):
        """
        Returns (Q1, Q2) for state-action pair.
        """
        shared = self._encode(s)                                   # (B, 24)

        # Q1
        h1 = F.leaky_relu(self.q1_layer_1(shared))
        h1 = F.leaky_relu(self.q1_layer_2_s(h1) + self.q1_layer_2_a(action))
        q1 = self.q1_layer_3(h1)

        # Q2
        h2 = F.leaky_relu(self.q2_layer_1(shared))
        h2 = F.leaky_relu(self.q2_layer_2_s(h2) + self.q2_layer_2_a(action))
        q2 = self.q2_layer_3(h2)

        return q1, q2


class TD3Agent(object):
    """
    Standard TD3 agent (no LSTM, no frame stacking).

    Same TD3 training algorithm as CNNRC but without temporal LSTM.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        model_name="TD3",
    ):
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.model_name = model_name

        # Actor
        self.actor = TD3Actor(action_dim, max_action).to(self.device)
        self.actor_target = TD3Actor(action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Critic
        self.critic = TD3Critic(action_dim).to(self.device)
        self.critic_target = TD3Critic(action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0

    def get_action(self, obs, add_noise):
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -float('inf')
        av_loss = 0

        for it in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device).reshape(-1, 1)
            done = torch.Tensor(batch_dones).to(self.device).reshape(-1, 1)

            # Target action with noise
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = (torch.randn_like(next_action) * policy_noise).clamp(
                    -noise_clip, noise_clip
                )
                next_action = (next_action + noise).clamp(
                    -self.max_action, self.max_action
                )

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q).item())

                target_Q = reward + ((1 - done) * discount * target_Q)

            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_Q1, _ = self.critic(state, self.actor(state))
                actor_loss = -actor_Q1.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                # Soft update targets
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss

        self.iter_count += 1
        n = float(iterations)
        self.writer.add_scalar("train/loss", av_loss / n, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / n, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.actor_target.state_dict(), "%s/%s_actor_target.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(self.critic_target.state_dict(), "%s/%s_critic_target.pth" % (directory, filename))


# ==================== 训练函数 ====================

def train_agent(
    agent,
    env,
    replay_buffer,
    total_steps,
    warmup_steps,
    collect_steps,
    train_iterations,
    batch_size,
    gamma,
    tau,
    policy_noise,
    noise_clip,
    policy_freq,
    max_steps_per_episode,
    writer,
    model_name,
    save_dir,
    save_every=25_000,
    action_dim=2,
):
    """
    Generic training loop — works with both RCPG (CNNRC) and TD3Agent.
    """
    episode = 0
    episode_rewards = []
    success_history = []
    best_mean_reward = -float('inf')
    steps_done = 0

    t_start = time.time()
    obs, _ = env.reset()

    while steps_done < total_steps:
        # ---- Collect phase ----
        collect_start = steps_done
        while (steps_done - collect_start < collect_steps
               and steps_done < total_steps):
            episode_reward = 0.0
            episode_steps_in_ep = 0

            for _ in range(max_steps_per_episode):
                if steps_done < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0,
                                               size=action_dim).astype(np.float32)
                else:
                    action = agent.get_action(obs, add_noise=True)

                # Map action: [-1,1] → env [v:0~1, w:-1~1]
                v_env = (float(action[0]) + 1.0) / 2.0
                w_env = float(action[1])
                env_action = np.array([v_env, w_env], dtype=np.float32)

                next_obs, reward, done, truncated, info = env.step(env_action)
                episode_reward += reward
                episode_steps_in_ep += 1
                steps_done += 1

                terminal = float(done or truncated)
                replay_buffer.add(obs, action, reward, terminal, next_obs)
                obs = next_obs

                if done or truncated:
                    break

            episode += 1
            episode_rewards.append(episode_reward)
            success_history.append(1 if info.get('result') == 'success' else 0)

            if episode % 10 == 0:
                recent_10 = episode_rewards[-10:]
                mean_reward = np.mean(recent_10)
                recent_successes = success_history[-50:] if len(success_history) >= 50 else success_history
                success_rate = np.mean(recent_successes) if recent_successes else 0.0
                elapsed = time.time() - t_start
                fps = steps_done / max(elapsed, 0.001)

                print(f"[{model_name}] ep={episode:5d} | steps={steps_done:8d} | "
                      f"rew={episode_reward:8.2f} | mean10={mean_reward:8.2f} | "
                      f"succ%={success_rate:6.1%} | fps={fps:6.0f} | "
                      f"result={info.get('result', 'timeout')}")

                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/mean_reward_10', mean_reward, episode)
                writer.add_scalar('episode/success_rate_50', success_rate, episode)
                writer.add_scalar('episode/steps', episode_steps_in_ep, episode)
                writer.add_scalar('train/total_steps', steps_done, episode)
                writer.add_scalar('train/fps', fps, episode)

            # Save checkpoint
            if steps_done % save_every < episode_steps_in_ep:
                agent.save(filename=f'{model_name}_step{steps_done}',
                           directory=Path(save_dir))
                recent_10 = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                mean_reward = np.mean(recent_10)
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    agent.save(filename='best_model', directory=Path(save_dir))

            obs, _ = env.reset()

        # ---- Train phase ----
        if steps_done >= warmup_steps and replay_buffer.size() >= batch_size:
            for _ in range(train_iterations):
                agent.train(
                    replay_buffer=replay_buffer,
                    iterations=1,
                    batch_size=batch_size,
                    discount=gamma,
                    tau=tau,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    policy_freq=policy_freq,
                )

    elapsed = time.time() - t_start
    print(f"\n[{model_name}] 训练完成: {steps_done} 步, {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Summary
    if success_history:
        total_episodes = len(success_history)
        total_successes = sum(success_history)
        recent_50 = success_history[-50:] if len(success_history) >= 50 else success_history
        avg_reward = np.mean(episode_rewards)
        print(f"[{model_name}] 总体成功率: {total_successes/total_episodes:.2%}")
        print(f"[{model_name}] 最近50-ep: {np.mean(recent_50):.2%}")
        print(f"[{model_name}] 平均奖励: {avg_reward:.2f}")

    return {
        'steps': steps_done,
        'episodes': episode,
        'success_rate': np.mean(success_history) if success_history else 0.0,
        'recent_50_rate': np.mean(success_history[-50:]) if len(success_history) >= 50 else (np.mean(success_history) if success_history else 0.0),
        'avg_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'train_time': elapsed,
    }


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='RCPG vs TD3 comparison')
    parser.add_argument('--steps', type=int, default=50_000,
                        help='Total training steps for EACH agent (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--yaml', type=str, default='./env/env.yaml',
                        help='Environment YAML config')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")
    print(f"每 agent 训练步数: {args.steps}")
    print(f"YAML: {args.yaml}")

    # Shared hyperparameters
    MAX_STEPS_PER_EPISODE = 1500
    ACTION_DIM = 2
    MAX_ACTION = 1.0
    LR = 1e-4
    GAMMA = 0.99
    TAU = 0.005
    POLICY_FREQ = 2
    POLICY_NOISE = 0.2
    NOISE_CLIP = 0.5
    BATCH_SIZE = 256
    BUFFER_CAPACITY = 200_000
    WARMUP_STEPS = 5_000
    COLLECT_STEPS = 1_000
    TRAIN_ITERATIONS = 500

    SINGLE_OBS_DIM = 89  # 84 LiDAR + 5 aux
    HIST_N_RCPG = 3

    results = {}

    # ========== Experiment 1: RCPG (with LSTM) ==========
    print("\n" + "="*70)
    print("Experiment 1: RCPG (CNN + LSTM + TD3)")
    print("="*70)

    env_rcpg = IrSimNavEnv(
        yaml_file=args.yaml,
        render_mode=None,
        hist_n=HIST_N_RCPG,
    )

    agent_rcpg = CNNRC(
        state_dim=SINGLE_OBS_DIM * HIST_N_RCPG,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        device=torch.device(DEVICE),
        lr=LR,
        hist_n=HIST_N_RCPG,
        save_every=0,
        model_name="RCPG_compare",
        save_directory=Path('./compare_models/RCPG'),
    )

    buffer_rcpg = ReplayBuffer(buffer_size=BUFFER_CAPACITY, random_seed=args.seed)
    writer_rcpg = SummaryWriter('./compare_tb/RCPG')

    results['RCPG'] = train_agent(
        agent=agent_rcpg,
        env=env_rcpg,
        replay_buffer=buffer_rcpg,
        total_steps=args.steps,
        warmup_steps=WARMUP_STEPS,
        collect_steps=COLLECT_STEPS,
        train_iterations=TRAIN_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP,
        policy_freq=POLICY_FREQ,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        writer=writer_rcpg,
        model_name='RCPG',
        save_dir='./compare_models/RCPG',
        save_every=25_000,
    )

    writer_rcpg.close()
    agent_rcpg.writer.close()
    env_rcpg.close()

    # ========== Experiment 2: Standard TD3 (no LSTM) ==========
    print("\n" + "="*70)
    print("Experiment 2: Standard TD3 (CNN + FC, no LSTM)")
    print("="*70)

    # Reset seeds for fair comparison
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_td3 = IrSimNavEnv(
        yaml_file=args.yaml,
        render_mode=None,
        hist_n=1,  # No frame stacking for standard TD3
    )

    agent_td3 = TD3Agent(
        state_dim=SINGLE_OBS_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        device=torch.device(DEVICE),
        lr=LR,
        model_name="TD3_compare",
    )

    buffer_td3 = ReplayBuffer(buffer_size=BUFFER_CAPACITY, random_seed=args.seed)
    writer_td3 = SummaryWriter('./compare_tb/TD3')

    results['TD3'] = train_agent(
        agent=agent_td3,
        env=env_td3,
        replay_buffer=buffer_td3,
        total_steps=args.steps,
        warmup_steps=WARMUP_STEPS,
        collect_steps=COLLECT_STEPS,
        train_iterations=TRAIN_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP,
        policy_freq=POLICY_FREQ,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        writer=writer_td3,
        model_name='TD3',
        save_dir='./compare_models/TD3',
        save_every=25_000,
    )

    writer_td3.close()
    agent_td3.writer.close()
    env_td3.close()

    # ========== Final Comparison ==========
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    print(f"{'指标':<25} {'RCPG (LSTM)':<20} {'TD3 (no LSTM)':<20} {'增益':<15}")
    print("-"*80)
    print(f"{'训练步数':<25} {results['RCPG']['steps']:<20} {results['TD3']['steps']:<20}")
    print(f"{'Episodes':<25} {results['RCPG']['episodes']:<20} {results['TD3']['episodes']:<20}")
    print(f"{'总体成功率':<25} {results['RCPG']['success_rate']:<20.2%} {results['TD3']['success_rate']:<20.2%} {results['RCPG']['success_rate'] - results['TD3']['success_rate']:>+.2%}")
    print(f"{'最近50-ep成功率':<25} {results['RCPG']['recent_50_rate']:<20.2%} {results['TD3']['recent_50_rate']:<20.2%} {results['RCPG']['recent_50_rate'] - results['TD3']['recent_50_rate']:>+.2%}")
    print(f"{'平均奖励':<25} {results['RCPG']['avg_reward']:<20.2f} {results['TD3']['avg_reward']:<20.2f} {results['RCPG']['avg_reward'] - results['TD3']['avg_reward']:>+.2f}")
    print(f"{'训练时间 (s)':<25} {results['RCPG']['train_time']:<20.0f} {results['TD3']['train_time']:<20.0f}")

    print(f"\nTensorBoard: tensorboard --logdir=./compare_tb/")
    print(f"Models: ./compare_models/")


if __name__ == '__main__':
    main()
