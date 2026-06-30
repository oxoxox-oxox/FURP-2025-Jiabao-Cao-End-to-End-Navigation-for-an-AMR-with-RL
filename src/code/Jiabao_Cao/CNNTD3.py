"""
RCPG: Recurrent + CNN + TD3 (Twin Delayed DDPG with LSTM).

Architecture:
    - Frame stacking: hist_n consecutive observations → temporal sequence.
    - CNN: 3-layer 1D Conv extracts 4-dim spatial features from each frame's LiDAR.
    - LSTM: temporal LSTM processes the sequence of per-frame CNN+embedding features.
    - Actor: LSTM last hidden → FC → Tanh → action.
    - Critic: TD3 twin Q-networks (Q1, Q2) share a CNN+LSTM encoder,
      with independent FC heads.  target_Q = min(Q1, Q2).

Differences from standard TD3:
    - Recurrent LSTM for temporal modeling across stacked frames.
    - Shared CNN+LSTM encoder between Q1 and Q2 (param efficiency).
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# ==================== Actor 网络 ====================

class Actor(nn.Module):
    """
    RCPG Actor: CNN per-frame + temporal LSTM + FC head.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    hist_n : int
        Number of stacked frames (temporal sequence length).
    lstm_hidden_dim : int
        LSTM hidden state dimension.
    """

    def __init__(self, action_dim, hist_n=3, lstm_hidden_dim=24):
        super(Actor, self).__init__()

        self.hist_n = hist_n
        self.lstm_hidden_dim = lstm_hidden_dim

        # ---- Per-frame CNN (共享权重) ----
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)       # (B, 1, 84) → (B, 4, 20)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)       # (B, 4, 20) → (B, 8, 4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)       # (B, 8, 4)  → (B, 4, 1)

        # ---- Per-frame embeddings ----
        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        # ---- Temporal LSTM ----
        # 每帧特征: CNN(4) + goal_embed(10) + action_embed(10) = 24
        self.temporal_lstm = nn.LSTM(
            input_size=24,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # 正交初始化 LSTM (RL 常用)
        for name, param in self.temporal_lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)   # forget gate bias = 1

        # ---- FC 头 ----
        self.layer_1 = nn.Linear(lstm_hidden_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def _encode_frame(self, laser, goal, act):
        """
        对单帧编码: CNN(laser) → 4 + goal_embed(10) + action_embed(10) = 24.

        Parameters
        ----------
        laser : (B, 84)
        goal  : (B, 3)
        act   : (B, 2)

        Returns
        -------
        (B, 24)
        """
        l = laser.unsqueeze(1)                     # (B, 1, 84)
        l = F.leaky_relu(self.cnn1(l))             # (B, 4, 20)
        l = F.leaky_relu(self.cnn2(l))             # (B, 8, 4)
        l = F.leaky_relu(self.cnn3(l))             # (B, 4, 1)
        l = l.flatten(start_dim=1)                 # (B, 4)

        g = F.leaky_relu(self.goal_embed(goal))    # (B, 10)
        a = F.leaky_relu(self.action_embed(act))   # (B, 10)

        return torch.concat((l, g, a), dim=-1)     # (B, 24)

    def forward(self, s):
        """
        Actor 前向传播: 堆叠帧 → CNN per-frame → LSTM → FC → action.

        Parameters
        ----------
        s : torch.Tensor
            堆叠状态, 形状 (batch_size, hist_n * 89) 或 (batch_size, hist_n, 89).

        Returns
        -------
        torch.Tensor
            动作, 形状 (batch_size, action_dim), 范围 [-1, 1].
        """
        # ---- 处理输入形状 ----
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        batch_size = s.shape[0]

        if len(s.shape) == 2:
            s = s.view(batch_size, self.hist_n, -1)   # (B, hist_n, 89)

        # ---- 分解每帧的 laser / goal / action ----
        laser = s[:, :, :-5]                            # (B, hist_n, 84)
        goal  = s[:, :, -5:-2]                          # (B, hist_n, 3)
        act   = s[:, :, -2:]                            # (B, hist_n, 2)

        # ---- 将所有帧展平为 batch，共享 CNN 权重处理 ----
        b = batch_size * self.hist_n
        frame_feats = self._encode_frame(
            laser.reshape(b, -1),
            goal.reshape(b, -1),
            act.reshape(b, -1),
        )                                               # (B*hist_n, 24)

        # ---- 恢复时序维度 → LSTM ----
        frame_feats = frame_feats.view(batch_size, self.hist_n, -1)  # (B, hist_n, 24)
        lstm_out, _ = self.temporal_lstm(frame_feats)                # (B, hist_n, hidden)
        temporal_feat = lstm_out[:, -1, :]                           # (B, hidden)

        # ---- FC → action ----
        h = F.leaky_relu(self.layer_1(temporal_feat))
        h = F.leaky_relu(self.layer_2(h))
        a = self.tanh(self.layer_3(h))
        return a


# ==================== Critic 网络 (TD3 Twin) ====================

class Critic(nn.Module):
    """
    RCPG Critic: TD3 twin Q-networks with shared CNN+LSTM encoder.

    Q1 和 Q2 共享 CNN+LSTM 编码器, 但拥有独立的 FC 头。
    目标值取 min(Q1, Q2) 以减少过高估计。

    Parameters
    ----------
    action_dim : int
        动作空间维度.
    hist_n : int
        堆叠帧数.
    lstm_hidden_dim : int
        LSTM 隐藏维度.
    """

    def __init__(self, action_dim, hist_n=3, lstm_hidden_dim=24):
        super(Critic, self).__init__()

        self.hist_n = hist_n
        self.lstm_hidden_dim = lstm_hidden_dim

        # ---- 共享 Per-frame CNN ----
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        # ---- 共享 Per-frame embeddings ----
        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        # ---- 共享 Temporal LSTM ----
        self.temporal_lstm = nn.LSTM(
            input_size=24,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        for name, param in self.temporal_lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        # ---- Q1 head ----
        self.q1_layer_1 = nn.Linear(lstm_hidden_dim, 400)
        torch.nn.init.kaiming_uniform_(self.q1_layer_1.weight, nonlinearity="leaky_relu")
        self.q1_layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.q1_layer_2_s.weight, nonlinearity="leaky_relu")
        self.q1_layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.q1_layer_2_a.weight, nonlinearity="leaky_relu")
        self.q1_layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.q1_layer_3.weight, nonlinearity="leaky_relu")

        # ---- Q2 head ----
        self.q2_layer_1 = nn.Linear(lstm_hidden_dim, 400)
        torch.nn.init.kaiming_uniform_(self.q2_layer_1.weight, nonlinearity="leaky_relu")
        self.q2_layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.q2_layer_2_s.weight, nonlinearity="leaky_relu")
        self.q2_layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.q2_layer_2_a.weight, nonlinearity="leaky_relu")
        self.q2_layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.q2_layer_3.weight, nonlinearity="leaky_relu")

    def _encode_frame(self, laser, goal, act):
        """与 Actor 完全相同的 per-frame 编码: CNN → 4 + embed → 24."""
        l = laser.unsqueeze(1)
        l = F.leaky_relu(self.cnn1(l))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))
        a = F.leaky_relu(self.action_embed(act))

        return torch.concat((l, g, a), dim=-1)

    def _encode(self, s):
        """
        共享编码器: stacked frames → CNN per-frame → LSTM → 24-dim temporal feature.

        Parameters
        ----------
        s : (B, hist_n * 89) or (B, hist_n, 89)

        Returns
        -------
        (B, lstm_hidden_dim)
        """
        batch_size = s.shape[0]
        if len(s.shape) == 2:
            s = s.view(batch_size, self.hist_n, -1)

        laser = s[:, :, :-5]
        goal  = s[:, :, -5:-2]
        act   = s[:, :, -2:]

        b = batch_size * self.hist_n
        frame_feats = self._encode_frame(
            laser.reshape(b, -1),
            goal.reshape(b, -1),
            act.reshape(b, -1),
        )
        frame_feats = frame_feats.view(batch_size, self.hist_n, -1)
        lstm_out, _ = self.temporal_lstm(frame_feats)
        return lstm_out[:, -1, :]                          # (B, lstm_hidden_dim)

    def forward(self, s, action):
        """
        前向传播: 编码状态 → Q1(s,a), Q2(s,a).

        Parameters
        ----------
        s      : (B, hist_n * 89)
        action : (B, action_dim)

        Returns
        -------
        (Q1, Q2) : 各 (B, 1)
        """
        shared = self._encode(s)                           # (B, hidden)

        # ---- Q1 ----
        h1 = F.leaky_relu(self.q1_layer_1(shared))
        s1_part = torch.mm(h1, self.q1_layer_2_s.weight.data.t())
        a1_part = torch.mm(action, self.q1_layer_2_a.weight.data.t())
        h1 = F.leaky_relu(s1_part + a1_part + self.q1_layer_2_a.bias.data)
        q1 = self.q1_layer_3(h1)

        # ---- Q2 ----
        h2 = F.leaky_relu(self.q2_layer_1(shared))
        s2_part = torch.mm(h2, self.q2_layer_2_s.weight.data.t())
        a2_part = torch.mm(action, self.q2_layer_2_a.weight.data.t())
        h2 = F.leaky_relu(s2_part + a2_part + self.q2_layer_2_a.bias.data)
        q2 = self.q2_layer_3(h2)

        return q1, q2


# ==================== RCPG 智能体 ====================

class CNNRC(object):
    """
    RCPG (Recurrent + CNN + TD3) 智能体, 用于连续控制导航任务.

    架构:
        - CNN: 3 层 1D Conv 从每帧 LiDAR 提取空间特征.
        - LSTM: 对 hist_n 帧的 CNN+embedding 特征序列进行时序建模.
        - Actor: LSTM 最后隐状态 → FC → Tanh → 动作 [-1, 1].
        - Critic: TD3 双 Q 网络 (Q1, Q2), 共享 CNN+LSTM 编码器,
          target_Q = min(Q1, Q2), 带目标策略噪声.

    Parameters
    ----------
    state_dim : int
        堆叠后的状态维度 (= single_obs_dim * hist_n).
    action_dim : int
        动作空间维度.
    max_action : float
        动作最大幅度.
    device : torch.device
    lr : float
        学习率.
    hist_n : int
        堆叠的历史帧数 (默认 3).
    lstm_hidden_dim : int
        LSTM 隐藏维度 (默认 24).
    save_every : int
        检查点保存间隔 (0 禁用).
    load_model : bool
    save_directory : Path
    model_name : str
    load_directory : Path
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        hist_n=3,
        lstm_hidden_dim=24,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/RCPG/checkpoint"),
        model_name="RCPG",
        load_directory=Path("robot_nav/models/RCPG/checkpoint"),
    ):
        self.device = device
        self.hist_n = hist_n
        self.lstm_hidden_dim = lstm_hidden_dim

        # Actor
        self.actor = Actor(action_dim, hist_n=hist_n, lstm_hidden_dim=lstm_hidden_dim).to(self.device)
        self.actor_target = Actor(action_dim, hist_n=hist_n, lstm_hidden_dim=lstm_hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Critic (TD3 twin)
        self.critic = Critic(action_dim, hist_n=hist_n, lstm_hidden_dim=lstm_hidden_dim).to(self.device)
        self.critic_target = Critic(action_dim, hist_n=hist_n, lstm_hidden_dim=lstm_hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0

        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

        # ---- 帧缓冲区 (用于 prepare_state 的 ROS/外部接口) ----
        self._frame_buffer = None

    def get_action(self, obs, add_noise):
        """
        为给定观测选择动作.

        Parameters
        ----------
        obs : np.ndarray
            堆叠观测, 形状 (state_dim,) = (hist_n * 89,).
        add_noise : bool
            是否添加探索噪声.

        Returns
        -------
        np.ndarray : 动作, 形状 (action_dim,).
        """
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        """确定性动作."""
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # ==================== TD3 训练循环 ====================

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
        """
        TD3 训练 (with temporal LSTM).

        Parameters
        ----------
        replay_buffer : ReplayBuffer
        iterations : int
            梯度步数.
        batch_size : int
        discount : float
        tau : float
            软更新率.
        policy_noise : float
            目标策略噪声标准差.
        noise_clip : float
            噪声裁剪范围.
        policy_freq : int
            延迟策略更新频率.
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0

        for it in range(iterations):
            # ---- 采样 ----
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

            # ---- 目标动作 (带噪声) ----
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = (torch.randn_like(next_action) * policy_noise).clamp(
                    -noise_clip, noise_clip
                )
                next_action = (next_action + noise).clamp(
                    -self.max_action, self.max_action
                )

                # ---- 目标 Q 值: min(Q1, Q2) ----
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q))

                # ---- Bellman 目标 ----
                target_Q = reward + ((1 - done) * discount * target_Q)

            # ---- 当前 Q 值 ----
            current_Q1, current_Q2 = self.critic(state, action)

            # ---- Critic 损失 ----
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # ---- 延迟策略更新 ----
            if it % policy_freq == 0:
                actor_Q1, _ = self.critic(state, self.actor(state))
                actor_loss = -actor_Q1.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ---- 软更新 target 网络 ----
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

        # TensorBoard
        n = float(iterations)
        self.writer.add_scalar("train/loss", av_loss / n, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / n, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    # ==================== 模型持久化 ====================

    def save(self, filename, directory):
        """保存模型权重."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(
            self.actor_target.state_dict(),
            "%s/%s_actor_target.pth" % (directory, filename),
        )
        torch.save(
            self.critic.state_dict(),
            "%s/%s_critic.pth" % (directory, filename),
        )
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory):
        """加载模型权重."""
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_target.load_state_dict(
            torch.load("%s/%s_actor_target.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"已加载权重: {directory}")

    # ==================== 状态准备 (ROS / 外部接口) ====================

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        将原始传感器数据整理为堆叠帧状态向量.

        内部维护一个长度为 hist_n 的帧缓冲区。首批帧用首帧填充,
        终止时清空缓冲区。

        Parameters
        ----------
        latest_scan : list 或 np.ndarray
            LiDAR 扫描数据.
        distance : float
            到目标距离.
        cos, sin : float
            航向角 cos/sin.
        collision : bool
        goal : bool
        action : list 或 np.ndarray
            上一步动作 [lin_vel, ang_vel].

        Returns
        -------
        state : list
            堆叠状态, 长度 = self.state_dim.
        terminal : int
            1 if collision or goal else 0.
        """
        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        latest_scan /= 7

        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        single_frame = latest_scan.tolist() + [distance, cos, sin] + [lin_vel, ang_vel]

        # 惰性初始化帧缓冲区
        if self._frame_buffer is None:
            self._frame_buffer = deque(maxlen=self.hist_n)

        self._frame_buffer.append(single_frame)

        # 缓冲区未满时用首帧填充 (episode 开始)
        while len(self._frame_buffer) < self.hist_n:
            self._frame_buffer.appendleft(single_frame)

        state = np.concatenate(list(self._frame_buffer)).tolist()

        assert len(state) == self.state_dim, (
            f"Expected state dim {self.state_dim}, got {len(state)}"
        )

        terminal = 1 if collision or goal else 0

        # 终止时清空缓冲区, 为下一 episode 准备
        if terminal:
            self._frame_buffer = None

        return state, terminal
