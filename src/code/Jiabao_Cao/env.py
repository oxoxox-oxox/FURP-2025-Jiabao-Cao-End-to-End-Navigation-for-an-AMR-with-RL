"""
IrSimNavEnv: gymnasium-compatible wrapper for ir-sim.

观测: [N条激光原始距离值, 目标距离/cos/sin, 自身速度编码] → 共 N+5 维
    - LiDAR: 原始距离 r / max_range=10 归一化到 [0,1]
    - 无效射线 (r≈max_range) → 归一化为 1.0 (单位圆边界)
    - 目标信息: dist/10([0,1]) + cos([-1,1]) + sin([-1,1])
    - 速度编码: v*2([0,2]), (w+1)/2([0,1])
    - N 从 YAML 中 lidar2d.number 动态读取
动作: [线速度v, 角速度w] (连续), 供 TD3 policy 映射
"""

from collections import deque

import gymnasium as gym
import numpy as np
import irsim
import matplotlib.pyplot as plt


class IrSimNavEnv(gym.Env):
    """
    将 ir-sim 包装成 Gymnasium 标准接口。

    通过 render_mode 控制渲染:
        - render_mode='human': 显示可视化窗口 (用于评估/演示)
        - render_mode=None  : 无渲染, 后台运行 (用于训练, 速度最快)
    """

    def __init__(self, yaml_file='./env/env_corridor_1.yaml', render_mode=None,
                 display=None, disable_all_plot=None, seed=None, hist_n=3):
        super().__init__()
        # 支持单个 YAML 或列表（每 episode 随机切换场景）
        if isinstance(yaml_file, (list, tuple)):
            self._yaml_files = list(yaml_file)
        else:
            self._yaml_files = [yaml_file]
        self.yaml_file = self._yaml_files[0]
        self.render_mode = render_mode

        # 根据 render_mode 自动决定是否开启渲染
        if display is None:
            display = (render_mode == 'human')
        if disable_all_plot is None:
            disable_all_plot = (render_mode != 'human')

        self._display = display
        self._disable_all_plot = disable_all_plot
        self._irsim_seed = seed

        self.env = irsim.make(
            self.yaml_file,
            disable_all_plot=self._disable_all_plot,
            log_level='WARNING',
            seed=self._irsim_seed,
            display=self._display
        )

        # ===== 动作空间 =====
        # 线速度 [0, 1.0] m/s，角速度 [-1.0, 1.0] rad/s
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ===== 观测空间 =====
        # 堆叠 hist_n 帧, 每帧: N 条 LiDAR 距离值 + 5 维 (dist/cos/sin + v_enc + w_enc)
        self.hist_n = hist_n
        lidar_data = self.env.robot_list[0].get_lidar_scan()
        self._num_lidar_rays = len(np.array(lidar_data['ranges']).flatten())
        self._single_obs_dim = self._num_lidar_rays + 5
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._single_obs_dim * self.hist_n,),
            dtype=np.float32
        )

        # 帧缓冲区 (FIFO)
        self._obs_history = deque(maxlen=self.hist_n)

        # 超参数
        self.max_steps = 1500  # 到达目标附近后需要额外时间绕过最后障碍
        self.lidar_range = 10.0   # 匹配 YAML 中 lidar2d range_max: 10
        self.max_linear_vel = 1.0    # 动作空间线速度上限, 用于归一化
        self.max_angular_vel = 1.0   # 动作空间角速度上限, 用于归一化
        self.goal_threshold = self.env.robot_list[0].goal_threshold
        self.current_step = 0
        self._prev_dist = None     # 用于 progress reward 的上一帧距离
        self._prev_obs = None      # 上一帧观测

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # 随机切换场景（多 YAML 时每个 episode 随机选一个）
        if len(self._yaml_files) > 1:
            chosen = np.random.choice(self._yaml_files)
            if chosen != self.yaml_file:
                self.yaml_file = chosen
                self.env.end()
                plt.close('all')          # 清理上个场景的 matplotlib figure
                self.env = irsim.make(
                    self.yaml_file,
                    disable_all_plot=self._disable_all_plot,
                    log_level='WARNING',
                    seed=self._irsim_seed,
                    display=self._display
                )
        self.env.reload()

        self.current_step = 0

        # Reset distance tracking for progress reward
        self._prev_dist = None

        single_obs = self._get_observation()
        self._prev_obs = single_obs.copy()

        # ---- 帧堆叠: 用首帧填充整个缓冲区 ----
        self._obs_history.clear()
        for _ in range(self.hist_n):
            self._obs_history.append(single_obs.copy())

        return self._get_stacked_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 执行动作：ir-sim 接受 [v, w] 速度指令
        v, w = float(action[0]), float(action[1])
        self.env.step(np.array([[v], [w]]))

        single_obs = self._get_observation()
        reward, done, info = self._compute_reward()

        # ---- 帧堆叠: 推入新帧, 返回堆叠观测 ----
        self._obs_history.append(single_obs.copy())
        obs = self._get_stacked_obs()

        self._prev_obs = single_obs.copy()

        # 超时截断 (truncated ≠ done by collision/success)
        truncated = self.current_step >= self.max_steps

        # Record episode length when episode ends
        if done or truncated:
            info['episode_length'] = self.current_step

        return obs, reward, done, truncated, info

    def _get_stacked_obs(self):
        """将帧缓冲区拼接为堆叠观测向量: (hist_n * single_obs_dim,)."""
        return np.concatenate(list(self._obs_history)).astype(np.float32)

    def _get_observation(self):
        """构建 CNNTD3 观测向量。

        排列: [ranges_norm(N), goal_dist/10, goal_cos, goal_sin, v*2, (w+1)/2]

        LiDAR → 原始距离值 r/range_max, CNN 从 1D 距离序列学习空间特征。
        速度编码匹配 CNNTD3.prepare_state():
          - lin_vel = action[0] * 2       (原始动作 v∈[0,1] → [0,2])
          - ang_vel = (action[1] + 1) / 2 (原始动作 w∈[-1,1] → [0,1])
        """
        robot = self.env.robot_list[0]

        # ===== 1. LiDAR → 原始距离值 =====
        lidar_data = robot.get_lidar_scan()
        ranges = np.array(lidar_data['ranges']).flatten()

        # 处理无效值: inf/nan → lidar_range (无 hit 方向)
        ranges = np.nan_to_num(ranges, nan=self.lidar_range,
                               posinf=self.lidar_range, neginf=0.0)
        ranges_norm = np.clip(ranges / self.lidar_range, 0.0, 1.0)

        # ===== 2. 自身速度 =====
        vel = robot.velocity.flatten()                           # [v, ω]
        v_norm = np.clip(vel[0] / self.max_linear_vel, -1, 1)   # [0, 1]
        w_norm = np.clip(vel[1] / self.max_angular_vel, -1, 1)  # [-1, 1]

        v_encoded = v_norm * 2.0           # [0, 2]
        w_encoded = (w_norm + 1.0) / 2.0   # [0, 1]

        # ===== 3. 目标信息 =====
        robot_pos = robot.state[:2].flatten()
        robot_theta = float(robot.state[2].item())
        goal_pos = np.array(robot.goal[:2]).flatten()
        diff = goal_pos - robot_pos
        dist_to_goal = np.linalg.norm(diff)
        angle_to_goal = np.arctan2(diff[1], diff[0]) - robot_theta

        goal_dist_norm = np.clip(dist_to_goal / 10.0, 0.0, 1.0)
        goal_cos = np.cos(angle_to_goal)
        goal_sin = np.sin(angle_to_goal)

        # ===== 4. 拼接 =====
        obs = np.concatenate([
            ranges_norm.astype(np.float32),                        # N 维
            np.array([goal_dist_norm, goal_cos, goal_sin],
                     dtype=np.float32),                            # 3 维
            np.array([v_encoded, w_encoded], dtype=np.float32)     # 2 维
        ])
        return obs  # N + 5 维

    def _compute_reward(self):
        robot = self.env.robot_list[0]
        robot_pos = robot.state[:2].flatten()
        goal_pos = np.array(robot.goal[:2]).flatten()
        dist = np.linalg.norm(goal_pos - robot_pos)

        done = False
        info = {}

        # ① 先判断是否到达目标
        if dist < self.goal_threshold:
            reward = 200.0
            done = True
            info['result'] = 'success'
            return reward, done, info

        # ② 再用 ir-sim 的 done() 判断碰撞
        if self.env.done():
            reward = -30.0
            done = True
            info['result'] = 'collision'
            return reward, done, info

        # ③ 正常步骤: dense reward 引导 agent 靠近目标
        # ---- 距离进度奖励 (核心) ----
        # >0 = 正在靠近目标, <0 = 正在远离目标
        if self._prev_dist is not None:
            progress = self._prev_dist - dist
            progress_reward = progress * 10.0   # 靠近 1m ≈ +10 reward, 单步~+1
        else:
            progress_reward = 0.0
        self._prev_dist = dist

        # ---- 靠近目标奖励 ----
        proximity_reward = max(0.0, 1.0 - dist / 50.0) * 0.3

        # # ---- 存活奖励 (鼓励探索) ----
        # alive_bonus = 0.03

        # ---- 激光空旷奖励 (鼓励朝向开阔方向, 间接帮助避障) ----
        lidar_data = robot.get_lidar_scan()
        ranges = np.array(lidar_data['ranges']).flatten()
        min_dist_raw = np.min(ranges)              # 原始距离（单位：米）
        min_dist_norm = min_dist_raw / self.lidar_range  # 归一化到 [0, 1]

        # 设定安全阈值（归一化后）。例如 0.2 表示实际距离 < 20% 的雷达量程时触发惩罚
        safety_threshold = 0.2  
        if min_dist_norm < safety_threshold:
            # 越近惩罚越大，呈线性增长。
            # 当完全贴近 (min_dist_norm=0) 时，惩罚为 -2.0 * 0.2 = -0.4
            # 当刚好在阈值边界 (min_dist_norm=0.2) 时，惩罚为 0
            clearance_penalty = -2.0 * (safety_threshold - min_dist_norm)
        else:
            clearance_penalty = 0.0

        # ---- 朝向奖励 (鼓励机器人面朝目标) ----
        # cos(angle_to_goal): +1=正对目标, 0=垂直, -1=背对目标
        diff = goal_pos - robot_pos
        robot_theta = float(robot.state[2].item())
        angle_to_goal = np.arctan2(diff[1], diff[0]) - robot_theta
        heading_alignment = np.cos(angle_to_goal)         # [-1, 1]
        heading_reward = heading_alignment * 0.15         # [-0.15, +0.15]

        # ---- 时间惩罚 (指数增长, 越晚惩罚急剧加大) ----
        progress_ratio = self.current_step / self.max_steps
        time_penalty = -0.2 * np.exp(2.0 * progress_ratio)   # -0.10 → -0.74

        reward = (progress_reward + proximity_reward + heading_reward
                  + time_penalty + clearance_penalty)
        info['result'] = 'running'
        return reward, done, info

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

    def close(self):
        if self.env is not None:
            self.env.end()
        plt.close('all')
