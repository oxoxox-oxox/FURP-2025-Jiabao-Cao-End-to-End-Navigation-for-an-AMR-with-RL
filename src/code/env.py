"""
IrSimNavEnv: gymnasium-compatible wrapper for ir-sim.

观测: [36条激光值(归一化), 目标相对距离, 目标相对角度]  → 共38维
动作: [线速度v, 角速度w] (连续)
"""

import gymnasium as gym
import numpy as np
import irsim


class IrSimNavEnv(gym.Env):
    """
    将 ir-sim 包装成 Gymnasium 标准接口。

    通过 render_mode 控制渲染:
        - render_mode='human': 显示可视化窗口 (用于评估/演示)
        - render_mode=None  : 无渲染, 后台运行 (用于训练, 速度最快)
    """

    def __init__(self, yaml_file='./env.yaml', render_mode=None,
                 display=None, disable_all_plot=None, seed=None):
        super().__init__()
        self.yaml_file = yaml_file
        self.render_mode = render_mode

        # 根据 render_mode 自动决定是否开启渲染
        if display is None:
            display = (render_mode == 'human')
        if disable_all_plot is None:
            disable_all_plot = (render_mode != 'human')

        self._display = display
        self._disable_all_plot = disable_all_plot
        self._irsim_seed = seed

        # 在 __init__ 中创建 irsim 环境（只创建一次），
        # 避免在 reset() 中反复 end() + make() 导致 X11 连接泄漏
        self.env = irsim.make(
            self.yaml_file,
            disable_all_plot=self._disable_all_plot,
            log_level='WARNING',
            seed=self._irsim_seed,
            display=self._display
        )

        # ===== 动作空间 =====
        # 线速度 [0, 1.0] m/s，角速度 [-1.5, 1.5] rad/s
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ===== 观测空间 =====
        # 36条激光(归一化到[0,1]) + 目标距离(归一化) + 目标角度(归一化)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(38,),
            dtype=np.float32
        )

        # 超参数
        self.max_steps = 500
        self.lidar_range = 5.0
        self.goal_threshold = 0.3  # 到达目标的距离阈值
        self.current_step = 0
        self._prev_dist = None     # 用于 progress reward 的上一帧距离

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 使用 reload() 而非 end()+make()：
        # reload() 重新解析 YAML、重新随机化障碍物位置，但复用已有的
        # matplotlib figure 和 pynput 连接，不会造成 X11 连接泄漏
        if seed is not None:
            np.random.seed(seed)
        self.env.reload()
        self.current_step = 0

        # Reset distance tracking for progress reward
        self._prev_dist = None

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # 执行动作：ir-sim 接受 [v, w] 速度指令
        v, w = float(action[0]), float(action[1])
        self.env.step(np.array([[v], [w]]))

        obs = self._get_observation()
        reward, done, info = self._compute_reward()

        # 超时截断 (truncated ≠ done by collision/success)
        truncated = self.current_step >= self.max_steps

        # Record episode length when episode ends (for logging in PPO buffer)
        if done or truncated:
            info['episode_length'] = self.current_step

        return obs, reward, done, truncated, info

    def _get_observation(self):
        robot = self.env.robot_list[0]

        # 直接从 robot 对象获取 lidar 数据，避免 get_lidar_scan(id=...) 的索引 bug
        lidar_data = robot.get_lidar_scan()

        ranges = np.array(lidar_data['ranges']).flatten()
        lidar_norm = np.clip(ranges / self.lidar_range,
                             0, 1).astype(np.float32)

        # 目标相对位置
        robot_pos = robot.state[:2].flatten()
        goal_pos = np.array(robot.goal[:2]).flatten()
        diff = goal_pos - robot_pos
        dist_to_goal = np.linalg.norm(diff)
        angle_to_goal = np.arctan2(
            diff[1], diff[0]) - float(robot.state[2].item())

        dist_norm = np.clip(dist_to_goal / 14.0, 0, 1)
        angle_norm = (angle_to_goal % (2 * np.pi)) / (2 * np.pi)

        obs = np.concatenate([
            lidar_norm,
            np.array([dist_norm, angle_norm], dtype=np.float32)
        ]).astype(np.float32)

        return obs

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
            reward = -100.0
            done = True
            info['result'] = 'collision'
            return reward, done, info

        # ③ 正常步骤: dense reward 引导 agent 靠近目标
        # ---- 距离进度奖励 (核心) ----
        # >0 = 正在靠近目标, <0 = 正在远离目标
        if self._prev_dist is not None:
            progress = self._prev_dist - dist
            progress_reward = progress * 10.0   # 靠近 1m ≈ +10 reward
        else:
            progress_reward = 0.0
        self._prev_dist = dist

        # ---- 靠近目标奖励 ----
        proximity_reward = (1.0 - dist / 14.0) * 0.5   # 近则奖励大

        # ---- 存活奖励 (抵消 step penalty, 鼓励探索) ----
        alive_bonus = 0.05

        # ---- 时间惩罚 (轻微, 促使尽快到达) ----
        progress_ratio = self.current_step / self.max_steps
        time_penalty = -0.01 - 0.05 * progress_ratio   # -0.01 → -0.06

        reward = progress_reward + proximity_reward + alive_bonus + time_penalty
        info['result'] = 'running'
        return reward, done, info

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

    def close(self):
        if self.env is not None:
            self.env.end()
