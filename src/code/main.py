from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
import irsim


class IrSimNavEnv(gym.Env):
    """
    将 ir-sim 包装成 Gymnasium 标准接口
    观测: [36条激光值(归一化), 目标相对距离, 目标相对角度]  → 共38维
    动作: [线速度v, 角速度w] (连续)
    """

    def __init__(self, yaml_file='./env/nav_world.yaml', render_mode=None):
        super().__init__()
        self.yaml_file = yaml_file
        self.render_mode = render_mode
        self.env = None

        # ===== 动作空间 =====
        # 线速度 [0, 1.0] m/s，角速度 [-1.5, 1.5] rad/s
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.5]),
            high=np.array([1.0,  1.5]),
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 关闭旧环境，重建新环境（每个episode随机化）
        if self.env is not None:
            self.env.end()
        self.env = irsim.make(self.yaml_file)
        self.current_step = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # 执行动作：ir-sim 接受 [v, w] 速度指令
        v, w = float(action[0]), float(action[1])
        self.env.step(vel=np.array([[v], [w]]))  # ir-sim 的速度输入格式

        obs = self._get_observation()
        reward, done, info = self._compute_reward()

        # 超时截断 (truncated ≠ done by collision/success)
        truncated = self.current_step >= self.max_steps

        return obs, reward, done, truncated, info

    def _get_observation(self):
        robot = self.env.robot_list[0]

        # ✅ 正确的 lidar2d 数据获取方式
        # get_scan() 返回一个 dict，包含 'range' 键
        lidar_data = self.env.get_lidar_scan(id=robot.id)
        # lidar_data 是 dict: {'range': array, 'angle': array, 'points': array}
        ranges = np.array(lidar_data['range']).flatten()
        lidar_norm = np.clip(ranges / self.lidar_range,
                             0, 1).astype(np.float32)

        # 目标相对位置
        robot_pos = robot.state[:2].flatten()
        goal_pos = np.array(robot.goal[:2]).flatten()
        diff = goal_pos - robot_pos
        dist_to_goal = np.linalg.norm(diff)
        angle_to_goal = np.arctan2(diff[1], diff[0]) - float(robot.state[2])

        dist_norm = np.clip(dist_to_goal / 14.0, 0, 1)
        angle_norm = (angle_to_goal % (2 * np.pi)) / (2 * np.pi)

        obs = np.concatenate([
            lidar_norm,
            np.array([dist_norm, angle_norm], dtype=np.float32)
        ]).astype(np.float32)
        return obs

    def _compute_reward(self):
        robot = self.env.robot_list[0]
        robot_pos = robot.state[:2]
        goal_pos = np.array(robot.goal[:2])
        dist = np.linalg.norm(goal_pos - robot_pos)

        done = False
        info = {}

        # ① 到达目标：大正奖励
        if dist < self.goal_threshold:
            reward = 200.0
            done = True
            info['result'] = 'success'

        # ② 碰撞：大负奖励
        elif self.env.is_collision():
            reward = -100.0
            done = True
            info['result'] = 'collision'

        # ③ 存活步骤奖励（距离塑形）
        else:
            # 每步给予基于"是否在靠近目标"的小奖励
            reward = -0.01  # 时间惩罚，鼓励尽快到达
            # 可选：加上势能塑形 (potential-based shaping)
            # reward += (prev_dist - dist) * 2.0

        return reward, done, info

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

    def close(self):
        if self.env is not None:
            self.env.end()

# ===================== main part ========================


# 创建训练环境（可并行多个加速训练）
train_env = make_vec_env(IrSimNavEnv, n_envs=4)
eval_env = IrSimNavEnv(render_mode='human')

# 定义回调：每5000步评估一次，自动保存最好的模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    eval_freq=5000,
    n_eval_episodes=10,
)

# 创建 SAC 智能体（适合连续动作空间导航任务）
model = SAC(
    policy='MlpPolicy',
    env=train_env,
    learning_rate=3e-4,
    buffer_size=100_000,   # Replay Buffer 大小
    batch_size=256,
    verbose=1,
    tensorboard_log='./tb_logs/'
)

# 开始训练
model.learn(total_timesteps=500_000, callback=eval_callback)
model.save('nav_sac_final')

# ====================== evaluation part========================

# 加载训练好的模型并可视化
model = SAC.load('models/best_model')
env = IrSimNavEnv(render_mode='human')

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        print(f"Result: {info.get('result', 'timeout')}")
        obs, _ = env.reset()
env.close()
