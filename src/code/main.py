from stable_baselines3.common.monitor import Monitor
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

    通过 render_mode 控制渲染:
        - render_mode='human': 显示可视化窗口 (用于评估/演示)
        - render_mode=None  : 无渲染, 后台运行 (用于训练, 速度最快)
    """

    def __init__(self, yaml_file='./env/nav_world.yaml', render_mode=None,
                 display=None, disable_all_plot=None, seed=None):
        super().__init__()
        self.yaml_file = yaml_file
        self.render_mode = render_mode

        # 根据 render_mode 自动决定是否开启渲染
        # human 模式 → 显示画面；None → 后台无渲染，加速训练
        if display is None:
            display = (render_mode == 'human')
        if disable_all_plot is None:
            disable_all_plot = (render_mode != 'human')

        self._display = display
        self._disable_all_plot = disable_all_plot
        self._irsim_seed = seed
        self.env = None

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 关闭旧环境，重建新环境（每个episode随机化）
        if self.env is not None:
            self.env.end()

        # 创建 irsim 环境：训练时 display=False 跳过渲染，大幅加速
        self.env = irsim.make(
            self.yaml_file,
            disable_all_plot=self._disable_all_plot,
            log_level='WARNING',       # 减少日志输出
            seed=self._irsim_seed,
            display=False
        )
        self.current_step = 0

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

        return obs, reward, done, truncated, info

    def _get_observation(self):
        robot = self.env.robot_list[0]

        # ✅ 正确的 lidar2d 数据获取方式
        # get_scan() 返回一个 dict，包含 'range' 键
        lidar_data = self.env.get_lidar_scan(id=robot.id)
        # lidar_data 是 dict: {'range': array, 'angle': array, 'points': array}

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

        # ② 再用 ir-sim 的 done() 判断碰撞（到达目标已经处理过，剩下的 done=True 就是碰撞）
        elif self.env.done():
            reward = -100.0
            done = True
            info['result'] = 'collision'

        # ③ 正常步骤
        else:
            reward = -0.01
            info['result'] = 'running'

        return reward, done, info

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

    def close(self):
        if self.env is not None:
            self.env.end()

# ===================== main part ========================


# 创建训练环境（可并行多个加速训练）
# render_mode=None → 自动开启 display=False, disable_all_plot=True，后台无渲染
train_env = make_vec_env(
    IrSimNavEnv,
    n_envs=4,
    env_kwargs={'render_mode': None},  # 无渲染，加速训练
)


eval_env = Monitor(IrSimNavEnv(render_mode=None))  # 评估时也不需要渲染

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
