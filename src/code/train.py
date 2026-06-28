"""
Training script for PPO navigation on ir-sim.

Each run creates tb_logs/experiment_N/ for TensorBoard logs.

Usage:
    python train.py
"""

import os
import re
from rl_ppo import PPO
from env import IrSimNavEnv


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

tb_dir = next_experiment_dir()
print(f"[Experiment] TensorBoard log dir: {tb_dir}")

# ===================== Training ========================

# Create PPO agent with custom PyTorch implementation
model = PPO(
    env=IrSimNavEnv,
    env_kwargs={'render_mode': None},
    n_envs=1,                  # irsim 不支持单进程多实例，多个 make() 会堆叠到同一世界
    learning_rate=3e-4,        # actor LR (standard PPO default)
    critic_lr=1e-3,            # critic LR (higher: critic must fit faster to track returns)
    n_steps=2048,            # rollout steps per collection
    batch_size=64,           # mini-batch size
    n_epochs=10,             # optimization epochs per rollout
    gamma=0.99,              # discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_range=0.1,          # PPO clip range (reduced: prevents policy collapse)
    ent_coef=0.005,          # minimal entropy bonus: prevent total std collapse
    target_kl=None,          # full n_epochs for thorough learning
    device='cuda',
    tensorboard_log=tb_dir,
)

# Train (no separate eval env — irsim is a global singleton)
model.learn(
    total_timesteps=500_000,  # 平衡: 足够学习但不过度坍缩 std
    eval_env=None,           # 不能用单独的 eval env——irsim 全局共享仿真
    eval_freq=5000,
    n_eval_episodes=10,
    best_model_save_path='./models/',
)

# Save final model
model.save('nav_ppo_final.pt')
print("\nTraining done. Model saved to nav_ppo_final.pt")
print("Run 'python eval.py' to visualize the trained policy.")
