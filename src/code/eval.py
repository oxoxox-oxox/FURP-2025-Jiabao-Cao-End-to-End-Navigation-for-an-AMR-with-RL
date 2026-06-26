"""
Evaluation / visualization script for trained PPO navigation model.

Usage:
    python eval.py [model_path]

Default model: nav_ppo_final.pt
"""

import sys

# ===== Set matplotlib backend BEFORE any library that imports pyplot =====
# Must be at the very top — once matplotlib is imported, the backend is locked.
import matplotlib
try:
    matplotlib.use('TkAgg')   # cross-platform, ships with tkinter
except ImportError:
    try:
        matplotlib.use('QtAgg')   # fallback: requires PyQt/PySide
    except ImportError:
        print("[WARN] No GUI backend available (TkAgg, QtAgg). "
              "Rendering will be disabled.")
# ========================================================================

from rl_ppo import PPO
from env import IrSimNavEnv


# ===================== Evaluation ========================

model_path = sys.argv[1] if len(sys.argv) > 1 else 'nav_ppo_final.pt'
print(f"Loading model: {model_path}")
model = PPO.load(model_path)

# Create a fresh environment for visualization
# (GUI backend was set at the top of this file, before import irsim)
env = IrSimNavEnv(render_mode='human')

print("\n=== Evaluation: running 3 episodes with deterministic policy ===\n")
episode_count = 0
obs, _ = env.reset()
step_count = 0

while episode_count < 3:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    step_count += 1

    if done or truncated:
        episode_count += 1
        result = info.get('result', 'timeout')
        print(f"Episode {episode_count}: result={result}, "
              f"steps={step_count}, reward={reward:.2f}")
        obs, _ = env.reset()
        step_count = 0

env.close()
print("\nEvaluation complete.")
