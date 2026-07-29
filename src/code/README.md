# End-to-End Navigation for an AMR with RL

## Overview

This project implements **Stall-Triggered Policy Switching (STPS)**, a runtime strategy that improves end-to-end navigation success by dynamically switching between two complementary CNNTD3 policies based on the robot's real-time behavior.

## Method

### Problem

A single CNNTD3 RL policy trained for general navigation often fails in specific hard scenarios:
- **U-trap / Double-U**: Robot gets stuck in concave obstacle structures and oscillates.
- **Narrow door**: Precision navigation is required; over-exploration causes collisions.

### STPS (Stall-Triggered Policy Switching)

Instead of a single policy, STPS maintains **two models** and switches at runtime:

| Role | Model | Training Goal |
|------|-------|---------------|
| **Main Policy** | `CNNTD3_v7_finetune_best` | Precise navigation with fine-tuned exploration annealing |
| **Escape Policy** | `CNNTD3_improved` | Strong exploration to escape traps |

The switching logic (v3) uses a **three-level detection** chain executed on the main policy:

1. **Position Stall** — If the robot moves less than 0.15 m in the last 20 steps, it is stuck.
2. **Oscillation** — If the displacement direction reverses ≥ 5 times in a 12-step window, the robot is bouncing against walls.
3. **Goal-Distance Stall** — If the distance to goal changes less than 0.05 m over 30 steps while still far (>0.5 m), progress has stopped.

When any trigger fires, the controller switches to the escape policy for a progressive duration (120 → 180 → 240 steps on repeated triggers), with a 30-step cooldown after returning to the main policy.

```
         ┌─────────────┐    stall/oscillation/goal_stall    ┌─────────────┐
         │             │ ──────────────────────────────────▶ │             │
         │  MAIN (v7)  │                                     │ ESCAPE (imp)│
         │  precise    │ ◀────────────────────────────────── │  explore    │
         │  navigation │      escaped ≥ 0.5 m or timeout     │  escape     │
         └─────────────┘                                     └─────────────┘
```

## Architecture

```
src/code/
├── robot_nav/                    # Core package (simulation, models, utilities)
│   ├── sim/                      # IR-SIM-based simulation environments
│   │   ├── sim_env.py            # Abstract base class (SIM_ENV)
│   │   ├── sim.py                # Single-robot environment (SIM)
│   │   └── marl_sim.py           # Multi-robot environment (MARL_SIM)
│   ├── models/                   # RL policy implementations
│   │   ├── cnntd3.py             # CNN-augmented TD3 (primary model)
│   │   ├── td3.py, ddpg.py       # Baseline TD3 / DDPG
│   │   ├── ppo.py, rcpg.py       # PPO / RCPG (for comparison)
│   │   ├── SAC/                  # Soft Actor-Critic
│   │   ├── MARL/                 # Multi-agent RL (MARL-TD3 + IGS attention)
│   │   └── HCM/                  # Hard-coded model baseline
│   ├── worlds/                   # YAML scene configuration files
│   ├── assets/                   # Pre-recorded experience data
│   ├── utils.py                  # Pretraining utilities, buffer helpers
│   ├── replay_buffer.py          # Experience replay buffers
│   └── eval_points.yaml          # Fixed evaluation positions
│
├── scripts/
│   ├── train/                    # Training scripts
│   │   ├── rl_train_v7_finetune.py   # → CNNTD3_v7_finetune_best (main policy)
│   │   └── rl_train_v6_anneal.py     # → CNNTD3_improved (escape policy)
│   ├── eval/                     # STPS evaluation scripts
│   │   ├── script3_stps_v3.py    # STPS v3: three-level switching evaluation
│   │   ├── eval_diagnose.py      # STPS diagnostic & parameter tuning
│   │   └── eval_stps_sensitivity.py  # Grid search over stall parameters
│   └── test/                     # Comparison benchmarks
│       ├── eval_unified.py       # CNNTD3 baseline vs STPS v2 (3 seeds)
│       └── script1_full_comparison.py  # Single-seed quick comparison
│
├── Wenjing_Chen/                 # Contributor workspace
└── Shengqin_Jiang/               # Contributor workspace
```

## CNNTD3 Model

The **CNNTD3** (CNN-augmented Twin Delayed DDPG) is the core RL agent:

- **State** (185-d): 180-beam LIDAR scan + distance to goal + cos/sin of goal heading + collision flag + goal flag + previous action
- **Action** (2-d): linear velocity [0, 0.5] m/s + angular velocity [-1, 1] rad/s
- **Architecture**: 1D CNN processes LIDAR → fused with goal/action embeddings → FC layers → action output
- **Training**: TD3 algorithm with clipped double Q-learning, target policy smoothing, delayed actor updates

## Workflow

### 1. Train Models

```bash
# Train the main policy (fine-tune from v5 checkpoint)
cd src/code
python scripts/train/rl_train_v7_finetune.py

# Train the escape policy (exploration annealing)
python scripts/train/rl_train_v6_anneal.py
```

Training uses curriculum learning: 65% standard scenes + 35% hard scenes (U-trap, U-shape). Exploration bonus rewards visiting new positions and penalizes stalling.

Model checkpoints are saved to `models/CNNTD3/checkpoint/`.

### 2. Run STPS Evaluation

```bash
cd src/code

# Full STPS v3 evaluation on 4 hard scenarios + standard environment
python scripts/eval/script3_stps_v3.py

# Compare CNNTD3 baseline vs STPS v2 (3 random seeds)
python scripts/test/eval_unified.py
```

### 3. Tune STPS Parameters

```bash
# Grid search: STALL_WINDOW × STALL_DIST on U-trap + Narrow door
python scripts/eval/eval_stps_sensitivity.py

# Diagnose U-trap escape behavior
python scripts/eval/eval_diagnose.py
```

## Key STPS v3 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STALL_WINDOW` | 20 | Steps to monitor for position stall |
| `STALL_DIST` | 0.15 m | Minimum movement to not be considered stalled |
| `OSC_WINDOW` | 12 | Steps to monitor for oscillation |
| `OSC_REVERSAL_THRESH` | 5 | Direction reversals triggering oscillation |
| `GOAL_STALL_WINDOW` | 30 | Steps to monitor goal-distance progress |
| `GOAL_STALL_THRESH` | 0.05 m | Minimum goal distance change |
| `ESCAPE_SCHEDULE` | [120, 180, 240] | Escape duration per trigger count |
| `PROGRESS_DIST` | 0.5 m | Movement required to confirm escape |
| `COOLDOWN_STEPS` | 30 | Post-escape delay before re-detection |

## Scenarios

| Scene | World File | Challenge |
|-------|------------|-----------|
| U-trap | `robot_nav/worlds/u_trap_world.yaml` | Concave trap; robot must reverse out |
| Double-U | `robot_nav/worlds/double_u_world.yaml` | Two consecutive U-shaped traps |
| Narrow Door | `robot_nav/worlds/narrow_door_world.yaml` | Tight passage requires precision |
| Symmetric Corridor | `robot_nav/worlds/symmetric_corridor_world.yaml` | Long corridor with obstacles |
| Standard | `robot_nav/worlds/robot_world.yaml` | 10×10 m with random obstacles and dynamic agents |

## Dependencies

```
torch    # RL model training & inference
numpy    # Numerical operations
pyyaml   # YAML config parsing
tensorboard  # Training metrics logging
matplotlib   # Result visualization
tqdm     # Progress bars
irsim    # IR-SIM simulator
```

Install: `pip install -r requirements.txt`
