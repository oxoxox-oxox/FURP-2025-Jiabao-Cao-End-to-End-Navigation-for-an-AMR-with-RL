"""
RCPG 困难场景全量测试
用法：
  conda activate neupan
  cd ~/DRL-robot-navigation-IR-SIM
  python test_rcpg_hard_scenarios.py
"""
import sys, os, numpy as np, torch, csv
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.SIM_ENV.sim import SIM

HISTORY_LEN = 10  # 和 rnn_train.py 一致
MAX_STEPS = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

model = RCPG(
    state_dim=185,
    action_dim=2,
    max_action=1,
    device=device,
    load_model=True,
    rnn="gru",
)

# ─── 所有困难场景定义 ─────────────────────────────────────────────

SCENARIOS = {
    "S1_U_trap": {
        "world": "robot_nav/worlds/u_trap_world.yaml",
        "max_steps": 300,
        "cases": [
            ([[7.5],[5.0],[0.0]],   [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[1.57]],  [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[3.14]],  [[9.0],[5.0],[0]]),
            ([[7.5],[5.0],[-1.57]], [[9.0],[5.0],[0]]),
            ([[7.5],[6.0],[0.0]],   [[9.0],[5.0],[0]]),
            ([[7.5],[4.0],[0.0]],   [[9.0],[5.0],[0]]),
            ([[7.0],[5.0],[3.14]],  [[9.0],[5.0],[0]]),
            ([[7.5],[5.5],[1.57]],  [[9.0],[5.0],[0]]),
            ([[7.5],[4.5],[-1.57]], [[9.0],[5.0],[0]]),
            ([[7.0],[5.0],[2.0]],   [[9.0],[5.0],[0]]),
        ],
    },
    "S2_double_U": {
        "world": "robot_nav/worlds/double_u_world.yaml",
        "max_steps": 300,
        "cases": [
            ([[5.0],[5.0],[0.0]],   [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[1.57]],  [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[-1.57]], [[9.0],[5.0],[0]]),
            ([[5.0],[5.0],[3.14]],  [[9.0],[5.0],[0]]),
            ([[5.0],[6.0],[1.57]],  [[9.0],[5.0],[0]]),
            ([[5.0],[4.0],[-1.57]], [[9.0],[5.0],[0]]),
        ],
    },
    "S3_narrow_door_0.45m": {
        "world": "robot_nav/worlds/narrow_door_world.yaml",
        "max_steps": 300,
        "cases": [
            ([[2.0],[5.0],[0.0]],   [[8.0],[5.0],[0]]),
            ([[2.0],[5.0],[0.3]],   [[8.0],[5.0],[0]]),
            ([[2.0],[5.0],[-0.3]],  [[8.0],[5.0],[0]]),
            ([[2.0],[6.0],[0.0]],   [[8.0],[5.0],[0]]),
            ([[2.0],[4.0],[0.0]],   [[8.0],[5.0],[0]]),
            ([[2.0],[7.0],[0.0]],   [[8.0],[5.0],[0]]),
            ([[2.0],[3.0],[0.0]],   [[8.0],[5.0],[0]]),
        ],
    },
    "S4_dead_end_maze": {
        "world": "robot_nav/worlds/dead_end_maze_world.yaml",
        "max_steps": 500,
        "cases": [
            ([[1.5],[8.0],[0.0]],    [[9.0],[6.0],[0]]),
            ([[1.5],[8.0],[1.57]],   [[9.0],[6.0],[0]]),
            ([[1.5],[8.0],[-1.57]],  [[9.0],[6.0],[0]]),
            ([[1.5],[8.0],[3.14]],   [[9.0],[6.0],[0]]),
            ([[1.5],[2.0],[0.0]],    [[9.0],[6.0],[0]]),
            ([[1.5],[6.0],[0.0]],    [[9.0],[6.0],[0]]),
        ],
    },
    "S5_symmetric_corridor": {
        "world": "robot_nav/worlds/symmetric_corridor_world.yaml",
        "max_steps": 300,
        "cases": [
            ([[1.0],[5.0],[0.0]],   [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[1.57]],  [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[-1.57]], [[9.0],[5.0],[0]]),
            ([[1.0],[5.0],[3.14]],  [[9.0],[5.0],[0]]),
            ([[1.0],[5.1],[0.0]],   [[9.0],[5.0],[0]]),
            ([[1.0],[4.9],[0.0]],   [[9.0],[5.0],[0]]),
        ],
    },
}

# ─── CNNTD3 已有结果（用于对比输出）─────────────────────────────

CNNTD3_RESULTS = {
    "S1_U_trap":              {"SR": "0%",    "CR": "0%",     "TR": "100%"},
    "S2_double_U":            {"SR": "33%",   "CR": "0%",     "TR": "67%"},
    "S3_narrow_door_0.45m":   {"SR": "4.8%",  "CR": "95.2%",  "TR": "0%"},
    "S4_dead_end_maze":       {"SR": "67%",   "CR": "0%",     "TR": "33%"},
    "S5_symmetric_corridor":  {"SR": "83%",   "CR": "0%",     "TR": "17%"},
}

# ─── 测试循环 ────────────────────────────────────────────────────

all_results = []
summary = {}

for scene_name, scene_cfg in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"  {scene_name}")
    print(f"{'='*60}")

    sim = SIM(world_file=scene_cfg["world"])
    max_steps = scene_cfg["max_steps"]
    total_goal, total_col, n = 0, 0, 0

    for idx, (robot_state, robot_goal) in enumerate(scene_cfg["cases"]):
        for rep in range(3):
            state_queue = deque(maxlen=HISTORY_LEN)
            fill_state = True

            scan, dist, cos, sin, col, goal, a, reward = sim.reset(
                robot_state=robot_state,
                robot_goal=robot_goal,
                random_obstacles=False,
            )

            done, steps = False, 0
            while not done and steps < max_steps:
                state, _ = model.prepare_state(scan, dist, cos, sin, col, goal, a)

                if fill_state:
                    state_queue.clear()
                    for _ in range(HISTORY_LEN):
                        state_queue.append(state)
                    fill_state = False
                state_queue.append(state)

                action = model.get_action(np.array(state_queue), False)
                a_in = [(action[0] + 1) / 4, action[1]]

                scan, dist, cos, sin, col, goal, a, reward = sim.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )
                steps += 1
                done = col or goal

            outcome = "goal" if goal else ("collision" if col else "timeout")
            total_goal += int(goal)
            total_col += int(col)
            n += 1

            theta = robot_state[2][0]
            print(f"  case={idx} rep={rep} | theta={theta:>6.2f} | "
                  f"steps={steps:>3} | {outcome}")

            all_results.append({
                "scene": scene_name, "case": idx, "rep": rep,
                "theta": theta, "steps": steps, "outcome": outcome,
            })

    sr = total_goal / n * 100
    cr = total_col / n * 100
    tr = (n - total_goal - total_col) / n * 100
    summary[scene_name] = {"SR": f"{sr:.1f}%", "CR": f"{cr:.1f}%", "TR": f"{tr:.1f}%"}
    print(f"\n  RCPG:   SR={sr:.1f}%  CR={cr:.1f}%  TR={tr:.1f}%")

    cnn = CNNTD3_RESULTS.get(scene_name, {})
    print(f"  CNNTD3: SR={cnn.get('SR','?')}  CR={cnn.get('CR','?')}  TR={cnn.get('TR','?')}")

# ─── 汇总对比表 ──────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  RCPG vs CNNTD3 全场景对比")
print(f"{'='*70}")
print(f"{'场景':<28} {'RCPG SR':<12} {'CNNTD3 SR':<12} {'差异'}")
print(f"{'-'*70}")

for scene_name in SCENARIOS:
    rcpg_sr = summary[scene_name]["SR"]
    cnntd3_sr = CNNTD3_RESULTS.get(scene_name, {}).get("SR", "?")
    rcpg_val = float(rcpg_sr.replace("%", ""))
    try:
        cnntd3_val = float(cnntd3_sr.replace("%", ""))
        diff = rcpg_val - cnntd3_val
        diff_str = f"{'+' if diff > 0 else ''}{diff:.1f}%"
    except:
        diff_str = "?"
    print(f"  {scene_name:<26} {rcpg_sr:<12} {cnntd3_sr:<12} {diff_str}")

print(f"{'='*70}")

# ─── 保存 CSV ────────────────────────────────────────────────────

csv_path = "rcpg_hard_scenario_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_results[0].keys())
    w.writeheader()
    w.writerows(all_results)
print(f"\n详细结果保存到 {csv_path}")
