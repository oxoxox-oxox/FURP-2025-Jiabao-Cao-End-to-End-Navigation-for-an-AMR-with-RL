"""
Generate representative trajectory figures for the final report/poster.

Run from the working experiment environment:

    conda activate <your_env>
    cd ~/DRL-robot-navigation-IR-SIM
    python ~/furp-2026-WenjingChen-RL-Navigation-for-AMR/src/code/Wenjing_Chen/CNNTD3/final_stps/generate_case_figures.py

Outputs are written to:

    ~/furp-2026-WenjingChen-RL-Navigation-for-AMR/docs/img/cases/
"""
import os
import sys
from collections import deque

import numpy as np
import torch
import yaml

# The original checkpoints may have been saved from CUDA. This keeps figure
# generation usable on CPU-only/headless machines without editing model code.
_orig_torch_load = torch.load


def _cpu_torch_load(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _orig_torch_load(*args, **kwargs)


torch.load = _cpu_torch_load


WORK_DIR = os.path.expanduser("~/DRL-robot-navigation-IR-SIM")
REPO_DIR = os.path.expanduser("~/furp-2026-WenjingChen-RL-Navigation-for-AMR")
OUT_DIR = os.path.join(REPO_DIR, "docs", "img", "cases")

sys.path.insert(0, os.path.join(WORK_DIR, "robot_nav"))
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
os.environ.pop("DISPLAY", None)

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3


device = torch.device("cpu")

STALL_WINDOW = 20
STALL_DIST = 0.15
BASE_ESCAPE_STEPS = 120
PROGRESS_DIST = 0.5
OSC_WINDOW = 12
OSC_REVERSAL_THRESH = 5
OSC_MIN_STEPS = 8

SCENARIOS = {
    "S1_U_trap": {
        "world": "robot_nav/worlds/u_trap_world.yaml",
        "start": [[7.5], [5.0], [0.0]],
        "goal": [[9.0], [5.0], [0.0]],
        "max_steps": 500,
    },
    "S2_Double_U": {
        "world": "robot_nav/worlds/double_u_world.yaml",
        "start": [[5.0], [5.0], [0.0]],
        "goal": [[9.0], [5.0], [0.0]],
        "max_steps": 500,
    },
    "S3_Narrow_door": {
        "world": "robot_nav/worlds/narrow_door_world.yaml",
        "start": [[2.0], [5.0], [0.0]],
        "goal": [[8.0], [5.0], [0.0]],
        "max_steps": 500,
    },
    "S5_Corridor": {
        "world": "robot_nav/worlds/symmetric_corridor_world.yaml",
        "start": [[1.0], [5.0], [0.0]],
        "goal": [[9.0], [5.0], [0.0]],
        "max_steps": 500,
    },
}


def detect_oscillation(pos_history):
    if len(pos_history) < OSC_WINDOW:
        return False
    recent = list(pos_history)[-OSC_WINDOW:]
    reversals = 0
    prev_dx, prev_dy = None, None
    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i - 1][0]
        dy = recent[i][1] - recent[i - 1][1]
        if prev_dx is not None and dx * prev_dx + dy * prev_dy < 0:
            reversals += 1
        prev_dx, prev_dy = dx, dy
    return reversals >= OSC_REVERSAL_THRESH


def load_models():
    ckpt_model = "models/CNNTD3/checkpoint"
    ckpt_robot = "robot_nav/models/CNNTD3/checkpoint"

    baseline = CNNTD3(
        state_dim=185, action_dim=2, max_action=1, device=device,
        load_model=False, model_name="case_baseline"
    )
    try:
        baseline.load("CNNTD3", ckpt_model)
    except Exception:
        baseline.load("CNNTD3", ckpt_robot)
    baseline.actor.eval()

    precise = CNNTD3(
        state_dim=185, action_dim=2, max_action=1, device=device,
        load_model=False, model_name="case_precise"
    )
    precise.load("CNNTD3_v7_finetune_best", ckpt_model)
    precise.actor.eval()

    explore = CNNTD3(
        state_dim=185, action_dim=2, max_action=1, device=device,
        load_model=False, model_name="case_explore"
    )
    try:
        explore.load("CNNTD3_improved", ckpt_model)
    except Exception:
        explore.load("CNNTD3_improved", ckpt_robot)
    explore.actor.eval()

    return baseline, precise, explore


def run_single_policy(model, cfg):
    sim = SIM(world_file=cfg["world"], disable_plotting=True)
    scan, dist, cos, sin, col, goal, action, reward = sim.reset(
        robot_state=cfg["start"], robot_goal=cfg["goal"], random_obstacles=False
    )
    prev = [0.0, 0.0]
    trace = []
    rewards = []

    for step in range(cfg["max_steps"]):
        rs = sim.env.get_robot_state()
        trace.append([float(rs[0].item()), float(rs[1].item())])

        state, _ = model.prepare_state(scan, dist, cos, sin, col, goal, prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0] + 1) / 4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan, dist, cos, sin, col, goal, action, reward = sim.step(lin, ang)
        rewards.append(float(reward))

        if goal:
            sim.env.end()
            return {"outcome": "goal", "trace": trace, "steps": step + 1, "return": sum(rewards)}
        if col:
            sim.env.end()
            return {"outcome": "collision", "trace": trace, "steps": step + 1, "return": sum(rewards)}

    sim.env.end()
    return {"outcome": "timeout", "trace": trace, "steps": cfg["max_steps"], "return": sum(rewards)}


def run_stps(precise, explore, cfg):
    sim = SIM(world_file=cfg["world"], disable_plotting=True)
    scan, dist, cos, sin, col, goal, action, reward = sim.reset(
        robot_state=cfg["start"], robot_goal=cfg["goal"], random_obstacles=False
    )
    prev = [0.0, 0.0]
    pos_hist = deque(maxlen=max(STALL_WINDOW, OSC_WINDOW + 2))
    trace = []
    modes = []
    rewards = []
    mode = "main"
    esc_cnt = 0
    esc_start = None
    switches = 0
    steps_main = 0
    esc_steps = BASE_ESCAPE_STEPS

    for step in range(cfg["max_steps"]):
        rs = sim.env.get_robot_state()
        cp = np.array([float(rs[0].item()), float(rs[1].item())])
        trace.append(cp.tolist())
        modes.append(mode)
        pos_hist.append(cp)

        if mode == "main":
            steps_main += 1
            trigger = False
            if len(pos_hist) >= STALL_WINDOW:
                trigger = np.linalg.norm(pos_hist[-1] - pos_hist[-STALL_WINDOW]) < STALL_DIST
            if not trigger and steps_main > OSC_MIN_STEPS:
                trigger = detect_oscillation(pos_hist)
            if trigger:
                mode = "escape"
                esc_cnt = 0
                esc_start = cp.copy()
                switches += 1
                steps_main = 0
                pos_hist.clear()
                if switches > 1:
                    esc_steps = min(BASE_ESCAPE_STEPS * 2, 240)
        else:
            esc_cnt += 1
            moved = np.linalg.norm(cp - esc_start)
            if esc_cnt >= esc_steps and moved > PROGRESS_DIST:
                mode = "main"
                steps_main = 0
                pos_hist.clear()
            elif esc_cnt >= esc_steps * 3:
                mode = "main"
                steps_main = 0
                pos_hist.clear()

        model = precise if mode == "main" else explore
        state, _ = model.prepare_state(scan, dist, cos, sin, col, goal, prev)
        action = model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0] + 1) / 4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan, dist, cos, sin, col, goal, action, reward = sim.step(lin, ang)
        rewards.append(float(reward))

        if goal:
            sim.env.end()
            return {
                "outcome": "goal", "trace": trace, "modes": modes,
                "steps": step + 1, "switches": switches, "return": sum(rewards)
            }
        if col:
            sim.env.end()
            return {
                "outcome": "collision", "trace": trace, "modes": modes,
                "steps": step + 1, "switches": switches, "return": sum(rewards)
            }

    sim.env.end()
    return {
        "outcome": "timeout", "trace": trace, "modes": modes,
        "steps": cfg["max_steps"], "switches": switches, "return": sum(rewards)
    }


def svg_text(x, y, text, size=14, anchor="middle", weight="normal", color="#222"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{text}</text>'
    )


def world_to_svg_elements(world_file, scale, margin, height_px):
    with open(world_file, "r") as f:
        world = yaml.safe_load(f)

    width = world.get("world", {}).get("width", 10)
    height = world.get("world", {}).get("height", 10)
    elems = []

    def sx(x):
        return margin + x * scale

    def sy(y):
        return height_px - margin - y * scale

    for obs in world.get("obstacle", []):
        shape = obs.get("shape", {})
        state = obs.get("state", [0, 0, 0])
        name = shape.get("name")
        if name == "rectangle":
            length = float(shape.get("length", 1.0))
            rect_width = float(shape.get("width", 1.0))
            x, y = float(state[0]), float(state[1])
            elems.append(
                f'<rect x="{sx(x - length / 2):.1f}" y="{sy(y + rect_width / 2):.1f}" '
                f'width="{length * scale:.1f}" height="{rect_width * scale:.1f}" '
                f'fill="#9aa3ad" stroke="#515760" stroke-width="1.4"/>'
            )
        elif name == "circle":
            radius = float(shape.get("radius", 0.2))
            x, y = float(state[0]), float(state[1])
            elems.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius * scale:.1f}" '
                f'fill="#9aa3ad" stroke="#515760" stroke-width="1.4"/>'
            )
        elif name == "linestring":
            vertices = np.array(shape.get("vertices", []), dtype=float)
            if len(vertices) > 0:
                pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in vertices[:, :2])
                elems.append(f'<polyline points="{pts}" fill="none" stroke="#515760" stroke-width="1.5"/>')

    return elems, width, height


def plot_case(filename, title, cfg, result):
    path = np.array(result["trace"], dtype=float)
    svg_w, svg_h = 720, 780
    margin = 70
    elems, world_w, world_h = world_to_svg_elements(cfg["world"], 58, margin, svg_h - 80)
    scale = min((svg_w - 2 * margin) / world_w, (svg_h - 170) / world_h)
    elems, world_w, world_h = world_to_svg_elements(cfg["world"], scale, margin, svg_h - 80)

    def sx(x):
        return margin + x * scale

    def sy(y):
        return svg_h - 80 - y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(svg_w / 2, 30, title, 20, weight="bold"),
        svg_text(
            svg_w / 2, 55,
            f"{result['outcome']}, steps={result['steps']}, return={result['return']:.1f}"
            + (f", switches={result['switches']}" if "switches" in result else ""),
            13,
            color="#444",
        ),
    ]

    # Grid and bounds.
    for gx in range(0, int(world_w) + 1):
        parts.append(f'<line x1="{sx(gx):.1f}" y1="{sy(0):.1f}" x2="{sx(gx):.1f}" y2="{sy(world_h):.1f}" stroke="#d5dae0" stroke-width="0.6"/>')
    for gy in range(0, int(world_h) + 1):
        parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(gy):.1f}" x2="{sx(world_w):.1f}" y2="{sy(gy):.1f}" stroke="#d5dae0" stroke-width="0.6"/>')
    parts.extend(elems)

    def polyline(points, color, width=3.0, opacity=1.0):
        if len(points) < 2:
            return ""
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"/>'

    if len(path) > 1 and "modes" in result:
        modes = result["modes"]
        main_pts = np.array([p for p, m in zip(path, modes) if m == "main"])
        esc_pts = np.array([p for p, m in zip(path, modes) if m == "escape"])
        parts.append(polyline(path, "#2f4858", 1.5, 0.35))
        parts.append(polyline(main_pts, "#277da1", 4.0, 1.0))
        parts.append(polyline(esc_pts, "#f3722c", 4.0, 1.0))
    elif len(path) > 1:
        parts.append(polyline(path, "#277da1", 4.0, 1.0))

    start = np.array(cfg["start"], dtype=float).reshape(-1)
    goal = np.array(cfg["goal"], dtype=float).reshape(-1)
    parts.append(f'<circle cx="{sx(start[0]):.1f}" cy="{sy(start[1]):.1f}" r="7" fill="#43aa8b" stroke="white" stroke-width="2"/>')
    parts.append(f'<polygon points="{sx(goal[0]):.1f},{sy(goal[1])-10:.1f} {sx(goal[0])+9:.1f},{sy(goal[1])+8:.1f} {sx(goal[0])-9:.1f},{sy(goal[1])+8:.1f}" fill="#f94144" stroke="white" stroke-width="1.5"/>')

    legend_y = svg_h - 38
    parts.append(f'<circle cx="95" cy="{legend_y}" r="6" fill="#43aa8b"/>')
    parts.append(svg_text(112, legend_y + 5, "start", 12, anchor="start"))
    parts.append(f'<polygon points="180,{legend_y-8} 188,{legend_y+7} 172,{legend_y+7}" fill="#f94144"/>')
    parts.append(svg_text(198, legend_y + 5, "goal", 12, anchor="start"))
    parts.append(f'<line x1="270" y1="{legend_y}" x2="310" y2="{legend_y}" stroke="#277da1" stroke-width="4"/>')
    parts.append(svg_text(318, legend_y + 5, "precision / trajectory", 12, anchor="start"))
    if "modes" in result:
        parts.append(f'<line x1="485" y1="{legend_y}" x2="525" y2="{legend_y}" stroke="#f3722c" stroke-width="4"/>')
        parts.append(svg_text(533, legend_y + 5, "escape", 12, anchor="start"))

    parts.append("</svg>")
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        f.write("\n".join(parts))


def plot_neupan_failure_panel():
    rows = [("U-trap", "0%"), ("Double-U", "0%"), ("Narrow door", "0%"), ("Corridor", "0%")]
    width, height = 720, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 55, "NeuPAN Compact-scene Failure Evidence", 22, weight="bold"),
        '<rect x="110" y="135" width="500" height="320" fill="#f5f7fa" stroke="#515760" stroke-width="1.2"/>',
        svg_text(235, 175, "Scenario", 16, weight="bold"),
        svg_text(475, 175, "SR in tested setup", 16, weight="bold"),
        '<line x1="110" y1="200" x2="610" y2="200" stroke="#515760" stroke-width="1.2"/>',
    ]
    for i, (scenario, sr) in enumerate(rows):
        y = 245 + i * 58
        parts.append(svg_text(235, y, scenario, 15))
        parts.append(svg_text(475, y, sr, 15, weight="bold", color="#9b3737"))
    parts.append(svg_text(width / 2, 550, "Report as compact-benchmark / configuration mismatch,", 15))
    parts.append(svg_text(width / 2, 576, "not as a general claim that NeuPAN fails.", 15))
    parts.append("</svg>")
    with open(os.path.join(OUT_DIR, "failure_neupan_compact.svg"), "w") as f:
        f.write("\n".join(parts))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline, precise, explore = load_models()

    cases = [
        ("success_u_trap.svg", "STPS success: U-trap", SCENARIOS["S1_U_trap"], "stps"),
        ("success_double_u.svg", "STPS success: Double-U", SCENARIOS["S2_Double_U"], "stps"),
        ("success_narrow_door.svg", "STPS success: narrow door", SCENARIOS["S3_Narrow_door"], "stps"),
        ("failure_baseline_u_trap.svg", "CNNTD3 failure: U-trap", SCENARIOS["S1_U_trap"], "baseline"),
        ("failure_explore_narrow_door.svg", "Exploration-policy failure: narrow door", SCENARIOS["S3_Narrow_door"], "explore"),
    ]

    for filename, title, cfg, method in cases:
        if method == "stps":
            result = run_stps(precise, explore, cfg)
        elif method == "baseline":
            result = run_single_policy(baseline, cfg)
        elif method == "explore":
            result = run_single_policy(explore, cfg)
        else:
            raise ValueError(method)
        plot_case(filename, title, cfg, result)
        print(f"{filename}: {result['outcome']}, steps={result['steps']}")

    plot_neupan_failure_panel()
    print(f"\nSaved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
