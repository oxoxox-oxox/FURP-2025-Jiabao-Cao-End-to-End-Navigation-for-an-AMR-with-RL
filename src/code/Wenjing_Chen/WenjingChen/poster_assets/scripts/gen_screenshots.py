"""
生成poster需要的6张轨迹截图
每个case跑一次，保存轨迹图为png
"""
import sys, os, numpy as np, torch
from collections import deque

sys.path.insert(0, 'robot_nav')
os.chdir(os.path.expanduser('~/DRL-robot-navigation-IR-SIM'))

from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import matplotlib
matplotlib.use('Agg')  # 无显示器保存图片
import matplotlib.pyplot as plt

device = torch.device('cpu')

# 加载模型
ckpt = 'models/CNNTD3/checkpoint'
ckpt_r = 'robot_nav/models/CNNTD3/checkpoint'

m_base = CNNTD3(state_dim=185, action_dim=2, max_action=1, device=device, load_model=False, model_name="s1")
try: m_base.load("CNNTD3", ckpt)
except: m_base.load("CNNTD3", ckpt_r)
m_base.actor.eval()

m_v7 = CNNTD3(state_dim=185, action_dim=2, max_action=1, device=device, load_model=False, model_name="s2")
m_v7.load("CNNTD3_v7_finetune_best", ckpt)
m_v7.actor.eval()

m_imp = CNNTD3(state_dim=185, action_dim=2, max_action=1, device=device, load_model=False, model_name="s3")
try: m_imp.load("CNNTD3_improved", ckpt)
except: m_imp.load("CNNTD3_improved", ckpt_r)
m_imp.actor.eval()

print("✅ Models loaded")

# STPS v2 参数
STALL_WINDOW = 20; STALL_DIST = 0.15
BASE_ESCAPE_STEPS = 120; PROGRESS_DIST = 0.5
OSC_WINDOW = 12; OSC_REVERSAL_THRESH = 5; OSC_MIN_STEPS = 8

def detect_oscillation(pos_history):
    if len(pos_history) < OSC_WINDOW: return False
    recent = list(pos_history)[-OSC_WINDOW:]
    rev=0; pdx,pdy=None,None
    for i in range(1,len(recent)):
        dx=recent[i][0]-recent[i-1][0]; dy=recent[i][1]-recent[i-1][1]
        if pdx is not None and dx*pdx+dy*pdy<0: rev+=1
        pdx,pdy=dx,dy
    return rev >= OSC_REVERSAL_THRESH


def run_and_record(model, world, robot_state, robot_goal, max_steps, mode='single'):
    """运行一个episode，记录轨迹"""
    sim = SIM(world_file=world, disable_plotting=True)
    scan,d,c,s,col,g,a,r = sim.reset(robot_state=robot_state, robot_goal=robot_goal, random_obstacles=False)
    prev = [0.0, 0.0]
    trajectory = []
    switch_points = []  # (step, pos, 'to_escape'/'to_main')

    # STPS state
    pos_hist = deque(maxlen=max(STALL_WINDOW, OSC_WINDOW+2))
    stps_mode = 'main'; esc_cnt = 0; esc_start = None; switches = 0; steps_main = 0
    esc_steps = BASE_ESCAPE_STEPS

    for step in range(max_steps):
        rs = sim.env.get_robot_state()
        cp = np.array([rs[0].item(), rs[1].item()])
        trajectory.append(cp.copy())

        if mode == 'stps':
            pos_hist.append(cp)
            if stps_mode == 'main':
                steps_main += 1
                trig = False
                if len(pos_hist) >= STALL_WINDOW:
                    if np.linalg.norm(pos_hist[-1]-pos_hist[-STALL_WINDOW]) < STALL_DIST:
                        trig = True
                if not trig and steps_main > OSC_MIN_STEPS:
                    if detect_oscillation(pos_hist):
                        trig = True
                if trig:
                    stps_mode='escape';esc_cnt=0;esc_start=cp.copy()
                    switches+=1;steps_main=0;pos_hist.clear()
                    switch_points.append((step, cp.copy(), 'to_escape'))
                    if switches>1: esc_steps=min(BASE_ESCAPE_STEPS*2,240)
            else:
                esc_cnt+=1
                dd=np.linalg.norm(cp-esc_start)
                if esc_cnt>=esc_steps and dd>PROGRESS_DIST:
                    stps_mode='main';steps_main=0;pos_hist.clear()
                    switch_points.append((step, cp.copy(), 'to_main'))
                elif esc_cnt>=esc_steps*3:
                    stps_mode='main';steps_main=0;pos_hist.clear()
                    switch_points.append((step, cp.copy(), 'to_main'))

            active_model = m_v7 if stps_mode == 'main' else m_imp
        else:
            active_model = model

        state,_ = active_model.prepare_state(scan,d,c,s,col,g,prev)
        action = active_model.get_action(np.array(state), False)
        prev = list(action)
        lin = float(np.clip((action[0]+1)/4, 0, 0.5))
        ang = float(np.clip(action[1], -1, 1))
        scan,d,c,s,col,g,a,r = sim.step(lin, ang)

        if g:
            trajectory.append(np.array([sim.env.get_robot_state()[0].item(),
                                         sim.env.get_robot_state()[1].item()]))
            sim.env.end()
            return 'goal', np.array(trajectory), switch_points
        if col:
            sim.env.end()
            return 'collision', np.array(trajectory), switch_points
    sim.env.end()
    return 'timeout', np.array(trajectory), switch_points


def plot_trajectory(traj, switch_points, world_type, title, outcome, filename,
                    robot_start, robot_goal, obstacles=None):
    """画轨迹图"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2)

    # 画障碍物
    if world_type == 'u_trap':
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((7.85, 2.85), 0.3, 4.3, fill=True, color='#2D3748', alpha=0.8))
        ax.add_patch(Rectangle((2.3, 6.85), 5.7, 0.3, fill=True, color='#2D3748', alpha=0.8))
        ax.add_patch(Rectangle((2.3, 2.85), 5.7, 0.3, fill=True, color='#2D3748', alpha=0.8))
    elif world_type == 'narrow_door':
        ax.add_patch(Rectangle((4.8, 0), 0.4, 4.5, fill=True, color='#2D3748', alpha=0.8))
        ax.add_patch(Rectangle((4.8, 5.5), 0.4, 4.5, fill=True, color='#2D3748', alpha=0.8))

    # 画边界
    ax.plot([0,10,10,0,0], [0,0,10,10,0], 'k-', linewidth=2)

    # 画轨迹
    if len(switch_points) > 0:
        # STPS: 不同颜色表示不同模式
        segments = []
        seg_start = 0
        seg_mode = 'main'
        for step, pos, sw_type in switch_points:
            segments.append((seg_start, step, seg_mode))
            seg_mode = 'escape' if sw_type == 'to_escape' else 'main'
            seg_start = step
        segments.append((seg_start, len(traj)-1, seg_mode))

        for s_start, s_end, s_mode in segments:
            if s_end <= s_start: continue
            seg = traj[s_start:s_end+1]
            color = '#2B6CB0' if s_mode == 'main' else '#E53E3E'
            ax.plot(seg[:, 0], seg[:, 1], '-', color=color, linewidth=2, alpha=0.8)

        # 画切换点
        for step, pos, sw_type in switch_points:
            marker = 'v' if sw_type == 'to_escape' else '^'
            color = '#E53E3E' if sw_type == 'to_escape' else '#38A169'
            ax.plot(pos[0], pos[1], marker, color=color, markersize=10, zorder=5)

        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], color='#2B6CB0', linewidth=2, label='Precision (πP)'),
            Line2D([0],[0], color='#E53E3E', linewidth=2, label='Escape (πE)'),
            Line2D([0],[0], marker='v', color='#E53E3E', linestyle='None', markersize=8, label='→ escape'),
            Line2D([0],[0], marker='^', color='#38A169', linestyle='None', markersize=8, label='→ main'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    else:
        color = '#38A169' if outcome == 'goal' else '#E53E3E'
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, alpha=0.8)

    # 起点和终点
    ax.plot(robot_start[0], robot_start[1], 'o', color='#2B6CB0', markersize=12, zorder=10, label='Start')
    ax.plot(robot_goal[0], robot_goal[1], '*', color='#D69E2E', markersize=15, zorder=10, label='Goal')

    # 结果标注
    result_color = '#38A169' if outcome == 'goal' else '#E53E3E'
    result_text = {'goal': 'SUCCESS', 'collision': 'COLLISION', 'timeout': 'TIMEOUT'}[outcome]
    ax.text(0.98, 0.02, f'{result_text} ({len(traj)} steps)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold', color=result_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {filename}")


# ===== 生成6张截图 =====
output_dir = os.path.expanduser('~/poster_screenshots')
os.makedirs(output_dir, exist_ok=True)

cases = [
    {
        'name': '1_fail_baseline_utrap',
        'title': 'Failure: Baseline in U-trap',
        'model': m_base, 'mode': 'single',
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'world_type': 'u_trap',
        'start': [[7.5],[5.0],[0.0]], 'goal': [[9.0],[5.0],[0]],
        'start_xy': [7.5, 5.0], 'goal_xy': [9.0, 5.0],
    },
    {
        'name': '2_fail_explore_narrow',
        'title': 'Failure: Exploration in narrow door',
        'model': m_imp, 'mode': 'single',
        'world': 'robot_nav/worlds/narrow_door_world.yaml',
        'world_type': 'narrow_door',
        'start': [[2.0],[5.0],[0.0]], 'goal': [[8.0],[5.0],[0]],
        'start_xy': [2.0, 5.0], 'goal_xy': [8.0, 5.0],
    },
    {
        'name': '3_fail_annealed_utrap',
        'title': 'Failure: Annealed policy in U-trap',
        'model': m_v7, 'mode': 'single',
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'world_type': 'u_trap',
        'start': [[7.5],[5.0],[3.14]], 'goal': [[9.0],[5.0],[0]],
        'start_xy': [7.5, 5.0], 'goal_xy': [9.0, 5.0],
    },
    {
        'name': '4_success_stps_utrap',
        'title': 'Success: STPS escapes U-trap',
        'model': None, 'mode': 'stps',
        'world': 'robot_nav/worlds/u_trap_world.yaml',
        'world_type': 'u_trap',
        'start': [[7.5],[5.0],[1.57]], 'goal': [[9.0],[5.0],[0]],
        'start_xy': [7.5, 5.0], 'goal_xy': [9.0, 5.0],
    },
    {
        'name': '5_success_stps_narrow',
        'title': 'Success: STPS passes narrow door',
        'model': None, 'mode': 'stps',
        'world': 'robot_nav/worlds/narrow_door_world.yaml',
        'world_type': 'narrow_door',
        'start': [[2.0],[5.0],[0.0]], 'goal': [[8.0],[5.0],[0]],
        'start_xy': [2.0, 5.0], 'goal_xy': [8.0, 5.0],
    },
    {
        'name': '6_success_stps_standard',
        'title': 'Success: STPS in standard env',
        'model': None, 'mode': 'stps',
        'world': 'robot_nav/worlds/robot_world.yaml',
        'world_type': 'standard',
        'start': [[2.0],[2.0],[0.0]], 'goal': [[9.0],[9.0],[0]],
        'start_xy': [2.0, 2.0], 'goal_xy': [9.0, 9.0],
    },
]

for case in cases:
    print(f"\nRunning: {case['name']}...")
    outcome, traj, sw = run_and_record(
        case['model'], case['world'],
        case['start'], case['goal'],
        500, mode=case['mode']
    )
    print(f"  Outcome: {outcome}, steps: {len(traj)}, switches: {len(sw)}")

    plot_trajectory(
        traj, sw, case['world_type'],
        case['title'], outcome,
        os.path.join(output_dir, f"{case['name']}.png"),
        case['start_xy'], case['goal_xy']
    )

print(f"\n✅ All 6 screenshots saved to {output_dir}/")
print("Files:")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")
